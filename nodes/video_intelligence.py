"""Fast temporal video understanding built from reusable perception signals.

The VLM is intentionally kept out of the per-frame loop.  This module selects
information-rich frames, preserves their real source timestamps, creates
track-aware semantic crops, and converts structured VLM responses into the
canonical temporal event and scene-state payloads.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .geometry import box_center, clip_box
from .modern_vlm import (
    ATTENTION_MODES,
    MEMORY_MODES,
    RECOMMENDED_MODEL_LABELS,
    ModernVLM,
)
from .vision_types import (
    VLM_EVENTS,
    VLM_SCENE_STATE,
    VLM_TRACKS,
    VLM_VIDEO_SELECTION,
    EventSequence,
    SceneObjectState,
    SceneState,
    SelectedVideoFrame,
    TemporalEvent,
    TrackSequence,
    VideoFrameSelection,
)

SAMPLING_STRATEGIES = (
    "Hybrid: scene + motion + tracks",
    "Uniform coverage",
    "Motion priority",
    "Scene-change priority",
    "Track-change priority",
)

VIDEO_TASKS = (
    "Action timeline",
    "Robotics scene understanding",
    "Safety and anomaly monitor",
    "Detailed temporal summary",
    "Answer a question",
)

VIDEO_REASONING_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "events"],
    "properties": {
        "summary": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "start_time",
                    "end_time",
                    "label",
                    "text",
                    "score",
                    "evidence_frame_indices",
                ],
                "properties": {
                    "start_time": {"type": "number", "minimum": 0},
                    "end_time": {"type": "number", "minimum": 0},
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_frame_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "uniqueItems": True,
                    },
                },
            },
        },
    },
}


def _video_tensor(frames: Any) -> torch.Tensor:
    if not isinstance(frames, torch.Tensor):
        raise TypeError("frames must be a ComfyUI IMAGE tensor.")
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError("frames must have shape [frames, height, width, 3|4].")
    if frames.shape[0] < 1 or frames.shape[1] < 1 or frames.shape[2] < 1:
        raise ValueError("frames must contain at least one non-empty image.")
    if not frames.dtype.is_floating_point:
        raise TypeError("ComfyUI IMAGE tensors must use a floating-point dtype.")
    if not bool(torch.isfinite(frames).all().item()):
        raise ValueError("frames contain NaN or infinite values.")
    return frames


def _positive_fps(value: Any) -> float:
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number.")
    return fps


def _robust_unit(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().to(device="cpu", dtype=torch.float32)
    if values.numel() == 0:
        return values
    low = torch.quantile(values, 0.05)
    high = torch.quantile(values, 0.95)
    span = high - low
    if float(span) <= 1.0e-8:
        return torch.zeros_like(values)
    return ((values - low) / span).clamp(0, 1)


def _visual_signals(
    frames: torch.Tensor,
    *,
    thumbnail_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized per-frame motion and coarse scene-change scores."""

    # Downsample before clamping or moving to CPU. Comfy IMAGE tensors are
    # normally float32 already, so this avoids a second full-resolution video
    # allocation merely to compute small scene/motion thumbnails.
    rgb = frames[..., :3].detach()
    if rgb.dtype != torch.float32:
        rgb = rgb.to(dtype=torch.float32)
    rgb = rgb.movedim(-1, 1)
    side = min(int(thumbnail_size), int(frames.shape[1]), int(frames.shape[2]))
    thumbnails = F.interpolate(
        rgb,
        size=(side, side),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    gray = (
        thumbnails[:, 0] * 0.299
        + thumbnails[:, 1] * 0.587
        + thumbnails[:, 2] * 0.114
    )
    frame_count = int(frames.shape[0])
    motion_raw = torch.zeros(frame_count, device=gray.device)
    scene_raw = torch.zeros(frame_count, device=gray.device)
    if frame_count > 1:
        centered = gray - gray.mean(dim=(1, 2), keepdim=True)
        structural_delta = (centered[1:] - centered[:-1]).abs().mean(dim=(1, 2))
        color_mean = thumbnails.mean(dim=(2, 3))
        color_std = thumbnails.std(dim=(2, 3), unbiased=False)
        appearance_delta = (color_mean[1:] - color_mean[:-1]).abs().mean(dim=1)
        appearance_delta += (
            color_std[1:] - color_std[:-1]
        ).abs().mean(dim=1)
        motion_raw[1:] = structural_delta
        scene_raw[1:] = structural_delta * 0.55 + appearance_delta * 0.45
    return _robust_unit(motion_raw), _robust_unit(scene_raw)


def _observed_detection(detection: Any) -> bool:
    state = detection.metadata.get("track_state")
    return state not in {"lost", "removed", "predicted"}


def _track_change_signal(
    tracks: TrackSequence | None,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> torch.Tensor:
    signal = torch.zeros(frame_count, dtype=torch.float32)
    if tracks is None:
        return signal
    if not isinstance(tracks, TrackSequence):
        raise TypeError("tracks must be a VLM Track Sequence.")
    if tracks.width != width or tracks.height != height:
        raise ValueError("Track dimensions must match the video frames.")
    diagonal = max(math.hypot(width, height), 1.0)
    for track in tracks.tracks:
        observed = sorted(
            (
                detection
                for detection in track.detections
                if _observed_detection(detection)
                and 0 <= detection.frame_index < frame_count
            ),
            key=lambda detection: detection.frame_index,
        )
        if not observed:
            continue
        signal[observed[0].frame_index] += 1.0
        if len(observed) > 1:
            signal[observed[-1].frame_index] += 0.75
        previous = observed[0]
        for current in observed[1:]:
            first_center = box_center(previous.bbox_xyxy)
            second_center = box_center(current.bbox_xyxy)
            displacement = math.hypot(
                second_center[0] - first_center[0],
                second_center[1] - first_center[1],
            )
            signal[current.frame_index] += min(1.0, displacement / diagonal * 8.0)
            previous = current
    return _robust_unit(signal)


def _uniform_indices(frame_count: int, count: int) -> list[int]:
    if count >= frame_count:
        return list(range(frame_count))
    if count <= 1:
        return [0]
    return sorted(
        {
            round(index * (frame_count - 1) / (count - 1))
            for index in range(count)
        }
    )


def _sampling_weights(strategy: str) -> tuple[float, float, float]:
    if strategy == "Uniform coverage":
        return 0.0, 0.0, 0.0
    if strategy == "Motion priority":
        return 0.75, 0.20, 0.05
    if strategy == "Scene-change priority":
        return 0.20, 0.75, 0.05
    if strategy == "Track-change priority":
        return 0.15, 0.15, 0.70
    if strategy == "Hybrid: scene + motion + tracks":
        return 0.40, 0.35, 0.25
    raise ValueError(f"Unsupported sampling strategy {strategy!r}.")


def sample_video_frames(
    frames: torch.Tensor,
    *,
    fps: float,
    max_frames: int,
    strategy: str = "Hybrid: scene + motion + tracks",
    minimum_gap_seconds: float = 0.15,
    thumbnail_size: int = 96,
    tracks: TrackSequence | None = None,
) -> tuple[torch.Tensor, VideoFrameSelection, dict[str, Any]]:
    """Select a bounded, timestamped frame batch without mutating the source."""

    frames = _video_tensor(frames)
    fps = _positive_fps(fps)
    frame_count, height, width = map(int, frames.shape[:3])
    max_frames = int(max_frames)
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")
    max_frames = min(max_frames, frame_count)
    minimum_gap_seconds = float(minimum_gap_seconds)
    if not math.isfinite(minimum_gap_seconds) or minimum_gap_seconds < 0:
        raise ValueError("minimum_gap_seconds must be finite and non-negative.")
    thumbnail_size = int(thumbnail_size)
    if not 16 <= thumbnail_size <= 256:
        raise ValueError("thumbnail_size must be between 16 and 256.")

    started = time.perf_counter()
    if strategy == "Uniform coverage" or frame_count == 1:
        motion = torch.zeros(frame_count)
        scene = torch.zeros(frame_count)
    else:
        motion, scene = _visual_signals(
            frames,
            thumbnail_size=thumbnail_size,
        )
    track_signal = _track_change_signal(
        tracks,
        frame_count=frame_count,
        width=width,
        height=height,
    )
    motion_weight, scene_weight, track_weight = _sampling_weights(strategy)
    combined = (
        motion * motion_weight
        + scene * scene_weight
        + track_signal * track_weight
    ).clamp(0, 1)

    if max_frames >= frame_count:
        selected = list(range(frame_count))
        coverage_anchors = set(selected)
    elif strategy == "Uniform coverage":
        selected = _uniform_indices(frame_count, max_frames)
        coverage_anchors = set(selected)
    else:
        anchor_count = min(
            max_frames,
            max(2 if max_frames > 1 else 1, round(max_frames * 0.35)),
        )
        coverage_anchors = set(_uniform_indices(frame_count, anchor_count))
        selected_set = set(coverage_anchors)
        if max_frames > 1:
            selected_set.update((0, frame_count - 1))
        gap_frames = max(1, round(minimum_gap_seconds * fps))
        candidates = sorted(
            range(frame_count),
            key=lambda index: (-float(combined[index]), index),
        )
        for candidate in candidates:
            if len(selected_set) >= max_frames:
                break
            if all(abs(candidate - existing) >= gap_frames for existing in selected_set):
                selected_set.add(candidate)
        if len(selected_set) < max_frames:
            for candidate in candidates:
                if len(selected_set) >= max_frames:
                    break
                selected_set.add(candidate)
        selected = sorted(selected_set)[:max_frames]

    records = []
    for index in selected:
        reasons = []
        if index == 0:
            reasons.append("first-frame")
        if index == frame_count - 1 and frame_count > 1:
            reasons.append("last-frame")
        if index in coverage_anchors:
            reasons.append("coverage")
        if float(scene[index]) >= 0.55:
            reasons.append("scene-change")
        if float(motion[index]) >= 0.55:
            reasons.append("motion")
        if float(track_signal[index]) >= 0.45:
            reasons.append("track-change")
        if not reasons:
            reasons.append("best-available")
        score = (
            1.0
            if index in {0, frame_count - 1}
            else max(float(combined[index]), 0.05 if index in coverage_anchors else 0.0)
        )
        records.append(
            SelectedVideoFrame(
                source_frame_index=index,
                timestamp=index / fps,
                score=min(1.0, score),
                reasons=tuple(reasons),
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    selection = VideoFrameSelection(
        width=width,
        height=height,
        source_frame_count=frame_count,
        fps=fps,
        frames=tuple(records),
        strategy=strategy,
        source="ComfyUI IMAGE batch",
        metadata={
            "minimum_gap_seconds": minimum_gap_seconds,
            "thumbnail_size": thumbnail_size,
            "weights": {
                "motion": motion_weight,
                "scene": scene_weight,
                "tracks": track_weight,
            },
        },
    )
    diagnostics = {
        "source_frames": frame_count,
        "selected_frames": len(selected),
        "visual_reduction_ratio": 1.0 - len(selected) / frame_count,
        "sampling_ms": elapsed_ms,
        "strategy": strategy,
        "indices": selected,
        "timestamps": [record.timestamp for record in records],
        "motion_peak": float(motion.max()) if motion.numel() else 0.0,
        "scene_peak": float(scene.max()) if scene.numel() else 0.0,
        "track_peak": float(track_signal.max()) if track_signal.numel() else 0.0,
    }
    index_tensor = torch.tensor(selected, dtype=torch.long, device=frames.device)
    return frames.index_select(0, index_tensor), selection, diagnostics


def resize_video_for_analysis(
    frames: torch.Tensor,
    *,
    max_side: int,
) -> torch.Tensor:
    """Downscale a sampled batch once while preserving aspect ratio and range."""

    frames = _video_tensor(frames)
    max_side = int(max_side)
    if max_side == 0:
        return frames
    if not 128 <= max_side <= 4096:
        raise ValueError("analysis max_side must be 0 or between 128 and 4096.")
    height, width = map(int, frames.shape[1:3])
    current_max = max(height, width)
    if current_max <= max_side:
        return frames
    scale = max_side / current_max
    target_height = max(1, round(height * scale))
    target_width = max(1, round(width * scale))
    resized = F.interpolate(
        frames[..., :3].movedim(-1, 1).to(dtype=torch.float32),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.movedim(1, -1).clamp(0, 1)


def _letterbox_crop(
    frame: torch.Tensor,
    box: Sequence[float],
    *,
    output_size: int,
    context_scale: float,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    height, width = map(int, frame.shape[:2])
    x1, y1, x2, y2 = clip_box(box, width, height)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    crop_width = max(2.0, (x2 - x1) * context_scale)
    crop_height = max(2.0, (y2 - y1) * context_scale)
    left = max(0, math.floor(center_x - crop_width * 0.5))
    top = max(0, math.floor(center_y - crop_height * 0.5))
    right = min(width, math.ceil(center_x + crop_width * 0.5))
    bottom = min(height, math.ceil(center_y + crop_height * 0.5))
    if right <= left or bottom <= top:
        raise ValueError("A tracked detection produced an empty crop.")
    crop = frame[top:bottom, left:right, :3].movedim(-1, 0).unsqueeze(0)
    scale = min(output_size / crop.shape[-1], output_size / crop.shape[-2])
    resized_width = max(1, round(crop.shape[-1] * scale))
    resized_height = max(1, round(crop.shape[-2] * scale))
    resized = F.interpolate(
        crop.to(dtype=torch.float32),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    canvas = torch.zeros(
        (1, 3, output_size, output_size),
        dtype=resized.dtype,
        device=resized.device,
    )
    x_offset = (output_size - resized_width) // 2
    y_offset = (output_size - resized_height) // 2
    canvas[
        :,
        :,
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas[0].movedim(0, -1), (left, top, right, bottom)


def track_aware_crops(
    frames: torch.Tensor,
    tracks: TrackSequence,
    *,
    crops_per_track: int,
    max_crops: int,
    output_size: int,
    context_scale: float,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    frames = _video_tensor(frames)
    if not isinstance(tracks, TrackSequence):
        raise TypeError("tracks must be a VLM Track Sequence.")
    if tracks.width != frames.shape[2] or tracks.height != frames.shape[1]:
        raise ValueError("Track dimensions must match the video frames.")
    crops_per_track = int(crops_per_track)
    max_crops = int(max_crops)
    output_size = int(output_size)
    context_scale = float(context_scale)
    if not 1 <= crops_per_track <= 16:
        raise ValueError("crops_per_track must be between 1 and 16.")
    if not 1 <= max_crops <= 256:
        raise ValueError("max_crops must be between 1 and 256.")
    if not 64 <= output_size <= 2048:
        raise ValueError("output_size must be between 64 and 2048.")
    if not 1.0 <= context_scale <= 4.0:
        raise ValueError("context_scale must be between 1 and 4.")

    candidates = []
    for track in tracks.tracks:
        observed = sorted(
            (
                detection
                for detection in track.detections
                if _observed_detection(detection)
                and detection.frame_index < frames.shape[0]
            ),
            key=lambda detection: detection.frame_index,
        )
        if not observed:
            continue
        positions = _uniform_indices(len(observed), min(crops_per_track, len(observed)))
        for position in positions:
            detection = observed[position]
            candidates.append((track, detection))
    candidates = candidates[:max_crops]
    if not candidates:
        raise ValueError("No observed track detections are available for cropping.")

    crops = []
    manifest = []
    for crop_index, (track, detection) in enumerate(candidates):
        crop, crop_bounds = _letterbox_crop(
            frames[detection.frame_index],
            detection.bbox_xyxy,
            output_size=output_size,
            context_scale=context_scale,
        )
        crops.append(crop)
        manifest.append(
            {
                "crop_index": crop_index,
                "track_id": track.track_id,
                "label": track.label or detection.label,
                "source_frame_index": detection.frame_index,
                "timestamp": detection.timestamp,
                "detection_bbox_xyxy": list(detection.bbox_xyxy),
                "crop_bounds_xyxy": list(crop_bounds),
            }
        )
    return torch.stack(crops), manifest


def _track_label(track: Any, observed: Sequence[Any]) -> str | None:
    if track.label:
        return track.label
    labels = [detection.label for detection in observed if detection.label]
    return Counter(labels).most_common(1)[0][0] if labels else None


def build_scene_state(
    tracks: TrackSequence,
    events: EventSequence | None = None,
) -> SceneState:
    if not isinstance(tracks, TrackSequence):
        raise TypeError("tracks must be a VLM Track Sequence.")
    if events is not None and not isinstance(events, EventSequence):
        raise TypeError("events must be a VLM Event Sequence.")
    objects = []
    for track in tracks.tracks:
        observed = sorted(
            (
                detection
                for detection in track.detections
                if _observed_detection(detection)
            ),
            key=lambda detection: (detection.timestamp, detection.frame_index),
        )
        if not observed:
            continue
        first, last = observed[0], observed[-1]
        elapsed = last.timestamp - first.timestamp
        if elapsed > 1.0e-6:
            first_center = box_center(first.bbox_xyxy)
            last_center = box_center(last.bbox_xyxy)
            velocity = (
                (last_center[0] - first_center[0]) / elapsed,
                (last_center[1] - first_center[1]) / elapsed,
            )
        else:
            velocity = (0.0, 0.0)
        scores = [
            detection.score
            for detection in observed
            if detection.score is not None
        ]
        latest_state = last.metadata.get("track_state")
        state = (
            latest_state
            if isinstance(latest_state, str) and latest_state
            else track.metadata.get("state", "active")
        )
        objects.append(
            SceneObjectState(
                track_id=track.track_id,
                label=_track_label(track, observed),
                first_seen=first.timestamp,
                last_seen=last.timestamp,
                last_bbox_xyxy=last.bbox_xyxy,
                observation_count=len(observed),
                mean_confidence=(
                    sum(scores) / len(scores) if scores else track.score
                ),
                velocity_xy_px_s=velocity,
                state=str(state),
                metadata={
                    "first_frame": first.frame_index,
                    "last_frame": last.frame_index,
                    "trajectory_samples": len(track.detections),
                },
            )
        )
    return SceneState(
        width=tracks.width,
        height=tracks.height,
        frame_count=tracks.frame_count,
        fps=tracks.fps,
        objects=tuple(sorted(objects, key=lambda item: item.track_id)),
        events=events.events if events is not None else (),
        source="VLM persistent scene state",
        metadata={
            "tracker_source": tracks.source,
            "track_count": len(tracks.tracks),
            "observed_object_count": len(objects),
        },
    )


def scene_state_summary(scene: SceneState) -> str:
    lines = [
        (
            f"Scene: {scene.width}x{scene.height}, {scene.frame_count} frames, "
            f"{scene.fps:g} FPS."
            if scene.fps is not None
            else f"Scene: {scene.width}x{scene.height}, {scene.frame_count} frames."
        )
    ]
    for item in scene.objects:
        label = item.label or "object"
        vx, vy = item.velocity_xy_px_s
        lines.append(
            f"#{item.track_id} {label}: {item.state}; "
            f"{item.first_seen:.3f}s–{item.last_seen:.3f}s; "
            f"{item.observation_count} observations; "
            f"velocity=({vx:.1f}, {vy:.1f}) px/s."
        )
    for event in scene.events:
        lines.append(
            f"Event {event.start_time:.3f}s–{event.end_time:.3f}s: "
            f"{event.label or event.text or 'event'}."
        )
    return "\n".join(lines)


def build_video_reasoning_prompt(
    selection: VideoFrameSelection,
    *,
    task: str,
    question: str,
    max_events: int,
    scene_state: SceneState | None = None,
) -> str:
    if not isinstance(selection, VideoFrameSelection):
        raise TypeError("selection must be a VLM Video Selection.")
    if task not in VIDEO_TASKS:
        raise ValueError(f"Unsupported video reasoning task {task!r}.")
    max_events = int(max_events)
    if not 1 <= max_events <= 256:
        raise ValueError("max_events must be between 1 and 256.")
    question = str(question).strip()
    if task == "Answer a question" and not question:
        raise ValueError("A question is required for the question-answering task.")
    mapping = "\n".join(
        f"- supplied image {position}: source frame "
        f"{frame.source_frame_index}, timestamp {frame.timestamp:.6f}s"
        for position, frame in enumerate(selection.frames)
    )
    task_guidance = {
        "Action timeline": (
            "Identify observable actions and changes in chronological order. "
            "Do not invent events between evidence frames."
        ),
        "Robotics scene understanding": (
            "Describe agents, manipulated objects, spatial relations, motion, "
            "object permanence, affordances, and uncertainty. Do not propose "
            "motor commands."
        ),
        "Safety and anomaly monitor": (
            "Identify only visible hazards, unsafe interactions, anomalies, "
            "and state changes. State uncertainty explicitly."
        ),
        "Detailed temporal summary": (
            "Summarize the scene and its temporal progression with precise "
            "timestamps and stable identities."
        ),
        "Answer a question": (
            f"Answer this question using only visible evidence: {question}"
        ),
    }[task]
    state_context = (
        "\nKnown track-derived scene state:\n" + scene_state_summary(scene_state)
        if scene_state is not None
        else ""
    )
    schema = json.dumps(
        VIDEO_REASONING_JSON_SCHEMA,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Analyze the sampled frames from one video. The samples are irregularly "
        "spaced; never assume adjacent supplied images are adjacent source frames.\n"
        f"Source video: {selection.source_frame_count} frames at "
        f"{selection.fps:g} FPS, duration {selection.duration:.6f}s.\n"
        f"Frame mapping:\n{mapping}{state_context}\n\n"
        f"Task: {task_guidance}\n"
        f"Return at most {max_events} non-overlapping or meaningfully overlapping "
        "events. Use seconds on the source timeline. evidence_frame_indices must "
        "contain only source frame indices listed above. Scores express confidence "
        "in visible evidence, not importance. If no event is supported, return an "
        "empty events array. Output one JSON object only: no Markdown and no prose "
        "outside JSON.\n"
        f"JSON Schema:\n{schema}"
    )


def _first_json_object(text: str) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The VLM response is empty.")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError("The VLM response does not contain a valid JSON object.")


def parse_video_reasoning_output(
    text: str,
    selection: VideoFrameSelection,
    *,
    max_events: int = 256,
) -> tuple[str, EventSequence, str]:
    if not isinstance(selection, VideoFrameSelection):
        raise TypeError("selection must be a VLM Video Selection.")
    value = _first_json_object(text)
    summary = value.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Structured video output requires a non-empty summary.")
    raw_events = value.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("Structured video output events must be an array.")
    if len(raw_events) > int(max_events):
        raise ValueError(f"Structured video output exceeds {int(max_events)} events.")
    allowed_indices = set(selection.indices)
    parsed = []
    for index, item in enumerate(raw_events):
        if not isinstance(item, Mapping):
            raise TypeError(f"Event {index} must be a JSON object.")
        try:
            start = float(item["start_time"])
            end = float(item["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Event {index} has invalid timestamps.") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
            or end > selection.duration + 1.0e-6
        ):
            raise ValueError(f"Event {index} lies outside the source timeline.")
        label = item.get("label")
        text_value = item.get("text")
        if label is not None and not isinstance(label, str):
            raise TypeError(f"Event {index} label must be a string.")
        if text_value is not None and not isinstance(text_value, str):
            raise TypeError(f"Event {index} text must be a string.")
        if not (label and label.strip()) and not (text_value and text_value.strip()):
            raise ValueError(f"Event {index} requires a label or description.")
        raw_score = item.get("score")
        score = None if raw_score is None else float(raw_score)
        if score is not None and (
            not math.isfinite(score) or not 0.0 <= score <= 1.0
        ):
            raise ValueError(f"Event {index} score must be between 0 and 1.")
        evidence = item.get("evidence_frame_indices", [])
        if not isinstance(evidence, list) or any(
            not isinstance(frame_index, int) for frame_index in evidence
        ):
            raise TypeError(
                f"Event {index} evidence_frame_indices must contain integers."
            )
        if len(evidence) != len(set(evidence)):
            raise ValueError(f"Event {index} contains duplicate evidence frames.")
        invalid_evidence = sorted(set(evidence) - allowed_indices)
        evidence_mode = "source-frame-index"
        if invalid_evidence:
            if all(0 <= frame_index < len(selection.frames) for frame_index in evidence):
                evidence = [
                    selection.frames[position].source_frame_index
                    for position in evidence
                ]
                evidence_mode = "supplied-image-position"
            else:
                raise ValueError(
                    f"Event {index} references frames that were not supplied "
                    "to the VLM."
                )
        parsed.append(
            TemporalEvent(
                start_time=start,
                end_time=end,
                label=label.strip() if isinstance(label, str) and label.strip() else None,
                text=(
                    text_value.strip()
                    if isinstance(text_value, str) and text_value.strip()
                    else None
                ),
                score=score,
                source="video-vlm",
                metadata={
                    "evidence_frame_indices": evidence,
                    "evidence_index_mode": evidence_mode,
                },
            )
        )
    parsed.sort(key=lambda event: (event.start_time, event.end_time))
    events = EventSequence(
        events=tuple(parsed),
        duration=selection.duration,
        source="video-vlm",
        metadata={
            "source_frame_count": selection.source_frame_count,
            "sampled_frame_count": len(selection.frames),
            "sampling_strategy": selection.strategy,
        },
    )
    normalized = json.dumps(
        {
            "summary": summary.strip(),
            "events": [event.to_dict() for event in events.events],
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    return summary.strip(), events, normalized


class VLMAdaptiveFrameSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "max_frames": (
                    "INT",
                    {"default": 16, "min": 1, "max": 512},
                ),
                "strategy": (
                    SAMPLING_STRATEGIES,
                    {"default": "Hybrid: scene + motion + tracks"},
                ),
                "minimum_gap_seconds": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 60.0, "step": 0.01},
                ),
                "thumbnail_size": (
                    "INT",
                    {"default": 96, "min": 16, "max": 256, "step": 16},
                ),
            },
            "optional": {"tracks": (VLM_TRACKS,)},
        }

    RETURN_TYPES = ("IMAGE", VLM_VIDEO_SELECTION, "STRING", "STRING")
    RETURN_NAMES = (
        "sampled_frames",
        "selection",
        "selection_json",
        "diagnostics_json",
    )
    FUNCTION = "sample"
    CATEGORY = "VLM Nodes/Video Intelligence"

    def sample(
        self,
        frames,
        fps,
        max_frames,
        strategy,
        minimum_gap_seconds,
        thumbnail_size,
        tracks=None,
    ):
        sampled, selection, diagnostics = sample_video_frames(
            frames,
            fps=fps,
            max_frames=max_frames,
            strategy=strategy,
            minimum_gap_seconds=minimum_gap_seconds,
            thumbnail_size=thumbnail_size,
            tracks=tracks,
        )
        return (
            sampled,
            selection,
            selection.to_json(indent=2),
            json.dumps(diagnostics, indent=2, sort_keys=True),
        )


class VLMTrackAwareCrops:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "tracks": (VLM_TRACKS,),
                "crops_per_track": (
                    "INT",
                    {"default": 3, "min": 1, "max": 16},
                ),
                "max_crops": (
                    "INT",
                    {"default": 32, "min": 1, "max": 256},
                ),
                "output_size": (
                    "INT",
                    {"default": 448, "min": 64, "max": 2048, "step": 32},
                ),
                "context_scale": (
                    "FLOAT",
                    {"default": 1.35, "min": 1.0, "max": 4.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("crops", "crop_manifest_json")
    FUNCTION = "crop"
    CATEGORY = "VLM Nodes/Video Intelligence"

    def crop(
        self,
        frames,
        tracks,
        crops_per_track,
        max_crops,
        output_size,
        context_scale,
    ):
        crops, manifest = track_aware_crops(
            frames,
            tracks,
            crops_per_track=crops_per_track,
            max_crops=max_crops,
            output_size=output_size,
            context_scale=context_scale,
        )
        return (
            crops,
            json.dumps(
                {"schema": "comfyui-vlm/track-crops", "crops": manifest},
                indent=2,
                sort_keys=True,
            ),
        )


class VLMBuildSceneState:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"tracks": (VLM_TRACKS,)},
            "optional": {"events": (VLM_EVENTS,)},
        }

    RETURN_TYPES = (VLM_SCENE_STATE, "STRING", "STRING")
    RETURN_NAMES = ("scene_state", "scene_state_json", "summary")
    FUNCTION = "build"
    CATEGORY = "VLM Nodes/Video Intelligence"

    def build(self, tracks, events=None):
        scene = build_scene_state(tracks, events)
        return scene, scene.to_json(indent=2), scene_state_summary(scene)


class VLMVideoReasoningPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selection": (VLM_VIDEO_SELECTION,),
                "task": (VIDEO_TASKS, {"default": "Action timeline"}),
                "question": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What changes over time?",
                    },
                ),
                "max_events": (
                    "INT",
                    {"default": 24, "min": 1, "max": 256},
                ),
            },
            "optional": {"scene_state": (VLM_SCENE_STATE,)},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "json_schema")
    FUNCTION = "build"
    CATEGORY = "VLM Nodes/Video Intelligence"

    def build(self, selection, task, question, max_events, scene_state=None):
        return (
            build_video_reasoning_prompt(
                selection,
                task=task,
                question=question,
                max_events=max_events,
                scene_state=scene_state,
            ),
            json.dumps(VIDEO_REASONING_JSON_SCHEMA, indent=2, sort_keys=True),
        )


class VLMEventsFromVideoJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "selection": (VLM_VIDEO_SELECTION,),
                "max_events": (
                    "INT",
                    {"default": 256, "min": 1, "max": 4096},
                ),
            }
        }

    RETURN_TYPES = (VLM_EVENTS, "STRING", "STRING")
    RETURN_NAMES = ("events", "summary", "normalized_json")
    FUNCTION = "parse"
    CATEGORY = "VLM Nodes/Video Intelligence"

    def parse(self, text, selection, max_events):
        summary, events, normalized = parse_video_reasoning_output(
            text,
            selection,
            max_events=max_events,
        )
        return events, summary, normalized


class VLMVideoTemporalReasoner(ModernVLM):
    """Convenience node: adaptively sample, reason, and validate in one call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "task": (VIDEO_TASKS, {"default": "Action timeline"}),
                "question": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What changes over time?",
                    },
                ),
                "model": (
                    list(RECOMMENDED_MODEL_LABELS),
                    {"default": "SmolVLM2 2.2B Video"},
                ),
                "custom_model_id": ("STRING", {"default": ""}),
                "memory_mode": (
                    MEMORY_MODES,
                    {"default": "ComfyUI managed (BF16)"},
                ),
                "max_frames": (
                    "INT",
                    {"default": 16, "min": 1, "max": 512},
                ),
                "max_events": (
                    "INT",
                    {"default": 24, "min": 1, "max": 256},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 768, "min": 32, "max": 16384},
                ),
            },
            "optional": {
                "tracks": (VLM_TRACKS,),
                "scene_state": (VLM_SCENE_STATE,),
                "strategy": (
                    SAMPLING_STRATEGIES,
                    {"default": "Hybrid: scene + motion + tracks"},
                ),
                "minimum_gap_seconds": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 60.0, "step": 0.01},
                ),
                "analysis_max_side": (
                    "INT",
                    {
                        "default": 448,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": (
                            "Downscale sampled frames before the VLM. 0 keeps "
                            "the source resolution; 448 is the fast default."
                        ),
                    },
                ),
                "attention_mode": (
                    ATTENTION_MODES,
                    {"default": "Auto (SDPA)"},
                ),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "strict_output": ("BOOLEAN", {"default": True}),
                "unload_after": ("BOOLEAN", {"default": False}),
                "stream_output": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "STRING",
        VLM_EVENTS,
        VLM_VIDEO_SELECTION,
        "IMAGE",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "summary",
        "events",
        "selection",
        "sampled_frames",
        "raw_response",
        "diagnostics_json",
        "events_json",
        "selection_json",
    )
    FUNCTION = "analyze"
    CATEGORY = "VLM Nodes/Video Intelligence"

    def analyze(
        self,
        frames,
        fps,
        task,
        question,
        model,
        custom_model_id,
        memory_mode,
        max_frames,
        max_events,
        max_new_tokens,
        tracks=None,
        scene_state=None,
        strategy="Hybrid: scene + motion + tracks",
        minimum_gap_seconds=0.15,
        analysis_max_side=448,
        attention_mode="Auto (SDPA)",
        enable_thinking=False,
        strict_output=True,
        unload_after=False,
        stream_output=True,
        unique_id=None,
    ):
        sampled, selection, diagnostics = sample_video_frames(
            frames,
            fps=fps,
            max_frames=max_frames,
            strategy=strategy,
            minimum_gap_seconds=minimum_gap_seconds,
            tracks=tracks,
        )
        prompt = build_video_reasoning_prompt(
            selection,
            task=task,
            question=question,
            max_events=max_events,
            scene_state=scene_state,
        )
        analysis_frames = resize_video_for_analysis(
            sampled,
            max_side=analysis_max_side,
        )
        diagnostics["source_resolution"] = [
            int(sampled.shape[2]),
            int(sampled.shape[1]),
        ]
        diagnostics["analysis_resolution"] = [
            int(analysis_frames.shape[2]),
            int(analysis_frames.shape[1]),
        ]
        diagnostics["analysis_pixel_reduction_ratio"] = 1.0 - (
            analysis_frames.shape[1] * analysis_frames.shape[2]
        ) / (sampled.shape[1] * sampled.shape[2])
        reasoning_started = time.perf_counter()
        raw_response = super().run(
            prompt=prompt,
            model=model,
            custom_model_id=custom_model_id,
            memory_mode=memory_mode,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            image=None,
            system_prompt=(
                "You are a precise temporal video analyst. Use only visible "
                "evidence, preserve source timestamps, and obey the JSON Schema."
            ),
            video_frames=analysis_frames,
            fps=fps,
            video_selection=selection,
            attention_mode=attention_mode,
            enable_thinking=enable_thinking,
            unload_after=unload_after,
            stream_output=stream_output,
            unique_id=unique_id,
        )[0]
        diagnostics["reasoning_ms"] = (
            time.perf_counter() - reasoning_started
        ) * 1000
        try:
            summary, events, _normalized = parse_video_reasoning_output(
                raw_response,
                selection,
                max_events=max_events,
            )
            diagnostics["structured_output_valid"] = True
        except (TypeError, ValueError) as exc:
            diagnostics["structured_output_valid"] = False
            diagnostics["structured_output_error"] = str(exc)
            if strict_output:
                raise ValueError(
                    "The video VLM did not return valid structured temporal "
                    f"output: {exc}"
                ) from exc
            summary = raw_response.strip()
            events = EventSequence(
                duration=selection.duration,
                source="video-vlm-unstructured",
                metadata={"validation_failed": True},
            )
        return (
            summary,
            events,
            selection,
            sampled,
            raw_response,
            json.dumps(diagnostics, indent=2, sort_keys=True),
            events.to_json(indent=2),
            selection.to_json(indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "VLMAdaptiveFrameSampler": VLMAdaptiveFrameSampler,
    "VLMTrackAwareCrops": VLMTrackAwareCrops,
    "VLMBuildSceneState": VLMBuildSceneState,
    "VLMVideoReasoningPrompt": VLMVideoReasoningPrompt,
    "VLMEventsFromVideoJSON": VLMEventsFromVideoJSON,
    "VLMVideoTemporalReasoner": VLMVideoTemporalReasoner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMAdaptiveFrameSampler": "VLM Adaptive Frame Sampler",
    "VLMTrackAwareCrops": "VLM Track-Aware Semantic Crops",
    "VLMBuildSceneState": "VLM Persistent Scene State",
    "VLMVideoReasoningPrompt": "VLM Video Reasoning Prompt",
    "VLMEventsFromVideoJSON": "VLM Temporal Events From JSON",
    "VLMVideoTemporalReasoner": "VLM Video Temporal Reasoner",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SAMPLING_STRATEGIES",
    "VIDEO_REASONING_JSON_SCHEMA",
    "VIDEO_TASKS",
    "build_scene_state",
    "build_video_reasoning_prompt",
    "parse_video_reasoning_output",
    "resize_video_for_analysis",
    "sample_video_frames",
    "scene_state_summary",
    "track_aware_crops",
]
