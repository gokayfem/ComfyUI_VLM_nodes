"""Guarded adapters for ComfyUI core ``SAM3_TRACK_DATA`` payloads.

Core SAM3 keeps masks bit-packed for memory efficiency. This module preserves
that payload untouched and emits small canonical ``VLM_TRACKS`` metadata with
mask references instead of embedding dense masks in JSON.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .geometry import bbox_from_mask, clip_box
from .vision_types import (
    VLM_DETECTIONS,
    VLM_TRACKS,
    Detection,
    DetectionSequence,
    Track,
    TrackSequence,
)

SAM3_TRACK_DATA = "SAM3_TRACK_DATA"
SAM3_ADAPTER_SOURCE = "comfyui-core-sam3"
_REQUIRED_KEYS = frozenset({"packed_masks", "n_frames", "scores", "orig_size"})


@dataclass(frozen=True, slots=True)
class SAM3TrackLayout:
    n_frames: int
    n_objects: int
    mask_height: int
    mask_width: int
    orig_height: int
    orig_width: int
    scores: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class _SeedIdentity:
    track_id: int | None
    label: str | None
    text: str | None
    score: float | None
    source: str | None


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _scores(
    values: Any,
    *,
    n_objects: int,
) -> tuple[float | None, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError("SAM3 scores tensor must have shape [objects].")
        items = values.detach().cpu().tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        items = list(values)
    else:
        raise TypeError("SAM3 scores must be a one-dimensional sequence.")
    if len(items) > n_objects:
        raise ValueError("SAM3 scores contain more entries than mask objects.")
    parsed: list[float | None] = []
    for value in items:
        if value is None:
            parsed.append(None)
            continue
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("SAM3 scores must be finite values from 0 to 1.")
        parsed.append(score)
    parsed.extend([None] * (n_objects - len(parsed)))
    return tuple(parsed)


def validate_sam3_track_data(track_data: Any) -> SAM3TrackLayout:
    """Validate the private core payload before interpreting its bit layout."""

    if not isinstance(track_data, Mapping):
        raise TypeError("SAM3_TRACK_DATA must be a mapping.")
    missing = sorted(_REQUIRED_KEYS - set(track_data))
    if missing:
        raise ValueError(
            "SAM3_TRACK_DATA is missing required keys: " + ", ".join(missing)
        )

    n_frames = _integer(track_data["n_frames"], "n_frames")
    orig_size = track_data["orig_size"]
    if not isinstance(orig_size, (tuple, list)) or len(orig_size) != 2:
        raise TypeError("SAM3 orig_size must be (height, width).")
    orig_height = _integer(orig_size[0], "orig_height", minimum=1)
    orig_width = _integer(orig_size[1], "orig_width", minimum=1)

    packed = track_data["packed_masks"]
    if packed is None:
        scores = _scores(track_data["scores"], n_objects=0)
        return SAM3TrackLayout(
            n_frames=n_frames,
            n_objects=0,
            mask_height=0,
            mask_width=0,
            orig_height=orig_height,
            orig_width=orig_width,
            scores=scores,
        )
    if not isinstance(packed, torch.Tensor):
        raise TypeError("SAM3 packed_masks must be a torch.Tensor or None.")
    if packed.dtype != torch.uint8:
        raise TypeError("SAM3 packed_masks must use torch.uint8.")
    if packed.ndim != 4:
        raise ValueError(
            "SAM3 packed_masks must have shape [frames, objects, height, packed_width]."
        )
    if packed.shape[0] != n_frames:
        raise ValueError("SAM3 n_frames does not match packed_masks.")
    n_objects = int(packed.shape[1])
    mask_height = int(packed.shape[2])
    packed_width = int(packed.shape[3])
    if n_objects < 1 or mask_height < 1 or packed_width < 1:
        raise ValueError("SAM3 packed_masks dimensions must be positive.")
    scores = _scores(track_data["scores"], n_objects=n_objects)
    return SAM3TrackLayout(
        n_frames=n_frames,
        n_objects=n_objects,
        mask_height=mask_height,
        mask_width=packed_width * 8,
        orig_height=orig_height,
        orig_width=orig_width,
        scores=scores,
    )


def unpack_sam3_mask(packed_mask: torch.Tensor) -> torch.Tensor:
    """Unpack exactly one object/frame mask, avoiding full-video expansion."""

    if not isinstance(packed_mask, torch.Tensor):
        raise TypeError("packed_mask must be a torch.Tensor.")
    if packed_mask.dtype != torch.uint8 or packed_mask.ndim != 2:
        raise ValueError("packed_mask must be uint8 with shape [height, packed_width].")
    bits = torch.tensor(
        (1, 2, 4, 8, 16, 32, 64, 128),
        dtype=torch.uint8,
        device=packed_mask.device,
    )
    return (
        torch.bitwise_and(packed_mask.unsqueeze(-1), bits)
        .ne(0)
        .reshape(packed_mask.shape[0], packed_mask.shape[1] * 8)
    )


def iter_sam3_masks(
    track_data: Mapping[str, Any],
    *,
    present_only: bool = False,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield one unpacked mask at a time as ``(frame, object, mask)``."""

    layout = validate_sam3_track_data(track_data)
    packed = track_data["packed_masks"]
    if packed is None:
        return
    for frame_index in range(layout.n_frames):
        for object_index in range(layout.n_objects):
            mask = unpack_sam3_mask(packed[frame_index, object_index])
            if present_only and not bool(mask.any().item()):
                continue
            yield frame_index, object_index, mask


def _seeds_from_detections(
    sequence: DetectionSequence,
) -> list[_SeedIdentity]:
    for frame in sequence.frames:
        if frame.detections:
            return [
                _SeedIdentity(
                    track_id=detection.track_id,
                    label=detection.label,
                    text=detection.text,
                    score=detection.score,
                    source=detection.source,
                )
                for detection in frame.detections
            ]
    return []


def _seeds_from_tracks(sequence: TrackSequence) -> list[_SeedIdentity]:
    return [
        _SeedIdentity(
            track_id=track.track_id,
            label=track.label or track.detections[0].label,
            text=track.detections[0].text,
            score=track.score,
            source=track.source,
        )
        for track in sequence.tracks
    ]


def _seed_identities(
    *,
    seed_detections: DetectionSequence | None,
    seed_tracks: TrackSequence | None,
    n_objects: int,
) -> tuple[tuple[_SeedIdentity, ...], int]:
    if seed_detections is not None and seed_tracks is not None:
        raise ValueError("Connect seed_detections or seed_tracks, not both.")
    if seed_detections is not None:
        if not isinstance(seed_detections, DetectionSequence):
            raise TypeError("seed_detections must be a DetectionSequence.")
        seeds = _seeds_from_detections(seed_detections)
    elif seed_tracks is not None:
        if not isinstance(seed_tracks, TrackSequence):
            raise TypeError("seed_tracks must be a TrackSequence.")
        seeds = _seeds_from_tracks(seed_tracks)
    else:
        seeds = []

    used_ids: set[int] = set()
    next_id = 0
    identities = []
    for object_index in range(n_objects):
        seed = seeds[object_index] if object_index < len(seeds) else None
        preferred = None if seed is None else seed.track_id
        if preferred is not None and preferred not in used_ids:
            track_id = preferred
        else:
            while next_id in used_ids:
                next_id += 1
            track_id = next_id
            next_id += 1
        used_ids.add(track_id)
        identities.append(
            _SeedIdentity(
                track_id=track_id,
                label=None if seed is None else seed.label,
                text=None if seed is None else seed.text,
                score=None if seed is None else seed.score,
                source=None if seed is None else seed.source,
            )
        )
    return tuple(identities), min(len(seeds), n_objects)


def _scaled_bbox(
    bbox: tuple[float, float, float, float],
    layout: SAM3TrackLayout,
) -> tuple[float, float, float, float]:
    scale_x = layout.orig_width / layout.mask_width
    scale_y = layout.orig_height / layout.mask_height
    x1, y1, x2, y2 = bbox
    return clip_box(
        (
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ),
        layout.orig_width,
        layout.orig_height,
    )


def sam3_track_data_to_tracks(
    track_data: Mapping[str, Any],
    *,
    seed_detections: DetectionSequence | None = None,
    seed_tracks: TrackSequence | None = None,
    fps: float | None = None,
    source: str = SAM3_ADAPTER_SOURCE,
) -> TrackSequence:
    """Create canonical sparse metadata while retaining packed masks separately."""

    layout = validate_sam3_track_data(track_data)
    if fps is not None:
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive or None.")
    identities, seed_count = _seed_identities(
        seed_detections=seed_detections,
        seed_tracks=seed_tracks,
        n_objects=layout.n_objects,
    )
    detections_by_object: list[list[Detection]] = [
        [] for _index in range(layout.n_objects)
    ]

    for frame_index, object_index, mask in iter_sam3_masks(
        track_data, present_only=True
    ):
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        identity = identities[object_index]
        timestamp = frame_index / fps if fps is not None else 0.0
        score = layout.scores[object_index]
        if score is None:
            score = identity.score
        detections_by_object[object_index].append(
            Detection(
                bbox_xyxy=_scaled_bbox(bbox, layout),
                label=identity.label,
                text=identity.text,
                score=score,
                frame_index=frame_index,
                timestamp=timestamp,
                track_id=identity.track_id,
                source=source,
                metadata={
                    "observation": "propagated",
                    "visibility": "visible",
                    "sam3_object_index": object_index,
                    "mask_ref": {
                        "type": SAM3_TRACK_DATA,
                        "frame_index": frame_index,
                        "object_index": object_index,
                    },
                },
            )
        )

    tracks = []
    for object_index, detections in enumerate(detections_by_object):
        if not detections:
            continue
        identity = identities[object_index]
        present_frames = len(detections)
        final_state = (
            "active" if detections[-1].frame_index == layout.n_frames - 1 else "lost"
        )
        score = layout.scores[object_index]
        if score is None:
            score = identity.score
        tracks.append(
            Track(
                track_id=identity.track_id,
                detections=tuple(detections),
                label=identity.label,
                score=score,
                source=source,
                metadata={
                    "state": final_state,
                    "sam3_object_index": object_index,
                    "first_frame": detections[0].frame_index,
                    "last_observed_frame": detections[-1].frame_index,
                    "present_frames": present_frames,
                    "presence_ratio": (
                        present_frames / layout.n_frames if layout.n_frames else 0.0
                    ),
                    "seeded": object_index < seed_count,
                },
            )
        )
    return TrackSequence(
        width=layout.orig_width,
        height=layout.orig_height,
        tracks=tuple(sorted(tracks, key=lambda item: item.track_id)),
        frame_count=layout.n_frames,
        fps=fps,
        source=source,
        metadata={
            "adapter": "sam3-track-data/v1",
            "mask_payload": {
                "type": SAM3_TRACK_DATA,
                "encoding": "little-endian-bitpack",
                "mask_width": layout.mask_width,
                "mask_height": layout.mask_height,
            },
            "object_slots": layout.n_objects,
        },
    )


def track_report_payload(tracks: TrackSequence) -> dict[str, Any]:
    """Return a compact, history-safe report with no tensor content."""

    if not isinstance(tracks, TrackSequence):
        raise TypeError("tracks must be a TrackSequence.")
    records = []
    state_counts: dict[str, int] = {}
    total_observations = 0
    for track in tracks.tracks:
        observations: dict[str, int] = {}
        for detection in track.detections:
            kind = str(detection.metadata.to_dict().get("observation", "detected"))
            observations[kind] = observations.get(kind, 0) + 1
        total_observations += len(track.detections)
        state = str(track.metadata.to_dict().get("state", "unknown"))
        state_counts[state] = state_counts.get(state, 0) + 1
        records.append(
            {
                "track_id": track.track_id,
                "label": track.label,
                "state": state,
                "score": track.score,
                "first_frame": track.detections[0].frame_index,
                "last_frame": track.detections[-1].frame_index,
                "observation_count": len(track.detections),
                "observations": observations,
            }
        )
    media: dict[str, Any] = {
        "width": tracks.width,
        "height": tracks.height,
        "frame_count": tracks.frame_count,
    }
    if tracks.fps is not None:
        media["fps"] = tracks.fps
    return {
        "schema": "comfyui-vlm/track-report",
        "version": 1,
        "media": media,
        "track_count": len(tracks.tracks),
        "observation_count": total_observations,
        "state_counts": dict(sorted(state_counts.items())),
        "tracks": records,
    }


def track_report_json(
    tracks: TrackSequence,
    *,
    indent: int | None = 2,
) -> str:
    return json.dumps(
        track_report_payload(tracks),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
    )


def track_report_text(tracks: TrackSequence) -> str:
    report = track_report_payload(tracks)
    media = report["media"]
    lines = [
        (
            f"Tracks: {report['track_count']} | "
            f"Observations: {report['observation_count']} | "
            f"Frames: {media['frame_count']} | "
            f"Size: {media['width']}x{media['height']}"
        )
    ]
    if "fps" in media:
        lines[0] += f" | FPS: {media['fps']:g}"
    for track in report["tracks"]:
        label = track["label"] or "(unlabeled)"
        lines.append(
            f"#{track['track_id']} {label}: {track['state']}, "
            f"frames {track['first_frame']}-{track['last_frame']}, "
            f"{track['observation_count']} observations"
        )
    return "\n".join(lines)


class VLMSAM3TrackAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": (SAM3_TRACK_DATA,),
                "fps": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "0 keeps timestamps unknown.",
                    },
                ),
            },
            "optional": {
                "seed_detections": (VLM_DETECTIONS,),
                "seed_tracks": (VLM_TRACKS,),
            },
        }

    RETURN_TYPES = (VLM_TRACKS, SAM3_TRACK_DATA)
    RETURN_NAMES = ("tracks", "track_data")
    FUNCTION = "adapt"
    CATEGORY = "VLM Nodes/Vision/Tracking"

    def adapt(
        self,
        track_data,
        fps,
        seed_detections=None,
        seed_tracks=None,
    ):
        tracks = sam3_track_data_to_tracks(
            track_data,
            seed_detections=seed_detections,
            seed_tracks=seed_tracks,
            fps=None if fps <= 0 else fps,
        )
        return tracks, track_data


class VLMTrackReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tracks": (VLM_TRACKS,)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report_json", "report_text")
    FUNCTION = "report"
    CATEGORY = "VLM Nodes/Vision/Tracking"
    OUTPUT_NODE = True

    def report(self, tracks):
        report_json = track_report_json(tracks)
        report_text = track_report_text(tracks)
        return {
            "ui": {"text": [report_text]},
            "result": (report_json, report_text),
        }


NODE_CLASS_MAPPINGS = {
    "VLMSAM3TrackAdapter": VLMSAM3TrackAdapter,
    "VLMTrackReport": VLMTrackReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMSAM3TrackAdapter": "VLM SAM3 Track Adapter",
    "VLMTrackReport": "VLM Track Report",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SAM3TrackLayout",
    "SAM3_ADAPTER_SOURCE",
    "SAM3_TRACK_DATA",
    "VLMSAM3TrackAdapter",
    "VLMTrackReport",
    "iter_sam3_masks",
    "sam3_track_data_to_tracks",
    "track_report_json",
    "track_report_payload",
    "track_report_text",
    "unpack_sam3_mask",
    "validate_sam3_track_data",
]
