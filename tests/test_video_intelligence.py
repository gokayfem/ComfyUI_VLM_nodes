from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ComfyUI_VLM_nodes.nodes.video_intelligence import (
    NODE_CLASS_MAPPINGS,
    VLMAdaptiveFrameSampler,
    build_scene_state,
    build_video_reasoning_prompt,
    parse_video_reasoning_output,
    resize_video_for_analysis,
    sample_video_frames,
    scene_state_summary,
    track_aware_crops,
)
from ComfyUI_VLM_nodes.nodes.vision_types import (
    Detection,
    EventSequence,
    SceneState,
    SelectedVideoFrame,
    Track,
    TrackSequence,
    VideoFrameSelection,
)


def _moving_video(frame_count=20, height=48, width=64):
    frames = torch.zeros((frame_count, height, width, 3), dtype=torch.float32)
    for frame_index in range(frame_count):
        x = min(width - 9, 2 + frame_index * 2)
        frames[frame_index, 16:28, x : x + 8, 0] = 1.0
        if frame_index >= frame_count // 2:
            frames[frame_index, :, :, 2] += 0.55
    return frames.clamp(0, 1)


def _tracks(width=64, height=48, frame_count=20, fps=10.0):
    detections = []
    for frame_index in (0, 5, 10, 15, 19):
        x = min(width - 12, 2 + frame_index * 2)
        detections.append(
            Detection(
                bbox_xyxy=(x, 14, x + 10, 30),
                label="red object",
                score=0.9 - frame_index * 0.005,
                frame_index=frame_index,
                timestamp=frame_index / fps,
                track_id=3,
                metadata={"track_state": "active"},
            )
        )
    return TrackSequence(
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        tracks=(
            Track(
                track_id=3,
                detections=tuple(detections),
                label="red object",
                score=0.85,
            ),
        ),
        source="unit-test-tracker",
    )


def _selection():
    return VideoFrameSelection(
        width=64,
        height=48,
        source_frame_count=20,
        fps=10.0,
        strategy="Hybrid: scene + motion + tracks",
        frames=(
            SelectedVideoFrame(0, 0.0, 1.0, ("first-frame",)),
            SelectedVideoFrame(5, 0.5, 0.7, ("motion",)),
            SelectedVideoFrame(10, 1.0, 0.9, ("scene-change",)),
            SelectedVideoFrame(19, 1.9, 1.0, ("last-frame",)),
        ),
    )


def test_uniform_sampling_is_deterministic_and_preserves_timestamps():
    frames = _moving_video()
    sampled, selection, diagnostics = sample_video_frames(
        frames,
        fps=10.0,
        max_frames=5,
        strategy="Uniform coverage",
    )
    assert sampled.shape == (5, 48, 64, 3)
    assert selection.indices == (0, 5, 10, 14, 19)
    assert selection.timestamps == pytest.approx((0.0, 0.5, 1.0, 1.4, 1.9))
    assert diagnostics["visual_reduction_ratio"] == pytest.approx(0.75)
    assert torch.equal(sampled[2], frames[10])


def test_hybrid_sampling_captures_boundaries_scene_change_and_motion():
    frames = _moving_video()
    sampled, selection, diagnostics = sample_video_frames(
        frames,
        fps=10.0,
        max_frames=7,
        strategy="Hybrid: scene + motion + tracks",
        minimum_gap_seconds=0.2,
    )
    assert sampled.shape[0] == 7
    assert selection.indices[0] == 0
    assert selection.indices[-1] == 19
    assert any(9 <= index <= 11 for index in selection.indices)
    assert diagnostics["motion_peak"] > 0
    assert diagnostics["scene_peak"] > 0
    assert selection.to_json() == VideoFrameSelection.from_json(
        selection.to_json()
    ).to_json()


def test_track_priority_uses_track_changes_and_validates_dimensions():
    frames = _moving_video()
    tracks = _tracks()
    _sampled, selection, diagnostics = sample_video_frames(
        frames,
        fps=10.0,
        max_frames=6,
        strategy="Track-change priority",
        tracks=tracks,
    )
    assert diagnostics["track_peak"] == pytest.approx(1.0)
    assert any(
        "track-change" in frame.reasons for frame in selection.frames
    )
    bad_tracks = TrackSequence(
        width=65,
        height=48,
        frame_count=20,
        fps=10,
        tracks=(),
    )
    with pytest.raises(ValueError, match="dimensions"):
        sample_video_frames(
            frames,
            fps=10,
            max_frames=4,
            tracks=bad_tracks,
        )


@pytest.mark.parametrize(
    "frames",
    [
        torch.zeros(4, 16, 16),
        torch.zeros(4, 16, 16, 2),
        torch.zeros(0, 16, 16, 3),
        torch.zeros(4, 16, 16, 3, dtype=torch.uint8),
    ],
)
def test_sampling_rejects_invalid_video_tensors(frames):
    with pytest.raises((TypeError, ValueError)):
        sample_video_frames(frames, fps=24, max_frames=4)


def test_track_aware_crops_preserve_identity_and_source_frame_mapping():
    frames = _moving_video()
    crops, manifest = track_aware_crops(
        frames,
        _tracks(),
        crops_per_track=3,
        max_crops=8,
        output_size=96,
        context_scale=1.4,
    )
    assert crops.shape == (3, 96, 96, 3)
    assert [item["track_id"] for item in manifest] == [3, 3, 3]
    assert [item["source_frame_index"] for item in manifest] == [0, 10, 19]
    assert crops.max().item() > 0.5


def test_analysis_resize_reduces_pixels_without_changing_batch_or_aspect():
    frames = _moving_video(height=128, width=256)
    resized = resize_video_for_analysis(frames, max_side=128)
    assert resized.shape == (20, 64, 128, 3)
    assert resized.min().item() >= 0
    assert resized.max().item() <= 1
    assert resize_video_for_analysis(frames, max_side=0) is frames


def test_scene_state_ignores_predicted_track_samples_and_computes_velocity():
    tracks = _tracks()
    predicted = Detection(
        bbox_xyxy=(48, 14, 58, 30),
        label="red object",
        frame_index=18,
        timestamp=1.8,
        track_id=3,
        metadata={"track_state": "predicted"},
    )
    track = tracks.tracks[0]
    with_prediction = TrackSequence(
        width=tracks.width,
        height=tracks.height,
        frame_count=tracks.frame_count,
        fps=tracks.fps,
        tracks=(
            Track(
                track_id=3,
                detections=tuple(
                    sorted(
                        (*track.detections, predicted),
                        key=lambda item: item.frame_index,
                    )
                ),
                label=track.label,
            ),
        ),
    )
    scene = build_scene_state(with_prediction)
    assert len(scene.objects) == 1
    item = scene.objects[0]
    assert item.observation_count == 5
    assert item.velocity_xy_px_s[0] > 0
    assert "#3 red object" in scene_state_summary(scene)
    assert SceneState.from_json(scene.to_json()).to_json() == scene.to_json()


def test_reasoning_prompt_explains_irregular_source_timeline():
    prompt = build_video_reasoning_prompt(
        _selection(),
        task="Robotics scene understanding",
        question="",
        max_events=12,
    )
    assert "irregularly spaced" in prompt
    assert "supplied image 2: source frame 10, timestamp 1.000000s" in prompt
    assert "Do not propose motor commands" in prompt
    assert "evidence_frame_indices" in prompt


def test_structured_video_output_parses_fenced_json_and_preserves_evidence():
    response = """Result:
```json
{
  "summary": "A red object moves to the right.",
  "events": [
    {
      "start_time": 0.0,
      "end_time": 1.9,
      "label": "object motion",
      "text": "The red object moves from left to right.",
      "score": 0.94,
      "evidence_frame_indices": [0, 10, 19]
    }
  ]
}
```
"""
    summary, events, normalized = parse_video_reasoning_output(
        response,
        _selection(),
    )
    assert summary == "A red object moves to the right."
    assert len(events.events) == 1
    assert events.events[0].metadata["evidence_frame_indices"] == (0, 10, 19)
    assert json.loads(normalized)["events"][0]["label"] == "object motion"


def test_structured_output_normalizes_supplied_image_positions_to_source_frames():
    response = json.dumps(
        {
            "summary": "A transition occurs.",
            "events": [
                {
                    "start_time": 0.5,
                    "end_time": 1.9,
                    "label": "transition",
                    "text": "The scene changes.",
                    "score": 0.8,
                    # Positions 1 and 3 in the supplied image batch.
                    "evidence_frame_indices": [1, 3],
                }
            ],
        }
    )
    _summary, events, _normalized = parse_video_reasoning_output(
        response,
        _selection(),
    )
    event = events.events[0]
    assert event.metadata["evidence_frame_indices"] == (5, 19)
    assert event.metadata["evidence_index_mode"] == "supplied-image-position"


@pytest.mark.parametrize(
    ("event_patch", "error"),
    [
        ({"end_time": 2.1}, "outside"),
        ({"score": 1.2}, "between"),
        ({"evidence_frame_indices": [0, 7]}, "not supplied"),
        ({"evidence_frame_indices": [0, 0]}, "duplicate"),
        ({"label": "", "text": ""}, "requires"),
    ],
)
def test_structured_video_output_rejects_unverifiable_events(event_patch, error):
    event = {
        "start_time": 0.0,
        "end_time": 1.0,
        "label": "motion",
        "text": "Object moves.",
        "score": 0.8,
        "evidence_frame_indices": [0, 10],
    }
    event.update(event_patch)
    with pytest.raises((TypeError, ValueError), match=error):
        parse_video_reasoning_output(
            json.dumps({"summary": "test", "events": [event]}),
            _selection(),
        )


def test_scene_state_accepts_validated_events():
    _summary, events, _normalized = parse_video_reasoning_output(
        json.dumps(
            {
                "summary": "motion",
                "events": [
                    {
                        "start_time": 0.0,
                        "end_time": 1.9,
                        "label": "motion",
                        "text": "Object moves.",
                        "score": 0.9,
                        "evidence_frame_indices": [0, 19],
                    }
                ],
            }
        ),
        _selection(),
    )
    scene = build_scene_state(_tracks(), events)
    assert isinstance(events, EventSequence)
    assert len(scene.events) == 1
    assert "Event 0.000s–1.900s" in scene_state_summary(scene)


def test_node_surface_registers_all_video_intelligence_nodes():
    assert set(NODE_CLASS_MAPPINGS) == {
        "VLMAdaptiveFrameSampler",
        "VLMTrackAwareCrops",
        "VLMBuildSceneState",
        "VLMVideoReasoningPrompt",
        "VLMEventsFromVideoJSON",
        "VLMVideoTemporalReasoner",
    }
    inputs = VLMAdaptiveFrameSampler.INPUT_TYPES()
    assert inputs["required"]["frames"][0] == "IMAGE"
    assert inputs["optional"]["tracks"][0] == "VLM_TRACKS"
    reasoner = NODE_CLASS_MAPPINGS["VLMVideoTemporalReasoner"]
    assert reasoner.RETURN_NAMES[-2:] == ("events_json", "selection_json")


def test_api_example_uses_direct_json_outputs_and_preview():
    example = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "vision"
            / "video_temporal_reasoning_api.json"
        ).read_text(encoding="utf-8")
    )
    assert example["3"]["class_type"] == "VLMVideoTemporalReasoner"
    assert example["5"]["inputs"]["text"] == ["3", 6]
    assert example["6"]["inputs"]["text"] == ["3", 7]
    assert example["8"]["inputs"]["images"] == ["3", 3]
