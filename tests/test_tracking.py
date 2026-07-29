from __future__ import annotations

import torch
from ComfyUI_VLM_nodes.nodes.tracking import (
    VLMByteTracker,
    associate_detection_sequence,
)
from ComfyUI_VLM_nodes.nodes.vision_types import (
    Detection,
    DetectionSequence,
    FrameDetections,
)


def _frame(
    frame_index,
    detections,
    *,
    width=100,
    height=100,
    fps=10.0,
):
    timestamp = frame_index / fps
    records = tuple(
        Detection(
            bbox_xyxy=record["box"],
            label=record.get("label"),
            score=record.get("score"),
            frame_index=frame_index,
            timestamp=timestamp,
            mask=record.get("mask"),
        )
        for record in detections
    )
    return FrameDetections(
        frame_index=frame_index,
        timestamp=timestamp,
        width=width,
        height=height,
        detections=records,
    )


def _sequence(frames, *, frame_count=None, fps=10.0, width=100, height=100):
    return DetectionSequence(
        width=width,
        height=height,
        frames=tuple(frames),
        frame_count=frame_count or 0,
        fps=fps,
        source="test-detector",
    )


def test_low_confidence_second_stage_keeps_the_track_id():
    sequence = _sequence(
        (
            _frame(
                0,
                [{"box": (10, 10, 30, 30), "score": 0.9, "label": "cat"}],
            ),
            _frame(
                1,
                [{"box": (11, 10, 31, 30), "score": 0.25, "label": "cat"}],
            ),
        )
    )
    tracks = associate_detection_sequence(
        sequence,
        high_threshold=0.6,
        low_threshold=0.1,
        min_hits=1,
    )
    assert len(tracks.tracks) == 1
    track = tracks.tracks[0]
    assert track.track_id == 0
    assert [item.track_id for item in track.detections] == [0, 0]
    assert track.detections[1].metadata["association_stage"] == "low"
    assert track.metadata["state"] == "active"


def test_low_confidence_detection_cannot_start_a_track():
    sequence = _sequence(
        (
            _frame(
                0,
                [{"box": (10, 10, 30, 30), "score": 0.2, "label": "cat"}],
            ),
        )
    )
    tracks = associate_detection_sequence(
        sequence,
        high_threshold=0.6,
        low_threshold=0.1,
        min_hits=1,
    )
    assert tracks.tracks == ()


def test_label_aware_matching_prevents_cross_class_identity_reuse():
    frames = (
        _frame(
            0,
            [{"box": (10, 10, 30, 30), "score": 0.9, "label": "cat"}],
        ),
        _frame(
            1,
            [{"box": (10, 10, 30, 30), "score": 0.9, "label": "dog"}],
        ),
    )
    label_aware = associate_detection_sequence(
        _sequence(frames),
        min_hits=1,
        label_aware=True,
        emit_predictions=False,
    )
    class_agnostic = associate_detection_sequence(
        _sequence(frames),
        min_hits=1,
        label_aware=False,
        emit_predictions=False,
    )
    assert len(label_aware.tracks) == 2
    assert [track.label for track in label_aware.tracks] == ["cat", "dog"]
    assert len(class_agnostic.tracks) == 1
    assert [
        detection.frame_index for detection in class_agnostic.tracks[0].detections
    ] == [0, 1]


def test_max_age_seconds_uses_fps_and_marks_removed_deterministically():
    sequence = _sequence(
        (
            _frame(
                0,
                [{"box": (10, 10, 30, 30), "score": 0.9}],
                fps=2.0,
            ),
        ),
        frame_count=4,
        fps=2.0,
    )
    tracks = associate_detection_sequence(
        sequence,
        min_hits=1,
        max_age_seconds=1.0,
        emit_predictions=True,
    )
    track = tracks.tracks[0]
    assert [item.frame_index for item in track.detections] == [0, 1, 2]
    assert [item.metadata["observation"] for item in track.detections] == [
        "detected",
        "predicted",
        "predicted",
    ]
    assert track.metadata["state"] == "removed"
    assert track.metadata["removed_frame"] == 3
    assert track.metadata["last_observed_frame"] == 0


def test_mask_iou_can_rescue_a_zero_bbox_iou_match():
    mask = torch.zeros(100, 100)
    mask[40:60, 40:60] = 1
    sequence = _sequence(
        (
            _frame(
                0,
                [
                    {
                        "box": (0, 0, 10, 10),
                        "score": 0.9,
                        "mask": mask,
                    }
                ],
            ),
            _frame(
                1,
                [
                    {
                        "box": (20, 0, 30, 10),
                        "score": 0.9,
                        "mask": mask,
                    }
                ],
            ),
        )
    )
    tracks = VLMByteTracker(
        min_hits=1,
        motion_gate=1.0e12,
    ).track(sequence)
    assert len(tracks.tracks) == 1
    assert len(tracks.tracks[0].detections) == 2


def test_hungarian_results_and_serialization_are_repeatable():
    sequence = _sequence(
        (
            _frame(
                0,
                [
                    {"box": (5, 5, 20, 20), "score": 0.9},
                    {"box": (40, 5, 55, 20), "score": 0.9},
                ],
            ),
            _frame(
                1,
                [
                    {"box": (41, 5, 56, 20), "score": 0.9},
                    {"box": (6, 5, 21, 20), "score": 0.9},
                ],
            ),
        )
    )
    options = {"min_hits": 1, "emit_predictions": False}
    first = associate_detection_sequence(sequence, **options)
    second = associate_detection_sequence(sequence, **options)
    assert first.to_json() == second.to_json()
    assert [
        [detection.bbox_xyxy for detection in track.detections]
        for track in first.tracks
    ] == [
        [(5.0, 5.0, 20.0, 20.0), (6.0, 5.0, 21.0, 20.0)],
        [(40.0, 5.0, 55.0, 20.0), (41.0, 5.0, 56.0, 20.0)],
    ]


def test_tracker_rejects_invalid_threshold_order():
    try:
        VLMByteTracker(high_threshold=0.2, low_threshold=0.3)
    except ValueError as error:
        assert "low_threshold" in str(error)
    else:
        raise AssertionError("Expected invalid threshold order to fail.")
