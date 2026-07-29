import importlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1].name
vision_types = importlib.import_module(f"{PACKAGE}.nodes.vision_types")

DETECTIONS_SCHEMA = vision_types.DETECTIONS_SCHEMA
EVENTS_SCHEMA = vision_types.EVENTS_SCHEMA
POINTS_SCHEMA = vision_types.POINTS_SCHEMA
SCHEMA_VERSION = vision_types.SCHEMA_VERSION
SCENE_STATE_SCHEMA = vision_types.SCENE_STATE_SCHEMA
TRACKS_SCHEMA = vision_types.TRACKS_SCHEMA
VIDEO_SELECTION_SCHEMA = vision_types.VIDEO_SELECTION_SCHEMA
VLM_DETECTIONS = vision_types.VLM_DETECTIONS
VLM_EVENTS = vision_types.VLM_EVENTS
VLM_POINTS = vision_types.VLM_POINTS
VLM_SCENE_STATE = vision_types.VLM_SCENE_STATE
VLM_TRACKS = vision_types.VLM_TRACKS
VLM_VIDEO_SELECTION = vision_types.VLM_VIDEO_SELECTION
Detection = vision_types.Detection
DetectionSequence = vision_types.DetectionSequence
EventSequence = vision_types.EventSequence
FrameDetections = vision_types.FrameDetections
FrozenDict = vision_types.FrozenDict
PointSequence = vision_types.PointSequence
TemporalEvent = vision_types.TemporalEvent
Track = vision_types.Track
TrackSequence = vision_types.TrackSequence
VisionPoint = vision_types.VisionPoint


def sample_sequence() -> DetectionSequence:
    mask = torch.zeros((24, 32), dtype=torch.float32)
    mask[3:12, 4:18] = 1
    first = Detection(
        bbox_xyxy=(4, 3, 18, 12),
        label="cat",
        text="sleeping cat",
        score=0.875,
        polygon=((4, 3), (18, 3), (18, 12), (4, 12)),
        quad=((4, 3), (18, 3), (18, 12), (4, 12)),
        frame_index=0,
        timestamp=0.0,
        track_id=7,
        source="unit-test",
        metadata={"attributes": ["small", "red"], "visible": True},
        mask=mask,
    )
    second = Detection(
        bbox_xyxy=(6.5, 5.0, 20.25, 15.0),
        label="cat",
        score=0.8,
        frame_index=1,
        timestamp=0.5,
        track_id=7,
    )
    return DetectionSequence(
        width=32,
        height=24,
        frame_count=2,
        fps=2.0,
        source="synthetic",
        metadata={"nested": {"value": 3}},
        frames=(
            FrameDetections(
                frame_index=0,
                timestamp=0.0,
                width=32,
                height=24,
                detections=(first,),
            ),
            FrameDetections(
                frame_index=1,
                timestamp=0.5,
                width=32,
                height=24,
                detections=(second,),
            ),
        ),
    )


def test_public_socket_and_schema_names_are_stable():
    assert VLM_DETECTIONS == "VLM_DETECTIONS"
    assert VLM_TRACKS == "VLM_TRACKS"
    assert VLM_POINTS == "VLM_POINTS"
    assert VLM_EVENTS == "VLM_EVENTS"
    assert VLM_VIDEO_SELECTION == "VLM_VIDEO_SELECTION"
    assert VLM_SCENE_STATE == "VLM_SCENE_STATE"
    assert SCHEMA_VERSION == 1
    assert DETECTIONS_SCHEMA == "comfyui-vlm/detections"
    assert TRACKS_SCHEMA == "comfyui-vlm/tracks"
    assert POINTS_SCHEMA == "comfyui-vlm/points"
    assert EVENTS_SCHEMA == "comfyui-vlm/events"
    assert VIDEO_SELECTION_SCHEMA == "comfyui-vlm/video-selection"
    assert SCENE_STATE_SCHEMA == "comfyui-vlm/scene-state"


def test_detection_payload_is_validated_immutable_and_mask_safe():
    original = torch.ones((4, 5))
    detection = Detection(
        bbox_xyxy=(0, 0, 5, 4),
        label="object",
        score=1.0,
        metadata={"items": [1, {"ready": True}]},
        mask=original,
    )
    original.zero_()
    assert detection.mask.sum().item() == 20
    assert isinstance(detection.metadata, FrozenDict)
    assert detection.metadata["items"][1]["ready"] is True

    with pytest.raises(FrozenInstanceError):
        detection.label = "changed"
    with pytest.raises(TypeError):
        detection.metadata["new"] = "value"
    with pytest.raises(TypeError, match="Metadata"):
        Detection(
            bbox_xyxy=(0, 0, 1, 1),
            metadata={"tensor": torch.ones(1)},
        )

    record = detection.to_dict()
    assert "mask" not in record
    assert "mask" not in json.dumps(record)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"bbox_xyxy": (2, 0, 1, 2)}, "x2"),
        ({"bbox_xyxy": (-1, 0, 1, 2)}, "non-negative"),
        ({"bbox_xyxy": (0, 0, 1, 2), "score": 1.1}, "between 0 and 1"),
        ({"bbox_xyxy": (0, 0, 1, 2), "frame_index": -1}, "frame_index"),
        ({"bbox_xyxy": (0, 0, 1, 2), "timestamp": -0.1}, "timestamp"),
        (
            {"bbox_xyxy": (0, 0, 1, 2), "quad": ((0, 0), (1, 0), (1, 1))},
            "exactly 4",
        ),
        (
            {"bbox_xyxy": (0, 0, 1, 2), "mask": torch.ones(1, 2, 3)},
            "shape",
        ),
    ],
)
def test_detection_rejects_invalid_values(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        Detection(**kwargs)


def test_detection_sequence_json_round_trip_is_versioned_and_tensor_free():
    sequence = sample_sequence()
    encoded = sequence.to_json(indent=2)
    decoded_json = json.loads(encoded)
    assert decoded_json["schema"] == DETECTIONS_SCHEMA
    assert decoded_json["version"] == SCHEMA_VERSION
    assert decoded_json["media"] == {
        "fps": 2.0,
        "frame_count": 2,
        "height": 24,
        "width": 32,
    }
    assert "mask" not in encoded

    restored = DetectionSequence.from_json(encoded)
    assert restored.to_dict() == sequence.to_dict()
    assert restored.frames[0].detections[0].mask is None
    assert restored.all_detections()[1].center == pytest.approx((13.375, 10.0))
    assert restored.frame(99) is None

    decoded_json["version"] = 99
    with pytest.raises(ValueError, match="Unsupported"):
        DetectionSequence.from_dict(decoded_json)
    decoded_json["version"] = SCHEMA_VERSION
    decoded_json["schema"] = "other"
    with pytest.raises(ValueError, match="Expected schema"):
        DetectionSequence.from_dict(decoded_json)


def test_frame_and_sequence_enforce_dimensions_order_and_timestamps():
    detection = Detection(
        bbox_xyxy=(0, 0, 11, 5),
        frame_index=0,
        timestamp=0,
    )
    with pytest.raises(ValueError, match="width"):
        FrameDetections(
            frame_index=0,
            timestamp=0,
            width=10,
            height=10,
            detections=(detection,),
        )
    mismatch = Detection(
        bbox_xyxy=(0, 0, 1, 1),
        frame_index=1,
        timestamp=0,
    )
    with pytest.raises(ValueError, match="frame_index"):
        FrameDetections(
            frame_index=0,
            timestamp=0,
            width=10,
            height=10,
            detections=(mismatch,),
        )

    frame_one = FrameDetections(1, 1.0, 10, 10)
    frame_zero = FrameDetections(0, 0.0, 10, 10)
    with pytest.raises(ValueError, match="increasing"):
        DetectionSequence(10, 10, frames=(frame_one, frame_zero))


def test_point_track_and_event_schemas_round_trip_without_tensors():
    sequence = sample_sequence()
    points = PointSequence(
        width=32,
        height=24,
        frame_count=2,
        fps=2,
        points=(
            VisionPoint(
                x=11,
                y=7.5,
                label="cat",
                score=0.8,
                frame_index=0,
                track_id=7,
            ),
        ),
    )
    assert PointSequence.from_json(points.to_json()).to_dict() == points.to_dict()

    track = Track(
        track_id=7,
        label="cat",
        score=0.8,
        detections=sequence.all_detections(),
    )
    tracks = TrackSequence(
        width=32,
        height=24,
        frame_count=2,
        fps=2,
        tracks=(track,),
    )
    restored_tracks = TrackSequence.from_json(tracks.to_json())
    assert restored_tracks.to_dict() == tracks.to_dict()
    assert "mask" not in tracks.to_json()

    events = EventSequence(
        duration=2.0,
        events=(
            TemporalEvent(
                start_time=0.25,
                end_time=1.5,
                label="movement",
                text="the cat moves",
                score=0.9,
            ),
        ),
    )
    assert EventSequence.from_json(events.to_json()).to_dict() == events.to_dict()
    with pytest.raises(ValueError, match="duration"):
        EventSequence(
            duration=1.0,
            events=(TemporalEvent(0.0, 2.0),),
        )
