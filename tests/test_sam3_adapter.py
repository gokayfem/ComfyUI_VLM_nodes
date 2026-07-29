from __future__ import annotations

import json

import pytest
import torch
from ComfyUI_VLM_nodes.nodes.sam3_adapter import (
    VLMTrackReport,
    iter_sam3_masks,
    sam3_track_data_to_tracks,
    track_report_json,
    track_report_payload,
    track_report_text,
    unpack_sam3_mask,
    validate_sam3_track_data,
)
from ComfyUI_VLM_nodes.nodes.vision_types import (
    Detection,
    DetectionSequence,
    FrameDetections,
)


def _pack_masks(masks):
    masks = masks.to(torch.uint8)
    width = masks.shape[-1]
    assert width % 8 == 0
    bits = 1 << torch.arange(8, dtype=torch.int64)
    grouped = masks.reshape(*masks.shape[:-1], width // 8, 8)
    return (grouped * bits).sum(dim=-1).to(torch.uint8)


def _sample_track_data():
    masks = torch.zeros(2, 2, 4, 8, dtype=torch.bool)
    masks[0, 0, 1:3, 2:5] = True
    masks[1, 0, 1:4, 3:6] = True
    masks[1, 1, 0:2, 0:2] = True
    return {
        "packed_masks": _pack_masks(masks),
        "n_frames": 2,
        "scores": [0.9, 0.75],
        "orig_size": (40, 80),
    }


def _seed_detections():
    detections = (
        Detection(
            bbox_xyxy=(20, 10, 50, 30),
            label="cat",
            text="the cat",
            score=0.95,
            frame_index=0,
            timestamp=0.0,
            track_id=7,
            source="seed",
        ),
        Detection(
            bbox_xyxy=(0, 0, 20, 20),
            label="fish",
            score=0.8,
            frame_index=0,
            timestamp=0.0,
            track_id=9,
            source="seed",
        ),
    )
    return DetectionSequence(
        width=80,
        height=40,
        frames=(
            FrameDetections(
                frame_index=0,
                timestamp=0.0,
                width=80,
                height=40,
                detections=detections,
            ),
        ),
        frame_count=2,
        fps=10.0,
    )


def test_validate_and_stream_unpack_without_expanding_the_video():
    track_data = _sample_track_data()
    layout = validate_sam3_track_data(track_data)
    assert (
        layout.n_frames,
        layout.n_objects,
        layout.mask_height,
        layout.mask_width,
    ) == (2, 2, 4, 8)
    yielded = list(iter_sam3_masks(track_data, present_only=True))
    assert [(frame, obj) for frame, obj, _mask in yielded] == [
        (0, 0),
        (1, 0),
        (1, 1),
    ]
    assert all(mask.shape == (4, 8) for _frame, _obj, mask in yielded)
    assert yielded[0][2].sum().item() == 6
    one = unpack_sam3_mask(track_data["packed_masks"][1, 1])
    assert one.dtype == torch.bool
    assert one.sum().item() == 4


def test_adapter_derives_scaled_boxes_and_preserves_seed_identity():
    tracks = sam3_track_data_to_tracks(
        _sample_track_data(),
        seed_detections=_seed_detections(),
        fps=10.0,
    )
    assert (tracks.width, tracks.height, tracks.frame_count, tracks.fps) == (
        80,
        40,
        2,
        10.0,
    )
    assert [track.track_id for track in tracks.tracks] == [7, 9]
    assert [track.label for track in tracks.tracks] == ["cat", "fish"]
    cat, fish = tracks.tracks
    assert cat.detections[0].bbox_xyxy == (20.0, 10.0, 50.0, 30.0)
    assert cat.detections[1].bbox_xyxy == (30.0, 10.0, 60.0, 40.0)
    assert fish.detections[0].frame_index == 1
    assert fish.detections[0].bbox_xyxy == (0.0, 0.0, 20.0, 20.0)
    assert cat.detections[0].metadata["mask_ref"]["object_index"] == 0
    assert fish.detections[0].metadata["mask_ref"]["object_index"] == 1
    assert cat.metadata["seeded"] is True
    assert fish.metadata["seeded"] is True

    serialized = tracks.to_json()
    assert "mask_ref" in serialized
    assert "packed_masks" not in serialized
    assert "tensor" not in serialized.casefold()


def test_adapter_handles_an_empty_core_result():
    tracks = sam3_track_data_to_tracks(
        {
            "packed_masks": None,
            "n_frames": 3,
            "scores": [],
            "orig_size": (48, 64),
        },
        fps=24.0,
    )
    assert tracks.tracks == ()
    assert tracks.frame_count == 3
    assert tracks.metadata["object_slots"] == 0


@pytest.mark.parametrize(
    ("track_data", "message"),
    (
        (
            {"packed_masks": None, "n_frames": 1, "scores": []},
            "orig_size",
        ),
        (
            {
                "packed_masks": torch.zeros(1, 1, 2, 1),
                "n_frames": 1,
                "scores": [0.5],
                "orig_size": (2, 8),
            },
            "uint8",
        ),
        (
            {
                "packed_masks": torch.zeros(2, 1, 2, 1, dtype=torch.uint8),
                "n_frames": 1,
                "scores": [0.5],
                "orig_size": (2, 8),
            },
            "n_frames",
        ),
        (
            {
                "packed_masks": torch.zeros(1, 1, 2, 1, dtype=torch.uint8),
                "n_frames": 1,
                "scores": [1.5],
                "orig_size": (2, 8),
            },
            "0 to 1",
        ),
    ),
)
def test_adapter_rejects_incompatible_private_payloads(track_data, message):
    with pytest.raises((TypeError, ValueError), match=message):
        validate_sam3_track_data(track_data)


def test_track_report_is_small_deterministic_and_history_safe():
    tracks = sam3_track_data_to_tracks(
        _sample_track_data(),
        seed_detections=_seed_detections(),
        fps=10.0,
    )
    payload = track_report_payload(tracks)
    assert payload["track_count"] == 2
    assert payload["observation_count"] == 3
    assert payload["state_counts"] == {"active": 2}
    encoded = track_report_json(tracks)
    assert json.loads(encoded) == payload
    assert "packed_masks" not in encoded
    text = track_report_text(tracks)
    assert "Tracks: 2" in text
    assert "#7 cat" in text
    assert "#9 fish" in text

    node_result = VLMTrackReport().report(tracks)
    assert node_result["result"] == (encoded, text)
    assert node_result["ui"]["text"] == [text]
