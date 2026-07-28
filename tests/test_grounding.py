from types import SimpleNamespace

import torch
from ComfyUI_VLM_nodes.nodes.grounding import (
    MODEL_SPECS,
    VLMOpenVocabularyDetection,
    core_bounding_box_frames,
    core_bounding_boxes,
    detection_box_masks,
    parse_labels,
    result_to_detections,
)
from ComfyUI_VLM_nodes.nodes.vision_types import (
    DetectionSequence,
    FrameDetections,
)


def test_detector_catalog_is_small_fast_and_portable():
    assert "Grounding DINO Tiny (fast)" in MODEL_SPECS
    assert "OmDet Turbo Swin Tiny (fast)" in MODEL_SPECS
    assert all("/" in spec.model_id for spec in MODEL_SPECS.values())
    schema = VLMOpenVocabularyDetection.INPUT_TYPES()
    assert tuple(MODEL_SPECS) == schema["required"]["model"][0]


def test_label_parser_preserves_phrases_and_removes_duplicates():
    assert parse_labels("red car, person\nsmall dog;person") == [
        "red car",
        "person",
        "small dog",
    ]


def test_transformers_results_are_clipped_sorted_and_normalized():
    result = {
        "boxes": torch.tensor([[-5.0, 2.0, 20.0, 12.0], [5.0, 5.0, 9.0, 9.0]]),
        "scores": torch.tensor([0.25, 0.9]),
        "text_labels": ["cat", "dog"],
    }
    detections = result_to_detections(
        result,
        labels=["cat", "dog"],
        width=16,
        height=10,
        frame_index=0,
        timestamp=0.0,
        source="test/model",
        max_detections=20,
    )
    assert [item.label for item in detections] == ["dog", "cat"]
    assert detections[1].bbox_xyxy == (0.0, 2.0, 16.0, 10.0)
    assert detections[0].metadata["model_id"] == "test/model"


def test_max_detections_is_applied_after_confidence_sorting():
    detections = result_to_detections(
        {
            "boxes": [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]],
            "scores": [0.1, 0.9, 0.8],
            "text_labels": ["low", "best", "second"],
        },
        labels=["low", "best", "second"],
        width=4,
        height=4,
        frame_index=0,
        timestamp=0.0,
        source="test/model",
        max_detections=2,
    )
    assert [item.label for item in detections] == ["best", "second"]


def test_box_masks_and_core_boxes_keep_geometry_and_metadata():
    detections = result_to_detections(
        {
            "boxes": torch.tensor([[1.0, 2.0, 4.0, 5.0]]),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([0]),
        },
        labels=["cat"],
        width=8,
        height=6,
        frame_index=0,
        timestamp=0.0,
        source="test/model",
        max_detections=5,
    )
    sequence = DetectionSequence(
        width=8,
        height=6,
        frames=(
            FrameDetections(
                frame_index=0,
                timestamp=0.0,
                width=8,
                height=6,
                detections=detections,
            ),
        ),
        frame_count=1,
    )
    masks = detection_box_masks(sequence)
    assert masks.shape == (1, 6, 8)
    assert masks.sum().item() == 9
    boxes = core_bounding_boxes(sequence)
    assert boxes == [
        {
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 3,
            "label": "cat",
            "score": detections[0].score,
            "metadata": {
                "frame_index": 0,
                "label": "cat",
                "score": detections[0].score,
                "source": "test/model",
            },
        }
    ]
    assert core_bounding_box_frames(sequence) == [boxes]


def test_result_label_indices_are_resolved():
    detections = result_to_detections(
        {
            "boxes": [[0, 0, 4, 4]],
            "scores": [SimpleNamespace(item=lambda: 0.5)],
            "classes": [SimpleNamespace(item=lambda: 1)],
        },
        labels=["cat", "dog"],
        width=4,
        height=4,
        frame_index=0,
        timestamp=0.0,
        source="omdet",
        max_detections=1,
    )
    assert detections[0].label == "dog"
