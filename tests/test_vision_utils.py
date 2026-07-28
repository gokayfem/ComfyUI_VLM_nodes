import importlib
import inspect
import json
from pathlib import Path

import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1].name
vision_types = importlib.import_module(f"{PACKAGE}.nodes.vision_types")
vision_utils = importlib.import_module(f"{PACKAGE}.nodes.vision_utils")

VLM_DETECTIONS = vision_types.VLM_DETECTIONS
VLM_POINTS = vision_types.VLM_POINTS
Detection = vision_types.Detection
DetectionSequence = vision_types.DetectionSequence
FrameDetections = vision_types.FrameDetections
BOUNDING_BOXES = vision_utils.BOUNDING_BOXES
NODE_CLASS_MAPPINGS = vision_utils.NODE_CLASS_MAPPINGS
VLMCropDetections = vision_utils.VLMCropDetections
VLMDetectionsFromJSON = vision_utils.VLMDetectionsFromJSON
VLMDetectionsToBoundingBoxes = vision_utils.VLMDetectionsToBoundingBoxes
VLMDetectionsToJSON = vision_utils.VLMDetectionsToJSON
VLMDetectionsToMasks = vision_utils.VLMDetectionsToMasks
VLMDetectionsToPoints = vision_utils.VLMDetectionsToPoints
VLMFilterDetections = vision_utils.VLMFilterDetections
VLMRenderDetections = vision_utils.VLMRenderDetections
VLMSelectDetection = vision_utils.VLMSelectDetection
bounding_boxes_payload = vision_utils.bounding_boxes_payload
crop_detections = vision_utils.crop_detections
detection_centers = vision_utils.detection_centers
filter_detection_sequence = vision_utils.filter_detection_sequence
render_detections = vision_utils.render_detections
select_detection_sequence = vision_utils.select_detection_sequence
sequence_masks = vision_utils.sequence_masks


def sample_sequence() -> DetectionSequence:
    cat_mask = torch.zeros((16, 20))
    cat_mask[2:8, 3:10] = 1
    return DetectionSequence(
        width=20,
        height=16,
        frame_count=2,
        fps=2,
        frames=(
            FrameDetections(
                frame_index=0,
                timestamp=0,
                width=20,
                height=16,
                detections=(
                    Detection(
                        bbox_xyxy=(3.2, 2.1, 10.0, 8.0),
                        label="cat",
                        score=0.9,
                        frame_index=0,
                        timestamp=0,
                        track_id=5,
                        mask=cat_mask,
                    ),
                    Detection(
                        bbox_xyxy=(12, 4, 18, 13),
                        label="dog",
                        score=0.55,
                        frame_index=0,
                        timestamp=0,
                    ),
                ),
            ),
            FrameDetections(
                frame_index=1,
                timestamp=0.5,
                width=20,
                height=16,
                detections=(
                    Detection(
                        bbox_xyxy=(5, 3, 12, 10),
                        label="cat",
                        score=0.8,
                        frame_index=1,
                        timestamp=0.5,
                        track_id=5,
                    ),
                ),
            ),
        ),
    )


def empty_sequence() -> DetectionSequence:
    return DetectionSequence(
        width=20,
        height=16,
        frame_count=2,
        frames=(
            FrameDetections(0, 0, 20, 16),
            FrameDetections(1, 1, 20, 16),
        ),
    )


def test_filter_and_select_by_all_supported_fields():
    sequence = sample_sequence()
    filtered = filter_detection_sequence(
        sequence,
        label="CAT",
        label_mode="exact",
        minimum_score=0.85,
        minimum_area=30,
        maximum_area=50,
        track_id=5,
        frame_index=0,
    )
    assert len(filtered.frames) == 1
    assert [item.label for item in filtered.all_detections()] == ["cat"]

    contains = filter_detection_sequence(
        sequence,
        label="o",
        minimum_score=0.5,
    )
    assert [item.label for item in contains.all_detections()] == ["dog"]

    selected = select_detection_sequence(sequence, 1)
    assert [item.label for item in selected.all_detections()] == ["dog"]
    missing = select_detection_sequence(sequence, 100)
    assert missing.all_detections() == ()
    assert len(missing.frames) == 2

    with pytest.raises(ValueError, match="maximum_area"):
        filter_detection_sequence(
            sequence,
            minimum_area=10,
            maximum_area=5,
        )


def test_core_bounding_boxes_are_integer_xywh_with_metadata():
    payload = bounding_boxes_payload(sample_sequence())
    assert payload[0] == {
        "x": 3,
        "y": 2,
        "width": 7,
        "height": 6,
        "metadata": {
            "bbox_xyxy": [3.2, 2.1, 10.0, 8.0],
            "frame_index": 0,
            "timestamp": 0.0,
            "label": "cat",
            "score": 0.9,
            "track_id": 5,
        },
    }
    assert payload[2]["x"] == 5
    assert payload[2]["metadata"]["frame_index"] == 1
    assert bounding_boxes_payload(empty_sequence()) == []


def test_centers_and_masks_preserve_frame_mapping_and_empty_shapes():
    sequence = sample_sequence()
    points = detection_centers(sequence)
    assert points.points[0].x == pytest.approx(6.6)
    assert points.points[0].y == pytest.approx(5.05)
    assert points.points[0].track_id == 5
    assert json.loads(points.to_json())["schema"] == "comfyui-vlm/points"

    unions, individuals, mapping = sequence_masks(sequence)
    assert unions.shape == (2, 16, 20)
    assert individuals.shape == (3, 16, 20)
    assert unions[0].sum() > 0
    assert mapping[0] == {
        "mask_index": 0,
        "frame_index": 0,
        "detection_index": 0,
        "label": "cat",
        "track_id": 5,
    }

    empty_unions, empty_individuals, empty_mapping = sequence_masks(empty_sequence())
    assert empty_unions.shape == (2, 16, 20)
    assert empty_unions.sum() == 0
    assert empty_individuals.shape == (0, 16, 20)
    assert empty_mapping == []


def test_rendering_is_deterministic_batch_safe_and_empty_safe():
    image = torch.zeros((2, 16, 20, 3), dtype=torch.float32)
    sequence = sample_sequence()
    first = render_detections(
        image,
        sequence,
        draw_masks=True,
        draw_labels=True,
    )
    second = render_detections(
        image,
        sequence,
        draw_masks=True,
        draw_labels=True,
    )
    assert first.shape == image.shape
    assert torch.equal(first, second)
    assert first.sum() > 0

    unchanged = render_detections(image, empty_sequence())
    assert torch.equal(unchanged, image)
    with pytest.raises(ValueError, match="dimensions"):
        render_detections(
            torch.zeros((2, 10, 10, 3)),
            sequence,
        )


def test_padded_square_crops_form_a_non_distorted_image_batch():
    image = torch.zeros((2, 16, 20, 3), dtype=torch.float32)
    image[0, 2:8, 3:10, 0] = 1
    image[0, 4:13, 12:18, 1] = 1
    image[1, 3:10, 5:12, 2] = 1
    crops, metadata = crop_detections(
        image,
        sample_sequence(),
        padding=1,
        square=True,
    )
    assert crops.shape[0] == 3
    assert crops.ndim == 4
    assert crops.shape[1] == max(record["valid_height"] for record in metadata)
    assert crops.shape[2] == max(record["valid_width"] for record in metadata)
    assert all(record["batch_width"] == crops.shape[2] for record in metadata)
    assert metadata[0]["track_id"] == 5
    valid_crop = crops[
        0,
        : metadata[0]["valid_height"],
        : metadata[0]["valid_width"],
    ]
    assert valid_crop.sum() > 0

    empty_crops, empty_metadata = crop_detections(image, empty_sequence())
    assert empty_crops.shape == (0, 1, 1, 3)
    assert empty_metadata == []


def test_utility_node_contracts_and_json_round_trip():
    sequence = sample_sequence()
    encoded = VLMDetectionsToJSON().serialize(sequence, pretty=False)[0]
    restored = VLMDetectionsFromJSON().parse(encoded)[0]
    assert restored.to_dict() == sequence.to_dict()

    filtered = VLMFilterDetections().filter(
        sequence,
        label="cat",
        label_mode="exact",
        minimum_score=0,
        minimum_area=0,
        maximum_area=0,
        track_id=-1,
        frame_index=-1,
    )[0]
    assert len(filtered.all_detections()) == 2
    assert len(VLMSelectDetection().select(sequence, 0)[0].all_detections()) == 1

    boxes, boxes_json = VLMDetectionsToBoundingBoxes().convert(sequence)
    assert json.loads(boxes_json) == boxes
    points, points_json = VLMDetectionsToPoints().convert(sequence)
    assert json.loads(points_json) == points.to_dict()
    union, individual, mask_json = VLMDetectionsToMasks().convert(sequence)
    assert union.shape[0] == 2
    assert individual.shape[0] == 3
    assert len(json.loads(mask_json)) == 3

    image = torch.zeros((2, 16, 20, 3))
    assert (
        VLMRenderDetections()
        .render(
            image,
            sequence,
            True,
            False,
            0.25,
            2,
        )[0]
        .shape
        == image.shape
    )
    crops, crop_json = VLMCropDetections().crop(image, sequence, 0, False)
    assert crops.shape[0] == 3
    assert len(json.loads(crop_json)) == 3


def test_every_utility_node_accepts_its_declared_inputs():
    expected = {
        "VLMDetectionsFromJSON",
        "VLMDetectionsToJSON",
        "VLMFilterDetections",
        "VLMSelectDetection",
        "VLMDetectionsToBoundingBoxes",
        "VLMDetectionsToPoints",
        "VLMDetectionsToMasks",
        "VLMRenderDetections",
        "VLMCropDetections",
    }
    assert set(NODE_CLASS_MAPPINGS) == expected
    assert VLMDetectionsFromJSON.RETURN_TYPES == (VLM_DETECTIONS,)
    assert VLMDetectionsToBoundingBoxes.RETURN_TYPES[0] == BOUNDING_BOXES
    assert VLMDetectionsToPoints.RETURN_TYPES[0] == VLM_POINTS

    for node_class in NODE_CLASS_MAPPINGS.values():
        schema = node_class.INPUT_TYPES()
        declared = {
            name
            for group in schema.values()
            if isinstance(group, dict)
            for name in group
        }
        function = getattr(node_class, node_class.FUNCTION)
        accepted = set(inspect.signature(function).parameters)
        assert declared <= accepted, (
            node_class.__name__,
            declared - accepted,
        )
