import importlib
from pathlib import Path

import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1].name
geometry = importlib.import_module(f"{PACKAGE}.nodes.geometry")
vision_types = importlib.import_module(f"{PACKAGE}.nodes.vision_types")

associate_detections = geometry.associate_detections
bbox_from_mask = geometry.bbox_from_mask
bbox_iou = geometry.bbox_iou
box_area = geometry.box_area
box_center = geometry.box_center
box_to_mask = geometry.box_to_mask
clip_box = geometry.clip_box
clip_polygon = geometry.clip_polygon
denormalize_box = geometry.denormalize_box
detection_to_mask = geometry.detection_to_mask
deterministic_color = geometry.deterministic_color
expand_box = geometry.expand_box
individual_detection_masks = geometry.individual_detection_masks
mask_iou = geometry.mask_iou
normalize_box = geometry.normalize_box
polygon_area = geometry.polygon_area
polygon_to_mask = geometry.polygon_to_mask
quad_to_mask = geometry.quad_to_mask
translate_box = geometry.translate_box
union_detection_mask = geometry.union_detection_mask
Detection = vision_types.Detection


def test_box_clipping_normalization_area_and_center():
    assert clip_box((0, 2, 25, 22), 20, 10) == (0, 2, 20, 10)
    normalized = normalize_box((5, 2, 15, 8), 20, 10)
    assert normalized == pytest.approx((0.25, 0.2, 0.75, 0.8))
    assert denormalize_box(normalized, 20, 10) == pytest.approx((5, 2, 15, 8))
    assert box_area((5, 2, 15, 8)) == 60
    assert box_center((5, 2, 15, 8)) == (10, 5)
    with pytest.raises(ValueError, match="Normalized"):
        denormalize_box((0, 0, 2, 1), 20, 10)
    with pytest.raises(ValueError, match="x2"):
        clip_box((2, 0, 1, 1), 20, 10)


def test_polygon_clipping_and_area():
    polygon = ((-2, -3), (8, 0), (8, 5), (0, 5))
    assert clip_polygon(polygon, 6, 4) == (
        (0, 0),
        (6, 0),
        (6, 4),
        (0, 4),
    )
    assert polygon_area(((0, 0), (5, 0), (5, 4), (0, 4))) == 20
    assert polygon_area(((0, 0), (0, 4), (5, 4), (5, 0))) == 20
    with pytest.raises(ValueError, match="at least three"):
        polygon_area(((0, 0), (1, 1)))


def test_bbox_and_mask_iou():
    assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    first = torch.zeros((4, 4))
    second = torch.zeros((4, 4))
    first[:2, :2] = 1
    second[1:3, :2] = 1
    assert mask_iou(first, second) == pytest.approx(1 / 3)
    assert mask_iou(torch.zeros((2, 2)), torch.zeros((2, 2))) == 0
    with pytest.raises(ValueError, match="same shape"):
        mask_iou(torch.zeros((2, 2)), torch.zeros((3, 2)))


def test_box_polygon_and_quad_rasterization():
    box = box_to_mask((1.2, 2.1, 4.1, 5.2), 8, 7)
    assert box.shape == (7, 8)
    assert box.sum().item() == 16
    polygon = polygon_to_mask(((1, 1), (5, 1), (5, 5), (1, 5)), 8, 8)
    quad = quad_to_mask(((1, 1), (5, 1), (5, 5), (1, 5)), 8, 8)
    assert torch.equal(polygon, quad)
    assert polygon.sum() > 0
    with pytest.raises(ValueError, match="exactly four"):
        quad_to_mask(((0, 0), (1, 0), (1, 1)), 4, 4)


def test_detection_mask_priority_union_individual_and_bbox():
    explicit = torch.zeros((8, 8))
    explicit[3:6, 2:5] = 1
    with_mask = Detection(
        bbox_xyxy=(0, 0, 8, 8),
        polygon=((0, 0), (8, 0), (8, 8), (0, 8)),
        mask=explicit,
    )
    polygon_only = Detection(
        bbox_xyxy=(1, 1, 5, 5),
        polygon=((1, 1), (5, 1), (5, 5), (1, 5)),
    )
    assert torch.equal(detection_to_mask(with_mask, 8, 8), explicit)
    masks = individual_detection_masks((with_mask, polygon_only), 8, 8)
    assert masks.shape == (2, 8, 8)
    union = union_detection_mask((with_mask, polygon_only), 8, 8)
    assert union.shape == (8, 8)
    assert torch.all(union >= masks[0])
    assert individual_detection_masks((), 8, 8).shape == (0, 8, 8)
    assert union_detection_mask((), 8, 8).sum() == 0
    assert bbox_from_mask(explicit) == (2, 3, 5, 6)
    assert bbox_from_mask(torch.zeros((2, 2))) is None


def test_deterministic_color_and_box_expansion():
    assert deterministic_color("track-1") == deterministic_color("track-1")
    assert deterministic_color("track-1") != deterministic_color("track-2")
    assert all(0 <= channel <= 255 for channel in deterministic_color("object"))
    assert expand_box((4, 4, 8, 6), 12, 12, padding=1) == (3, 3, 9, 7)
    squared = expand_box((4, 4, 8, 6), 12, 12, square=True)
    assert squared == (4, 3, 8, 7)
    assert translate_box((1, 2, 3, 4), 2, 1) == (3, 3, 5, 5)


def test_label_aware_stable_association_with_motion():
    previous = (
        Detection(
            bbox_xyxy=(0, 0, 10, 10),
            label="cat",
            track_id=4,
        ),
        Detection(
            bbox_xyxy=(20, 0, 30, 10),
            label="dog",
            track_id=9,
        ),
    )
    current = (
        Detection(bbox_xyxy=(5, 0, 15, 10), label="cat"),
        Detection(bbox_xyxy=(20, 0, 30, 10), label="bird"),
        Detection(bbox_xyxy=(40, 0, 50, 10), label="dog"),
    )
    without_motion = associate_detections(
        previous,
        current,
        minimum_iou=0.3,
    )
    assert without_motion.matches == ((0, 0, pytest.approx(1 / 3)),)
    assert without_motion.unmatched_previous == (1,)
    assert without_motion.unmatched_current == (1, 2)

    with_motion = associate_detections(
        previous,
        current,
        minimum_iou=0.9,
        motion_by_track={4: (5, 0), 9: (20, 0)},
    )
    assert with_motion.matches == (
        (0, 0, 1.0),
        (1, 2, 1.0),
    )
    assert with_motion.unmatched_previous == ()
    assert with_motion.unmatched_current == (1,)

    label_agnostic = associate_detections(
        previous,
        current,
        minimum_iou=0.9,
        label_aware=False,
    )
    assert label_agnostic.matches == ((1, 1, 1.0),)
