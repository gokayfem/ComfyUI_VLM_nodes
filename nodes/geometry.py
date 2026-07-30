"""Dependency-light geometry, mask, color, and association primitives."""

from __future__ import annotations

import colorsys
import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw

from .vision_types import BoxXYXY, Detection, PointXY, Polygon


def _dimensions(width: int, height: int) -> tuple[int, int]:
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer.")
    if not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer.")
    return width, height


def _ordered_box(box: Iterable[float]) -> BoxXYXY:
    values = tuple(float(value) for value in box)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("A box must contain four finite xyxy values.")
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise ValueError("A box must satisfy x2 >= x1 and y2 >= y1.")
    return x1, y1, x2, y2


def clip_box(box: Iterable[float], width: int, height: int) -> BoxXYXY:
    """Clamp a pixel xyxy box to an image, preserving exclusive x2/y2."""

    width, height = _dimensions(width, height)
    x1, y1, x2, y2 = _ordered_box(box)
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def clip_polygon(
    polygon: Iterable[Iterable[float]],
    width: int,
    height: int,
) -> Polygon:
    width, height = _dimensions(width, height)
    points = []
    for point in polygon:
        values = tuple(float(value) for value in point)
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("Polygon points must contain two finite values.")
        points.append(
            (
                min(max(values[0], 0.0), float(width)),
                min(max(values[1], 0.0), float(height)),
            )
        )
    if len(points) < 3:
        raise ValueError("A polygon requires at least three points.")
    return tuple(points)


def normalize_box(
    box: Iterable[float],
    width: int,
    height: int,
) -> BoxXYXY:
    width, height = _dimensions(width, height)
    x1, y1, x2, y2 = clip_box(box, width, height)
    return x1 / width, y1 / height, x2 / width, y2 / height


def denormalize_box(
    box: Iterable[float],
    width: int,
    height: int,
) -> BoxXYXY:
    width, height = _dimensions(width, height)
    x1, y1, x2, y2 = _ordered_box(box)
    if any(value < 0.0 or value > 1.0 for value in (x1, y1, x2, y2)):
        raise ValueError("Normalized box coordinates must be between 0 and 1.")
    return x1 * width, y1 * height, x2 * width, y2 * height


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = _ordered_box(box)
    return (x2 - x1) * (y2 - y1)


def box_center(box: Iterable[float]) -> PointXY:
    x1, y1, x2, y2 = _ordered_box(box)
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def polygon_area(polygon: Iterable[Iterable[float]]) -> float:
    points = [tuple(float(value) for value in point) for point in polygon]
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise ValueError("A polygon requires at least three xy points.")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("Polygon coordinates must be finite.")
    twice_area = sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )
    return abs(twice_area) * 0.5


def bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = _ordered_box(first)
    bx1, by1, bx2, by2 = _ordered_box(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def mask_iou(
    first: torch.Tensor | np.ndarray,
    second: torch.Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    first_tensor = torch.as_tensor(first)
    second_tensor = torch.as_tensor(second)
    if first_tensor.ndim != 2 or second_tensor.ndim != 2:
        raise ValueError("Masks must have shape [height, width].")
    if first_tensor.shape != second_tensor.shape:
        raise ValueError("Masks must have the same shape.")
    first_bool = first_tensor > float(threshold)
    second_bool = second_tensor > float(threshold)
    intersection = torch.logical_and(first_bool, second_bool).sum().item()
    union = torch.logical_or(first_bool, second_bool).sum().item()
    return float(intersection / union) if union else 0.0


def deterministic_color(value: object) -> tuple[int, int, int]:
    """Return a readable RGB color that is stable across Python processes."""

    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.62 + digest[2] / 255.0 * 0.22
    brightness = 0.78 + digest[3] / 255.0 * 0.17
    return tuple(
        round(channel * 255)
        for channel in colorsys.hsv_to_rgb(hue, saturation, brightness)
    )


def box_to_mask(
    box: Iterable[float],
    width: int,
    height: int,
) -> torch.Tensor:
    width, height = _dimensions(width, height)
    x1, y1, x2, y2 = clip_box(box, width, height)
    left = max(0, min(width, math.floor(x1)))
    top = max(0, min(height, math.floor(y1)))
    right = max(left, min(width, math.ceil(x2)))
    bottom = max(top, min(height, math.ceil(y2)))
    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[top:bottom, left:right] = 1.0
    return mask


def polygon_to_mask(
    polygon: Iterable[Iterable[float]],
    width: int,
    height: int,
) -> torch.Tensor:
    width, height = _dimensions(width, height)
    points = clip_polygon(polygon, width, height)
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy())


def quad_to_mask(
    quad: Iterable[Iterable[float]],
    width: int,
    height: int,
) -> torch.Tensor:
    points = tuple(tuple(point) for point in quad)
    if len(points) != 4:
        raise ValueError("A quad must contain exactly four points.")
    return polygon_to_mask(points, width, height)


def detection_to_mask(
    detection: Detection,
    width: int,
    height: int,
) -> torch.Tensor:
    """Rasterize the most precise geometry available on a detection."""

    width, height = _dimensions(width, height)
    if not isinstance(detection, Detection):
        raise TypeError("detection must be a Detection.")
    if detection.mask is not None:
        if tuple(detection.mask.shape) != (height, width):
            raise ValueError("Detection mask shape does not match the image.")
        return detection.mask.detach().to(dtype=torch.float32).clamp(0, 1).clone()
    if detection.polygon is not None:
        return polygon_to_mask(detection.polygon, width, height)
    if detection.quad is not None:
        return quad_to_mask(detection.quad, width, height)
    return box_to_mask(detection.bbox_xyxy, width, height)


def individual_detection_masks(
    detections: Iterable[Detection],
    width: int,
    height: int,
) -> torch.Tensor:
    width, height = _dimensions(width, height)
    masks = [detection_to_mask(detection, width, height) for detection in detections]
    if not masks:
        return torch.zeros((0, height, width), dtype=torch.float32)
    return torch.stack(masks).to(dtype=torch.float32)


def union_detection_mask(
    detections: Iterable[Detection],
    width: int,
    height: int,
) -> torch.Tensor:
    masks = individual_detection_masks(detections, width, height)
    if masks.shape[0] == 0:
        return torch.zeros((height, width), dtype=torch.float32)
    return masks.amax(dim=0).clamp(0, 1)


def bbox_from_mask(
    mask: torch.Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
) -> BoxXYXY | None:
    value = torch.as_tensor(mask)
    if value.ndim != 2:
        raise ValueError("mask must have shape [height, width].")
    locations = torch.nonzero(value > float(threshold), as_tuple=False)
    if locations.numel() == 0:
        return None
    y1, x1 = locations.amin(dim=0).tolist()
    y2, x2 = locations.amax(dim=0).tolist()
    return float(x1), float(y1), float(x2 + 1), float(y2 + 1)


def translate_box(
    box: Iterable[float],
    dx: float,
    dy: float,
) -> BoxXYXY:
    x1, y1, x2, y2 = _ordered_box(box)
    dx = float(dx)
    dy = float(dy)
    if not math.isfinite(dx) or not math.isfinite(dy):
        raise ValueError("Box motion must be finite.")
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


def expand_box(
    box: Iterable[float],
    width: int,
    height: int,
    *,
    padding: float = 0.0,
    square: bool = False,
) -> BoxXYXY:
    """Pad and optionally square a box around its center, then clip it."""

    width, height = _dimensions(width, height)
    if not math.isfinite(float(padding)) or padding < 0:
        raise ValueError("padding must be finite and non-negative.")
    x1, y1, x2, y2 = _ordered_box(box)
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding
    if square:
        center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        half = max(x2 - x1, y2 - y1) * 0.5
        x1, y1, x2, y2 = (
            center_x - half,
            center_y - half,
            center_x + half,
            center_y + half,
        )
        side = x2 - x1
        if side <= width:
            if x1 < 0:
                x2 -= x1
                x1 = 0.0
            elif x2 > width:
                x1 -= x2 - width
                x2 = float(width)
        if side <= height:
            if y1 < 0:
                y2 -= y1
                y1 = 0.0
            elif y2 > height:
                y1 -= y2 - height
                y2 = float(height)
    return clip_box((x1, y1, x2, y2), width, height)


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """Stable one-to-one detection assignment by descending overlap."""

    matches: tuple[tuple[int, int, float], ...]
    unmatched_previous: tuple[int, ...]
    unmatched_current: tuple[int, ...]


def associate_detections(
    previous: Iterable[Detection],
    current: Iterable[Detection],
    *,
    minimum_iou: float = 0.3,
    label_aware: bool = True,
    motion_by_track: Mapping[int, tuple[float, float]] | None = None,
) -> AssociationResult:
    """Associate detections without SciPy or backend-specific operators.

    Candidates are greedily selected by descending IoU with deterministic
    index tie-breaks. Optional per-track motion offsets predict the previous
    box before overlap is measured.
    """

    previous_items = tuple(previous)
    current_items = tuple(current)
    if not 0.0 <= float(minimum_iou) <= 1.0:
        raise ValueError("minimum_iou must be between 0 and 1.")
    if any(not isinstance(item, Detection) for item in previous_items):
        raise TypeError("previous must contain Detection values.")
    if any(not isinstance(item, Detection) for item in current_items):
        raise TypeError("current must contain Detection values.")

    candidates = []
    for previous_index, old in enumerate(previous_items):
        old_box = old.bbox_xyxy
        if old.track_id is not None and motion_by_track:
            motion = motion_by_track.get(old.track_id)
            if motion is not None:
                old_box = translate_box(old_box, motion[0], motion[1])
        for current_index, new in enumerate(current_items):
            if (
                label_aware
                and old.label is not None
                and new.label is not None
                and " ".join(old.label.casefold().split())
                != " ".join(new.label.casefold().split())
            ):
                continue
            overlap = bbox_iou(old_box, new.bbox_xyxy)
            if overlap >= float(minimum_iou):
                candidates.append((-overlap, previous_index, current_index, overlap))

    matched_previous: set[int] = set()
    matched_current: set[int] = set()
    matches = []
    for _negative, previous_index, current_index, overlap in sorted(candidates):
        if previous_index in matched_previous or current_index in matched_current:
            continue
        matched_previous.add(previous_index)
        matched_current.add(current_index)
        matches.append((previous_index, current_index, overlap))

    return AssociationResult(
        matches=tuple(matches),
        unmatched_previous=tuple(
            index
            for index in range(len(previous_items))
            if index not in matched_previous
        ),
        unmatched_current=tuple(
            index for index in range(len(current_items)) if index not in matched_current
        ),
    )


__all__ = [
    "AssociationResult",
    "associate_detections",
    "bbox_from_mask",
    "bbox_iou",
    "box_area",
    "box_center",
    "box_to_mask",
    "clip_box",
    "clip_polygon",
    "denormalize_box",
    "detection_to_mask",
    "deterministic_color",
    "expand_box",
    "individual_detection_masks",
    "mask_iou",
    "normalize_box",
    "polygon_area",
    "polygon_to_mask",
    "quad_to_mask",
    "translate_box",
    "union_detection_mask",
]
