"""Composable utility nodes for canonical VLM spatial payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .geometry import (
    detection_to_mask,
    deterministic_color,
    expand_box,
    individual_detection_masks,
    union_detection_mask,
)
from .vision_types import (
    VLM_DETECTIONS,
    VLM_POINTS,
    Detection,
    DetectionSequence,
    FrameDetections,
    PointSequence,
    VisionPoint,
)

BOUNDING_BOXES = "BOUNDING_BOXES"


def _copy_sequence(
    sequence: DetectionSequence,
    frames: Iterable[FrameDetections],
) -> DetectionSequence:
    return replace(sequence, frames=tuple(frames))


def filter_detection_sequence(
    sequence: DetectionSequence,
    *,
    label: str = "",
    label_mode: str = "contains",
    minimum_score: float = 0.0,
    minimum_area: float = 0.0,
    maximum_area: float = 0.0,
    track_id: int = -1,
    frame_index: int = -1,
) -> DetectionSequence:
    if not isinstance(sequence, DetectionSequence):
        raise TypeError("detections must be a DetectionSequence.")
    if label_mode not in {"contains", "exact"}:
        raise ValueError("label_mode must be 'contains' or 'exact'.")
    if not 0.0 <= float(minimum_score) <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1.")
    if minimum_area < 0 or maximum_area < 0:
        raise ValueError("Area filters must be non-negative.")
    if maximum_area and maximum_area < minimum_area:
        raise ValueError("maximum_area cannot be smaller than minimum_area.")
    if track_id < -1 or frame_index < -1:
        raise ValueError("track_id and frame_index must be -1 or non-negative.")

    query = label.strip().casefold()

    def keep(detection: Detection) -> bool:
        candidate = (detection.label or "").casefold()
        if query:
            if label_mode == "exact" and candidate != query:
                return False
            if label_mode == "contains" and query not in candidate:
                return False
        if minimum_score > 0 and (
            detection.score is None or detection.score < minimum_score
        ):
            return False
        if detection.area < minimum_area:
            return False
        if maximum_area and detection.area > maximum_area:
            return False
        if track_id >= 0 and detection.track_id != track_id:
            return False
        return frame_index < 0 or detection.frame_index == frame_index

    frames = []
    for frame in sequence.frames:
        if frame_index >= 0 and frame.frame_index != frame_index:
            continue
        frames.append(
            replace(
                frame,
                detections=tuple(
                    detection for detection in frame.detections if keep(detection)
                ),
            )
        )
    return _copy_sequence(sequence, frames)


def select_detection_sequence(
    sequence: DetectionSequence,
    index: int,
) -> DetectionSequence:
    if not isinstance(sequence, DetectionSequence):
        raise TypeError("detections must be a DetectionSequence.")
    if not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer.")
    all_detections = sequence.all_detections()
    selected = all_detections[index] if index < len(all_detections) else None
    frames = tuple(
        replace(
            frame,
            detections=(
                (selected,)
                if selected is not None and selected.frame_index == frame.frame_index
                else ()
            ),
        )
        for frame in sequence.frames
    )
    return _copy_sequence(sequence, frames)


def bounding_boxes_payload(
    sequence: DetectionSequence,
) -> list[dict[str, Any]]:
    """Return ComfyUI's BOUNDING_BOXES list-of-dicts contract.

    Core bounding boxes use integer ``x/y/width/height`` and an arbitrary
    metadata dictionary. The original sub-pixel xyxy coordinates remain in
    metadata so the conversion is not lossy.
    """

    if not isinstance(sequence, DetectionSequence):
        raise TypeError("detections must be a DetectionSequence.")
    result = []
    for detection in sequence.all_detections():
        x1, y1, x2, y2 = detection.bbox_xyxy
        left, top = math.floor(x1), math.floor(y1)
        right, bottom = math.ceil(x2), math.ceil(y2)
        metadata: dict[str, Any] = {
            "bbox_xyxy": list(detection.bbox_xyxy),
            "frame_index": detection.frame_index,
            "timestamp": detection.timestamp,
        }
        for key, value in (
            ("label", detection.label),
            ("text", detection.text),
            ("score", detection.score),
            ("track_id", detection.track_id),
            ("source", detection.source),
        ):
            if value is not None:
                metadata[key] = value
        if detection.metadata:
            metadata["vlm_metadata"] = detection.metadata.to_dict()
        result.append(
            {
                "x": left,
                "y": top,
                "width": max(0, right - left),
                "height": max(0, bottom - top),
                "metadata": metadata,
            }
        )
    return result


def detection_centers(sequence: DetectionSequence) -> PointSequence:
    if not isinstance(sequence, DetectionSequence):
        raise TypeError("detections must be a DetectionSequence.")
    return PointSequence(
        width=sequence.width,
        height=sequence.height,
        frame_count=sequence.frame_count,
        fps=sequence.fps,
        source=sequence.source,
        metadata=sequence.metadata,
        points=tuple(
            VisionPoint(
                x=detection.center[0],
                y=detection.center[1],
                label=detection.label,
                text=detection.text,
                score=detection.score,
                frame_index=detection.frame_index,
                timestamp=detection.timestamp,
                track_id=detection.track_id,
                source=detection.source,
                metadata=detection.metadata,
            )
            for detection in sequence.all_detections()
        ),
    )


def sequence_masks(
    sequence: DetectionSequence,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Return per-frame union masks and flattened individual masks."""

    if not isinstance(sequence, DetectionSequence):
        raise TypeError("detections must be a DetectionSequence.")
    frame_lookup = {frame.frame_index: frame for frame in sequence.frames}
    union_masks = []
    individual_masks = []
    mapping = []
    for frame_index in range(sequence.frame_count):
        frame = frame_lookup.get(frame_index)
        detections = frame.detections if frame is not None else ()
        union_masks.append(
            union_detection_mask(
                detections,
                sequence.width,
                sequence.height,
            )
        )
        frame_masks = individual_detection_masks(
            detections,
            sequence.width,
            sequence.height,
        )
        for detection_index, (detection, mask) in enumerate(
            zip(detections, frame_masks)
        ):
            individual_masks.append(mask)
            mapping.append(
                {
                    "mask_index": len(individual_masks) - 1,
                    "frame_index": frame_index,
                    "detection_index": detection_index,
                    "label": detection.label,
                    "track_id": detection.track_id,
                }
            )
    unions = (
        torch.stack(union_masks)
        if union_masks
        else torch.zeros(
            (0, sequence.height, sequence.width),
            dtype=torch.float32,
        )
    )
    individuals = (
        torch.stack(individual_masks)
        if individual_masks
        else torch.zeros(
            (0, sequence.height, sequence.width),
            dtype=torch.float32,
        )
    )
    return unions, individuals, mapping


def normalize_masks(mask: torch.Tensor) -> torch.Tensor:
    """Normalize common Comfy mask layouts to finite float ``[N, H, W]``."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(
            "mask must have shape [height, width], [batch, height, width], "
            "[batch, 1, height, width], or [batch, height, width, 1]."
        )
    value = mask.to(dtype=torch.float32)
    return torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)


def masks_to_images(mask: torch.Tensor) -> torch.Tensor:
    """Return ready-to-preview black-and-white RGB images for a mask batch."""

    value = normalize_masks(mask)
    return value.unsqueeze(-1).expand(-1, -1, -1, 3).clone()


def instance_map_images(sequence: DetectionSequence) -> torch.Tensor:
    """Render stable object/track colors over black for every source frame."""

    if not isinstance(sequence, DetectionSequence):
        raise TypeError("detections must be a DetectionSequence.")
    output = torch.zeros(
        (sequence.frame_count, sequence.height, sequence.width, 3),
        dtype=torch.float32,
    )
    frame_lookup = {frame.frame_index: frame for frame in sequence.frames}
    for frame_index in range(sequence.frame_count):
        frame = frame_lookup.get(frame_index)
        if frame is None:
            continue
        for detection_index, detection in enumerate(frame.detections):
            identity = (
                detection.track_id
                if detection.track_id is not None
                else detection.label or detection_index
            )
            color = torch.tensor(
                deterministic_color(identity),
                dtype=torch.float32,
            ).div_(255.0)
            mask = detection_to_mask(
                detection,
                sequence.width,
                sequence.height,
            )
            output[frame_index][mask > 0.5] = color
    return output


def _morph_masks(mask: torch.Tensor, grow_shrink: int) -> torch.Tensor:
    radius = abs(int(grow_shrink))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    value = mask.unsqueeze(1)
    if grow_shrink > 0:
        value = F.max_pool2d(value, kernel, stride=1, padding=radius)
    else:
        value = -F.max_pool2d(
            -F.pad(
                value,
                (radius, radius, radius, radius),
                mode="constant",
                value=0.0,
            ),
            kernel,
            stride=1,
        )
    return value[:, 0].clamp(0, 1)


def _blur_masks(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask
    sigma = max(radius / 3.0, 0.5)
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=mask.device,
        dtype=mask.dtype,
    )
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    value = mask.unsqueeze(1)
    value = F.pad(value, (radius, radius, 0, 0), mode="replicate")
    value = F.conv2d(value, kernel.reshape(1, 1, 1, -1))
    value = F.pad(value, (0, 0, radius, radius), mode="replicate")
    value = F.conv2d(value, kernel.reshape(1, 1, -1, 1))
    return value[:, 0].clamp(0, 1)


def process_masks(
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    grow_shrink: int = 0,
    feather_radius: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a soft matte, strict binary mask, and inverse matte."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if not isinstance(grow_shrink, int):
        raise TypeError("grow_shrink must be an integer.")
    if not isinstance(feather_radius, int) or feather_radius < 0:
        raise ValueError("feather_radius must be a non-negative integer.")
    binary = (normalize_masks(mask) >= float(threshold)).to(torch.float32)
    binary = _morph_masks(binary, grow_shrink)
    processed = _blur_masks(binary, feather_radius)
    return processed, binary, 1.0 - processed


def _rgb_images(value: torch.Tensor, name: str) -> torch.Tensor:
    images = _images(value)
    if images.shape[-1] == 1:
        images = images.expand(-1, -1, -1, 3)
    elif images.shape[-1] == 4:
        images = images[..., :3]
    return torch.nan_to_num(
        images.to(dtype=torch.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0, 1)


def _parse_color(value: str) -> tuple[float, float, float]:
    text = str(value).strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3 and all(
        character in "0123456789abcdefABCDEF" for character in text
    ):
        text = "".join(character * 2 for character in text)
    if len(text) == 6 and all(
        character in "0123456789abcdefABCDEF" for character in text
    ):
        return tuple(int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
    try:
        channels = tuple(float(channel.strip()) for channel in text.split(","))
    except ValueError as error:
        raise ValueError(
            "background_color must be #RRGGBB, #RGB, or three 0-255 channels."
        ) from error
    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError(
            "background_color must be #RRGGBB, #RGB, or three 0-255 channels."
        )
    return tuple(channel / 255.0 for channel in channels)


def _repeat_batch(value: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if value.shape[0] == batch:
        return value
    if value.shape[0] == 1:
        return value.expand(batch, *value.shape[1:])
    raise ValueError(f"{name} batch must contain one item or {batch} items.")


def composite_with_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    background: torch.Tensor | None = None,
    background_color: str = "#000000",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split and composite an image/video batch with predictable broadcasting."""

    images = _rgb_images(image, "image")
    masks = normalize_masks(mask)
    if tuple(masks.shape[-2:]) != tuple(images.shape[1:3]):
        raise ValueError("Mask dimensions must match the image dimensions.")
    background_images = (
        _rgb_images(background, "background") if background is not None else None
    )
    if background_images is not None and tuple(background_images.shape[1:3]) != tuple(
        images.shape[1:3]
    ):
        raise ValueError("Background dimensions must match the image dimensions.")

    batch = max(
        images.shape[0],
        masks.shape[0],
        background_images.shape[0] if background_images is not None else 1,
    )
    images = _repeat_batch(images, batch, "image")
    masks = _repeat_batch(masks, batch, "mask")
    if background_images is None:
        color = images.new_tensor(_parse_color(background_color))
        background_images = color.reshape(1, 1, 1, 3).expand(
            batch,
            images.shape[1],
            images.shape[2],
            3,
        )
    else:
        background_images = _repeat_batch(background_images, batch, "background")

    alpha = masks.unsqueeze(-1)
    foreground = images * alpha
    background_only = images * (1.0 - alpha)
    composite = foreground + background_images * (1.0 - alpha)
    return composite, foreground, background_only, masks_to_images(masks)


def _images(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] not in (1, 3, 4):
        raise ValueError("image must have shape [batch, height, width, channels].")
    return value


def _frame_image_pairs(
    images: torch.Tensor,
    sequence: DetectionSequence,
) -> tuple[tuple[int, FrameDetections | None], ...]:
    batch = images.shape[0]
    if images.shape[2] != sequence.width or images.shape[1] != sequence.height:
        raise ValueError("Image dimensions do not match the detection sequence.")
    if not sequence.frames:
        return tuple((index, None) for index in range(batch))
    if batch == sequence.frame_count:
        lookup = {frame.frame_index: frame for frame in sequence.frames}
        return tuple((index, lookup.get(index)) for index in range(batch))
    if batch == len(sequence.frames):
        return tuple((index, frame) for index, frame in enumerate(sequence.frames))
    raise ValueError(
        "Image batch must match frame_count or the number of stored frames."
    )


def _pil_image(image: torch.Tensor) -> Image.Image:
    value = image.detach().to(device="cpu", dtype=torch.float32)
    value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
    if value.shape[-1] == 1:
        value = value.repeat(1, 1, 3)
    elif value.shape[-1] == 4:
        value = value[..., :3]
    array = value.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(array, "RGB")


def _tensor_image(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy())


def _display_label(detection: Detection, fallback_index: int) -> str:
    identity = (
        f"#{detection.track_id}"
        if detection.track_id is not None
        else str(fallback_index)
    )
    label = detection.label or detection.text or "object"
    score = f" {detection.score:.2f}" if detection.score is not None else ""
    return f"{identity} {label}{score}"


def render_detections(
    image: torch.Tensor,
    sequence: DetectionSequence,
    *,
    draw_masks: bool = True,
    draw_labels: bool = True,
    mask_opacity: float = 0.35,
    line_width: int = 3,
) -> torch.Tensor:
    images = _images(image)
    if not isinstance(sequence, DetectionSequence):
        raise TypeError("detections must be a DetectionSequence.")
    if not 0.0 <= float(mask_opacity) <= 1.0:
        raise ValueError("mask_opacity must be between 0 and 1.")
    if not isinstance(line_width, int) or line_width < 1:
        raise ValueError("line_width must be a positive integer.")
    if not sequence.all_detections():
        return images.clone()

    rendered = []
    for image_index, frame in _frame_image_pairs(images, sequence):
        visual = _pil_image(images[image_index])
        if frame is None or not frame.detections:
            rendered.append(_tensor_image(visual))
            continue

        if draw_masks:
            pixels = np.asarray(visual, dtype=np.float32).copy()
            for detection_index, detection in enumerate(frame.detections):
                identity = (
                    detection.track_id
                    if detection.track_id is not None
                    else detection.label or detection_index
                )
                color = np.asarray(
                    deterministic_color(identity),
                    dtype=np.float32,
                )
                mask = (
                    detection_to_mask(
                        detection,
                        sequence.width,
                        sequence.height,
                    )
                    .cpu()
                    .numpy()
                )[..., None]
                alpha = mask * float(mask_opacity)
                pixels = pixels * (1.0 - alpha) + color * alpha
            visual = Image.fromarray(
                np.clip(pixels, 0, 255).astype(np.uint8),
                "RGB",
            )

        draw = ImageDraw.Draw(visual)
        for detection_index, detection in enumerate(frame.detections):
            identity = (
                detection.track_id
                if detection.track_id is not None
                else detection.label or detection_index
            )
            color = deterministic_color(identity)
            x1, y1, x2, y2 = detection.bbox_xyxy
            outline = (
                round(x1),
                round(y1),
                max(round(x1), round(x2) - 1),
                max(round(y1), round(y2) - 1),
            )
            draw.rectangle(outline, outline=color, width=line_width)
            if detection.polygon is not None:
                draw.line(
                    [*detection.polygon, detection.polygon[0]],
                    fill=color,
                    width=line_width,
                )
            if detection.quad is not None:
                draw.line(
                    [*detection.quad, detection.quad[0]],
                    fill=color,
                    width=line_width,
                )
            if draw_labels:
                label = _display_label(detection, detection_index)
                text_box = draw.textbbox((0, 0), label)
                text_width = text_box[2] - text_box[0]
                text_height = text_box[3] - text_box[1]
                text_x = min(max(0, round(x1)), max(0, visual.width - text_width - 4))
                text_y = round(y1) - text_height - 5
                if text_y < 0:
                    text_y = max(
                        0,
                        min(
                            visual.height - text_height - 4,
                            round(y1) + 2,
                        ),
                    )
                draw.rectangle(
                    (
                        text_x,
                        text_y,
                        text_x + text_width + 4,
                        text_y + text_height + 4,
                    ),
                    fill=color,
                )
                draw.text((text_x + 2, text_y + 2), label, fill="black")
        rendered.append(_tensor_image(visual))
    return torch.stack(rendered) if rendered else images.clone()


def crop_detections(
    image: torch.Tensor,
    sequence: DetectionSequence,
    *,
    padding: float = 0.0,
    square: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Crop every detection and pad crops to one non-distorted IMAGE batch."""

    images = _images(image)
    if not isinstance(sequence, DetectionSequence):
        raise TypeError("detections must be a DetectionSequence.")
    if not math.isfinite(float(padding)) or padding < 0:
        raise ValueError("padding must be finite and non-negative.")

    crops = []
    records = []
    for image_index, frame in _frame_image_pairs(images, sequence):
        if frame is None:
            continue
        for detection_index, detection in enumerate(frame.detections):
            crop_box = expand_box(
                detection.bbox_xyxy,
                sequence.width,
                sequence.height,
                padding=float(padding),
                square=bool(square),
            )
            x1, y1, x2, y2 = crop_box
            left, top = math.floor(x1), math.floor(y1)
            right, bottom = math.ceil(x2), math.ceil(y2)
            crop = images[image_index, top:bottom, left:right].clone()
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            crops.append(crop)
            records.append(
                {
                    "crop_index": len(crops) - 1,
                    "frame_index": frame.frame_index,
                    "detection_index": detection_index,
                    "bbox_xyxy": list(detection.bbox_xyxy),
                    "crop_bbox_xyxy": [left, top, right, bottom],
                    "valid_width": crop.shape[1],
                    "valid_height": crop.shape[0],
                    "label": detection.label,
                    "track_id": detection.track_id,
                }
            )

    channels = images.shape[-1]
    if not crops:
        return images.new_zeros((0, 1, 1, channels)), []
    maximum_height = max(crop.shape[0] for crop in crops)
    maximum_width = max(crop.shape[1] for crop in crops)
    output = images.new_zeros((len(crops), maximum_height, maximum_width, channels))
    for index, crop in enumerate(crops):
        output[index, : crop.shape[0], : crop.shape[1]] = crop
        records[index]["batch_width"] = maximum_width
        records[index]["batch_height"] = maximum_height
    return output, records


class VLMDetectionsFromJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            '{"schema":"comfyui-vlm/detections","version":1,'
                            '"media":{"width":1,"height":1,"frame_count":0},'
                            '"frames":[]}'
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = (VLM_DETECTIONS,)
    RETURN_NAMES = ("detections",)
    FUNCTION = "parse"
    CATEGORY = "VLM Nodes/Vision/Utilities"

    def parse(self, json_text):
        return (DetectionSequence.from_json(json_text),)


class VLMDetectionsToJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detections": (VLM_DETECTIONS,),
                "pretty": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "serialize"
    CATEGORY = "VLM Nodes/Vision/Utilities"

    def serialize(self, detections, pretty=True):
        if not isinstance(detections, DetectionSequence):
            raise TypeError("detections must be a DetectionSequence.")
        return (detections.to_json(indent=2 if pretty else None),)


class VLMFilterDetections:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detections": (VLM_DETECTIONS,),
                "label": ("STRING", {"default": ""}),
                "label_mode": (["contains", "exact"],),
                "minimum_score": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "minimum_area": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "step": 1.0},
                ),
                "maximum_area": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 1.0,
                        "tooltip": "Zero disables the maximum-area filter.",
                    },
                ),
                "track_id": (
                    "INT",
                    {"default": -1, "min": -1, "step": 1},
                ),
                "frame_index": (
                    "INT",
                    {"default": -1, "min": -1, "step": 1},
                ),
            }
        }

    RETURN_TYPES = (VLM_DETECTIONS,)
    RETURN_NAMES = ("detections",)
    FUNCTION = "filter"
    CATEGORY = "VLM Nodes/Vision/Utilities"

    def filter(
        self,
        detections,
        label="",
        label_mode="contains",
        minimum_score=0.0,
        minimum_area=0.0,
        maximum_area=0.0,
        track_id=-1,
        frame_index=-1,
    ):
        return (
            filter_detection_sequence(
                detections,
                label=label,
                label_mode=label_mode,
                minimum_score=minimum_score,
                minimum_area=minimum_area,
                maximum_area=maximum_area,
                track_id=track_id,
                frame_index=frame_index,
            ),
        )


class VLMSelectDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detections": (VLM_DETECTIONS,),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = (VLM_DETECTIONS,)
    RETURN_NAMES = ("selection",)
    FUNCTION = "select"
    CATEGORY = "VLM Nodes/Vision/Utilities"

    def select(self, detections, index):
        return (select_detection_sequence(detections, index),)


class VLMDetectionsToBoundingBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"detections": (VLM_DETECTIONS,)}}

    RETURN_TYPES = (BOUNDING_BOXES, "STRING")
    RETURN_NAMES = ("bounding_boxes", "metadata_json")
    FUNCTION = "convert"
    CATEGORY = "VLM Nodes/Vision/Utilities"

    def convert(self, detections):
        payload = bounding_boxes_payload(detections)
        return (
            payload,
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
        )


class VLMDetectionsToPoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"detections": (VLM_DETECTIONS,)}}

    RETURN_TYPES = (VLM_POINTS, "STRING")
    RETURN_NAMES = ("points", "points_json")
    FUNCTION = "convert"
    CATEGORY = "VLM Nodes/Vision/Utilities"

    def convert(self, detections):
        points = detection_centers(detections)
        return points, points.to_json()


class VLMDetectionsToMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"detections": (VLM_DETECTIONS,)}}

    RETURN_TYPES = (
        "MASK",
        "MASK",
        "STRING",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "union_masks",
        "individual_masks",
        "mask_map_json",
        "inverse_union_masks",
        "union_mask_images",
        "individual_mask_images",
        "instance_maps",
    )
    FUNCTION = "convert"
    CATEGORY = "VLM Nodes/Vision/Utilities"
    DESCRIPTION = (
        "Convert detections to combined and per-object black/white masks, "
        "inverse masks, previewable mask images, and stable-color instance maps."
    )

    def convert(self, detections):
        unions, individuals, mapping = sequence_masks(detections)
        return (
            unions,
            individuals,
            json.dumps(mapping, ensure_ascii=False, allow_nan=False),
            1.0 - unions,
            masks_to_images(unions),
            masks_to_images(individuals),
            instance_map_images(detections),
        )


class VLMMaskProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "grow_shrink": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1024,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "Positive grows; negative shrinks the mask.",
                    },
                ),
                "feather_radius": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1024, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = (
        "processed_mask",
        "binary_mask",
        "inverse_mask",
        "mask_image",
    )
    FUNCTION = "process"
    CATEGORY = "VLM Nodes/Vision/Mask Tools"
    DESCRIPTION = (
        "Threshold, grow/shrink, and feather any VLM or Comfy mask. Returns "
        "the soft matte, strict binary mask, inverse matte, and black/white image."
    )

    def process(
        self,
        mask,
        threshold=0.5,
        grow_shrink=0,
        feather_radius=0,
    ):
        processed, binary, inverse = process_masks(
            mask,
            threshold=threshold,
            grow_shrink=grow_shrink,
            feather_radius=feather_radius,
        )
        return processed, binary, inverse, masks_to_images(processed)


class VLMMaskComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "background_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "#RRGGBB, #RGB, or R,G,B in the 0-255 range.",
                    },
                ),
            },
            "optional": {
                "background": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "composite",
        "foreground",
        "background_only",
        "mask_image",
    )
    FUNCTION = "composite"
    CATEGORY = "VLM Nodes/Vision/Mask Tools"
    DESCRIPTION = (
        "Apply a mask to still-image or video batches. Composite the selected "
        "foreground over a solid color or optional background image and also "
        "return isolated foreground, original background, and mask preview."
    )

    def composite(
        self,
        image,
        mask,
        background_color="#000000",
        background=None,
    ):
        return composite_with_mask(
            image,
            mask,
            background=background,
            background_color=background_color,
        )


class VLMRenderDetections:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detections": (VLM_DETECTIONS,),
                "draw_masks": ("BOOLEAN", {"default": True}),
                "draw_labels": ("BOOLEAN", {"default": True}),
                "mask_opacity": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "line_width": (
                    "INT",
                    {"default": 3, "min": 1, "max": 32, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "render"
    CATEGORY = "VLM Nodes/Vision/Utilities"

    def render(
        self,
        image,
        detections,
        draw_masks=True,
        draw_labels=True,
        mask_opacity=0.35,
        line_width=3,
    ):
        return (
            render_detections(
                image,
                detections,
                draw_masks=draw_masks,
                draw_labels=draw_labels,
                mask_opacity=mask_opacity,
                line_width=line_width,
            ),
        )


class VLMCropDetections:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detections": (VLM_DETECTIONS,),
                "padding": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "step": 1.0},
                ),
                "square": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("crops", "crop_metadata_json")
    FUNCTION = "crop"
    CATEGORY = "VLM Nodes/Vision/Utilities"

    def crop(self, image, detections, padding=0.0, square=False):
        crops, metadata = crop_detections(
            image,
            detections,
            padding=padding,
            square=square,
        )
        return (
            crops,
            json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "VLMDetectionsFromJSON": VLMDetectionsFromJSON,
    "VLMDetectionsToJSON": VLMDetectionsToJSON,
    "VLMFilterDetections": VLMFilterDetections,
    "VLMSelectDetection": VLMSelectDetection,
    "VLMDetectionsToBoundingBoxes": VLMDetectionsToBoundingBoxes,
    "VLMDetectionsToPoints": VLMDetectionsToPoints,
    "VLMDetectionsToMasks": VLMDetectionsToMasks,
    "VLMMaskProcessor": VLMMaskProcessor,
    "VLMMaskComposite": VLMMaskComposite,
    "VLMRenderDetections": VLMRenderDetections,
    "VLMCropDetections": VLMCropDetections,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMDetectionsFromJSON": "VLM Detections from JSON",
    "VLMDetectionsToJSON": "VLM Detections to JSON",
    "VLMFilterDetections": "Filter VLM Detections",
    "VLMSelectDetection": "Select VLM Detection",
    "VLMDetectionsToBoundingBoxes": "VLM Detections to Bounding Boxes",
    "VLMDetectionsToPoints": "VLM Detection Centers",
    "VLMDetectionsToMasks": "VLM Detections to Masks",
    "VLMMaskProcessor": "VLM Mask Processor",
    "VLMMaskComposite": "VLM Mask Composite",
    "VLMRenderDetections": "Render VLM Detections",
    "VLMCropDetections": "Crop VLM Detections",
}

__all__ = [
    "BOUNDING_BOXES",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "VLMCropDetections",
    "VLMDetectionsFromJSON",
    "VLMDetectionsToBoundingBoxes",
    "VLMDetectionsToJSON",
    "VLMDetectionsToMasks",
    "VLMDetectionsToPoints",
    "VLMFilterDetections",
    "VLMMaskComposite",
    "VLMMaskProcessor",
    "VLMRenderDetections",
    "VLMSelectDetection",
    "bounding_boxes_payload",
    "composite_with_mask",
    "crop_detections",
    "detection_centers",
    "filter_detection_sequence",
    "instance_map_images",
    "masks_to_images",
    "normalize_masks",
    "process_masks",
    "render_detections",
    "select_detection_sequence",
    "sequence_masks",
]
