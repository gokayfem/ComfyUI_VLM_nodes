"""Florence-2 multitask caption, OCR, detection and segmentation node."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real

import torch
from PIL import Image, ImageDraw

from .runtime import (
    CachedModelNode,
    ManagedTorchModel,
    batch_text,
    inference_context,
    model_device,
    move_inputs,
    pil_mask_to_tensor,
    pil_to_tensor,
    require_module,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)

MODELS = {
    "Florence-2 base FT (fast)": "florence-community/Florence-2-base-ft",
    "Florence-2 large FT (recommended)": ("florence-community/Florence-2-large-ft"),
}


@dataclass(frozen=True)
class FlorenceTaskSpec:
    """Declarative contract for one official Florence-2 task."""

    token: str
    input_kind: str
    output_kind: str


TASKS = {
    "Caption": FlorenceTaskSpec("<CAPTION>", "none", "text"),
    "Detailed caption": FlorenceTaskSpec("<DETAILED_CAPTION>", "none", "text"),
    "More detailed caption": FlorenceTaskSpec(
        "<MORE_DETAILED_CAPTION>", "none", "text"
    ),
    "OCR": FlorenceTaskSpec("<OCR>", "none", "text"),
    "OCR with regions": FlorenceTaskSpec("<OCR_WITH_REGION>", "none", "quad_boxes"),
    "Object detection": FlorenceTaskSpec("<OD>", "none", "boxes"),
    "Dense region caption": FlorenceTaskSpec("<DENSE_REGION_CAPTION>", "none", "boxes"),
    "Caption to phrase grounding": FlorenceTaskSpec(
        "<CAPTION_TO_PHRASE_GROUNDING>", "text", "boxes"
    ),
    "Referring expression segmentation": FlorenceTaskSpec(
        "<REFERRING_EXPRESSION_SEGMENTATION>", "text", "polygons"
    ),
    "Region to segmentation": FlorenceTaskSpec(
        "<REGION_TO_SEGMENTATION>", "region", "polygons"
    ),
    "Open vocabulary detection": FlorenceTaskSpec(
        "<OPEN_VOCABULARY_DETECTION>", "text", "mixed"
    ),
    "Region to category": FlorenceTaskSpec("<REGION_TO_CATEGORY>", "region", "text"),
    "Region to description": FlorenceTaskSpec(
        "<REGION_TO_DESCRIPTION>", "region", "text"
    ),
    "Region to OCR": FlorenceTaskSpec("<REGION_TO_OCR>", "region", "text"),
    "Region proposals": FlorenceTaskSpec("<REGION_PROPOSAL>", "none", "boxes"),
}


def _clean_decoded_text(value):
    """Remove generation wrappers without discarding Florence location tokens."""

    text = str(value)
    for token in ("<s>", "</s>", "<pad>"):
        text = text.replace(token, "")
    return text.strip()


def _select_region(region, image_index, batch_size):
    """Select one core BOUNDING_BOX for the current image.

    Core primitive boxes are dictionaries. Detection nodes may emit either a
    flat per-image list or a nested batch list, so both common shapes are
    accepted while ambiguous multi-region inputs fail explicitly.
    """

    if region is None or isinstance(region, dict):
        return region
    if not isinstance(region, (list, tuple)):
        raise TypeError("region must be a core BOUNDING_BOX dictionary.")
    if not region:
        return None

    if all(isinstance(item, dict) for item in region):
        if len(region) == 1:
            return region[0]
        if len(region) == batch_size:
            return region[image_index]
        raise ValueError("Region tasks require exactly one BOUNDING_BOX per image.")

    if len(region) != batch_size:
        raise ValueError("Batched BOUNDING_BOX input must contain one entry per image.")
    frame_regions = region[image_index]
    if isinstance(frame_regions, dict):
        return frame_regions
    if not isinstance(frame_regions, (list, tuple)) or len(frame_regions) != 1:
        raise ValueError(
            "Region tasks require exactly one BOUNDING_BOX per image; "
            "select a detection before connecting it."
        )
    if not isinstance(frame_regions[0], dict):
        raise TypeError("Each BOUNDING_BOX entry must be a dictionary.")
    return frame_regions[0]


def _encode_region(region, image_size):
    """Encode an absolute-pixel core BOUNDING_BOX as Florence location tokens."""

    if not isinstance(region, dict):
        raise TypeError("region must be a core BOUNDING_BOX dictionary.")

    try:
        x = float(region["x"])
        y = float(region["y"])
        box_width = float(region["width"])
        box_height = float(region["height"])
    except KeyError as exc:
        raise ValueError("region must contain x, y, width, and height.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("region coordinates must be numeric.") from exc

    values = (x, y, box_width, box_height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("region coordinates must be finite.")
    if box_width <= 0 or box_height <= 0:
        raise ValueError("region width and height must be greater than zero.")

    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be greater than zero.")

    x0 = max(0.0, min(float(image_width), x))
    y0 = max(0.0, min(float(image_height), y))
    x1 = max(0.0, min(float(image_width), x + box_width))
    y1 = max(0.0, min(float(image_height), y + box_height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("region does not overlap the input image.")

    coordinates = (
        x0 / image_width,
        y0 / image_height,
        x1 / image_width,
        y1 / image_height,
    )
    bins = [
        max(0, min(999, math.floor(coordinate * 1000))) for coordinate in coordinates
    ]
    return "".join(f"<loc_{value}>" for value in bins)


def _task_extra_input(task_name, text_input, region, image_size):
    """Validate and prepare the optional suffix for a Florence task prompt."""

    try:
        spec = TASKS[task_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Florence-2 task: {task_name}") from exc

    text = (text_input or "").strip()
    if spec.input_kind == "none":
        if text:
            raise ValueError(f"{task_name} does not accept text input.")
        if region is not None:
            raise ValueError(f"{task_name} does not accept a region input.")
        return ""
    if spec.input_kind == "text":
        if not text:
            raise ValueError(f"{task_name} requires text input.")
        if region is not None:
            raise ValueError(f"{task_name} does not accept a region input.")
        return text
    if spec.input_kind == "region":
        if text:
            raise ValueError(
                f"{task_name} uses the region input and does not accept text."
            )
        if region is None:
            raise ValueError(f"{task_name} requires a connected BOUNDING_BOX region.")
        return _encode_region(region, image_size)
    raise RuntimeError(f"Unknown Florence task input kind: {spec.input_kind}")


class FlorencePredictor:
    def __init__(self, model_label):
        transformers = require_module("transformers")
        repo_id = MODELS[model_label]
        path = snapshot_download(
            repo_id,
            f"florence2/{repo_id.replace('/', '--')}",
            ignore_patterns=["*.bin"],
        )
        self.dtype = torch_dtype("float16")
        self.processor = transformers.Florence2Processor.from_pretrained(path)
        model = transformers.Florence2ForConditionalGeneration.from_pretrained(
            path,
            dtype=self.dtype,
        )
        model.eval()
        self.handle = ManagedTorchModel(model, processor=self.processor)

    def close(self):
        self.handle.close()
        self.processor = None

    def run(self, image, task_token, text, max_new_tokens, beams):
        prompt = task_token + (text.strip() if text.strip() else "")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        model = self.handle.ensure_loaded()
        device = model_device(model)
        inputs = move_inputs(inputs, device, floating_dtype=self.dtype)
        with torch.inference_mode(), inference_context(device, self.dtype):
            generated = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                num_beams=int(beams),
                do_sample=False,
                early_stopping=int(beams) > 1,
            )
        raw = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            raw, task=task_token, image_size=image.size
        )
        return raw, parsed


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


_SPATIAL_KEYS = frozenset(
    {
        "bboxes",
        "quad_boxes",
        "polygons",
        "labels",
        "bboxes_labels",
        "polygons_labels",
    }
)


def _spatial_result(parsed):
    if not isinstance(parsed, dict):
        return {}
    if _SPATIAL_KEYS.intersection(parsed):
        return parsed
    result = next(iter(parsed.values()), {})
    return result if isinstance(result, dict) else {}


def _stable_color(kind, index, label):
    key = f"{kind}:{index}:{label}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(key, digest_size=3).digest()
    return tuple(64 + channel % 192 for channel in digest)


def _points(values, image_size):
    if not isinstance(values, (list, tuple)) or len(values) < 6:
        return []
    width, height = image_size
    points = []
    for index in range(0, len(values) - 1, 2):
        x, y = values[index], values[index + 1]
        if not isinstance(x, Real) or not isinstance(y, Real):
            return []
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            return []
        points.append(
            (
                max(0, min(width - 1, round(float(x)))),
                max(0, min(height - 1, round(float(y)))),
            )
        )
    return points


def _box(values, image_size):
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return None
    if not all(isinstance(value, Real) for value in values[:4]):
        return None
    coordinates = [float(value) for value in values[:4]]
    if not all(math.isfinite(value) for value in coordinates):
        return None
    x0, y0, x1, y1 = coordinates
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    width, height = image_size
    x0 = max(0, min(width - 1, round(x0)))
    x1 = max(0, min(width - 1, round(x1)))
    y0 = max(0, min(height - 1, round(y0)))
    y1 = max(0, min(height - 1, round(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _polygon_list(group):
    if not isinstance(group, (list, tuple)) or not group:
        return []
    if isinstance(group[0], Real):
        return [group]
    return [item for item in group if isinstance(item, (list, tuple))]


def _label_with_score(labels, scores, index):
    label = str(labels[index]) if index < len(labels) else ""
    if index < len(scores) and isinstance(scores[index], Real):
        score = f"{float(scores[index]):.3f}"
        return f"{label} {score}".strip()
    return label


def _draw_label(draw, position, text, color, image_size):
    if not text:
        return
    x, y = position
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text)
        text_width, text_height = right - left, bottom - top
    except AttributeError:
        text_width, text_height = draw.textlength(text), 11
    width, height = image_size
    x = max(0, min(width - text_width - 4, x))
    y = max(0, min(height - text_height - 4, y))
    background = (0, 0, 0) if sum(color) > 360 else (255, 255, 255)
    foreground = (255, 255, 255) if background == (0, 0, 0) else (0, 0, 0)
    draw.rectangle(
        (x, y, x + text_width + 4, y + text_height + 4),
        fill=background,
    )
    draw.text((x + 2, y + 2), text, fill=foreground)


def _visualize(image, parsed):
    result = _spatial_result(parsed)
    mask = Image.new("L", image.size, 0)
    visual = image.copy().convert("RGB")
    mask_draw = ImageDraw.Draw(mask)
    draw = ImageDraw.Draw(visual)
    width = max(2, min(8, round(min(image.size) / 256 * 3)))
    labels = result.get("labels", [])
    scores = result.get("scores", [])

    box_labels = result.get("bboxes_labels", labels)
    for index, values in enumerate(result.get("bboxes", [])):
        box = _box(values, image.size)
        if box is None:
            continue
        label = _label_with_score(box_labels, scores, index)
        color = _stable_color("box", index, label)
        mask_draw.rectangle(box, fill=255)
        draw.rectangle(box, outline=color, width=width)
        _draw_label(draw, (box[0], box[1]), label, color, image.size)

    for index, values in enumerate(result.get("quad_boxes", [])):
        points = _points(values, image.size)
        if len(points) < 3:
            continue
        label = _label_with_score(labels, scores, index)
        color = _stable_color("quad", index, label)
        mask_draw.polygon(points, fill=255)
        draw.line(points + [points[0]], fill=color, width=width)
        _draw_label(draw, points[0], label, color, image.size)

    polygon_labels = result.get("polygons_labels", labels)
    for index, group in enumerate(result.get("polygons", [])):
        label = _label_with_score(polygon_labels, scores, index)
        color = _stable_color("polygon", index, label)
        label_drawn = False
        for polygon in _polygon_list(group):
            points = _points(polygon, image.size)
            if len(points) < 3:
                continue
            mask_draw.polygon(points, fill=255)
            draw.line(points + [points[0]], fill=color, width=width)
            if not label_drawn:
                _draw_label(draw, points[0], label, color, image.size)
                label_drawn = True
    return mask, visual


class Florence2(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "task": (list(TASKS),),
                "text_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Required only for phrase grounding, referring-expression "
                            "segmentation, and open-vocabulary detection."
                        ),
                    },
                ),
                "model": (
                    list(MODELS),
                    {"default": "Florence-2 large FT (recommended)"},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 4096},
                ),
                "beams": ("INT", {"default": 3, "min": 1, "max": 8}),
            },
            "optional": {
                "unload_after": ("BOOLEAN", {"default": False}),
                "region": (
                    "BOUNDING_BOX",
                    {
                        "tooltip": (
                            "Core bounding box input required by Region to "
                            "Segmentation/Category/Description/OCR."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("text", "structured_json", "mask", "visualization")
    FUNCTION = "run"
    CATEGORY = "VLM Nodes/Florence-2"

    def run(
        self,
        image,
        task,
        text_input,
        model,
        max_new_tokens,
        beams,
        unload_after=False,
        region=None,
    ):
        images = tensor_batch_to_pil(image)
        if not images:
            raise ValueError("Florence-2 requires at least one input image.")
        try:
            spec = TASKS[task]
        except KeyError as exc:
            raise ValueError(f"Unsupported Florence-2 task: {task}") from exc

        extra_inputs = []
        for index, pil_image in enumerate(images):
            selected_region = _select_region(region, index, len(images))
            extra_inputs.append(
                _task_extra_input(
                    task,
                    text_input,
                    selected_region,
                    pil_image.size,
                )
            )

        predictor = self.get_or_create_model(model, lambda: FlorencePredictor(model))
        texts, records, masks, visuals = [], [], [], []
        try:
            for pil_image, extra_input in zip(images, extra_inputs):
                raw, parsed = predictor.run(
                    pil_image,
                    spec.token,
                    extra_input,
                    max_new_tokens,
                    beams,
                )
                texts.append(_clean_decoded_text(raw))
                records.append(parsed)
                mask, visual = _visualize(pil_image, parsed)
                masks.append(pil_mask_to_tensor(mask))
                visuals.append(pil_to_tensor(visual))
            return (
                batch_text(texts),
                json.dumps(
                    records,
                    ensure_ascii=False,
                    default=_json_default,
                    sort_keys=True,
                ),
                torch.cat(masks),
                torch.cat(visuals),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"Florence2": Florence2}
NODE_DISPLAY_NAME_MAPPINGS = {"Florence2": "Florence-2 Multitask Vision"}
