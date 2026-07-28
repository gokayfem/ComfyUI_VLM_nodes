"""Fast open-vocabulary object detection with maintained Transformers models.

The node deliberately presents one stable ComfyUI interface while keeping
model-specific preprocessing and postprocessing behind a small adapter.  Model
downloads are lazy, inference participates in ComfyUI's VRAM management, and
all spatial output uses the pack's versioned pixel-coordinate contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import ImageDraw

from .geometry import deterministic_color
from .runtime import (
    CachedModelNode,
    ManagedTorchModel,
    inference_context,
    model_device,
    move_inputs,
    require_module,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)
from .vision_types import (
    VLM_DETECTIONS,
    Detection,
    DetectionSequence,
    FrameDetections,
)


@dataclass(frozen=True)
class DetectorSpec:
    model_id: str
    cache_name: str
    family: str
    description: str


MODEL_SPECS = {
    "Grounding DINO Tiny (fast)": DetectorSpec(
        "IDEA-Research/grounding-dino-tiny",
        "grounding-dino-tiny",
        "grounding_dino",
        "Fast, accurate open-vocabulary grounding.",
    ),
    "Grounding DINO Base": DetectorSpec(
        "IDEA-Research/grounding-dino-base",
        "grounding-dino-base",
        "grounding_dino",
        "Higher-quality open-vocabulary grounding.",
    ),
    "OWLv2 Base Ensemble": DetectorSpec(
        "google/owlv2-base-patch16-ensemble",
        "owlv2-base-patch16-ensemble",
        "owlv2",
        "Strong zero-shot detector for lists of visual concepts.",
    ),
    "OmDet Turbo Swin Tiny (fast)": DetectorSpec(
        "omlab/omdet-turbo-swin-tiny-hf",
        "omdet-turbo-swin-tiny",
        "omdet",
        "Efficient real-time-oriented open-vocabulary detector.",
    ),
}


def parse_labels(value: str) -> list[str]:
    """Parse user concepts without splitting meaningful multi-word labels."""

    labels: list[str] = []
    for line in str(value or "").replace(";", "\n").splitlines():
        for candidate in line.split(","):
            label = " ".join(candidate.strip().split())
            if label and label not in labels:
                labels.append(label)
    if not labels:
        raise ValueError("Enter at least one object label or referring phrase.")
    return labels


def _safe_score(value: Any) -> float:
    score = float(value.item() if hasattr(value, "item") else value)
    return min(1.0, max(0.0, score))


def _result_labels(result: dict[str, Any], labels: list[str]) -> list[str]:
    text_labels = result.get("text_labels")
    if text_labels is not None:
        return [str(label) for label in text_labels]

    raw_labels = result.get("labels", result.get("classes", []))
    resolved = []
    for value in raw_labels:
        if isinstance(value, str):
            resolved.append(value)
            continue
        index = int(value.item() if hasattr(value, "item") else value)
        resolved.append(labels[index] if 0 <= index < len(labels) else str(index))
    return resolved


def result_to_detections(
    result: dict[str, Any],
    *,
    labels: list[str],
    width: int,
    height: int,
    frame_index: int,
    timestamp: float,
    source: str,
    max_detections: int,
) -> tuple[Detection, ...]:
    """Normalize a Transformers detector result into immutable detections."""

    boxes = result.get("boxes", ())
    scores = result.get("scores", ())
    resolved_labels = _result_labels(result, labels)
    count = min(len(boxes), len(scores), len(resolved_labels))
    records = []
    for index in range(count):
        box_value = boxes[index]
        if hasattr(box_value, "detach"):
            box_value = box_value.detach().to(device="cpu").tolist()
        x1, y1, x2, y2 = (float(value) for value in box_value)
        x1 = min(float(width), max(0.0, x1))
        y1 = min(float(height), max(0.0, y1))
        x2 = min(float(width), max(x1, x2))
        y2 = min(float(height), max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        records.append(
            Detection(
                bbox_xyxy=(x1, y1, x2, y2),
                label=resolved_labels[index].strip() or None,
                score=_safe_score(scores[index]),
                frame_index=frame_index,
                timestamp=timestamp,
                source=source,
                metadata={"model_id": source},
            )
        )
    records.sort(
        key=lambda item: (
            -(item.score or 0.0),
            item.label or "",
            item.bbox_xyxy,
        )
    )
    return tuple(records[:max_detections])


def _post_process(
    processor: Any,
    spec: DetectorSpec,
    outputs: Any,
    inputs: dict[str, Any],
    labels: list[str],
    sizes: list[tuple[int, int]],
    box_threshold: float,
    text_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    if spec.family == "grounding_dino":
        kwargs = {
            "threshold": float(box_threshold),
            "text_threshold": float(text_threshold),
            "target_sizes": sizes,
        }
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        return processor.post_process_grounded_object_detection(outputs, **kwargs)
    if spec.family == "omdet":
        return processor.post_process_grounded_object_detection(
            outputs,
            text_labels=[labels] * len(sizes),
            threshold=float(box_threshold),
            nms_threshold=float(nms_threshold),
            target_sizes=sizes,
            max_num_det=int(max_detections),
        )
    return processor.post_process_grounded_object_detection(
        outputs,
        threshold=float(box_threshold),
        target_sizes=sizes,
        text_labels=[labels] * len(sizes),
    )


class OpenVocabularyDetector:
    def __init__(self, spec: DetectorSpec, precision: str = "auto"):
        transformers = require_module("transformers")
        model_path = snapshot_download(
            spec.model_id,
            spec.cache_name,
            ignore_patterns=["*.bin", "*.gguf", "*.onnx", "*.tflite"],
        )
        processor = transformers.AutoProcessor.from_pretrained(model_path)
        model_class = transformers.AutoModelForZeroShotObjectDetection
        dtype = torch_dtype(precision)
        model = model_class.from_pretrained(model_path, dtype=dtype)
        model.eval()
        self.spec = spec
        self.dtype = dtype
        self.processor = processor
        self.handle = ManagedTorchModel(model, processor=processor)

    def close(self):
        self.handle.close()

    def detect(
        self,
        images: torch.Tensor,
        labels: list[str],
        *,
        box_threshold: float,
        text_threshold: float,
        nms_threshold: float,
        max_detections: int,
        fps: float,
        batch_size: int,
    ) -> DetectionSequence:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        frames = []
        pil_images = tensor_batch_to_pil(images)
        model = self.handle.ensure_loaded()
        device = model_device(model)
        for start in range(0, len(pil_images), batch_size):
            image_batch = pil_images[start : start + batch_size]
            text = [labels] * len(image_batch)
            inputs = self.processor(
                images=image_batch,
                text=text,
                return_tensors="pt",
            )
            inputs = move_inputs(inputs, device, floating_dtype=self.dtype)
            with torch.inference_mode(), inference_context(device, self.dtype):
                outputs = model(**inputs)
            results = _post_process(
                self.processor,
                self.spec,
                outputs,
                inputs,
                labels,
                [(image.height, image.width) for image in image_batch],
                box_threshold,
                text_threshold,
                nms_threshold,
                max_detections,
            )
            if len(results) != len(image_batch):
                raise RuntimeError(
                    f"{self.spec.model_id} returned {len(results)} result sets "
                    f"for a batch of {len(image_batch)} images."
                )
            for offset, (image, result) in enumerate(
                zip(image_batch, results, strict=True)
            ):
                frame_index = start + offset
                detections = result_to_detections(
                    result,
                    labels=labels,
                    width=image.width,
                    height=image.height,
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    source=self.spec.model_id,
                    max_detections=max_detections,
                )
                frames.append(
                    FrameDetections(
                        frame_index=frame_index,
                        timestamp=frame_index / fps,
                        width=image.width,
                        height=image.height,
                        detections=detections,
                    )
                )
        first = pil_images[0]
        return DetectionSequence(
            width=first.width,
            height=first.height,
            frames=tuple(frames),
            frame_count=len(frames),
            fps=fps,
            source=self.spec.model_id,
            metadata={"labels": labels, "model_family": self.spec.family},
        )


def render_detections(
    images: torch.Tensor, detections: DetectionSequence
) -> torch.Tensor:
    rendered = []
    for index, image in enumerate(tensor_batch_to_pil(images)):
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        frame = detections.frame(index)
        for detection in frame.detections if frame else ():
            color = deterministic_color(
                detection.track_id
                if detection.track_id is not None
                else detection.label or "object"
            )
            color = tuple(int(component) for component in color)
            x1, y1, x2, y2 = detection.bbox_xyxy
            draw.rectangle(
                (x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
                outline=color,
                width=max(2, round(min(image.size) / 256)),
            )
            label = detection.label or "object"
            if detection.score is not None:
                label += f" {detection.score:.2f}"
            text_box = draw.textbbox((x1, y1), label)
            draw.rectangle(text_box, fill=color)
            draw.text((x1, y1), label, fill=(0, 0, 0))
        array = torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy())
        rendered.append(array / 255.0)
    return torch.stack(rendered)


def detection_box_masks(
    detections: DetectionSequence,
) -> torch.Tensor:
    masks = torch.zeros(
        (detections.frame_count, detections.height, detections.width),
        dtype=torch.float32,
    )
    for frame in detections.frames:
        for detection in frame.detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            ix1, iy1 = int(x1), int(y1)
            ix2, iy2 = int(math.ceil(x2)), int(math.ceil(y2))
            masks[frame.frame_index, iy1:iy2, ix1:ix2] = 1.0
    return masks


def _core_box(detection: Detection) -> dict[str, Any]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    left, top = math.floor(x1), math.floor(y1)
    right, bottom = math.ceil(x2), math.ceil(y2)
    return {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
        "label": detection.label,
        "score": detection.score,
        "metadata": {
            "frame_index": detection.frame_index,
            "label": detection.label,
            "score": detection.score,
            "source": detection.source,
        },
    }


def core_bounding_box_frames(
    detections: DetectionSequence,
) -> list[list[dict[str, Any]]]:
    """Return the nested per-frame convention used by core BOUNDING_BOX."""

    frames = [[] for _index in range(detections.frame_count)]
    for frame in detections.frames:
        frames[frame.frame_index] = [
            _core_box(detection) for detection in frame.detections
        ]
    return frames


def core_bounding_boxes(detections: DetectionSequence) -> list[dict[str, Any]]:
    """Return the flat metadata-rich BOUNDING_BOXES contract."""

    result = []
    for frame in detections.frames:
        for detection in frame.detections:
            result.append(_core_box(detection))
    return result


class VLMOpenVocabularyDetection(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (tuple(MODEL_SPECS),),
                "labels": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "person, animal, vehicle",
                        "tooltip": "Comma, semicolon, or newline-separated concepts.",
                    },
                ),
                "box_threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "text_threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_detections": (
                    "INT",
                    {"default": 100, "min": 1, "max": 1000},
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": (
                            "Connect Get Video Components fps for video batches."
                        ),
                    },
                ),
            },
            "optional": {
                "nms_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "precision": (("auto", "bfloat16", "float16", "float32"),),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "tooltip": (
                            "Frames per model call. Increase only when VRAM allows."
                        ),
                    },
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        VLM_DETECTIONS,
        "STRING",
        "IMAGE",
        "MASK",
        "BOUNDING_BOX",
        "BOUNDING_BOXES",
    )
    RETURN_NAMES = (
        "detections",
        "json",
        "preview",
        "box_mask",
        "bounding_boxes",
        "bounding_boxes_with_metadata",
    )
    FUNCTION = "detect"
    CATEGORY = "VLM Nodes/Vision/Detection"
    DESCRIPTION = (
        "Detect text-specified objects with one portable interface. Outputs "
        "versioned detections, JSON, preview, box masks, and core boxes."
    )

    def detect(
        self,
        image,
        model,
        labels,
        box_threshold,
        text_threshold,
        max_detections,
        fps,
        nms_threshold=0.5,
        precision="auto",
        batch_size=1,
        unload_after=False,
    ):
        concepts = parse_labels(labels)
        fps_value = float(fps)
        batch_size_value = int(batch_size)
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ValueError("fps must be finite and positive.")
        if batch_size_value < 1:
            raise ValueError("batch_size must be a positive integer.")
        spec = MODEL_SPECS[model]
        predictor = self.get_or_create_model(
            (spec.model_id, precision),
            lambda: OpenVocabularyDetector(spec, precision),
        )
        try:
            detections = predictor.detect(
                image,
                concepts,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                nms_threshold=nms_threshold,
                max_detections=max_detections,
                fps=fps_value,
                batch_size=batch_size_value,
            )
            return (
                detections,
                detections.to_json(indent=2),
                render_detections(image, detections),
                detection_box_masks(detections),
                core_bounding_box_frames(detections),
                core_bounding_boxes(detections),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {
    "VLMOpenVocabularyDetection": VLMOpenVocabularyDetection,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMOpenVocabularyDetection": "VLM Open-Vocabulary Detection",
}
