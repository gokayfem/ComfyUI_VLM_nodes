"""Prompt-seeded SAM2.1 image-batch/video segmentation and tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .geometry import bbox_from_mask, deterministic_color
from .runtime import (
    CachedModelNode,
    ManagedTorchModel,
    inference_context,
    model_device,
    require_module,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)
from .vision_types import (
    VLM_DETECTIONS,
    VLM_TRACKS,
    Detection,
    DetectionSequence,
    Track,
    TrackSequence,
)


@dataclass(frozen=True)
class Sam2Spec:
    model_id: str
    cache_name: str


SAM2_MODELS = {
    "SAM2.1 Hiera Tiny (fast)": Sam2Spec(
        "facebook/sam2.1-hiera-tiny", "sam2.1-hiera-tiny"
    ),
    "SAM2.1 Hiera Small": Sam2Spec("facebook/sam2.1-hiera-small", "sam2.1-hiera-small"),
    "SAM2.1 Hiera Base+": Sam2Spec(
        "facebook/sam2.1-hiera-base-plus", "sam2.1-hiera-base-plus"
    ),
    "SAM2.1 Hiera Large": Sam2Spec("facebook/sam2.1-hiera-large", "sam2.1-hiera-large"),
}


def _core_box(value: dict[str, Any]) -> tuple[float, float, float, float]:
    try:
        x, y = float(value["x"]), float(value["y"])
        width, height = float(value["width"]), float(value["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "BOUNDING_BOX must contain numeric x, y, width, and height."
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError("BOUNDING_BOX width and height must be positive.")
    return x, y, x + width, y + height


def _core_box_frames(value: Any) -> list[list[dict[str, Any]]]:
    """Normalize core dict/flat/nested BOUNDING_BOX values to frame lists."""

    if value is None:
        return []
    if isinstance(value, dict):
        return [[value]]
    if not isinstance(value, list):
        raise TypeError("BOUNDING_BOX must be a dict, list of dicts, or frame list.")
    if not value:
        return []
    if all(isinstance(item, dict) for item in value):
        return [value]
    if all(
        isinstance(frame, list) and all(isinstance(item, dict) for item in frame)
        for frame in value
    ):
        return value
    raise TypeError("BOUNDING_BOX contains an unsupported nested value.")


def _box_label(value: dict[str, Any]) -> str | None:
    label = value.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        label = metadata.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return None


def seed_boxes(
    *,
    width: int,
    height: int,
    frame_index: int,
    detections: DetectionSequence | None,
    bounding_box: Any,
) -> tuple[list[list[float]], list[int], dict[int, str | None]]:
    boxes: list[list[float]] = []
    object_ids: list[int] = []
    labels: dict[int, str | None] = {}
    if detections is not None:
        if not isinstance(detections, DetectionSequence):
            raise TypeError("detections must be a VLM Detection Sequence.")
        if detections.width != width or detections.height != height:
            raise ValueError(
                "Detection dimensions must exactly match the SAM2 video frames."
            )
        frame = detections.frame(frame_index)
        if frame is None and len(detections.frames) == 1:
            # A detector run over a selected single image is an explicit seed
            # annotation and may be applied to any chosen video frame.
            frame = detections.frames[0]
        elif frame is None and detections.frames:
            raise ValueError(
                f"Detections do not contain the requested seed frame {frame_index}."
            )
        for index, detection in enumerate(frame.detections if frame else (), 1):
            track_id = detection.track_id
            object_id = int(track_id if track_id is not None else index)
            while object_id in object_ids:
                object_id += 1
            x1, y1, x2, y2 = detection.bbox_xyxy
            boxes.append(
                [
                    min(width, max(0.0, x1)),
                    min(height, max(0.0, y1)),
                    min(width, max(0.0, x2)),
                    min(height, max(0.0, y2)),
                ]
            )
            object_ids.append(object_id)
            labels[object_id] = detection.label
    box_frames = _core_box_frames(bounding_box)
    if len(box_frames) == 1:
        selected_boxes = box_frames[0]
    elif box_frames and frame_index < len(box_frames):
        selected_boxes = box_frames[frame_index]
    elif box_frames:
        raise ValueError(
            f"BOUNDING_BOX has {len(box_frames)} frames but seed_frame is "
            f"{frame_index}."
        )
    else:
        selected_boxes = []
    for value in selected_boxes:
        x1, y1, x2, y2 = _core_box(value)
        object_id = max(object_ids, default=0) + 1
        boxes.append(
            [
                min(width, max(0.0, x1)),
                min(height, max(0.0, y1)),
                min(width, max(0.0, x2)),
                min(height, max(0.0, y2)),
            ]
        )
        object_ids.append(object_id)
        labels[object_id] = _box_label(value)
    valid = []
    for box, object_id in zip(boxes, object_ids):
        if box[2] > box[0] and box[3] > box[1]:
            valid.append((box, object_id))
    return (
        [box for box, _object_id in valid],
        [object_id for _box, object_id in valid],
        labels,
    )


def _normalize_processed_masks(value: torch.Tensor) -> torch.Tensor:
    masks = value.detach().to(device="cpu")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise RuntimeError(
            f"SAM2 returned an unsupported mask shape {tuple(masks.shape)}."
        )
    return masks if masks.dtype == torch.bool else masks > 0.5


class Sam2VideoPredictor:
    def __init__(self, spec: Sam2Spec, precision: str):
        transformers = require_module("transformers")
        if not hasattr(transformers, "Sam2VideoModel"):
            raise RuntimeError(
                "SAM2 video requires Transformers with Sam2VideoModel support."
            )
        model_path = snapshot_download(
            spec.model_id,
            spec.cache_name,
            ignore_patterns=["*.pt", "*.bin", "*.onnx", "*.tflite"],
        )
        self.processor = transformers.Sam2VideoProcessor.from_pretrained(model_path)
        self.dtype = torch_dtype(precision)
        model = transformers.Sam2VideoModel.from_pretrained(
            model_path, dtype=self.dtype
        )
        model.eval()
        self.handle = ManagedTorchModel(model, processor=self.processor)
        self.spec = spec

    def close(self):
        self.handle.close()

    def propagate(
        self,
        images: torch.Tensor,
        *,
        seed_frame: int,
        fps: float,
        detections: DetectionSequence | None,
        bounding_box: Any,
        seed_mask: torch.Tensor | None,
        mask_threshold: float,
        keep_video_on_cpu: bool,
        mask_output: str,
        render_preview: bool,
    ) -> tuple[TrackSequence, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive.")
        if mask_output not in {"union_only", "union_and_objects"}:
            raise ValueError(f"Unsupported mask_output mode {mask_output!r}.")
        pil_images = tensor_batch_to_pil(images)
        if not pil_images:
            raise ValueError("SAM2 requires at least one image.")
        if not 0 <= seed_frame < len(pil_images):
            raise ValueError(
                f"seed_frame {seed_frame} is outside the {len(pil_images)}-frame batch."
            )
        width, height = pil_images[0].size
        if any(image.size != (width, height) for image in pil_images):
            raise ValueError("Every video frame must have identical dimensions.")

        boxes, object_ids, labels = seed_boxes(
            width=width,
            height=height,
            frame_index=seed_frame,
            detections=detections,
            bounding_box=bounding_box,
        )
        masks_for_seed = None
        if not boxes and seed_mask is not None:
            masks_for_seed = seed_mask.detach().to(device="cpu", dtype=torch.float32)
            if masks_for_seed.ndim == 2:
                masks_for_seed = masks_for_seed.unsqueeze(0)
            if masks_for_seed.ndim != 3 or tuple(masks_for_seed.shape[-2:]) != (
                height,
                width,
            ):
                raise ValueError("seed_mask must have shape [objects, height, width].")
            object_ids = list(range(1, masks_for_seed.shape[0] + 1))
            labels = dict.fromkeys(object_ids)
        if not object_ids:
            raise ValueError(
                "Connect detections, a BOUNDING_BOX, or at least one seed mask."
            )

        model = self.handle.ensure_loaded()
        device = model_device(model)
        state_device = torch.device("cpu") if keep_video_on_cpu else device
        processing_device = state_device if keep_video_on_cpu else device
        object_count = len(object_ids)
        union = torch.zeros((len(pil_images), height, width), dtype=torch.float32)
        if mask_output == "union_and_objects":
            individual = torch.zeros(
                (len(pil_images) * object_count, height, width),
                dtype=torch.float32,
            )
        else:
            individual = torch.zeros((0, height, width), dtype=torch.float32)
        source_preview = images.detach().to(device="cpu", dtype=torch.float32)
        preview = source_preview.clone() if render_preview else source_preview
        with torch.inference_mode(), inference_context(device, self.dtype):
            session = self.processor.init_video_session(
                video=pil_images,
                inference_device=device,
                inference_state_device=state_device,
                processing_device=processing_device,
                video_storage_device=state_device,
                max_vision_features_cache_size=1,
                dtype=self.dtype,
            )
            seed_kwargs: dict[str, Any] = {
                "inference_session": session,
                "frame_idx": int(seed_frame),
                "obj_ids": object_ids,
            }
            if boxes:
                seed_kwargs["input_boxes"] = [boxes]
            else:
                seed_kwargs["input_masks"] = [
                    masks_for_seed[index] for index in range(len(object_ids))
                ]
            self.processor.add_inputs_to_inference_session(**seed_kwargs)

            per_object: dict[int, dict[int, Detection]] = {
                object_id: {} for object_id in object_ids
            }

            def record_output(output):
                frame_index = int(output.frame_idx)
                processed = self.processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[[height, width]],
                    mask_threshold=float(mask_threshold),
                    binarize=True,
                )[0]
                processed = _normalize_processed_masks(processed)
                current_ids = list(getattr(session, "obj_ids", object_ids))
                if render_preview:
                    # The Transformers iterator may revisit the seed frame in
                    # both directions. Rebuild that frame from the immutable
                    # source so opacity is never accumulated across visits.
                    preview[frame_index].copy_(source_preview[frame_index])
                for object_index, object_id in enumerate(current_ids):
                    if object_index >= processed.shape[0]:
                        continue
                    mask = processed[object_index]
                    float_mask = mask.to(dtype=torch.float32)
                    union[frame_index] = torch.maximum(union[frame_index], float_mask)
                    if mask_output == "union_and_objects":
                        individual[frame_index * object_count + object_index] = (
                            float_mask
                        )
                    if render_preview:
                        color = torch.tensor(
                            deterministic_color(object_id),
                            dtype=preview.dtype,
                        ).div(255.0)
                        alpha = float_mask.unsqueeze(-1) * 0.45
                        preview[frame_index] = (
                            preview[frame_index] * (1.0 - alpha) + color * alpha
                        )
                    bbox = bbox_from_mask(mask)
                    if bbox is None:
                        continue
                    per_object.setdefault(int(object_id), {})[frame_index] = Detection(
                        bbox_xyxy=bbox,
                        label=labels.get(int(object_id)),
                        frame_index=frame_index,
                        timestamp=frame_index / fps,
                        track_id=int(object_id),
                        source=self.spec.model_id,
                        metadata={
                            "observation": (
                                "detected"
                                if frame_index == seed_frame
                                else "propagated"
                            ),
                            **(
                                {
                                    "mask_batch_index": (
                                        frame_index * object_count + object_index
                                    )
                                }
                                if mask_output == "union_and_objects"
                                else {"object_mask_output": "disabled"}
                            ),
                        },
                    )

            # SAM2 does not consider prompt insertion itself an inference pass.
            # Running the conditioned frame establishes the track start before
            # either propagation direction is requested.
            record_output(model(inference_session=session, frame_idx=seed_frame))
            for output in model.propagate_in_video_iterator(
                inference_session=session,
                start_frame_idx=seed_frame,
                show_progress_bar=False,
            ):
                record_output(output)
            if seed_frame > 0:
                for output in model.propagate_in_video_iterator(
                    inference_session=session,
                    start_frame_idx=seed_frame,
                    reverse=True,
                    show_progress_bar=False,
                ):
                    record_output(output)

        tracks = tuple(
            Track(
                track_id=object_id,
                detections=tuple(
                    records[frame_index] for frame_index in sorted(records)
                ),
                label=labels.get(object_id),
                source=self.spec.model_id,
                metadata={"backend": "transformers-sam2-video"},
            )
            for object_id, records in sorted(per_object.items())
            if records
        )
        track_sequence = TrackSequence(
            width=width,
            height=height,
            tracks=tracks,
            frame_count=len(pil_images),
            fps=fps,
            source=self.spec.model_id,
            metadata={
                "seed_frame": seed_frame,
                "object_ids": object_ids,
                "mask_output": mask_output,
                "mask_order": (
                    "frame_major_object_minor"
                    if mask_output == "union_and_objects"
                    else None
                ),
            },
        )
        if render_preview:
            preview.clamp_(0, 1)
        return track_sequence, union, individual, preview


class VLMSAM2VideoSegmentation(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (tuple(SAM2_MODELS),),
                "seed_frame": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.001,
                    },
                ),
            },
            "optional": {
                "detections": (VLM_DETECTIONS,),
                "bounding_box": ("BOUNDING_BOX",),
                "seed_mask": ("MASK",),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05},
                ),
                "precision": (("auto", "bfloat16", "float16", "float32"),),
                "keep_video_on_cpu": ("BOOLEAN", {"default": True}),
                "mask_output": (
                    ("union_only", "union_and_objects"),
                    {
                        "default": "union_only",
                        "tooltip": (
                            "Per-object full-resolution masks can be very large."
                        ),
                    },
                ),
                "render_preview": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Disable to return the input batch without another "
                            "full-size overlay copy."
                        ),
                    },
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (VLM_TRACKS, "STRING", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = (
        "tracks",
        "json",
        "union_masks",
        "object_masks",
        "preview",
    )
    FUNCTION = "segment"
    CATEGORY = "VLM Nodes/Vision/Segmentation"
    DESCRIPTION = (
        "Track detection boxes or masks through an IMAGE batch with SAM2.1. "
        "Connect the fps output of Get Video Components for correct timestamps."
    )

    def segment(
        self,
        images,
        model,
        seed_frame,
        fps,
        detections=None,
        bounding_box=None,
        seed_mask=None,
        mask_threshold=0.0,
        precision="auto",
        keep_video_on_cpu=True,
        mask_output="union_only",
        render_preview=True,
        unload_after=False,
    ):
        fps_value = float(fps)
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ValueError("fps must be finite and positive.")
        spec = SAM2_MODELS[model]
        predictor = self.get_or_create_model(
            (spec.model_id, precision),
            lambda: Sam2VideoPredictor(spec, precision),
        )
        try:
            tracks, union, individual, preview = predictor.propagate(
                images,
                seed_frame=int(seed_frame),
                fps=fps_value,
                detections=detections,
                bounding_box=bounding_box,
                seed_mask=seed_mask,
                mask_threshold=float(mask_threshold),
                keep_video_on_cpu=bool(keep_video_on_cpu),
                mask_output=mask_output,
                render_preview=bool(render_preview),
            )
            return tracks, tracks.to_json(indent=2), union, individual, preview
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {
    "VLMSAM2VideoSegmentation": VLMSAM2VideoSegmentation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMSAM2VideoSegmentation": "VLM SAM2.1 Video Segmentation",
}
