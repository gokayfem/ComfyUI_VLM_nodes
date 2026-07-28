"""Florence-2 multitask caption, OCR, detection and segmentation node."""

from __future__ import annotations

import json

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
    "Florence-2 large FT (recommended)": (
        "florence-community/Florence-2-large-ft"
    ),
}
TASKS = {
    "Caption": "<CAPTION>",
    "Detailed caption": "<DETAILED_CAPTION>",
    "More detailed caption": "<MORE_DETAILED_CAPTION>",
    "OCR": "<OCR>",
    "OCR with regions": "<OCR_WITH_REGION>",
    "Object detection": "<OD>",
    "Dense region caption": "<DENSE_REGION_CAPTION>",
    "Region proposals": "<REGION_PROPOSAL>",
    "Referring expression segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "Open vocabulary detection": "<OPEN_VOCABULARY_DETECTION>",
}


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
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        )
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
        raw = self.processor.batch_decode(
            generated, skip_special_tokens=False
        )[0]
        parsed = self.processor.post_process_generation(
            raw, task=task_token, image_size=image.size
        )
        return raw, parsed


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _visualize(image, parsed):
    result = next(iter(parsed.values()), parsed) if isinstance(parsed, dict) else {}
    mask = Image.new("L", image.size, 0)
    visual = image.copy().convert("RGB")
    mask_draw = ImageDraw.Draw(mask)
    draw = ImageDraw.Draw(visual)
    labels = result.get("labels", []) if isinstance(result, dict) else []

    for index, box in enumerate(result.get("bboxes", [])):
        box = [float(value) for value in box]
        draw.rectangle(box, outline="#00ff88", width=3)
        if index < len(labels):
            draw.text((box[0] + 3, box[1] + 3), str(labels[index]), fill="#00ff88")

    for quad in result.get("quad_boxes", []):
        points = [
            (float(quad[index]), float(quad[index + 1]))
            for index in range(0, len(quad), 2)
        ]
        draw.line(points + [points[0]], fill="#00c8ff", width=3)

    polygons = result.get("polygons", [])
    for group in polygons:
        # Florence may return either one flat polygon or a list of polygons.
        groups = [group] if group and isinstance(group[0], (int, float)) else group
        for polygon in groups:
            points = [
                (float(polygon[index]), float(polygon[index + 1]))
                for index in range(0, len(polygon), 2)
            ]
            if len(points) >= 3:
                mask_draw.polygon(points, fill=255)
                draw.line(points + [points[0]], fill="#ff4da6", width=3)
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
                        "tooltip": "Required for referring-expression and open-vocabulary tasks.",
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
    ):
        predictor = self.get_or_create_model(
            model, lambda: FlorencePredictor(model)
        )
        texts, records, masks, visuals = [], [], [], []
        try:
            for pil_image in tensor_batch_to_pil(image):
                raw, parsed = predictor.run(
                    pil_image,
                    TASKS[task],
                    text_input,
                    max_new_tokens,
                    beams,
                )
                texts.append(raw)
                records.append(parsed)
                mask, visual = _visualize(pil_image, parsed)
                masks.append(pil_mask_to_tensor(mask))
                visuals.append(pil_to_tensor(visual))
            return (
                batch_text(texts),
                json.dumps(records, ensure_ascii=False, default=_json_default),
                torch.cat(masks),
                torch.cat(visuals),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"Florence2": Florence2}
NODE_DISPLAY_NAME_MAPPINGS = {"Florence2": "Florence-2 Multitask Vision"}
