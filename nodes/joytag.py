"""JoyTag image tagging with cached, ComfyUI-managed model weights."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .runtime import (
    CachedModelNode,
    ManagedTorchModel,
    batch_text,
    inference_context,
    model_device,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)

MODEL_ID = "fancyfeast/joytag"


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image.convert("RGB"), ((side - width) // 2, (side - height) // 2))
    if side != target_size:
        canvas = canvas.resize(
            (target_size, target_size), Image.Resampling.BICUBIC
        )
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
    return (tensor - mean) / std


def clean_tag(tag: str) -> str:
    return (
        tag.replace("(medium)", "")
        .replace("\\", "")
        .replace("m/", "")
        .replace("_", " ")
        .strip(" -")
    )


class JoyTagPredictor:
    def __init__(self):
        from .joytagger import Models

        path = snapshot_download(MODEL_ID, "joytag")
        model = Models.VisionModel.load_model(path, device=None).eval()
        self.tags = [
            line.strip()
            for line in (path / "top_tags.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        self.dtype = torch_dtype("float16")
        self.handle = ManagedTorchModel(model)

    def close(self):
        self.handle.close()
        self.tags = []

    def predict(self, images, count: int, threshold: float):
        results = []
        for image in tensor_batch_to_pil(images):
            model = self.handle.ensure_loaded()
            device = model_device(model)
            tensor = prepare_image(image, model.image_size).unsqueeze(0).to(device)
            with torch.inference_mode(), inference_context(device, self.dtype):
                predictions = model({"image": tensor})["tags"].sigmoid()[0]
            scores = predictions.float().cpu()
            ranked = torch.argsort(scores, descending=True).tolist()
            selected = [
                index
                for index in ranked
                if scores[index].item() >= float(threshold)
            ][: int(count)]
            # Always return up to tag_number useful results, even when the
            # threshold is deliberately high.
            if not selected:
                selected = ranked[: int(count)]
            tags = [clean_tag(self.tags[index]) for index in selected]
            results.append(", ".join(tag for tag in tags if tag))
        return batch_text(results)


class Joytag(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tag_number": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "number",
                    },
                ),
            },
            "optional": {
                "threshold": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "tags"
    CATEGORY = "VLM Nodes/Vision/Tagging"

    def tags(
        self,
        image,
        tag_number,
        threshold=0.4,
        unload_after=False,
    ):
        predictor = self.get_or_create_model(MODEL_ID, JoyTagPredictor)
        try:
            return (
                predictor.predict(image, tag_number, threshold),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"Joytag": Joytag}
NODE_DISPLAY_NAME_MAPPINGS = {"Joytag": "JoyTag"}
