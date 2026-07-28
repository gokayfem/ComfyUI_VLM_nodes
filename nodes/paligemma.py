"""PaLI-Gemma captioning, VQA and official VQ-VAE segmentation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageFilter

from .runtime import (
    CachedModelNode,
    ExternalTorchModel,
    ManagedTorchModel,
    batch_text,
    external_device_map,
    hf_download,
    inference_context,
    model_device,
    move_inputs,
    normalize_hf_model_id,
    pil_mask_to_tensor,
    pil_to_tensor,
    require_quantization_backend,
    require_module,
    reserve_external_vram,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)


PALIGEMMA_MODELS = [
    "gokaygokay/sd3-long-captioner-v2",
    "google/paligemma-3b-ft-refcoco-seg-896",
    "google/paligemma-3b-ft-cococap-448",
    "google/paligemma-3b-ft-vqav2-448",
    "google/paligemma-3b-mix-448",
    "google/paligemma-3b-mix-224",
    "Custom",
]
SEGMENT_PATTERN = re.compile(
    r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>"
    r"((?:<seg\d{3}>){16})\s*([^;]*)"
)


class _Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 1),
        )

    def forward(self, value):
        return value + self.net(value)


class PaliMaskDecoder(torch.nn.Module):
    """Decoder architecture and weights published with Google's PaliGemma guide."""

    def __init__(self, weights_path: Path):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, 1),
            torch.nn.ReLU(),
            _Residual(),
            _Residual(),
            torch.nn.ConvTranspose2d(128, 128, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1),
        )
        arrays = np.load(weights_path)
        self.register_buffer(
            "codebook",
            torch.from_numpy(arrays["_vq_vae._embedding"]).float(),
        )
        state = self.decoder.state_dict()
        for key in state:
            source = f"decoder.{key}"
            if source not in arrays:
                raise RuntimeError(f"Official mask decoder is missing {source}.")
            state[key] = torch.from_numpy(arrays[source])
        self.decoder.load_state_dict(state)
        self.eval()

    def forward(self, codes: list[int]) -> torch.Tensor:
        if len(codes) != 16 or any(code not in range(128) for code in codes):
            raise ValueError("A PaliGemma mask must contain 16 <seg000..127> tokens.")
        indices = torch.tensor(codes, device=self.codebook.device)
        latent = self.codebook[indices].reshape(1, 4, 4, 512)
        latent = latent.permute(0, 3, 1, 2)
        # Google's decoder maps its tanh-like output back into [0, 1].
        return (self.decoder(latent) * 0.5 + 0.5).clamp(0, 1)


def parse_segments(text: str):
    parsed = []
    for match in SEGMENT_PATTERN.finditer(text):
        y1, x1, y2, x2 = (int(match.group(i)) / 1024 for i in range(1, 5))
        codes = [int(value) for value in re.findall(r"<seg(\d{3})>", match.group(5))]
        parsed.append(((y1, x1, y2, x2), codes, match.group(6).strip()))
    return parsed


class PaliPredictor:
    def __init__(self, repo_id: str, precision: str, quantization: str):
        transformers = require_module("transformers")
        external = quantization != "None"
        if external:
            # Validate before downloading a multi-gigabyte checkpoint.
            require_quantization_backend(f"PaLI-Gemma {quantization}")
        path = snapshot_download(
            repo_id,
            f"paligemma/{repo_id.replace('/', '--')}",
            ignore_patterns=["*.bin", "*.msgpack", "*.h5"],
        )
        self.dtype = torch_dtype(precision)
        self.processor = transformers.AutoProcessor.from_pretrained(path)
        kwargs: dict[str, Any] = {"dtype": self.dtype}
        if external:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = external_device_map()
            reserve_external_vram(3 * 1024**3)
        model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
            path, **kwargs
        ).eval()
        self.handle = (
            ExternalTorchModel(model, processor=self.processor)
            if external
            else ManagedTorchModel(model, processor=self.processor)
        )
        self.mask_decoder = None

    def close(self):
        self.handle.close()
        self.processor = None
        self.mask_decoder = None

    def generate(self, image, prompt, **generation):
        model = self.handle.ensure_loaded()
        device = model_device(model)
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        )
        inputs = move_inputs(inputs, device)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode(), inference_context(device, self.dtype):
            output = model.generate(**inputs, **generation)
        return self.processor.decode(
            output[0, input_length:], skip_special_tokens=False
        ).strip()

    def decoder(self):
        if self.mask_decoder is None:
            path = hf_download(
                "big-vision/paligemma",
                "vae-oid.npz",
                "paligemma/mask-decoder",
                repo_type="space",
            )
            self.mask_decoder = PaliMaskDecoder(path)
        return self.mask_decoder


def _render_masks(image: Image.Image, segments, decoder, threshold, blur, color, opacity):
    width, height = image.size
    combined = torch.zeros((1, 1, height, width), dtype=torch.float32)
    for (y1, x1, y2, x2), codes, _label in segments:
        left = max(0, min(width - 1, round(x1 * width)))
        top = max(0, min(height - 1, round(y1 * height)))
        right = max(left + 1, min(width, round(x2 * width)))
        bottom = max(top + 1, min(height, round(y2 * height)))
        decoded = decoder(codes)
        resized = F.interpolate(
            decoded,
            size=(bottom - top, right - left),
            mode="bilinear",
            align_corners=False,
        )
        combined[:, :, top:bottom, left:right] = torch.maximum(
            combined[:, :, top:bottom, left:right], resized
        )
    mask = (combined[0, 0] >= float(threshold)).float().numpy() * 255
    mask_image = Image.fromarray(mask.astype(np.uint8), "L")
    if blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(float(blur)))
    rgb = ImageColor.getrgb(color)
    overlay = Image.new("RGBA", image.size, (*rgb, 0))
    overlay.putalpha(mask_image.point(lambda value: round(value * float(opacity))))
    visual = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return mask_image, visual


class Paligemma(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_id": (PALIGEMMA_MODELS,),
                "custom_model_id": ("STRING", {"default": ""}),
                "task_type": (
                    ["Captioning", "Segmentation", "Question Answering"],
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe this image in detail."},
                ),
            },
            "optional": {
                "precision": (["bfloat16", "float32"],),
                # Retained in place so older workflows keep their widget
                # indexes. ComfyUI remains the source of truth for placement.
                "device": (
                    ["auto", "cuda", "cpu", "mps", "xpu"],
                    {"default": "auto"},
                ),
                "quantization": (["None", "8bit", "4bit"],),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0},
                ),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "min_tokens": ("INT", {"default": 0, "min": 0, "max": 512}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0},
                ),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8}),
                "do_sample": (["False", "True"],),
                "early_stopping": (["False", "True"],),
                "fill_mask": (["True", "False"],),
                "mask_color": ("STRING", {"default": "#00ff88"}),
                "mask_opacity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("description", "mask", "visualization")
    FUNCTION = "process_task"
    CATEGORY = "VLM Nodes/Paligemma"

    def process_task(
        self,
        image,
        prompt,
        task_type,
        model_id=None,
        precision="bfloat16",
        device="auto",
        quantization="None",
        custom_model_id="",
        mask_threshold=0.5,
        mask_blur=0,
        max_tokens=256,
        min_tokens=0,
        temperature=0.0,
        num_beams=1,
        do_sample="False",
        early_stopping="False",
        fill_mask="True",
        mask_color="#00ff88",
        mask_opacity=0.5,
        unload_after=False,
        **_legacy,
    ):
        del device
        repo_id = (
            normalize_hf_model_id(custom_model_id)
            if model_id == "Custom"
            else model_id or PALIGEMMA_MODELS[0]
        )
        predictor = self.get_or_create_model(
            (repo_id, precision, quantization),
            lambda: PaliPredictor(repo_id, precision, quantization),
        )
        descriptions, masks, visuals = [], [], []
        try:
            for pil_image in tensor_batch_to_pil(image):
                effective_prompt = (
                    f"segment {prompt}" if task_type == "Segmentation" else prompt
                )
                generated = predictor.generate(
                    pil_image,
                    effective_prompt,
                    max_new_tokens=int(max_tokens),
                    min_new_tokens=int(min_tokens),
                    do_sample=do_sample == "True" and float(temperature) > 0,
                    temperature=max(float(temperature), 1e-5),
                    num_beams=int(num_beams),
                    early_stopping=early_stopping == "True",
                )
                descriptions.append(generated)
                if task_type == "Segmentation":
                    segments = parse_segments(generated)
                    if segments:
                        mask, visual = _render_masks(
                            pil_image,
                            segments,
                            predictor.decoder(),
                            mask_threshold,
                            mask_blur,
                            mask_color,
                            mask_opacity if fill_mask == "True" else 0,
                        )
                    else:
                        mask, visual = (
                            Image.new("L", pil_image.size),
                            pil_image,
                        )
                else:
                    mask, visual = Image.new("L", pil_image.size), pil_image
                masks.append(pil_mask_to_tensor(mask))
                visuals.append(pil_to_tensor(visual))
            return (
                batch_text(descriptions),
                torch.cat(masks),
                torch.cat(visuals),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"Paligemma": Paligemma}
NODE_DISPLAY_NAME_MAPPINGS = {"Paligemma": "PaLI-Gemma (Official Segmentation)"}
