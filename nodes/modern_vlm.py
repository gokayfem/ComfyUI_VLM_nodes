"""Modern, chat-template based vision-language models.

This node intentionally uses the Transformers multimodal auto classes instead
of model-specific glue.  It provides one stable ComfyUI surface for current
Qwen, Gemma and SmolVLM checkpoints while keeping downloads and VRAM allocation
lazy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .runtime import (
    CachedModelNode,
    ExternalTorchModel,
    ManagedTorchModel,
    batch_text,
    inference_context,
    model_device,
    move_inputs,
    normalize_hf_model_id,
    require_module,
    reserve_external_vram,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    family: str
    estimated_gib: float
    gated: bool = False
    video: bool = False


# Deliberately curated: these are useful tiers, not every redundant checkpoint.
MODEL_CATALOG = {
    "Qwen 3.5 0.8B (fastest current)": ModelSpec(
        "Qwen/Qwen3.5-0.8B", "Qwen 3.5", 2.0, video=True
    ),
    "Qwen 3.5 2B": ModelSpec(
        "Qwen/Qwen3.5-2B", "Qwen 3.5", 4.5, video=True
    ),
    "Qwen 3.5 4B (recommended)": ModelSpec(
        "Qwen/Qwen3.5-4B", "Qwen 3.5", 8.5, video=True
    ),
    "Qwen 3.5 9B": ModelSpec(
        "Qwen/Qwen3.5-9B", "Qwen 3.5", 19.0, video=True
    ),
    "Qwen 3 VL 2B Instruct": ModelSpec(
        "Qwen/Qwen3-VL-2B-Instruct", "Qwen 3 VL", 5.0, video=True
    ),
    "Qwen 3 VL 4B Instruct": ModelSpec(
        "Qwen/Qwen3-VL-4B-Instruct", "Qwen 3 VL", 9.0, video=True
    ),
    "Qwen 3 VL 8B Instruct": ModelSpec(
        "Qwen/Qwen3-VL-8B-Instruct", "Qwen 3 VL", 18.0, video=True
    ),
    "Qwen 3 VL 30B-A3B Instruct (4-bit recommended)": ModelSpec(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen 3 VL", 61.0, video=True
    ),
    "Gemma 3 4B IT (license acceptance required)": ModelSpec(
        "google/gemma-3-4b-it", "Gemma 3", 9.0, gated=True
    ),
    "Gemma 3 12B IT (license acceptance required)": ModelSpec(
        "google/gemma-3-12b-it", "Gemma 3", 25.0, gated=True
    ),
    "Gemma 3 27B IT (4-bit recommended, gated)": ModelSpec(
        "google/gemma-3-27b-it", "Gemma 3", 55.0, gated=True
    ),
    "SmolVLM2 500M Video (low VRAM)": ModelSpec(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "SmolVLM2",
        1.8,
        video=True,
    ),
    "SmolVLM2 2.2B Video": ModelSpec(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "SmolVLM2",
        5.2,
        video=True,
    ),
    "Custom Hugging Face model": ModelSpec("", "Custom", 8.0),
}

MEMORY_MODES = (
    "ComfyUI managed (BF16)",
    "4-bit NF4 (bitsandbytes)",
    "8-bit (bitsandbytes)",
    "CPU",
)
ATTENTION_MODES = ("Auto (SDPA)", "Flash Attention 2", "Eager")


def _model_class(transformers):
    for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    raise RuntimeError(
        "Modern VLMs require a current Transformers release with "
        "AutoModelForImageTextToText support."
    )


class ModernVLMPredictor:
    def __init__(
        self,
        model_label: str,
        custom_model_id: str,
        memory_mode: str,
        attention_mode: str,
    ) -> None:
        transformers = require_module("transformers")
        spec = MODEL_CATALOG[model_label]
        repo_id = (
            normalize_hf_model_id(custom_model_id)
            if spec.family == "Custom"
            else spec.repo_id
        )
        self.spec = spec
        self.dtype = torch_dtype("bfloat16")
        model_path = snapshot_download(
            repo_id,
            f"modern-vlm/{repo_id.replace('/', '--')}",
            ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx"],
        )
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        attention = {
            "Auto (SDPA)": "sdpa",
            "Flash Attention 2": "flash_attention_2",
            "Eager": "eager",
        }[attention_mode]
        kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "trust_remote_code": True,
            "attn_implementation": attention,
        }
        external = memory_mode != "ComfyUI managed (BF16)"
        if memory_mode in {"4-bit NF4 (bitsandbytes)", "8-bit (bitsandbytes)"}:
            require_module("bitsandbytes")
            require_module("accelerate")
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=memory_mode.startswith("4-bit"),
                load_in_8bit=memory_mode.startswith("8-bit"),
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
            divisor = 4 if memory_mode.startswith("4-bit") else 2
            reserve_external_vram(int(spec.estimated_gib * 1024**3 / divisor))
        elif memory_mode == "CPU":
            kwargs["dtype"] = torch.float32

        try:
            model = _model_class(transformers).from_pretrained(
                model_path, **kwargs
            ).eval()
        except OSError as exc:
            if spec.gated:
                raise RuntimeError(
                    f"{repo_id} is gated. Accept its Hugging Face license and "
                    "set HF_TOKEN before running this node."
                ) from exc
            raise
        except ImportError as exc:
            if attention_mode == "Flash Attention 2":
                raise RuntimeError(
                    "Flash Attention 2 is unavailable for this Python/PyTorch "
                    "build. Select Auto (SDPA), or install a matching wheel."
                ) from exc
            raise

        self.handle = (
            ExternalTorchModel(model, processor=self.processor)
            if external
            else ManagedTorchModel(model, processor=self.processor)
        )

    def close(self) -> None:
        self.handle.close()
        self.processor = None

    def _inputs(self, messages):
        """Use the standard multimodal template, with an older-template fallback."""

        try:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError, KeyError):
            media = []
            portable_messages = []
            for message in messages:
                content = []
                for part in message["content"]:
                    if part["type"] == "image":
                        media.append(part["image"])
                        content.append({"type": "image"})
                    elif part["type"] == "video":
                        media.extend(part["video"])
                        content.extend({"type": "image"} for _ in part["video"])
                    else:
                        content.append(part)
                portable_messages.append(
                    {"role": message["role"], "content": content}
                )
            prompt = self.processor.apply_chat_template(
                portable_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            return self.processor(
                text=[prompt], images=media, return_tensors="pt"
            )

    def generate(
        self,
        images,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        video_frames=None,
        fps: float = 1.0,
    ) -> str:
        primary_images = tensor_batch_to_pil(images)
        video = (
            tensor_batch_to_pil(video_frames)
            if video_frames is not None
            else None
        )
        if video is not None and not self.spec.video:
            raise ValueError(
                f"{self.spec.family} does not advertise video support. "
                "Disconnect video_frames or select Qwen/SmolVLM2."
            )

        results = []
        runs = [primary_images[0]] if video is not None else primary_images
        for image in runs:
            messages = []
            if system_prompt.strip():
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_prompt.strip()}
                        ],
                    }
                )
            content = [{"type": "image", "image": image}]
            if video is not None:
                content.append(
                    {"type": "video", "video": video, "fps": float(fps)}
                )
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})

            inputs = self._inputs(messages)
            model = self.handle.ensure_loaded()
            device = model_device(model)
            inputs = move_inputs(inputs, device)
            input_length = inputs["input_ids"].shape[-1]
            generation: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": float(temperature) > 0,
            }
            if generation["do_sample"]:
                generation.update(
                    temperature=float(temperature), top_p=float(top_p)
                )
            with torch.inference_mode(), inference_context(device, self.dtype):
                output = model.generate(**inputs, **generation)
            new_tokens = output[:, input_length:]
            results.append(
                self.processor.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
            )
        return batch_text(results)


class ModernVLM(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image precisely and in detail.",
                    },
                ),
                "model": (
                    list(MODEL_CATALOG),
                    {"default": "Qwen 3.5 4B (recommended)"},
                ),
                "custom_model_id": ("STRING", {"default": ""}),
                "memory_mode": (
                    MEMORY_MODES,
                    {"default": "ComfyUI managed (BF16)"},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 16384},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an expert visual analyst.",
                    },
                ),
                "video_frames": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 60.0, "step": 0.1},
                ),
                "attention_mode": (
                    ATTENTION_MODES,
                    {"default": "Auto (SDPA)"},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "VLM Nodes/Modern"

    def run(
        self,
        image,
        prompt,
        model,
        custom_model_id,
        memory_mode,
        max_new_tokens,
        temperature,
        top_p,
        system_prompt="You are an expert visual analyst.",
        video_frames=None,
        fps=1.0,
        attention_mode="Auto (SDPA)",
        unload_after=False,
    ):
        effective_custom_id = (
            normalize_hf_model_id(custom_model_id)
            if model == "Custom Hugging Face model"
            else ""
        )
        key = (model, effective_custom_id, memory_mode, attention_mode)
        predictor = self.get_or_create_model(
            key,
            lambda: ModernVLMPredictor(
                model, effective_custom_id, memory_mode, attention_mode
            ),
        )
        try:
            return (
                predictor.generate(
                    image,
                    prompt,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    video_frames,
                    fps,
                ),
            )
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"ModernVLM": ModernVLM}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModernVLM": "Modern VLM (Qwen 3.5 / Qwen 3 VL / Gemma 3 / SmolVLM2)"
}
