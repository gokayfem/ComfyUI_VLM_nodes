"""Qwen2-VL with real image/video batches and ComfyUI-aware VRAM handling."""

from __future__ import annotations

from typing import Any

import torch

from .runtime import (
    CachedModelNode,
    ExternalTorchModel,
    ManagedTorchModel,
    accelerator_backend,
    batch_text,
    execution_device,
    external_device_map,
    inference_context,
    model_device,
    move_inputs,
    require_quantization_backend,
    require_module,
    reserve_external_vram,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)


QWEN2_VL_MODELS = {
    "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2-VL-72B": "Qwen/Qwen2-VL-72B-Instruct",
}
# Old workflows used separate AWQ/GPTQ repositories whose integration breaks
# across Transformers/AutoGPTQ releases. Resolve those labels to the same base
# weights and the maintained bitsandbytes path instead.
LEGACY_QUANTIZED_ALIASES = {
    f"Qwen2-VL-{size}-{quant}": (
        f"Qwen2-VL-{size}",
        (
            "Balanced (8-bit)"
            if quant.endswith("Int8")
            else "Maximum Savings (4-bit)"
        ),
    )
    for size in ("2B", "7B", "72B")
    for quant in ("AWQ", "GPTQ-Int4", "GPTQ-Int8")
}
QWEN2_VL_CHOICES = ("Qwen2-VL-2B", "Qwen2-VL-7B")

MEMORY_MODES = [
    "ComfyUI managed (BF16)",
    "Balanced (8-bit)",
    "Maximum Savings (4-bit)",
    "CPU Offload",
    "Default",
]

ESTIMATED_MODEL_BYTES = {
    "Qwen2-VL-2B": 5 * 1024**3,
    "Qwen2-VL-7B": 16 * 1024**3,
    "Qwen2-VL-72B": 145 * 1024**3,
}


def _model_class(transformers):
    for name in (
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "This Transformers version does not include a Qwen2-VL model class."
    )


def _attention_value(mode: str) -> str:
    return {
        "Auto (SDPA)": "sdpa",
        "Flash Attention 2": "flash_attention_2",
        "Eager": "eager",
    }[mode]


class Qwen2VLPredictor:
    def __init__(
        self,
        model_name: str,
        memory_mode: str,
        attention_mode: str,
        min_pixels: int,
        max_pixels: int,
    ):
        transformers = require_module("transformers")
        if model_name in LEGACY_QUANTIZED_ALIASES:
            model_name, memory_mode = LEGACY_QUANTIZED_ALIASES[model_name]
        if (
            attention_mode == "Flash Attention 2"
            and accelerator_backend(execution_device())
            not in {"nvidia-cuda", "amd-rocm"}
        ):
            raise RuntimeError(
                "Flash Attention 2 requires a supported CUDA or ROCm build. "
                "Select Auto (SDPA) on Apple Metal, Intel XPU, or CPU."
            )
        if memory_mode in {"Balanced (8-bit)", "Maximum Savings (4-bit)"}:
            # Validate before downloading a multi-gigabyte checkpoint.
            require_quantization_backend(memory_mode)
        repo_id = QWEN2_VL_MODELS[model_name]
        model_path = snapshot_download(
            repo_id,
            f"qwen2vl/{model_name}",
            ignore_patterns=["*.bin"],
        )
        self.dtype = torch_dtype("bfloat16")
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
        )
        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "attn_implementation": _attention_value(attention_mode),
        }
        external = memory_mode in {
            "Balanced (8-bit)",
            "Maximum Savings (4-bit)",
            "CPU Offload",
        }

        if memory_mode in {"Balanced (8-bit)", "Maximum Savings (4-bit)"}:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_8bit=memory_mode == "Balanced (8-bit)",
                load_in_4bit=memory_mode == "Maximum Savings (4-bit)",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        if external:
            require_module("accelerate")
            estimate = ESTIMATED_MODEL_BYTES.get(
                model_name.split("-AWQ", 1)[0].split("-GPTQ", 1)[0],
                8 * 1024**3,
            )
            reserve_external_vram(
                estimate // (4 if memory_mode == "Maximum Savings (4-bit)" else 2)
            )
            kwargs["device_map"] = external_device_map(
                allow_auto_offload=memory_mode == "CPU Offload"
            )

        try:
            model = _model_class(transformers).from_pretrained(
                model_path, **kwargs
            ).eval()
        except ImportError as exc:
            if attention_mode == "Flash Attention 2":
                raise RuntimeError(
                    "Flash Attention 2 was selected but flash-attn is not "
                    "installed for this PyTorch accelerator build. Use Auto "
                    "(SDPA), or install a matching flash-attn wheel."
                ) from exc
            raise

        self.handle = (
            ExternalTorchModel(model, processor=self.processor)
            if external
            else ManagedTorchModel(model, processor=self.processor)
        )

    def close(self):
        self.handle.close()
        self.processor = None

    def _generate_messages(
        self,
        messages,
        *,
        max_new_tokens,
        temperature,
        top_p,
    ) -> str:
        process_vision_info = require_module(
            "qwen_vl_utils", "qwen-vl-utils"
        ).process_vision_info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model = self.handle.ensure_loaded()
        device = model_device(model)
        inputs = move_inputs(inputs, device)
        generation: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": float(temperature) > 0.0,
        }
        if generation["do_sample"]:
            generation.update(
                temperature=float(temperature), top_p=float(top_p)
            )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            generation["pad_token_id"] = tokenizer.pad_token_id
            generation["eos_token_id"] = tokenizer.eos_token_id

        with torch.inference_mode(), inference_context(device, self.dtype):
            output_ids = model.generate(**inputs, **generation)
        trimmed = [
            output[len(input_ids) :]
            for input_ids, output in zip(inputs["input_ids"], output_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_images(
        self, images, prompt, max_new_tokens, temperature, top_p
    ) -> str:
        results = []
        for image in tensor_batch_to_pil(images):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            results.append(
                self._generate_messages(
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        return batch_text(results)

    def generate_video(
        self,
        primary_image,
        frames,
        prompt,
        max_new_tokens,
        temperature,
        top_p,
        fps,
    ) -> str:
        # The still IMAGE socket is required by ComfyUI for backwards
        # compatibility, but a connected frame batch is the visual source for
        # video inference. Mixing both causes small VLMs to answer from the
        # still and ignore temporal content.
        del primary_image
        frame_list = tensor_batch_to_pil(frames)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_list,
                        "fps": float(fps),
                    },
                    {
                        "type": "text",
                        "text": (
                            f"The video frames are sampled at {float(fps):g} "
                            f"FPS.\n\n{prompt}"
                        ),
                    },
                ],
            }
        ]
        return self._generate_messages(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )


class Qwen2VLNode(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image in detail.",
                    },
                ),
                "model_name": (list(QWEN2_VL_CHOICES),),
                "memory_mode": (
                    MEMORY_MODES,
                    {"default": "ComfyUI managed (BF16)"},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_frames": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 60.0, "step": 0.1},
                ),
                "attention_mode": (
                    ["Auto (SDPA)", "Flash Attention 2", "Eager"],
                    {"default": "Auto (SDPA)"},
                ),
                "min_pixels": (
                    "INT",
                    {"default": 256 * 28 * 28, "min": 28 * 28},
                ),
                "max_pixels": (
                    "INT",
                    {"default": 1280 * 28 * 28, "min": 28 * 28},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "VLM Nodes/Legacy/Model Loaders"

    def generate(
        self,
        text_input,
        model_name,
        memory_mode="ComfyUI managed (BF16)",
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        image=None,
        video_frames=None,
        fps=1.0,
        attention_mode="Auto (SDPA)",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        unload_after=False,
    ):
        if min_pixels > max_pixels:
            raise ValueError("min_pixels cannot be greater than max_pixels.")
        if image is None and video_frames is None:
            raise ValueError("Connect either image or video_frames.")
        key = (
            model_name,
            memory_mode,
            attention_mode,
            int(min_pixels),
            int(max_pixels),
        )
        predictor = self.get_or_create_model(
            key,
            lambda: Qwen2VLPredictor(
                model_name,
                memory_mode,
                attention_mode,
                min_pixels,
                max_pixels,
            ),
        )
        try:
            if video_frames is None:
                result = predictor.generate_images(
                    image,
                    text_input,
                    max_new_tokens,
                    temperature,
                    top_p,
                )
            else:
                result = predictor.generate_video(
                    image,
                    video_frames,
                    text_input,
                    max_new_tokens,
                    temperature,
                    top_p,
                    fps,
                )
            return (result,)
        finally:
            self.maybe_clear_model(unload_after)


NODE_CLASS_MAPPINGS = {"Qwen2VLNode": Qwen2VLNode}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen2VLNode": "Qwen2-VL"}
