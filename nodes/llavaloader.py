"""llama.cpp multimodal nodes with lazy loading and owned GPU cleanup."""

from __future__ import annotations

from typing import Any

import folder_paths

from .runtime import (
    LLAMA_VISION_HANDLER_CHOICES,
    LlamaHandle,
    LlavaClipConfig,
    batch_text,
    close_handle,
    default_llama_threads,
    image_data_uri,
    llama_chat_content,
    llama_runtime_input_types,
    llama_runtime_options,
    resolve_model_path,
    tensor_batch_to_pil,
    unwrap_llm,
)


def _clip_factory(clip: Any):
    if isinstance(clip, LlavaClipConfig):
        return clip.create
    if callable(getattr(clip, "create", None)):
        return clip.create
    # Compatibility with workflows that pass a pre-created llama.cpp handler.
    return lambda: clip


def _make_handle(
    ckpt_name: str,
    max_ctx: int,
    gpu_layers: int,
    n_threads: int,
    clip: Any,
    *,
    seed: int = 42,
    runtime_options: dict[str, Any] | None = None,
) -> LlamaHandle:
    options = dict(runtime_options or {})
    if isinstance(clip, LlavaClipConfig):
        options.setdefault("projector_path", clip.model_path)
    return LlamaHandle(
        resolve_model_path(ckpt_name),
        n_ctx=max_ctx,
        n_gpu_layers=gpu_layers,
        n_threads=n_threads,
        chat_handler_factory=_clip_factory(clip),
        seed=seed,
        **options,
    )


def _vision_messages(system_msg: str, prompt: str, data_uri: str):
    return [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _run_batch(
    image,
    model,
    *,
    system_msg: str,
    prompt: str,
    **generation: Any,
) -> str:
    llm = unwrap_llm(model)
    responses = []
    for pil_image in tensor_batch_to_pil(image):
        response = llm.create_chat_completion(
            messages=_vision_messages(system_msg, prompt, image_data_uri(pil_image)),
            **generation,
        )
        responses.append(llama_chat_content(response))
    return batch_text(responses)


class LLavaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
                "max_ctx": (
                    "INT",
                    {"default": 4096, "min": 128, "max": 131072, "step": 64},
                ),
                "gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 1000, "step": 1},
                ),
                "n_threads": (
                    "INT",
                    {
                        "default": default_llama_threads(),
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "clip": ("CUSTOM", {"default": ""}),
            },
            "optional": llama_runtime_input_types(),
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_llava_checkpoint"
    CATEGORY = "VLM Nodes/LLava"

    def load_llava_checkpoint(
        self,
        ckpt_name,
        max_ctx,
        gpu_layers,
        n_threads,
        clip,
        n_batch=512,
        n_ubatch=512,
        flash_attention="Auto",
        use_mmap=True,
        split_mode="Layer",
        main_gpu=0,
        tensor_split="",
    ):
        # The GGUF and mmproj are loaded only when a sampler actually executes.
        return (
            _make_handle(
                ckpt_name,
                max_ctx,
                gpu_layers,
                n_threads,
                clip,
                runtime_options=llama_runtime_options(
                    n_batch=n_batch,
                    n_ubatch=n_ubatch,
                    flash_attention=flash_attention,
                    use_mmap=use_mmap,
                    split_mode=split_mode,
                    main_gpu=main_gpu,
                    tensor_split=tensor_split,
                ),
            ),
        )


class LlavaClipLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
            },
            "optional": {
                "handler": (
                    list(LLAMA_VISION_HANDLER_CHOICES),
                    {"default": "Auto (GGUF chat template)"},
                ),
            },
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip_checkpoint"
    CATEGORY = "VLM Nodes/LLava"

    def load_clip_checkpoint(self, clip_name, handler="LLaVA 1.5"):
        return (LlavaClipConfig(resolve_model_path(clip_name), handler),)


class LLavaSamplerSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "VLM Nodes/LLava"

    def generate_text(self, image, prompt, model, temperature):
        return (
            _run_batch(
                image,
                model,
                system_msg="You are an assistant who accurately describes images.",
                prompt=prompt,
                temperature=temperature,
            ),
        )


class LLavaSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "system_msg": (
                    "STRING",
                    {
                        "default": (
                            "You are an assistant who accurately describes images."
                        )
                    },
                ),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 40, "min": 0, "step": 1}),
                "frequency_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 42, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLava"

    def generate_text_advanced(
        self,
        image,
        system_msg,
        prompt,
        model,
        max_tokens,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        repeat_penalty,
        seed,
    ):
        return (
            _run_batch(
                image,
                model,
                system_msg=system_msg,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                seed=seed,
            ),
        )


class _CachedLlavaBase:
    def __init__(self):
        self._handle = None
        self._key = None

    def _model(
        self,
        ckpt_name,
        clip_name,
        max_ctx,
        gpu_layers,
        n_threads,
        seed=42,
        handler="LLaVA 1.5",
        **runtime_options,
    ):
        key = (
            ckpt_name,
            clip_name,
            int(max_ctx),
            int(gpu_layers),
            int(n_threads),
            int(seed),
            handler,
            tuple(sorted(runtime_options.items())),
        )
        if self._handle is None or self._key != key:
            close_handle(self._handle)
            clip = LlavaClipConfig(resolve_model_path(clip_name), handler)
            self._handle = _make_handle(
                ckpt_name,
                max_ctx,
                gpu_layers,
                n_threads,
                clip,
                seed=seed,
                runtime_options=runtime_options,
            )
            self._key = key
        return self._handle

    def _maybe_unload(self, unload):
        if unload:
            close_handle(self._handle)
            self._handle = None
            self._key = None


class LLavaOptionalMemoryFreeSimple(_CachedLlavaBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
                "clip_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
                "max_ctx": (
                    "INT",
                    {"default": 4096, "min": 128, "max": 131072, "step": 64},
                ),
                "gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 1000, "step": 1},
                ),
                "n_threads": (
                    "INT",
                    {
                        "default": default_llama_threads(),
                        "min": 1,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "unload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "handler": (
                    list(LLAMA_VISION_HANDLER_CHOICES),
                    {"default": "Auto (GGUF chat template)"},
                ),
                **llama_runtime_input_types(),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "VLM Nodes/LLava"

    def generate_text(
        self,
        ckpt_name,
        clip_name,
        max_ctx,
        gpu_layers,
        n_threads,
        image,
        prompt,
        temperature,
        unload,
        handler="LLaVA 1.5",
        n_batch=512,
        n_ubatch=512,
        flash_attention="Auto",
        use_mmap=True,
        split_mode="Layer",
        main_gpu=0,
        tensor_split="",
    ):
        options = llama_runtime_options(
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attention=flash_attention,
            use_mmap=use_mmap,
            split_mode=split_mode,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
        )
        model = self._model(
            ckpt_name,
            clip_name,
            max_ctx,
            gpu_layers,
            n_threads,
            handler=handler,
            **options,
        )
        try:
            result = _run_batch(
                image,
                model,
                system_msg="You are an assistant who accurately describes images.",
                prompt=prompt,
                temperature=temperature,
            )
            return (result,)
        finally:
            self._maybe_unload(unload)


class LLavaOptionalMemoryFreeAdvanced(_CachedLlavaBase):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
            "clip_name": (folder_paths.get_filename_list("LLavacheckpoints"),),
            "max_ctx": (
                "INT",
                {"default": 4096, "min": 128, "max": 131072, "step": 64},
            ),
            "gpu_layers": (
                "INT",
                {"default": -1, "min": -1, "max": 1000, "step": 1},
            ),
            "n_threads": (
                "INT",
                {
                    "default": default_llama_threads(),
                    "min": 1,
                    "max": 256,
                    "step": 1,
                },
            ),
            "image": ("IMAGE",),
            "system_msg": (
                "STRING",
                {"default": ("You are an assistant who accurately describes images.")},
            ),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "max_tokens": (
                "INT",
                {"default": 512, "min": 1, "max": 8192, "step": 1},
            ),
            "temperature": (
                "FLOAT",
                {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
            ),
            "top_p": (
                "FLOAT",
                {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "top_k": ("INT", {"default": 40, "min": 0, "step": 1}),
            "frequency_penalty": (
                "FLOAT",
                {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
            ),
            "presence_penalty": (
                "FLOAT",
                {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
            ),
            "repeat_penalty": (
                "FLOAT",
                {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01},
            ),
            "seed": ("INT", {"default": 42, "step": 1}),
            "unload": ("BOOLEAN", {"default": False}),
        }
        return {
            "required": required,
            "optional": {
                "handler": (
                    list(LLAMA_VISION_HANDLER_CHOICES),
                    {"default": "Auto (GGUF chat template)"},
                ),
                **llama_runtime_input_types(),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLava"

    def generate_text_advanced(
        self,
        ckpt_name,
        clip_name,
        max_ctx,
        gpu_layers,
        n_threads,
        image,
        system_msg,
        prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        repeat_penalty,
        seed,
        unload,
        handler="LLaVA 1.5",
        n_batch=512,
        n_ubatch=512,
        flash_attention="Auto",
        use_mmap=True,
        split_mode="Layer",
        main_gpu=0,
        tensor_split="",
    ):
        options = llama_runtime_options(
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attention=flash_attention,
            use_mmap=use_mmap,
            split_mode=split_mode,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
        )
        model = self._model(
            ckpt_name,
            clip_name,
            max_ctx,
            gpu_layers,
            n_threads,
            seed,
            handler,
            **options,
        )
        try:
            result = _run_batch(
                image,
                model,
                system_msg=system_msg,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                seed=seed,
            )
            return (result,)
        finally:
            self._maybe_unload(unload)


NODE_CLASS_MAPPINGS = {
    "LLava Loader Simple": LLavaLoader,
    "LLavaSamplerSimple": LLavaSamplerSimple,
    "LlavaClipLoader": LlavaClipLoader,
    "LLavaSamplerAdvanced": LLavaSamplerAdvanced,
    "LLavaOptionalMemoryFreeSimple": LLavaOptionalMemoryFreeSimple,
    "LLavaOptionalMemoryFreeAdvanced": LLavaOptionalMemoryFreeAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLava Loader Simple": "LLaVA Loader",
    "LLavaSamplerSimple": "LLaVA Sampler",
    "LlavaClipLoader": "LLaVA Vision Projector Loader",
    "LLavaSamplerAdvanced": "LLaVA Sampler (Advanced)",
    "LLavaOptionalMemoryFreeSimple": "LLaVA (Managed Cache)",
    "LLavaOptionalMemoryFreeAdvanced": "LLaVA (Managed Cache, Advanced)",
}
