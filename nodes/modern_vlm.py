"""Modern, chat-template based vision-language models.

This node intentionally uses the Transformers multimodal auto classes instead
of model-specific glue. It provides one stable ComfyUI surface for current
small and large VLM families while keeping downloads and VRAM allocation lazy.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

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
    normalize_hf_model_id,
    require_quantization_backend,
    require_module,
    reserve_external_vram,
    snapshot_download,
    tensor_batch_to_pil,
    torch_dtype,
)
from .vision_types import VLM_VIDEO_SELECTION, VideoFrameSelection


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    family: str
    estimated_gib: float
    gated: bool = False
    video: bool = False
    small_fast: bool = False
    trust_remote_code: bool = False


# Deliberately curated: these are useful tiers, not every redundant checkpoint.
MODEL_CATALOG = {
    "Qwen 3.5 0.8B (fastest current)": ModelSpec(
        "Qwen/Qwen3.5-0.8B",
        "Qwen 3.5",
        2.0,
        video=True,
        small_fast=True,
    ),
    "Qwen 3.5 2B": ModelSpec(
        "Qwen/Qwen3.5-2B",
        "Qwen 3.5",
        4.5,
        video=True,
        small_fast=True,
    ),
    "Qwen 3.5 4B (recommended)": ModelSpec(
        "Qwen/Qwen3.5-4B",
        "Qwen 3.5",
        8.5,
        video=True,
        small_fast=True,
    ),
    "Qwen 3.5 9B": ModelSpec(
        "Qwen/Qwen3.5-9B", "Qwen 3.5", 19.0, video=True
    ),
    "Qwen 3.5 27B (4-bit recommended)": ModelSpec(
        "Qwen/Qwen3.5-27B", "Qwen 3.5", 55.0, video=True
    ),
    "Qwen 3.5 35B-A3B (4-bit recommended)": ModelSpec(
        "Qwen/Qwen3.5-35B-A3B", "Qwen 3.5", 72.0, video=True
    ),
    "Qwen 3.6 27B (4-bit recommended)": ModelSpec(
        "Qwen/Qwen3.6-27B", "Qwen 3.6", 55.0, video=True
    ),
    "Qwen 3 VL 2B Instruct": ModelSpec(
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen 3 VL",
        5.0,
        video=True,
        small_fast=True,
    ),
    "Qwen 3 VL 4B Instruct": ModelSpec(
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen 3 VL",
        9.0,
        video=True,
        small_fast=True,
    ),
    "Qwen 3 VL 8B Instruct": ModelSpec(
        "Qwen/Qwen3-VL-8B-Instruct", "Qwen 3 VL", 18.0, video=True
    ),
    "Qwen 3 VL 30B-A3B Instruct (4-bit recommended)": ModelSpec(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen 3 VL", 61.0, video=True
    ),
    "Qwen 2.5 VL 3B Instruct (legacy workflows)": ModelSpec(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen 2.5 VL",
        7.0,
        video=True,
        small_fast=True,
    ),
    "Qwen 2.5 VL 7B Instruct (legacy workflows)": ModelSpec(
        "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen 2.5 VL", 16.0, video=True
    ),
    "Gemma 3 4B IT (license acceptance required)": ModelSpec(
        "google/gemma-3-4b-it",
        "Gemma 3",
        9.0,
        gated=True,
        small_fast=True,
    ),
    "Gemma 3 12B IT (license acceptance required)": ModelSpec(
        "google/gemma-3-12b-it", "Gemma 3", 25.0, gated=True
    ),
    "Gemma 3 27B IT (4-bit recommended, gated)": ModelSpec(
        "google/gemma-3-27b-it", "Gemma 3", 55.0, gated=True
    ),
    "SmolVLM2 256M Video (smallest)": ModelSpec(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "SmolVLM2",
        1.4,
        video=True,
        small_fast=True,
    ),
    "SmolVLM2 500M Video (low VRAM)": ModelSpec(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "SmolVLM2",
        1.8,
        video=True,
        small_fast=True,
    ),
    "SmolVLM2 2.2B Video": ModelSpec(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "SmolVLM2",
        5.2,
        video=True,
        small_fast=True,
    ),
    "LFM2.5 VL 450M (edge)": ModelSpec(
        "LiquidAI/LFM2.5-VL-450M",
        "LFM2.5 VL",
        1.5,
        small_fast=True,
    ),
    "LFM2.5 VL 1.6B": ModelSpec(
        "LiquidAI/LFM2.5-VL-1.6B",
        "LFM2.5 VL",
        4.0,
        small_fast=True,
    ),
    "InternVL 3.5 1B HF": ModelSpec(
        "OpenGVLab/InternVL3_5-1B-HF",
        "InternVL 3.5",
        2.5,
        video=True,
        small_fast=True,
    ),
    "InternVL 3.5 2B HF": ModelSpec(
        "OpenGVLab/InternVL3_5-2B-HF",
        "InternVL 3.5",
        5.0,
        video=True,
        small_fast=True,
    ),
    "Granite Vision 3.3 2B (documents/OCR)": ModelSpec(
        "ibm-granite/granite-vision-3.3-2b",
        "Granite Vision 3.3",
        6.5,
        small_fast=True,
    ),
    "Granite Vision 4.1 4B (structured documents)": ModelSpec(
        "ibm-granite/granite-vision-4.1-4b",
        "Granite Vision 4.1",
        9.0,
        small_fast=True,
    ),
    "Custom Hugging Face model": ModelSpec(
        "",
        "Custom",
        8.0,
        trust_remote_code=True,
    ),
}

RECOMMENDED_MODEL_LABELS = (
    "Qwen 3.5 0.8B (fastest current)",
    "Qwen 3.5 4B (recommended)",
    "Qwen 3 VL 2B Instruct",
    "Qwen 3 VL 4B Instruct",
    "Qwen 3 VL 8B Instruct",
    "SmolVLM2 500M Video (low VRAM)",
    "SmolVLM2 2.2B Video",
    "LFM2.5 VL 450M (edge)",
    "InternVL 3.5 1B HF",
    "Granite Vision 4.1 4B (structured documents)",
    "Gemma 3 4B IT (license acceptance required)",
    "Custom Hugging Face model",
)
LEGACY_MODEL_LABELS = tuple(
    label for label in MODEL_CATALOG if label not in RECOMMENDED_MODEL_LABELS
)

MEMORY_MODES = (
    "ComfyUI managed (BF16)",
    "4-bit NF4 (bitsandbytes)",
    "8-bit (bitsandbytes)",
    "CPU",
)
ATTENTION_MODES = ("Auto (SDPA)", "Flash Attention 2", "Eager")


def _progress_text_sender(node_id: str | None) -> Callable[[str], None] | None:
    """Return a best-effort sender for ComfyUI's native progress-text channel."""

    if node_id is None:
        return None
    try:
        from server import PromptServer

        server = PromptServer.instance
    except (ImportError, AttributeError):
        return None

    def send(text: str) -> None:
        try:
            server.send_progress_text(
                text,
                str(node_id),
                server.client_id,
            )
        except Exception:
            # Streaming is a UI enhancement and must never fail inference.
            return

    return send


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
        self.streamer_class = getattr(transformers, "TextIteratorStreamer", None)
        spec = MODEL_CATALOG[model_label]
        repo_id = (
            normalize_hf_model_id(custom_model_id)
            if spec.family == "Custom"
            else spec.repo_id
        )
        self.spec = spec
        self.dtype = torch_dtype("bfloat16")
        if (
            attention_mode == "Flash Attention 2"
            and accelerator_backend(execution_device())
            not in {"nvidia-cuda", "amd-rocm"}
        ):
            raise RuntimeError(
                "Flash Attention 2 requires a supported CUDA or ROCm build. "
                "Select Auto (SDPA) on Apple Metal, Intel XPU, or CPU."
            )
        quantization_device = None
        if memory_mode in {
            "4-bit NF4 (bitsandbytes)",
            "8-bit (bitsandbytes)",
        }:
            # Validate before downloading a multi-gigabyte checkpoint.
            quantization_device = require_quantization_backend(memory_mode)
        try:
            model_path = snapshot_download(
                repo_id,
                f"modern-vlm/{repo_id.replace('/', '--')}",
                ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx"],
            )
        except Exception as exc:
            if spec.gated:
                raise RuntimeError(
                    f"{repo_id} is gated. Accept its Hugging Face license and "
                    "set HF_TOKEN before running this node."
                ) from exc
            raise
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=spec.trust_remote_code,
        )

        attention = {
            # Let each architecture choose its maintained native kernel. Most
            # current PyTorch models select SDPA here, while hybrid edge models
            # can retain their own attention implementation.
            "Auto (SDPA)": None,
            "Flash Attention 2": "flash_attention_2",
            "Eager": "eager",
        }[attention_mode]
        kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "trust_remote_code": spec.trust_remote_code,
        }
        if attention is not None:
            kwargs["attn_implementation"] = attention
        external = memory_mode != "ComfyUI managed (BF16)"
        if memory_mode in {"4-bit NF4 (bitsandbytes)", "8-bit (bitsandbytes)"}:
            assert quantization_device is not None
            needs_offload = spec.estimated_gib >= 40.0
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=memory_mode.startswith("4-bit"),
                load_in_8bit=memory_mode.startswith("8-bit"),
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=needs_offload,
            )
            if needs_offload:
                # Automatic CPU/disk placement is maintained for CUDA, ROCm,
                # and XPU. MPS uses unified memory and CPU already runs in RAM,
                # so both stay on their explicit active device.
                kwargs["device_map"] = external_device_map(
                    allow_auto_offload=True
                )
                kwargs["offload_folder"] = str(model_path / ".offload")
            else:
                # Avoid accidental dispatch to device zero when ComfyUI chose
                # another GPU, Apple Metal, Intel XPU, or CPU.
                kwargs["device_map"] = external_device_map()
            divisor = 4 if memory_mode.startswith("4-bit") else 2
            if quantization_device.type != "cpu":
                reserve_external_vram(
                    int(spec.estimated_gib * 1024**3 / divisor)
                )
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

    def _inputs(
        self,
        messages,
        enable_thinking: bool = False,
        *,
        video_metadata: dict[str, Any] | None = None,
    ):
        """Use the standard multimodal template, with an older-template fallback."""

        template_kwargs = (
            {"enable_thinking": bool(enable_thinking)}
            if self.spec.family in {"Qwen 3.5", "Qwen 3.6"}
            else {}
        )
        processor_kwargs = (
            {
                "video_metadata": [[video_metadata]],
                # ComfyUI already supplied the selected frames as a batch.
                "do_sample_frames": False,
            }
            if video_metadata is not None
            else None
        )
        if processor_kwargs is not None and self.spec.family == "InternVL 3.5":
            # The published InternVL 3.5 video preprocessor uses 384px, which
            # makes a 27x27 patch grid with its 14px vision patches. The
            # model's 0.5 pixel shuffle requires even spatial dimensions.
            image_size = getattr(self.processor.image_processor, "size", None)
            processor_kwargs["size"] = (
                dict(image_size)
                if image_size is not None
                else {"height": 448, "width": 448}
            )
        try:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=processor_kwargs,
                **template_kwargs,
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
                **template_kwargs,
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
        enable_thinking: bool = False,
        stream_callback: Callable[[str], None] | None = None,
        video_selection: VideoFrameSelection | None = None,
    ) -> str:
        primary_images = (
            tensor_batch_to_pil(images) if images is not None else []
        )
        video = (
            tensor_batch_to_pil(video_frames)
            if video_frames is not None
            else None
        )
        if video is None and not primary_images:
            raise ValueError("Connect either image or video_frames.")
        if video is not None and not self.spec.video:
            raise ValueError(
                f"{self.spec.family} does not advertise video support. "
                "Disconnect video_frames or select Qwen/SmolVLM2."
            )
        if video_selection is not None:
            if video is None:
                raise ValueError(
                    "video_selection requires a connected video_frames batch."
                )
            if not isinstance(video_selection, VideoFrameSelection):
                raise TypeError("video_selection must be a VLM Video Selection.")
            if len(video_selection.frames) != len(video):
                raise ValueError(
                    "video_selection frame count must match video_frames."
                )
            source_aspect = video_selection.width / video_selection.height
            analysis_aspect = video[0].width / video[0].height
            if abs(source_aspect - analysis_aspect) > max(
                0.01,
                source_aspect * 0.01,
            ):
                raise ValueError(
                    "video_selection and video_frames must have the same "
                    "aspect ratio."
                )

        results = []
        # A connected video is the primary visual input. Including ComfyUI's
        # required still image as well makes small video models attend to the
        # still and silently ignore the frames.
        runs = [None] if video is not None else primary_images
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
            content = (
                [{"type": "video", "video": video}]
                if video is not None
                else [{"type": "image", "image": image}]
            )
            if video is not None and video_selection is not None:
                timeline = ", ".join(
                    f"{position}=frame {frame.source_frame_index} "
                    f"at {frame.timestamp:.6f}s"
                    for position, frame in enumerate(video_selection.frames)
                )
                effective_prompt = (
                    "The supplied video images are irregular samples from one "
                    f"{video_selection.source_frame_count}-frame video at "
                    f"{video_selection.fps:g} FPS. Supplied-image mapping: "
                    f"{timeline}.\n\n{prompt}"
                )
            elif video is not None:
                effective_prompt = (
                    f"The video frames are sampled at {float(fps):g} FPS.\n\n"
                    f"{prompt}"
                )
            else:
                effective_prompt = prompt
            content.append({"type": "text", "text": effective_prompt})
            messages.append({"role": "user", "content": content})

            metadata = None
            if video is not None:
                if video_selection is not None:
                    metadata = {
                        "total_num_frames": video_selection.source_frame_count,
                        "fps": video_selection.fps,
                        "duration": video_selection.duration,
                        "frames_indices": list(video_selection.indices),
                        "width": video[0].width,
                        "height": video[0].height,
                    }
                else:
                    frame_rate = float(fps)
                    metadata = {
                        "total_num_frames": len(video),
                        "fps": frame_rate,
                        "duration": len(video) / frame_rate,
                        "frames_indices": list(range(len(video))),
                        "width": video[0].width,
                        "height": video[0].height,
                    }
            inputs = self._inputs(
                messages,
                enable_thinking,
                video_metadata=metadata,
            )
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
            streamer_class = self.streamer_class
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if stream_callback is not None and streamer_class is not None:
                streamer = streamer_class(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                generated = []
                errors: list[BaseException] = []

                def generate_in_background() -> None:
                    try:
                        with (
                            torch.inference_mode(),
                            inference_context(device, self.dtype),
                        ):
                            generated.append(
                                model.generate(
                                    **inputs,
                                    **generation,
                                    streamer=streamer,
                                )
                            )
                    except BaseException as exc:
                        errors.append(exc)
                        # Unblock TextIteratorStreamer if generation exits
                        # before it can publish its normal stop signal.
                        streamer.end()

                worker = threading.Thread(
                    target=generate_in_background,
                    name="ComfyUI-VLM-token-stream",
                    daemon=True,
                )
                worker.start()
                chunks = []
                for chunk in streamer:
                    chunks.append(chunk)
                    current = batch_text(
                        [*results, "".join(chunks).strip()]
                    )
                    if current:
                        stream_callback(current)
                worker.join()
                if errors:
                    raise errors[0]

                decoded = "".join(chunks).strip()
                if not decoded and generated:
                    new_tokens = generated[0][:, input_length:]
                    decoded = self.processor.batch_decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                results.append(decoded)
            else:
                with (
                    torch.inference_mode(),
                    inference_context(device, self.dtype),
                ):
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
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image precisely and in detail.",
                    },
                ),
                "model": (
                    list(RECOMMENDED_MODEL_LABELS),
                    {"default": "Qwen 3 VL 2B Instruct"},
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
                "image": ("IMAGE",),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an expert visual analyst.",
                    },
                ),
                "video_frames": ("IMAGE",),
                "video_selection": (VLM_VIDEO_SELECTION,),
                "fps": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 60.0, "step": 0.1},
                ),
                "attention_mode": (
                    ATTENTION_MODES,
                    {"default": "Auto (SDPA)"},
                ),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "unload_after": ("BOOLEAN", {"default": False}),
                "stream_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Stream generated text through ComfyUI's native "
                            "progress-text WebSocket while inference runs."
                        ),
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "VLM Nodes/Modern"

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        # The visible combo is deliberately curated. Accepting every known
        # catalog value here keeps workflows saved before the curation fully
        # executable even when their model now lives under Legacy.
        if model not in MODEL_CATALOG:
            return f"Unsupported Modern VLM model {model!r}."
        return True

    def run(
        self,
        prompt,
        model,
        custom_model_id,
        memory_mode,
        max_new_tokens,
        temperature,
        top_p,
        image=None,
        system_prompt="You are an expert visual analyst.",
        video_frames=None,
        video_selection=None,
        fps=1.0,
        attention_mode="Auto (SDPA)",
        enable_thinking=False,
        unload_after=False,
        stream_output=True,
        unique_id=None,
    ):
        stream_callback = (
            _progress_text_sender(unique_id) if stream_output else None
        )
        if stream_callback is not None:
            stream_callback("Preparing model…")
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
                    images=image,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    video_frames=video_frames,
                    fps=fps,
                    video_selection=video_selection,
                    enable_thinking=enable_thinking,
                    stream_callback=stream_callback,
                ),
            )
        finally:
            self.maybe_clear_model(unload_after)


class LegacyModernVLM(ModernVLM):
    """Compatibility surface for redundant, superseded, and very large tiers."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["model"] = (
            list(LEGACY_MODEL_LABELS),
            {"default": LEGACY_MODEL_LABELS[0]},
        )
        return inputs

    CATEGORY = "VLM Nodes/Legacy/Model Loaders"


NODE_CLASS_MAPPINGS = {
    "ModernVLM": ModernVLM,
    "LegacyModernVLM": LegacyModernVLM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModernVLM": (
        "Modern VLM (Qwen / SmolVLM2 / LFM / InternVL / Granite / Gemma)"
    ),
    "LegacyModernVLM": "[Legacy] Modern VLM Compatibility",
}
