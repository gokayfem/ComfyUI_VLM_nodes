"""Shared runtime helpers for ComfyUI VLM nodes.

The important design rule in this module is that importing a node must never
download a model, install a package, or allocate VRAM.  Models are created on
first execution and, where possible, registered with ComfyUI's own model
manager so they participate in smart VRAM offloading.
"""

from __future__ import annotations

import base64
import gc
import importlib
import inspect
import io
import logging
import os
import platform
import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import folder_paths
import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger("ComfyUI_VLM_nodes")
GGUF_EXTENSIONS = {".gguf"}
LLAMA_FLASH_ATTENTION_CHOICES = ("Auto", "Enabled", "Disabled")
LLAMA_SPLIT_MODE_CHOICES = ("Layer", "Row", "Single GPU")
LLAMA_VISION_HANDLER_CHOICES = (
    "Auto (GGUF chat template)",
    "LLaVA 1.5",
    "LLaVA 1.6",
    "MiniCPM-V 2.6",
    "Moondream2",
    "NanoLLaVA",
    "Qwen2.5-VL",
    "Gemma 4",
    "Llama 3 Vision Alpha",
    "Obsidian",
)


class OptionalDependencyError(RuntimeError):
    """Raised only when a node that needs an optional package is executed."""


def require_module(import_name: str, package_name: str | None = None):
    """Import an optional dependency with an actionable, non-destructive error."""

    try:
        return importlib.import_module(import_name)
    except Exception as exc:
        package = package_name or import_name.split(".", 1)[0]
        raise OptionalDependencyError(
            f"This node requires the optional package '{package}'. "
            f"Install it into ComfyUI's Python environment, then restart ComfyUI. "
            "The node pack intentionally does not run pip or compile packages at startup."
        ) from exc


def register_model_folder() -> Path:
    """Register the shared GGUF/model directory once and return its first path."""

    model_dir = Path(folder_paths.models_dir) / "LLavacheckpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    existing = folder_paths.folder_names_and_paths.get("LLavacheckpoints")
    if existing:
        paths, extensions = existing
        normalized_paths = [str(Path(path)) for path in paths]
        if str(model_dir) not in normalized_paths:
            normalized_paths.append(str(model_dir))
        folder_paths.folder_names_and_paths["LLavacheckpoints"] = (
            normalized_paths,
            set(extensions) | GGUF_EXTENSIONS,
        )
    else:
        folder_paths.folder_names_and_paths["LLavacheckpoints"] = (
            [str(model_dir)],
            GGUF_EXTENSIONS,
        )
    return model_dir


def model_root() -> Path:
    paths = folder_paths.get_folder_paths("LLavacheckpoints")
    if not paths:
        return register_model_folder()
    root = Path(paths[0])
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_cache_dir(name: str) -> Path:
    path = model_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_path(filename: str) -> Path:
    getter = getattr(folder_paths, "get_full_path_or_raise", None)
    if getter is not None:
        return Path(getter("LLavacheckpoints", filename))
    path = folder_paths.get_full_path("LLavacheckpoints", filename)
    if path is None:
        raise FileNotFoundError(f"Model '{filename}' was not found in {model_root()}.")
    return Path(path)


def normalize_hf_model_id(value: str) -> str:
    model_id = (value or "").strip().rstrip("/")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/"):
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix) :]
            break
    if not model_id or "/" not in model_id:
        raise ValueError(
            "Enter a Hugging Face repository as 'owner/model' or a full "
            "https://huggingface.co/owner/model URL."
        )
    return model_id


def snapshot_download(repo_id: str, subdirectory: str, **kwargs: Any) -> Path:
    hub = require_module("huggingface_hub", "huggingface-hub")
    destination = model_cache_dir(subdirectory)
    download_kwargs = {
        "repo_id": repo_id,
        "local_dir": str(destination),
        "local_files_only": False,
    }
    download_kwargs.update(kwargs)
    # local_dir_use_symlinks was removed from newer huggingface-hub versions.
    if "local_dir_use_symlinks" in inspect.signature(hub.snapshot_download).parameters:
        download_kwargs.setdefault("local_dir_use_symlinks", False)
    return Path(hub.snapshot_download(**download_kwargs))


def hf_download(repo_id: str, filename: str, subdirectory: str, **kwargs: Any) -> Path:
    hub = require_module("huggingface_hub", "huggingface-hub")
    destination = model_cache_dir(subdirectory)
    download_kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": str(destination),
    }
    download_kwargs.update(kwargs)
    if "local_dir_use_symlinks" in inspect.signature(hub.hf_hub_download).parameters:
        download_kwargs.setdefault("local_dir_use_symlinks", False)
    return Path(hub.hf_hub_download(**download_kwargs))


def tensor_to_pil(image: torch.Tensor, index: int = 0) -> Image.Image:
    """Convert a Comfy IMAGE tensor to an RGB PIL image without torchvision."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(image).__name__}.")
    value = image.detach()
    if value.ndim == 4:
        if not 0 <= index < value.shape[0]:
            raise IndexError(f"Image batch index {index} is out of range.")
        value = value[index]
    if value.ndim == 2:
        value = value.unsqueeze(-1)
    if value.ndim != 3:
        raise ValueError(
            f"Expected an HWC/BHWC or CHW/BCHW image tensor, got {tuple(value.shape)}."
        )

    # ComfyUI uses HWC. CHW is accepted for compatibility with older callers.
    if value.shape[-1] not in (1, 3, 4) and value.shape[0] in (1, 3, 4):
        value = value.permute(1, 2, 0)
    if value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported image channel shape: {tuple(value.shape)}.")

    value = torch.nan_to_num(
        value.to(device="cpu", dtype=torch.float32), nan=0.0, posinf=1.0, neginf=0.0
    )
    if value.numel() and (value.max() > 1.0 or value.min() < 0.0):
        value = value / 255.0
    array = value.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array, mode="RGB")


def tensor_batch_to_pil(images: torch.Tensor) -> list[Image.Image]:
    if images.ndim == 3:
        return [tensor_to_pil(images)]
    if images.ndim != 4:
        raise ValueError(f"Expected an IMAGE batch, got {tuple(images.shape)}.")
    return [tensor_to_pil(images, index) for index in range(images.shape[0])]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def pil_mask_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def image_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def batch_text(responses: Iterable[str]) -> str:
    items = [str(item).strip() for item in responses]
    if len(items) <= 1:
        return items[0] if items else ""
    return "\n\n".join(
        f"--- Image {index} ---\n{text}" for index, text in enumerate(items, 1)
    )


def llama_chat_content(response: Any) -> str:
    """Extract non-empty text from llama.cpp's OpenAI-compatible response."""

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"llama.cpp returned an unexpected response: {response!r}"
        ) from exc
    if content is None or not str(content).strip():
        raise RuntimeError(
            "llama.cpp returned an empty response. Check that the GGUF chat "
            "template/vision handler matches the model and that max_tokens is positive."
        )
    return str(content).strip()


def execution_device() -> torch.device:
    """Return ComfyUI's selected device, with portable standalone fallbacks."""

    try:
        import comfy.model_management as model_management

        return model_management.get_torch_device()
    except Exception:
        if torch.cuda.is_available():
            # PyTorch intentionally exposes both NVIDIA CUDA and AMD ROCm
            # devices through torch.cuda.
            return torch.device("cuda", torch.cuda.current_device())
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            try:
                if xpu.is_available():
                    return torch.device("xpu", xpu.current_device())
            except Exception:
                pass
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None:
            try:
                if mps.is_available():
                    return torch.device("mps")
            except Exception:
                pass
        return torch.device("cpu")


def accelerator_backend(device: torch.device | None = None) -> str:
    """Return a stable, user-facing name for the active PyTorch backend."""

    device = device or execution_device()
    if device.type == "cuda":
        return (
            "amd-rocm"
            if getattr(getattr(torch, "version", None), "hip", None)
            else "nvidia-cuda"
        )
    return {
        "mps": "apple-metal",
        "xpu": "intel-xpu",
        "cpu": "cpu",
        "privateuseone": "directml-or-privateuse1",
        "npu": "ascend-npu",
        "mlu": "cambricon-mlu",
    }.get(device.type, device.type)


def supports_bfloat16(device: torch.device | None = None) -> bool:
    """Feature-detect BF16 without initializing an unavailable accelerator."""

    device = device or execution_device()
    if device.type == "cuda":
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        try:
            return bool(checker()) if checker is not None else False
        except Exception:
            return False
    if device.type == "xpu":
        checker = getattr(getattr(torch, "xpu", None), "is_bf16_supported", None)
        try:
            return bool(checker()) if checker is not None else False
        except Exception:
            return False
    if device.type == "mps":
        # MPS BF16 requires macOS 14+. Older PyTorch releases may not expose
        # the version probe, in which case FP16 is the safe portable choice.
        checker = getattr(
            getattr(getattr(torch, "backends", None), "mps", None),
            "is_macos_or_newer",
            None,
        )
        try:
            return bool(checker(14, 0)) if checker is not None else False
        except Exception:
            return False
    return False


def torch_dtype(
    name: str | None = None,
    device: torch.device | None = None,
) -> torch.dtype:
    """Choose a dtype that the selected ComfyUI backend can execute safely."""

    device = device or execution_device()
    requested = (name or "auto").lower()
    if requested in {"float32", "fp32"}:
        return torch.float32
    if requested in {"float16", "fp16"}:
        return torch.float16 if device.type in {"cuda", "mps", "xpu"} else torch.float32
    if requested in {"bfloat16", "bf16"}:
        if supports_bfloat16(device):
            return torch.bfloat16
        return torch.float16 if device.type in {"cuda", "mps", "xpu"} else torch.float32
    if supports_bfloat16(device):
        return torch.bfloat16
    return torch.float16 if device.type in {"cuda", "mps", "xpu"} else torch.float32


def _release_tuple(distribution: str) -> tuple[int, ...]:
    try:
        value = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return ()
    parts = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def require_quantization_backend(feature: str) -> torch.device:
    """Validate the maintained bitsandbytes backend for the selected device."""

    device = execution_device()
    backend = accelerator_backend(device)
    supported = {"nvidia-cuda", "amd-rocm", "intel-xpu", "apple-metal", "cpu"}
    if backend not in supported:
        raise RuntimeError(
            f"{feature} is not supported on the active {backend} backend. "
            "Use ComfyUI managed precision or CPU mode."
        )
    if platform.system() == "Darwin" and platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        raise RuntimeError(
            f"{feature} needs bitsandbytes, which has no official Intel-macOS "
            "wheel. Use ComfyUI managed precision, or use an Apple Silicon Mac."
        )
    require_module("bitsandbytes")
    require_module("accelerate")
    if backend != "nvidia-cuda" and _release_tuple("bitsandbytes") < (0, 50):
        raise RuntimeError(
            f"{feature} on {backend} requires bitsandbytes 0.50 or newer. "
            "Upgrade requirements.txt in ComfyUI's Python environment."
        )
    return device


def external_device_map(*, allow_auto_offload: bool = False):
    """Create an Accelerate device map without assuming CUDA device zero."""

    device = execution_device()
    if allow_auto_offload and device.type in {"cuda", "xpu"}:
        return "auto"
    return {"": str(device)}


def runtime_diagnostics() -> dict[str, Any]:
    """Return support information suitable for bug reports and CI logs."""

    device = execution_device()
    packages = {}
    for distribution in (
        "accelerate",
        "bitsandbytes",
        "diffusers",
        "huggingface-hub",
        "llama-cpp-python",
        "qwen-vl-utils",
        "transformers",
    ):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "backend": accelerator_backend(device),
        "bf16": supports_bfloat16(device),
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "torch_hip": getattr(getattr(torch, "version", None), "hip", None),
        "packages": packages,
        "llama_cpp": llama_cpp_diagnostics(),
    }


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return execution_device()


def move_inputs(
    inputs: Mapping[str, Any],
    device: torch.device,
    *,
    floating_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
        elif floating_dtype is not None and value.is_floating_point():
            moved[key] = value.to(device=device, dtype=floating_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


class _ManagedModelAdapter(torch.nn.Module):
    """Give arbitrary HF modules the mutable ``device`` ComfyUI expects.

    Recent Transformers models expose ``device`` as a read-only property.
    ModelPatcher writes that attribute as residency changes, so wrapping the
    original module is necessary for current Qwen/Gemma and harmless for older
    torch modules. Attribute access remains transparent to node predictors.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        super().__init__()
        self.wrapped_model = model
        self.device = device

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_model, name)


class ManagedTorchModel:
    """Register an ordinary torch module with ComfyUI's smart VRAM manager."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        processor: Any = None,
        load_device: torch.device | None = None,
        offload_device: torch.device | None = None,
    ) -> None:
        import comfy.model_management as model_management
        from comfy.model_patcher import ModelPatcher

        self.load_device = load_device or model_management.get_torch_device()
        self.offload_device = offload_device or (
            torch.device("cpu") if self.load_device.type != "cpu" else self.load_device
        )
        self.model = _ManagedModelAdapter(model.eval(), self.offload_device)
        self.processor = processor
        self.patcher = ModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=self.offload_device,
        )
        self._lock = threading.RLock()
        self._closed = False

    def ensure_loaded(self) -> torch.nn.Module:
        if self._closed:
            raise RuntimeError("This model handle has already been closed.")
        with self._lock:
            import comfy.model_management as model_management

            model_management.load_models_gpu([self.patcher])
            return self.model

    def unload(self) -> None:
        if self._closed:
            return
        with self._lock:
            import comfy.model_management as model_management

            model_management.unload_model_and_clones(self.patcher)

    def close(self) -> None:
        if self._closed:
            return
        self.unload()
        self._closed = True
        self.processor = None
        self.model = None
        self.patcher = None
        gc.collect()


class CachedModelNode:
    """Reusable node-instance cache that closes only the model it owns."""

    def __init__(self) -> None:
        self._model_handle = None
        self._model_key = None

    def get_or_create_model(self, key: Any, factory: Callable[[], Any]):
        if self._model_handle is None or self._model_key != key:
            close_handle(self._model_handle)
            self._model_handle = factory()
            self._model_key = key
        return self._model_handle

    def clear_model(self) -> None:
        close_handle(self._model_handle)
        self._model_handle = None
        self._model_key = None

    def maybe_clear_model(self, unload_after: bool) -> None:
        if unload_after:
            self.clear_model()

    def __del__(self):
        try:
            self.clear_model()
        except Exception:
            pass


class ExternalTorchModel:
    """Handle for models whose quantizer/device map cannot use ModelPatcher."""

    def __init__(self, model: Any, *, processor: Any = None) -> None:
        self.model = model
        self.processor = processor
        self._closed = False

    def ensure_loaded(self):
        if self._closed:
            raise RuntimeError("This model handle has already been closed.")
        return self.model

    def unload(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        model = self.model
        self.model = None
        self.processor = None
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        gc.collect()
        try:
            import comfy.model_management as model_management

            model_management.soft_empty_cache()
        except Exception:
            pass


def reserve_external_vram(memory_required: int) -> None:
    """Ask ComfyUI to make room before an external accelerator allocator."""

    device = execution_device()
    if memory_required <= 0 or device.type == "cpu":
        return
    try:
        import comfy.model_management as model_management

        model_management.free_memory(int(memory_required), device)
    except Exception as exc:
        LOGGER.debug("Could not reserve VRAM through ComfyUI: %s", exc)


def default_llama_threads() -> int:
    """A portable generation-thread default that avoids oversubscribing ComfyUI."""

    return max(1, min(16, (os.cpu_count() or 4) // 2))


def llama_runtime_input_types() -> dict[str, tuple[Any, ...]]:
    """Advanced llama.cpp inputs shared by every GGUF loader."""

    return {
        "n_batch": (
            "INT",
            {
                "default": 512,
                "min": 1,
                "max": 8192,
                "step": 1,
                "tooltip": "Logical prompt batch. Lower this if context loading runs out of memory.",
            },
        ),
        "n_ubatch": (
            "INT",
            {
                "default": 512,
                "min": 1,
                "max": 8192,
                "step": 1,
                "tooltip": "Physical prompt micro-batch. Never exceeds n_batch.",
            },
        ),
        "flash_attention": (
            list(LLAMA_FLASH_ATTENTION_CHOICES),
            {
                "default": "Auto",
                "tooltip": "Auto enables llama.cpp flash attention only with accelerator offload and safely retries without it when unsupported.",
            },
        ),
        "use_mmap": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "Memory-map GGUF weights when the installed backend supports it.",
            },
        ),
        "split_mode": (
            list(LLAMA_SPLIT_MODE_CHOICES),
            {
                "default": "Layer",
                "tooltip": "How llama.cpp distributes tensors across multiple accelerators.",
            },
        ),
        "main_gpu": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 31,
                "step": 1,
            },
        ),
        "tensor_split": (
            "STRING",
            {
                "default": "",
                "tooltip": "Optional comma-separated accelerator proportions, for example 0.6,0.4.",
            },
        ),
    }


def llama_runtime_options(
    *,
    n_batch: int = 512,
    n_ubatch: int = 512,
    flash_attention: str = "Auto",
    use_mmap: bool = True,
    split_mode: str = "Layer",
    main_gpu: int = 0,
    tensor_split: str = "",
) -> dict[str, Any]:
    """Normalize node inputs into keyword arguments accepted by LlamaHandle."""

    return {
        "n_batch": int(n_batch),
        "n_ubatch": int(n_ubatch),
        "flash_attention": str(flash_attention),
        "use_mmap": bool(use_mmap),
        "split_mode": str(split_mode),
        "main_gpu": int(main_gpu),
        "tensor_split": str(tensor_split),
    }


def _call_llama_probe(module: Any, name: str) -> bool | None:
    for source in (module, getattr(module, "llama_cpp", None)):
        probe = getattr(source, name, None)
        if callable(probe):
            try:
                return bool(probe())
            except Exception:
                return None
    return None


def _llama_system_info(module: Any) -> str | None:
    for source in (module, getattr(module, "llama_cpp", None)):
        probe = getattr(source, "llama_print_system_info", None)
        if callable(probe):
            try:
                value = probe()
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                return str(value).strip() or None
            except Exception:
                return None
    return None


def _llama_backend_labels(system_info: str | None) -> list[str]:
    if not system_info:
        return []
    value = system_info.upper()
    candidates = (
        ("CUDA", "cuda"),
        ("ROCM", "rocm"),
        ("HIP", "hip"),
        ("METAL", "metal"),
        ("VULKAN", "vulkan"),
        ("SYCL", "sycl"),
        ("CANN", "cann"),
        ("OPENCL", "opencl"),
        ("BLAS", "blas"),
    )
    return [label for marker, label in candidates if marker in value]


def llama_cpp_diagnostics(module: Any | None = None) -> dict[str, Any]:
    """Describe the installed llama.cpp build without allocating model memory."""

    try:
        version = metadata.version("llama-cpp-python")
    except metadata.PackageNotFoundError:
        version = None
    if module is None:
        try:
            module = importlib.import_module("llama_cpp")
        except Exception as exc:
            return {
                "installed": version is not None,
                "version": version,
                "import_error": f"{type(exc).__name__}: {exc}",
                "gpu_offload": None,
                "mmap": None,
                "mlock": None,
                "backends": [],
                "system_info": None,
            }
    system_info = _llama_system_info(module)
    return {
        "installed": True,
        "version": getattr(module, "__version__", None) or version,
        "import_error": None,
        "gpu_offload": _call_llama_probe(module, "llama_supports_gpu_offload"),
        "mmap": _call_llama_probe(module, "llama_supports_mmap"),
        "mlock": _call_llama_probe(module, "llama_supports_mlock"),
        "backends": _llama_backend_labels(system_info),
        "system_info": system_info,
    }


def _parse_tensor_split(value: str | Iterable[float] | None) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                "tensor_split must be comma-separated numbers, for example 0.6,0.4."
            ) from exc
    else:
        values = [float(part) for part in value]
    if not values or any(not np.isfinite(part) or part < 0 for part in values):
        raise ValueError("tensor_split values must be finite and non-negative.")
    if not any(values):
        raise ValueError("tensor_split must give at least one accelerator a weight.")
    return values


def _resolve_split_mode(module: Any, value: str) -> Any:
    constants = {
        "Layer": "LLAMA_SPLIT_MODE_LAYER",
        "Row": "LLAMA_SPLIT_MODE_ROW",
        "Single GPU": "LLAMA_SPLIT_MODE_NONE",
    }
    if value not in constants:
        raise ValueError(
            f"Unknown llama.cpp split mode {value!r}; choose one of {LLAMA_SPLIT_MODE_CHOICES}."
        )
    return getattr(module, constants[value], None)


@dataclass(frozen=True)
class LlavaClipConfig:
    model_path: Path
    # Preserve the prompt format used by workflows saved before handler
    # selection existed. New loader widgets explicitly pass the Auto choice.
    handler: str = "LLaVA 1.5"

    def create(self, *, use_gpu: bool = True):
        module = require_module("llama_cpp.llama_chat_format", "llama-cpp-python")
        handlers = {
            "LLaVA 1.5": "Llava15ChatHandler",
            "LLaVA 1.6": "Llava16ChatHandler",
            "MiniCPM-V 2.6": "MiniCPMv26ChatHandler",
            "Moondream2": "MoondreamChatHandler",
            "NanoLLaVA": "NanoLlavaChatHandler",
            "Qwen2.5-VL": "Qwen25VLChatHandler",
            "Gemma 4": "Gemma4ChatHandler",
            "Llama 3 Vision Alpha": "Llama3VisionAlphaChatHandler",
            "Obsidian": "ObsidianChatHandler",
        }
        if self.handler == "Auto (GGUF chat template)":
            handler_class = getattr(module, "MTMDChatHandler", None)
            if handler_class is not None:
                return handler_class(
                    clip_model_path=str(self.model_path),
                    verbose=False,
                    use_gpu=use_gpu,
                )
            handler_name = "Llava15ChatHandler"
        else:
            try:
                handler_name = handlers[self.handler]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown vision handler {self.handler!r}; choose one of "
                    f"{LLAMA_VISION_HANDLER_CHOICES}."
                ) from exc
        handler_class = getattr(module, handler_name, None)
        if handler_class is None:
            raise RuntimeError(
                f"The installed llama-cpp-python does not provide {handler_name}. "
                "Install a current wheel for CUDA, ROCm/HIP, Metal, Vulkan, SYCL, or CPU."
            )
        return handler_class(clip_model_path=str(self.model_path), verbose=False)


class LlamaHandle:
    """Lazy llama.cpp handle that owns and closes its accelerator allocations."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        chat_format: str | None = None,
        chat_handler_factory: Callable[..., Any] | None = None,
        seed: int = 42,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads_batch: int | None = None,
        flash_attention: str = "Auto",
        use_mmap: bool = True,
        split_mode: str = "Layer",
        main_gpu: int = 0,
        tensor_split: str | Iterable[float] | None = None,
        projector_path: Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_threads = int(n_threads)
        self.chat_format = chat_format
        self.chat_handler_factory = chat_handler_factory
        self.seed = int(seed)
        self.n_batch = max(1, int(n_batch))
        self.n_ubatch = max(1, int(n_ubatch))
        self.n_threads_batch = (
            None if n_threads_batch is None else max(1, int(n_threads_batch))
        )
        if flash_attention not in LLAMA_FLASH_ATTENTION_CHOICES:
            raise ValueError(
                f"flash_attention must be one of {LLAMA_FLASH_ATTENTION_CHOICES}."
            )
        self.flash_attention = flash_attention
        self.use_mmap = bool(use_mmap)
        self.split_mode = split_mode
        self.main_gpu = max(0, int(main_gpu))
        self.tensor_split = _parse_tensor_split(tensor_split)
        self.projector_path = None if projector_path is None else Path(projector_path)
        self._llm = None
        self._chat_handler = None
        self._lock = threading.RLock()

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return (
            str(self.model_path),
            self.n_ctx,
            self.n_gpu_layers,
            self.n_threads,
            self.chat_format,
            self.seed,
            self.n_batch,
            self.n_ubatch,
            self.n_threads_batch,
            self.flash_attention,
            self.use_mmap,
            self.split_mode,
            self.main_gpu,
            tuple(self.tensor_split or ()),
            str(self.projector_path) if self.projector_path else None,
        )

    @staticmethod
    def _filter_constructor_kwargs(llama_class: Any, requested: dict[str, Any]):
        signature = inspect.signature(llama_class.__init__)
        named_parameters = {
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        return {
            key: value
            for key, value in requested.items()
            if value is not None
            and (
                key in named_parameters
                or (accepts_kwargs and not named_parameters)
            )
        }

    def _create_chat_handler(self, *, use_gpu: bool):
        if self.chat_handler_factory is None:
            return None
        try:
            signature = inspect.signature(self.chat_handler_factory)
            if "use_gpu" in signature.parameters:
                return self.chat_handler_factory(use_gpu=use_gpu)
        except (TypeError, ValueError):
            pass
        return self.chat_handler_factory()

    def ensure_loaded(self):
        if self._llm is not None:
            return self._llm
        with self._lock:
            if self._llm is not None:
                return self._llm
            if not self.model_path.is_file():
                raise FileNotFoundError(f"GGUF model not found: {self.model_path}")

            llama_cpp = require_module("llama_cpp", "llama-cpp-python")
            capabilities = llama_cpp_diagnostics(llama_cpp)
            supports_gpu = capabilities["gpu_offload"]
            effective_gpu_layers = self.n_gpu_layers
            if effective_gpu_layers != 0 and supports_gpu is False:
                LOGGER.warning(
                    "The installed llama-cpp-python build has no accelerator "
                    "offload; falling back from n_gpu_layers=%s to CPU.",
                    effective_gpu_layers,
                )
                effective_gpu_layers = 0

            # llama.cpp owns its allocator, so ask ComfyUI to make room without
            # globally clearing unrelated accelerator caches.
            if effective_gpu_layers != 0:
                memory_required = self.model_path.stat().st_size
                if self.projector_path is not None and self.projector_path.is_file():
                    memory_required += self.projector_path.stat().st_size
                reserve_external_vram(memory_required)

            use_gpu = effective_gpu_layers != 0 and supports_gpu is not False
            self._chat_handler = self._create_chat_handler(use_gpu=use_gpu)

            effective_batch = min(
                self.n_batch, self.n_ctx if self.n_ctx > 0 else self.n_batch
            )
            effective_ubatch = min(self.n_ubatch, effective_batch)
            flash_attention = self.flash_attention == "Enabled" or (
                self.flash_attention == "Auto" and use_gpu
            )

            requested = {
                "model_path": str(self.model_path),
                "chat_handler": self._chat_handler,
                "chat_format": self.chat_format,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": effective_gpu_layers,
                "n_threads": self.n_threads,
                # None lets llama.cpp choose its optimized prompt-processing
                # thread count independently from token-generation threads.
                "n_threads_batch": self.n_threads_batch,
                "n_batch": effective_batch,
                "n_ubatch": effective_ubatch,
                "split_mode": _resolve_split_mode(llama_cpp, self.split_mode),
                "main_gpu": self.main_gpu,
                "tensor_split": self.tensor_split,
                "offload_kqv": use_gpu,
                "op_offload": use_gpu,
                "flash_attn": flash_attention,
                "use_mmap": self.use_mmap and capabilities["mmap"] is not False,
                "use_mlock": False,
                "embedding": False,
                "verbose": False,
                "seed": self.seed,
            }
            kwargs = self._filter_constructor_kwargs(llama_cpp.Llama, requested)
            try:
                self._llm = llama_cpp.Llama(**kwargs)
            except Exception as exc:
                # Some backend/model pairs reject flash attention at context
                # creation. Auto is an optimization, never a compatibility
                # requirement, so retry once with the official safe default.
                if (
                    self.flash_attention == "Auto"
                    and kwargs.get("flash_attn")
                    and re.search(r"flash|attn|attention", str(exc), re.IGNORECASE)
                ):
                    LOGGER.warning(
                        "llama.cpp flash attention was unavailable; retrying "
                        "with the portable attention path: %s",
                        exc,
                    )
                    kwargs["flash_attn"] = False
                    gc.collect()
                    self._llm = llama_cpp.Llama(**kwargs)
                else:
                    raise
            return self._llm

    def close(self) -> None:
        with self._lock:
            llm, handler = self._llm, self._chat_handler
            self._llm = None
            self._chat_handler = None
            if llm is not None:
                close = getattr(llm, "close", None)
                if callable(close):
                    close()
            if handler is not None:
                close = getattr(handler, "close", None)
                if callable(close):
                    close()
            gc.collect()
            try:
                import comfy.model_management as model_management

                model_management.soft_empty_cache()
            except Exception:
                pass

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.ensure_loaded(), name)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def unwrap_llm(model: Any):
    ensure_loaded = getattr(model, "ensure_loaded", None)
    return ensure_loaded() if callable(ensure_loaded) else model


def close_handle(handle: Any) -> None:
    if handle is None:
        return
    close = getattr(handle, "close", None)
    if callable(close):
        close()


def inference_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "xpu"} and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device.type, dtype=dtype)
    return nullcontext()
