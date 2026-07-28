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
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
from PIL import Image

import folder_paths

LOGGER = logging.getLogger("ComfyUI_VLM_nodes")
GGUF_EXTENSIONS = {".gguf"}


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
        raise FileNotFoundError(
            f"Model '{filename}' was not found in {model_root()}."
        )
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
    if "local_dir_use_symlinks" in inspect.signature(
        hub.snapshot_download
    ).parameters:
        download_kwargs.setdefault("local_dir_use_symlinks", False)
    return Path(hub.snapshot_download(**download_kwargs))


def hf_download(
    repo_id: str, filename: str, subdirectory: str, **kwargs: Any
) -> Path:
    hub = require_module("huggingface_hub", "huggingface-hub")
    destination = model_cache_dir(subdirectory)
    download_kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": str(destination),
    }
    download_kwargs.update(kwargs)
    if "local_dir_use_symlinks" in inspect.signature(
        hub.hf_hub_download
    ).parameters:
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
    array = (
        value.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    )
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


def torch_dtype(name: str | None = None) -> torch.dtype:
    requested = (name or "auto").lower()
    if requested in {"float32", "fp32"}:
        return torch.float32
    if requested in {"float16", "fp16"}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if requested in {"bfloat16", "bf16"}:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def execution_device() -> torch.device:
    try:
        import comfy.model_management as model_management

        return model_management.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            torch.device("cpu")
            if self.load_device.type != "cpu"
            else self.load_device
        )
        self.model = _ManagedModelAdapter(
            model.eval(), self.offload_device
        )
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
    """Ask ComfyUI to make room before an external CUDA allocator is used."""

    if memory_required <= 0 or not torch.cuda.is_available():
        return
    try:
        import comfy.model_management as model_management

        model_management.free_memory(
            int(memory_required), model_management.get_torch_device()
        )
    except Exception as exc:
        LOGGER.debug("Could not reserve VRAM through ComfyUI: %s", exc)


@dataclass(frozen=True)
class LlavaClipConfig:
    model_path: Path

    def create(self):
        module = require_module("llama_cpp.llama_chat_format", "llama-cpp-python")
        return module.Llava15ChatHandler(
            clip_model_path=str(self.model_path), verbose=False
        )


class LlamaHandle:
    """Lazy llama.cpp handle that owns and closes its exact GPU allocations."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        chat_format: str | None = None,
        chat_handler_factory: Callable[[], Any] | None = None,
        seed: int = 42,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_threads = int(n_threads)
        self.chat_format = chat_format
        self.chat_handler_factory = chat_handler_factory
        self.seed = int(seed)
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
        )

    def ensure_loaded(self):
        if self._llm is not None:
            return self._llm
        with self._lock:
            if self._llm is not None:
                return self._llm
            if not self.model_path.is_file():
                raise FileNotFoundError(f"GGUF model not found: {self.model_path}")

            # llama.cpp owns its CUDA allocator, so reserve enough room through
            # ComfyUI first instead of blindly emptying the global CUDA cache.
            if self.n_gpu_layers != 0:
                reserve_external_vram(self.model_path.stat().st_size)

            llama_cpp = require_module("llama_cpp", "llama-cpp-python")
            if self.chat_handler_factory is not None:
                self._chat_handler = self.chat_handler_factory()

            requested = {
                "model_path": str(self.model_path),
                "chat_handler": self._chat_handler,
                "chat_format": self.chat_format,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "n_threads": self.n_threads,
                "n_batch": min(1024, self.n_ctx),
                "offload_kqv": True,
                "flash_attn": True,
                "use_mlock": False,
                "embedding": False,
                "verbose": False,
                "seed": self.seed,
            }
            signature = inspect.signature(llama_cpp.Llama.__init__)
            kwargs = {
                key: value
                for key, value in requested.items()
                if key in signature.parameters and value is not None
            }
            self._llm = llama_cpp.Llama(**kwargs)
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
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()
