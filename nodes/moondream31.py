"""Native Moondream 3/3.1 skills with isolated Photon inference.

Moondream returns normalized boxes and points. The Moondream 3 Preview segment
skill additionally returns a native SVG path. This module converts those
results into the canonical spatial types shared by this pack and into ordinary
ComfyUI IMAGE/MASK outputs. The final 3.1 model officially exposes query,
caption, detect, and point; it is not falsely advertised as a segment model.

The official ``moondream`` package intentionally runs in a dedicated virtual
environment because its Pillow constraint conflicts with current ComfyUI
releases.  The exact worker process is owned and terminated by the model
handle, so ``unload_after`` reliably releases its GPU allocation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw

from .grounding import (
    core_bounding_box_frames,
    core_bounding_boxes,
    detection_box_masks,
)
from .runtime import (
    CachedModelNode,
    model_cache_dir,
    reserve_external_vram,
    tensor_batch_to_pil,
)
from .vision_types import (
    VLM_DETECTIONS,
    VLM_POINTS,
    Detection,
    DetectionSequence,
    FrameDetections,
    PointSequence,
    VisionPoint,
)
from .vision_utils import (
    composite_with_mask,
    masks_to_images,
    render_detections,
    sequence_masks,
)

LOGGER = logging.getLogger("ComfyUI_VLM_nodes")

MOONDREAM31_MODEL = "MOONDREAM31_MODEL"
MODEL_ID = "moondream3.1-9B-A2B"
MODEL_SOURCE = "moondream/moondream3.1-9B-A2B"
PREVIEW_MODEL_ID = "moondream3-preview"
PREVIEW_MODEL_SOURCE = "moondream/moondream3-preview"
MODEL_LICENSE_URL = "https://moondream.ai/licenses/model/1.0"
RUNTIME_VERSION = "1.3.0"
MAX_SVG_PATH_CHARS = 1_000_000
MAX_SVG_POINTS = 250_000

DEVICE_CHOICES = ("Auto", "NVIDIA CUDA", "Apple Silicon MPS")
KV_CACHE_PROFILES = {
    "Low VRAM (4K pages)": 4096,
    "Balanced (8K pages)": 8192,
    "Maximum throughput (16K pages)": 16384,
    "Photon automatic": 0,
}


def _progress_text_sender(node_id: str | None) -> Callable[[str], None] | None:
    if node_id is None:
        return None
    try:
        from server import PromptServer

        server = PromptServer.instance
    except (ImportError, AttributeError):
        return None

    def send(text: str) -> None:
        try:
            server.send_progress_text(text, str(node_id), server.client_id)
        except Exception as exc:  # noqa: BLE001 - UI enhancement only.
            LOGGER.debug("Could not send Moondream progress text: %s", exc)

    return send


def _device_name(value: str) -> str:
    if value == "NVIDIA CUDA":
        return "cuda"
    if value == "Apple Silicon MPS":
        return "mps"
    if value == "Auto":
        # Photon performs the authoritative availability check in its isolated
        # environment. Host OS is enough to choose the only supported options.
        return "mps" if os.sys.platform == "darwin" else "cuda"
    raise ValueError(f"Unsupported Moondream device {value!r}.")


def _runtime_root() -> Path:
    return model_cache_dir("moondream31-runtime")


def _runtime_python(root: Path) -> Path:
    override = os.environ.get("MOONDREAM_PYTHON", "").strip()
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(
        (
            root / ".venv" / "bin" / "python",
            root / ".venv" / "Scripts" / "python.exe",
            root / "venv" / "bin" / "python",
            root / "venv" / "Scripts" / "python.exe",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            # Do not resolve a POSIX venv's python symlink: executing the
            # resolved base interpreter bypasses the venv's site-packages.
            return candidate.absolute()
    raise RuntimeError(
        "Moondream 3.1 needs its isolated runtime. Create it on the same drive "
        "as ComfyUI by following README.md → Moondream 3.1 isolated runtime. "
        "You can instead set the server-side MOONDREAM_PYTHON environment "
        "variable to an existing compatible runtime."
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    return buffer.getvalue()


def _worker_error(message: dict[str, Any]) -> RuntimeError:
    detail = str(message.get("error") or "Unknown Moondream worker error")
    trace = str(message.get("traceback") or "").strip()
    if trace:
        detail = f"{detail}\n\nIsolated worker traceback:\n{trace}"
    return RuntimeError(detail)


def _base_model_name(value: str) -> str:
    return str(value).split("/", 1)[0]


def _model_source(value: str) -> str:
    return (
        PREVIEW_MODEL_SOURCE
        if _base_model_name(value) == PREVIEW_MODEL_ID
        else MODEL_SOURCE
    )


def _safe_log_tail(path: Path | None, limit: int = 12_000) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            text = handle.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""
    # Worker logs should not contain credentials, but redact common secret
    # assignments defensively before a tail is surfaced in ComfyUI.
    return re.sub(
        r"(?i)\b(api[_ -]?key|authorization|token)\b(\s*[:=]\s*)\S+",
        r"\1\2<redacted>",
        text,
    ).strip()


def _worker_environment(
    cache: Path,
    auth_key: bytes,
    model_name: str,
) -> dict[str, str]:
    """Build a minimal inherited environment without unrelated credentials."""

    blocked_names = {
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        # ComfyUI may select cudaMallocAsync for its own allocator. Photon
        # captures CUDA graphs in a separate process, where inheriting that
        # override can abort during warmup on an uncaptured/captured free.
        # Let the sidecar use PyTorch's native allocator instead.
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
    }
    secret_name = re.compile(
        r"(?:API[_-]?KEY|AUTHORIZATION|CREDENTIAL|PASSWORD|SECRET|TOKEN)",
        re.IGNORECASE,
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() not in blocked_names and not secret_name.search(name)
    }
    # These are the only upstream credentials the worker can legitimately use.
    # They remain server-side environment values and never become node inputs.
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        environment["HF_TOKEN"] = hf_token
    moondream_key = os.environ.get("MOONDREAM_API_KEY", "").strip()
    if "/" in model_name and moondream_key:
        environment["MOONDREAM_API_KEY"] = moondream_key
    environment["HF_HOME"] = str(cache / "huggingface")
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["DO_NOT_TRACK"] = "1"
    environment["MOONDREAM_WORKER_AUTH"] = auth_key.hex()
    return environment


@dataclass(frozen=True)
class Moondream31Config:
    model: str
    device: str
    max_batch_size: int
    kv_cache_pages: int


class Moondream31Model:
    """Restartable owner for one exact isolated Photon process."""

    def __init__(self, config: Moondream31Config):
        self.config = config
        self.root = _runtime_root()
        self.python = _runtime_python(self.root)
        self.cache = self.root / "cache"
        self.logs = self.root / "logs"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._process: subprocess.Popen | None = None
        self._listener = None
        self._log_handle = None
        self._request_id = 0
        self._lock = threading.RLock()
        self._log_path: Path | None = None
        self.runtime_info: dict[str, Any] = {}

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _close_transport(self) -> None:
        connection, listener = self._connection, self._listener
        self._connection = None
        self._listener = None
        for value in (connection, listener):
            if value is not None:
                try:
                    value.close()
                except Exception as exc:  # noqa: BLE001 - heterogeneous handles.
                    LOGGER.debug("Could not close Moondream transport: %s", exc)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception as exc:  # noqa: BLE001 - interpreter shutdown.
                LOGGER.debug("Could not close Moondream log handle: %s", exc)
            self._log_handle = None

    def _unexpected_exit(self, operation: str) -> RuntimeError:
        process = self._process
        code = None
        if process is not None:
            code = process.poll()
            if code is None:
                try:
                    code = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        log_path = self._log_path
        self.close()
        tail = _safe_log_tail(log_path)
        detail = (
            f"Moondream Photon worker stopped during {operation} (code {code})."
        )
        if "cudaLibraryLoadData" in tail:
            detail += (
                " Photon's CUDA 12 kernels require libcudart 12.9 or newer; "
                "CUDA runtime 12.6 does not export cudaLibraryLoadData. Re-run "
                "the README isolated-runtime install command so "
                "requirements-moondream31.txt upgrades only this sidecar to "
                "nvidia-cuda-runtime-cu12 12.9.79."
            )
        if tail:
            detail += f"\n\nSanitized worker log tail:\n{tail}"
        elif log_path is not None:
            detail += f" See {log_path}."
        return RuntimeError(detail)

    def ensure_started(
        self,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        with self._lock:
            if self.is_running and self._connection is not None:
                return
            self.close()
            if progress is not None:
                progress("Preparing isolated Moondream 3.1 runtime…")

            # The model repository is about 10.5 GB. This is intentionally an
            # estimate rather than a promise; Photon sizes its KV cache from
            # the selected profile and performs the final device check.
            reserve_external_vram(18 * 1024**3)
            host = "127.0.0.1"
            port = _free_loopback_port()
            auth_key = secrets.token_bytes(32)
            listener = Listener((host, port), authkey=auth_key)
            try:
                listener._listener._socket.settimeout(30)  # type: ignore[attr-defined]
            except (AttributeError, OSError):
                # Timeout is a guard for CPython's current Listener internals;
                # the worker connects before importing any heavy dependency.
                pass
            self._listener = listener

            worker = Path(__file__).with_name("moondream31_worker.py").resolve()
            log_path = self.logs / (
                f"worker-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
            )
            self._log_path = log_path
            self._log_handle = log_path.open("ab", buffering=0)
            command = [
                str(self.python),
                str(worker),
                "--host",
                host,
                "--port",
                str(port),
                "--model",
                self.config.model,
                "--device",
                self.config.device,
                "--max-batch-size",
                str(self.config.max_batch_size),
                "--kv-cache-pages",
                str(self.config.kv_cache_pages),
            ]
            # Keep unrelated provider keys out of the sidecar and keep the
            # IPC authentication secret out of process listings.
            environment = _worker_environment(
                self.cache,
                auth_key,
                self.config.model,
            )
            startup_info = None
            creation_flags = 0
            if os.name == "nt":
                startup_info = subprocess.STARTUPINFO()
                startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creation_flags = subprocess.CREATE_NO_WINDOW
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=str(self.root),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    startupinfo=startup_info,
                    creationflags=creation_flags,
                    close_fds=os.name != "nt",
                )
                self._connection = listener.accept()
            except Exception:
                self.close()
                raise

            deadline = time.monotonic() + 3600
            loading_announced = False
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise self._unexpected_exit("startup")
                if not self._connection.poll(0.25):
                    if progress is not None and not loading_announced:
                        progress(
                            "Loading Moondream 3.1 weights on the GPU "
                            "(the first run downloads them to D)…"
                        )
                        loading_announced = True
                    continue
                try:
                    message = self._connection.recv()
                except (EOFError, OSError) as exc:
                    raise self._unexpected_exit("startup") from exc
                status = message.get("status")
                if message.get("type") == "fatal":
                    error = _worker_error(message)
                    self.close()
                    raise error
                if message.get("type") == "status" and status == "ready":
                    self.runtime_info = dict(message)
                    self.runtime_info.update(
                        {
                            "model": self.config.model,
                            "device": self.config.device,
                            "max_batch_size": self.config.max_batch_size,
                            "kv_cache_pages": self.config.kv_cache_pages,
                            "python": str(self.python),
                            "cache": str(self.cache),
                            "log": str(log_path),
                        }
                    )
                    if progress is not None:
                        progress("Moondream 3.1 ready.")
                    return
            self.close()
            raise TimeoutError(
                f"Moondream 3.1 did not finish loading within one hour. See {log_path}."
            )

    def request(
        self,
        operation: str,
        *,
        progress: Callable[[str], None] | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self.ensure_started(progress)
            self._request_id += 1
            request_id = self._request_id
            self._connection.send({"id": request_id, "operation": operation, **payload})
            streamed = ""
            while True:
                if self._process is None or self._process.poll() is not None:
                    raise self._unexpected_exit(operation)
                if not self._connection.poll(0.25):
                    continue
                try:
                    message = self._connection.recv()
                except (EOFError, OSError) as exc:
                    raise self._unexpected_exit(operation) from exc
                if message.get("id") != request_id:
                    continue
                message_type = message.get("type")
                if message_type == "chunk":
                    streamed += str(message.get("text") or "")
                    if progress is not None:
                        progress(streamed)
                    continue
                if message_type == "error":
                    raise _worker_error(message)
                if message_type == "result":
                    return dict(message.get("result") or {})

    def close(self) -> None:
        with self._lock:
            process = self._process
            connection = self._connection
            if (
                process is not None
                and process.poll() is None
                and connection is not None
            ):
                try:
                    self._request_id += 1
                    connection.send({"id": self._request_id, "operation": "shutdown"})
                    if connection.poll(5):
                        connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            self._process = None
            self._close_transport()

    unload = close

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110 - destructors must not raise.
            pass


def _normalized_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and normalized to [0, 1].")
    return number


def _normalized_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        raise TypeError("Moondream bbox must be an object.")
    x1 = _normalized_number(value.get("x_min"), "bbox x_min")
    y1 = _normalized_number(value.get("y_min"), "bbox y_min")
    x2 = _normalized_number(value.get("x_max"), "bbox x_max")
    y2 = _normalized_number(value.get("y_max"), "bbox y_max")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Moondream bbox must have positive width and height.")
    return x1, y1, x2, y2


def _pixel_bbox(
    value: Any,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _normalized_bbox(value)
    return x1 * width, y1 * height, x2 * width, y2 * height


def _curve_contours(
    svg_path: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    precision_px: float,
) -> list[list[tuple[float, float]]]:
    if not isinstance(svg_path, str) or not svg_path.strip():
        raise ValueError("Moondream returned an empty SVG path.")
    if len(svg_path) > MAX_SVG_PATH_CHARS:
        raise ValueError("Moondream SVG path is larger than the safety limit.")
    try:
        from svgelements import Close, Move
        from svgelements import Path as SVGPath
    except Exception as exc:
        raise RuntimeError(
            "SVG mask conversion needs `svgelements>=1.9.6,<2`. "
            "Reinstall this node pack's base requirements."
        ) from exc

    try:
        path = SVGPath(svg_path)
    except Exception as exc:
        raise ValueError(f"Moondream returned an invalid SVG path: {exc}") from exc
    if not path:
        raise ValueError("Moondream returned an empty SVG path.")

    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    contours: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    point_count = 0

    def transform(point: Any) -> tuple[float, float]:
        nonlocal point_count
        px = float(point.real)
        py = float(point.imag)
        if not math.isfinite(px) or not math.isfinite(py):
            raise ValueError("SVG path contains non-finite coordinates.")
        # Native Moondream path coordinates are normalized within its bbox.
        output = (
            min(float(width), max(0.0, (x1 + px * box_width) * width)),
            min(float(height), max(0.0, (y1 + py * box_height) * height)),
        )
        point_count += 1
        if point_count > MAX_SVG_POINTS:
            raise ValueError("SVG path exceeds the flattened point safety limit.")
        return output

    def finish() -> None:
        nonlocal current
        clean = []
        for point in current:
            if not clean or point != clean[-1]:
                clean.append(point)
        if len(clean) >= 3:
            if clean[0] == clean[-1]:
                clean.pop()
            if len(clean) >= 3:
                contours.append(clean)
        current = []

    for segment in path:
        if isinstance(segment, Move):
            finish()
            if segment.end is not None:
                current = [transform(segment.end)]
            continue
        if not current and segment.start is not None:
            current = [transform(segment.start)]
        if isinstance(segment, Close):
            finish()
            continue
        try:
            normalized_length = float(segment.length(error=1e-5))
        except TypeError:
            normalized_length = float(segment.length())
        if not math.isfinite(normalized_length):
            raise ValueError("SVG path contains a non-finite curve length.")
        pixel_length = normalized_length * max(
            box_width * width,
            box_height * height,
        )
        samples = min(
            2048,
            max(1, math.ceil(pixel_length / float(precision_px))),
        )
        for index in range(1, samples + 1):
            current.append(transform(segment.point(index / samples)))
    finish()
    if not contours:
        raise ValueError("SVG path did not contain a fillable closed shape.")
    return contours


def svg_path_to_mask(
    svg_path: str,
    bbox: Any,
    width: int,
    height: int,
    *,
    supersample: int = 4,
    precision_px: float = 1.0,
) -> tuple[
    torch.Tensor, tuple[tuple[float, float], ...], list[list[tuple[float, float]]]
]:
    """Rasterize a native Moondream SVG path into an antialiased Comfy mask.

    SVG subpaths use an even-odd fill, preserving ordinary holes and disjoint
    islands without requiring a system SVG library.
    """

    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer.")
    if not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer.")
    if not isinstance(supersample, int) or not 1 <= supersample <= 8:
        raise ValueError("supersample must be between 1 and 8.")
    if not math.isfinite(precision_px) or precision_px <= 0:
        raise ValueError("precision_px must be finite and positive.")
    normalized_box = _normalized_bbox(bbox)
    contours = _curve_contours(
        svg_path,
        normalized_box,
        width,
        height,
        precision_px,
    )

    canvas_size = (width * supersample, height * supersample)
    combined = Image.new("1", canvas_size, 0)
    for contour in contours:
        layer = Image.new("1", canvas_size, 0)
        points = [
            (
                round(x * supersample),
                round(y * supersample),
            )
            for x, y in contour
        ]
        ImageDraw.Draw(layer).polygon(points, fill=1)
        combined = ImageChops.logical_xor(combined, layer)
    high_resolution = combined.convert("L").point(lambda value: 255 if value else 0)
    if supersample > 1:
        high_resolution = high_resolution.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )
    array = np.asarray(high_resolution, dtype=np.float32) / 255.0
    mask = torch.from_numpy(array.copy()).clamp(0, 1)
    primary = max(contours, key=lambda value: abs(_polygon_area(value)))
    return mask, tuple(primary), contours


def _polygon_area(points: Iterable[tuple[float, float]]) -> float:
    values = list(points)
    return 0.5 * sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(values, values[1:] + values[:1])
    )


def _sampled_images(
    image: torch.Tensor,
    frame_stride: int,
) -> tuple[list[Image.Image], list[int], int, int]:
    if not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer.")
    images = tensor_batch_to_pil(image)
    if not images:
        raise ValueError("At least one image/frame is required.")
    dimensions = {item.size for item in images}
    if len(dimensions) != 1:
        raise ValueError("Every image in a batch must have the same dimensions.")
    indices = list(range(0, len(images), frame_stride))
    width, height = images[0].size
    return [images[index] for index in indices], indices, width, height


def _performance(
    *,
    result: dict[str, Any],
    total_elapsed: float,
    source_frames: int,
    processed_frames: int,
    source_fps: float,
    frame_stride: int,
) -> dict[str, Any]:
    worker_elapsed = max(float(result.get("elapsed_seconds", 0.0)), 1e-9)
    total_elapsed = max(float(total_elapsed), 1e-9)
    target_processed_fps = source_fps / frame_stride
    sustained_fps = processed_frames / total_elapsed
    return {
        "source_frames": source_frames,
        "processed_frames": processed_frames,
        "skipped_frames": source_frames - processed_frames,
        "source_fps": source_fps,
        "frame_stride": frame_stride,
        "target_processed_fps": target_processed_fps,
        "worker_seconds": worker_elapsed,
        "end_to_end_seconds": total_elapsed,
        "worker_fps": processed_frames / worker_elapsed,
        "sustained_fps": sustained_fps,
        "realtime_factor": sustained_fps / target_processed_fps,
        "parallel_requests": int(result.get("parallel_requests", 1)),
        "warm_runtime": True,
    }


def _frames_to_payload(images: list[Image.Image]) -> list[bytes]:
    return [_image_bytes(image) for image in images]


def _validate_fps(value: float) -> float:
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive.")
    return fps


def _detect_sequence(
    items: list[dict[str, Any]],
    *,
    selected_indices: list[int],
    source_frames: int,
    width: int,
    height: int,
    object_prompt: str,
    fps: float,
    max_objects: int,
    metadata: dict[str, Any],
    source: str,
) -> DetectionSequence:
    frames = []
    for frame_index, item in zip(selected_indices, items):
        timestamp = frame_index / fps
        detections = []
        objects = item.get("objects", [])
        if not isinstance(objects, list):
            raise TypeError("Moondream detect result must contain an object list.")
        for record in objects[:max_objects]:
            detections.append(
                Detection(
                    bbox_xyxy=_pixel_bbox(record, width, height),
                    label=object_prompt,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source=source,
                    metadata={"native_bbox": record},
                )
            )
        frames.append(
            FrameDetections(
                frame_index=frame_index,
                timestamp=timestamp,
                width=width,
                height=height,
                detections=tuple(detections),
            )
        )
    return DetectionSequence(
        width=width,
        height=height,
        frames=tuple(frames),
        frame_count=source_frames,
        fps=fps,
        source=source,
        metadata={
            "skill": "detect",
            "object": object_prompt,
            "performance": metadata,
        },
    )


def _point_sequence(
    items: list[dict[str, Any]],
    *,
    selected_indices: list[int],
    source_frames: int,
    width: int,
    height: int,
    object_prompt: str,
    fps: float,
    max_points: int,
    metadata: dict[str, Any],
    source: str,
) -> PointSequence:
    points = []
    for frame_index, item in zip(selected_indices, items):
        timestamp = frame_index / fps
        records = item.get("points", [])
        if not isinstance(records, list):
            raise TypeError("Moondream point result must contain a point list.")
        for record in records[:max_points]:
            if not isinstance(record, dict):
                raise TypeError("Moondream points must be objects.")
            x = _normalized_number(record.get("x"), "point x") * width
            y = _normalized_number(record.get("y"), "point y") * height
            points.append(
                VisionPoint(
                    x=x,
                    y=y,
                    label=object_prompt,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source=source,
                    metadata={"native_point": record},
                )
            )
    return PointSequence(
        width=width,
        height=height,
        points=tuple(points),
        frame_count=source_frames,
        fps=fps,
        source=source,
        metadata={
            "skill": "point",
            "object": object_prompt,
            "performance": metadata,
        },
    )


def _render_points(image: torch.Tensor, points: PointSequence) -> torch.Tensor:
    output = []
    by_frame: dict[int, list[VisionPoint]] = {}
    for point in points.points:
        by_frame.setdefault(point.frame_index, []).append(point)
    for frame_index, frame in enumerate(tensor_batch_to_pil(image)):
        canvas = frame.copy()
        draw = ImageDraw.Draw(canvas)
        radius = max(4, round(min(canvas.size) / 100))
        width = max(2, round(radius / 3))
        for point in by_frame.get(frame_index, []):
            x, y = point.x, point.y
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(0, 255, 128),
                width=width,
            )
            draw.line((x - radius, y, x + radius, y), fill=(0, 255, 128), width=width)
            draw.line((x, y - radius, x, y + radius), fill=(0, 255, 128), width=width)
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        output.append(torch.from_numpy(array.copy()))
    return torch.stack(output)


def _spatial_refs(
    value: str,
    width: int,
    height: int,
    points: PointSequence | None,
    detections: DetectionSequence | None,
) -> list[list[float]]:
    refs: list[list[float]] = []
    text = str(value or "").strip()
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid spatial_refs_json: {exc.msg}.") from exc
        if not isinstance(parsed, list):
            raise ValueError("spatial_refs_json must be a JSON array.")
        refs.extend(parsed)
    if points is not None:
        if not isinstance(points, PointSequence):
            raise TypeError("points must be a VLM point sequence.")
        refs.extend([[point.x / width, point.y / height] for point in points.points])
    if detections is not None:
        if not isinstance(detections, DetectionSequence):
            raise TypeError("detections must be a VLM detection sequence.")
        for detection in detections.all_detections():
            x1, y1, x2, y2 = detection.bbox_xyxy
            refs.append([x1 / width, y1 / height, x2 / width, y2 / height])
    if len(refs) > 64:
        raise ValueError("At most 64 spatial references are accepted.")
    normalized = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, (list, tuple)) or len(ref) not in (2, 4):
            raise ValueError(
                f"Spatial reference {index} must contain 2 point values or 4 bbox values."
            )
        values = [
            _normalized_number(component, f"spatial reference {index}")
            for component in ref
        ]
        if len(values) == 4 and (values[2] <= values[0] or values[3] <= values[1]):
            raise ValueError(f"Spatial bbox reference {index} has no positive area.")
        normalized.append(values)
    return normalized


class Moondream31Loader(CachedModelNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "license_accepted": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Required acknowledgement of Moondream Model License "
                            f"1.0: {MODEL_LICENSE_URL}"
                        ),
                    },
                ),
                "device": (DEVICE_CHOICES, {"default": "Auto"}),
                "max_batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "tooltip": (
                            "Maximum concurrent Photon requests. 4 is a strong "
                            "starting point for video detection."
                        ),
                    },
                ),
                "kv_cache_profile": (
                    tuple(KV_CACHE_PROFILES),
                    {"default": "Balanced (8K pages)"},
                ),
            },
            "optional": {
                "model_or_adapter": (
                    "STRING",
                    {
                        "default": MODEL_ID,
                        "tooltip": (
                            "Use moondream3.1-9B-A2B for query/caption/detect/"
                            "point, or moondream3-preview for SVG segment. "
                            "Adapters may use the upstream base/adapter syntax."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (MOONDREAM31_MODEL, "STRING")
    RETURN_NAMES = ("model", "runtime_info")
    FUNCTION = "load"
    CATEGORY = "VLM Nodes/Moondream 3"
    DESCRIPTION = (
        "Load official Moondream 3/3.1 Photon in a conflict-free sidecar. "
        "The model cache, worker environment, and logs stay under ComfyUI models."
    )

    def load(
        self,
        license_accepted,
        device,
        max_batch_size,
        kv_cache_profile,
        model_or_adapter=MODEL_ID,
    ):
        if not license_accepted:
            raise ValueError(
                "Moondream Model License 1.0 acceptance is required before "
                f"loading. Read {MODEL_LICENSE_URL}"
            )
        model_name = str(model_or_adapter).strip()
        if not model_name or any(character in model_name for character in "\r\n\0"):
            raise ValueError("model_or_adapter must be one non-empty line.")
        config = Moondream31Config(
            model=model_name,
            device=_device_name(device),
            max_batch_size=int(max_batch_size),
            kv_cache_pages=KV_CACHE_PROFILES[kv_cache_profile],
        )
        model = self.get_or_create_model(config, lambda: Moondream31Model(config))
        model.ensure_started()
        return model, json.dumps(
            model.runtime_info,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )


class Moondream31Query:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MOONDREAM31_MODEL,),
                "question": (
                    "STRING",
                    {"multiline": True, "default": "Describe this image precisely."},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192},
                ),
                "reasoning": ("BOOLEAN", {"default": False}),
                "stream_output": ("BOOLEAN", {"default": True}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1_000_000},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "reasoning_json", "performance_json")
    FUNCTION = "query"
    CATEGORY = "VLM Nodes/Moondream 3"

    def query(
        self,
        model,
        question,
        max_tokens,
        reasoning,
        stream_output,
        unload_after,
        image=None,
        image_index=0,
        unique_id=None,
    ):
        if not isinstance(model, Moondream31Model):
            raise TypeError("model must come from Moondream 3.1 Loader.")
        prompt = str(question).strip()
        if not prompt:
            raise ValueError("question cannot be empty.")
        payload = None
        if image is not None:
            images = tensor_batch_to_pil(image)
            index = int(image_index)
            if not 0 <= index < len(images):
                raise IndexError("image_index is outside the input batch.")
            payload = _image_bytes(images[index])
        progress = _progress_text_sender(unique_id) if bool(stream_output) else None
        started = time.perf_counter()
        try:
            result = model.request(
                "query",
                progress=progress,
                image=payload,
                question=prompt,
                max_tokens=int(max_tokens),
                reasoning=bool(reasoning),
                stream=bool(stream_output),
            )
            performance = {
                "worker_seconds": result.get("elapsed_seconds"),
                "end_to_end_seconds": time.perf_counter() - started,
                "warm_runtime": True,
            }
            return (
                str(result.get("answer") or ""),
                json.dumps(
                    result.get("reasoning"),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                ),
                json.dumps(performance, allow_nan=False, sort_keys=True, indent=2),
            )
        finally:
            if unload_after:
                model.close()


class Moondream31Caption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MOONDREAM31_MODEL,),
                "image": ("IMAGE",),
                "length": (("short", "normal", "long"), {"default": "normal"}),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8192},
                ),
                "stream_output": ("BOOLEAN", {"default": True}),
                "unload_after": ("BOOLEAN", {"default": False}),
                "image_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1_000_000},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "performance_json")
    FUNCTION = "caption"
    CATEGORY = "VLM Nodes/Moondream 3"

    def caption(
        self,
        model,
        image,
        length,
        max_tokens,
        stream_output,
        unload_after,
        image_index,
        unique_id=None,
    ):
        if not isinstance(model, Moondream31Model):
            raise TypeError("model must come from Moondream 3.1 Loader.")
        images = tensor_batch_to_pil(image)
        index = int(image_index)
        if not 0 <= index < len(images):
            raise IndexError("image_index is outside the input batch.")
        progress = _progress_text_sender(unique_id) if bool(stream_output) else None
        started = time.perf_counter()
        try:
            result = model.request(
                "caption",
                progress=progress,
                image=_image_bytes(images[index]),
                length=length,
                max_tokens=int(max_tokens),
                stream=bool(stream_output),
            )
            performance = {
                "worker_seconds": result.get("elapsed_seconds"),
                "end_to_end_seconds": time.perf_counter() - started,
                "warm_runtime": True,
            }
            return str(result.get("caption") or ""), json.dumps(
                performance,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
        finally:
            if unload_after:
                model.close()


class Moondream31Detect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MOONDREAM31_MODEL,),
                "image": ("IMAGE",),
                "object": (
                    "STRING",
                    {"multiline": False, "default": "person"},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.001, "max": 1000.0, "step": 0.001},
                ),
                "frame_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100_000,
                        "tooltip": "1 analyzes every frame; 2 analyzes every other frame.",
                    },
                ),
                "parallel_requests": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "tooltip": (
                            "Concurrent frame requests let Photon form dynamic GPU batches."
                        ),
                    },
                ),
                "max_objects": (
                    "INT",
                    {"default": 100, "min": 1, "max": 1000},
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        VLM_DETECTIONS,
        "STRING",
        "IMAGE",
        "MASK",
        "BOUNDING_BOX",
        "BOUNDING_BOXES",
        "STRING",
    )
    RETURN_NAMES = (
        "detections",
        "detections_json",
        "preview",
        "box_masks",
        "bounding_boxes",
        "bounding_boxes_with_metadata",
        "performance_json",
    )
    FUNCTION = "detect"
    CATEGORY = "VLM Nodes/Moondream 3"
    DESCRIPTION = (
        "Warm, dynamically batched Moondream detection for images or video "
        "frame batches. Reports measured worker and end-to-end FPS."
    )

    def detect(
        self,
        model,
        image,
        object,
        fps,
        frame_stride,
        parallel_requests,
        max_objects,
        unload_after,
    ):
        if not isinstance(model, Moondream31Model):
            raise TypeError("model must come from Moondream 3.1 Loader.")
        object_prompt = str(object).strip()
        if not object_prompt:
            raise ValueError("object cannot be empty.")
        fps_value = _validate_fps(fps)
        sampled, indices, width, height = _sampled_images(image, int(frame_stride))
        started = time.perf_counter()
        try:
            result = model.request(
                "detect",
                images=_frames_to_payload(sampled),
                object=object_prompt,
                parallel_requests=int(parallel_requests),
                max_tokens=64,
            )
            performance = _performance(
                result=result,
                total_elapsed=time.perf_counter() - started,
                source_frames=len(tensor_batch_to_pil(image)),
                processed_frames=len(indices),
                source_fps=fps_value,
                frame_stride=int(frame_stride),
            )
            items = result.get("items", [])
            if not isinstance(items, list) or len(items) != len(indices):
                raise RuntimeError("Moondream returned an incomplete detect batch.")
            sequence = _detect_sequence(
                items,
                selected_indices=indices,
                source_frames=int(image.shape[0]) if image.ndim == 4 else 1,
                width=width,
                height=height,
                object_prompt=object_prompt,
                fps=fps_value,
                max_objects=int(max_objects),
                metadata=performance,
                source=_model_source(model.config.model),
            )
            return (
                sequence,
                sequence.to_json(indent=2),
                render_detections(image, sequence),
                detection_box_masks(sequence),
                core_bounding_box_frames(sequence),
                core_bounding_boxes(sequence),
                json.dumps(performance, allow_nan=False, sort_keys=True, indent=2),
            )
        finally:
            if unload_after:
                model.close()


class Moondream31Point:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = Moondream31Detect.INPUT_TYPES()
        required = dict(inputs["required"])
        required["object"] = (
            "STRING",
            {"multiline": False, "default": "person"},
        )
        required["max_points"] = required.pop("max_objects")
        return {"required": required}

    RETURN_TYPES = (VLM_POINTS, "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("points", "points_json", "preview", "performance_json")
    FUNCTION = "point"
    CATEGORY = "VLM Nodes/Moondream 3"

    def point(
        self,
        model,
        image,
        object,
        fps,
        frame_stride,
        parallel_requests,
        max_points,
        unload_after,
    ):
        if not isinstance(model, Moondream31Model):
            raise TypeError("model must come from Moondream 3.1 Loader.")
        object_prompt = str(object).strip()
        if not object_prompt:
            raise ValueError("object cannot be empty.")
        fps_value = _validate_fps(fps)
        sampled, indices, width, height = _sampled_images(image, int(frame_stride))
        started = time.perf_counter()
        try:
            result = model.request(
                "point",
                images=_frames_to_payload(sampled),
                object=object_prompt,
                parallel_requests=int(parallel_requests),
                max_tokens=64,
            )
            performance = _performance(
                result=result,
                total_elapsed=time.perf_counter() - started,
                source_frames=int(image.shape[0]) if image.ndim == 4 else 1,
                processed_frames=len(indices),
                source_fps=fps_value,
                frame_stride=int(frame_stride),
            )
            items = result.get("items", [])
            if not isinstance(items, list) or len(items) != len(indices):
                raise RuntimeError("Moondream returned an incomplete point batch.")
            sequence = _point_sequence(
                items,
                selected_indices=indices,
                source_frames=int(image.shape[0]) if image.ndim == 4 else 1,
                width=width,
                height=height,
                object_prompt=object_prompt,
                fps=fps_value,
                max_points=int(max_points),
                metadata=performance,
                source=_model_source(model.config.model),
            )
            return (
                sequence,
                sequence.to_json(indent=2),
                _render_points(image, sequence),
                json.dumps(performance, allow_nan=False, sort_keys=True, indent=2),
            )
        finally:
            if unload_after:
                model.close()


class Moondream31Segment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MOONDREAM31_MODEL,),
                "image": ("IMAGE",),
                "object": (
                    "STRING",
                    {"multiline": False, "default": "foreground object"},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.001, "max": 1000.0, "step": 0.001},
                ),
                "frame_stride": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100_000},
                ),
                "parallel_requests": (
                    "INT",
                    {"default": 2, "min": 1, "max": 32},
                ),
                "svg_supersample": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "tooltip": "Higher values produce smoother mask edges.",
                    },
                ),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "spatial_refs_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]",
                        "tooltip": (
                            "Normalized [x,y] points and/or [x1,y1,x2,y2] boxes."
                        ),
                    },
                ),
                "points": (VLM_POINTS,),
                "detections": (VLM_DETECTIONS,),
            },
        }

    RETURN_TYPES = (
        VLM_DETECTIONS,
        "STRING",
        "STRING",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "BOUNDING_BOX",
        "BOUNDING_BOXES",
        "STRING",
    )
    RETURN_NAMES = (
        "segments",
        "segments_json",
        "native_svg_paths",
        "combined_masks",
        "black_white_masks",
        "cutouts",
        "overlays",
        "bounding_boxes",
        "bounding_boxes_with_metadata",
        "performance_json",
    )
    FUNCTION = "segment"
    CATEGORY = "VLM Nodes/Moondream 3"
    DESCRIPTION = (
        "Preserve Moondream's native SVG geometry and convert it into "
        "antialiased masks, polygons, cutouts, overlays, and core boxes."
    )

    def segment(
        self,
        model,
        image,
        object,
        fps,
        frame_stride,
        parallel_requests,
        svg_supersample,
        unload_after,
        spatial_refs_json="[]",
        points=None,
        detections=None,
    ):
        if not isinstance(model, Moondream31Model):
            raise TypeError("model must come from Moondream 3.1 Loader.")
        if _base_model_name(model.config.model) != PREVIEW_MODEL_ID:
            raise ValueError(
                "SVG segment is a Moondream 3 Preview skill. The official "
                "Moondream 3.1 model exposes query, caption, detect, and point "
                "but does not list segment. Load model_or_adapter="
                "'moondream3-preview' for this node."
            )
        object_prompt = str(object).strip()
        if not object_prompt:
            raise ValueError("object cannot be empty.")
        fps_value = _validate_fps(fps)
        sampled, indices, width, height = _sampled_images(image, int(frame_stride))
        refs = _spatial_refs(
            spatial_refs_json,
            width,
            height,
            points,
            detections,
        )
        started = time.perf_counter()
        try:
            result = model.request(
                "segment",
                images=_frames_to_payload(sampled),
                object=object_prompt,
                spatial_refs=refs,
                parallel_requests=int(parallel_requests),
                max_tokens=2048,
            )
            performance = _performance(
                result=result,
                total_elapsed=time.perf_counter() - started,
                source_frames=int(image.shape[0]) if image.ndim == 4 else 1,
                processed_frames=len(indices),
                source_fps=fps_value,
                frame_stride=int(frame_stride),
            )
            items = result.get("items", [])
            if not isinstance(items, list) or len(items) != len(indices):
                raise RuntimeError("Moondream returned an incomplete segment batch.")

            frames = []
            paths = []
            for frame_index, item in zip(indices, items):
                if not isinstance(item, dict):
                    raise TypeError("Moondream segment items must be objects.")
                path = str(item.get("path") or "")
                native_bbox = item.get(
                    "bbox",
                    {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
                )
                mask, polygon, contours = svg_path_to_mask(
                    path,
                    native_bbox,
                    width,
                    height,
                    supersample=int(svg_supersample),
                )
                timestamp = frame_index / fps_value
                detection = Detection(
                    bbox_xyxy=_pixel_bbox(native_bbox, width, height),
                    label=object_prompt,
                    polygon=polygon,
                    mask=mask,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source=_model_source(model.config.model),
                    metadata={
                        "native_bbox": native_bbox,
                        "native_svg_path": path,
                        "svg_subpaths": len(contours),
                        "svg_fill_rule": "evenodd",
                    },
                )
                frames.append(
                    FrameDetections(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        width=width,
                        height=height,
                        detections=(detection,),
                    )
                )
                paths.append(
                    {
                        "frame_index": frame_index,
                        "bbox": native_bbox,
                        "path": path,
                    }
                )
            sequence = DetectionSequence(
                width=width,
                height=height,
                frames=tuple(frames),
                frame_count=int(image.shape[0]) if image.ndim == 4 else 1,
                fps=fps_value,
                source=_model_source(model.config.model),
                metadata={
                    "skill": "segment",
                    "object": object_prompt,
                    "spatial_refs": refs,
                    "performance": performance,
                },
            )
            masks, _individual, _mapping = sequence_masks(sequence)
            _composite, cutouts, _background, _mask_preview = composite_with_mask(
                image,
                masks,
                background_color="#000000",
            )
            return (
                sequence,
                sequence.to_json(indent=2),
                json.dumps(paths, ensure_ascii=False, allow_nan=False, indent=2),
                masks,
                masks_to_images(masks),
                cutouts,
                render_detections(image, sequence),
                core_bounding_box_frames(sequence),
                core_bounding_boxes(sequence),
                json.dumps(performance, allow_nan=False, sort_keys=True, indent=2),
            )
        finally:
            if unload_after:
                model.close()


NODE_CLASS_MAPPINGS = {
    "Moondream31Loader": Moondream31Loader,
    "Moondream31Query": Moondream31Query,
    "Moondream31Caption": Moondream31Caption,
    "Moondream31Detect": Moondream31Detect,
    "Moondream31Point": Moondream31Point,
    "Moondream31Segment": Moondream31Segment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Moondream31Loader": "Moondream 3 / 3.1 Loader (Isolated Photon)",
    "Moondream31Query": "Moondream 3 / 3.1 Query",
    "Moondream31Caption": "Moondream 3 / 3.1 Caption",
    "Moondream31Detect": "Moondream 3 / 3.1 Detect (Image / Video)",
    "Moondream31Point": "Moondream 3 / 3.1 Point (Image / Video)",
    "Moondream31Segment": "Moondream 3 Preview SVG Segment (Image / Video)",
}
