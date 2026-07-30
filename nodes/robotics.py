"""Dependency-isolated robotics/VLA workflow primitives.

This module deliberately stops at the policy/safety boundary.  It prepares
observations, calls an isolated policy server, validates returned action
chunks, and visualizes them.  It never imports a robot SDK or sends commands
to hardware.

Heavy policy stacks (LeRobot, openpi, Isaac-GR00T, OpenVLA/OFT, and their
accelerator-specific dependencies) belong in a separate environment.  The
ComfyUI process remains portable and communicates through small, explicit
client protocols.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch
from PIL import Image, ImageDraw

from .runtime import require_module, tensor_to_pil

VLA_EMBODIMENT = "VLA_EMBODIMENT"
VLA_OBSERVATION = "VLA_OBSERVATION"
VLA_ACTIONS = "VLA_ACTIONS"

ROBOTICS_SCHEMA_VERSION = 1
EMBODIMENT_SCHEMA = "comfyui-vlm/robot-embodiment"
OBSERVATION_SCHEMA = "comfyui-vlm/robot-observation"
ACTIONS_SCHEMA = "comfyui-vlm/robot-actions"

MAX_TASK_CHARS = 16_384
MAX_STATE_DIM = 2_048
MAX_ACTION_DIM = 2_048
MAX_ACTION_HORIZON = 4_096
MAX_CAMERAS = 16
MAX_HISTORY_FRAMES = 256
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_HTTP_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_WIRE_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_WIRE_IMAGE_BYTES = 45 * 1024 * 1024
MAX_NATIVE_MESSAGE_BYTES = 128 * 1024 * 1024
MAX_ERROR_CHARS = 1_000

SAFETY_MODES = (
    "Block unsafe",
    "Clamp safely",
    "Hold position on unsafe",
    "Report only",
)
OPENPI_LAYOUTS = (
    "Flat keys (DROID / LIBERO)",
    "Nested images (ALOHA)",
)
TOKEN_ENV_BY_PROTOCOL = {
    "http": "VLA_POLICY_TOKEN",
    "openpi": "OPENPI_API_KEY",
    "groot": "GROOT_API_TOKEN",
}


def _json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _finite_vector(
    values: Sequence[Any],
    *,
    name: str,
    expected: int | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a JSON array of numbers.")
    result = tuple(_finite_float(value, f"{name}[{index}]") for index, value in enumerate(values))
    if expected is not None and len(result) != expected:
        raise ValueError(f"{name} must contain {expected} values, got {len(result)}.")
    return result


def _name_vector(
    values: Sequence[Any],
    *,
    name: str,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a JSON array of names.")
    result = tuple(str(value).strip() for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) > maximum:
        raise ValueError(f"{name} exceeds the supported limit of {maximum}.")
    if any(not value for value in result):
        raise ValueError(f"{name} must not contain empty names.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate names.")
    return result


def _parse_json(value: str, *, name: str) -> Any:
    try:
        return json.loads((value or "").strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{name} is invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from None


def _optional_json_array(value: str, *, name: str) -> list[Any] | None:
    raw = (value or "").strip()
    if not raw:
        return None
    parsed = _parse_json(raw, name=name)
    if not isinstance(parsed, list):
        raise TypeError(f"{name} must be a JSON array.")
    return parsed


@dataclass(frozen=True, slots=True)
class RobotEmbodiment:
    """The action contract for one robot/policy pairing.

    Presets are conservative workflow templates, not certified robot limits.
    Real hardware deployments must replace them with limits from the robot
    manufacturer and the trained policy's dataset metadata.
    """

    name: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]
    max_delta: tuple[float, ...]
    control_hz: float
    action_mode: str
    camera_names: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        clean_name = str(self.name).strip()
        if not clean_name:
            raise ValueError("Embodiment name must not be empty.")
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(
            self,
            "state_names",
            _name_vector(self.state_names, name="state_names", maximum=MAX_STATE_DIM),
        )
        action_names = _name_vector(
            self.action_names,
            name="action_names",
            maximum=MAX_ACTION_DIM,
        )
        object.__setattr__(self, "action_names", action_names)
        action_min = _finite_vector(
            self.action_min,
            name="action_min",
            expected=len(action_names),
        )
        action_max = _finite_vector(
            self.action_max,
            name="action_max",
            expected=len(action_names),
        )
        max_delta = _finite_vector(
            self.max_delta,
            name="max_delta",
            expected=len(action_names),
        )
        if any(low >= high for low, high in zip(action_min, action_max, strict=True)):
            raise ValueError("Every action_min value must be smaller than action_max.")
        if any(value <= 0 for value in max_delta):
            raise ValueError("Every max_delta value must be positive.")
        object.__setattr__(self, "action_min", action_min)
        object.__setattr__(self, "action_max", action_max)
        object.__setattr__(self, "max_delta", max_delta)
        control_hz = _finite_float(self.control_hz, "control_hz")
        if control_hz <= 0 or control_hz > 10_000:
            raise ValueError("control_hz must be in the interval (0, 10000].")
        object.__setattr__(self, "control_hz", control_hz)
        action_mode = str(self.action_mode).strip()
        if not action_mode:
            raise ValueError("action_mode must not be empty.")
        object.__setattr__(self, "action_mode", action_mode)
        cameras = _name_vector(
            self.camera_names,
            name="camera_names",
            maximum=MAX_CAMERAS,
        )
        object.__setattr__(self, "camera_names", cameras)
        object.__setattr__(self, "notes", str(self.notes).strip())

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EMBODIMENT_SCHEMA,
            "version": ROBOTICS_SCHEMA_VERSION,
            "name": self.name,
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "action_min": list(self.action_min),
            "action_max": list(self.action_max),
            "max_delta_per_step": list(self.max_delta),
            "control_hz": self.control_hz,
            "action_mode": self.action_mode,
            "camera_names": list(self.camera_names),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RobotEmbodiment:
        if not isinstance(value, Mapping):
            raise TypeError("An embodiment must be a JSON object.")
        schema = value.get("schema")
        if schema not in (None, EMBODIMENT_SCHEMA):
            raise ValueError(f"Unsupported embodiment schema {schema!r}.")
        version = int(value.get("version", ROBOTICS_SCHEMA_VERSION))
        if version != ROBOTICS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported embodiment schema version {version}.")
        return cls(
            name=value["name"],
            state_names=tuple(value["state_names"]),
            action_names=tuple(value["action_names"]),
            action_min=tuple(value["action_min"]),
            action_max=tuple(value["action_max"]),
            max_delta=tuple(value.get("max_delta_per_step", value.get("max_delta", ()))),
            control_hz=value["control_hz"],
            action_mode=value["action_mode"],
            camera_names=tuple(value["camera_names"]),
            notes=value.get("notes", ""),
        )


def _numbered_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{index}" for index in range(count)]


_EMBODIMENT_PRESET_SPECS: dict[str, dict[str, Any]] = {
    "Generic 7-DoF joint + gripper": {
        "name": "generic_7dof_joint_gripper",
        "state_names": [*_numbered_names("joint", 7), "gripper"],
        "action_names": [*_numbered_names("joint", 7), "gripper"],
        "action_min": [-math.pi] * 7 + [0.0],
        "action_max": [math.pi] * 7 + [1.0],
        "max_delta": [0.12] * 7 + [0.15],
        "control_hz": 20.0,
        "action_mode": "absolute joint position",
        "camera_names": ["observation.images.front"],
    },
    "Generic EEF delta + gripper": {
        "name": "generic_eef_delta_gripper",
        "state_names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        "action_names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        "action_min": [-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, -1.0],
        "action_max": [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0],
        "max_delta": [0.025, 0.025, 0.025, 0.12, 0.12, 0.12, 0.5],
        "control_hz": 20.0,
        "action_mode": "delta end-effector pose",
        "camera_names": ["observation.images.front", "observation.images.wrist"],
    },
    "LeRobot SO-100 / SO-101 template": {
        "name": "lerobot_so100_so101_template",
        "state_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "action_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "action_min": [-1.0] * 6,
        "action_max": [1.0] * 6,
        "max_delta": [0.10] * 5 + [0.15],
        "control_hz": 30.0,
        "action_mode": "dataset-normalized template",
        "camera_names": ["observation.images.front", "observation.images.wrist"],
    },
    "LIBERO Panda simulation template": {
        "name": "libero_panda_template",
        "state_names": _numbered_names("state", 8),
        "action_names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        "action_min": [-1.0] * 7,
        "action_max": [1.0] * 7,
        "max_delta": [0.25] * 6 + [1.0],
        "control_hz": 20.0,
        "action_mode": "LIBERO normalized delta template",
        "camera_names": ["observation.image", "observation.wrist_image"],
    },
    "ALOHA bimanual template": {
        "name": "aloha_bimanual_template",
        "state_names": _numbered_names("joint", 14),
        "action_names": _numbered_names("joint", 14),
        "action_min": [-math.pi] * 14,
        "action_max": [math.pi] * 14,
        "max_delta": [0.12] * 14,
        "control_hz": 50.0,
        "action_mode": "absolute bimanual joint position",
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
}

EMBODIMENT_PRESETS = tuple(_EMBODIMENT_PRESET_SPECS)
_PRESET_WARNING = (
    "Template limits are workflow defaults only. Replace them with the trained "
    "checkpoint's dataset statistics and manufacturer/controller limits before deployment."
)


def make_embodiment(
    preset: str,
    *,
    state_names: Sequence[Any] | None = None,
    action_names: Sequence[Any] | None = None,
    action_min: Sequence[Any] | None = None,
    action_max: Sequence[Any] | None = None,
    max_delta: Sequence[Any] | None = None,
    control_hz: float | None = None,
    camera_names: Sequence[Any] | None = None,
    action_mode: str | None = None,
) -> RobotEmbodiment:
    try:
        spec = dict(_EMBODIMENT_PRESET_SPECS[preset])
    except KeyError:
        raise ValueError(f"Unknown embodiment preset {preset!r}.") from None
    overrides = {
        "state_names": state_names,
        "action_names": action_names,
        "action_min": action_min,
        "action_max": action_max,
        "max_delta": max_delta,
        "control_hz": control_hz,
        "camera_names": camera_names,
        "action_mode": action_mode,
    }
    spec.update({key: value for key, value in overrides.items() if value is not None})
    spec["notes"] = _PRESET_WARNING
    return RobotEmbodiment(**spec)


def _canonical_image_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a ComfyUI IMAGE tensor.")
    image = value
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"{name} must have HWC/BHWC or CHW/BCHW shape.")
    if image.shape[-1] not in (1, 3, 4) and image.shape[1] in (1, 3, 4):
        image = image.permute(0, 2, 3, 1)
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{name} has unsupported channel shape {tuple(value.shape)}.")
    if image.shape[0] < 1 or image.shape[0] > MAX_HISTORY_FRAMES:
        raise ValueError(
            f"{name} must contain 1 to {MAX_HISTORY_FRAMES} observation frames."
        )
    if image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"{name} contains an empty image.")
    pixels = int(image.shape[1]) * int(image.shape[2])
    if pixels > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"{name} has {pixels} pixels per frame; limit is {MAX_IMAGE_PIXELS}. "
            "Resize it with VLM Image Pixel Budget."
        )
    if not image.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point ComfyUI IMAGE dtype.")
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError(f"{name} contains NaN or infinite pixel values.")
    return image


@dataclass(frozen=True, slots=True)
class RobotObservation:
    task: str
    state: tuple[float, ...]
    images: tuple[tuple[str, torch.Tensor], ...]
    timestamp: float
    history_fps: float
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        task = str(self.task).strip()
        if not task:
            raise ValueError("Robot task/instruction must not be empty.")
        if len(task) > MAX_TASK_CHARS:
            raise ValueError(f"Robot task exceeds {MAX_TASK_CHARS} characters.")
        object.__setattr__(self, "task", task)
        state = _finite_vector(self.state, name="state")
        if not state or len(state) > MAX_STATE_DIM:
            raise ValueError(f"state must contain 1 to {MAX_STATE_DIM} values.")
        object.__setattr__(self, "state", state)
        images = tuple(self.images)
        if not images or len(images) > MAX_CAMERAS:
            raise ValueError(f"images must contain 1 to {MAX_CAMERAS} cameras.")
        clean_images = []
        for key, image in images:
            clean_key = str(key).strip()
            if not clean_key:
                raise ValueError("Camera names must not be empty.")
            clean_images.append((clean_key, _canonical_image_batch(image, name=clean_key)))
        keys = [key for key, _image in clean_images]
        if len(keys) != len(set(keys)):
            raise ValueError("Camera names must be unique.")
        frame_counts = {int(image.shape[0]) for _key, image in clean_images}
        maximum = max(frame_counts)
        if any(count not in {1, maximum} for count in frame_counts):
            raise ValueError(
                "Camera histories must have the same frame count; a one-frame "
                "camera may broadcast across the longest history."
            )
        object.__setattr__(self, "images", tuple(clean_images))
        timestamp = _finite_float(self.timestamp, "timestamp")
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative.")
        object.__setattr__(self, "timestamp", timestamp)
        history_fps = _finite_float(self.history_fps, "history_fps")
        if history_fps <= 0:
            raise ValueError("history_fps must be positive.")
        object.__setattr__(self, "history_fps", history_fps)
        metadata = tuple((str(key), value) for key, value in self.metadata)
        object.__setattr__(self, "metadata", metadata)

    @property
    def history_frames(self) -> int:
        return max(int(image.shape[0]) for _key, image in self.images)

    @property
    def image_map(self) -> dict[str, torch.Tensor]:
        return dict(self.images)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "version": ROBOTICS_SCHEMA_VERSION,
            "task": self.task,
            "timestamp": self.timestamp,
            "history_fps": self.history_fps,
            "history_frames": self.history_frames,
            "state": list(self.state),
            "cameras": {
                key: {
                    "frames": int(image.shape[0]),
                    "height": int(image.shape[1]),
                    "width": int(image.shape[2]),
                    "channels": int(image.shape[3]),
                    "dtype": str(image.dtype),
                    "device": str(image.device),
                }
                for key, image in self.images
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RobotActions:
    values: torch.Tensor
    action_names: tuple[str, ...]
    source: str
    stream_slices: tuple[tuple[str, int, int], ...] = field(default_factory=tuple)
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("Action values must be a torch.Tensor.")
        values = self.values.detach().to(device="cpu", dtype=torch.float32)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2:
            raise ValueError("Action values must have shape [horizon, action_dim].")
        horizon, dimension = map(int, values.shape)
        if not 1 <= horizon <= MAX_ACTION_HORIZON:
            raise ValueError(f"Action horizon must be in [1, {MAX_ACTION_HORIZON}].")
        if not 1 <= dimension <= MAX_ACTION_DIM:
            raise ValueError(f"Action dimension must be in [1, {MAX_ACTION_DIM}].")
        object.__setattr__(self, "values", values.contiguous())
        names = tuple(str(name).strip() for name in self.action_names)
        if len(names) != dimension or any(not name for name in names):
            raise ValueError("action_names must contain one non-empty name per action dimension.")
        object.__setattr__(self, "action_names", names)
        object.__setattr__(self, "source", str(self.source).strip() or "unknown")
        slices = tuple(self.stream_slices)
        for stream, start, end in slices:
            if not str(stream).strip() or not 0 <= int(start) < int(end) <= dimension:
                raise ValueError("stream_slices contains an invalid range.")
        object.__setattr__(self, "stream_slices", slices)
        object.__setattr__(
            self,
            "metadata",
            tuple((str(key), value) for key, value in self.metadata),
        )

    @property
    def horizon(self) -> int:
        return int(self.values.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.values.shape[1])

    def to_dict(self, *, include_values: bool = True) -> dict[str, Any]:
        if include_values and not bool(torch.isfinite(self.values).all().item()):
            raise ValueError(
                "Action values contain NaN or infinity. Run VLA Action Safety "
                "before serializing or executing this trajectory."
            )
        result: dict[str, Any] = {
            "schema": ACTIONS_SCHEMA,
            "version": ROBOTICS_SCHEMA_VERSION,
            "source": self.source,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "action_names": list(self.action_names),
            "streams": [
                {"name": name, "start": start, "end": end}
                for name, start, end in self.stream_slices
            ],
            "metadata": dict(self.metadata),
        }
        if include_values:
            result["actions"] = self.values.tolist()
        return result


@dataclass(frozen=True, slots=True)
class VLAModelInfo:
    label: str
    family: str
    policy_type: str
    checkpoint: str
    backend: str
    scale: str
    status: str
    best_for: str
    license_note: str
    official_url: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "family": self.family,
            "policy_type": self.policy_type,
            "checkpoint": self.checkpoint,
            "backend": self.backend,
            "scale": self.scale,
            "status": self.status,
            "best_for": self.best_for,
            "license_note": self.license_note,
            "official_url": self.official_url,
            "notes": self.notes,
        }


_MODEL_INFOS = (
    VLAModelInfo(
        "SmolVLA 450M — fast consumer hardware",
        "SmolVLA",
        "smolvla",
        "lerobot/smolvla_base",
        "LeRobot sidecar",
        "450M",
        "base; fine-tune for the target embodiment",
        "small multi-camera manipulation and asynchronous control",
        "Apache-2.0 code; verify checkpoint/model card",
        "https://huggingface.co/lerobot/smolvla_base",
        "The preferred small starting point. Do not send its base actions to "
        "a robot without embodiment fine-tuning.",
    ),
    VLAModelInfo(
        "X-VLA 0.9B — cross-embodiment",
        "X-VLA",
        "xvla",
        "lerobot/xvla-base",
        "LeRobot sidecar",
        "0.9B",
        "base plus deployable domain checkpoints",
        "soft-prompt embodiment adaptation and multi-camera control",
        "Apache-2.0 code; verify checkpoint/model card",
        "https://huggingface.co/lerobot/xvla-base",
        "Use a domain checkpoint such as xvla-libero/widowx/folding where it matches the task.",
    ),
    VLAModelInfo(
        "π0 base — generalist VLA",
        "π0",
        "pi0",
        "lerobot/pi0_base",
        "LeRobot or openpi sidecar",
        "large",
        "base and LIBERO checkpoints",
        "generalist manipulation and embodiment fine-tuning",
        "Model terms vary; review the checkpoint",
        "https://huggingface.co/lerobot/pi0_base",
        "Use isolated inference; the upstream openpi runtime is primarily tested on Ubuntu.",
    ),
    VLAModelInfo(
        "π0-FAST — tokenized fast actions",
        "π0-FAST",
        "pi0_fast",
        "lerobot/pi0fast-base",
        "LeRobot or openpi sidecar",
        "large",
        "base and LIBERO checkpoints",
        "lower-latency autoregressive action generation",
        "Model terms vary; review the checkpoint",
        "https://huggingface.co/lerobot/pi0fast-base",
        "FAST action tokenization is supported by current LeRobot and openpi runtimes.",
    ),
    VLAModelInfo(
        "π0.5 base — open-world generalization",
        "π0.5",
        "pi05",
        "lerobot/pi05_base",
        "LeRobot or openpi sidecar",
        "large",
        "base and LIBERO checkpoints",
        "language-conditioned manipulation and open-world tasks",
        "Model terms vary; review the checkpoint",
        "https://huggingface.co/lerobot/pi05_base",
        "Requires an embodiment-compatible checkpoint and observation keys.",
    ),
    VLAModelInfo(
        "GR00T N1.7 3B — cross-embodiment",
        "Isaac GR00T N1.7",
        "groot",
        "nvidia/GR00T-N1.7-3B",
        "GR00T ZMQ or LeRobot sidecar",
        "3B",
        "base plus DROID/LIBERO/SimplerEnv checkpoints",
        "cross-embodiment manipulation and NVIDIA deployment",
        "Apache-2.0 repository; verify checkpoint terms",
        "https://huggingface.co/nvidia/GR00T-N1.7-3B",
        "Use strict modality validation during development; action streams "
        "are returned in physical units.",
    ),
    VLAModelInfo(
        "WALL-OSS — mixture-of-experts VLA",
        "WALL-OSS",
        "wall_x",
        "x-square-robot/wall-oss-flow",
        "LeRobot sidecar",
        "large MoE",
        "pretrained flow checkpoint",
        "general manipulation with flow/diffusion or FAST actions",
        "Apache-2.0 integration; verify model card",
        "https://huggingface.co/x-square-robot/wall-oss-flow",
        "Fine-tune and validate on the target embodiment before use.",
    ),
    VLAModelInfo(
        "MolmoAct2 — action reasoning",
        "MolmoAct2",
        "molmoact2",
        "lerobot/MolmoAct2-SO100_101-LeRobot",
        "LeRobot sidecar",
        "large",
        "converted SO-100/SO-101 checkpoint",
        "robot action reasoning and SO-100/SO-101 experiments",
        "Review AllenAI and checkpoint terms",
        "https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot",
        "The converted checkpoint uses the current LeRobot processor convention.",
    ),
    VLAModelInfo(
        "VLA-JEPA — predictive world representation",
        "VLA-JEPA",
        "vla_jepa",
        "lerobot/VLA-JEPA-Pretrain",
        "LeRobot sidecar",
        "research",
        "DROID pretrain plus LIBERO/SimplerEnv checkpoints",
        "multi-camera temporal prediction and representation learning",
        "Verify original and converted checkpoint terms",
        "https://huggingface.co/lerobot/VLA-JEPA-Pretrain",
        "Action/state projection dimensions may need explicit reinitialization when fine-tuning.",
    ),
    VLAModelInfo(
        "LingBot-VA — video-action model",
        "LingBot-VA",
        "lingbot_va",
        "lerobot/lingbot_va_base",
        "LeRobot sidecar",
        "large",
        "base plus LIBERO-Long and RoboTwin",
        "long-horizon video-conditioned manipulation",
        "Verify checkpoint/model card",
        "https://huggingface.co/lerobot/lingbot_va_base",
        "Select the post-trained domain checkpoint where available.",
    ),
    VLAModelInfo(
        "FastWAM — world action model",
        "FastWAM",
        "fastwam",
        "ZibinDong/fastwam_libero_uncond_2cam224",
        "LeRobot sidecar",
        "5B backbone class",
        "released LIBERO research checkpoint",
        "world-action modeling and video prediction research",
        "Verify checkpoint and Wan component terms",
        "https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224",
        "Heavy multi-component runtime; not a low-latency default.",
    ),
    VLAModelInfo(
        "EO-1 — efficient VLA architecture",
        "EO-1",
        "eo1",
        "",
        "LeRobot sidecar",
        "3B VLM class",
        "architecture/training integration; supply your checkpoint",
        "efficient action policy research",
        "Verify backbone and checkpoint terms",
        "https://huggingface.co/docs/lerobot/eo1",
        "Current LeRobot integration documents training but does not prescribe "
        "a universal robot-ready checkpoint.",
    ),
    VLAModelInfo(
        "EVO-1 — embodied multimodal architecture",
        "EVO-1",
        "evo1",
        "",
        "LeRobot sidecar",
        "research",
        "architecture/training integration; supply your checkpoint",
        "embodied multimodal policy research",
        "Verify backbone and checkpoint terms",
        "https://huggingface.co/docs/lerobot/evo1",
        "Train or load an embodiment-specific LeRobot checkpoint.",
    ),
    VLAModelInfo(
        "OpenVLA-OFT 7B — optimized OpenVLA",
        "OpenVLA-OFT",
        "openvla_oft",
        "openvla/openvla-7b",
        "dedicated OpenVLA/OFT sidecar",
        "7B",
        "base plus OFT fine-tunes",
        "high-frequency multi-image OpenVLA deployment",
        "MIT code; checkpoint/data terms vary",
        "https://github.com/openvla/openvla",
        "OFT is the recommended OpenVLA path for substantially faster action generation.",
    ),
    VLAModelInfo(
        "Octo small 27M — legacy lightweight research",
        "Octo",
        "octo",
        "hf://rail-berkeley/octo-small",
        "dedicated JAX sidecar",
        "27M",
        "legacy research checkpoint",
        "very small goal/language-conditioned research baseline",
        "MIT code; verify checkpoint terms",
        "https://github.com/octo-models/octo",
        "Useful for research comparisons; the JAX stack is isolated from ComfyUI.",
    ),
)

VLA_MODEL_CATALOG = {info.label: info for info in _MODEL_INFOS}
VLA_MODEL_LABELS = tuple(VLA_MODEL_CATALOG)


def _state_from_json(
    value: str,
    embodiment: RobotEmbodiment | None,
) -> tuple[float, ...]:
    parsed = _parse_json(value, name="state_json")
    if isinstance(parsed, Mapping):
        if embodiment is None:
            raise ValueError(
                "Object-form state_json requires an embodiment so key order is explicit."
            )
        missing = [name for name in embodiment.state_names if name not in parsed]
        extras = [name for name in parsed if name not in embodiment.state_names]
        if missing or extras:
            raise ValueError(
                f"State keys do not match embodiment. Missing={missing}, extra={extras}."
            )
        parsed = [parsed[name] for name in embodiment.state_names]
    state = _finite_vector(parsed, name="state")
    if embodiment is not None and len(state) != embodiment.state_dim:
        raise ValueError(
            f"State dimension {len(state)} does not match embodiment dimension "
            f"{embodiment.state_dim}."
        )
    return state


def _safe_camera_name(value: str, *, name: str) -> str:
    result = str(value).strip()
    if not result or len(result) > 256:
        raise ValueError(f"{name} must contain 1 to 256 characters.")
    if any(ord(char) < 32 for char in result):
        raise ValueError(f"{name} contains a control character.")
    return result


def build_observation(
    *,
    task: str,
    state_json: str,
    primary_image: torch.Tensor,
    primary_camera: str,
    history_fps: float,
    timestamp: float,
    embodiment: RobotEmbodiment | None = None,
    wrist_image: torch.Tensor | None = None,
    wrist_camera: str = "observation.images.wrist",
    secondary_image: torch.Tensor | None = None,
    secondary_camera: str = "observation.images.secondary",
    goal_image: torch.Tensor | None = None,
    goal_camera: str = "observation.images.goal",
) -> RobotObservation:
    state = _state_from_json(state_json, embodiment)
    images = [
        (
            _safe_camera_name(primary_camera, name="primary_camera"),
            primary_image,
        )
    ]
    for image, key, field_name in (
        (wrist_image, wrist_camera, "wrist_camera"),
        (secondary_image, secondary_camera, "secondary_camera"),
        (goal_image, goal_camera, "goal_camera"),
    ):
        if image is not None:
            images.append((_safe_camera_name(key, name=field_name), image))
    observation = RobotObservation(
        task=task,
        state=state,
        images=tuple(images),
        timestamp=timestamp,
        history_fps=history_fps,
        metadata=(
            ("embodiment", embodiment.name if embodiment else "unspecified"),
            ("goal_conditioned", goal_image is not None),
        ),
    )
    if embodiment is not None:
        unknown = [key for key, _image in images if key not in embodiment.camera_names]
        if unknown:
            raise ValueError(
                f"Camera names {unknown} are not declared by embodiment "
                f"{embodiment.name!r}: {list(embodiment.camera_names)}."
            )
    return observation


def _loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    value = hostname.rstrip(".").lower()
    if value == "localhost" or value.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def validate_policy_url(value: str, *, allow_remote: bool) -> str:
    raw = str(value or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Policy endpoint must be an absolute HTTP(S) URL.")
    if parsed.username or parsed.password:
        raise ValueError("Policy endpoint must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("Policy endpoint must not contain a query string or fragment.")
    loopback = _loopback_host(parsed.hostname)
    if parsed.scheme.lower() == "http" and not loopback:
        raise ValueError("Remote policy endpoints must use HTTPS.")
    if not loopback and not allow_remote:
        raise ValueError(
            "Remote policy access is disabled. Enable allow_remote only for a "
            "trusted HTTPS policy server."
        )
    path = (parsed.path or "").rstrip("/")
    if not path:
        path = "/v1/infer"
    elif not path.endswith("/v1/infer"):
        path += "/v1/infer"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def validate_ws_url(value: str, *, allow_remote: bool) -> str:
    raw = str(value or "").strip()
    if "://" not in raw:
        raw = f"ws://{raw}"
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("OpenPI endpoint must be an absolute ws:// or wss:// URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("OpenPI endpoint must not contain credentials, query, or fragment.")
    loopback = _loopback_host(parsed.hostname)
    if parsed.scheme.lower() == "ws" and not loopback:
        raise ValueError("Remote OpenPI endpoints must use WSS.")
    if not loopback and not allow_remote:
        raise ValueError("Remote OpenPI access is disabled.")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path, "", ""))


def validate_zmq_host(value: str, *, allow_remote: bool) -> str:
    host = str(value or "").strip()
    if not host or len(host) > 253:
        raise ValueError("GR00T host is invalid.")
    if "://" in host or any(char in host for char in "/?#@"):
        raise ValueError("GR00T host must be a hostname or IP address only.")
    if not _loopback_host(host) and not allow_remote:
        raise ValueError("Remote GR00T access is disabled.")
    return host


def _redact(text: Any, secrets: Sequence[str] = ()) -> str:
    result = str(text)
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    result = re.sub(
        r"(?i)(authorization|api[-_ ]?key|token)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        result,
    )
    return result[:MAX_ERROR_CHARS]


def _image_png_payload(image: torch.Tensor) -> tuple[str, int]:
    pil = tensor_to_pil(image)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=92, optimize=True)
    payload = buffer.getvalue()
    if len(payload) > MAX_WIRE_IMAGE_BYTES:
        raise ValueError(
            f"Encoded observation image is {len(payload)} bytes; limit is "
            f"{MAX_WIRE_IMAGE_BYTES}. Resize it with VLM Image Pixel Budget."
        )
    return base64.b64encode(payload).decode("ascii"), len(payload)


def observation_to_http_payload(
    observation: RobotObservation,
    *,
    include_history: bool,
) -> dict[str, Any]:
    total_bytes = 0
    cameras = {}
    for key, images in observation.images:
        indices = range(int(images.shape[0])) if include_history else (int(images.shape[0]) - 1,)
        frames = []
        for index in indices:
            encoded, byte_count = _image_png_payload(images[index])
            total_bytes += byte_count
            if total_bytes > MAX_TOTAL_WIRE_IMAGE_BYTES:
                raise ValueError(
                    "Encoded observation images exceed the total wire limit. "
                    "Reduce history or pixels before policy inference."
                )
            frames.append(
                {
                    "encoding": "base64-jpeg",
                    "data": encoded,
                    "frame_index": int(index),
                }
            )
        cameras[key] = frames
    return {
        "schema": OBSERVATION_SCHEMA,
        "version": ROBOTICS_SCHEMA_VERSION,
        "task": observation.task,
        "state": list(observation.state),
        "timestamp": observation.timestamp,
        "history_fps": observation.history_fps,
        "cameras": cameras,
    }


def _normalize_action_array(value: Any, *, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    except Exception as exc:
        raise TypeError(f"Action stream {name!r} is not a numeric array.") from exc
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Action stream {name!r} has batch size {tensor.shape[0]}; "
                "ComfyUI robotics nodes currently require batch size 1."
            )
        tensor = tensor[0]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Action stream {name!r} must have [horizon, dim] or [1, horizon, dim] shape."
        )
    return tensor


_ACTION_METADATA_KEYS = {
    "info",
    "metadata",
    "server_timing",
    "policy_timing",
    "timing",
    "latency_ms",
}


def actions_from_response(
    response: Any,
    *,
    source: str,
    action_names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RobotActions:
    if (
        isinstance(response, (list, tuple))
        and len(response) == 2
        and isinstance(response[0], Mapping)
    ):
        response = response[0]
    if isinstance(response, Mapping):
        for key in ("actions", "action"):
            if key in response:
                candidate = response[key]
                break
        else:
            candidate = {
                key: value
                for key, value in response.items()
                if key not in _ACTION_METADATA_KEYS
            }
    else:
        candidate = response

    slices: list[tuple[str, int, int]] = []
    if isinstance(candidate, Mapping):
        streams = []
        offset = 0
        horizon = None
        for stream_name, stream_value in candidate.items():
            tensor = _normalize_action_array(stream_value, name=str(stream_name))
            if horizon is None:
                horizon = int(tensor.shape[0])
            elif int(tensor.shape[0]) != horizon:
                raise ValueError("All action streams must have the same horizon.")
            streams.append(tensor)
            end = offset + int(tensor.shape[1])
            slices.append((str(stream_name), offset, end))
            offset = end
        if not streams:
            raise ValueError("Policy response contains no action streams.")
        values = torch.cat(streams, dim=1)
    else:
        values = _normalize_action_array(candidate, name="actions")
        slices = [("actions", 0, int(values.shape[1]))]

    if action_names is None:
        generated = []
        for stream, start, end in slices:
            generated.extend(f"{stream}[{index}]" for index in range(end - start))
        names = tuple(generated)
    else:
        names = tuple(str(name) for name in action_names)
    return RobotActions(
        values=values,
        action_names=names,
        source=source,
        stream_slices=tuple(slices),
        metadata=tuple((str(key), value) for key, value in (metadata or {}).items()),
    )


def actions_from_json(value: str) -> RobotActions:
    parsed = _parse_json(value, name="actions_json")
    if isinstance(parsed, Mapping) and parsed.get("schema") == ACTIONS_SCHEMA:
        version = int(parsed.get("version", 0))
        if version != ROBOTICS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported robot-actions schema version {version}.")
        names = parsed.get("action_names")
        streams = tuple(
            (str(item["name"]), int(item["start"]), int(item["end"]))
            for item in parsed.get("streams", [])
        )
        result = actions_from_response(
            parsed["actions"],
            source=str(parsed.get("source", "json")),
            action_names=names,
            metadata=parsed.get("metadata", {}),
        )
        if streams:
            result = RobotActions(
                values=result.values,
                action_names=result.action_names,
                source=result.source,
                stream_slices=streams,
                metadata=result.metadata,
            )
        return result
    return actions_from_response(parsed, source="json")


def blend_action_chunks(
    previous: RobotActions,
    new: RobotActions,
    *,
    executed_steps: int,
    transition_steps: int,
    max_horizon: int,
) -> tuple[RobotActions, dict[str, Any]]:
    if not isinstance(previous, RobotActions) or not isinstance(new, RobotActions):
        raise TypeError("previous and new must both be VLA action trajectories.")
    if previous.action_dim != new.action_dim:
        raise ValueError("Action chunk dimensions do not match.")
    if previous.action_names != new.action_names:
        raise ValueError("Action chunk names/order do not match.")
    executed = int(executed_steps)
    if not 0 <= executed <= previous.horizon:
        raise ValueError("executed_steps is outside the previous action horizon.")
    transition = int(transition_steps)
    if transition < 0:
        raise ValueError("transition_steps must be non-negative.")
    horizon = int(max_horizon)
    if not 1 <= horizon <= MAX_ACTION_HORIZON:
        raise ValueError(f"max_horizon must be in [1, {MAX_ACTION_HORIZON}].")
    output = new.values[:horizon].clone()
    remaining = previous.values[executed:]
    overlap = min(transition, int(remaining.shape[0]), int(output.shape[0]))
    for index in range(overlap):
        # Begin near the already-planned command and continuously hand control
        # to the new plan. With one transition step, average the two.
        alpha = (index + 1) / (overlap + 1)
        output[index] = remaining[index] * (1.0 - alpha) + output[index] * alpha
    blended = RobotActions(
        values=output,
        action_names=new.action_names,
        source=f"replan:{new.source}",
        stream_slices=new.stream_slices,
        metadata=(
            *new.metadata,
            ("previous_source", previous.source),
            ("executed_steps", executed),
            ("transition_steps", overlap),
        ),
    )
    report = {
        "schema": "comfyui-vlm/robot-action-replan",
        "version": ROBOTICS_SCHEMA_VERSION,
        "previous_source": previous.source,
        "new_source": new.source,
        "previous_horizon": previous.horizon,
        "new_horizon": new.horizon,
        "executed_steps": executed,
        "remaining_previous_steps": int(remaining.shape[0]),
        "transition_steps_requested": transition,
        "transition_steps_applied": overlap,
        "output_horizon": blended.horizon,
        "note": (
            "Deterministic chunk-boundary smoothing only; it does not replace "
            "a policy runtime's asynchronous or real-time chunking controller."
        ),
    }
    return blended, report


def call_http_policy(
    observation: RobotObservation,
    *,
    endpoint: str,
    timeout_seconds: float,
    allow_remote: bool,
    include_history: bool,
) -> tuple[RobotActions, dict[str, Any]]:
    url = validate_policy_url(endpoint, allow_remote=allow_remote)
    timeout = _finite_float(timeout_seconds, "timeout_seconds")
    if not 0.1 <= timeout <= 600:
        raise ValueError("timeout_seconds must be between 0.1 and 600.")
    token = os.environ.get(TOKEN_ENV_BY_PROTOCOL["http"], "").strip()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "ComfyUI-VLM-Nodes/robotics",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = observation_to_http_payload(observation, include_history=include_history)
    httpx = require_module("httpx")
    started = time.perf_counter()
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client, client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            declared_length = response.headers.get("content-length")
            if (
                declared_length is not None
                and int(declared_length) > MAX_HTTP_RESPONSE_BYTES
            ):
                raise RuntimeError(
                    "Policy response exceeds the 32 MiB safety limit."
                )
            content = bytearray()
            for chunk in response.iter_bytes():
                if len(content) + len(chunk) > MAX_HTTP_RESPONSE_BYTES:
                    raise RuntimeError(
                        "Policy response exceeds the 32 MiB safety limit."
                    )
                content.extend(chunk)
            result = json.loads(content)
    except Exception as exc:
        raise RuntimeError(
            f"Policy request failed: {_redact(exc, (token,))}"
        ) from None
    elapsed_ms = (time.perf_counter() - started) * 1000
    names = result.get("action_names") if isinstance(result, Mapping) else None
    actions = actions_from_response(
        result,
        source=f"http:{urlsplit(url).hostname}",
        action_names=names,
        metadata={"client_latency_ms": elapsed_ms},
    )
    report = {
        "protocol": "comfyui-vla-http-v1",
        "endpoint": f"{urlsplit(url).scheme}://{urlsplit(url).netloc}",
        "client_latency_ms": elapsed_ms,
        "horizon": actions.horizon,
        "action_dim": actions.action_dim,
        "history_sent": include_history,
        "authenticated": bool(token),
        "server_timing": _json_safe(
            result.get("server_timing", {}) if isinstance(result, Mapping) else {}
        ),
        "policy": _json_safe(
            result.get("policy", {}) if isinstance(result, Mapping) else {}
        ),
    }
    return actions, report


def _observation_image_uint8(
    image: torch.Tensor,
    *,
    layout: str,
) -> np.ndarray:
    pil = tensor_to_pil(image, int(image.shape[0]) - 1)
    array = np.asarray(pil, dtype=np.uint8)
    if layout == "chw":
        return np.ascontiguousarray(array.transpose(2, 0, 1))
    return np.ascontiguousarray(array)


def observation_to_openpi_payload(
    observation: RobotObservation,
    *,
    layout: str,
    state_key: str,
    prompt_key: str,
) -> dict[str, Any]:
    clean_state_key = _safe_camera_name(state_key, name="state_key")
    clean_prompt_key = _safe_camera_name(prompt_key, name="prompt_key")
    if layout == "Nested images (ALOHA)":
        return {
            clean_state_key: np.asarray(observation.state, dtype=np.float32),
            "images": {
                key: _observation_image_uint8(image, layout="chw")
                for key, image in observation.images
            },
            clean_prompt_key: observation.task,
        }
    if layout == "Flat keys (DROID / LIBERO)":
        result: dict[str, Any] = {
            clean_state_key: np.asarray(observation.state, dtype=np.float32),
            clean_prompt_key: observation.task,
        }
        result.update(
            {
                key: _observation_image_uint8(image, layout="hwc")
                for key, image in observation.images
            }
        )
        return result
    raise ValueError(f"Unknown OpenPI observation layout {layout!r}.")


def _openpi_pack_array(value: Any) -> Any:
    if (
        isinstance(value, (np.ndarray, np.generic))
        and value.dtype.kind in {"V", "O", "c"}
    ):
        raise ValueError(f"OpenPI protocol does not support dtype {value.dtype}.")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"OpenPI protocol cannot encode {type(value).__name__}.")


def _openpi_unpack_array(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if b"__ndarray__" in value or "__ndarray__" in value:
        marker = b"__ndarray__" if b"__ndarray__" in value else "__ndarray__"
        data_key = b"data" if b"data" in value else "data"
        dtype_key = b"dtype" if b"dtype" in value else "dtype"
        shape_key = b"shape" if b"shape" in value else "shape"
        if not value.get(marker):
            return value
        dtype = np.dtype(value[dtype_key])
        if dtype.kind in {"V", "O", "c"}:
            raise ValueError(f"Refusing unsafe OpenPI response dtype {dtype}.")
        shape = tuple(int(item) for item in value[shape_key])
        if math.prod(shape) > MAX_ACTION_HORIZON * MAX_ACTION_DIM * 16:
            raise ValueError("OpenPI response array exceeds the safety limit.")
        return np.frombuffer(value[data_key], dtype=dtype).reshape(shape).copy()
    if b"__npgeneric__" in value or "__npgeneric__" in value:
        data_key = b"data" if b"data" in value else "data"
        dtype_key = b"dtype" if b"dtype" in value else "dtype"
        dtype = np.dtype(value[dtype_key])
        if dtype.kind in {"V", "O", "c"}:
            raise ValueError(f"Refusing unsafe OpenPI response dtype {dtype}.")
        return dtype.type(value[data_key])
    return value


def call_openpi_policy(
    observation: RobotObservation,
    *,
    endpoint: str,
    timeout_seconds: float,
    allow_remote: bool,
    layout: str,
    state_key: str,
    prompt_key: str,
) -> tuple[RobotActions, dict[str, Any]]:
    uri = validate_ws_url(endpoint, allow_remote=allow_remote)
    timeout = _finite_float(timeout_seconds, "timeout_seconds")
    if not 0.1 <= timeout <= 600:
        raise ValueError("timeout_seconds must be between 0.1 and 600.")
    websockets = require_module("websockets.sync.client", "websockets")
    msgpack = require_module("msgpack")
    token = os.environ.get(TOKEN_ENV_BY_PROTOCOL["openpi"], "").strip()
    headers = {"Authorization": f"Api-Key {token}"} if token else None
    payload = observation_to_openpi_payload(
        observation,
        layout=layout,
        state_key=state_key,
        prompt_key=prompt_key,
    )
    started = time.perf_counter()
    try:
        with websockets.connect(
            uri,
            compression=None,
            max_size=MAX_HTTP_RESPONSE_BYTES,
            additional_headers=headers,
            open_timeout=timeout,
            close_timeout=min(timeout, 10.0),
        ) as connection:
            metadata_bytes = connection.recv(timeout=timeout)
            if isinstance(metadata_bytes, str):
                raise RuntimeError("OpenPI metadata response must be binary MessagePack.")
            server_metadata = msgpack.unpackb(
                metadata_bytes,
                object_hook=_openpi_unpack_array,
            )
            request = msgpack.packb(payload, default=_openpi_pack_array)
            connection.send(request)
            response_bytes = connection.recv(timeout=timeout)
            if isinstance(response_bytes, str):
                raise RuntimeError(_redact(response_bytes, (token,)))
            result = msgpack.unpackb(
                response_bytes,
                object_hook=_openpi_unpack_array,
            )
    except Exception as exc:
        raise RuntimeError(
            f"OpenPI request failed: {_redact(exc, (token,))}"
        ) from None
    elapsed_ms = (time.perf_counter() - started) * 1000
    actions = actions_from_response(
        result,
        source=f"openpi:{urlsplit(uri).hostname}",
        metadata={"client_latency_ms": elapsed_ms},
    )
    report = {
        "protocol": "openpi-websocket",
        "endpoint": f"{urlsplit(uri).scheme}://{urlsplit(uri).netloc}",
        "client_latency_ms": elapsed_ms,
        "horizon": actions.horizon,
        "action_dim": actions.action_dim,
        "authenticated": bool(token),
        "server_metadata": _json_safe(server_metadata),
        "server_timing": _json_safe(
            result.get("server_timing", {}) if isinstance(result, Mapping) else {}
        ),
    }
    return actions, report


def _groot_encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V"}:
            raise TypeError("Refusing object/void ndarray in GR00T request.")
        if value.dtype.kind == "c":
            return {
                b"nd": True,
                b"type": value.dtype.str,
                b"kind": b"c",
                b"shape": value.shape,
                b"real": value.real.tobytes(),
                b"imag": value.imag.tobytes(),
            }
        return {
            b"nd": True,
            b"type": value.dtype.str,
            b"kind": value.dtype.kind.encode(),
            b"shape": value.shape,
            b"data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        return {
            b"nd": False,
            b"type": value.dtype.str,
            b"data": value.item(),
        }
    raise TypeError(f"GR00T protocol cannot encode {type(value).__name__}.")


def _groot_decode(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    nd = value.get(b"nd", value.get("nd"))
    if nd is None:
        return value
    kind = value.get(b"kind", value.get("kind"))
    if kind in (b"O", "O", b"V", "V"):
        raise ValueError("Refusing object/void ndarray in GR00T response.")
    dtype_value = value.get(b"type", value.get("type"))
    data = value.get(b"data", value.get("data"))
    dtype = np.dtype(dtype_value)
    if nd:
        shape = tuple(int(item) for item in value.get(b"shape", value.get("shape")))
        if math.prod(shape) > MAX_ACTION_HORIZON * MAX_ACTION_DIM * 16:
            raise ValueError("GR00T response array exceeds the safety limit.")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    return dtype.type(data)


def observation_to_groot_payload(observation: RobotObservation) -> dict[str, Any]:
    history = observation.history_frames
    video = {}
    total_bytes = 0
    for key, images in observation.images:
        frames = images
        if int(frames.shape[0]) == 1 and history > 1:
            frames = frames.expand(history, *frames.shape[1:])
        arrays = [
            np.asarray(tensor_to_pil(frames, index), dtype=np.uint8)
            for index in range(int(frames.shape[0]))
        ]
        camera = np.ascontiguousarray(np.stack(arrays, axis=0)[None, ...])
        total_bytes += int(camera.nbytes)
        if total_bytes > MAX_NATIVE_MESSAGE_BYTES:
            raise ValueError(
                "GR00T observation images exceed the 128 MiB message limit. "
                "Reduce history or pixels before policy inference."
            )
        video[key] = camera
    state = np.asarray(observation.state, dtype=np.float32)
    state = np.broadcast_to(state, (1, history, len(state))).copy()
    return {
        "video": video,
        "state": {"state": state},
        "language": {"task": [[observation.task]]},
    }


def call_groot_policy(
    observation: RobotObservation,
    *,
    host: str,
    port: int,
    timeout_seconds: float,
    allow_remote: bool,
) -> tuple[RobotActions, dict[str, Any]]:
    clean_host = validate_zmq_host(host, allow_remote=allow_remote)
    clean_port = int(port)
    if not 1 <= clean_port <= 65_535:
        raise ValueError("GR00T port must be in [1, 65535].")
    timeout = _finite_float(timeout_seconds, "timeout_seconds")
    if not 0.1 <= timeout <= 600:
        raise ValueError("timeout_seconds must be between 0.1 and 600.")
    zmq = require_module("zmq", "pyzmq")
    msgpack = require_module("msgpack")
    token = os.environ.get(TOKEN_ENV_BY_PROTOCOL["groot"], "").strip()
    request = {
        "endpoint": "get_action",
        "data": {
            "observation": observation_to_groot_payload(observation),
            "options": None,
        },
    }
    if token:
        request["api_token"] = token
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    timeout_ms = int(timeout * 1000)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    if hasattr(zmq, "MAXMSGSIZE"):
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_HTTP_RESPONSE_BYTES)
    started = time.perf_counter()
    try:
        socket.connect(f"tcp://{clean_host}:{clean_port}")
        request_bytes = msgpack.packb(request, default=_groot_encode)
        if len(request_bytes) > MAX_NATIVE_MESSAGE_BYTES:
            raise ValueError("GR00T request exceeds the 128 MiB message limit.")
        socket.send(request_bytes)
        response_bytes = socket.recv()
        if len(response_bytes) > MAX_HTTP_RESPONSE_BYTES:
            raise RuntimeError("GR00T response exceeds the 32 MiB safety limit.")
        result = msgpack.unpackb(
            response_bytes,
            object_hook=_groot_decode,
            raw=False,
        )
        if isinstance(result, Mapping) and "error" in result:
            raise RuntimeError(result["error"])
    except Exception as exc:
        raise RuntimeError(
            f"GR00T request failed: {_redact(exc, (token,))}"
        ) from None
    finally:
        socket.close(linger=0)
        context.term()
    elapsed_ms = (time.perf_counter() - started) * 1000
    info = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else {}
    actions = actions_from_response(
        result,
        source=f"groot:{clean_host}",
        metadata={"client_latency_ms": elapsed_ms},
    )
    report = {
        "protocol": "isaac-groot-zmq",
        "endpoint": f"tcp://{clean_host}:{clean_port}",
        "client_latency_ms": elapsed_ms,
        "horizon": actions.horizon,
        "action_dim": actions.action_dim,
        "authenticated": bool(token),
        "policy_info": _json_safe(info),
    }
    return actions, report


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        if value.size > 1024:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "omitted": True,
            }
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        if value.numel() > 1024:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "omitted": True,
            }
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _previous_action(
    value: str,
    *,
    dimension: int,
) -> torch.Tensor | None:
    raw = (value or "").strip()
    if not raw:
        return None
    parsed = _parse_json(raw, name="previous_action_json")
    vector = _finite_vector(parsed, name="previous_action", expected=dimension)
    return torch.tensor(vector, dtype=torch.float32)


def validate_action_trajectory(
    actions: RobotActions,
    embodiment: RobotEmbodiment,
    *,
    mode: str,
    execution_horizon: int,
    previous_action_json: str = "",
) -> tuple[RobotActions, dict[str, Any]]:
    if not isinstance(actions, RobotActions):
        raise TypeError("actions must be a VLA action trajectory.")
    if not isinstance(embodiment, RobotEmbodiment):
        raise TypeError("embodiment must be a VLA embodiment profile.")
    if mode not in SAFETY_MODES:
        raise ValueError(f"Unknown action safety mode {mode!r}.")
    if actions.action_dim != embodiment.action_dim:
        raise ValueError(
            f"Policy returned {actions.action_dim} action values, but embodiment "
            f"{embodiment.name!r} requires {embodiment.action_dim}."
        )
    horizon = int(execution_horizon)
    if horizon < 1:
        raise ValueError("execution_horizon must be at least 1.")
    horizon = min(horizon, actions.horizon)
    original = actions.values[:horizon].clone()
    result = original.clone()
    low = torch.tensor(embodiment.action_min, dtype=torch.float32)
    high = torch.tensor(embodiment.action_max, dtype=torch.float32)
    delta_limit = torch.tensor(embodiment.max_delta, dtype=torch.float32)
    previous = _previous_action(
        previous_action_json,
        dimension=embodiment.action_dim,
    )

    finite_mask = torch.isfinite(original)
    nonfinite_count = int((~finite_mask).sum().item())
    fallback = previous if previous is not None else (low + high) * 0.5
    result = torch.where(finite_mask, result, fallback.unsqueeze(0))
    below = result < low
    above = result > high
    bounds_count = int((below | above).sum().item())

    delta_count = 0
    running_previous = previous
    for index in range(horizon):
        if running_previous is not None:
            delta = result[index] - running_previous
            delta_count += int((delta.abs() > delta_limit).sum().item())
        running_previous = result[index]
    violation_count = nonfinite_count + bounds_count + delta_count

    if violation_count and mode == "Block unsafe":
        raise ValueError(
            "Unsafe policy action blocked: "
            f"{nonfinite_count} non-finite, {bounds_count} bounds, "
            f"{delta_count} per-step delta violations."
        )
    if violation_count and mode == "Hold position on unsafe":
        if previous is None:
            raise ValueError(
                "Hold position on unsafe requires previous_action_json with one "
                "current/commanded value per action dimension."
            )
        result = previous.unsqueeze(0).expand(horizon, -1).clone()
    elif mode == "Clamp safely":
        result = torch.maximum(torch.minimum(result, high), low)
        running_previous = previous
        for index in range(horizon):
            if running_previous is not None:
                result[index] = torch.maximum(
                    torch.minimum(result[index], running_previous + delta_limit),
                    running_previous - delta_limit,
                )
                result[index] = torch.maximum(torch.minimum(result[index], high), low)
            running_previous = result[index].clone()
    elif mode == "Report only":
        result = original

    changed = not torch.equal(result, original)
    finite_after = bool(torch.isfinite(result).all().item())
    in_bounds_after = (
        bool(((result >= low) & (result <= high)).all().item())
        if finite_after
        else False
    )
    safe_after = finite_after and in_bounds_after
    if safe_after:
        running_previous = previous
        for index in range(horizon):
            if running_previous is not None and bool(
                ((result[index] - running_previous).abs() > delta_limit).any().item()
            ):
                safe_after = False
                break
            running_previous = result[index]
    validated = RobotActions(
        values=result,
        action_names=embodiment.action_names,
        source=actions.source,
        stream_slices=actions.stream_slices,
        metadata=(
            *actions.metadata,
            ("safety_mode", mode),
            ("embodiment", embodiment.name),
            ("validated", safe_after),
        ),
    )
    report = {
        "schema": "comfyui-vlm/robot-action-safety",
        "version": ROBOTICS_SCHEMA_VERSION,
        "embodiment": embodiment.name,
        "action_mode": embodiment.action_mode,
        "policy_source": actions.source,
        "input_horizon": actions.horizon,
        "execution_horizon": horizon,
        "action_dim": actions.action_dim,
        "mode": mode,
        "previous_action_provided": previous is not None,
        "violations": {
            "non_finite_values": nonfinite_count,
            "out_of_bounds_values": bounds_count,
            "per_step_delta_values": delta_count,
            "total": violation_count,
        },
        "changed": changed,
        "safe_for_handoff": safe_after,
        "warning": (
            "This is a workflow safety gate, not a certified robot safety "
            "controller. A hardware deadman, watchdog, collision limits, and "
            "manufacturer controller remain mandatory."
        ),
    }
    return validated, report


def render_action_preview(
    actions: RobotActions,
    *,
    embodiment: RobotEmbodiment | None = None,
    width: int = 960,
    height: int = 480,
) -> torch.Tensor:
    width = int(width)
    height = int(height)
    if width < 320 or height < 240:
        raise ValueError("Preview dimensions must be at least 320x240.")
    canvas = Image.new("RGB", (width, height), (18, 21, 28))
    draw = ImageDraw.Draw(canvas)
    margin_left, margin_right, margin_top, margin_bottom = 72, 24, 42, 48
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    values = actions.values
    finite = torch.isfinite(values)
    plot_values = torch.where(finite, values, torch.zeros_like(values))
    if embodiment is not None and embodiment.action_dim == actions.action_dim:
        global_low = min(embodiment.action_min)
        global_high = max(embodiment.action_max)
    else:
        global_low = float(plot_values.min().item())
        global_high = float(plot_values.max().item())
        if math.isclose(global_low, global_high):
            global_low -= 1.0
            global_high += 1.0
    span = max(global_high - global_low, 1.0e-6)
    draw.text(
        (margin_left, 14),
        f"{actions.source} — {actions.horizon} × {actions.action_dim}",
        fill=(235, 238, 245),
    )
    draw.rectangle(
        (
            margin_left,
            margin_top,
            margin_left + chart_width,
            margin_top + chart_height,
        ),
        outline=(87, 94, 111),
        width=1,
    )
    for tick in range(5):
        y = margin_top + round(chart_height * tick / 4)
        value = global_high - span * tick / 4
        draw.line((margin_left, y, margin_left + chart_width, y), fill=(47, 53, 65))
        draw.text((6, y - 7), f"{value:.3g}", fill=(165, 171, 184))
    palette = (
        (96, 165, 250),
        (244, 114, 182),
        (52, 211, 153),
        (251, 191, 36),
        (167, 139, 250),
        (248, 113, 113),
        (34, 211, 238),
        (163, 230, 53),
    )
    for dimension in range(actions.action_dim):
        color = palette[dimension % len(palette)]
        points = []
        for step in range(actions.horizon):
            x = (
                margin_left + chart_width // 2
                if actions.horizon == 1
                else margin_left + round(chart_width * step / (actions.horizon - 1))
            )
            y = margin_top + round(
                chart_height
                * (global_high - float(plot_values[step, dimension].item()))
                / span
            )
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        else:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    if not bool(finite.all().item()):
        draw.text(
            (margin_left, height - 28),
            "WARNING: raw trajectory contains NaN or infinity",
            fill=(248, 113, 113),
        )
    else:
        draw.text(
            (margin_left, height - 28),
            "Preview only — validate with VLA Action Safety before controller handoff",
            fill=(165, 171, 184),
        )
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


class VLAEmbodimentProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (EMBODIMENT_PRESETS, {"default": EMBODIMENT_PRESETS[0]}),
                "control_hz": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10_000.0, "step": 1.0},
                ),
                "state_names_json": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "action_names_json": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "action_min_json": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "action_max_json": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "max_delta_json": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "camera_names_json": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "action_mode_override": (
                    "STRING",
                    {"default": "", "multiline": False, "dynamicPrompts": False},
                ),
            }
        }

    RETURN_TYPES = (VLA_EMBODIMENT, "STRING", "INT", "INT")
    RETURN_NAMES = ("embodiment", "profile_json", "state_dim", "action_dim")
    FUNCTION = "build"
    CATEGORY = "VLM Nodes/Robotics/Schemas"
    DESCRIPTION = (
        "Create an explicit robot state/action/camera contract. Presets are "
        "templates; override their bounds with controller-approved limits."
    )

    def build(
        self,
        preset,
        control_hz,
        state_names_json,
        action_names_json,
        action_min_json,
        action_max_json,
        max_delta_json,
        camera_names_json,
        action_mode_override,
    ):
        profile = make_embodiment(
            preset,
            state_names=_optional_json_array(state_names_json, name="state_names_json"),
            action_names=_optional_json_array(action_names_json, name="action_names_json"),
            action_min=_optional_json_array(action_min_json, name="action_min_json"),
            action_max=_optional_json_array(action_max_json, name="action_max_json"),
            max_delta=_optional_json_array(max_delta_json, name="max_delta_json"),
            control_hz=float(control_hz) if float(control_hz) > 0 else None,
            camera_names=_optional_json_array(camera_names_json, name="camera_names_json"),
            action_mode=(action_mode_override or "").strip() or None,
        )
        return profile, _json(profile.to_dict()), profile.state_dim, profile.action_dim


class VLAObservationBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (
                    "STRING",
                    {
                        "default": "Pick up the object and place it in the container.",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "state_json": (
                    "STRING",
                    {
                        "default": "[0, 0, 0, 0, 0, 0, 0, 0]",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "primary_image": ("IMAGE",),
                "primary_camera": (
                    "STRING",
                    {"default": "observation.images.front", "dynamicPrompts": False},
                ),
                "history_fps": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.01, "max": 1_000.0, "step": 0.1},
                ),
                "timestamp": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0e12, "step": 0.001},
                ),
            },
            "optional": {
                "embodiment": (VLA_EMBODIMENT,),
                "wrist_image": ("IMAGE",),
                "wrist_camera": (
                    "STRING",
                    {"default": "observation.images.wrist", "dynamicPrompts": False},
                ),
                "secondary_image": ("IMAGE",),
                "secondary_camera": (
                    "STRING",
                    {"default": "observation.images.secondary", "dynamicPrompts": False},
                ),
                "goal_image": ("IMAGE",),
                "goal_camera": (
                    "STRING",
                    {"default": "observation.images.goal", "dynamicPrompts": False},
                ),
            },
        }

    RETURN_TYPES = (VLA_OBSERVATION, "STRING", "INT")
    RETURN_NAMES = ("observation", "observation_summary", "history_frames")
    FUNCTION = "build"
    CATEGORY = "VLM Nodes/Robotics/Observations"
    DESCRIPTION = (
        "Build one validated language + state + multi-camera observation. "
        "IMAGE batches become temporal history; no policy or robot SDK is loaded."
    )

    def build(
        self,
        task,
        state_json,
        primary_image,
        primary_camera,
        history_fps,
        timestamp,
        embodiment=None,
        wrist_image=None,
        wrist_camera="observation.images.wrist",
        secondary_image=None,
        secondary_camera="observation.images.secondary",
        goal_image=None,
        goal_camera="observation.images.goal",
    ):
        observation = build_observation(
            task=task,
            state_json=state_json,
            primary_image=primary_image,
            primary_camera=primary_camera,
            history_fps=history_fps,
            timestamp=timestamp,
            embodiment=embodiment,
            wrist_image=wrist_image,
            wrist_camera=wrist_camera,
            secondary_image=secondary_image,
            secondary_camera=secondary_camera,
            goal_image=goal_image,
            goal_camera=goal_camera,
        )
        return observation, _json(observation.summary()), observation.history_frames


class VLAHTTPPolicy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "observation": (VLA_OBSERVATION,),
                "endpoint": (
                    "STRING",
                    {"default": "http://127.0.0.1:8787", "dynamicPrompts": False},
                ),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": 120.0, "min": 0.1, "max": 600.0, "step": 1.0},
                ),
                "include_history": ("BOOLEAN", {"default": True}),
                "allow_remote": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (VLA_ACTIONS, "STRING")
    RETURN_NAMES = ("actions", "inference_report")
    FUNCTION = "infer"
    CATEGORY = "VLM Nodes/Robotics/Policies"
    DESCRIPTION = (
        "Call the isolated universal VLA HTTP sidecar. Remote servers require "
        "HTTPS and explicit opt-in. VLA_POLICY_TOKEN stays server-side."
    )

    def infer(
        self,
        observation,
        endpoint,
        timeout_seconds,
        include_history,
        allow_remote,
    ):
        actions, report = call_http_policy(
            observation,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            allow_remote=allow_remote,
            include_history=include_history,
        )
        return actions, _json(report)


class VLAOpenPIWebSocketPolicy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "observation": (VLA_OBSERVATION,),
                "endpoint": (
                    "STRING",
                    {"default": "ws://127.0.0.1:8000", "dynamicPrompts": False},
                ),
                "observation_layout": (
                    OPENPI_LAYOUTS,
                    {"default": OPENPI_LAYOUTS[0]},
                ),
                "state_key": (
                    "STRING",
                    {"default": "observation/state", "dynamicPrompts": False},
                ),
                "prompt_key": (
                    "STRING",
                    {"default": "prompt", "dynamicPrompts": False},
                ),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": 120.0, "min": 0.1, "max": 600.0, "step": 1.0},
                ),
                "allow_remote": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (VLA_ACTIONS, "STRING")
    RETURN_NAMES = ("actions", "inference_report")
    FUNCTION = "infer"
    CATEGORY = "VLM Nodes/Robotics/Policies"
    DESCRIPTION = (
        "Native client for the official openpi MessagePack WebSocket server. "
        "Install the lightweight robotics client extra; OPENPI_API_KEY is "
        "never stored in workflows."
    )

    def infer(
        self,
        observation,
        endpoint,
        observation_layout,
        state_key,
        prompt_key,
        timeout_seconds,
        allow_remote,
    ):
        actions, report = call_openpi_policy(
            observation,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            allow_remote=allow_remote,
            layout=observation_layout,
            state_key=state_key,
            prompt_key=prompt_key,
        )
        return actions, _json(report)


class VLAGr00tZMQPolicy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "observation": (VLA_OBSERVATION,),
                "host": (
                    "STRING",
                    {"default": "127.0.0.1", "dynamicPrompts": False},
                ),
                "port": (
                    "INT",
                    {"default": 5555, "min": 1, "max": 65_535, "step": 1},
                ),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": 120.0, "min": 0.1, "max": 600.0, "step": 1.0},
                ),
                "allow_remote": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (VLA_ACTIONS, "STRING")
    RETURN_NAMES = ("actions", "inference_report")
    FUNCTION = "infer"
    CATEGORY = "VLM Nodes/Robotics/Policies"
    DESCRIPTION = (
        "Native client for the official Isaac-GR00T N1.7 ZeroMQ policy server. "
        "GROOT_API_TOKEN stays in the environment; no hardware commands are emitted."
    )

    def infer(self, observation, host, port, timeout_seconds, allow_remote):
        actions, report = call_groot_policy(
            observation,
            host=host,
            port=port,
            timeout_seconds=timeout_seconds,
            allow_remote=allow_remote,
        )
        return actions, _json(report)


class VLAActionSafety:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "actions": (VLA_ACTIONS,),
                "embodiment": (VLA_EMBODIMENT,),
                "mode": (SAFETY_MODES, {"default": "Block unsafe"}),
                "execution_horizon": (
                    "INT",
                    {"default": 8, "min": 1, "max": MAX_ACTION_HORIZON, "step": 1},
                ),
                "previous_action_json": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
            }
        }

    RETURN_TYPES = (VLA_ACTIONS, "STRING", "BOOLEAN")
    RETURN_NAMES = ("safe_actions", "safety_report", "safe_for_handoff")
    FUNCTION = "validate"
    CATEGORY = "VLM Nodes/Robotics/Safety"
    DESCRIPTION = (
        "Validate dimensions, finite values, bounds, per-step deltas, and "
        "execution horizon. This does not replace a hardware safety controller."
    )

    def validate(
        self,
        actions,
        embodiment,
        mode,
        execution_horizon,
        previous_action_json,
    ):
        safe_actions, report = validate_action_trajectory(
            actions,
            embodiment,
            mode=mode,
            execution_horizon=execution_horizon,
            previous_action_json=previous_action_json,
        )
        return safe_actions, _json(report), report["safe_for_handoff"]


class VLAActionsFromJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "actions_json": (
                    "STRING",
                    {
                        "default": "[[0, 0, 0, 0, 0, 0, 0, 0]]",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                )
            }
        }

    RETURN_TYPES = (VLA_ACTIONS, "STRING")
    RETURN_NAMES = ("actions", "actions_summary")
    FUNCTION = "parse"
    CATEGORY = "VLM Nodes/Robotics/Actions"
    DESCRIPTION = (
        "Parse a raw numeric action array, named action-stream object, or "
        "versioned VLA action JSON for replay, simulation, and testing."
    )

    def parse(self, actions_json):
        actions = actions_from_json(actions_json)
        return actions, _json(actions.to_dict())


class VLAActionChunkReplan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_actions": (VLA_ACTIONS,),
                "new_actions": (VLA_ACTIONS,),
                "executed_steps": (
                    "INT",
                    {"default": 1, "min": 0, "max": MAX_ACTION_HORIZON, "step": 1},
                ),
                "transition_steps": (
                    "INT",
                    {"default": 2, "min": 0, "max": 128, "step": 1},
                ),
                "max_horizon": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": MAX_ACTION_HORIZON,
                        "step": 1,
                    },
                ),
            }
        }

    RETURN_TYPES = (VLA_ACTIONS, "STRING")
    RETURN_NAMES = ("replanned_actions", "replan_report")
    FUNCTION = "replan"
    CATEGORY = "VLM Nodes/Robotics/Actions"
    DESCRIPTION = (
        "Blend the unexecuted edge of a previous chunk into a new plan to "
        "reduce chunk-boundary discontinuities. Validate the result afterward."
    )

    def replan(
        self,
        previous_actions,
        new_actions,
        executed_steps,
        transition_steps,
        max_horizon,
    ):
        actions, report = blend_action_chunks(
            previous_actions,
            new_actions,
            executed_steps=executed_steps,
            transition_steps=transition_steps,
            max_horizon=max_horizon,
        )
        return actions, _json(report)


class VLAActionInspect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "actions": (VLA_ACTIONS,),
                "step_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_ACTION_HORIZON - 1, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("trajectory_json", "selected_step_json", "horizon", "action_dim")
    FUNCTION = "inspect"
    CATEGORY = "VLM Nodes/Robotics/Actions"
    DESCRIPTION = (
        "Serialize a validated action trajectory and inspect one step. "
        "It never writes to a robot transport."
    )

    def inspect(self, actions, step_index):
        if not isinstance(actions, RobotActions):
            raise TypeError("actions must be a VLA action trajectory.")
        index = int(step_index)
        if not 0 <= index < actions.horizon:
            raise IndexError(
                f"step_index {index} is outside the trajectory horizon {actions.horizon}."
            )
        step = {
            "schema": "comfyui-vlm/robot-action-step",
            "version": ROBOTICS_SCHEMA_VERSION,
            "source": actions.source,
            "step_index": index,
            "action": {
                name: float(value)
                for name, value in zip(
                    actions.action_names,
                    actions.values[index].tolist(),
                    strict=True,
                )
            },
        }
        return (
            _json(actions.to_dict()),
            _json(step),
            actions.horizon,
            actions.action_dim,
        )


class VLATrajectoryPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "actions": (VLA_ACTIONS,),
                "width": ("INT", {"default": 960, "min": 320, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 480, "min": 240, "max": 1536, "step": 16}),
            },
            "optional": {"embodiment": (VLA_EMBODIMENT,)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trajectory_plot",)
    FUNCTION = "render"
    CATEGORY = "VLM Nodes/Robotics/Actions"
    DESCRIPTION = "Render every predicted action dimension as a workflow preview."

    def render(self, actions, width, height, embodiment=None):
        return (render_action_preview(actions, embodiment=embodiment, width=width, height=height),)


class VLAModelCatalog:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (VLA_MODEL_LABELS, {"default": VLA_MODEL_LABELS[0]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_info_json", "checkpoint", "policy_type", "backend")
    FUNCTION = "lookup"
    CATEGORY = "VLM Nodes/Robotics"
    DESCRIPTION = (
        "Curated official VLA runtime/checkpoint map. Entries distinguish "
        "robot-ready fine-tunes from base or architecture-only research models."
    )

    def lookup(self, model):
        info = VLA_MODEL_CATALOG[model]
        return _json(info.to_dict()), info.checkpoint, info.policy_type, info.backend


NODE_CLASS_MAPPINGS = {
    "VLAEmbodimentProfile": VLAEmbodimentProfile,
    "VLAObservationBuilder": VLAObservationBuilder,
    "VLAHTTPPolicy": VLAHTTPPolicy,
    "VLAOpenPIWebSocketPolicy": VLAOpenPIWebSocketPolicy,
    "VLAGr00tZMQPolicy": VLAGr00tZMQPolicy,
    "VLAActionSafety": VLAActionSafety,
    "VLAActionsFromJSON": VLAActionsFromJSON,
    "VLAActionChunkReplan": VLAActionChunkReplan,
    "VLAActionInspect": VLAActionInspect,
    "VLATrajectoryPreview": VLATrajectoryPreview,
    "VLAModelCatalog": VLAModelCatalog,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLAEmbodimentProfile": "VLA Embodiment Profile",
    "VLAObservationBuilder": "VLA Observation Builder",
    "VLAHTTPPolicy": "VLA Policy — Universal HTTP",
    "VLAOpenPIWebSocketPolicy": "VLA Policy — OpenPI WebSocket",
    "VLAGr00tZMQPolicy": "VLA Policy — GR00T N1.7 ZMQ",
    "VLAActionSafety": "VLA Action Safety Gate",
    "VLAActionsFromJSON": "VLA Actions From JSON",
    "VLAActionChunkReplan": "VLA Action Chunk Replan",
    "VLAActionInspect": "VLA Action Inspect",
    "VLATrajectoryPreview": "VLA Trajectory Preview",
    "VLAModelCatalog": "VLA Model Catalog",
}
