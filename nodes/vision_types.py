"""Canonical, immutable spatial payloads shared by VLM vision nodes.

Coordinates use source-image pixels. Bounding boxes are always ``xyxy`` with
an exclusive right/bottom edge. Masks are optional in-process tensors and are
deliberately omitted from every JSON representation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

VLM_DETECTIONS = "VLM_DETECTIONS"
VLM_TRACKS = "VLM_TRACKS"
VLM_POINTS = "VLM_POINTS"
VLM_EVENTS = "VLM_EVENTS"
VLM_VIDEO_SELECTION = "VLM_VIDEO_SELECTION"
VLM_SCENE_STATE = "VLM_SCENE_STATE"

SCHEMA_VERSION = 1
DETECTIONS_SCHEMA = "comfyui-vlm/detections"
TRACKS_SCHEMA = "comfyui-vlm/tracks"
POINTS_SCHEMA = "comfyui-vlm/points"
EVENTS_SCHEMA = "comfyui-vlm/events"
VIDEO_SELECTION_SCHEMA = "comfyui-vlm/video-selection"
SCENE_STATE_SCHEMA = "comfyui-vlm/scene-state"

PointXY = tuple[float, float]
BoxXYXY = tuple[float, float, float, float]
Polygon = tuple[PointXY, ...]


class FrozenDict(Mapping[str, Any]):
    """Small recursively immutable mapping used for metadata."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, values: Mapping[str, Any] | None = None):
        items = []
        for key, value in (values or {}).items():
            if not isinstance(key, str):
                raise TypeError("Metadata keys must be strings.")
            items.append((key, _freeze_json(value)))
        self._items = tuple(sorted(items))
        self._lookup = dict(self._items)

    def __getitem__(self, key: str) -> Any:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._items)!r})"

    def __hash__(self) -> int:
        return hash(self._items)

    def to_dict(self) -> dict[str, Any]:
        return {key: _thaw_json(value) for key, value in self._items}


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Metadata numbers must be finite.")
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"Metadata must contain JSON values, got {type(value).__name__}.")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _metadata(value: Mapping[str, Any] | FrozenDict | None) -> FrozenDict:
    return value if isinstance(value, FrozenDict) else FrozenDict(value)


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _non_negative(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative.")
    return number


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    score = _finite(value, "score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1.")
    return score


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    return value


def _box(value: Any) -> BoxXYXY:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError("bbox_xyxy must contain exactly four numbers.")
    x1, y1, x2, y2 = (
        _non_negative(component, "bbox coordinate") for component in value
    )
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox_xyxy must satisfy x2 >= x1 and y2 >= y1.")
    return x1, y1, x2, y2


def _point(value: Any) -> PointXY:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError("A point must contain exactly two numbers.")
    return (
        _non_negative(value[0], "point x"),
        _non_negative(value[1], "point y"),
    )


def _polygon(
    value: Any,
    *,
    name: str,
    exact_points: int | None = None,
) -> Polygon | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of points.")
    points = tuple(_point(item) for item in value)
    if exact_points is not None and len(points) != exact_points:
        raise ValueError(f"{name} must contain exactly {exact_points} points.")
    if exact_points is None and len(points) < 3:
        raise ValueError(f"{name} must contain at least three points.")
    return points


def _mask(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor or None.")
    if value.ndim != 2:
        raise ValueError("mask must have shape [height, width].")
    return value.detach().to(dtype=torch.float32).clamp(0, 1).clone()


def _base_record(
    *,
    label: str | None,
    text: str | None,
    score: float | None,
    frame_index: int,
    timestamp: float,
    track_id: int | None,
    source: str | None,
    metadata: FrozenDict,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "frame_index": frame_index,
        "timestamp": timestamp,
    }
    if label is not None:
        record["label"] = label
    if text is not None:
        record["text"] = text
    if score is not None:
        record["score"] = score
    if track_id is not None:
        record["track_id"] = track_id
    if source is not None:
        record["source"] = source
    if metadata:
        record["metadata"] = metadata.to_dict()
    return record


@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: BoxXYXY
    label: str | None = None
    text: str | None = None
    score: float | None = None
    polygon: Polygon | None = None
    quad: Polygon | None = None
    frame_index: int = 0
    timestamp: float = 0.0
    track_id: int | None = None
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)
    mask: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_xyxy", _box(self.bbox_xyxy))
        object.__setattr__(self, "label", _optional_text(self.label, "label"))
        object.__setattr__(self, "text", _optional_text(self.text, "text"))
        object.__setattr__(self, "score", _optional_score(self.score))
        object.__setattr__(
            self,
            "polygon",
            _polygon(self.polygon, name="polygon"),
        )
        object.__setattr__(
            self,
            "quad",
            _polygon(self.quad, name="quad", exact_points=4),
        )
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer.")
        object.__setattr__(
            self,
            "timestamp",
            _non_negative(self.timestamp, "timestamp"),
        )
        if self.track_id is not None and (
            not isinstance(self.track_id, int) or self.track_id < 0
        ):
            raise ValueError("track_id must be a non-negative integer or None.")
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "mask", _mask(self.mask))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> PointXY:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def to_dict(self) -> dict[str, Any]:
        record = _base_record(
            label=self.label,
            text=self.text,
            score=self.score,
            frame_index=self.frame_index,
            timestamp=self.timestamp,
            track_id=self.track_id,
            source=self.source,
            metadata=self.metadata,
        )
        record["bbox_xyxy"] = list(self.bbox_xyxy)
        if self.polygon is not None:
            record["polygon"] = [list(point) for point in self.polygon]
        if self.quad is not None:
            record["quad"] = [list(point) for point in self.quad]
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Detection:
        if not isinstance(value, Mapping):
            raise TypeError("A detection must be a JSON object.")
        return cls(
            bbox_xyxy=value["bbox_xyxy"],
            label=value.get("label"),
            text=value.get("text"),
            score=value.get("score"),
            polygon=value.get("polygon"),
            quad=value.get("quad"),
            frame_index=value.get("frame_index", 0),
            timestamp=value.get("timestamp", 0.0),
            track_id=value.get("track_id"),
            source=value.get("source"),
            metadata=value.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class FrameDetections:
    frame_index: int
    timestamp: float
    width: int
    height: int
    detections: tuple[Detection, ...] = ()
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer.")
        object.__setattr__(
            self,
            "timestamp",
            _non_negative(self.timestamp, "timestamp"),
        )
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer.")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer.")
        detections = tuple(self.detections)
        for detection in detections:
            if not isinstance(detection, Detection):
                raise TypeError("detections must contain Detection values.")
            if detection.frame_index != self.frame_index:
                raise ValueError("Detection frame_index does not match its frame.")
            if not math.isclose(detection.timestamp, self.timestamp):
                raise ValueError("Detection timestamp does not match its frame.")
            x1, y1, x2, y2 = detection.bbox_xyxy
            if x1 > self.width or x2 > self.width:
                raise ValueError("Detection x coordinates exceed the frame width.")
            if y1 > self.height or y2 > self.height:
                raise ValueError("Detection y coordinates exceed the frame height.")
            for shape in (detection.polygon, detection.quad):
                if shape is not None and any(
                    x > self.width or y > self.height for x, y in shape
                ):
                    raise ValueError("Detection geometry exceeds the frame bounds.")
            if detection.mask is not None and tuple(detection.mask.shape) != (
                self.height,
                self.width,
            ):
                raise ValueError("Detection mask shape does not match its frame.")
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "detections": [item.to_dict() for item in self.detections],
        }
        if self.metadata:
            record["metadata"] = self.metadata.to_dict()
        return record

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        width: int,
        height: int,
    ) -> FrameDetections:
        if not isinstance(value, Mapping):
            raise TypeError("A frame must be a JSON object.")
        frame_index = value.get("frame_index", 0)
        timestamp = value.get("timestamp", 0.0)
        detections = []
        for record in value.get("detections", []):
            merged = dict(record)
            merged.setdefault("frame_index", frame_index)
            merged.setdefault("timestamp", timestamp)
            detections.append(Detection.from_dict(merged))
        return cls(
            frame_index=frame_index,
            timestamp=timestamp,
            width=width,
            height=height,
            detections=tuple(detections),
            metadata=value.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class DetectionSequence:
    width: int
    height: int
    frames: tuple[FrameDetections, ...] = ()
    frame_count: int = 0
    fps: float | None = None
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported detection schema version {self.version}.")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer.")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer.")
        frames = tuple(self.frames)
        if any(not isinstance(frame, FrameDetections) for frame in frames):
            raise TypeError("frames must contain FrameDetections values.")
        indices = [frame.frame_index for frame in frames]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("Frame indices must be unique and increasing.")
        timestamps = [frame.timestamp for frame in frames]
        if timestamps != sorted(timestamps):
            raise ValueError("Frame timestamps must be increasing.")
        if any(
            frame.width != self.width or frame.height != self.height for frame in frames
        ):
            raise ValueError("Every frame must match the sequence dimensions.")
        frame_count = self.frame_count
        if not isinstance(frame_count, int) or frame_count < 0:
            raise ValueError("frame_count must be a non-negative integer.")
        minimum_count = indices[-1] + 1 if indices else 0
        if frame_count == 0:
            frame_count = minimum_count
        elif frame_count < minimum_count:
            raise ValueError("frame_count is smaller than the largest frame index.")
        fps = None if self.fps is None else _finite(self.fps, "fps")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive.")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def all_detections(self) -> tuple[Detection, ...]:
        return tuple(
            detection for frame in self.frames for detection in frame.detections
        )

    def frame(self, frame_index: int) -> FrameDetections | None:
        return next(
            (frame for frame in self.frames if frame.frame_index == frame_index),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        media: dict[str, Any] = {
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
        }
        if self.fps is not None:
            media["fps"] = self.fps
        record: dict[str, Any] = {
            "schema": DETECTIONS_SCHEMA,
            "version": self.version,
            "media": media,
            "frames": [frame.to_dict() for frame in self.frames],
        }
        if self.source is not None:
            record["source"] = self.source
        if self.metadata:
            record["metadata"] = self.metadata.to_dict()
        return record

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DetectionSequence:
        if not isinstance(value, Mapping):
            raise TypeError("Detection JSON must contain an object.")
        if value.get("schema") != DETECTIONS_SCHEMA:
            raise ValueError(f"Expected schema {DETECTIONS_SCHEMA!r}.")
        if value.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported detection schema version {value.get('version')!r}."
            )
        media = value.get("media")
        if not isinstance(media, Mapping):
            raise ValueError("Detection JSON requires a media object.")
        width, height = media.get("width"), media.get("height")
        frames = tuple(
            FrameDetections.from_dict(frame, width=width, height=height)
            for frame in value.get("frames", [])
        )
        return cls(
            width=width,
            height=height,
            frames=frames,
            frame_count=media.get("frame_count", 0),
            fps=media.get("fps"),
            source=value.get("source"),
            metadata=value.get("metadata"),
            version=value["version"],
        )

    @classmethod
    def from_json(cls, value: str) -> DetectionSequence:
        if not isinstance(value, str):
            raise TypeError("Detection JSON must be a string.")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid detection JSON: {exc.msg}.") from exc
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True)
class VisionPoint:
    x: float
    y: float
    label: str | None = None
    text: str | None = None
    score: float | None = None
    frame_index: int = 0
    timestamp: float = 0.0
    track_id: int | None = None
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _non_negative(self.x, "point x"))
        object.__setattr__(self, "y", _non_negative(self.y, "point y"))
        object.__setattr__(self, "label", _optional_text(self.label, "label"))
        object.__setattr__(self, "text", _optional_text(self.text, "text"))
        object.__setattr__(self, "score", _optional_score(self.score))
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer.")
        object.__setattr__(
            self,
            "timestamp",
            _non_negative(self.timestamp, "timestamp"),
        )
        if self.track_id is not None and (
            not isinstance(self.track_id, int) or self.track_id < 0
        ):
            raise ValueError("track_id must be a non-negative integer or None.")
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        record = _base_record(
            label=self.label,
            text=self.text,
            score=self.score,
            frame_index=self.frame_index,
            timestamp=self.timestamp,
            track_id=self.track_id,
            source=self.source,
            metadata=self.metadata,
        )
        record.update(x=self.x, y=self.y)
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VisionPoint:
        return cls(
            x=value["x"],
            y=value["y"],
            label=value.get("label"),
            text=value.get("text"),
            score=value.get("score"),
            frame_index=value.get("frame_index", 0),
            timestamp=value.get("timestamp", 0.0),
            track_id=value.get("track_id"),
            source=value.get("source"),
            metadata=value.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class PointSequence:
    width: int
    height: int
    points: tuple[VisionPoint, ...] = ()
    frame_count: int = 0
    fps: float | None = None
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported point schema version {self.version}.")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer.")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer.")
        points = tuple(self.points)
        if any(not isinstance(point, VisionPoint) for point in points):
            raise TypeError("points must contain VisionPoint values.")
        if any(point.x > self.width or point.y > self.height for point in points):
            raise ValueError("Point coordinates exceed the sequence bounds.")
        minimum_count = max((point.frame_index for point in points), default=-1) + 1
        frame_count = self.frame_count or minimum_count
        if not isinstance(frame_count, int) or frame_count < minimum_count:
            raise ValueError("frame_count is inconsistent with point frame indices.")
        fps = None if self.fps is None else _finite(self.fps, "fps")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        media: dict[str, Any] = {
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
        }
        if self.fps is not None:
            media["fps"] = self.fps
        result: dict[str, Any] = {
            "schema": POINTS_SCHEMA,
            "version": self.version,
            "media": media,
            "points": [point.to_dict() for point in self.points],
        }
        if self.source is not None:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PointSequence:
        if not isinstance(value, Mapping):
            raise TypeError("Point JSON must contain an object.")
        if value.get("schema") != POINTS_SCHEMA:
            raise ValueError(f"Expected schema {POINTS_SCHEMA!r}.")
        if value.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported point schema version {value.get('version')!r}."
            )
        media = value.get("media")
        if not isinstance(media, Mapping):
            raise ValueError("Point JSON requires a media object.")
        return cls(
            width=media.get("width"),
            height=media.get("height"),
            points=tuple(
                VisionPoint.from_dict(point) for point in value.get("points", [])
            ),
            frame_count=media.get("frame_count", 0),
            fps=media.get("fps"),
            source=value.get("source"),
            metadata=value.get("metadata"),
            version=value["version"],
        )

    @classmethod
    def from_json(cls, value: str) -> PointSequence:
        try:
            return cls.from_dict(json.loads(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid point JSON: {exc.msg}.") from exc


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    detections: tuple[Detection, ...]
    label: str | None = None
    score: float | None = None
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer.")
        detections = tuple(self.detections)
        if not detections:
            raise ValueError("A track requires at least one detection.")
        if any(not isinstance(item, Detection) for item in detections):
            raise TypeError("detections must contain Detection values.")
        indices = [item.frame_index for item in detections]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("Track detections must have increasing unique frames.")
        if any(
            item.track_id is not None and item.track_id != self.track_id
            for item in detections
        ):
            raise ValueError("Detection track_id does not match its Track.")
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "label", _optional_text(self.label, "label"))
        object.__setattr__(self, "score", _optional_score(self.score))
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "track_id": self.track_id,
            "detections": [item.to_dict() for item in self.detections],
        }
        if self.label is not None:
            record["label"] = self.label
        if self.score is not None:
            record["score"] = self.score
        if self.source is not None:
            record["source"] = self.source
        if self.metadata:
            record["metadata"] = self.metadata.to_dict()
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Track:
        if not isinstance(value, Mapping):
            raise TypeError("A track must be a JSON object.")
        return cls(
            track_id=value["track_id"],
            detections=tuple(
                Detection.from_dict(item) for item in value.get("detections", [])
            ),
            label=value.get("label"),
            score=value.get("score"),
            source=value.get("source"),
            metadata=value.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class TrackSequence:
    width: int
    height: int
    tracks: tuple[Track, ...]
    frame_count: int
    fps: float | None = None
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported track schema version {self.version}.")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer.")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer.")
        if not isinstance(self.frame_count, int) or self.frame_count < 0:
            raise ValueError("frame_count must be a non-negative integer.")
        tracks = tuple(self.tracks)
        if any(not isinstance(track, Track) for track in tracks):
            raise TypeError("tracks must contain Track values.")
        ids = [track.track_id for track in tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("Track IDs must be unique.")
        for track in tracks:
            for detection in track.detections:
                if detection.frame_index >= self.frame_count:
                    raise ValueError("Track detection exceeds frame_count.")
                x1, y1, x2, y2 = detection.bbox_xyxy
                if x1 > self.width or x2 > self.width:
                    raise ValueError("Track detection exceeds the frame width.")
                if y1 > self.height or y2 > self.height:
                    raise ValueError("Track detection exceeds the frame height.")
                if detection.mask is not None and tuple(detection.mask.shape) != (
                    self.height,
                    self.width,
                ):
                    raise ValueError("Track mask shape does not match the sequence.")
        fps = None if self.fps is None else _finite(self.fps, "fps")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive.")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        media: dict[str, Any] = {
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
        }
        if self.fps is not None:
            media["fps"] = self.fps
        result: dict[str, Any] = {
            "schema": TRACKS_SCHEMA,
            "version": self.version,
            "media": media,
            "tracks": [track.to_dict() for track in self.tracks],
        }
        if self.source is not None:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrackSequence:
        if not isinstance(value, Mapping):
            raise TypeError("Track JSON must contain an object.")
        if value.get("schema") != TRACKS_SCHEMA:
            raise ValueError(f"Expected schema {TRACKS_SCHEMA!r}.")
        if value.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported track schema version {value.get('version')!r}."
            )
        media = value.get("media")
        if not isinstance(media, Mapping):
            raise ValueError("Track JSON requires a media object.")
        return cls(
            width=media.get("width"),
            height=media.get("height"),
            frame_count=media.get("frame_count", 0),
            fps=media.get("fps"),
            tracks=tuple(Track.from_dict(track) for track in value.get("tracks", [])),
            source=value.get("source"),
            metadata=value.get("metadata"),
            version=value["version"],
        )

    @classmethod
    def from_json(cls, value: str) -> TrackSequence:
        try:
            return cls.from_dict(json.loads(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid track JSON: {exc.msg}.") from exc


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    start_time: float
    end_time: float
    label: str | None = None
    text: str | None = None
    score: float | None = None
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        start = _non_negative(self.start_time, "start_time")
        end = _non_negative(self.end_time, "end_time")
        if end < start:
            raise ValueError("end_time must be greater than or equal to start_time.")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "label", _optional_text(self.label, "label"))
        object.__setattr__(self, "text", _optional_text(self.text, "text"))
        object.__setattr__(self, "score", _optional_score(self.score))
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        if self.label is not None:
            result["label"] = self.label
        if self.text is not None:
            result["text"] = self.text
        if self.score is not None:
            result["score"] = self.score
        if self.source is not None:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TemporalEvent:
        if not isinstance(value, Mapping):
            raise TypeError("An event must be a JSON object.")
        return cls(
            start_time=value["start_time"],
            end_time=value["end_time"],
            label=value.get("label"),
            text=value.get("text"),
            score=value.get("score"),
            source=value.get("source"),
            metadata=value.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class EventSequence:
    events: tuple[TemporalEvent, ...] = ()
    duration: float | None = None
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported event schema version {self.version}.")
        events = tuple(self.events)
        if any(not isinstance(event, TemporalEvent) for event in events):
            raise TypeError("events must contain TemporalEvent values.")
        if list(events) != sorted(
            events,
            key=lambda event: (event.start_time, event.end_time),
        ):
            raise ValueError("Events must be ordered by start_time.")
        duration = (
            None if self.duration is None else _non_negative(self.duration, "duration")
        )
        if duration is not None and any(event.end_time > duration for event in events):
            raise ValueError("An event extends beyond the media duration.")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": EVENTS_SCHEMA,
            "version": self.version,
            "events": [event.to_dict() for event in self.events],
        }
        if self.duration is not None:
            result["duration"] = self.duration
        if self.source is not None:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EventSequence:
        if not isinstance(value, Mapping):
            raise TypeError("Event JSON must contain an object.")
        if value.get("schema") != EVENTS_SCHEMA:
            raise ValueError(f"Expected schema {EVENTS_SCHEMA!r}.")
        if value.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported event schema version {value.get('version')!r}."
            )
        return cls(
            events=tuple(
                TemporalEvent.from_dict(event) for event in value.get("events", [])
            ),
            duration=value.get("duration"),
            source=value.get("source"),
            metadata=value.get("metadata"),
            version=value["version"],
        )

    @classmethod
    def from_json(cls, value: str) -> EventSequence:
        try:
            return cls.from_dict(json.loads(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid event JSON: {exc.msg}.") from exc


@dataclass(frozen=True, slots=True)
class SelectedVideoFrame:
    """One source-frame reference preserved through adaptive sampling."""

    source_frame_index: int
    timestamp: float
    score: float = 0.0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_frame_index, int)
            or self.source_frame_index < 0
        ):
            raise ValueError("source_frame_index must be a non-negative integer.")
        object.__setattr__(
            self,
            "timestamp",
            _non_negative(self.timestamp, "timestamp"),
        )
        score = _finite(self.score, "selection score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("selection score must be between 0 and 1.")
        object.__setattr__(self, "score", score)
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
            raise TypeError("selection reasons must be non-empty strings.")
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_frame_index": self.source_frame_index,
            "timestamp": self.timestamp,
            "score": self.score,
        }
        if self.reasons:
            result["reasons"] = list(self.reasons)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SelectedVideoFrame:
        if not isinstance(value, Mapping):
            raise TypeError("A selected video frame must be a JSON object.")
        return cls(
            source_frame_index=value["source_frame_index"],
            timestamp=value["timestamp"],
            score=value.get("score", 0.0),
            reasons=tuple(value.get("reasons", ())),
        )


@dataclass(frozen=True, slots=True)
class VideoFrameSelection:
    """Immutable map from a sampled IMAGE batch back to its source video."""

    width: int
    height: int
    source_frame_count: int
    fps: float
    frames: tuple[SelectedVideoFrame, ...]
    strategy: str = "adaptive"
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported video selection schema version {self.version}."
            )
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer.")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer.")
        if (
            not isinstance(self.source_frame_count, int)
            or self.source_frame_count <= 0
        ):
            raise ValueError("source_frame_count must be a positive integer.")
        fps = _finite(self.fps, "fps")
        if fps <= 0:
            raise ValueError("fps must be positive.")
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("A video selection requires at least one frame.")
        if any(not isinstance(frame, SelectedVideoFrame) for frame in frames):
            raise TypeError("frames must contain SelectedVideoFrame values.")
        indices = [frame.source_frame_index for frame in frames]
        if indices != sorted(set(indices)):
            raise ValueError(
                "Selected source frame indices must be unique and increasing."
            )
        if indices[-1] >= self.source_frame_count:
            raise ValueError("A selected frame lies outside the source video.")
        expected_timestamps = [index / fps for index in indices]
        if any(
            abs(frame.timestamp - expected) > max(1.0e-6, 0.51 / fps)
            for frame, expected in zip(frames, expected_timestamps)
        ):
            raise ValueError(
                "Selected frame timestamps do not match source indices and fps."
            )
        strategy = str(self.strategy).strip()
        if not strategy:
            raise ValueError("strategy must not be empty.")
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def duration(self) -> float:
        return self.source_frame_count / self.fps

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(frame.source_frame_index for frame in self.frames)

    @property
    def timestamps(self) -> tuple[float, ...]:
        return tuple(frame.timestamp for frame in self.frames)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": VIDEO_SELECTION_SCHEMA,
            "version": self.version,
            "media": {
                "width": self.width,
                "height": self.height,
                "source_frame_count": self.source_frame_count,
                "fps": self.fps,
                "duration": self.duration,
            },
            "strategy": self.strategy,
            "frames": [frame.to_dict() for frame in self.frames],
        }
        if self.source is not None:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VideoFrameSelection:
        if not isinstance(value, Mapping):
            raise TypeError("Video selection JSON must contain an object.")
        if value.get("schema") != VIDEO_SELECTION_SCHEMA:
            raise ValueError(f"Expected schema {VIDEO_SELECTION_SCHEMA!r}.")
        if value.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported video selection schema version "
                f"{value.get('version')!r}."
            )
        media = value.get("media")
        if not isinstance(media, Mapping):
            raise ValueError("Video selection JSON requires a media object.")
        return cls(
            width=media["width"],
            height=media["height"],
            source_frame_count=media["source_frame_count"],
            fps=media["fps"],
            frames=tuple(
                SelectedVideoFrame.from_dict(frame)
                for frame in value.get("frames", ())
            ),
            strategy=value.get("strategy", "adaptive"),
            source=value.get("source"),
            metadata=value.get("metadata"),
            version=value["version"],
        )

    @classmethod
    def from_json(cls, value: str) -> VideoFrameSelection:
        try:
            return cls.from_dict(json.loads(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid video selection JSON: {exc.msg}.") from exc


@dataclass(frozen=True, slots=True)
class SceneObjectState:
    """Compact latest state derived from a temporally consistent object track."""

    track_id: int
    first_seen: float
    last_seen: float
    last_bbox_xyxy: BoxXYXY
    observation_count: int
    label: str | None = None
    state: str = "active"
    mean_confidence: float | None = None
    velocity_xy_px_s: PointXY = (0.0, 0.0)
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer.")
        first_seen = _non_negative(self.first_seen, "first_seen")
        last_seen = _non_negative(self.last_seen, "last_seen")
        if last_seen < first_seen:
            raise ValueError("last_seen must be at or after first_seen.")
        if (
            not isinstance(self.observation_count, int)
            or self.observation_count <= 0
        ):
            raise ValueError("observation_count must be a positive integer.")
        state = str(self.state).strip()
        if not state:
            raise ValueError("state must not be empty.")
        velocity = tuple(_finite(value, "velocity") for value in self.velocity_xy_px_s)
        if len(velocity) != 2:
            raise ValueError("velocity_xy_px_s must contain exactly two values.")
        object.__setattr__(self, "first_seen", first_seen)
        object.__setattr__(self, "last_seen", last_seen)
        object.__setattr__(self, "last_bbox_xyxy", _box(self.last_bbox_xyxy))
        object.__setattr__(self, "label", _optional_text(self.label, "label"))
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "mean_confidence",
            _optional_score(self.mean_confidence),
        )
        object.__setattr__(self, "velocity_xy_px_s", velocity)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "track_id": self.track_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "last_bbox_xyxy": list(self.last_bbox_xyxy),
            "observation_count": self.observation_count,
            "state": self.state,
            "velocity_xy_px_s": list(self.velocity_xy_px_s),
        }
        if self.label is not None:
            result["label"] = self.label
        if self.mean_confidence is not None:
            result["mean_confidence"] = self.mean_confidence
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneObjectState:
        if not isinstance(value, Mapping):
            raise TypeError("A scene object must be a JSON object.")
        return cls(
            track_id=value["track_id"],
            first_seen=value["first_seen"],
            last_seen=value["last_seen"],
            last_bbox_xyxy=value["last_bbox_xyxy"],
            observation_count=value["observation_count"],
            label=value.get("label"),
            state=value.get("state", "active"),
            mean_confidence=value.get("mean_confidence"),
            velocity_xy_px_s=tuple(value.get("velocity_xy_px_s", (0.0, 0.0))),
            metadata=value.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class SceneState:
    """Persistent, serializable world-state summary for video reasoning."""

    width: int
    height: int
    frame_count: int
    fps: float | None
    objects: tuple[SceneObjectState, ...] = ()
    events: tuple[TemporalEvent, ...] = ()
    source: str | None = None
    metadata: FrozenDict = field(default_factory=FrozenDict)
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported scene state schema version {self.version}.")
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer.")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer.")
        if not isinstance(self.frame_count, int) or self.frame_count < 0:
            raise ValueError("frame_count must be a non-negative integer.")
        fps = None if self.fps is None else _finite(self.fps, "fps")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive.")
        objects = tuple(self.objects)
        if any(not isinstance(item, SceneObjectState) for item in objects):
            raise TypeError("objects must contain SceneObjectState values.")
        ids = [item.track_id for item in objects]
        if ids != sorted(set(ids)):
            raise ValueError("Scene objects must have unique increasing track IDs.")
        events = tuple(self.events)
        if any(not isinstance(item, TemporalEvent) for item in events):
            raise TypeError("events must contain TemporalEvent values.")
        if list(events) != sorted(
            events,
            key=lambda event: (event.start_time, event.end_time),
        ):
            raise ValueError("Scene events must be ordered by start_time.")
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "source", _optional_text(self.source, "source"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def duration(self) -> float | None:
        return (
            self.frame_count / self.fps
            if self.fps is not None and self.frame_count
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        media: dict[str, Any] = {
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
        }
        if self.fps is not None:
            media["fps"] = self.fps
        if self.duration is not None:
            media["duration"] = self.duration
        result: dict[str, Any] = {
            "schema": SCENE_STATE_SCHEMA,
            "version": self.version,
            "media": media,
            "objects": [item.to_dict() for item in self.objects],
            "events": [item.to_dict() for item in self.events],
        }
        if self.source is not None:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneState:
        if not isinstance(value, Mapping):
            raise TypeError("Scene state JSON must contain an object.")
        if value.get("schema") != SCENE_STATE_SCHEMA:
            raise ValueError(f"Expected schema {SCENE_STATE_SCHEMA!r}.")
        if value.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported scene state schema version "
                f"{value.get('version')!r}."
            )
        media = value.get("media")
        if not isinstance(media, Mapping):
            raise ValueError("Scene state JSON requires a media object.")
        return cls(
            width=media["width"],
            height=media["height"],
            frame_count=media["frame_count"],
            fps=media.get("fps"),
            objects=tuple(
                SceneObjectState.from_dict(item)
                for item in value.get("objects", ())
            ),
            events=tuple(
                TemporalEvent.from_dict(item)
                for item in value.get("events", ())
            ),
            source=value.get("source"),
            metadata=value.get("metadata"),
            version=value["version"],
        )

    @classmethod
    def from_json(cls, value: str) -> SceneState:
        try:
            return cls.from_dict(json.loads(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid scene state JSON: {exc.msg}.") from exc


__all__ = [
    "BoxXYXY",
    "DETECTIONS_SCHEMA",
    "Detection",
    "DetectionSequence",
    "EVENTS_SCHEMA",
    "EventSequence",
    "FrameDetections",
    "FrozenDict",
    "POINTS_SCHEMA",
    "PointSequence",
    "PointXY",
    "Polygon",
    "SCENE_STATE_SCHEMA",
    "SCHEMA_VERSION",
    "SceneObjectState",
    "SceneState",
    "SelectedVideoFrame",
    "TRACKS_SCHEMA",
    "TemporalEvent",
    "Track",
    "TrackSequence",
    "VLM_DETECTIONS",
    "VLM_EVENTS",
    "VLM_POINTS",
    "VLM_SCENE_STATE",
    "VLM_TRACKS",
    "VLM_VIDEO_SELECTION",
    "VIDEO_SELECTION_SCHEMA",
    "VideoFrameSelection",
    "VisionPoint",
]
