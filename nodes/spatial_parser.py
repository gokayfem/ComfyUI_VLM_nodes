"""Strict, model-agnostic parsing for structured spatial VLM responses."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .vision_types import (
    VLM_DETECTIONS,
    VLM_POINTS,
    Detection,
    DetectionSequence,
    FrameDetections,
    PointSequence,
    VisionPoint,
)

COORDINATE_MODES = ("pixel", "normalized_0_1", "normalized_0_1000")

_FRAME_COLLECTION_KEYS = ("frames", "images", "batches", "batch")
_DETECTION_COLLECTION_KEYS = ("detections", "objects", "predictions")
_BBOX_KEYS = ("bbox_xyxy", "bbox", "box")
_POLYGON_KEYS = ("polygon", "segmentation")
_POINT_KEYS = ("point", "points")
_LABEL_KEYS = ("label", "class", "name")
_SCORE_KEYS = ("score", "confidence")
_TEXT_KEYS = ("text", "description", "caption")

_RECORD_RESERVED_KEYS = frozenset(
    {
        *_BBOX_KEYS,
        *_POLYGON_KEYS,
        "quad",
        *_POINT_KEYS,
        "x",
        "y",
        *_LABEL_KEYS,
        *_SCORE_KEYS,
        *_TEXT_KEYS,
        "frame_index",
        "timestamp",
        "track_id",
        "source",
        "coordinate_mode",
        "metadata",
    }
)
_FRAME_RESERVED_KEYS = frozenset(
    {
        *_DETECTION_COLLECTION_KEYS,
        "items",
        "points",
        "frame_index",
        "timestamp",
        "width",
        "height",
        "coordinate_mode",
        "metadata",
    }
)
_ROOT_RESERVED_KEYS = frozenset(
    {
        *_FRAME_COLLECTION_KEYS,
        *_DETECTION_COLLECTION_KEYS,
        "items",
        "points",
        "media",
        "width",
        "height",
        "frame_count",
        "fps",
        "coordinate_mode",
        "source",
        "metadata",
    }
)
_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n?```[ \t]*\Z",
    re.IGNORECASE,
)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _positive_dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _coordinate_mode(value: Any) -> str:
    if value not in COORDINATE_MODES:
        choices = ", ".join(COORDINATE_MODES)
        raise ValueError(f"coordinate_mode must be one of: {choices}.")
    return str(value)


def _optional_fps(value: Any) -> float | None:
    number = _finite_number(value, "fps")
    if number < 0:
        raise ValueError("fps must be non-negative.")
    return number or None


def _without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed.")


def load_json_document(text: str) -> Mapping[str, Any] | list[Any]:
    """Load a complete JSON document or a complete JSON Markdown fence.

    The function deliberately does not search prose for an embedded object.
    An empty response is treated as an empty result.
    """

    if not isinstance(text, str):
        raise TypeError("response must be a string.")
    stripped = text.strip()
    if not stripped:
        return {}

    if stripped.startswith("```"):
        match = _FENCE.fullmatch(stripped)
        if match is None:
            raise ValueError(
                "A fenced response must contain only one ```json JSON block."
            )
        stripped = match.group("body").strip()
        if not stripped:
            return {}
    elif "```" in stripped:
        raise ValueError("JSON fences cannot be mixed with prose.")

    try:
        parsed = json.loads(
            stripped,
            object_pairs_hook=_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The response must be a complete JSON object or array ({exc.msg})."
        ) from exc
    if not isinstance(parsed, (Mapping, list)):
        raise ValueError("The JSON root must be an object or array.")
    return parsed


def _alias(
    record: Mapping[str, Any],
    names: Sequence[str],
    field: str,
) -> tuple[str | None, Any]:
    found = [(name, record[name]) for name in names if record.get(name) is not None]
    if not found:
        return None, None
    first_name, first_value = found[0]
    if any(value != first_value for _name, value in found[1:]):
        aliases = ", ".join(name for name, _value in found)
        raise ValueError(f"Conflicting aliases for {field}: {aliases}.")
    return first_name, first_value


def _optional_text_alias(
    record: Mapping[str, Any],
    names: Sequence[str],
    field: str,
) -> str | None:
    _name, value = _alias(record, names, field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    return value


def _optional_score(record: Mapping[str, Any]) -> float | None:
    _name, value = _alias(record, _SCORE_KEYS, "score")
    if value is None:
        return None
    score = _finite_number(value, "score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("score/confidence must be between 0 and 1.")
    return score


def _metadata(
    value: Mapping[str, Any],
    reserved: frozenset[str],
) -> dict[str, Any]:
    supplied = value.get("metadata", {})
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise TypeError("metadata must be a JSON object.")
    result = dict(supplied)
    for key, item in value.items():
        if key not in reserved:
            result[key] = item
    return result


def _coordinate_scale(
    coordinate_mode: str,
    width: int,
    height: int,
) -> tuple[float, float]:
    if coordinate_mode == "pixel":
        return 1.0, 1.0
    denominator = 1.0 if coordinate_mode == "normalized_0_1" else 1000.0
    return width / denominator, height / denominator


def _point_xy(
    value: Any,
    *,
    coordinate_mode: str,
    width: int,
    height: int,
    name: str,
) -> tuple[float, float]:
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            raise ValueError(f"{name} must contain x and y.")
        raw_x, raw_y = value["x"], value["y"]
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        raw_x, raw_y = value
    else:
        raise ValueError(f"{name} must contain exactly two coordinates.")

    scale_x, scale_y = _coordinate_scale(coordinate_mode, width, height)
    x = _finite_number(raw_x, f"{name} x") * scale_x
    y = _finite_number(raw_y, f"{name} y") * scale_y
    return (
        min(max(x, 0.0), float(width)),
        min(max(y, 0.0), float(height)),
    )


def _bbox_xyxy(
    value: Any,
    *,
    coordinate_mode: str,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xywh = False
    if isinstance(value, Mapping):
        if all(name in value for name in ("x1", "y1", "x2", "y2")):
            raw = (value["x1"], value["y1"], value["x2"], value["y2"])
        elif all(name in value for name in ("xmin", "ymin", "xmax", "ymax")):
            raw = (
                value["xmin"],
                value["ymin"],
                value["xmax"],
                value["ymax"],
            )
        elif all(name in value for name in ("x", "y", "width", "height")):
            raw = (value["x"], value["y"], value["width"], value["height"])
            xywh = True
        else:
            raise ValueError(
                "A box object must use x1/y1/x2/y2, "
                "xmin/ymin/xmax/ymax, or x/y/width/height."
            )
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
    ):
        raw = tuple(value)
    else:
        raise ValueError("bbox/bbox_xyxy/box must contain four coordinates.")

    x1, y1, third, fourth = (
        _finite_number(component, "bbox coordinate") for component in raw
    )
    if xywh:
        if third < 0 or fourth < 0:
            raise ValueError("Box width and height must be non-negative.")
        x2, y2 = x1 + third, y1 + fourth
    else:
        x2, y2 = third, fourth
    if x2 < x1 or y2 < y1:
        raise ValueError("A box must satisfy x2 >= x1 and y2 >= y1.")

    scale_x, scale_y = _coordinate_scale(coordinate_mode, width, height)
    return (
        min(max(x1 * scale_x, 0.0), float(width)),
        min(max(y1 * scale_y, 0.0), float(height)),
        min(max(x2 * scale_x, 0.0), float(width)),
        min(max(y2 * scale_y, 0.0), float(height)),
    )


def _polygon_points(
    value: Any,
    *,
    coordinate_mode: str,
    width: int,
    height: int,
    exact_points: int | None = None,
    name: str = "polygon",
) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a coordinate array.")
    values: Sequence[Any] = value
    if (
        len(values) == 1
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], (str, bytes))
        and values[0]
        and isinstance(values[0][0], Real)
    ):
        values = values[0]
    if values and all(
        isinstance(item, Real) and not isinstance(item, bool) for item in values
    ):
        if len(values) % 2:
            raise ValueError(f"{name} has an odd number of coordinates.")
        values = [
            (values[index], values[index + 1]) for index in range(0, len(values), 2)
        ]
    points = tuple(
        _point_xy(
            item,
            coordinate_mode=coordinate_mode,
            width=width,
            height=height,
            name=f"{name} point",
        )
        for item in values
    )
    if exact_points is not None and len(points) != exact_points:
        raise ValueError(f"{name} must contain exactly {exact_points} points.")
    if exact_points is None and len(points) < 3:
        raise ValueError(f"{name} must contain at least three points.")
    return points


def _point_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("point/points must contain coordinates.")
    if len(value) == 2 and all(
        isinstance(item, Real) and not isinstance(item, bool) for item in value
    ):
        return [value]
    return list(value)


def _validate_declared_mode(
    value: Mapping[str, Any],
    coordinate_mode: str,
) -> None:
    declared = value.get("coordinate_mode")
    if declared is not None and declared != coordinate_mode:
        raise ValueError(
            f"JSON declares coordinate_mode {declared!r}, "
            f"but the parser is set to {coordinate_mode!r}."
        )


def _frame_index(value: Mapping[str, Any], default: int) -> int:
    return _non_negative_integer(value.get("frame_index", default), "frame_index")


def _frame_timestamp(
    value: Mapping[str, Any],
    frame_index: int,
    fps: float | None,
) -> float:
    default = frame_index / fps if fps is not None else 0.0
    timestamp = _finite_number(value.get("timestamp", default), "timestamp")
    if timestamp < 0:
        raise ValueError("timestamp must be non-negative.")
    return timestamp


def _record_context(
    record: Mapping[str, Any],
    *,
    frame_index: int,
    timestamp: float,
    default_source: str | None,
    coordinate_mode: str,
) -> dict[str, Any]:
    _validate_declared_mode(record, coordinate_mode)
    declared_index = _frame_index(record, frame_index)
    if declared_index != frame_index:
        raise ValueError("A record frame_index must match its containing frame.")
    declared_timestamp = _finite_number(
        record.get("timestamp", timestamp),
        "timestamp",
    )
    if declared_timestamp < 0 or not math.isclose(declared_timestamp, timestamp):
        raise ValueError("A record timestamp must match its containing frame.")
    track_id = record.get("track_id")
    if track_id is not None:
        track_id = _non_negative_integer(track_id, "track_id")
    source = record.get("source", default_source)
    if source is not None and not isinstance(source, str):
        raise TypeError("source must be a string.")
    return {
        "label": _optional_text_alias(record, _LABEL_KEYS, "label"),
        "text": _optional_text_alias(record, _TEXT_KEYS, "text"),
        "score": _optional_score(record),
        "frame_index": frame_index,
        "timestamp": timestamp,
        "track_id": track_id,
        "source": source,
        "metadata": _metadata(record, _RECORD_RESERVED_KEYS),
    }


def _parse_record(
    record: Mapping[str, Any],
    *,
    frame_index: int,
    timestamp: float,
    width: int,
    height: int,
    coordinate_mode: str,
    default_source: str | None,
) -> tuple[Detection | None, list[VisionPoint]]:
    if not isinstance(record, Mapping):
        raise TypeError("Each spatial record must be a JSON object.")
    context = _record_context(
        record,
        frame_index=frame_index,
        timestamp=timestamp,
        default_source=default_source,
        coordinate_mode=coordinate_mode,
    )

    _bbox_name, bbox_value = _alias(record, _BBOX_KEYS, "bbox")
    _polygon_name, polygon_value = _alias(record, _POLYGON_KEYS, "polygon")
    quad_value = record.get("quad")
    if polygon_value is not None and quad_value is not None:
        raise ValueError("Use either polygon/segmentation or quad, not both.")

    polygon = (
        _polygon_points(
            polygon_value,
            coordinate_mode=coordinate_mode,
            width=width,
            height=height,
        )
        if polygon_value is not None
        else None
    )
    quad = (
        _polygon_points(
            quad_value,
            coordinate_mode=coordinate_mode,
            width=width,
            height=height,
            exact_points=4,
            name="quad",
        )
        if quad_value is not None
        else None
    )
    bbox = (
        _bbox_xyxy(
            bbox_value,
            coordinate_mode=coordinate_mode,
            width=width,
            height=height,
        )
        if bbox_value is not None
        else None
    )
    shape = polygon or quad
    if bbox is None and shape is not None:
        bbox = (
            min(point[0] for point in shape),
            min(point[1] for point in shape),
            max(point[0] for point in shape),
            max(point[1] for point in shape),
        )

    point_key, point_value = _alias(record, _POINT_KEYS, "point")
    if point_value is None and ("x" in record or "y" in record):
        if "x" not in record or "y" not in record:
            raise ValueError("A direct point record requires both x and y.")
        point_key, point_value = "xy", {"x": record["x"], "y": record["y"]}
    points = []
    if point_value is not None:
        for value in _point_values(point_value):
            x, y = _point_xy(
                value,
                coordinate_mode=coordinate_mode,
                width=width,
                height=height,
                name=point_key or "point",
            )
            points.append(VisionPoint(x=x, y=y, **context))

    detection = (
        Detection(
            bbox_xyxy=bbox,
            polygon=polygon,
            quad=quad,
            **context,
        )
        if bbox is not None
        else None
    )
    if detection is None and not points:
        raise ValueError(
            "A spatial record requires bbox/bbox_xyxy/box, "
            "polygon/segmentation/quad, or point/points."
        )
    return detection, points


def _records(value: Any, name: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON object or array.")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f"Every {name} item must be a JSON object.")
    return list(value)


def _point_records(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    if (
        isinstance(values, list)
        and len(values) == 2
        and all(
            isinstance(item, Real) and not isinstance(item, bool) for item in values
        )
    ):
        values = [values]
    records = []
    for item in values:
        if isinstance(item, Mapping):
            records.append(item)
        else:
            records.append({"point": item})
    return records


def _frame_records(
    frame: Mapping[str, Any] | list[Any],
    *,
    default_index: int,
    width: int,
    height: int,
    coordinate_mode: str,
    fps: float | None,
    default_source: str | None,
) -> tuple[FrameDetections, list[VisionPoint]]:
    if isinstance(frame, list):
        frame = {"frame_index": default_index, "items": frame}
    if not isinstance(frame, Mapping):
        raise TypeError("Each frame/batch entry must be an object or array.")
    _validate_declared_mode(frame, coordinate_mode)
    for dimension_name, expected in (("width", width), ("height", height)):
        if dimension_name in frame:
            declared = _positive_dimension(frame[dimension_name], dimension_name)
            if declared != expected:
                raise ValueError(
                    f"Frame {dimension_name} {declared} does not match {expected}."
                )

    index = _frame_index(frame, default_index)
    timestamp = _frame_timestamp(frame, index, fps)
    container_names = [
        name for name in _DETECTION_COLLECTION_KEYS if frame.get(name) is not None
    ]
    if len(container_names) > 1:
        raise ValueError(
            "A frame cannot use multiple detection collection aliases: "
            + ", ".join(container_names)
            + "."
        )
    has_record_collection = bool(container_names) or frame.get("items") is not None
    records = (
        _records(frame[container_names[0]], container_names[0])
        if container_names
        else []
    )
    records.extend(_records(frame.get("items"), "items"))

    geometry_keys = {
        *_BBOX_KEYS,
        *_POLYGON_KEYS,
        "quad",
        "point",
        "x",
        "y",
        *_LABEL_KEYS,
        *_SCORE_KEYS,
        *_TEXT_KEYS,
        "track_id",
    }
    direct_record = not has_record_collection and any(
        frame.get(key) is not None for key in geometry_keys
    )
    if direct_record:
        records.append(frame)
    else:
        records.extend(_point_records(frame.get("points")))

    detections = []
    points = []
    for record in records:
        detection, record_points = _parse_record(
            record,
            frame_index=index,
            timestamp=timestamp,
            width=width,
            height=height,
            coordinate_mode=coordinate_mode,
            default_source=default_source,
        )
        if detection is not None:
            detections.append(detection)
        points.extend(record_points)
    return (
        FrameDetections(
            frame_index=index,
            timestamp=timestamp,
            width=width,
            height=height,
            detections=tuple(detections),
            metadata=({} if direct_record else _metadata(frame, _FRAME_RESERVED_KEYS)),
        ),
        points,
    )


def _group_root_records(records: list[Any]) -> list[Mapping[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in records:
        if not isinstance(item, Mapping):
            raise TypeError("A top-level JSON array must contain objects.")
        grouped[_frame_index(item, 0)].append(item)
    return [
        {"frame_index": index, "items": grouped[index]} for index in sorted(grouped)
    ]


def _frame_documents(
    parsed: Mapping[str, Any] | list[Any],
) -> list[Mapping[str, Any] | list[Any]]:
    if isinstance(parsed, list):
        if not parsed:
            return []
        is_frame_list = all(
            isinstance(item, Mapping)
            and (
                any(item.get(key) is not None for key in _DETECTION_COLLECTION_KEYS)
                or item.get("items") is not None
            )
            for item in parsed
        )
        return list(parsed) if is_frame_list else _group_root_records(parsed)

    frame_keys = [key for key in _FRAME_COLLECTION_KEYS if parsed.get(key) is not None]
    if len(frame_keys) > 1:
        raise ValueError(
            "Use only one frame collection key: " + ", ".join(frame_keys) + "."
        )
    if frame_keys:
        value = parsed[frame_keys[0]]
        if not isinstance(value, list):
            raise TypeError(f"{frame_keys[0]} must be a JSON array.")
        return list(value)

    has_spatial_payload = (
        any(parsed.get(key) is not None for key in _DETECTION_COLLECTION_KEYS)
        or parsed.get("items") is not None
        or parsed.get("points") is not None
        or any(
            parsed.get(key) is not None
            for key in (
                *_BBOX_KEYS,
                *_POLYGON_KEYS,
                "quad",
                "point",
                "x",
                "y",
                *_LABEL_KEYS,
                *_SCORE_KEYS,
                *_TEXT_KEYS,
                "track_id",
            )
        )
    )
    return [parsed] if has_spatial_payload else []


def _declared_media(
    parsed: Mapping[str, Any] | list[Any],
) -> Mapping[str, Any]:
    if not isinstance(parsed, Mapping):
        return {}
    media = parsed.get("media", {})
    if media is None:
        return {}
    if not isinstance(media, Mapping):
        raise TypeError("media must be a JSON object.")
    return media


def _validate_media_dimensions(
    parsed: Mapping[str, Any] | list[Any],
    media: Mapping[str, Any],
    width: int,
    height: int,
) -> None:
    root = parsed if isinstance(parsed, Mapping) else {}
    for name, expected in (("width", width), ("height", height)):
        declarations = [
            owner[name]
            for owner in (root, media)
            if name in owner and owner[name] is not None
        ]
        for value in declarations:
            declared = _positive_dimension(value, name)
            if declared != expected:
                raise ValueError(
                    f"JSON {name} {declared} does not match parser {expected}."
                )


def _resolve_timing(
    parsed: Mapping[str, Any] | list[Any],
    media: Mapping[str, Any],
    frame_count: int,
    fps: float | None,
) -> tuple[int, float | None]:
    root = parsed if isinstance(parsed, Mapping) else {}
    declared_counts = [
        owner["frame_count"]
        for owner in (root, media)
        if owner.get("frame_count") is not None
    ]
    for value in declared_counts:
        declared = _non_negative_integer(value, "frame_count")
        if frame_count and declared and declared != frame_count:
            raise ValueError(
                f"JSON frame_count {declared} does not match parser {frame_count}."
            )
        if frame_count == 0:
            frame_count = declared

    declared_fps = [
        owner["fps"] for owner in (root, media) if owner.get("fps") is not None
    ]
    for value in declared_fps:
        declared = _optional_fps(value)
        if fps is not None and declared is not None and not math.isclose(fps, declared):
            raise ValueError(f"JSON fps {declared} does not match parser {fps}.")
        if fps is None:
            fps = declared
    return frame_count, fps


def _sequence_metadata(
    parsed: Mapping[str, Any] | list[Any],
    media: Mapping[str, Any],
    coordinate_mode: str,
) -> dict[str, Any]:
    if not isinstance(parsed, Mapping):
        return {"coordinate_mode": coordinate_mode}
    result = _metadata(parsed, _ROOT_RESERVED_KEYS)
    media_extra = {
        key: value
        for key, value in media.items()
        if key not in {"width", "height", "frame_count", "fps"}
    }
    if media_extra:
        result["media_metadata"] = media_extra
    result.setdefault("coordinate_mode", coordinate_mode)
    return result


def parse_spatial_response(
    response: str,
    *,
    width: int,
    height: int,
    coordinate_mode: str = "pixel",
    frame_count: int = 0,
    fps: float = 0.0,
    source: str = "",
) -> tuple[DetectionSequence, PointSequence, str]:
    """Convert a strict VLM JSON response into canonical spatial payloads."""

    width = _positive_dimension(width, "width")
    height = _positive_dimension(height, "height")
    coordinate_mode = _coordinate_mode(coordinate_mode)
    frame_count = _non_negative_integer(frame_count, "frame_count")
    resolved_fps = _optional_fps(fps)
    if not isinstance(source, str):
        raise TypeError("source must be a string.")

    parsed = load_json_document(response)
    if isinstance(parsed, Mapping):
        _validate_declared_mode(parsed, coordinate_mode)
    media = _declared_media(parsed)
    _validate_media_dimensions(parsed, media, width, height)
    frame_count, resolved_fps = _resolve_timing(
        parsed,
        media,
        frame_count,
        resolved_fps,
    )
    root_source = parsed.get("source") if isinstance(parsed, Mapping) else None
    if root_source is not None and not isinstance(root_source, str):
        raise TypeError("source must be a string.")
    resolved_source = root_source or source.strip() or None

    frames = []
    points = []
    seen_indices = set()
    for default_index, frame_document in enumerate(_frame_documents(parsed)):
        frame, frame_points = _frame_records(
            frame_document,
            default_index=default_index,
            width=width,
            height=height,
            coordinate_mode=coordinate_mode,
            fps=resolved_fps,
            default_source=resolved_source,
        )
        if frame.frame_index in seen_indices:
            raise ValueError(f"Duplicate frame_index {frame.frame_index}.")
        seen_indices.add(frame.frame_index)
        frames.append(frame)
        points.extend(frame_points)
    frames.sort(key=lambda item: item.frame_index)
    points.sort(key=lambda item: (item.frame_index, item.timestamp))

    metadata = _sequence_metadata(parsed, media, coordinate_mode)
    detections = DetectionSequence(
        width=width,
        height=height,
        frames=tuple(frames),
        frame_count=frame_count,
        fps=resolved_fps,
        source=resolved_source,
        metadata=metadata,
    )
    point_sequence = PointSequence(
        width=width,
        height=height,
        points=tuple(points),
        frame_count=detections.frame_count,
        fps=resolved_fps,
        source=resolved_source,
        metadata=metadata,
    )
    combined = {
        "schema": "comfyui-vlm/spatial",
        "version": 1,
        "detections": detections.to_dict(),
        "points": point_sequence.to_dict(),
    }
    return (
        detections,
        point_sequence,
        json.dumps(
            combined,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ),
    )


def build_spatial_prompt(
    request: str,
    *,
    coordinate_mode: str,
    width: int,
    height: int,
    frame_count: int = 1,
    fps: float = 0.0,
) -> str:
    """Build a deterministic, provider-neutral strict spatial JSON prompt."""

    if not isinstance(request, str):
        raise TypeError("request must be a string.")
    request = request.strip()
    if not request:
        raise ValueError("request cannot be empty.")
    coordinate_mode = _coordinate_mode(coordinate_mode)
    width = _positive_dimension(width, "width")
    height = _positive_dimension(height, "height")
    frame_count = _non_negative_integer(frame_count, "frame_count")
    resolved_fps = _optional_fps(fps)

    if coordinate_mode == "pixel":
        range_text = f"x coordinates in [0, {width}] and y coordinates in [0, {height}]"
    elif coordinate_mode == "normalized_0_1":
        range_text = "both x and y coordinates in [0, 1]"
    else:
        range_text = "both x and y coordinates in [0, 1000]"

    media: dict[str, int | float] = {
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }
    if resolved_fps is not None:
        media["fps"] = resolved_fps
    media_json = json.dumps(media, ensure_ascii=False, separators=(",", ":"))

    return (
        "Perform this visual analysis task:\n"
        f"{request}\n\n"
        "Return only one valid JSON object. Do not add Markdown, prose, comments, "
        "or trailing text. Use exactly the declared coordinate_mode. "
        "All boxes are [x1,y1,x2,y2] with x2 >= x1 and y2 >= y1. "
        "Polygons are [[x,y],...] with at least three points; quads have exactly "
        "four points; points are [x,y]. Clip every coordinate to the media bounds. "
        "Scores/confidences are numbers from 0 to 1. Use an empty array when "
        "nothing is found. For image batches or video, emit one frames entry per "
        "analyzed image/frame and include its zero-based frame_index. Do not merge "
        "objects across frames. Optional metadata must be a JSON object.\n\n"
        f"coordinate_mode: {coordinate_mode} ({range_text})\n"
        "Required JSON shape:\n"
        "{"
        f'"coordinate_mode":"{coordinate_mode}",'
        f'"media":{media_json},'
        '"metadata":{},'
        '"frames":[{'
        '"frame_index":0,'
        '"detections":[{'
        '"label":"object class",'
        '"score":0.0,'
        '"bbox_xyxy":[0,0,0,0],'
        '"polygon":[[0,0],[0,0],[0,0]],'
        '"metadata":{}'
        "}],"
        '"points":[{"label":"point label","score":0.0,"point":[0,0],'
        '"metadata":{}}],'
        '"metadata":{}'
        "}]"
        "}"
    )


class VLMSpatialPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "request": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Detect and segment every visible object, and return "
                            "useful key points."
                        ),
                    },
                ),
                "coordinate_mode": (list(COORDINATE_MODES),),
                "width": ("INT", {"default": 1024, "min": 1, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "step": 1}),
                "frame_count": ("INT", {"default": 1, "min": 0, "step": 1}),
                "fps": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.01,
                        "tooltip": "Zero means FPS is unknown/not applicable.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "VLM Nodes/Vision/Spatial"

    def build(
        self,
        request,
        coordinate_mode,
        width,
        height,
        frame_count=1,
        fps=0.0,
    ):
        return (
            build_spatial_prompt(
                request,
                coordinate_mode=coordinate_mode,
                width=width,
                height=height,
                frame_count=frame_count,
                fps=fps,
            ),
        )


class VLMStructuredSpatialParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "response": (
                    "STRING",
                    {"multiline": True, "default": '{"frames":[]}'},
                ),
                "coordinate_mode": (list(COORDINATE_MODES),),
                "width": ("INT", {"default": 1024, "min": 1, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "step": 1}),
                "frame_count": ("INT", {"default": 0, "min": 0, "step": 1}),
                "fps": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.01,
                        "tooltip": "Zero uses JSON FPS or leaves FPS unset.",
                    },
                ),
                "source": ("STRING", {"default": "structured_vlm"}),
            }
        }

    RETURN_TYPES = (VLM_DETECTIONS, VLM_POINTS, "STRING")
    RETURN_NAMES = ("detections", "points", "normalized_json")
    FUNCTION = "parse"
    CATEGORY = "VLM Nodes/Vision/Spatial"

    def parse(
        self,
        response,
        coordinate_mode,
        width,
        height,
        frame_count=0,
        fps=0.0,
        source="structured_vlm",
    ):
        return parse_spatial_response(
            response,
            width=width,
            height=height,
            coordinate_mode=coordinate_mode,
            frame_count=frame_count,
            fps=fps,
            source=source,
        )


NODE_CLASS_MAPPINGS = {
    "VLMSpatialPromptBuilder": VLMSpatialPromptBuilder,
    "VLMStructuredSpatialParser": VLMStructuredSpatialParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMSpatialPromptBuilder": "VLM Spatial Prompt Builder",
    "VLMStructuredSpatialParser": "VLM Structured Spatial Parser",
}
