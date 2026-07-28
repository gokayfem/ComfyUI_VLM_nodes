"""Deterministic tracking-by-detection for canonical VLM vision payloads.

The tracker intentionally owns only temporal association and identity. Dense
mask propagation remains the responsibility of SAM-style video models. This
keeps the baseline portable across CUDA, ROCm, MPS, XPU, and CPU systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import bbox_iou, clip_box, mask_iou
from .vision_types import (
    VLM_DETECTIONS,
    VLM_TRACKS,
    Detection,
    DetectionSequence,
    FrozenDict,
    Track,
    TrackSequence,
)

TRACKER_SOURCE = "vlm-bytetrack"
_CHI_SQUARE_FOUR_DOF_99 = 13.2767
_MIN_SIZE = 1.0e-3


def _normalized_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.casefold().split())
    return normalized or None


def _score_or_one(detection: Detection) -> float:
    return 1.0 if detection.score is None else detection.score


def _box_to_measurement(box: Iterable[float]) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in box)
    return np.asarray(
        (
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            max(x2 - x1, _MIN_SIZE),
            max(y2 - y1, _MIN_SIZE),
        ),
        dtype=np.float64,
    )


def _measurement_to_box(measurement: np.ndarray) -> tuple[float, ...]:
    center_x, center_y, width, height = measurement[:4]
    width = max(float(width), _MIN_SIZE)
    height = max(float(height), _MIN_SIZE)
    return (
        float(center_x - width * 0.5),
        float(center_y - height * 0.5),
        float(center_x + width * 0.5),
        float(center_y + height * 0.5),
    )


class _BoxKalmanFilter:
    """Small constant-velocity Kalman filter with no optional dependencies."""

    _observation = np.concatenate(
        (np.eye(4, dtype=np.float64), np.zeros((4, 4), dtype=np.float64)),
        axis=1,
    )

    def __init__(self, box: Iterable[float]):
        measurement = _box_to_measurement(box)
        self.mean = np.concatenate((measurement, np.zeros(4, dtype=np.float64)))
        scale = max(measurement[2], measurement[3], 1.0)
        self.covariance = np.diag(
            (
                scale * scale * 0.01,
                scale * scale * 0.01,
                scale * scale * 0.04,
                scale * scale * 0.04,
                scale * scale,
                scale * scale,
                scale * scale * 0.25,
                scale * scale * 0.25,
            )
        )

    @property
    def box(self) -> tuple[float, ...]:
        return _measurement_to_box(self.mean)

    def predict(self, delta_seconds: float) -> None:
        delta = max(float(delta_seconds), 1.0e-6)
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4, dtype=np.float64) * delta
        scale = max(self.mean[2], self.mean[3], 1.0)
        position_noise = max(scale * 0.02 * delta, 1.0e-3)
        velocity_noise = max(scale * 0.01 * math.sqrt(delta), 1.0e-3)
        process_noise = np.diag((position_noise,) * 4 + (velocity_noise,) * 4) ** 2
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.mean[2:4] = np.maximum(self.mean[2:4], _MIN_SIZE)

    def projected(self) -> tuple[np.ndarray, np.ndarray]:
        scale = max(self.mean[2], self.mean[3], 1.0)
        measurement_noise = (
            np.diag(
                (
                    max(scale * 0.025, 1.0e-3),
                    max(scale * 0.025, 1.0e-3),
                    max(scale * 0.05, 1.0e-3),
                    max(scale * 0.05, 1.0e-3),
                )
            )
            ** 2
        )
        projected_mean = self._observation @ self.mean
        projected_covariance = (
            self._observation @ self.covariance @ self._observation.T
            + measurement_noise
        )
        return projected_mean, projected_covariance

    def gating_distance(self, box: Iterable[float]) -> float:
        measurement = _box_to_measurement(box)
        projected_mean, projected_covariance = self.projected()
        residual = measurement - projected_mean
        try:
            solved = np.linalg.solve(projected_covariance, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(projected_covariance) @ residual
        return float(residual @ solved)

    def update(self, box: Iterable[float]) -> None:
        measurement = _box_to_measurement(box)
        projected_mean, projected_covariance = self.projected()
        cross_covariance = self.covariance @ self._observation.T
        try:
            gain = np.linalg.solve(projected_covariance, cross_covariance.T).T
        except np.linalg.LinAlgError:
            gain = cross_covariance @ np.linalg.pinv(projected_covariance)
        innovation = measurement - projected_mean
        self.mean = self.mean + gain @ innovation
        identity = np.eye(8, dtype=np.float64)
        residual_projection = identity - gain @ self._observation
        self.covariance = residual_projection @ self.covariance @ residual_projection.T
        self.mean[2:4] = np.maximum(self.mean[2:4], _MIN_SIZE)


def _merged_metadata(
    detection: Detection,
    *,
    observation: str,
    track_state: str,
    association_stage: str,
    association_score: float | None,
) -> FrozenDict:
    metadata = detection.metadata.to_dict()
    if detection.track_id is not None:
        metadata.setdefault("source_track_id", detection.track_id)
    metadata.update(
        {
            "observation": observation,
            "track_state": track_state,
            "association_stage": association_stage,
            "association_score": association_score,
        }
    )
    return FrozenDict(metadata)


def _tracked_detection(
    detection: Detection,
    *,
    track_id: int,
    timestamp: float,
    track_state: str,
    association_stage: str,
    association_score: float | None,
) -> Detection:
    return Detection(
        bbox_xyxy=detection.bbox_xyxy,
        label=detection.label,
        text=detection.text,
        score=detection.score,
        polygon=detection.polygon,
        quad=detection.quad,
        frame_index=detection.frame_index,
        timestamp=timestamp,
        track_id=track_id,
        source=detection.source,
        metadata=_merged_metadata(
            detection,
            observation="detected",
            track_state=track_state,
            association_stage=association_stage,
            association_score=association_score,
        ),
        mask=detection.mask,
    )


@dataclass(slots=True)
class _TrackState:
    track_id: int
    filter: _BoxKalmanFilter
    detections: list[Detection]
    label: str | None
    text: str | None
    state: str
    hits: int
    first_frame: int
    last_observed_frame: int
    last_observed_timestamp: float
    last_timestamp: float
    last_mask: object | None = None
    misses: int = 0
    removed_frame: int | None = None
    observed_scores: list[float] = field(default_factory=list)

    @property
    def predicted_box(self) -> tuple[float, ...]:
        return self.filter.box


def _labels_compatible(
    track: _TrackState,
    detection: Detection,
    *,
    label_aware: bool,
) -> bool:
    if not label_aware:
        return True
    old_label = _normalized_label(track.label)
    new_label = _normalized_label(detection.label)
    return old_label is None or new_label is None or old_label == new_label


def _overlap(track: _TrackState, detection: Detection) -> float:
    overlap = bbox_iou(track.predicted_box, detection.bbox_xyxy)
    if track.last_mask is not None and detection.mask is not None:
        try:
            overlap = max(
                overlap,
                mask_iou(track.last_mask, detection.mask),
            )
        except (TypeError, ValueError):
            # Boxes remain a valid association primitive when mask resolutions
            # differ across detector backends.
            pass
    return overlap


def _hungarian_matches(
    tracks: list[_TrackState],
    detections: list[Detection],
    *,
    minimum_iou: float,
    label_aware: bool,
    motion_gate: float,
) -> tuple[
    list[tuple[int, int, float]],
    list[int],
    list[int],
]:
    if not tracks or not detections:
        return (
            [],
            list(range(len(tracks))),
            list(range(len(detections))),
        )

    cost = np.full((len(tracks), len(detections)), np.inf, dtype=np.float64)
    overlaps = np.zeros_like(cost)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            if not _labels_compatible(track, detection, label_aware=label_aware):
                continue
            if track.filter.gating_distance(detection.bbox_xyxy) > motion_gate:
                continue
            overlap = _overlap(track, detection)
            if overlap < minimum_iou:
                continue
            overlaps[track_index, detection_index] = overlap
            cost[track_index, detection_index] = 1.0 - overlap

    finite = np.isfinite(cost)
    if not finite.any():
        return (
            [],
            list(range(len(tracks))),
            list(range(len(detections))),
        )
    safe_cost = np.where(finite, cost, 1.0e6)
    row_indices, column_indices = linear_sum_assignment(safe_cost)
    matches = sorted(
        (
            (int(row), int(column), float(overlaps[row, column]))
            for row, column in zip(row_indices, column_indices)
            if finite[row, column]
        ),
        key=lambda item: (tracks[item[0]].track_id, item[1]),
    )
    matched_tracks = {track_index for track_index, _index, _score in matches}
    matched_detections = {
        detection_index for _index, detection_index, _score in matches
    }
    return (
        matches,
        [index for index in range(len(tracks)) if index not in matched_tracks],
        [index for index in range(len(detections)) if index not in matched_detections],
    )


class VLMByteTracker:
    """ByteTrack-style high/low confidence association over a whole sequence."""

    def __init__(
        self,
        *,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        match_iou_threshold: float = 0.3,
        low_match_iou_threshold: float = 0.2,
        max_age_seconds: float = 1.0,
        min_hits: int = 2,
        label_aware: bool = True,
        emit_predictions: bool = True,
        motion_gate: float = _CHI_SQUARE_FOUR_DOF_99,
        fps_fallback: float = 30.0,
    ):
        values = (
            high_threshold,
            low_threshold,
            match_iou_threshold,
            low_match_iou_threshold,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("Thresholds must be between 0 and 1.")
        if low_threshold > high_threshold:
            raise ValueError(
                "low_threshold must be less than or equal to high_threshold."
            )
        if not math.isfinite(float(max_age_seconds)) or max_age_seconds < 0:
            raise ValueError("max_age_seconds must be finite and non-negative.")
        if not isinstance(min_hits, int) or min_hits < 1:
            raise ValueError("min_hits must be a positive integer.")
        if not math.isfinite(float(motion_gate)) or motion_gate <= 0:
            raise ValueError("motion_gate must be finite and positive.")
        if not math.isfinite(float(fps_fallback)) or fps_fallback <= 0:
            raise ValueError("fps_fallback must be finite and positive.")
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.match_iou_threshold = float(match_iou_threshold)
        self.low_match_iou_threshold = float(low_match_iou_threshold)
        self.max_age_seconds = float(max_age_seconds)
        self.min_hits = min_hits
        self.label_aware = bool(label_aware)
        self.emit_predictions = bool(emit_predictions)
        self.motion_gate = float(motion_gate)
        self.fps_fallback = float(fps_fallback)
        self._tracks: list[_TrackState] = []
        self._next_track_id = 0

    def _timestamp(
        self,
        frame_index: int,
        frame_timestamp: float | None,
        fps: float,
    ) -> float:
        expected = frame_index / fps
        if frame_timestamp is None or (frame_index > 0 and frame_timestamp <= 0.0):
            return expected
        return max(float(frame_timestamp), expected)

    def _spawn(
        self,
        detection: Detection,
        *,
        timestamp: float,
    ) -> None:
        state = "active" if self.min_hits == 1 else "tentative"
        track_id = self._next_track_id
        self._next_track_id += 1
        tracked = _tracked_detection(
            detection,
            track_id=track_id,
            timestamp=timestamp,
            track_state=state,
            association_stage="new",
            association_score=None,
        )
        scores = [] if detection.score is None else [detection.score]
        self._tracks.append(
            _TrackState(
                track_id=track_id,
                filter=_BoxKalmanFilter(detection.bbox_xyxy),
                detections=[tracked],
                label=detection.label,
                text=detection.text,
                state=state,
                hits=1,
                first_frame=detection.frame_index,
                last_observed_frame=detection.frame_index,
                last_observed_timestamp=timestamp,
                last_timestamp=timestamp,
                last_mask=detection.mask,
                observed_scores=scores,
            )
        )

    def _update_track(
        self,
        track: _TrackState,
        detection: Detection,
        *,
        timestamp: float,
        stage: str,
        association_score: float,
    ) -> None:
        track.filter.update(detection.bbox_xyxy)
        track.hits += 1
        track.misses = 0
        track.state = "active" if track.hits >= self.min_hits else "tentative"
        if track.label is None:
            track.label = detection.label
        if track.text is None:
            track.text = detection.text
        track.last_observed_frame = detection.frame_index
        track.last_observed_timestamp = timestamp
        track.last_timestamp = timestamp
        track.last_mask = detection.mask
        if detection.score is not None:
            track.observed_scores.append(detection.score)
        track.detections.append(
            _tracked_detection(
                detection,
                track_id=track.track_id,
                timestamp=timestamp,
                track_state=track.state,
                association_stage=stage,
                association_score=association_score,
            )
        )

    def _mark_missed(
        self,
        track: _TrackState,
        *,
        frame_index: int,
        timestamp: float,
        width: int,
        height: int,
    ) -> None:
        track.misses += 1
        elapsed = max(0.0, timestamp - track.last_observed_timestamp)
        if track.state == "tentative" or elapsed > self.max_age_seconds:
            track.state = "removed"
            track.removed_frame = frame_index
            return
        track.state = "lost"
        if not self.emit_predictions:
            return
        box = clip_box(track.predicted_box, width, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            return
        track.detections.append(
            Detection(
                bbox_xyxy=box,
                label=track.label,
                text=track.text,
                score=None,
                frame_index=frame_index,
                timestamp=timestamp,
                track_id=track.track_id,
                source=TRACKER_SOURCE,
                metadata={
                    "observation": "predicted",
                    "track_state": "lost",
                    "association_stage": "unmatched",
                    "association_score": None,
                },
            )
        )

    def _predict(
        self,
        *,
        timestamp: float,
    ) -> list[_TrackState]:
        candidates = [track for track in self._tracks if track.state != "removed"]
        for track in candidates:
            delta = max(timestamp - track.last_timestamp, 1.0e-6)
            track.filter.predict(delta)
            track.last_timestamp = timestamp
        return candidates

    def _process_frame(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        timestamp: float,
        width: int,
        height: int,
    ) -> None:
        candidates = self._predict(timestamp=timestamp)
        high = [
            detection
            for detection in detections
            if _score_or_one(detection) >= self.high_threshold
        ]
        low = [
            detection
            for detection in detections
            if self.low_threshold <= _score_or_one(detection) < self.high_threshold
        ]

        high_matches, unmatched_candidate_indices, unmatched_high_indices = (
            _hungarian_matches(
                candidates,
                high,
                minimum_iou=self.match_iou_threshold,
                label_aware=self.label_aware,
                motion_gate=self.motion_gate,
            )
        )
        matched_track_ids = set()
        for track_index, detection_index, overlap in high_matches:
            track = candidates[track_index]
            self._update_track(
                track,
                high[detection_index],
                timestamp=timestamp,
                stage="high",
                association_score=overlap,
            )
            matched_track_ids.add(track.track_id)

        low_candidates = [
            candidates[index]
            for index in unmatched_candidate_indices
            if candidates[index].state in {"active", "lost"}
        ]
        low_matches, _unmatched_low_track_indices, _unmatched_low_indices = (
            _hungarian_matches(
                low_candidates,
                low,
                minimum_iou=self.low_match_iou_threshold,
                label_aware=self.label_aware,
                motion_gate=self.motion_gate,
            )
        )
        for track_index, detection_index, overlap in low_matches:
            track = low_candidates[track_index]
            self._update_track(
                track,
                low[detection_index],
                timestamp=timestamp,
                stage="low",
                association_score=overlap,
            )
            matched_track_ids.add(track.track_id)

        for track in candidates:
            if track.track_id not in matched_track_ids:
                self._mark_missed(
                    track,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                )

        for detection_index in unmatched_high_indices:
            self._spawn(high[detection_index], timestamp=timestamp)

    def track(self, sequence: DetectionSequence) -> TrackSequence:
        if not isinstance(sequence, DetectionSequence):
            raise TypeError("sequence must be a DetectionSequence.")
        self._tracks = []
        self._next_track_id = 0
        fps = sequence.fps or self.fps_fallback
        frames = {frame.frame_index: frame for frame in sequence.frames}
        for frame_index in range(sequence.frame_count):
            frame = frames.get(frame_index)
            timestamp = self._timestamp(
                frame_index,
                None if frame is None else frame.timestamp,
                fps,
            )
            detections = (
                []
                if frame is None
                else [
                    Detection(
                        bbox_xyxy=detection.bbox_xyxy,
                        label=detection.label,
                        text=detection.text,
                        score=detection.score,
                        polygon=detection.polygon,
                        quad=detection.quad,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        track_id=detection.track_id,
                        source=detection.source,
                        metadata=detection.metadata,
                        mask=detection.mask,
                    )
                    for detection in frame.detections
                ]
            )
            self._process_frame(
                detections,
                frame_index=frame_index,
                timestamp=timestamp,
                width=sequence.width,
                height=sequence.height,
            )

        tracks = []
        for track in sorted(self._tracks, key=lambda item: item.track_id):
            score = (
                sum(track.observed_scores) / len(track.observed_scores)
                if track.observed_scores
                else None
            )
            tracks.append(
                Track(
                    track_id=track.track_id,
                    detections=tuple(track.detections),
                    label=track.label,
                    score=score,
                    source=TRACKER_SOURCE,
                    metadata={
                        "state": track.state,
                        "hits": track.hits,
                        "misses": track.misses,
                        "first_frame": track.first_frame,
                        "last_observed_frame": track.last_observed_frame,
                        "removed_frame": track.removed_frame,
                    },
                )
            )
        metadata = sequence.metadata.to_dict()
        metadata["tracker"] = {
            "algorithm": "bytetrack-style-hungarian",
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "match_iou_threshold": self.match_iou_threshold,
            "low_match_iou_threshold": self.low_match_iou_threshold,
            "max_age_seconds": self.max_age_seconds,
            "min_hits": self.min_hits,
            "label_aware": self.label_aware,
            "emit_predictions": self.emit_predictions,
        }
        return TrackSequence(
            width=sequence.width,
            height=sequence.height,
            tracks=tuple(tracks),
            frame_count=sequence.frame_count,
            fps=sequence.fps,
            source=TRACKER_SOURCE,
            metadata=metadata,
        )


def associate_detection_sequence(
    sequence: DetectionSequence,
    **tracker_options,
) -> TrackSequence:
    """Convenience function for callers that do not need a reusable tracker."""

    return VLMByteTracker(**tracker_options).track(sequence)


class VLMTrackDetections:
    """ComfyUI node wrapper for deterministic tracking-by-detection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detections": (VLM_DETECTIONS,),
                "high_threshold": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "low_threshold": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "match_iou_threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "low_match_iou_threshold": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_age_seconds": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.05},
                ),
                "min_hits": (
                    "INT",
                    {"default": 2, "min": 1, "max": 100},
                ),
                "label_aware": ("BOOLEAN", {"default": True}),
                "emit_predictions": ("BOOLEAN", {"default": True}),
                "fps_fallback": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.01, "max": 1000.0},
                ),
            }
        }

    RETURN_TYPES = (VLM_TRACKS,)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "track"
    CATEGORY = "VLM Nodes/Vision/Tracking"

    def track(
        self,
        detections,
        high_threshold,
        low_threshold,
        match_iou_threshold,
        low_match_iou_threshold,
        max_age_seconds,
        min_hits,
        label_aware,
        emit_predictions,
        fps_fallback,
    ):
        tracker = VLMByteTracker(
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            match_iou_threshold=match_iou_threshold,
            low_match_iou_threshold=low_match_iou_threshold,
            max_age_seconds=max_age_seconds,
            min_hits=min_hits,
            label_aware=label_aware,
            emit_predictions=emit_predictions,
            fps_fallback=fps_fallback,
        )
        return (tracker.track(detections),)


NODE_CLASS_MAPPINGS = {"VLMTrackDetections": VLMTrackDetections}
NODE_DISPLAY_NAME_MAPPINGS = {"VLMTrackDetections": "VLM Track Detections"}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TRACKER_SOURCE",
    "VLMByteTracker",
    "VLMTrackDetections",
    "associate_detection_sequence",
]
