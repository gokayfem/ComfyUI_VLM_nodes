from types import SimpleNamespace

import pytest
import torch
from ComfyUI_VLM_nodes.nodes.sam2 import (
    SAM2_MODELS,
    Sam2Spec,
    Sam2VideoPredictor,
    VLMSAM2VideoSegmentation,
    _core_box,
    _normalize_processed_masks,
    seed_boxes,
)
from ComfyUI_VLM_nodes.nodes.vision_types import (
    Detection,
    DetectionSequence,
    FrameDetections,
)


def test_sam2_catalog_leads_with_tiny_and_has_no_30b_models():
    assert next(iter(SAM2_MODELS)) == "SAM2.1 Hiera Tiny (fast)"
    assert all("30b" not in spec.model_id.lower() for spec in SAM2_MODELS.values())
    assert VLMSAM2VideoSegmentation.RETURN_NAMES[0:2] == ("tracks", "json")


def test_core_box_is_converted_from_xywh():
    assert _core_box({"x": 2, "y": 3, "width": 5, "height": 7}) == (
        2.0,
        3.0,
        7.0,
        10.0,
    )
    with pytest.raises(ValueError, match="positive"):
        _core_box({"x": 2, "y": 3, "width": 0, "height": 7})


def test_detection_seeds_keep_labels_and_ids():
    detection = Detection(
        bbox_xyxy=(1, 2, 9, 10),
        label="cat",
        frame_index=2,
        timestamp=0.2,
        track_id=7,
    )
    sequence = DetectionSequence(
        width=10,
        height=10,
        frames=(
            FrameDetections(
                frame_index=2,
                timestamp=0.2,
                width=10,
                height=10,
                detections=(detection,),
            ),
        ),
        frame_count=3,
        fps=10,
    )
    boxes, ids, labels = seed_boxes(
        width=10,
        height=10,
        frame_index=2,
        detections=sequence,
        bounding_box=None,
    )
    assert boxes == [[1.0, 2.0, 9.0, 10.0]]
    assert ids == [7]
    assert labels == {7: "cat"}


def test_processed_mask_shapes_are_normalized():
    assert _normalize_processed_masks(torch.zeros(2, 1, 4, 5)).shape == (
        2,
        4,
        5,
    )
    assert _normalize_processed_masks(torch.zeros(4, 5)).shape == (1, 4, 5)
    with pytest.raises(RuntimeError, match="unsupported"):
        _normalize_processed_masks(torch.zeros(1, 2, 3, 4, 5))


def test_video_session_runs_seed_frame_before_both_propagation_directions():
    class FakeProcessor:
        def init_video_session(self, **_kwargs):
            return SimpleNamespace(obj_ids=[1], seed_inferred=False)

        def add_inputs_to_inference_session(self, **kwargs):
            kwargs["inference_session"].seed_frame = kwargs["frame_idx"]

        def post_process_masks(self, masks, **_kwargs):
            return masks

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seed_calls = []
            self.propagation_directions = []

        def forward(self, inference_session, frame_idx):
            inference_session.seed_inferred = True
            self.seed_calls.append(frame_idx)
            return SimpleNamespace(
                frame_idx=frame_idx,
                pred_masks=torch.ones(1, 1, 4, 4),
            )

        def propagate_in_video_iterator(
            self,
            inference_session,
            start_frame_idx,
            reverse=False,
            **_kwargs,
        ):
            assert inference_session.seed_inferred
            self.propagation_directions.append(reverse)
            indices = (
                range(start_frame_idx, 3)
                if not reverse
                else range(start_frame_idx, -1, -1)
            )
            for frame_index in indices:
                yield SimpleNamespace(
                    frame_idx=frame_index,
                    pred_masks=torch.ones(1, 1, 4, 4),
                )

    model = FakeModel()
    predictor = Sam2VideoPredictor.__new__(Sam2VideoPredictor)
    predictor.processor = FakeProcessor()
    predictor.dtype = torch.float32
    predictor.spec = Sam2Spec("test/sam2", "test-sam2")
    predictor.handle = SimpleNamespace(ensure_loaded=lambda: model)
    detections = DetectionSequence(
        width=4,
        height=4,
        frames=(
            FrameDetections(
                frame_index=0,
                timestamp=0.0,
                width=4,
                height=4,
                detections=(
                    Detection(
                        bbox_xyxy=(0, 0, 4, 4),
                        label="object",
                        frame_index=0,
                        timestamp=0.0,
                    ),
                ),
            ),
        ),
        frame_count=1,
    )
    tracks, union, individual, preview = predictor.propagate(
        torch.zeros(3, 4, 4, 3),
        seed_frame=1,
        fps=10.0,
        detections=detections,
        bounding_box=None,
        seed_mask=None,
        mask_threshold=0.0,
        keep_video_on_cpu=True,
        mask_output="union_and_objects",
        render_preview=True,
    )
    assert model.seed_calls == [1]
    assert model.propagation_directions == [False, True]
    assert [item.frame_index for item in tracks.tracks[0].detections] == [0, 1, 2]
    assert union.shape == (3, 4, 4)
    assert individual.shape == (3, 4, 4)
    assert preview.shape == (3, 4, 4, 3)
    assert torch.equal(preview[0], preview[1])
    assert torch.equal(preview[1], preview[2])


def test_multi_object_mask_seeds_are_passed_as_one_mask_per_object():
    class FakeProcessor:
        def __init__(self):
            self.received_masks = None

        def init_video_session(self, **kwargs):
            assert str(kwargs["processing_device"]) == "cpu"
            return SimpleNamespace(obj_ids=[1, 2], seed_inferred=False)

        def add_inputs_to_inference_session(self, **kwargs):
            self.received_masks = kwargs["input_masks"]

        def post_process_masks(self, masks, **_kwargs):
            return masks

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, inference_session, frame_idx):
            inference_session.seed_inferred = True
            return SimpleNamespace(
                frame_idx=frame_idx,
                pred_masks=torch.ones(2, 1, 4, 4),
            )

        def propagate_in_video_iterator(self, **_kwargs):
            return iter(())

    processor = FakeProcessor()
    predictor = Sam2VideoPredictor.__new__(Sam2VideoPredictor)
    predictor.processor = processor
    predictor.dtype = torch.float32
    predictor.spec = Sam2Spec("test/sam2", "test-sam2")
    predictor.handle = SimpleNamespace(ensure_loaded=FakeModel)
    images = torch.zeros(1, 4, 4, 3)
    tracks, _union, individual, preview = predictor.propagate(
        images,
        seed_frame=0,
        fps=24.0,
        detections=None,
        bounding_box=None,
        seed_mask=torch.ones(2, 4, 4),
        mask_threshold=0.0,
        keep_video_on_cpu=True,
        mask_output="union_only",
        render_preview=False,
    )
    assert isinstance(processor.received_masks, list)
    assert len(processor.received_masks) == 2
    assert len(tracks.tracks) == 2
    assert individual.shape == (0, 4, 4)
    assert preview.data_ptr() == images.data_ptr()


def test_nested_core_boxes_select_seed_frame_and_keep_top_level_labels():
    boxes, ids, labels = seed_boxes(
        width=20,
        height=20,
        frame_index=1,
        detections=None,
        bounding_box=[
            [{"x": 0, "y": 0, "width": 2, "height": 2, "label": "old"}],
            [
                {"x": 3, "y": 4, "width": 5, "height": 6, "label": "person"},
                {"x": 10, "y": 11, "width": 4, "height": 3},
            ],
        ],
    )
    assert boxes == [[3.0, 4.0, 8.0, 10.0], [10.0, 11.0, 14.0, 14.0]]
    assert ids == [1, 2]
    assert labels == {1: "person", 2: None}
