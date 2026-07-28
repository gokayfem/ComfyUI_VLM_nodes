from __future__ import annotations

import json

import pytest
from ComfyUI_VLM_nodes.nodes.spatial_parser import (
    COORDINATE_MODES,
    VLMSpatialPromptBuilder,
    VLMStructuredSpatialParser,
    build_spatial_prompt,
    load_json_document,
    parse_spatial_response,
)
from ComfyUI_VLM_nodes.nodes.vision_types import (
    VLM_DETECTIONS,
    VLM_POINTS,
    DetectionSequence,
    PointSequence,
)


def test_prompt_builder_is_explicit_and_provider_neutral():
    prompt = build_spatial_prompt(
        "Find every vehicle.",
        coordinate_mode="normalized_0_1000",
        width=1920,
        height=1080,
        frame_count=12,
        fps=24.0,
    )

    assert prompt.startswith("Perform this visual analysis task:")
    assert "Find every vehicle." in prompt
    assert "Return only one valid JSON object" in prompt
    assert "normalized_0_1000" in prompt
    assert '"frame_count":12' in prompt
    assert '"fps":24.0' in prompt
    assert "zero-based frame_index" in prompt
    assert "bbox_xyxy" in prompt
    assert "polygon" in prompt
    assert '"point"' in prompt
    assert "score" in prompt
    example = json.loads(prompt.split("Required JSON shape:\n", 1)[1])
    assert example["coordinate_mode"] == "normalized_0_1000"


def test_node_contracts_use_canonical_spatial_types():
    assert tuple(COORDINATE_MODES) == (
        "pixel",
        "normalized_0_1",
        "normalized_0_1000",
    )
    assert VLMSpatialPromptBuilder.RETURN_TYPES == ("STRING",)
    assert VLMStructuredSpatialParser.RETURN_TYPES == (
        VLM_DETECTIONS,
        VLM_POINTS,
        "STRING",
    )
    schema = VLMStructuredSpatialParser.INPUT_TYPES()
    assert tuple(schema["required"]["coordinate_mode"][0]) == COORDINATE_MODES


def test_parser_accepts_only_complete_plain_or_fenced_json():
    assert load_json_document(" ") == {}
    assert load_json_document('```json\n{"frames":[]}\n```') == {"frames": []}

    for invalid in (
        'Here is the result: {"frames":[]}',
        'Result:\n```json\n{"frames":[]}\n```',
        '{"frames":[]} trailing',
        "```python\n{}\n```",
        '{"x": 1, "x": 2}',
        '{"score": NaN}',
    ):
        with pytest.raises(ValueError):
            load_json_document(invalid)


def test_normalized_video_parse_clips_and_preserves_metadata():
    response = json.dumps(
        {
            "coordinate_mode": "normalized_0_1",
            "media": {
                "width": 200,
                "height": 100,
                "frame_count": 3,
                "fps": 2,
                "codec": "test-codec",
            },
            "source": "unit-vlm",
            "metadata": {"request_id": "abc"},
            "vendor": {"latency_ms": 12},
            "frames": [
                {
                    "frame_index": 0,
                    "metadata": {"scene": "start"},
                    "detections": [
                        {
                            "class": "cat",
                            "confidence": 0.75,
                            "bbox": [-0.1, 0.2, 1.2, 0.8],
                            "polygon": [
                                [-0.5, 0.2],
                                [0.5, 0.2],
                                [0.5, 1.5],
                            ],
                            "instance_id": "cat-1",
                            "metadata": {"occluded": False},
                        }
                    ],
                    "points": [
                        {
                            "name": "nose",
                            "point": [0.25, 0.5],
                            "confidence": 0.9,
                            "landmark_id": 7,
                        }
                    ],
                },
                {
                    "frame_index": 2,
                    "detections": [
                        {
                            "label": "sign",
                            "quad": [
                                [0.1, 0.1],
                                [0.9, 0.1],
                                [0.9, 0.9],
                                [0.1, 0.9],
                            ],
                            "text": "STOP",
                        }
                    ],
                },
            ],
        }
    )

    detections, points, normalized_json = parse_spatial_response(
        f"```json\n{response}\n```",
        width=200,
        height=100,
        coordinate_mode="normalized_0_1",
    )

    assert isinstance(detections, DetectionSequence)
    assert isinstance(points, PointSequence)
    assert detections.frame_count == points.frame_count == 3
    assert detections.fps == points.fps == 2.0
    assert [frame.frame_index for frame in detections.frames] == [0, 2]
    cat, sign = detections.all_detections()
    assert cat.bbox_xyxy == (0.0, 20.0, 200.0, 80.0)
    assert cat.polygon == ((0.0, 20.0), (100.0, 20.0), (100.0, 100.0))
    assert cat.label == "cat"
    assert cat.score == 0.75
    assert cat.metadata.to_dict() == {
        "instance_id": "cat-1",
        "occluded": False,
    }
    assert sign.bbox_xyxy == (20.0, 10.0, 180.0, 90.0)
    assert sign.quad is not None and len(sign.quad) == 4
    assert sign.text == "STOP"
    assert points.points[0].x == 50.0
    assert points.points[0].y == 50.0
    assert points.points[0].label == "nose"
    assert points.points[0].metadata["landmark_id"] == 7
    assert detections.frames[0].metadata["scene"] == "start"
    assert detections.metadata.to_dict() == {
        "coordinate_mode": "normalized_0_1",
        "media_metadata": {"codec": "test-codec"},
        "request_id": "abc",
        "vendor": {"latency_ms": 12},
    }

    normalized = json.loads(normalized_json)
    assert normalized["schema"] == "comfyui-vlm/spatial"
    assert normalized["detections"] == detections.to_dict()
    assert normalized["points"] == points.to_dict()


def test_pixel_aliases_xywh_flat_segmentation_and_multiple_points():
    response = json.dumps(
        {
            "objects": [
                {
                    "name": "panel",
                    "box": {"x": -2, "y": 5, "width": 15, "height": 30},
                    "segmentation": [0, 5, 13, 5, 13, 35, 0, 35],
                    "points": [[2, 7], [12, 30]],
                    "score": 1,
                }
            ]
        }
    )
    detections, points, _json = parse_spatial_response(
        response,
        width=10,
        height=20,
        coordinate_mode="pixel",
        frame_count=1,
    )

    detection = detections.all_detections()[0]
    assert detection.bbox_xyxy == (0.0, 5.0, 10.0, 20.0)
    assert detection.polygon == (
        (0.0, 5.0),
        (10.0, 5.0),
        (10.0, 20.0),
        (0.0, 20.0),
    )
    assert [(point.x, point.y) for point in points.points] == [
        (2.0, 7.0),
        (10.0, 20.0),
    ]


def test_direct_multi_point_record_keeps_shared_fields_without_duplicates():
    detections, points, _json = parse_spatial_response(
        '{"label":"hand","confidence":0.8,"points":[[1,2],[3,4]],'
        '"metadata":{"side":"left"}}',
        width=10,
        height=10,
    )

    assert detections.all_detections() == ()
    assert [(point.x, point.y) for point in points.points] == [
        (1.0, 2.0),
        (3.0, 4.0),
    ]
    assert {point.label for point in points.points} == {"hand"}
    assert {point.score for point in points.points} == {0.8}
    assert points.points[0].metadata["side"] == "left"
    assert detections.frames[0].metadata.to_dict() == {}


def test_top_level_record_batch_groups_video_frame_indices_and_timestamps():
    response = json.dumps(
        [
            {"frame_index": 2, "bbox_xyxy": [1, 2, 3, 4], "label": "late"},
            {"frame_index": 0, "point": [5, 6], "label": "early"},
            {"frame_index": 0, "box": [0, 0, 4, 5], "label": "first"},
        ]
    )
    detections, points, _json = parse_spatial_response(
        response,
        width=20,
        height=10,
        fps=4,
        coordinate_mode="pixel",
    )

    assert [frame.frame_index for frame in detections.frames] == [0, 2]
    assert detections.frames[1].timestamp == 0.5
    assert detections.frame_count == 3
    assert [item.label for item in detections.all_detections()] == [
        "first",
        "late",
    ]
    assert points.points[0].frame_index == 0


def test_normalized_1000_polygon_without_box_derives_clipped_bbox():
    detections, points, _json = parse_spatial_response(
        '{"polygon":[[-100,100],[500,100],[1200,900]],"label":"shape"}',
        width=300,
        height=200,
        coordinate_mode="normalized_0_1000",
    )

    detection = detections.all_detections()[0]
    assert detection.bbox_xyxy == (0.0, 20.0, 300.0, 180.0)
    assert not points.points


def test_empty_payloads_are_valid_and_predictable():
    for response in ("", "{}", "[]", '{"frames":[]}'):
        detections, points, normalized_json = parse_spatial_response(
            response,
            width=640,
            height=480,
            coordinate_mode="pixel",
            frame_count=5,
            fps=25,
            source="empty-test",
        )
        assert detections.frames == ()
        assert detections.frame_count == 5
        assert detections.fps == 25
        assert points.points == ()
        assert points.frame_count == 5
        normalized = json.loads(normalized_json)
        assert normalized["detections"]["frames"] == []
        assert normalized["points"]["points"] == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ('{"bbox":[4,4,2,5]}', "x2 >= x1"),
        ('{"polygon":[[1,1],[2,2]]}', "at least three"),
        ('{"quad":[[0,0],[1,0],[1,1]]}', "exactly 4"),
        ('{"point":[1]}', "two coordinates"),
        ('{"score":2,"point":[1,1]}', "between 0 and 1"),
        (
            '{"bbox":[0,0,1,1],"polygon":[[0,0],[1,0],[0,1]],'
            '"quad":[[0,0],[1,0],[1,1],[0,1]]}',
            "either polygon",
        ),
        (
            '{"bbox":[0,0,1,1],"coordinate_mode":"normalized_0_1"}',
            "parser is set",
        ),
        ('{"label":"no geometry"}', "requires bbox"),
    ],
)
def test_strict_validation(response, message):
    with pytest.raises((TypeError, ValueError), match=message):
        parse_spatial_response(
            response,
            width=10,
            height=10,
            coordinate_mode="pixel",
        )


def test_dimensions_timing_and_alias_conflicts_are_rejected():
    cases = [
        ('{"media":{"width":20}}', {"width": 10}, "does not match"),
        ('{"media":{"fps":30}}', {"fps": 24}, "does not match"),
        (
            '{"label":"a","class":"b","point":[1,1]}',
            {},
            "Conflicting aliases",
        ),
        (
            '{"frames":[{"frame_index":0},{"frame_index":0}]}',
            {},
            "Duplicate frame_index",
        ),
        (
            '{"detections":[],"objects":[]}',
            {},
            "multiple detection collection",
        ),
    ]
    for response, overrides, message in cases:
        arguments = {
            "width": 10,
            "height": 10,
            "coordinate_mode": "pixel",
            **overrides,
        }
        with pytest.raises(ValueError, match=message):
            parse_spatial_response(response, **arguments)


def test_node_methods_return_direct_canonical_payloads():
    prompt = VLMSpatialPromptBuilder().build(
        "Locate the subject.",
        "pixel",
        100,
        80,
        1,
        0,
    )[0]
    detections, points, normalized = VLMStructuredSpatialParser().parse(
        '{"bbox_xyxy":[1,2,30,40],"point":[5,6]}',
        "pixel",
        100,
        80,
    )

    assert isinstance(prompt, str)
    assert isinstance(detections, DetectionSequence)
    assert isinstance(points, PointSequence)
    assert json.loads(normalized)["version"] == 1
