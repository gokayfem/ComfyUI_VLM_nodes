import json
from pathlib import Path

import pytest

import ComfyUI_VLM_nodes as package
from ComfyUI_VLM_nodes.nodes import simpletext


def test_simple_text_preserves_legacy_default_and_appends_metrics():
    result = simpletext.SimpleText().simple_text("  one\r\ntwo  ")
    assert result == ("  one\r\ntwo  ", 12, 2, 2)

    normalized = simpletext.SimpleText().simple_text(
        "  one\r\ntwo  ",
        trim_edges=True,
        normalize_newlines=True,
    )
    assert normalized == ("one\ntwo", 7, 2, 2)


def test_json_to_text_keeps_legacy_smart_rendering_and_adds_canonical_output():
    response = simpletext.JsonToText().json_to_text(
        '{"prompt":"Create a red kite","suggestion1":"at sunset","tags":["red","sky"]}'
    )
    assert response["result"][0] == "a red kite\n\nat sunset\n\ntags: red, sky"
    assert json.loads(response["result"][1])["tags"] == ["red", "sky"]
    assert response["result"][2] == 3


def test_json_to_text_parses_fenced_model_response_and_json_paths():
    response = simpletext.JsonToText().json_to_text(
        'Model response:\n```json\n{"result":{"items":[{"name":"café"}]}}\n```',
        format_mode="Pretty JSON",
        json_path="$.result.items[0]",
    )
    assert json.loads(response["result"][0]) == {"name": "café"}
    assert response["result"][1] == '{"name":"café"}'

    pointer = simpletext.VLMJSONExtract().extract(
        '{"a/b":{"~key":[10,20]}}',
        "/a~1b/~0key/1",
        "Text",
        "Error",
        "",
    )
    assert pointer == ("20", True, "integer", "20")


def test_json_extract_handles_negative_indexes_and_missing_policy():
    node = simpletext.VLMJSONExtract()
    assert node.extract(
        '{"items":["first","last"]}',
        "$.items[-1]",
        "Text",
        "Error",
        "",
    )[:3] == ("last", True, "string")
    assert node.extract(
        '{"items":[]}',
        "$.missing",
        "Text",
        "Default value",
        "fallback",
    )[:3] == ("fallback", False, "string")
    with pytest.raises(ValueError, match="not found"):
        node.extract("{}", "$.missing", "Text", "Error", "")


def test_text_join_drops_empty_and_duplicate_parts():
    result = simpletext.VLMTextJoin().join(
        " first ",
        "Blank line",
        "|",
        True,
        True,
        True,
        text_b="second",
        text_c="first",
    )
    assert result == (
        "first\n\nsecond",
        '["first","second"]',
        2,
    )


def test_text_template_is_safe_explicit_and_supports_literal_braces():
    result = simpletext.VLMTextTemplate().render(
        "{{schema}} {subject}: {text1}",
        '{"subject":"robot"}',
        "Error",
        text1="moving a box",
    )
    assert result[0] == "{schema} robot: moving a box"
    assert json.loads(result[1]) == {
        "subject": "robot",
        "text1": "moving a box",
    }
    assert result[2] == "[]"

    with pytest.raises(ValueError, match="missing"):
        simpletext.VLMTextTemplate().render(
            "{known} {unknown}",
            '{"known":"yes"}',
            "Error",
        )


def test_text_clean_normalizes_fences_duplicates_and_length():
    result, diagnostics_json = simpletext.VLMTextClean().clean(
        "```text\r\nＡ  B\r\nＡ  B\r\nC\r\n```",
        "NFKC",
        "Collapse horizontal",
        True,
        True,
        True,
        5,
    )
    assert result == "A B\nC"
    diagnostics = json.loads(diagnostics_json)
    assert diagnostics["changed"] is True
    assert diagnostics["duplicate_lines_removed"] == 1
    assert diagnostics["truncated"] is False


def test_text_replace_literal_regex_and_errors():
    node = simpletext.VLMTextReplace()
    assert node.replace(
        "Cat cat cat",
        "cat",
        "dog",
        "Literal",
        False,
        2,
        "Keep text",
    )[:2] == ("dog dog cat", 2)
    assert node.replace(
        "a1 b22",
        r"\d+",
        "#",
        "Regular expression",
        True,
        0,
        "Keep text",
    )[:2] == ("a# b#", 2)
    with pytest.raises(ValueError, match="not found"):
        node.replace("hello", "x", "y", "Literal", True, 0, "Error")


def test_text_split_outputs_real_list_and_stable_json():
    items, items_json, count = simpletext.VLMTextSplit().split(
        '[" first ","second","first",""]',
        "JSON array",
        ",",
        True,
        True,
        True,
        0,
    )
    assert items == ["first", "second"]
    assert json.loads(items_json) == items
    assert count == 2
    assert simpletext.VLMTextSplit.OUTPUT_IS_LIST == (True, False, False)


def test_text_inspector_and_view_text_report_same_metrics():
    inspected = simpletext.VLMTextInspect().inspect("hello\nworld")
    assert inspected[1:6] == (11, 11, 2, 2, 3)
    assert len(inspected[6]) == 64
    assert json.loads(inspected[7])["words"] == 2

    viewed = simpletext.ViewText().view_text("hello\nworld")
    assert viewed["result"][:4] == ("hello\nworld", 11, 2, 2)
    assert viewed["ui"]["text"] == ["hello\nworld"]


def test_text_node_categories_aliases_and_legacy_ids_are_stable():
    assert simpletext.NODE_CLASS_MAPPINGS["SimpleText"] is simpletext.SimpleText
    assert simpletext.NODE_CLASS_MAPPINGS["JsonToText"] is simpletext.JsonToText
    assert simpletext.NODE_CLASS_MAPPINGS["ViewText"] is simpletext.ViewText
    assert set(simpletext.NODE_CLASS_MAPPINGS) == {
        "SimpleText",
        "JsonToText",
        "ViewText",
        "VLMTextJoin",
        "VLMTextTemplate",
        "VLMTextClean",
        "VLMTextReplace",
        "VLMJSONExtract",
        "VLMTextSplit",
        "VLMTextInspect",
    }
    assert simpletext.SimpleText.CATEGORY == "VLM Nodes/Text/Create"
    assert simpletext.ViewText.CATEGORY == "VLM Nodes/Text/Inspect"


def test_text_toolkit_api_example_uses_registered_inputs_and_output_indexes():
    root = Path(package.__file__).parent
    prompt = json.loads(
        (root / "examples" / "text_toolkit_api.json").read_text("utf-8")
    )
    assert prompt["5"]["inputs"]["text"] == ["4", 0]
    for node in prompt.values():
        node_class = package.NODE_CLASS_MAPPINGS[node["class_type"]]
        declared = {
            name
            for group in node_class.INPUT_TYPES().values()
            if isinstance(group, dict)
            for name in group
        }
        assert set(node["inputs"]) <= declared
