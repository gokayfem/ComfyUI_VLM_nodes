"""Reliable, composable text utilities for VLM and creator workflows.

The original node IDs and their first STRING outputs are intentionally kept
stable.  New outputs are appended so existing workflow links remain valid.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from typing import Any


TEXT_CATEGORY = "VLM Nodes/Text"
CREATE_CATEGORY = f"{TEXT_CATEGORY}/Create"
TRANSFORM_CATEGORY = f"{TEXT_CATEGORY}/Transform"
JSON_CATEGORY = f"{TEXT_CATEGORY}/JSON"
INSPECT_CATEGORY = f"{TEXT_CATEGORY}/Inspect"

_MISSING = object()
_JSON_FENCE = re.compile(
    r"```(?:json|javascript|js)?\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_WHOLE_FENCE = re.compile(
    r"^\s*```[^\n`]*\n?(.*?)\n?```\s*$",
    flags=re.DOTALL,
)
_TEMPLATE_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.-]*)\}")


def _metrics(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {
        "characters": len(text),
        "utf8_bytes": len(encoded),
        "words": len(re.findall(r"\S+", text)),
        "lines": 0 if not text else text.count("\n") + 1,
        "approx_tokens": 0 if not text else max(1, (len(encoded) + 3) // 4),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _json_text(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _json_text(value)


def _load_json_document(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("JSON input is empty.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as direct_error:
        for match in _JSON_FENCE.finditer(stripped):
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", stripped):
            try:
                value, _end = decoder.raw_decode(stripped[match.start() :])
                return value
            except json.JSONDecodeError:
                continue
        raise ValueError(
            "Input does not contain a valid JSON document "
            f"(line {direct_error.lineno}, column {direct_error.colno})."
        ) from direct_error


def _path_tokens(path: str) -> list[str | int]:
    value = path.strip()
    if not value or value == "$":
        return []
    if value.startswith("/"):
        return [
            token.replace("~1", "/").replace("~0", "~")
            for token in value[1:].split("/")
        ]

    if value.startswith("$"):
        value = value[1:]
    tokens: list[str | int] = []
    index = 0
    while index < len(value):
        if value[index] == ".":
            index += 1
            continue
        if value[index] == "[":
            end = value.find("]", index + 1)
            if end < 0:
                raise ValueError(f"Unclosed bracket in JSON path: {path!r}.")
            raw = value[index + 1 : end].strip()
            if not raw:
                raise ValueError(f"Empty bracket in JSON path: {path!r}.")
            if raw[0] in "\"'" and raw[-1:] == raw[0]:
                if raw[0] == '"':
                    token: str | int = json.loads(raw)
                else:
                    token = raw[1:-1].replace("\\'", "'")
            elif re.fullmatch(r"-?\d+", raw):
                token = int(raw)
            else:
                token = raw
            tokens.append(token)
            index = end + 1
            continue
        end = index
        while end < len(value) and value[end] not in ".[":
            end += 1
        token_text = value[index:end]
        if not token_text:
            raise ValueError(f"Invalid JSON path: {path!r}.")
        tokens.append(token_text)
        index = end
    return tokens


def _resolve_json_path(document: Any, path: str) -> Any:
    current = document
    for token in _path_tokens(path):
        if isinstance(current, dict):
            key = str(token)
            if key not in current:
                return _MISSING
            current = current[key]
        elif isinstance(current, list):
            if isinstance(token, str) and re.fullmatch(r"-?\d+", token):
                token = int(token)
            if not isinstance(token, int):
                return _MISSING
            resolved = token if token >= 0 else len(current) + token
            if resolved < 0 or resolved >= len(current):
                return _MISSING
            current = current[resolved]
        else:
            return _MISSING
    return current


def _flatten_json(value: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        flattened: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(_flatten_json(child, child_path))
        return flattened or [(prefix, {})]
    if isinstance(value, list):
        flattened = []
        for index, child in enumerate(value):
            flattened.extend(_flatten_json(child, f"{prefix}[{index}]"))
        return flattened or [(prefix, [])]
    return [(prefix, value)]


def _readable_json(value: Any, separator: str, strip_generation_verbs: bool) -> str:
    if not isinstance(value, dict):
        if isinstance(value, list):
            return separator.join(_display_value(item) for item in value)
        return _display_value(value)

    parts = []
    for key, child in value.items():
        if key == "prompt" and isinstance(child, str):
            if strip_generation_verbs:
                child = re.sub(
                    r"\b(?:create|generate)\b",
                    "",
                    child,
                    flags=re.IGNORECASE,
                )
                child = re.sub(r"[ \t]{2,}", " ", child).strip()
            parts.append(child)
        elif key.lower().startswith("sugg"):
            parts.append(_display_value(child))
        elif isinstance(child, list):
            parts.append(f"{key}: {', '.join(_display_value(item) for item in child)}")
        elif isinstance(child, dict):
            parts.append(f"{key}: {_json_text(child, pretty=True)}")
        else:
            parts.append(f"{key}: {_display_value(child)}")
    return separator.join(parts)


class SimpleText:
    """A stable text source with optional normalization and useful metadata."""

    DESCRIPTION = (
        "Create reusable text while preserving the exact original SimpleText "
        "node ID and first output."
    )
    SEARCH_ALIASES = ["text", "prompt", "string", "multiline text"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamicPrompts": True,
                        "placeholder": "Enter text or connect a STRING.",
                        "tooltip": "Text passed to downstream nodes.",
                    },
                ),
            },
            "optional": {
                "trim_edges": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Remove whitespace only at the start and end.",
                    },
                ),
                "normalize_newlines": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Convert CRLF and CR line endings to LF.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("text", "characters", "words", "lines")
    OUTPUT_TOOLTIPS = (
        "The text value.",
        "Unicode character count.",
        "Whitespace-delimited word count.",
        "Line count.",
    )
    FUNCTION = "simple_text"
    CATEGORY = CREATE_CATEGORY

    def simple_text(
        self,
        input_text: str,
        trim_edges: bool = False,
        normalize_newlines: bool = False,
    ):
        text = str(input_text)
        if normalize_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if trim_edges:
            text = text.strip()
        metrics = _metrics(text)
        return (
            text,
            metrics["characters"],
            metrics["words"],
            metrics["lines"],
        )


class JsonToText:
    """Render model JSON without losing the original compatibility behavior."""

    DESCRIPTION = (
        "Parse plain or fenced JSON and render it as readable text, values, "
        "key/value lines, or canonical JSON."
    )
    SEARCH_ALIASES = ["json to string", "json render", "format json"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "{}",
                        "placeholder": "JSON or a response containing a JSON block.",
                    },
                ),
            },
            "optional": {
                "format_mode": (
                    [
                        "Smart readable (legacy compatible)",
                        "Pretty JSON",
                        "Compact JSON",
                        "Values only",
                        "Key/value lines",
                    ],
                    {"default": "Smart readable (legacy compatible)"},
                ),
                "json_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "$.result.items[0] or /result/items/0",
                    },
                ),
                "separator": (
                    "STRING",
                    {
                        "default": "\n\n",
                        "multiline": True,
                        "tooltip": "Used by readable and values-only modes.",
                    },
                ),
                "strip_generation_verbs": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Preserves the original node's prompt-field behavior."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "canonical_json", "items")
    FUNCTION = "json_to_text"
    OUTPUT_NODE = True
    CATEGORY = JSON_CATEGORY

    def json_to_text(
        self,
        text: str,
        format_mode: str = "Smart readable (legacy compatible)",
        json_path: str = "",
        separator: str = "\n\n",
        strip_generation_verbs: bool = True,
    ):
        document = _load_json_document(text)
        selected = _resolve_json_path(document, json_path)
        if selected is _MISSING:
            raise ValueError(f"JSON path was not found: {json_path!r}.")

        if format_mode == "Pretty JSON":
            output = _json_text(selected, pretty=True)
        elif format_mode == "Compact JSON":
            output = _json_text(selected)
        elif format_mode == "Values only":
            output = separator.join(
                _display_value(value) for _path, value in _flatten_json(selected)
            )
        elif format_mode == "Key/value lines":
            output = "\n".join(
                f"{path}: {_display_value(value)}"
                for path, value in _flatten_json(selected)
            )
        elif format_mode == "Smart readable (legacy compatible)":
            output = _readable_json(
                selected,
                separator,
                bool(strip_generation_verbs),
            )
        else:
            raise ValueError(f"Unknown JSON render mode: {format_mode!r}.")

        canonical = _json_text(selected)
        item_count = len(selected) if isinstance(selected, (dict, list)) else 1
        return {
            "ui": {"text": [output]},
            "result": (output, canonical, item_count),
        }


class ViewText:
    """Display final and streamed text safely in a read-only frontend widget."""

    DESCRIPTION = (
        "Inspect, copy, and download text. Connected VLM/API responses stream "
        "into the widget without changing the final STRING output."
    )
    SEARCH_ALIASES = ["show text", "preview text", "display string", "text output"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Connect any STRING output.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("text", "characters", "words", "lines", "stats_json")
    FUNCTION = "view_text"
    OUTPUT_NODE = True
    CATEGORY = INSPECT_CATEGORY

    def view_text(self, text: str):
        value = str(text)
        metrics = _metrics(value)
        stats = _json_text(metrics)
        return {
            "ui": {"text": [value], "text_stats": [metrics]},
            "result": (
                value,
                metrics["characters"],
                metrics["words"],
                metrics["lines"],
                stats,
            ),
        }


class VLMTextJoin:
    DESCRIPTION = (
        "Join up to eight text values, optionally dropping empty or duplicate "
        "parts. This replaces long chains of concatenation nodes."
    )
    SEARCH_ALIASES = ["text concat", "join text", "merge strings", "combine prompts"]

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"text_{letter}": (
                "STRING",
                {"default": "", "multiline": True},
            )
            for letter in "bcdefgh"
        }
        return {
            "required": {
                "text_a": ("STRING", {"default": "", "multiline": True}),
                "separator": (
                    [
                        "New line",
                        "Blank line",
                        "Space",
                        "Comma",
                        "Custom",
                    ],
                    {"default": "New line"},
                ),
                "custom_separator": (
                    "STRING",
                    {
                        "default": " | ",
                        "tooltip": "Used only when separator is Custom.",
                    },
                ),
                "trim_parts": ("BOOLEAN", {"default": True}),
                "drop_empty": ("BOOLEAN", {"default": True}),
                "deduplicate": ("BOOLEAN", {"default": False}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "parts_json", "part_count")
    FUNCTION = "join"
    CATEGORY = CREATE_CATEGORY

    def join(
        self,
        text_a: str,
        separator: str,
        custom_separator: str,
        trim_parts: bool,
        drop_empty: bool,
        deduplicate: bool,
        **kwargs,
    ):
        separators = {
            "New line": "\n",
            "Blank line": "\n\n",
            "Space": " ",
            "Comma": ", ",
            "Custom": custom_separator,
        }
        if separator not in separators:
            raise ValueError(f"Unknown separator: {separator!r}.")
        values = [str(text_a)]
        values.extend(str(kwargs.get(f"text_{letter}", "")) for letter in "bcdefgh")
        if trim_parts:
            values = [value.strip() for value in values]
        if drop_empty:
            values = [value for value in values if value]
        if deduplicate:
            values = list(dict.fromkeys(values))
        return (
            separators[separator].join(values),
            _json_text(values),
            len(values),
        )


class VLMTextTemplate:
    DESCRIPTION = (
        "Render {named} placeholders from a JSON object plus four convenient "
        "{text1}…{text4} sockets. Missing values are handled explicitly."
    )
    SEARCH_ALIASES = ["prompt template", "format text", "text variables"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (
                    "STRING",
                    {
                        "default": "{instruction}\n\nContext:\n{text1}",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "variables_json": (
                    "STRING",
                    {
                        "default": '{"instruction":"Describe the input accurately."}',
                        "multiline": True,
                    },
                ),
                "missing_values": (
                    ["Keep placeholder", "Empty string", "Error"],
                    {"default": "Error"},
                ),
            },
            "optional": {
                f"text{index}": (
                    "STRING",
                    {"default": "", "multiline": True},
                )
                for index in range(1, 5)
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "variables_json", "missing_keys_json")
    FUNCTION = "render"
    CATEGORY = CREATE_CATEGORY

    def render(
        self,
        template: str,
        variables_json: str,
        missing_values: str,
        **kwargs,
    ):
        variables = _load_json_document(variables_json or "{}")
        if not isinstance(variables, dict):
            raise ValueError("Template variables must be a JSON object.")
        variables = dict(variables)
        for index in range(1, 5):
            value = kwargs.get(f"text{index}", _MISSING)
            if value is not _MISSING and value != "":
                variables[f"text{index}"] = str(value)

        left_token = "\x00VLM_LEFT_BRACE\x00"
        right_token = "\x00VLM_RIGHT_BRACE\x00"
        prepared = str(template).replace("{{", left_token).replace("}}", right_token)
        missing: list[str] = []

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in variables:
                return _display_value(variables[name])
            if name not in missing:
                missing.append(name)
            if missing_values == "Keep placeholder":
                return match.group(0)
            if missing_values == "Empty string":
                return ""
            if missing_values == "Error":
                return match.group(0)
            raise ValueError(f"Unknown missing-value policy: {missing_values!r}.")

        rendered = _TEMPLATE_FIELD.sub(replace, prepared)
        if missing and missing_values == "Error":
            raise ValueError(
                "Template variables are missing: " + ", ".join(missing) + "."
            )
        rendered = rendered.replace(left_token, "{").replace(right_token, "}")
        return (
            rendered,
            _json_text(variables),
            _json_text(missing),
        )


class VLMTextClean:
    DESCRIPTION = (
        "Normalize Unicode/newlines, remove an enclosing Markdown fence, "
        "collapse whitespace, deduplicate lines, and cap length deterministically."
    )
    SEARCH_ALIASES = ["clean prompt", "normalize text", "strip markdown"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "unicode_normalization": (
                    ["NFC", "NFKC", "None"],
                    {"default": "NFC"},
                ),
                "whitespace": (
                    [
                        "Normalize line endings",
                        "Preserve",
                        "Collapse horizontal",
                        "Collapse all",
                    ],
                    {"default": "Normalize line endings"},
                ),
                "trim_edges": ("BOOLEAN", {"default": True}),
                "remove_outer_markdown_fence": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "deduplicate_lines": ("BOOLEAN", {"default": False}),
                "max_characters": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10_000_000,
                        "tooltip": "0 keeps the complete text.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "diagnostics_json")
    FUNCTION = "clean"
    CATEGORY = TRANSFORM_CATEGORY

    def clean(
        self,
        text: str,
        unicode_normalization: str,
        whitespace: str,
        trim_edges: bool,
        remove_outer_markdown_fence: bool,
        deduplicate_lines: bool,
        max_characters: int,
    ):
        original = str(text)
        value = original
        if unicode_normalization != "None":
            if unicode_normalization not in {"NFC", "NFKC"}:
                raise ValueError(
                    f"Unknown Unicode normalization: {unicode_normalization!r}."
                )
            value = unicodedata.normalize(unicode_normalization, value)

        if whitespace != "Preserve":
            value = value.replace("\r\n", "\n").replace("\r", "\n")
        if remove_outer_markdown_fence:
            match = _WHOLE_FENCE.fullmatch(value)
            if match:
                value = match.group(1)
        if whitespace == "Collapse horizontal":
            value = "\n".join(
                re.sub(r"[^\S\n]+", " ", line) for line in value.split("\n")
            )
        elif whitespace == "Collapse all":
            value = re.sub(r"\s+", " ", value)
        elif whitespace not in {"Normalize line endings", "Preserve"}:
            raise ValueError(f"Unknown whitespace mode: {whitespace!r}.")

        removed_duplicates = 0
        if deduplicate_lines:
            seen: set[str] = set()
            lines = []
            for line in value.splitlines():
                key = line.strip()
                if key and key in seen:
                    removed_duplicates += 1
                    continue
                if key:
                    seen.add(key)
                lines.append(line)
            value = "\n".join(lines)
        if trim_edges:
            value = value.strip()

        truncated = int(max_characters) > 0 and len(value) > int(max_characters)
        if truncated:
            value = value[: int(max_characters)]
        diagnostics = {
            "changed": value != original,
            "characters_before": len(original),
            "characters_after": len(value),
            "duplicate_lines_removed": removed_duplicates,
            "truncated": truncated,
        }
        return value, _json_text(diagnostics)


class VLMTextReplace:
    DESCRIPTION = (
        "Perform literal or regular-expression replacement with an exact "
        "replacement count and explicit missing-pattern behavior."
    )
    SEARCH_ALIASES = ["find replace", "regex replace", "substitute text"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "search": ("STRING", {"default": "", "multiline": True}),
                "replacement": ("STRING", {"default": "", "multiline": True}),
                "mode": (
                    ["Literal", "Regular expression"],
                    {"default": "Literal"},
                ),
                "case_sensitive": ("BOOLEAN", {"default": True}),
                "max_replacements": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1_000_000,
                        "tooltip": "0 replaces every match.",
                    },
                ),
                "if_not_found": (
                    ["Keep text", "Error"],
                    {"default": "Keep text"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "replacement_count", "diagnostics_json")
    FUNCTION = "replace"
    CATEGORY = TRANSFORM_CATEGORY

    def replace(
        self,
        text: str,
        search: str,
        replacement: str,
        mode: str,
        case_sensitive: bool,
        max_replacements: int,
        if_not_found: str,
    ):
        value = str(text)
        if not search:
            raise ValueError("Search text or regular expression cannot be empty.")
        limit = int(max_replacements)
        if mode == "Literal" and case_sensitive:
            available = value.count(search)
            count = available if limit == 0 else min(available, limit)
            output = value.replace(search, replacement, limit if limit else -1)
        elif mode in {"Literal", "Regular expression"}:
            pattern = re.escape(search) if mode == "Literal" else search
            flags = re.MULTILINE | (0 if case_sensitive else re.IGNORECASE)
            try:
                output, count = re.subn(
                    pattern,
                    replacement,
                    value,
                    count=limit,
                    flags=flags,
                )
            except re.error as exc:
                raise ValueError(f"Invalid regular expression: {exc}.") from exc
        else:
            raise ValueError(f"Unknown replacement mode: {mode!r}.")

        if count == 0 and if_not_found == "Error":
            raise ValueError("The requested search pattern was not found.")
        if if_not_found not in {"Keep text", "Error"}:
            raise ValueError(f"Unknown missing-pattern policy: {if_not_found!r}.")
        diagnostics = {
            "mode": mode,
            "case_sensitive": bool(case_sensitive),
            "replacement_count": count,
        }
        return output, count, _json_text(diagnostics)


class VLMJSONExtract:
    DESCRIPTION = (
        "Safely extract a value using $.dot[0] syntax or RFC 6901 JSON "
        "Pointer, including JSON embedded in a fenced model response."
    )
    SEARCH_ALIASES = ["json path", "json pointer", "extract json field"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "{}", "multiline": True}),
                "path": (
                    "STRING",
                    {
                        "default": "$",
                        "placeholder": "$.result.items[0] or /result/items/0",
                    },
                ),
                "output_format": (
                    ["Text", "Compact JSON", "Pretty JSON"],
                    {"default": "Text"},
                ),
                "if_missing": (
                    ["Error", "Empty string", "Default value"],
                    {"default": "Error"},
                ),
                "default_value": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("value", "found", "value_type", "canonical_json")
    FUNCTION = "extract"
    CATEGORY = JSON_CATEGORY

    def extract(
        self,
        text: str,
        path: str,
        output_format: str,
        if_missing: str,
        default_value: str,
    ):
        document = _load_json_document(text)
        selected = _resolve_json_path(document, path)
        found = selected is not _MISSING
        if not found:
            if if_missing == "Error":
                raise ValueError(f"JSON path was not found: {path!r}.")
            if if_missing == "Empty string":
                selected = ""
            elif if_missing == "Default value":
                selected = default_value
            else:
                raise ValueError(f"Unknown missing-value policy: {if_missing!r}.")

        if output_format == "Text":
            output = _display_value(selected)
        elif output_format == "Compact JSON":
            output = _json_text(selected)
        elif output_format == "Pretty JSON":
            output = _json_text(selected, pretty=True)
        else:
            raise ValueError(f"Unknown JSON output format: {output_format!r}.")

        value_type = (
            "null"
            if selected is None
            else "boolean"
            if isinstance(selected, bool)
            else "integer"
            if isinstance(selected, int)
            else "number"
            if isinstance(selected, float)
            else "string"
            if isinstance(selected, str)
            else "array"
            if isinstance(selected, list)
            else "object"
            if isinstance(selected, dict)
            else type(selected).__name__
        )
        return output, found, value_type, _json_text(selected)


class VLMTextSplit:
    DESCRIPTION = (
        "Split text into a real Comfy list for batched execution, with stable "
        "JSON and count outputs for inspection or API use."
    )
    SEARCH_ALIASES = ["text list", "split prompt", "batch strings", "csv to list"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "mode": (
                    [
                        "Lines",
                        "Paragraphs",
                        "Literal delimiter",
                        "Regular expression",
                        "CSV",
                        "JSON array",
                    ],
                    {"default": "Lines"},
                ),
                "delimiter": (
                    "STRING",
                    {
                        "default": ",",
                        "tooltip": "Used by literal, regex, and CSV modes.",
                    },
                ),
                "trim_items": ("BOOLEAN", {"default": True}),
                "remove_empty": ("BOOLEAN", {"default": True}),
                "deduplicate": ("BOOLEAN", {"default": False}),
                "max_items": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100_000,
                        "tooltip": "0 keeps every item.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("items", "items_json", "item_count")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "split"
    CATEGORY = TRANSFORM_CATEGORY

    def split(
        self,
        text: str,
        mode: str,
        delimiter: str,
        trim_items: bool,
        remove_empty: bool,
        deduplicate: bool,
        max_items: int,
    ):
        value = str(text)
        if mode == "Lines":
            items = value.splitlines()
        elif mode == "Paragraphs":
            items = re.split(r"\n\s*\n+", value)
        elif mode == "Literal delimiter":
            if not delimiter:
                raise ValueError("Literal delimiter cannot be empty.")
            items = value.split(delimiter)
        elif mode == "Regular expression":
            if not delimiter:
                raise ValueError("Regular-expression delimiter cannot be empty.")
            try:
                items = re.split(delimiter, value)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression: {exc}.") from exc
        elif mode == "CSV":
            if len(delimiter) != 1:
                raise ValueError("CSV delimiter must be exactly one character.")
            items = [
                cell
                for row in csv.reader(io.StringIO(value), delimiter=delimiter)
                for cell in row
            ]
        elif mode == "JSON array":
            parsed = _load_json_document(value)
            if not isinstance(parsed, list):
                raise ValueError("JSON array mode requires a top-level array.")
            items = [_display_value(item) for item in parsed]
        else:
            raise ValueError(f"Unknown split mode: {mode!r}.")

        items = [str(item) for item in items]
        if trim_items:
            items = [item.strip() for item in items]
        if remove_empty:
            items = [item for item in items if item]
        if deduplicate:
            items = list(dict.fromkeys(items))
        if int(max_items) > 0:
            items = items[: int(max_items)]
        return items, _json_text(items), len(items)


class VLMTextInspect:
    DESCRIPTION = (
        "Pass text through while reporting deterministic size, line, word, "
        "rough token-budget, and SHA-256 metadata without network calls."
    )
    SEARCH_ALIASES = ["text stats", "count words", "string hash", "prompt size"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "text",
        "characters",
        "utf8_bytes",
        "words",
        "lines",
        "approx_tokens",
        "sha256",
        "stats_json",
    )
    FUNCTION = "inspect"
    CATEGORY = INSPECT_CATEGORY

    def inspect(self, text: str):
        value = str(text)
        metrics = _metrics(value)
        return (
            value,
            metrics["characters"],
            metrics["utf8_bytes"],
            metrics["words"],
            metrics["lines"],
            metrics["approx_tokens"],
            metrics["sha256"],
            _json_text(metrics),
        )


NODE_CLASS_MAPPINGS = {
    "SimpleText": SimpleText,
    "JsonToText": JsonToText,
    "ViewText": ViewText,
    "VLMTextJoin": VLMTextJoin,
    "VLMTextTemplate": VLMTextTemplate,
    "VLMTextClean": VLMTextClean,
    "VLMTextReplace": VLMTextReplace,
    "VLMJSONExtract": VLMJSONExtract,
    "VLMTextSplit": VLMTextSplit,
    "VLMTextInspect": VLMTextInspect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleText": "Text",
    "JsonToText": "JSON to Text",
    "ViewText": "View Text (Streaming)",
    "VLMTextJoin": "Text Join",
    "VLMTextTemplate": "Text Template",
    "VLMTextClean": "Text Clean",
    "VLMTextReplace": "Text Replace",
    "VLMJSONExtract": "JSON Extract",
    "VLMTextSplit": "Text Split / Batch",
    "VLMTextInspect": "Text Inspector",
}
