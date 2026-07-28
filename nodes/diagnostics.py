"""A zero-download runtime report for portable support requests."""

from __future__ import annotations

import json

from .runtime import runtime_diagnostics


class VLMRuntimeDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("runtime_report",)
    FUNCTION = "report"
    CATEGORY = "VLM Nodes/Diagnostics"
    OUTPUT_NODE = True

    def report(self):
        return (
            json.dumps(
                runtime_diagnostics(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )


NODE_CLASS_MAPPINGS = {"VLMRuntimeDiagnostics": VLMRuntimeDiagnostics}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMRuntimeDiagnostics": "VLM Runtime Diagnostics"
}
