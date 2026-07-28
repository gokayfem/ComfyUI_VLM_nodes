"""Validate curated Hugging Face IDs without downloading model weights.

This opt-in network check resolves each repository's configuration and
processor through the installed Transformers version. It complements, but does
not replace, the real-weight smoke tests.

    python tests/manual_catalog_probe.py
"""

from __future__ import annotations

import json

from transformers import AutoConfig, AutoProcessor

from ComfyUI_VLM_nodes.nodes.modern_vlm import MODEL_CATALOG


def main() -> int:
    records = []
    for label, spec in MODEL_CATALOG.items():
        if not spec.small_fast or spec.gated:
            continue
        config = AutoConfig.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(
            spec.repo_id,
            trust_remote_code=spec.trust_remote_code,
        )
        records.append(
            {
                "label": label,
                "repo_id": spec.repo_id,
                "model_type": config.model_type,
                "config_class": type(config).__name__,
                "processor_class": type(processor).__name__,
            }
        )
    print("CATALOG_PROBE_JSON=" + json.dumps(records, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
