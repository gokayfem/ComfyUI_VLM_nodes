"""Opt-in real-weight smoke test for the GGUF *node classes*.

`manual_llama_cpp_smoke.py` proves the shared `LlamaHandle` runtime loads and
generates. This script goes one level up and drives the actual ComfyUI node
classes end to end against real weights, which covers the parts the offline
suite deliberately stubs:

* `LLMLoader` resolving a real file through ComfyUI's `folder_paths`
* `LLMSampler` producing real text from real sampling arguments
* `StructuredOutput` constraining a real model to a generated JSON Schema —
  the llama.cpp grammar path, which cannot be verified with a stub
* `LLMOptionalMemoryFreeSimple` releasing a real llama.cpp allocation

Never run in CI: it downloads weights and needs `llama-cpp-python`.

Example:
    python tests/manual_llm_node_smoke.py --download
    python tests/manual_llm_node_smoke.py --model /models/qwen.gguf
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import folder_paths  # noqa: E402
from ComfyUI_VLM_nodes.nodes.runtime import hf_download, model_root  # noqa: E402
from ComfyUI_VLM_nodes.nodes.suggest import (  # noqa: E402
    LLMLoader,
    LLMOptionalMemoryFreeSimple,
    LLMSampler,
    StructuredOutput,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--repo", default="ggml-org/Qwen3.5-0.8B-GGUF")
    parser.add_argument("--filename", default="Qwen3.5-0.8B-Q4_0.gguf")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    return parser.parse_args()


def stage_model(args: argparse.Namespace) -> str:
    """Put the GGUF where ComfyUI's folder_paths can enumerate it."""

    if args.model is None:
        if not args.download:
            raise SystemExit(
                "Pass --model /path/to/model.gguf, or allow the small default "
                "download with --download."
            )
        source = hf_download(args.repo, args.filename, "llm-node-smoke")
    else:
        source = args.model.resolve()
        if not source.is_file():
            raise SystemExit(f"{source} is not a file.")

    destination = model_root() / source.name
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    # The loader nodes offer whatever folder_paths enumerates, so the staged
    # file has to actually show up there. Staging happens before the first
    # get_filename_list call in this process, so there is no cache to clear.
    listed = folder_paths.get_filename_list("LLavacheckpoints")
    if source.name not in listed:
        raise SystemExit(
            f"{source.name} is not enumerated in LLavacheckpoints: {listed}"
        )
    return source.name


def main() -> None:
    args = parse_args()
    checkpoint = stage_model(args)
    results: dict[str, object] = {"checkpoint": checkpoint}

    # 1. The loader must hand back a lazy handle that has not loaded yet.
    started = time.perf_counter()
    (model,) = LLMLoader().load_llm_checkpoint(
        ckpt_name=checkpoint,
        max_ctx=2048,
        gpu_layers=args.n_gpu_layers,
        n_threads=4,
    )
    results["loader_returned_without_loading"] = model._llm is None
    results["loader_seconds"] = round(time.perf_counter() - started, 3)

    # 2. Real generation through the real sampler node.
    #
    # Deliberately no assertion on what the model *says*: at 0.8B/Q4 the answer
    # is often factually wrong, and that is model quality, not node
    # correctness. What the node owns is that generation happens and that its
    # sampling arguments actually reach llama.cpp — so assert determinism for a
    # fixed seed at temperature 0 instead.
    def sample(seed: int) -> tuple[str, float]:
        started = time.perf_counter()
        (text,) = LLMSampler().generate_text_advanced(
            system_msg="You answer with a single short sentence.",
            prompt="Name the largest planet in the solar system.",
            model=model,
            max_tokens=48,
            temperature=0.0,
            top_p=0.95,
            top_k=40,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repeat_penalty=1.1,
            seed=seed,
        )
        return text, round(time.perf_counter() - started, 3)

    text, elapsed = sample(42)
    repeat, _ = sample(42)
    results["sampler_seconds"] = elapsed
    results["sampler_text"] = text
    results["sampler_produced_text"] = bool(text.strip())
    results["sampler_deterministic_for_fixed_seed"] = text == repeat

    # 3. The grammar-constrained path. A stub cannot prove this works.
    started = time.perf_counter()
    (value,) = StructuredOutput().keyword_extract(
        prompt="The photograph shows a calm, empty beach at sunrise.",
        model=model,
        temperature=0.0,
        attribute_name="mood",
        attribute_type="Category",
        attribute_description="The overall mood of the described scene.",
        categories="calm, tense, joyful, melancholy",
    )
    results["structured_seconds"] = round(time.perf_counter() - started, 3)
    results["structured_value"] = value
    # The whole point of the schema is that the model cannot answer off-menu.
    results["structured_respected_enum"] = value in {
        "calm",
        "tense",
        "joyful",
        "melancholy",
    }
    model.close()

    # 4. A managed-cache node must really release its allocation.
    node = LLMOptionalMemoryFreeSimple()
    (cached_text,) = node.generate_text(
        ckpt_name=checkpoint,
        max_ctx=2048,
        gpu_layers=args.n_gpu_layers,
        n_threads=4,
        prompt="Say the word: ready",
        temperature=0.0,
        unload=True,
    )
    results["managed_cache_text"] = cached_text
    results["managed_cache_released"] = node._handle is None and node._key is None

    checks = {
        key: value for key, value in results.items() if isinstance(value, bool)
    }
    results["ALL_CHECKS_PASSED"] = all(checks.values())
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not results["ALL_CHECKS_PASSED"]:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    main()
