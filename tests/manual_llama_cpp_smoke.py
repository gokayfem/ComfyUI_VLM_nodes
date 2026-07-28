"""Opt-in real-weight smoke test for the shared llama.cpp runtime.

The default model is a small official ggml-org Qwen checkpoint. Nothing is
downloaded unless --download is supplied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ComfyUI_VLM_nodes.nodes.runtime import (
    LlamaHandle,
    default_llama_threads,
    hf_download,
    llama_cpp_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--repo",
        default="ggml-org/Qwen3.5-0.8B-GGUF",
    )
    parser.add_argument(
        "--filename",
        default="Qwen3.5-0.8B-Q4_0.gguf",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: llama.cpp runtime ready",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model
    if model_path is None:
        if not args.download:
            raise SystemExit(
                "Pass --model /path/to/model.gguf, or explicitly allow the "
                "small default download with --download."
            )
        model_path = hf_download(
            args.repo,
            args.filename,
            "llama-cpp-smoke",
        )

    started = time.perf_counter()
    handle = LlamaHandle(
        model_path,
        n_ctx=2048,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=default_llama_threads(),
        n_batch=512,
        n_ubatch=512,
        flash_attention="Auto",
    )
    try:
        llm = handle.ensure_loaded()
        loaded = time.perf_counter()
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=0.0,
            seed=42,
        )
        finished = time.perf_counter()
        content = response["choices"][0]["message"]["content"]
        print(
            json.dumps(
                {
                    "model": str(model_path),
                    "model_bytes": model_path.stat().st_size,
                    "llama_cpp": llama_cpp_diagnostics(),
                    "load_seconds": round(loaded - started, 3),
                    "generation_seconds": round(finished - loaded, 3),
                    "response": content,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        handle.close()


if __name__ == "__main__":
    main()
