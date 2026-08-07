"""Quality-gated benchmark for OpenAI-compatible VLM servers.

The runner intentionally depends only on packages already required by this
repository. It is suitable for SGLang and TensorRT-LLM chat endpoints and keeps
the raw evidence required to audit every aggregate number.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import platform
import re
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


def score_output(output: str, evaluator: str, expected: Any) -> float:
    normalized = normalize_text(output)
    if evaluator == "exact":
        return float(normalized == normalize_text(str(expected)))
    if evaluator == "keywords":
        terms = [normalize_text(str(term)) for term in expected]
        terms = [term for term in terms if term]
        return sum(term in normalized for term in terms) / len(terms) if terms else 0.0
    if evaluator == "number":
        match = re.search(r"-?\d+", output.replace(",", ""))
        return float(match is not None and int(match.group()) == int(expected))
    raise ValueError(f"Unsupported evaluator: {evaluator!r}")


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def file_data_url(path: Path) -> tuple[str, str]:
    content = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{encoded}", hashlib.sha256(content).hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_sse_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def run_request(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_url: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_content_at: float | None = None
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    with client.stream(
        "POST", f"{base_url.rstrip('/')}/chat/completions", json=payload
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            event = parse_sse_line(line)
            if event is None:
                continue
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    pieces.append(content)
    finished = time.perf_counter()
    output = "".join(pieces)
    completion_tokens = usage.get("completion_tokens")
    decode_seconds = finished - (first_content_at or finished)
    return {
        "output": output,
        "latency_ms": round((finished - started) * 1000, 3),
        "ttft_ms": round(((first_content_at or finished) - started) * 1000, 3),
        "completion_tokens": completion_tokens,
        "output_tokens_per_second": (
            round(completion_tokens / decode_seconds, 3)
            if completion_tokens and decode_seconds > 0
            else None
        ),
        "usage": usage,
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(sample["latency_ms"]) for sample in samples]
    ttfts = [float(sample["ttft_ms"]) for sample in samples]
    rates = [
        float(sample["output_tokens_per_second"])
        for sample in samples
        if sample.get("output_tokens_per_second") is not None
    ]
    return {
        "requests": len(samples),
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
        },
        "ttft_ms": {
            "p50": round(percentile(ttfts, 0.50), 3),
            "p95": round(percentile(ttfts, 0.95), 3),
            "p99": round(percentile(ttfts, 0.99), 3),
        },
        "output_tokens_per_second_mean": round(statistics.fmean(rates), 3) if rates else None,
        "quality_mean": round(statistics.fmean(sample["quality"] for sample in samples), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--backend", choices=("sglang", "tensorrt-llm", "other"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1:
        parser.error("--warmups must be non-negative and --runs must be positive")

    suite_path = args.suite.resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = suite.get("cases") or []
    if not cases:
        raise ValueError("The suite must contain at least one case.")

    prepared = []
    for case in cases:
        media_path = (suite_path.parent / case["image"]).resolve()
        if not media_path.is_file():
            raise FileNotFoundError(f"Missing benchmark media: {media_path}")
        data_url, digest = file_data_url(media_path)
        prepared.append((case, data_url, digest))

    samples: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for index in range(args.warmups + args.runs):
            case, data_url, digest = prepared[index % len(prepared)]
            result = run_request(
                client,
                base_url=args.base_url,
                model=suite["model"],
                prompt=case["prompt"],
                image_url=data_url,
                max_tokens=int(suite.get("max_tokens", 128)),
                temperature=float(suite.get("temperature", 0.0)),
            )
            if index < args.warmups:
                continue
            result.update(
                {
                    "sample": index - args.warmups,
                    "case_id": case["id"],
                    "task": case["task"],
                    "media_sha256": digest,
                    "quality": score_output(
                        result["output"], case["evaluator"], case["expected"]
                    ),
                }
            )
            samples.append(result)

    summary = aggregate(samples)
    tolerance = float(suite.get("quality_tolerance", 0.98))
    artifact = {
        "schema": "comfyui-vlm/benchmark-run",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "backend": args.backend,
        "suite": suite["name"],
        "model": suite["model"],
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "server_base_url": args.base_url,
        },
        "settings": {
            "warmups": args.warmups,
            "runs": args.runs,
            "max_tokens": suite.get("max_tokens", 128),
            "temperature": suite.get("temperature", 0.0),
            "quality_tolerance": tolerance,
        },
        "summary": summary,
        "quality_gate": {
            "threshold": tolerance,
            "passed": summary["quality_mean"] >= tolerance,
        },
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"{timestamp}-{args.label}.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

