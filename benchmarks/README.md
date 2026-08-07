# VLM Speed Lab

This directory turns performance work into a sequence of reproducible,
quality-gated experiments. The first target is the repository default:
`Qwen/Qwen3-VL-2B-Instruct`.

## Rule zero

A result is a speedup only when it uses the same checkpoint revision, media,
prompts, seed, precision policy, and decoding settings as its baseline, and its
task-quality score remains inside the declared tolerance. A faster result that
misses the quality gate is recorded as a regression.

## Iteration order

1. Transformers BF16 + SDPA baseline.
2. Existing adaptive sampling and pixel-budget nodes.
3. Flash Attention 2.
4. `torch.compile` / CUDA graph experiments.
5. SGLang with its declared attention backend (including FlashInfer where
   selected by the runtime).
6. TensorRT-LLM.

Change one performance variable at a time. Run single-request latency first,
then concurrency sweeps. Never mix cold-start and steady-state samples.

## Reproduce the first RTX 3090 matrix in WSL

The committed `qwen3-vl-2b-matrix-tf5-rubric.json` artifact was generated on
Ubuntu 22.04 under WSL2 with an RTX 3090, PyTorch 2.8.0+cu128, and Transformers
5.12.1. Model files, the virtual environment, media, and results all lived on
the WSL ext4 disk rather than a `/mnt/c` or `/mnt/d` mount.

```bash
HF_ENABLE_PARALLEL_LOADING=true \
HF_PARALLEL_LOADING_WORKERS=8 \
./.venv-bench/bin/python benchmarks/qwen3_vl_matrix.py \
  --image benchmarks/media/qwen-demo.jpeg \
  --runs 10 \
  --max-new-tokens 96 \
  --output benchmarks/results/qwen3-vl-2b-matrix-tf5-rubric.json
```

Ten measured runs follow two warmups for dynamic-cache variants and six for
the compiled static-cache variant. The one-time compilation sample remains in
`warmup_samples`; it is never mixed into steady-state percentiles.

| Iteration | Input | TTFT p50 | E2E p50 | Output tok/s | Peak VRAM | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00 SDPA + dynamic | 2048x1365 | 700.3 ms | 1395.7 ms | 42.3 | 4.55 GiB | rubric pass |
| 01a SDPA + dynamic | 672x448 | 112.8 ms | 844.0 ms | 42.2 | 4.04 GiB | rubric pass |
| 01b SDPA + dynamic | 448x299 | 88.2 ms | 774.1 ms | 43.7 | 4.00 GiB | rubric pass |
| 02 SDPA + static compiled | 448x299 | 76.6 ms | 290.1 ms | 139.4 | 4.02 GiB | rubric + exact-output pass vs 01b |
| 03a FA2 + dynamic | 448x299 | 106.6 ms | 1033.3 ms | 32.3 | 4.00 GiB | exact pass; performance regression |
| 03b FA2 + static compiled | 448x299 | 265.0 ms | 3028.0 ms | 34.4 | 4.02 GiB | **fail; corrupted repetitive output** |
| 04 SDPA + static + scoped TF32 | 448x299 | 74.3 ms | 274.5 ms | 149.2 | 4.03 GiB | rubric + exact-output pass vs 01b |

Iteration 04 is 9.43x faster to first token, 5.08x faster end to end, and
3.53x higher output throughput than iteration 00. Resizing preserves the task
rubric but is not byte-identical to source-resolution output; the artifact
records both facts. The cache/compiler change is byte-identical to iteration
01b, as is scoped TF32. Flash Attention 2 is retained as negative evidence:
its dynamic-cache run was correct but slower, while its static-cache pairing
failed the exact-output gate. These are single-image, batch-one latency
results—not yet a general VLM quality claim.

Parallel safetensor loading reduced warm-filesystem model/processor setup from
88.351 seconds to 6.858 seconds. Treat this as a warm-cache startup result;
network download time is outside the measurement.

## Run the OpenAI-compatible benchmark

SGLang and TensorRT-LLM both expose OpenAI-compatible chat endpoints. Start
one server, copy `suite.example.json`, point its cases to local benchmark media,
and run:

```bash
python benchmarks/vlm_bench.py \
  --suite benchmarks/suite.local.json \
  --base-url http://127.0.0.1:8000/v1 \
  --backend sglang \
  --label qwen3-vl-2b-sglang \
  --warmups 3 \
  --runs 30
```

The runner writes one immutable JSON artifact under `benchmarks/results/`.
It records raw model output, per-request latency and time-to-first-token,
aggregate percentiles, quality scores, media hashes, server identity, and the
local Git commit. Do not hand-edit result artifacts.

## Required suite fields

Each case declares a task and an evaluator:

- `keywords`: case-insensitive keyword recall for captions.
- `exact`: normalized exact match for OCR and constrained answers.
- `number`: extracts the first integer for counting tasks.

Detection, segmentation, and tracking evaluators will be added after the first
text-output baseline is frozen. Their artifacts will use the same run envelope
and add box, mask, or track data rather than creating a separate leaderboard.

## Result review

The comparison site lives in `benchmarks/site`. It intentionally shows planned
iterations as planned and never substitutes estimates for missing GPU runs.
The existing `11.38×` figure is explicitly labeled as frame-by-pixel input-work
reduction, not end-to-end model acceleration.
