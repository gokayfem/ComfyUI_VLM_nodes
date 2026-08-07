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

