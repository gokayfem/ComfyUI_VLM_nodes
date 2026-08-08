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
6. TensorRT component engines where the model is exportable; TensorRT-LLM only
   where the upstream runtime supports the complete architecture.

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

## SGLang and FlashInfer matrix

The same 448x299 image, prompt, greedy decode, 96-token cap, RTX 3090, three
warmups, and ten measured requests were used for the serving-runtime matrix.
SGLang 0.5.10.post1 ran with PyTorch 2.9.1+cu128, Transformers 5.3.0, and
FlashInfer 0.6.7.post3. The concept gate requires the woman, golden retriever,
beach, and high-five action; inflection aliases such as `high-fiving` are
accepted within that action concept.

| Iteration | Runtime change | TTFT p50 / p95 | E2E p50 / p95 | Output tok/s | Quality |
| --- | --- | ---: | ---: | ---: | --- |
| 05a SGLang 0.5.9 native | FlashInfer + SDPA vision | 38.3 / 42.9 ms | 43.3 / 48.0 ms | 393.9 | **fail; output was only a code fence** |
| 05b SGLang 0.5.10 Transformers backend | Version + model implementation | 75.6 / 79.2 ms | 254.2 / 257.5 ms | 173.5 | pass; exact vs 01b |
| 05c SGLang 0.5.10 native | Native model implementation | 35.2 / 38.3 ms | 240.6 / 243.6 ms | 194.7 | concept pass |
| 05d Triton multimodal attention | SDPA vision -> Triton vision | 35.5 / 37.9 ms | 193.6 / 195.3 ms | 196.4 | pass; exact vs 01b |
| 05e compiled decode | `torch.compile`, max batch 4 | 37.5 / 41.0 ms | 190.5 / 194.7 ms | 202.6 | pass; exact vs 01b |

Iteration 05e is 7.33x faster end to end and delivers 4.79x higher output
throughput than iteration 00. Iteration 05c retains the best TTFT at 19.88x
faster than iteration 00, while 05e trades 2.2 ms of TTFT for the best E2E and
decode throughput. The one-request 0.5.10 cold probe took 17.6 seconds because
of one-time compilation and is kept separate from steady-state percentiles.

The 0.5.9 result demonstrates why latency cannot be promoted without output
evidence: its apparently extraordinary timing came from terminating after two
invalid tokens. The 0.5.10 release fixed the native vision path for this case.
The current 0.5.15.post1 release was also installed and audited, but its CUDA
13 / PyTorch 2.11 build cannot initialize CUDA on the machine's NVIDIA 560.94
driver, so it is recorded as incompatible rather than benchmarked.

## TensorRT vision engine

Current TensorRT-LLM does not list Qwen3-VL as a supported multimodal serving
architecture, so iteration 06 does not mislabel its PyTorch backend as a
TensorRT engine. Instead, Torch-TensorRT 2.9.0 and TensorRT 10.13.3 compile the
fixed-shape Qwen3-VL vision tower into one real BF16 engine on the RTX 3090.
The graph has zero PyTorch fallback partitions.

```bash
./.venv-tensorrt/bin/python benchmarks/qwen3_vl_tensorrt.py \
  --image benchmarks/media/qwen-demo.jpeg \
  --longest-edge 448 \
  --warmups 3 \
  --runs 10 \
  --generation-warmups 1 \
  --generation-runs 3 \
  --output benchmarks/results/qwen3-vl-2b-tensorrt-vision-full.json
```

| Path | Vision p50 | TTFT p50 / p95 | E2E p50 / p95 | Output tok/s | Quality |
| --- | ---: | ---: | ---: | ---: | --- |
| Torch 2.9 eager control | 2385.3 ms | 2452.9 / 2464.3 ms | 2660.6 / 2674.7 ms | 143.3 | 3/3 identical |
| TensorRT vision + unchanged decoder | 9.1 ms | 61.4 / 62.4 ms | 273.4 / 274.0 ms | 142.1 | exact output vs eager |

Engine construction took 98.070 seconds and is reported separately from
inference. TensorRT produced the same 31-token sentence in every full-model
sample. Its isolated 262.8x vision speedup is real relative to the Torch 2.9
eager control but is not the cross-stack headline: the established Torch 2.8
Transformers path already runs end to end in 274.5 ms, and SGLang iteration
05e remains the overall winner at 190.5 ms. The useful result is a verified
9.1 ms vision engine and a new 61.4 ms Transformers TTFT.

Iteration 07 serializes that engine and injects its packed pooler plus three
deep-stack tensors into SGLang's native decoder. The static bridge only accepts
the compiled `(1, 18, 28)` grid; other image shapes fall back to SGLang's
unchanged vision path.

| Path | TTFT p50 / p95 | E2E p50 / p95 | Output tok/s | Semantic gate | Exact gate |
| --- | ---: | ---: | ---: | --- | --- |
| 05e SGLang control | 37.5 / 41.0 ms | **190.5 / 194.7 ms** | **202.6** | pass | pass; 31 tokens |
| 07 TensorRT + SGLang | **34.9 / 37.8 ms** | 250.7 / 366.1 ms | 176.1 | pass | **fail; 40 tokens** |

The bridge reduced TTFT by 7.0%, but numerical differences in the Transformers
vision engine changed greedy decoding to a longer, semantically correct
caption. That makes iteration 07 a measured regression rather than a promoted
speedup. The next experiment is to compile SGLang-native vision weights and
preserve the exact 31-token output.

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
- `concepts`: required semantic concepts, each with one or more accepted aliases.
- `exact`: normalized exact match for OCR and constrained answers.
- `number`: extracts the first integer for counting tasks.

Detection, segmentation, and tracking evaluators will be added after the first
text-output baseline is frozen. Their artifacts will use the same run envelope
and add box, mask, or track data rather than creating a separate leaderboard.

## Result review

The comparison site lives in `benchmarks/site`. It shows regressions alongside
winners and never substitutes estimates for missing GPU runs.
The existing `11.38×` figure is explicitly labeled as frame-by-pixel input-work
reduction, not end-to-end model acceleration.
