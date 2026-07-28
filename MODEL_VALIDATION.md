# Model validation

Validated on 2026-07-29 with ComfyUI 0.28.0, Python 3.12, Transformers 5.14.1,
PyTorch 2.13.0+cu126, and an RTX 3090 24 GB. All models and caches were stored
on the D drive and executed through WSL.

## Real-weight passes

| Family | Representative result | Peak CUDA |
| --- | --- | ---: |
| Qwen 3.5 | 0.8B BF16 image/video; 0.8B NF4; 2B, 4B, and 9B images | 0.82–17.62 GiB |
| Qwen 3 VL | 2B, 4B, and 8B images returned the correct red object | 3.99–16.37 GiB |
| SmolVLM2 | 500M image/video and 2.2B video returned the correct object | 2.29–5.41 GiB |
| LFM2.5 VL | 450M returned “red … rectangle” | 0.88 GiB |
| InternVL 3.5 | 1B video returned “green rectangle” after the 448px patch-grid fix | 2.14 GiB |
| Granite Vision 4.1 | 4B returned “solid red square” through native Transformers code | 7.61 GiB |
| Florence-2 | Native converted base-FT returned and parsed a bright-red-square caption | 0.59 GiB |
| llama.cpp GGUF | Official Qwen3.5-0.8B Q4_0 with llama-cpp-python 0.3.34 CUDA loaded in 20.015s and generated the exact requested response in 0.654s | < 1 GiB model weights |

One checkpoint covers sibling sizes that use the same architecture and loader.
The node does not download every size simply to repeat the same integration
test.

## ComfyUI API pass

ComfyUI started from the D-drive WSL installation with all four repaired custom
node repositories enabled and no custom-node import failures. A real local API
workflow (`EmptyImage` -> `ModernVLM` -> `ViewText`) ran the cached LFM2.5-VL
450M checkpoint on a solid red input, returned `Red.`, and completed with
`unload_after=true`. Prompt ID:
`919f92cd-ecb2-487b-abf0-19f5e4d88229`.

A second real local API workflow (`LLMLoader` -> `LLMSampler` -> `ViewText`)
used the official 563 MB `ggml-org/Qwen3.5-0.8B-GGUF` Q4_0 checkpoint with
full GPU offload, `n_batch=256`, `n_ubatch=128`, mmap, and Auto flash
attention. It returned exactly `ComfyUI llama API ready` and completed
successfully. Prompt ID: `eed8458d-de7f-47ac-8ebf-e48e4dacc2d6`.

The installed llama.cpp CUDA 12.4 wheel reported GPU offload, mmap, and mlock
support directly. CPU-only fallback, Metal/Vulkan/SYCL/ROCm-independent
capability detection, multi-GPU options, and flash-attention retry are covered
by simulated backend contract tests; those vendor kernels were not claimed as
real hardware passes on the NVIDIA test machine.

## Catalog validation

Configuration and processor resolution passed for all 15 ungated entries in
the small/fast catalog: Qwen 3.5 0.8B/2B/4B, Qwen 3 VL 2B/4B, Qwen 2.5 VL 3B,
SmolVLM2 256M/500M/2.2B, LFM2.5 VL 450M/1.6B, InternVL 3.5 1B/2B, and Granite
Vision 3.3 2B/4.1 4B. Gemma 3 4B is the sixteenth entry and correctly requires
license acceptance plus `HF_TOKEN`.

## Not marked passed

- Qwen 3 VL 30B-A3B: weights are available locally, but inference validation
  was stopped at the user's request and will not be repeated.
- Moondream2 2025-06-21: its pinned remote wrapper needed Transformers 5 loading
  metadata, but this Torch/CUDA stack produced NaN probabilities when sampling
  and immediate EOS with greedy decoding. The node defaults to the
  non-destructive greedy path and raises an actionable error on an empty result.
- PaLI-Gemma and Gemma 3: gated checkpoints were not accessible without an
  accepted license and token.
