# Qwen3-VL cold-start research

This note separates weight loading from warm inference. The current promoted
runtime remains SGLang 0.5.10 native + Triton multimodal attention + compiled
decode at 190.5 ms end to end. FlashPack does not make a resident model decode
faster; it targets the much larger cold-start path.

## Local profile

Host: RTX 3090 24 GB, WSL2 ext4, one 4,255,140,312-byte
`Qwen/Qwen3-VL-2B-Instruct` safetensors checkpoint. Each cold sample ran in a
fresh process after `POSIX_FADV_DONTNEED` was applied only to the measured file.
Conversion to FlashPack was excluded. Three tensors spanning the packed file
were checked bit-for-bit against safetensors and all passed.

| Loader | Reader staging | Cold seconds | Cold p50 / p95 | Effective p50 | Warm p50 | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| safetensors | library default | 58.55, 59.31, 60.18 | 59.31 / 60.18 s | 0.574 Gbit/s | 1.058 s | control |
| safetensors fast GPU | library default | 62.31, 58.51, 60.63 | 60.63 / 62.31 s | 0.561 Gbit/s | 1.137 s | slower |
| FlashPack direct I/O | 4 readers x 2 buffers x 32 MiB = 256 MiB | 44.78, 38.82, 43.74 | **43.74 / 44.78 s** | **0.779 Gbit/s** | not applicable to direct I/O | **26.3% faster** |
| FlashPack direct I/O | 8 readers x 2 buffers x 16 MiB = 256 MiB | 45.74 (probe) | — | 0.744 Gbit/s | — | no improvement |
| FlashPack buffered legacy | bounded internal buffer | 87.24 (probe) | — | 0.390 Gbit/s | 1.00 s | cold regression |

The upstream FlashPack default at the audited `a923a6c` revision attempted
16 readers x 2 buffers x 64 MiB, a 2 GiB pinned staging pool, and failed with a
CUDA pinned-allocation out-of-memory error on this host. The local profiler
therefore defaults to the measured 256 MiB configuration. Production code
must budget pinned memory from available host and GPU pressure rather than
assuming that the upstream default is safe.

These are local storage results, not fal `/data` results. The approximately
56x gap between cold safetensors (59.31 s) and warm safetensors (1.06 s) shows
that this WSL profile is storage-bound. fal documents up to 25 Gbit/s for
FlashPack on its infrastructure, but that number must not be presented as this
model's measured startup speed until the same profiler runs inside the target
fal machine.

## What FlashPack and ComfyUI contribute

FlashPack flattens a state dictionary into large dtype-grouped blocks, reads
chunks in parallel, overlaps host reads with CUDA copies, and creates parameter
views without a second GPU allocation. fal's persistent `/data` cache makes the
packed file reusable across runners and deployments.

Current ComfyUI adds a complementary set of mechanisms:

- read-only safetensors memory maps annotated with exact file offsets;
- direct file-slice-to-device reads where AIMDO is available;
- bounded host buffers and asynchronous device copies otherwise;
- pressure-aware pinned-memory registration and eviction;
- model deduplication, residency, partial unload, and reuse;
- module-ahead prefetch with stream synchronization;
- two asynchronous offload streams by default on supported NVIDIA systems.

ComfyUI's dynamic-VRAM path is primarily a memory-capacity and model-switching
feature. For a 2B checkpoint that fits comfortably on a 24 GB GPU, eagerly
loading the complete pack once and retaining the SGLang process minimizes first
request latency. Lazy layer materialization should be an explicit low-VRAM or
multi-model mode, not the fast default.

## Proposed combined loader: FlashSlice

1. Convert the pinned checkpoint revision to one FlashPack file during image
   build or a one-time `/data` preparation job. Store its index, checksum,
   dtype, model revision, FlashPack revision, Torch version, and CUDA version.
2. Instantiate the model with empty/meta parameters and map each parameter to
   the packed file's offset, borrowing ComfyUI's `TensorFileSlice` abstraction.
3. For the latency path, eagerly stream the entire pack through a bounded pool.
   Start with a 256 MiB budget, four read workers, two buffers per worker, and
   two CUDA copy streams; autotune against the target machine and checkpoint.
4. Pipeline file read, host staging, H2D copy, parameter binding, and runtime
   initialization. Never allocate a second full GPU state dictionary.
5. Keep the initialized SGLang engine resident and reuse it for every ComfyUI
   execution. Do not reconstruct the engine per graph run.
6. For low-VRAM or rapid model switching, retain the file-offset map and enable
   ComfyUI-style layer-ahead prefetch, bounded pinning, and pressure-aware
   eviction. Record this as a distinct runtime because its first-request shape
   differs from the eager path.

```text
/data packed checkpoint
        |
        v
bounded parallel reads --> pinned ring --> 2 CUDA streams --> empty parameters
        |                                                        |
        +------ file offsets for optional lazy/prefetch mode -----+
                                                                 |
                                                                 v
                                                  resident SGLang engine
```

## End-to-end startup ladder

Every deployment benchmark should emit timestamps for these phases. A single
"cold start" duration is not actionable.

| Mark | Phase | Optimization |
| --- | --- | --- |
| T0 | request accepted | client region, upload size, connection reuse |
| T1 | runner allocated | fal `min_concurrency`, `keep_alive`, capacity |
| T2 | imports complete | small image, pinned dependencies, lazy imports |
| T3 | checkpoint available | persistent `/data`, checksum hit, no download |
| T4 | model skeleton ready | empty/meta initialization |
| T5 | weights resident | bounded FlashPack/FlashSlice pipeline |
| T6 | kernels ready | synchronized Inductor cache, GPU/version key |
| T7 | serving ready | in-process engine or explicit readiness barrier |
| T8 | first token | preprocessed fixed shape, CUDA graph/compile cache |
| T9 | final token | existing SGLang steady-state benchmark |

Recommended production sequence:

1. Measure a true zero-runner fal cold start and a `/data`-cached cold start.
2. Add the bounded packed loader; accept it only with exact tensor and output
   gates.
3. Persist the compiled Inductor cache and warm the real 448-edge, batch-one
   image/decode shape during setup.
4. Reuse the model process. For latency-critical traffic, compare
   `min_concurrency=1` against cost; for sporadic traffic, start with a longer
   `keep_alive` such as 300 seconds and measure the hit rate.
5. Stream output so perceived latency follows TTFT, resize media before upload,
   and avoid base64 copies when a region-local URL is available.

## Primary sources

- [fal FlashPack optimization](https://fal.ai/docs/documentation/serverless/optimizations/flashpack)
- [fal cold-start phases](https://fal.ai/docs/documentation/serverless/optimizations/optimize-cold-starts)
- [fal compiled-cache synchronization](https://fal.ai/docs/documentation/serverless/optimizations/optimize-startup-with-compiled-caches)
- [fal cold-start scaling controls](https://fal.ai/docs/documentation/serverless/optimizations/cold-start-scaling)
- [fal parallel file loading](https://fal.ai/docs/documentation/serverless/optimizations/parallel-file-loading)
- [FlashPack source](https://github.com/fal-ai/flashpack)
- [ComfyUI tensor loading and mmap metadata](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/utils.py)
- [ComfyUI model residency and loading](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py)
- [ComfyUI file-slice-to-device pipeline](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/memory_management.py)
- [ComfyUI module prefetch](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_prefetch.py)
- [ComfyUI bounded pinned memory](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/pinned_memory.py)
