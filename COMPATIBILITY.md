# Platform and accelerator compatibility

ComfyUI owns PyTorch. This node pack deliberately does not depend on `torch`,
`torchvision`, or a vendor wheel, because installing a generic PyPI build can
silently replace a working CUDA, ROCm, XPU, or Metal environment.

Install `requirements.txt` with the same Python executable that starts ComfyUI.
The **VLM Runtime Diagnostics** node reports the environment seen by the pack
without downloading a model.

## Support matrix

| Platform | Managed Transformers | bitsandbytes 4/8-bit | GGUF acceleration |
| --- | --- | --- | --- |
| Linux + NVIDIA | CUDA, BF16/FP16 capability detected | Official wheel | CUDA or Vulkan |
| Windows + NVIDIA | CUDA, BF16/FP16 capability detected | Official wheel | CUDA or Vulkan |
| Linux + AMD | ROCm through PyTorch's `cuda` API | Official ROCm wheel for listed GPU architectures | ROCm/HIP or Vulkan |
| Windows + AMD | Current ComfyUI/AMD ROCm PyTorch builds | Official ROCm Windows wheel for listed GPU architectures | HIP Radeon or Vulkan |
| Apple Silicon macOS | MPS, BF16 on supported macOS/PyTorch; FP16 fallback | Official arm64 wheel | Metal |
| Intel GPU | XPU with BF16 capability detection | Official XPU/CPU wheel | SYCL or Vulkan |
| CPU | FP32 | Official wheels on supported architectures | OpenBLAS or default CPU |
| Intel macOS | CPU/legacy MPS environment as provided by ComfyUI | No official bitsandbytes wheel; dependency is skipped | CPU build |

The default **ComfyUI managed** mode is the portable path. Quantization is an
optional optimization, not an import requirement. DirectML/private-use devices
receive a safe FP32 fallback, but are best-effort because current ComfyUI itself
does not treat DirectML as a primary performance backend.

## Detection and segmentation backends

The structured vision nodes do not install a second PyTorch build. Grounding
DINO, OWLv2, OmDet Turbo, Florence-2, and SAM2.1 use the device selected by
ComfyUI and participate in its model loading/offloading lifecycle. The core
SAM3.1 adapter performs schema validation and report generation on the compact
core payload; ComfyUI itself owns SAM3 inference and mask packing.

| Backend | Detection / Florence | SAM2.1 video | Comfy core SAM3.1 | Practical limitation |
| --- | --- | --- | --- | --- |
| NVIDIA CUDA | Managed BF16 when supported, otherwise FP16 | Preferred accelerated path; CPU state/storage is the default | Supported when the installed ComfyUI version recognizes the checkpoint | Resolution, frame count, and object count still dominate VRAM/RAM |
| AMD ROCm on Linux | Uses PyTorch's `cuda` device and BF16/FP16 capability checks | Same managed path; keep inference state on CPU unless measured otherwise | Follows ComfyUI core ROCm support | Individual Transformers kernels may fall back or differ in performance |
| AMD ROCm on Windows | Uses the device exposed by the selected ComfyUI PyTorch build | Same API contract | Follows that ComfyUI build | Treat as hardware-validation pending, not equivalent to a Linux ROCm pass |
| Apple Metal / MPS | FP16, or BF16 only when macOS/PyTorch report support | Supported contract with CPU video storage; use Tiny and short slices first | Follows ComfyUI core MPS support | Unified memory is shared with the OS; unsupported operators may fall back to CPU |
| Intel XPU | BF16/FP16 capability-selected managed path | Supported contract; use CPU state for portability | Follows ComfyUI core XPU support | Model-specific operator coverage and real throughput require hardware validation |
| CPU | FP32 portable path | Functionally supported but slow; use Tiny, low resolution, and short slices | Adapter/report works; core SAM3 inference is memory intensive | No half-precision speed assumption and no accelerator kernel |

`precision=auto` is the safe default for open-vocabulary detection and SAM2.1.
Explicit BF16 silently falls back to FP16 or FP32 when the selected backend
cannot execute BF16. This is a portability fallback, not proof that every
model family has been run on every vendor device. See
[MODEL_VALIDATION.md](MODEL_VALIDATION.md) for real-hardware evidence.

### Moondream 3 / 3.1 Photon

Moondream Photon is deliberately isolated from ComfyUI's main Python environment
because `moondream==1.3.0` requires Pillow 10 while current ComfyUI uses a
newer Pillow. Its worker cache, virtual environment, and logs live under
`models/LLavacheckpoints/moondream31-runtime`; it never replaces ComfyUI's
PyTorch or Pillow.

| Platform | Official local Photon support | This integration |
| --- | --- | --- |
| Linux/WSL + NVIDIA Ampere or newer | Supported | 3.1 query/caption/detection/pointing; 3 Preview SVG segmentation |
| Windows + NVIDIA Ampere or newer | Supported | Same isolated worker contract |
| Apple Silicon macOS 13+ | Supported with MPS | Same contract; use a conservative KV-cache profile on low-memory systems |
| AMD ROCm, Intel GPU, CPU | Not currently provided upstream | Node stays importable and fails before model work with an actionable support message |

The final Moondream 3.1 model card lists query, caption, detect, and point; it
does not list segment. Native SVG segment uses `moondream3-preview`, and the
loader rejects a 3.1/segment mismatch before inference.

`max_batch_size` controls Photon's scheduler capacity. The detection, point,
and preview-segmentation nodes issue `parallel_requests` frame requests concurrently,
allowing Photon to build GPU batches. `frame_stride` bounds work for high-frame
rate sources. Performance JSON records warm worker time, end-to-end time,
processed/skipped frames, worker/sustained FPS, target sampled FPS, and
real-time factor; it is a measurement from the current run, not a universal
benchmark claim.

### Video memory and chunking

- Core `Video Slice` should bound work before `GetVideoComponents` materializes
  frames. Scale the resulting `IMAGE` batch before running detection or
  segmentation.
- Open-vocabulary detection runs frame by frame. SAM2.1 keeps source frames on
  CPU, defaults its inference state to CPU, and caches at most one vision
  feature in the video session.
- SAM2.1 output masks and previews are CPU tensors. Core SAM3 keeps its track
  masks bit-packed; `VLMSAM3TrackAdapter` does not unpack the complete volume.
- `unload_after=true` releases the node's owned detector/SAM2 model after a
  run. Leave it false for repeated work with one model; set it true before a
  different large family must load on a constrained accelerator.
- Each slice or queue run starts a new propagation/tracking session. Carrying
  an ID across independent chunks requires an explicit application-level
  overlap/reconciliation step; the nodes never claim cross-run identity.

### Model licenses and access

Model licenses are independent from this repository's code license. Check the
model card before redistributing weights or outputs.

- The `facebook/sam2.1-hiera-*` Transformers checkpoints are published under
  Apache-2.0.
- Meta SAM3 uses the SAM License. The upstream `facebook/sam3` repository is
  access-gated and asks the Hugging Face account holder to accept its terms and
  share the requested contact information.
- ComfyUI's `Comfy-Org/sam3.1` checkpoint is marked `sam-license`; the example
  expects `sam3.1_multiplex_fp16.safetensors` under
  `ComfyUI/models/checkpoints`.
- `HF_TOKEN` is used when Hugging Face requires authenticated access. Tokens
  must be supplied by the environment and must not be embedded in workflows.
- Moondream 3.1 uses the Moondream Model License 1.0. The Loader requires an
  explicit workflow acknowledgement. The license permits local product use
  but restricts offering general-purpose hosted Moondream access; review the
  current upstream terms for the intended deployment.

Authoritative references:

- [Meta SAM3 model and access terms](https://huggingface.co/facebook/sam3)
- [Meta SAM3 license](https://huggingface.co/facebook/sam3/blob/main/LICENSE)
- [ComfyUI SAM3.1 checkpoint](https://huggingface.co/Comfy-Org/sam3.1)
- [SAM2.1 Hiera Tiny model card](https://huggingface.co/facebook/sam2.1-hiera-tiny)
- [Moondream 3.1 model card](https://huggingface.co/moondream/moondream3.1-9B-A2B)
- [Moondream Model License 1.0](https://moondream.ai/licenses/model/1.0)

## Dependency behavior

- Python 3.10 through 3.13 is covered by CI.
- `transformers>=5.4,<6` and `huggingface-hub>=1.5,<2` are paired intentionally;
  Transformers 5.4 requires Hub 1.5 or newer.
- `bitsandbytes>=0.50` is the first dependency floor used here for the current
  multi-backend releases. Environment markers prevent an unsupported wheel
  from blocking the whole node pack.
- `requirements-quantization.txt` is available for an explicit quantization
  install or source-build environment.
- `requirements-moondream31.txt` belongs only in the isolated Photon sidecar;
  installing it into ComfyUI's environment would create a Pillow conflict.
- Model downloads, imports, and package compilation never occur during node
  discovery.

## Robotics / VLA policy compatibility

ComfyUI's robotics schemas, safety gate, trajectory tools, and universal HTTP
client run wherever this node pack runs. Policy runtime compatibility is
separate:

| Policy route | ComfyUI client | Policy environment | Practical boundary |
| --- | --- | --- | --- |
| Universal VLA HTTP | Windows, Linux, macOS; CUDA, ROCm, Metal, XPU, CPU | Any host that implements `comfyui-vla-http-v1` | Loopback HTTP or trusted HTTPS; no pickle |
| LeRobot sidecar | Same universal client | Current LeRobot supports Linux, Windows, and macOS; individual policy extras/operators vary | Python/PyTorch live outside ComfyUI; fine-tuned checkpoint required for the target embodiment |
| openpi WebSocket | Lightweight optional client on every ComfyUI platform | Upstream currently tests Ubuntu 22.04 + NVIDIA, inference above 8 GB VRAM | Use WSL/Docker/Linux server; remote transport must be WSS |
| Isaac-GR00T N1.7 ZMQ | Lightweight optional client on every ComfyUI platform | NVIDIA CUDA/Jetson Linux according to upstream deployment matrix | ZMQ has no transport encryption; use a private network/tunnel |
| OpenVLA-OFT | Universal client with a project-specific bridge | Upstream PyTorch/CUDA environment | OFT is the preferred high-frequency multi-image OpenVLA route |
| Octo | Universal client with a project-specific bridge | Isolated JAX environment | Kept as a lightweight research baseline, not the default maintained runtime |

Install only the native client protocols into ComfyUI:

```bash
python -m pip install -r requirements-robotics-client.txt
```

Do not install `lerobot[all]`, openpi, Isaac-GR00T, OpenVLA, or JAX into
ComfyUI's Python. The included LeRobot HTTP sidecar belongs in its own
environment and optionally moves its owned policy to CPU after an idle
interval. It does not flush ComfyUI's accelerator cache.

An embodiment profile is a workflow contract, not a hardware certification.
The supplied profiles are visibly labeled templates. Before real deployment,
replace action bounds/deltas with the trained dataset's semantics and the
manufacturer/controller limits. ComfyUI never opens ROS, serial, CAN, or robot
SDK transports.

Install manually:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI_VLM_nodes/requirements.txt
```

If quantization was skipped but the machine has a supported custom build:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI_VLM_nodes/requirements-quantization.txt
```

## llama.cpp / GGUF

`llama-cpp-python` must be compiled or selected for the actual backend. Its
official project currently publishes backend indexes and documents source
build flags:

```bash
# NVIDIA; choose a wheel supported by the installed driver.
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Apple Metal
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal

# Linux ROCm
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/rocm72

# Linux or Windows Vulkan
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
```

If an Apple Metal wheel is unavailable or fails archive validation, build the
same optional requirement from source:

```bash
CMAKE_ARGS="-DGGML_METAL=on" python -m pip install \
  --no-cache-dir --no-binary llama-cpp-python \
  -r ComfyUI/custom_nodes/ComfyUI_VLM_nodes/requirements-llama-cpp.txt
```

The official Windows HIP Radeon index is:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/hip-radeon
```

Source builds use `GGML_CUDA=on`, `GGML_METAL=on`, `GGML_HIP=on`,
`GGML_VULKAN=on`, or `GGML_SYCL=on` through `CMAKE_ARGS`. Use an arm64 Python
on Apple Silicon; an x86 Python builds the wrong architecture and is
dramatically slower.

The llama.cpp wheel is an independent native runtime; it does not have to use
the same accelerator API as ComfyUI's PyTorch wheel. For example, a Vulkan
llama.cpp wheel can coexist with a CUDA or CPU PyTorch build. The nodes query
`llama_supports_gpu_offload`, `llama_supports_mmap`, and llama.cpp's system
information at runtime. They never label a wheel CUDA/ROCm/Metal based only on
`torch`.

### GGUF runtime controls

- `gpu_layers=-1` requests full accelerator offload. A build that reports no
  offload support is automatically clamped to `0` and continues on CPU.
- `n_batch` is the logical prompt batch and `n_ubatch` is the physical
  micro-batch. The runtime clamps both to the selected context and guarantees
  `n_ubatch <= n_batch`.
- **Auto** flash attention enables the optimized path for accelerator offload
  and retries once without it only when llama.cpp reports an attention-related
  initialization failure. **Enabled** remains strict; **Disabled** is the
  maximum-compatibility setting.
- `use_mmap` is honored only when the compiled backend reports mmap support.
- Layer, row, and single-device split modes plus `main_gpu` and
  comma-separated `tensor_split` weights are passed through when supported by
  the installed binding. Parallel multi-GPU is primarily a CUDA/ROCm feature;
  Vulkan and SYCL support is more limited.
- Current multimodal GGUFs should use **Auto (GGUF chat template)**, which maps
  to llama.cpp's MTMD handler. Named legacy handlers remain selectable for
  model cards that require an exact prompt format.
- Every model handle is lazy, mutex-protected, cache-keyed by all performance
  settings, and closes its exact model and projector handler on unload.

Authoritative installation references:

- [ComfyUI installation and hardware backends](https://github.com/Comfy-Org/ComfyUI)
- [bitsandbytes installation and supported hardware](https://huggingface.co/docs/bitsandbytes/installation)
- [llama-cpp-python supported backends](https://github.com/abetlen/llama-cpp-python#supported-backends)
- [llama-cpp-python API reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/)
- [llama.cpp backend feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)

## Attention and offloading

- **Auto (SDPA)** lets PyTorch choose its maintained kernel and is the default
  on every backend.
- **Flash Attention 2** is preflighted for CUDA/ROCm only. A compatible
  `flash-attn` build is still required.
- ComfyUI-managed models participate in its normal model patcher lifecycle.
- External bitsandbytes and llama.cpp allocations ask ComfyUI to free space
  first, then release only their owned model on unload.
- Automatic CPU/disk device mapping is used for large CUDA/ROCm/XPU models.
  MPS unified memory and CPU use an explicit active-device map.
- AudioLDM2 uses FP16 on capable accelerators, FP32 on CPU, CUDA-API CPU
  offload for NVIDIA/ROCm, and a portable CPU random generator on MPS.

## What CI proves

Every push installs current ComfyUI plus this complete `requirements.txt` and
runs imports, schemas, runtime contracts, tests, and byte-compilation on:

- Ubuntu, Python 3.10
- Ubuntu, Python 3.13
- Windows, Python 3.12
- macOS, Python 3.12

Hosted runners do not contain production NVIDIA, AMD, or Intel GPUs. CI
therefore tests backend selection and dtype/device-map contracts, while real
GPU model smoke tests remain explicit hardware validation. It does not claim
that a CPU simulation executed a vendor kernel.
