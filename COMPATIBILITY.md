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

## Dependency behavior

- Python 3.10 through 3.13 is covered by CI.
- `transformers>=5.4,<6` and `huggingface-hub>=1.5,<2` are paired intentionally;
  Transformers 5.4 requires Hub 1.5 or newer.
- `bitsandbytes>=0.50` is the first dependency floor used here for the current
  multi-backend releases. Environment markers prevent an unsupported wheel
  from blocking the whole node pack.
- `requirements-quantization.txt` is available for an explicit quantization
  install or source-build environment.
- Model downloads, imports, and package compilation never occur during node
  discovery.

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
