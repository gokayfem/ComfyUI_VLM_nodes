# ComfyUI VLM Nodes

Production-oriented vision-language, structured prompting, audio, and utility
nodes for ComfyUI. Version 2.3 supports ComfyUI's selected NVIDIA CUDA, AMD
ROCm, Apple Metal, Intel XPU, and CPU device without replacing its PyTorch
build. It removes startup installers and global accelerator cache flushes,
adds real image/video batches and live token streaming, and uses ComfyUI model
residency and offloading.

## Modern model coverage

The **Modern VLM** node provides one stable interface for:

- Qwen 3.5 0.8B, 2B, 4B, 9B, 27B, and 35B-A3B
- Qwen 3.6 27B
- Qwen 3 VL 2B, 4B, 8B, and 30B-A3B Instruct
- Qwen 2.5 VL 3B and 7B for existing workflows
- Gemma 3 4B, 12B, and 27B IT
- SmolVLM2 256M, 500M, and 2.2B video models
- Liquid LFM2.5-VL 450M and 1.6B edge models
- InternVL 3.5 1B and 2B standard Hugging Face checkpoints
- Granite Vision 3.3 2B and 4.1 4B for documents, charts, and OCR
- a compatible custom Hugging Face image-to-text repository

Sixteen curated sub-4B/low-VRAM choices are marked internally as the
small-and-fast tier. The default is Qwen 3 VL 2B: it is much quicker to load
than larger checkpoints while retaining broad image and video understanding.
The catalog intentionally uses official model repositories and maintained
Transformers interfaces rather than unverified community quantizations.
Curated models use native Transformers implementations; remote repository code
is enabled only when the explicit custom-model option requires it. Florence-2
uses the Transformers-native converted checkpoints instead of Microsoft’s
legacy repository code.

## Live text output

`Modern VLM` streams decoded text through ComfyUI's native `progress_text`
WebSocket channel by default. A connected `ViewText` node updates while tokens
arrive, shows the final response after execution, and restores the last result
when ComfyUI rehydrates workflow output history. Disable `stream_output` for
API-only or headless runs that do not need incremental UI updates. Streaming is
best-effort and never changes the final `STRING` output or makes inference fail.

Specialized nodes remain available where a generic chat node would discard
useful model capabilities:

- **Florence-2**: captioning, OCR, detection, region captioning, and referring
  expression segmentation, with structured JSON, mask, and overlay outputs.
- **PaLI-Gemma**: caption/VQA plus the official 16-token VQ-VAE segmentation
  decoder; segmentation tokens are no longer misinterpreted as polygon points.
- **Moondream2**: pinned query API with explicit decoding controls. Its current
  checkpoint is not marked passed on the tested Torch/Transformers stack; use a
  small Modern VLM preset for production.
- **Qwen2-VL**: image batches and real video-frame batches.
- **Molmo, Kosmos-2, UForm, MCLLaVA, JoyTag, and MiniCPM-V 2.6 GGUF**.
- **llama.cpp LLaVA/GGUF**, structured prompt suggestions, OpenAI-compatible
  prompting, and AudioLDM2.

## Install

Install through ComfyUI Manager, or clone into `ComfyUI/custom_nodes` and run:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI_VLM_nodes/requirements.txt
```

Run that command with ComfyUI's Python. Do not install or replace `torch` from
this repository: ComfyUI's own installer selects CUDA, ROCm, XPU, Metal, or CPU.
Current official bitsandbytes wheels are installed automatically only on their
supported OS/architecture combinations. Unsupported machines retain all
non-quantized nodes.

GGUF nodes use optional `llama-cpp-python`. Install a wheel built for the
desired CUDA, ROCm/HIP, Metal, Vulkan, SYCL, or CPU backend:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI_VLM_nodes/requirements-llama-cpp.txt
```

See [COMPATIBILITY.md](COMPATIBILITY.md) for the tested matrix and official
backend-specific GGUF commands.

The GGUF loaders now query the installed llama.cpp build instead of inferring
its capabilities from PyTorch. Accelerator offload automatically falls back to
CPU when a CPU-only wheel is installed. Advanced optional inputs expose logical
and physical prompt batching (`n_batch`/`n_ubatch`), flash-attention policy,
mmap, and CUDA/ROCm multi-GPU layer/row splitting without changing legacy
workflow sockets. `Auto` flash attention retries the portable path if a
backend/model pair rejects it.

The **LLaVA Vision Projector Loader** supports metadata-driven MTMD plus
explicit handlers for LLaVA 1.5/1.6, MiniCPM-V 2.6, Moondream2, NanoLLaVA,
Qwen2.5-VL, Gemma 4, Llama 3 Vision Alpha, and Obsidian. Use the default
metadata-driven handler for current GGUF + mmproj pairs; select the named
legacy handler when a model card requires it.

Models are downloaded only when their node first executes and are stored below
`ComfyUI/models/LLavacheckpoints`. Hugging Face downloads respect `HF_TOKEN`.
Gemma 3 and PaLI-Gemma require accepting their model licenses on Hugging Face.

## GPU lifecycle

- **ComfyUI managed (BF16)** is the default and preferred path. BF16 is used
  only when the active device reports support; otherwise the node safely falls
  back to FP16 on CUDA/ROCm/Metal/XPU or FP32 on CPU.
- **4-bit/8-bit** models and llama.cpp own external allocators. Before loading,
  the nodes ask ComfyUI to free the required space; unloading closes the exact
  owned model and then requests a soft cache cleanup. Small quantized models
  stay on ComfyUI's active device instead of assuming GPU zero. Large-model
  Accelerate placement is enabled on CUDA/ROCm/XPU; any disk offload remains
  inside the model's ComfyUI directory.
- llama.cpp model and projector bytes are included in the pre-load reservation.
  The runtime reports llama.cpp's own compiled backend, GPU-offload, mmap, and
  mlock capabilities in **VLM Runtime Diagnostics**.
- `unload_after=false` caches one model per node instance for fast repeated
  queues. Turn it on for maximum reclamation between prompts.
- A connected `video_frames` batch becomes the primary visual input. The
  optional still-image socket is ignored for video inference so smaller models
  cannot silently answer from the wrong media.
- Qwen 3.5/3.6 thinking is off by default for lower latency and predictable
  output length; enable it explicitly for tasks that benefit from visual
  reasoning.
- **Auto (SDPA)** is portable and preferred. Flash Attention 2 is accepted only
  on supported CUDA/ROCm builds and otherwise fails before model loading.
- **VLM Runtime Diagnostics** produces a zero-download JSON report containing
  OS, Python, PyTorch, backend, dtype capability, and optional package versions.
- Visualization-only companion repositories do not allocate accelerator memory.

Avoid placing several independently quantized VLMs in one workflow unless the
GPU can hold them. On a 24 GB card, Qwen 3 VL 2B is the fast default,
Qwen 3 VL 8B fits in BF16, and larger models should use NF4. Qwen 3.5/3.6 can
be substantially slower when their optional optimized linear-attention kernels
are not available for the installed PyTorch/backend combination.

## API nodes

`PromptGenerateAPI` supports the current OpenAI Responses API, the legacy Chat
Completions API, and compatible base URLs. API keys can be supplied by node or
environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`GROQ_API_KEY`). Keys are never persisted by this repository.

## Reliability guarantees

- Importing the pack performs no network access, compilation, or package install.
- Missing optional backends fail only the node that needs them, with an
  actionable error.
- Image inputs use ComfyUI `BHWC` batches; text responses preserve every batch
  item. Florence/PaLI masks use `BHW`.
- `forceInput` string hacks were removed, preventing frontend widget-index drift.
- Downloads stay inside the configured ComfyUI model directory.
- CI installs and imports the full pack on Linux Python 3.10/3.13, Windows
  Python 3.12, and macOS Python 3.12. Backend contracts for CUDA, ROCm, Metal,
  XPU, and CPU are exercised without pretending hosted CPU runners are GPUs.

Run local checks with:

```bash
PYTHONPATH=/path/to:/path/to/ComfyUI python -m pytest -q
```

Real-weight checks are opt-in because they download multi-gigabyte checkpoints:

```bash
python tests/manual_model_smoke.py --model "Qwen 3 VL 4B Instruct"
python tests/manual_specialized_smoke.py --backend florence-large
python tests/manual_llama_cpp_smoke.py --download
```

See [MODEL_VALIDATION.md](MODEL_VALIDATION.md) for the exact real-weight and
catalog-only evidence matrix.

Please report reproducible bugs at the
[issue tracker](https://github.com/gokayfem/ComfyUI_VLM_nodes/issues).
