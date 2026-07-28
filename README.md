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

## Structured detection, segmentation, and tracking

The vision nodes use stable, typed sockets instead of passing model-specific
lists between nodes:

| Socket | JSON schema | Purpose |
| --- | --- | --- |
| `VLM_DETECTIONS` | `comfyui-vlm/detections`, version 1 | Per-frame boxes, labels, scores, optional polygons/quads, and in-process masks |
| `VLM_TRACKS` | `comfyui-vlm/tracks`, version 1 | Durable object IDs with ordered observations over time |
| `VLM_POINTS` | `comfyui-vlm/points`, version 1 | Pixel-coordinate points, including detection centers |
| `VLM_EVENTS` | `comfyui-vlm/events`, version 1 | Ordered temporal events for downstream video analysis |

All spatial coordinates are source-image pixels. Bounding boxes are
`[x1, y1, x2, y2]` with an exclusive right/bottom edge; polygons contain at
least three points and quads exactly four. JSON roots contain `schema`,
`version`, media dimensions/frame count/FPS, and their ordered records. Dense
mask tensors remain in-process and are deliberately omitted from JSON so API
results do not unexpectedly grow by hundreds of megabytes.

The utility layer converts without model-specific glue:

- `VLMStructuredSpatialParser` strictly parses pixel, normalized 0–1, or
  normalized 0–1000 JSON from any VLM into `VLM_DETECTIONS` and `VLM_POINTS`.
  `VLMSpatialPromptBuilder` creates the matching constrained prompt.
- `VLMDetectionsToBoundingBoxes`, `VLMDetectionsToPoints`, and
  `VLMDetectionsToMasks` emit Comfy core boxes, center points, and union plus
  individual masks. Polygon/quad masks are rasterized when present, otherwise
  the bounding box is used.
- `VLMFilterDetections`, `VLMSelectDetection`, `VLMCropDetections`, and
  `VLMRenderDetections` provide label/score/area/frame selection, padded crops,
  and deterministic overlays.
- `VLMDetectionsFromJSON` and `VLMDetectionsToJSON` are the explicit API and
  persistence boundary for the versioned detection schema.

### Open-vocabulary image and video detection

`VLMOpenVocabularyDetection` exposes one interface for:

- Grounding DINO Tiny and Base
- OWLv2 Base Ensemble
- OmDet Turbo Swin Tiny

It accepts a still image or an `IMAGE` batch of video frames and processes the
batch frame by frame. Outputs, in socket order, are `detections`, `json`,
`preview`, `box_mask`, and Comfy core `bounding_boxes`. Connect the FPS output
of `GetVideoComponents` when the input is video so every timestamp is correct.
For tracking-by-detection, run detection over the complete bounded batch and
connect it to `VLMTrackDetections`.

`VLMTrackDetections` uses a ByteTrack-style two-stage high/low-confidence
association, motion prediction, label-aware matching, and time-based expiry.
IDs are durable within the supplied sequence and survive short missed
detections when `emit_predictions` is enabled. Independent Comfy queue runs or
independently sliced chunks are separate tracking sessions; they do not
silently reuse IDs.

### SAM2.1 and Comfy core SAM3.1

`VLMSAM2VideoSegmentation` propagates first-frame detections, one core
`BOUNDING_BOX`, or seed masks through an `IMAGE` batch using SAM2.1 Hiera Tiny,
Small, Base+, or Large. It returns `VLM_TRACKS`, report JSON, per-frame union
masks, frame-major individual object masks, and an overlay batch. The object
IDs assigned at the seed frame remain stable for that video session.

`VLMSAM3TrackAdapter` is intentionally an adapter, not a second SAM3 loader. It
validates ComfyUI core `SAM3_TRACK_DATA`, preserves the core bit-packed mask
payload unchanged, and exposes lightweight `VLM_TRACKS` metadata with mask
references. Connect its passthrough output to core `SAM3_TrackPreview` or
`SAM3_TrackToMask`, and connect `tracks` to `VLMTrackReport`. This avoids
duplicating dense masks in memory or JSON.

SAM3 weights use Meta's SAM License. The upstream `facebook/sam3` repository
requires accepting access terms and sharing the requested account information;
the ComfyUI checkpoint is also marked `sam-license`. Review and accept the
license before downloading. The example names ComfyUI's
`sam3.1_multiplex_fp16.safetensors`; if it is unavailable, use the SAM2.1
workflow rather than substituting an unrelated checkpoint.

### Florence-2 task coverage

`Florence2` exposes all 15 supported task contracts:

| Task | Extra input | Structured result |
| --- | --- | --- |
| Caption | none | text |
| Detailed caption | none | text |
| More detailed caption | none | text |
| OCR | none | text |
| OCR with regions | none | text plus quadrilateral regions |
| Object detection | none | labeled boxes |
| Dense region caption | none | captions with boxes |
| Caption to phrase grounding | `text_input` | phrase boxes |
| Referring expression segmentation | `text_input` | polygons and mask |
| Region to segmentation | one `BOUNDING_BOX` per image | polygons and mask |
| Open vocabulary detection | `text_input` | model-provided spatial records |
| Region to category | one `BOUNDING_BOX` per image | text |
| Region to description | one `BOUNDING_BOX` per image | text |
| Region to OCR | one `BOUNDING_BOX` per image | text |
| Region proposals | none | boxes |

Every task returns `text`, `structured_json`, `mask`, and `visualization`.
Tasks that do not produce a spatial result return an empty mask and the source
image visualization. Region tasks reject ambiguous multi-box input; use
`VLMSelectDetection` to isolate the record, then supply exactly one core
`BOUNDING_BOX` with the same pixel coordinates.

### Video memory strategy

- Trim long media with core `Video Slice`, then use `GetVideoComponents`.
  Downscale the complete frame batch before detection or segmentation and keep
  every frame at identical dimensions.
- Grounding detection supports configurable micro-batches; keep `batch_size=1`
  for minimum VRAM or increase it when memory allows. It returns both nested
  per-frame core `BOUNDING_BOX` values and flat metadata-rich
  `BOUNDING_BOXES`.
- SAM2.1 stores source video frames on CPU, keeps its inference state on CPU by
  default, and limits the vision-feature cache to one frame. Union masks and
  previews return on CPU. Full per-object mask volumes are opt-in with
  `mask_output=union_and_objects`; disable `render_preview` to avoid another
  full-resolution overlay copy on long clips.
- Start with Grounding DINO Tiny plus SAM2.1 Hiera Tiny. Increase detector or
  segmenter size only after the pipeline is correct. `unload_after=false`
  caches one model per node instance; use `true` when another large model must
  run immediately afterward.
- A `Video Slice` is an independent propagation session. For very long media,
  use bounded slices, reseed each slice, and keep the overlap/output mapping in
  the caller. The pack does not pretend IDs are globally stable across separate
  queues.
- The SAM3 adapter never unpacks the complete mask volume for its report. Use
  core `SAM3_TrackToMask` only when a dense selected mask is actually needed.

API-format examples are in [`examples/vision`](examples/vision):

- [`grounding_dino_image_api.json`](examples/vision/grounding_dino_image_api.json)
- [`sam2_video_tracking_api.json`](examples/vision/sam2_video_tracking_api.json)
- [`sam3_core_adapter_blueprint_api.json`](examples/vision/sam3_core_adapter_blueprint_api.json)

Upload the named media to ComfyUI's input directory, adjust the filenames and
labels, then submit the JSON object as the `prompt` value to `/prompt`. These
are API graphs, not frontend workflow-export JSON.

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
