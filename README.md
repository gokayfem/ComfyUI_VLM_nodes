# ComfyUI VLM Nodes

Production-oriented vision-language, structured prompting, audio, and utility
nodes for ComfyUI. Version 3.4 supports ComfyUI's selected NVIDIA CUDA, AMD
ROCm, Apple Metal, Intel XPU, and CPU device without replacing its PyTorch
build. It removes startup installers and global accelerator cache flushes,
adds real image/video batches and live token streaming, and uses ComfyUI model
residency and offloading.

## Modern model coverage

The **Modern VLM** node provides one stable interface with a deliberately
small, 12-choice production picker:

- Qwen 3.5 0.8B and 4B
- Qwen 3 VL 2B, 4B, and 8B Instruct
- SmolVLM2 500M and 2.2B Video
- Liquid LFM2.5-VL 450M
- InternVL 3.5 1B
- Granite Vision 4.1 4B
- Gemma 3 4B IT
- a compatible custom Hugging Face image-to-text repository

The separate **[Legacy] Modern VLM Compatibility** node contains redundant,
superseded, experimental, and very large tiers:

- Qwen 3.5 2B, 9B, 27B, and 35B-A3B
- Qwen 3.6 27B
- Qwen 3 VL 30B-A3B Instruct
- Qwen 2.5 VL 3B and 7B for existing workflows
- Gemma 3 12B and 27B IT
- SmolVLM2 256M Video
- Liquid LFM2.5-VL 1.6B
- InternVL 3.5 2B
- Granite Vision 3.3 2B

Previously saved `ModernVLM` workflows remain valid even when their selected
model moved to Legacy. The server accepts every known catalog value for
backward compatibility; only the visible new-workflow picker is curated.
Dedicated Molmo, PaLI-Gemma, Qwen2-VL, MiniCPM-V, Kosmos-2, MC-LLaVA, UForm,
and script-style MoonDream nodes are also collected under
`VLM Nodes/Legacy/Model Loaders`. Maintained creator-facing Florence-2,
Moondream2, JoyTag, llama.cpp/GGUF, detection, segmentation, tracking, API,
and video-intelligence nodes stay in their functional categories.

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

## Text workflow toolkit

The original `SimpleText`, `JsonToText`, and `ViewText` node IDs and their
first `STRING` outputs remain stable for saved workflows. They now live in
organized `VLM Nodes/Text` subcategories and expose descriptive names, search
aliases, tooltips, appended metrics, and strict error messages:

| Node | Purpose |
| --- | --- |
| `Text` (`SimpleText`) | Multiline/dynamic prompt source with optional edge/newline normalization and character, word, and line outputs |
| `View Text (Streaming)` | Read-only live output with counts, copy, UTF-8 download, line wrapping, stream following, reroute traversal, and history rehydration |
| `JSON to Text` | Plain or fenced JSON parsing with readable, values-only, key/value, pretty, and compact render modes |
| `Text Join` | Join up to eight prompt/context values with empty-value removal and stable deduplication |
| `Text Template` | Safe named placeholders from a JSON object plus four convenient live text sockets, with explicit missing-key policy |
| `Text Clean` | Unicode NFC/NFKC, newline/whitespace cleanup, enclosing Markdown-fence removal, line deduplication, and deterministic length caps |
| `Text Replace` | Literal or regex substitution with case, count, and missing-pattern controls |
| `JSON Extract` | JSONPath-lite (`$.items[0]`) and RFC 6901 JSON Pointer extraction from plain or fenced model responses |
| `Text Split / Batch` | Lines, paragraphs, delimiters, regex, CSV, or JSON arrays converted to a real mapped Comfy `STRING` list |
| `Text Inspector` | Pass-through text plus characters, UTF-8 bytes, words, lines, rough token budget, SHA-256, and JSON metadata |

The JSON utilities never evaluate code, follow references, access files, or
make network requests. Template fields are direct names rather than Python
attribute/index expressions. `approx_tokens` is deliberately labeled as a
rough UTF-8 budget estimate; use the target model tokenizer when exact billing
or context accounting matters.

Specialized nodes remain available where a generic chat node would discard
useful model capabilities:

- **Moondream 3.1 9B-A2B**: official 2B-active Photon runtime with query,
  caption, and high-throughput image/video detection and pointing.
- **Moondream 3 Preview segment**: native SVG segmentation through the same
  isolated Photon loader. The SVG is preserved and also converted into antialiased
  `MASK`, black/white previews, foreground cutouts, overlays, polygons,
  canonical `VLM_DETECTIONS`, and core bounding boxes. Detection/pointing
  submit frames concurrently so Photon can dynamically batch them; every run
  reports measured worker FPS, end-to-end FPS, and real-time factor.
- **Florence-2**: captioning, OCR, detection, region captioning, and referring
  expression segmentation, with structured JSON, mask, and overlay outputs.
- **PaLI-Gemma**: caption/VQA plus the official 16-token VQ-VAE segmentation
  decoder; segmentation tokens are no longer misinterpreted as polygon points.
- **Moondream2**: pinned query API with explicit decoding controls. The official
  checkpoint is loaded through its native safetensors state dict, avoiding the
  silent empty-output regression in Transformers 5 while retaining ComfyUI
  managed loading and unloading.
- **Qwen2-VL**: image batches and real video-frame batches.
- **Legacy Molmo, Kosmos-2, UForm, MCLLaVA, and MiniCPM-V 2.6 GGUF**, plus
  maintained JoyTag.
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
| `VLM_VIDEO_SELECTION` | `comfyui-vlm/video-selection`, version 1 | Exact mapping from sampled images to source frame indices and timestamps |
| `VLM_SCENE_STATE` | `comfyui-vlm/scene-state`, version 1 | Compact persistent objects, motion, visibility, and validated events |

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
  `VLMDetectionsToMasks` emit Comfy core boxes, center points, combined and
  individual binary masks, inverse masks, ready-to-preview black-and-white
  images, and stable-color instance maps. Polygon/quad masks are rasterized
  when present, otherwise the bounding box is used. Existing output indexes
  remain stable; the creator-facing mask images and instance map are appended.
- `VLMFilterDetections`, `VLMSelectDetection`, `VLMCropDetections`, and
  `VLMRenderDetections` provide label/score/area/frame selection, padded crops,
  and deterministic overlays.
- `VLMMaskProcessor` accepts any Comfy `MASK`, including SAM2/SAM3 masks, and
  returns a feathered matte, strict binary mask, inverse mask, and
  black-and-white image. Its grow/shrink and Gaussian feathering run in Torch
  without OpenCV or SciPy.
- `VLMMaskComposite` applies still-image or video mask batches to a source and
  returns the replacement composite, isolated foreground, original
  background-only plate, and black-and-white mask image. A single mask or
  background broadcasts safely across a video batch.
- `VLMDetectionsFromJSON` and `VLMDetectionsToJSON` are the explicit API and
  persistence boundary for the versioned detection schema.

### Universal VLM performance utilities

The performance nodes sit before any local or hosted VLM, so their savings do
not depend on CUDA, ROCm, MPS, XPU, CPU, Transformers, llama.cpp, or Photon:

- `VLM Performance Profile` emits coherent `max_frames`, pixel budget,
  longest-edge, batch-size, and `unload_after` values. `Live / robotics`,
  `Fast video`, `Balanced`, `High detail`, and `Low VRAM handoff` are explicit
  starting points rather than hidden global flags.
- `VLM Adaptive Frame Sampler` is the existing track-aware temporal gate. It
  combines uniform coverage, scene changes, motion, and optional track changes
  while preserving source frame indices and timestamps.
- `VLM Image Pixel Budget` downsizes the selected analysis copy once, preserves
  aspect ratio, never upscales, and can align dimensions to 14/28-pixel VLM
  patches or 32-pixel detector backbones. Fast area and antialiased bicubic
  modes are available.

The recommended order is `Video Slice` → `VLM Adaptive Frame Sampler` →
`VLM Image Pixel Budget` → any VLM. A model's own official processor still
performs its required normalization/crop; the pixel-budget node simply prevents
every downstream model from repeatedly receiving unnecessary source pixels.
Local torch models remain registered with ComfyUI's smart model manager, while
external allocators reserve space before loading and close only the handle they
own.

On the real `vlm_api_people_birds.mp4` input in this repository's D-drive test
environment, the utilities selected 10 of 60 1280×720 frames and resized them
to 938×518 in about 0.44 seconds on a cold WSL run. That reduced the
frame×pixel analysis workload by 11.38× before model inference. This is an
input-work reduction measurement, not a claim that every model runs 11.38×
faster; token generation and model-specific vision encoders still determine
end-to-end speed.

### Adaptive video intelligence

The video-intelligence layer keeps generative VLM inference out of the
per-frame loop:

- `VLMAdaptiveFrameSampler` combines scene-change, motion, track-change, and
  uniform-coverage signals. It always preserves the real source frame index
  and timestamp, enforces a frame budget, and returns selection/diagnostic
  JSON. `Uniform coverage`, motion, scene, and track-priority modes remain
  available for deterministic experiments.
- `VLMVideoTemporalReasoner` is the one-node path. It adaptively samples the
  input, downsizes only the VLM analysis copy (448-pixel longest side by
  default), runs a recommended video-capable model, parses the result into
  validated `VLM_EVENTS`, and returns summary, events, selection, sampled
  previews, raw response, diagnostics, event JSON, and selection JSON.
- `VLMVideoReasoningPrompt` and `VLMEventsFromVideoJSON` expose the same strict
  timestamp/evidence contract for custom local or hosted VLM workflows.
- `VLMTrackAwareCrops` chooses representative observations for each durable
  track, adds configurable context, and letterboxes crops to one batch size.
  This lets a VLM label identities without rereading every full frame.
- `VLMBuildSceneState` converts tracks plus optional events into a compact
  persistent world-state summary with first/last observation, current box,
  confidence, state, and pixel velocity.

Small VLMs commonly return evidence as positions in the supplied image batch
even when asked for source indices. The parser accepts that form only when
every value is an unambiguous valid supplied-image position, maps it back to
the immutable source selection, and records the normalization mode. Arbitrary
or unsupplied evidence frames, out-of-range timestamps, invalid confidence,
duplicate evidence, malformed JSON, and non-finite values fail validation.

On the repository's real-data smoke test (RTX 3090, Qwen3-VL 2B, 157-frame
896x448 H.264 clip), hybrid sampling selected 12 frames in 0.30 seconds,
reduced temporal inputs by 92.36%, reduced analysis pixels by 75%, used
4.24 GiB peak allocated VRAM in the standalone runner, and produced a valid
timestamped result in 35.17 seconds. The equivalent live ComfyUI `/prompt`
graph completed in 37.45 seconds. These are one-machine measurements, not
portable performance guarantees.

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
- [`moondream3_preview_svg_segment_api.json`](examples/vision/moondream3_preview_svg_segment_api.json)
- [`moondream31_video_detect_api.json`](examples/vision/moondream31_video_detect_api.json)
- [`sam2_video_tracking_api.json`](examples/vision/sam2_video_tracking_api.json)
- [`sam3_core_adapter_blueprint_api.json`](examples/vision/sam3_core_adapter_blueprint_api.json)
- [`video_temporal_reasoning_api.json`](examples/vision/video_temporal_reasoning_api.json)
- [`vlm_performance_preflight_api.json`](examples/vision/vlm_performance_preflight_api.json)

The dependency-free text-toolkit example is
[`examples/text_toolkit_api.json`](examples/text_toolkit_api.json).
Robotics policy, safety, and sidecar examples are in
[`examples/robotics`](examples/robotics), including a complete universal
HTTP policy graph.

Upload the named media to ComfyUI's input directory, adjust the filenames and
labels, then submit the JSON object as the `prompt` value to `/prompt`. These
are API graphs, not frontend workflow-export JSON.

## Node reference

All 89 registered nodes, grouped by their menu category. The **Node ID** is the
`class_type` written into workflow and API JSON — search for that string when
you need to find a node you saw on a canvas.

### Modern VLM

The main entry point for current vision-language models.

| Node | Node ID | Outputs |
| --- | --- | --- |
| Modern VLM (Qwen / SmolVLM2 / LFM / InternVL / Granite / Gemma) | `ModernVLM` | `STRING` |
| Moondream 2 | `Moondream2model` | `STRING` |

### Moondream 3

Moondream 3 / 3.1 in an isolated Photon runtime. Load once, then reuse the
`MOONDREAM31_MODEL` output across the task nodes.

| Node | Node ID | Outputs |
| --- | --- | --- |
| Moondream 3 / 3.1 Loader (Isolated Photon) | `Moondream31Loader` | `MOONDREAM31_MODEL`, `STRING` |
| Moondream 3 / 3.1 Caption | `Moondream31Caption` | `STRING`, `STRING` |
| Moondream 3 / 3.1 Query | `Moondream31Query` | `STRING`, `STRING`, `STRING` |
| Moondream 3 / 3.1 Detect (Image / Video) | `Moondream31Detect` | `VLM_DETECTIONS`, `STRING`, `IMAGE`, `MASK`, `BOUNDING_BOX`, `BOUNDING_BOXES`, `STRING` |
| Moondream 3 / 3.1 Point (Image / Video) | `Moondream31Point` | `VLM_POINTS`, `STRING`, `IMAGE`, `STRING` |
| Moondream 3 Preview SVG Segment (Image / Video) | `Moondream31Segment` | `VLM_DETECTIONS`, `STRING`, `STRING`, `MASK`, `IMAGE`, `IMAGE`, `IMAGE`, `BOUNDING_BOX`, `BOUNDING_BOXES`, `STRING` |

### Florence-2

| Node | Node ID | Outputs |
| --- | --- | --- |
| Florence-2 Multitask Vision | `Florence2` | `STRING`, `STRING`, `MASK`, `IMAGE` |

### Vision: detection, segmentation, tracking

Open-vocabulary detection and video segmentation. These emit the structured
`VLM_DETECTIONS` / `VLM_POINTS` / `VLM_TRACKS` types rather than loose strings.

| Node | Node ID | Outputs |
| --- | --- | --- |
| VLM Open-Vocabulary Detection | `VLMOpenVocabularyDetection` | `VLM_DETECTIONS`, `STRING`, `IMAGE`, `MASK`, `BOUNDING_BOX`, `BOUNDING_BOXES` |
| VLM SAM2.1 Video Segmentation | `VLMSAM2VideoSegmentation` | `VLM_TRACKS`, `STRING`, `MASK`, `MASK`, `IMAGE` |
| VLM SAM3 Track Adapter | `VLMSAM3TrackAdapter` | `VLM_TRACKS`, `SAM3_TRACK_DATA` |
| VLM Track Detections | `VLMTrackDetections` | `VLM_TRACKS` |
| VLM Track Report | `VLMTrackReport` | `STRING`, `STRING` |
| JoyTag | `Joytag` | `STRING` |

### Vision: spatial reasoning

| Node | Node ID | Outputs |
| --- | --- | --- |
| VLM Spatial Prompt Builder | `VLMSpatialPromptBuilder` | `STRING` |
| VLM Structured Spatial Parser | `VLMStructuredSpatialParser` | `VLM_DETECTIONS`, `VLM_POINTS`, `STRING` |

### Vision: detection utilities

Converters and filters between structured detections and ordinary Comfy types.

| Node | Node ID | Outputs |
| --- | --- | --- |
| Filter VLM Detections | `VLMFilterDetections` | `VLM_DETECTIONS` |
| Select VLM Detection | `VLMSelectDetection` | `VLM_DETECTIONS` |
| Crop VLM Detections | `VLMCropDetections` | `IMAGE`, `STRING` |
| Render VLM Detections | `VLMRenderDetections` | `IMAGE` |
| VLM Detection Centers | `VLMDetectionsToPoints` | `VLM_POINTS`, `STRING` |
| VLM Detections from JSON | `VLMDetectionsFromJSON` | `VLM_DETECTIONS` |
| VLM Detections to JSON | `VLMDetectionsToJSON` | `STRING` |
| VLM Detections to Bounding Boxes | `VLMDetectionsToBoundingBoxes` | `BOUNDING_BOXES`, `STRING` |
| VLM Detections to Masks | `VLMDetectionsToMasks` | `MASK`, `MASK`, `STRING`, `MASK`, `IMAGE`, `IMAGE`, `IMAGE` |

### Vision: mask tools

| Node | Node ID | Outputs |
| --- | --- | --- |
| VLM Mask Processor | `VLMMaskProcessor` | `MASK`, `MASK`, `MASK`, `IMAGE` |
| VLM Mask Composite | `VLMMaskComposite` | `IMAGE`, `IMAGE`, `IMAGE`, `IMAGE` |

### Video intelligence

Adaptive frame selection and temporal reasoning for long videos.

| Node | Node ID | Outputs |
| --- | --- | --- |
| VLM Adaptive Frame Sampler | `VLMAdaptiveFrameSampler` | `IMAGE`, `VLM_VIDEO_SELECTION`, `STRING`, `STRING` |
| VLM Video Reasoning Prompt | `VLMVideoReasoningPrompt` | `STRING`, `STRING` |
| VLM Video Temporal Reasoner | `VLMVideoTemporalReasoner` | `STRING`, `VLM_EVENTS`, `VLM_VIDEO_SELECTION`, `IMAGE`, `STRING`, `STRING`, `STRING`, `STRING` |
| VLM Temporal Events From JSON | `VLMEventsFromVideoJSON` | `VLM_EVENTS`, `STRING`, `STRING` |
| VLM Persistent Scene State | `VLMBuildSceneState` | `VLM_SCENE_STATE`, `STRING`, `STRING` |
| VLM Track-Aware Semantic Crops | `VLMTrackAwareCrops` | `IMAGE`, `STRING` |

### LLM (local GGUF)

llama.cpp text models. `LLM Loader (GGUF)` produces the `CUSTOM` model handle
the samplers consume; the *Managed Cache* variants own their own handle and can
release it after each run.

| Node | Node ID | Outputs |
| --- | --- | --- |
| LLM Loader (GGUF) | `LLMLoader` | `CUSTOM` |
| LLM Sampler | `LLMSampler` | `STRING` |
| LLM Prompt Generator | `LLMPromptGenerator` | `STRING` |
| LLM (Managed Cache) | `LLMOptionalMemoryFreeSimple` | `STRING` |
| LLM (Managed Cache, Advanced) | `LLMOptionalMemoryFreeAdvanced` | `STRING` |
| Structured Output | `StructuredOutput` | `STRING` |
| Structured Keyword Extraction | `KeywordExtraction` | `STRING` |
| Structured Prompt Generator | `LLavaPromptGenerator` | `STRING` |
| Creative Art Prompt Generator | `CreativeArtPromptGenerator` | `STRING` |
| Prompt Suggester | `Suggester` | `STRING` |

### LLaVA (local GGUF multimodal)

Vision models through llama.cpp. These need both a GGUF and its vision
projector (mmproj).

| Node | Node ID | Outputs |
| --- | --- | --- |
| LLaVA Loader | `LLava Loader Simple` | `CUSTOM` |
| LLaVA Vision Projector Loader | `LlavaClipLoader` | `CUSTOM` |
| LLaVA Sampler | `LLavaSamplerSimple` | `STRING` |
| LLaVA Sampler (Advanced) | `LLavaSamplerAdvanced` | `STRING` |
| LLaVA (Managed Cache) | `LLavaOptionalMemoryFreeSimple` | `STRING` |
| LLaVA (Managed Cache, Advanced) | `LLavaOptionalMemoryFreeAdvanced` | `STRING` |

### Hosted APIs

| Node | Node ID | Outputs |
| --- | --- | --- |
| Hosted VLM API (Secure) | `HostedVLMAPI` | `STRING`, `STRING`, `INT` |
| Hosted LLM API (Secure) | `PromptGenerateAPI` | `STRING` |

### Robotics / VLA policies

These nodes build and inspect policy observations/actions. They never send
commands to robot hardware. Heavy policy runtimes stay in isolated LeRobot,
openpi, GR00T, OpenVLA/OFT, or JAX environments.

| Node | Node ID | Outputs |
| --- | --- | --- |
| VLA Embodiment Profile | `VLAEmbodimentProfile` | `VLA_EMBODIMENT`, `STRING`, `INT`, `INT` |
| VLA Observation Builder | `VLAObservationBuilder` | `VLA_OBSERVATION`, `STRING`, `INT` |
| VLA Policy — Universal HTTP | `VLAHTTPPolicy` | `VLA_ACTIONS`, `STRING` |
| VLA Policy — OpenPI WebSocket | `VLAOpenPIWebSocketPolicy` | `VLA_ACTIONS`, `STRING` |
| VLA Policy — GR00T N1.7 ZMQ | `VLAGr00tZMQPolicy` | `VLA_ACTIONS`, `STRING` |
| VLA Action Safety Gate | `VLAActionSafety` | `VLA_ACTIONS`, `STRING`, `BOOLEAN` |
| VLA Actions From JSON | `VLAActionsFromJSON` | `VLA_ACTIONS`, `STRING` |
| VLA Action Chunk Replan | `VLAActionChunkReplan` | `VLA_ACTIONS`, `STRING` |
| VLA Action Inspect | `VLAActionInspect` | `STRING`, `STRING`, `INT`, `INT` |
| VLA Trajectory Preview | `VLATrajectoryPreview` | `IMAGE` |
| VLA Model Catalog | `VLAModelCatalog` | `STRING`, `STRING`, `STRING`, `STRING` |

### Text toolkit

Dependency-free string handling, so a VLM response can be shaped without an
extra node pack.

| Node | Node ID | Outputs |
| --- | --- | --- |
| Text | `SimpleText` | `STRING`, `INT`, `INT`, `INT` |
| Text Join | `VLMTextJoin` | `STRING`, `STRING`, `INT` |
| Text Template | `VLMTextTemplate` | `STRING`, `STRING`, `STRING` |
| Text Clean | `VLMTextClean` | `STRING`, `STRING` |
| Text Replace | `VLMTextReplace` | `STRING`, `INT`, `STRING` |
| Text Split / Batch | `VLMTextSplit` | `STRING`, `STRING`, `INT` |
| Text Inspector | `VLMTextInspect` | `STRING`, `INT`, `INT`, `INT`, `INT`, `INT`, `STRING`, `STRING` |
| View Text (Streaming) | `ViewText` | `STRING`, `INT`, `INT`, `INT`, `STRING` |
| JSON Extract | `VLMJSONExtract` | `STRING`, `BOOLEAN`, `STRING`, `STRING` |
| JSON to Text | `JsonToText` | `STRING`, `STRING`, `INT` |

### Performance and diagnostics

Run **VLM Runtime Diagnostics** before reporting a bug — it reports your
device, backend, and which optional packages are installed.

| Node | Node ID | Outputs |
| --- | --- | --- |
| VLM Runtime Diagnostics | `VLMRuntimeDiagnostics` | `STRING` |
| VLM Performance Profile | `VLMPerformanceProfile` | `INT`, `FLOAT`, `INT`, `INT`, `BOOLEAN`, `STRING` |
| VLM Image Pixel Budget | `VLMImagePixelBudget` | `IMAGE`, `INT`, `INT`, `STRING` |

### Audio

| Node | Node ID | Outputs |
| --- | --- | --- |
| AudioLDM2 | `AudioLDM2Node` | `*`, `INT`, `AUDIO` |
| Chat Musician | `ChatMusician` | `STRING`, `*`, `INT`, `AUDIO` |
| PlayMusic Node | `PlayMusic` | `*` |
| Save Audio | `SaveAudioNode` | — |

### Legacy model loaders

Kept for existing workflows. New graphs should prefer **Modern VLM**, which
covers most of these architectures through one interface.

| Node | Node ID | Outputs |
| --- | --- | --- |
| Qwen2-VL | `Qwen2VLNode` | `STRING` |
| MiniCPM-V 2.6 (GGUF) | `MiniCPMNode` | `STRING` |
| Molmo Vision-Language Model | `MolmoNode` | `STRING` |
| PaLI-Gemma (Official Segmentation) | `Paligemma` | `STRING`, `MASK`, `IMAGE` |
| Kosmos-2 | `Kosmos2model` | `STRING` |
| MC-LLaVA | `MCLLaVAModel` | `STRING` |
| UForm Gen2 Qwen | `UformGen2QwenNode` | `STRING` |
| MoonDream (Moondream 2) | `MoonDream` | `STRING` |
| [Legacy] Modern VLM Compatibility | `LegacyModernVLM` | `STRING` |

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

### Robotics / VLA isolated runtimes

The robotics nodes keep policy dependencies outside ComfyUI. The universal
HTTP client works without another package. Native openpi WebSocket and
GR00T ZeroMQ clients use the lightweight optional extra:

```bash
python -m pip install \
  -r ComfyUI/custom_nodes/ComfyUI_VLM_nodes/requirements-robotics-client.txt
```

`VLA Model Catalog` covers current SmolVLA, X-VLA, π0/π0-FAST/π0.5,
GR00T N1.7, WALL-OSS, MolmoAct2, VLA-JEPA, LingBot-VA, FastWAM, EO-1,
EVO-1, OpenVLA-OFT, and Octo routes. “Available” means a supported isolated
runtime/checkpoint path; base and architecture-only entries still require
embodiment-specific training and transforms.

Start with SmolVLA for small consumer hardware. The included authenticated
LeRobot sidecar loads one chosen policy, uses its serialized processors,
returns action chunks over bounded JSON/JPEG, keeps it resident for speed,
and can offload it to CPU after an idle timeout. Remote policy URLs require
encrypted transport and explicit opt-in. Tokens are fixed environment
variables (`VLA_POLICY_TOKEN`, `OPENPI_API_KEY`, or `GROOT_API_TOKEN`) and are
never workflow inputs.

See [`examples/robotics/README.md`](examples/robotics/README.md) for D-drive
WSL setup, platform boundaries, current model readiness, observation schemas,
action safety semantics, and the runnable API example.

### Moondream 3 / 3.1 isolated runtime

Moondream's official Photon package pins Pillow below version 11 while
current ComfyUI uses a newer Pillow. It therefore runs in a dedicated sidecar
environment and never changes ComfyUI's Python packages. Read and accept the
[Moondream Model License 1.0](https://moondream.ai/licenses/model/1.0), then
create the environment under the registered `LLavacheckpoints` model folder.

Linux/WSL/macOS:

```bash
runtime="ComfyUI/models/LLavacheckpoints/moondream31-runtime"
uv venv "$runtime/.venv" --python 3.12
uv pip install --python "$runtime/.venv/bin/python" \
  -r ComfyUI/custom_nodes/ComfyUI_VLM_nodes/requirements-moondream31.txt
```

Windows PowerShell:

```powershell
$runtime = "ComfyUI\models\LLavacheckpoints\moondream31-runtime"
uv venv "$runtime\.venv" --python 3.12
uv pip install --python "$runtime\.venv\Scripts\python.exe" `
  -r "ComfyUI\custom_nodes\ComfyUI_VLM_nodes\requirements-moondream31.txt"
```

The first Loader execution downloads the selected official model below that
runtime's `cache` directory. Use `moondream3.1-9B-A2B` for query, caption,
detection, and pointing. Use `moondream3-preview` only for the SVG segment
skill; the final 3.1 model card does not list segment. Set the server-side
`MOONDREAM_PYTHON` environment variable
when using a different isolated environment. Do not put this path or any
credential in a workflow.

Official Photon local inference currently supports NVIDIA Ampere-or-newer on
Linux/Windows and Apple Silicon on macOS 13 or newer. It does not currently
provide local ROCm, Intel GPU, or CPU execution. Those platforms retain every
portable Transformers, GGUF, API, and vision utility node in this pack.

On CUDA 12 x86-64 systems the isolated requirements deliberately install
`nvidia-cuda-runtime-cu12==12.9.79`. Kestrel 0.4.6's AOT kernels require the
`cudaLibraryLoadData` entry point, which is absent from the CUDA 12.6 runtime
bundled by cu126 PyTorch. This pin updates only Photon's private runtime; it
does not replace ComfyUI's PyTorch build or the host NVIDIA driver.

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
  queues. Cache creation is serialized, so concurrent API work cannot make the
  same node allocate duplicate model handles. Turn it on for maximum
  reclamation between prompts.
- Moondream Photon asks ComfyUI to make room before it starts, then owns one
  exact isolated process. `unload_after=true` gracefully shuts it down and
  terminates that process if necessary, which releases Photon model, KV-cache,
  and CUDA-graph allocations without flushing unrelated ComfyUI models. The
  sidecar intentionally does not inherit ComfyUI's PyTorch allocator override;
  Photon's CUDA-graph capture uses the native allocator in its own process. The
  worker does not inherit unrelated provider keys or proxy credentials; only
  `HF_TOKEN`, and `MOONDREAM_API_KEY` for an explicitly selected adapter, may
  cross into its server-side environment. Base-model sidecars honor
  `DO_NOT_TRACK` locally and do not start Kestrel's anonymous telemetry task.
  Its random IPC secret is not placed on the process command line.
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

**Hosted LLM API (Secure)** and **Hosted VLM API (Secure)** share a provider
layer built around the current OpenAI Responses and Chat Completions request
shapes, with Anthropic using its native Messages/vision contract and Gemini
switching to its native multimodal contract for grounded or structured calls.
The VLM node
accepts a still image or a video-frame batch, samples
frames uniformly, resizes and JPEG-compresses them, and enforces per-image and
total request limits before upload. Both nodes can stream text into a connected
`ViewText` node.

Both API nodes also expose:

- **Native web search** for OpenAI, Gemini, Anthropic, xAI, and any compatible
  model routed through OpenRouter. Unsupported presets fail clearly before a
  model request instead of silently pretending to search. Search can add
  provider cost and has provider-specific data terms, so it is off by default.
- **JSON object** and **JSON Schema** output. Completed JSON is always parsed
  locally, JSON Schema results are validated locally, and invalid results fail
  the node instead of flowing into downstream automation.
- **Open-source structured VLM output** through Custom / Local endpoints.
  OpenAI-standard mode supports vLLM, Ollama, and compatible servers;
  `llama.cpp JSON Schema` emits llama.cpp's direct schema dialect; and
  `JSON object + local validation` is a portable fallback for servers that
  implement only JSON mode.

User-provided schemas are capped at 64,000 characters, bounded by depth/node
count, checked against their declared JSON Schema draft, and may use only local
fragment `$ref` values. Remote/file references are rejected so validation can
never turn into an unexpected network or filesystem lookup.

Curated production profiles include:

| Provider | Presets | Server environment variable |
| --- | --- | --- |
| OpenAI | GPT-5.6 Terra, Sol, Luna | `OPENAI_API_KEY` |
| Google | Gemini 3.6 Flash, 3.5 Flash, 3.5 Flash-Lite | `GEMINI_API_KEY` |
| Anthropic | Claude Fable 5, Opus 5, Sonnet 5, Haiku 4.5 | `ANTHROPIC_API_KEY` |
| xAI | Grok 4.5 | `XAI_API_KEY` |
| DeepSeek | V4 Flash, V4 Pro | `DEEPSEEK_API_KEY` |
| Groq | Qwen 3.6 27B Vision, GPT-OSS 20B | `GROQ_API_KEY` |
| Mistral | Mistral Large, Mistral Small, Ministral 14B | `MISTRAL_API_KEY` |
| Together AI | Kimi K2.5, Qwen 3.5 9B | `TOGETHER_API_KEY` |
| OpenRouter | Any compatible model ID | `OPENROUTER_API_KEY` |
| Custom/local | OpenAI-compatible endpoint | `CUSTOM_API_KEY` |

Preset IDs were reviewed on 2026-07-29 against the official
[OpenAI](https://developers.openai.com/api/docs/models),
[Gemini](https://ai.google.dev/gemini-api/docs/models),
[Claude](https://platform.claude.com/docs/en/about-claude/models/overview),
[xAI](https://docs.x.ai/developers/models),
[DeepSeek](https://api-docs.deepseek.com/updates/),
[Groq](https://console.groq.com/docs/models),
[Mistral](https://docs.mistral.ai/models/), and
[Together](https://docs.together.ai/docs/inference/recommended-models), plus
[OpenRouter's multimodal compatibility](https://openrouter.ai/docs/guides/overview/multimodal/overview)
catalogs. Use `model_override` when a provider exposes a newer compatible model
before the next node-pack release.

The capability routing follows the current official
[OpenAI web-search](https://developers.openai.com/api/docs/guides/tools-web-search)
and [structured-output](https://developers.openai.com/api/docs/guides/structured-outputs)
contracts,
[Gemini grounding](https://ai.google.dev/gemini-api/docs/google-search) and
[structured output](https://ai.google.dev/gemini-api/docs/structured-output),
[Claude web-search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
and [structured-output](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
contracts, [xAI web search](https://docs.x.ai/developers/tools/web-search) and
[structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs),
and [OpenRouter server-side search](https://openrouter.ai/docs/guides/features/server-tools/web-search).
The local dialect is based on the
[llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

API keys are not node inputs. A workflow contains only the provider selection,
and the server resolves that provider's fixed environment variable at execution
time. Built-in credentials are pinned to the provider's official HTTPS host;
only the custom profile accepts a URL, and it can read only `CUSTOM_API_KEY`.
Remote custom URLs require HTTPS, while keyless HTTP is restricted to
`localhost`/loopback. Redirect following and environment proxies are disabled
by default, API calls are stateless, OpenAI Responses explicitly use
`store=false`, and provider exceptions are redacted before ComfyUI receives
them.

Web search sends the prompt (and, where supported, the same multimodal request)
to the selected provider's server-side search system. Do not enable it for
content that must not be processed under that provider's search terms.

Opening an older `PromptGenerateAPI` workflow automatically clears its former
plaintext key widget before the graph is configured. Save the migrated workflow
to overwrite the old file, and rotate any key that was previously saved or
shared. See [SECURITY.md](SECURITY.md) for setup and the exact threat model.

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
