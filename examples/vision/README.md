# Vision API examples

These files contain ComfyUI API prompt graphs: the object that belongs under
the `prompt` key in a `POST /prompt` request. They are not frontend workflow
exports and are not intended for drag-and-drop import into the canvas.

Before queueing:

1. Copy the named image/video into `ComfyUI/input`, or change the `image`/`file`
   widget value to an existing input filename.
2. Restart ComfyUI after installing or updating this node pack.
3. Confirm every `class_type` is present in `/object_info`.
4. Wrap the loaded JSON as `{"prompt": graph}` in the API request.

## Examples

### `grounding_dino_image_api.json`

Runs Grounding DINO Tiny over `grounding_input.png`. Node 2 outputs:

| Index | Output |
| ---: | --- |
| 0 | `VLM_DETECTIONS` |
| 1 | Structured detection JSON |
| 2 | Detection overlay |
| 3 | Box mask |
| 4 | Core nested per-frame `BOUNDING_BOX` |
| 5 | Flat metadata-rich `BOUNDING_BOXES` |

`PreviewImage` displays output 2 and `ViewText` reports output 1.

### `vlm_performance_preflight_api.json`

Loads `vlm_api_people_birds.mp4` with Comfy core video nodes, applies the
`Fast video` performance profile, runs the track-aware adaptive sampler, and
then applies a 14-pixel-aligned image budget. The preview shows the exact batch
that can be connected to any local or hosted VLM. Three `ViewText` nodes report
the selected source indices/timestamps, pixel reduction, and active profile.

### `moondream3_preview_svg_segment_api.json`

Runs the official Moondream 3 Preview SVG segmentation skill over
`moondream_segment_input.png`. Read the linked model license and change
`license_accepted` to `true` before queueing. The graph previews the
black/white mask, isolated foreground cutout, and mask/box/polygon overlay;
`ViewText` receives the exact native SVG path plus its normalized bbox.

Moondream's path coordinates are normalized within the returned bbox. The
node preserves that path verbatim, safely flattens curves/arcs, applies an
even-odd fill for subpath holes, and supersamples the raster edge. The
canonical detection keeps both the primary polygon and the full in-process
mask.

### `moondream31_video_detect_api.json`

Loads `moondream_video_input.mp4`, passes the real frame batch and source FPS
to Moondream, and analyzes every frame with four concurrent requests. Photon
uses the Loader's `max_batch_size=4` scheduler capacity to form dynamic
batches. `ViewText` reports measured throughput and real-time factor. Increase
`frame_stride` to 2, 3, or more when full-frame analysis cannot keep up with
the source FPS; the canonical results preserve original frame indices and
timestamps.

### `sam2_video_tracking_api.json`

Runs this bounded pipeline:

`LoadVideo` → `Video Slice` → `GetVideoComponents` → `ImageScale` →
`ImageFromBatch` → Grounding DINO first-frame detection → SAM2.1 propagation.

The example limits the source to two seconds, scales its largest dimension to
768 pixels while preserving aspect ratio, unloads Grounding DINO after
seeding, and keeps SAM2.1 video state on CPU. The example requests only the
union mask volume; change `mask_output` to `union_and_objects` only when every
per-object mask is required. `VLMTrackReport` is an output node and the final
`PreviewImage` displays SAM2.1 output index 4.

For a longer source, change `start_time` and keep a bounded `duration`.
Independent slices create independent object-ID sessions.

### `sam3_core_adapter_blueprint_api.json`

Uses ComfyUI core nodes to load and run SAM3.1, then passes core
`SAM3_TRACK_DATA` through `VLMSAM3TrackAdapter`. The adapter's output 1 is the
unchanged core payload consumed by `SAM3_TrackPreview`; output 0 is canonical
`VLM_TRACKS` consumed by `VLMTrackReport`.

The graph intentionally names:

`ComfyUI/models/checkpoints/sam3.1_multiplex_fp16.safetensors`

The checkpoint is not bundled. Review the SAM License before downloading
[Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1). ComfyUI rejects
the graph at prompt validation when the named checkpoint is absent. Use the
SAM2.1 example when SAM3.1 access or compatible core support is unavailable.

## Output history

ComfyUI returns image/video previews in the execution history and text reports
in the output-node UI payload. Canonical JSON is also available on the linked
string outputs. Dense masks intentionally stay as tensors rather than being
embedded in the JSON report.

## Creator mask outputs

`VLM Detections to Masks` preserves its original first three outputs and
appends creator-ready derivatives:

| Index | Output |
| ---: | --- |
| 0 | Per-frame combined/union `MASK` |
| 1 | Flattened per-object `MASK` batch |
| 2 | JSON mapping each object mask to its frame/detection/track |
| 3 | Per-frame inverse/background `MASK` |
| 4 | Combined masks as black-and-white `IMAGE` batches |
| 5 | Individual masks as black-and-white `IMAGE` batches |
| 6 | Stable-color per-frame instance maps |

All binary mask values are exactly zero or one. `VLM Mask Processor` can grow,
shrink, and feather any of these masks and returns processed, binary, inverse,
and black-and-white image outputs. `VLM Mask Composite` accepts the resulting
mask plus still-image or video frames and returns a composite, isolated
foreground, background-only plate, and mask image. Connect an optional
background image/video batch to replace the solid background color.
