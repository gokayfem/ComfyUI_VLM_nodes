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

## Robotics VLA pass

Validated on 2026-07-31 through the included isolated LeRobot HTTP policy
server, entirely from WSL and D-drive storage:

- Runtime: Python 3.12.12, LeRobot 0.6.1 from current upstream source,
  PyTorch 2.11.0+cu128, and an NVIDIA RTX 3090.
- Checkpoint: `lerobot/smolvla_base` (about 2.5 GiB of D-drive cache), backed
  by `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
- Real input: local `image (23).png`, a 256x256 outdoor photograph, repeated
  across the checkpoint's three declared camera keys with a six-value state
  vector and the task “Move the end effector toward the backpack and prepare
  to grasp it.”
- Contract: three camera tensors, `observation.state`, the LeRobot
  preprocessor, `predict_action_chunk`, the checkpoint postprocessor, bounded
  JSON/JPEG transport, action parsing, and the ComfyUI safety layer all ran.
  The native checkpoint advertises a 50-step chunk; the server returned four
  steps of six actions for this test.
- Five warm requests after one discarded warm-up measured 241.374 ms mean
  server inference (242.957 ms median, 234.726–249.980 ms range) and
  270.404 ms mean HTTP client time (271.483 ms median,
  261.477–282.218 ms range).
- The final raw action chunk was:

  ```json
  [
    [0.06258623, -0.11250310, -0.13713294, -0.06168950, -0.00926633, -0.08506130],
    [0.15420279, -0.05678255, -0.20159233, 0.06734322, -0.00563951, -0.09575561],
    [0.16482556, -0.07453565, -0.17410603, 0.02461835, -0.00256573, 0.15842065],
    [0.27048433, -0.09272483, -0.19934477, 0.05491992, 0.05286619, 0.07594281]
  ]
  ```

  Applying the SO-100/SO-101 template limits from an all-zero previous action
  found five per-step delta violations, no bounds violations, and no
  non-finite values. `Clamp safely` produced:

  ```json
  [
    [0.06258623, -0.1, -0.1, -0.06168950, -0.00926633, -0.08506130],
    [0.15420279, -0.05678255, -0.2, 0.03831051, -0.00563951, -0.09575561],
    [0.16482556, -0.07453565, -0.17410603, 0.02461835, -0.00256573, 0.05424440],
    [0.26482555, -0.09272483, -0.19934477, 0.05491992, 0.05286619, 0.07594281]
  ]
  ```

This is an end-to-end loading, preprocessing, inference, transport, parsing,
and safety-contract pass. It is not evidence that a base SmolVLA checkpoint can
control an SO-100 from an arbitrary Internet-style photograph. Actual robot
deployment still requires embodiment-matched fine-tuning, calibrated state and
camera inputs, hardware-certified limits, a deadman/watchdog, collision
handling, and an external emergency stop.

The same real checkpoint was then exercised through ComfyUI's actual local
`POST /prompt` API, not by calling the Python node directly. The graph loaded
and center-cropped the real image to 256x256, constructed the three-camera
checkpoint contract, called the isolated GPU policy, applied the SO-100/SO-101
template safety gate, rendered a 960x480 trajectory preview, and emitted all
three text reports. Final prompt
`86b31f5a-5a8c-4abc-abdb-634e46da5c93` completed successfully: the policy
returned `[4, 6]` actions, the safety gate found three rate violations and
clamped them, `safe_for_handoff` was true under the declared template, and
ComfyUI wrote preview `ComfyUI_temp_icynu_00001_.png`. The reusable acceptance
harness is `tests/manual_robotics_smoke.py`.

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

## Structured vision validation

The versioned detection/track/point/event payloads, geometry and mask
conversion, strict spatial parser, Grounding-family adapters, SAM2.1 session
plumbing, SAM3 bit-packed payload adapter, and ByteTrack-style association pass
the local WSL contract suite. Those tests validate schemas, shapes, output
ordering, bounds, timestamps, deterministic IDs, and error handling.

Representative real-weight checks were then submitted through ComfyUI's local
`POST /prompt` API and verified from `/history/{prompt_id}`. The test machine
used ComfyUI 0.28.0, Python 3.12.12, PyTorch 2.13.0+cu126, Transformers 5.14.1,
and an NVIDIA RTX 3090. Input media, checkpoints, model caches, ComfyUI, and
this checkout all remained on the D drive under WSL.

| Family | Representative checkpoint policy | Real-weight status |
| --- | --- | --- |
| Grounding DINO | Tiny; Base uses the same loader/processor contract | **Passed**: FP16, four real 640x360 video frames in two-frame micro-batches; person and bird boxes/labels were visually checked, serialized, timestamped, and in bounds |
| OWLv2 | Base Ensemble | Pending |
| OmDet Turbo | Swin Tiny | Pending |
| SAM2.1 video | Hiera Tiny; sibling sizes use the same session adapter | **Passed**: FP16, real 12-frame 640x360 clip at 24 FPS with CPU preprocessing/state. Grounding's core `BOUNDING_BOX` output connected directly: the forward union-only run kept one person ID on frames 0-11; a last-frame reverse run kept two IDs for 24 observations and emitted 24 frame-major object masks. All geometry was in bounds and first/last overlays and masks were visually checked |
| Comfy core SAM3.1 | `sam3.1_multiplex_fp16.safetensors`, only after license/access is available | Pending |
| SAM3 adapter/report | Synthetic core payload contract | Passed without weights; real core handoff pending |
| ByteTrack-style tracker | Deterministic synthetic crossing, missed-frame, and expiry cases | Passed; no model weights exist |
| Florence-2 multitask | Base FT; Large uses the same native Transformers contract | **Passed**: real object-detection API run produced bounded woman, face, and clothing boxes plus a visually checked overlay |

The SAM2 API check initially exposed a real session-lifecycle defect that unit
fixtures did not: prompt insertion must be followed by inference on the seeded
frame before propagation. The implementation now performs that seed pass and
also propagates in reverse when `seed_frame` is greater than zero. Later live
checks exercised nested multi-object core boxes, CPU preprocessing/state,
union-only low-memory output, optional object-mask output, disabled preview
rendering, reverse propagation, and `unload_after=true` for both models. The
final unload run returned total reported GPU memory use to within 4 MiB of the
pre-run `nvidia-smi` baseline.

Grounding DINO and SAM2 sibling sizes are catalog-available but were not
downloaded or executed. OWLv2, OmDet Turbo, and gated SAM3 remain explicitly
unverified; the UI never presents them as locally tested simply because their
schemas import.

The acceptance run for each model family must record:

1. Exact checkpoint revision, ComfyUI/Python/PyTorch/Transformers versions,
   device, dtype, peak accelerator allocation, and wall time.
2. A real image or short bounded video with manually verified boxes, labels,
   masks, timestamps, and stable IDs.
3. The canonical JSON schema/version and every advertised output socket,
   including preview/report output through ComfyUI's local `/prompt` API.
4. A second queue using the cached model, followed by an `unload_after=true`
   run where that option exists.
5. Failure behavior for an absent checkpoint or gated access without exposing
   a token.

One checkpoint per distinct implementation family is enough for sibling model
sizes that share the same code path. Validation prioritizes the smallest useful
checkpoint and will not download or execute a 30B model. A larger variant is
tested only when it has a different loader, processor, postprocessor, or
quantization path.

## Not marked passed

- Qwen 3 VL 30B-A3B: weights are available locally, but inference validation
  was stopped at the user's request and will not be repeated.
- Moondream2 2025-06-21: its pinned remote wrapper needed Transformers 5 loading
  metadata, but this Torch/CUDA stack produced NaN probabilities when sampling
  and immediate EOS with greedy decoding. The node defaults to the
  non-destructive greedy path and raises an actionable error on an empty result.
- PaLI-Gemma and Gemma 3: gated checkpoints were not accessible without an
  accepted license and token.
