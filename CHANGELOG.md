# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are published to the [Comfy Registry](https://registry.comfy.org/)
from `pyproject.toml`. A release is only published when `version` changes, so
every user-visible fix needs a version bump.

## [3.4.0] - 2026-07-31

### Added

- A robotics-safe VLA layer with typed embodiment, observation, and action
  contracts; bounded multi-camera history; trajectory inspection and preview;
  action-chunk replanning; and explicit bounds, rate, dimension, horizon, and
  non-finite-value checks before handoff.
- Native policy clients for OpenPI's WebSocket protocol and NVIDIA Isaac
  GR00T's ZeroMQ protocol, plus a portable authenticated HTTP/JPEG protocol for
  isolated policy runtimes.
- An isolated current-LeRobot policy server with pre/postprocessor support,
  serialized inference, optional idle CPU offload, checkpoint feature metadata,
  and environment-only bearer authentication.
- A curated 15-model VLA catalog covering SmolVLA, X-VLA, the OpenPI family,
  GR00T N1.7, WALL-OSS, MolmoAct2, VLA-JEPA, LingBot-VA, FastWAM, EO-1, EVO-1,
  OpenVLA-OFT, and Octo with explicit readiness and fine-tuning requirements.
- A complete API workflow, setup guide, compatibility matrix, security
  guidance, and real-weight SmolVLA validation on an RTX 3090.

### Security

- Workflow JSON never stores robotics API keys. The clients read only
  `VLA_POLICY_TOKEN`, `OPENPI_API_KEY`, or `GROOT_API_TOKEN` from the
  environment, redact them from errors/reports, reject embedded URL
  credentials, and require encrypted transports plus explicit opt-in for
  remote endpoints where the upstream protocol supports encryption.
- The included policy server bounds request, camera, history, and response
  sizes and never uses pickle across the network.

## [3.3.1] - 2026-07-30

### Fixed

- Package metadata declared `license = "MIT"` while the bundled `LICENSE` has
  been Apache-2.0 since the initial commit. Built wheels therefore contained
  contradictory MIT metadata and Apache-2.0 license text. The Registry already
  referenced the license file and was unaffected. Metadata now says
  `Apache-2.0`.
- Moondream 2 and Moondream 3.1 local inference (`b8ae298`).
- SmolVLM setup dependencies (`c13ee23`).

  The two fixes above landed on `main` after 3.3.0 without a version bump, so
  the Registry publish workflow saw 3.3.0 already published and skipped them.
  They reach Registry users for the first time in 3.3.1.

### Added

- Test coverage for the GGUF text and multimodal node families, which
  previously had none: `nodes/suggest.py` (0% to 98%) and
  `nodes/llavaloader.py` (0% to 99%). The new cases pin the behaviours behind
  the pack's longest-running bug reports: widget ordering (#156), sampling
  kwarg plumbing (#144), and handle teardown on both success and failure
  (#137).
- `ruff` lint gate and a coverage floor in CI, plus `requirements-dev.txt`
  for the tooling.
- `CHANGELOG.md`, `CONTRIBUTING.md`, issue and pull request templates, and a
  Dependabot configuration.
- A complete node reference in the README covering all 78 registered nodes.

## [3.3.0] - 2026-07-29

### Added

- Moondream Photon support and universal VLM acceleration utilities, including
  the image pixel-budget and performance-profile nodes (`102f166`).

## [3.2.0] - 2026-07-29

### Added

- Adaptive video intelligence with temporal reasoning, plus the text workflow
  toolkit (join, template, clean, replace, split, JSON extract, inspect)
  (`44fefcb`).

## [3.1.0] - 2026-07-29

### Changed

- Hosted LLM and VLM API nodes modernized and hardened, with provider profiles
  for OpenAI, Google Gemini, Anthropic, xAI, DeepSeek, and others (`505b324`).

## [3.0.0] - 2026-07-29

### Added

- Unified vision stack: open-vocabulary detection (Grounding DINO, OWLv2,
  OmDet), SAM2.1 and SAM3.1 segmentation, tracking, and creator mask tools,
  with structured detection/segmentation schemas (`39fc116`).

### Changed

- **Breaking:** detection and segmentation nodes now emit structured data
  types rather than loose strings. Workflows wiring these outputs into text
  nodes need the new converter utilities.

## [2.3.0] - 2026-07-29

### Added

- Reliable streaming VLM text output (`239c904`).

## [2.2.0] - 2026-07-29

### Changed

- llama.cpp GGUF runtime modernized. `llama-cpp-agent` was removed in favour
  of llama-cpp-python's native JSON Schema support, which resolves the
  unstable wrapper API behind the `unexpected keyword argument 'temperature'`
  crashes (#144).

## [2.1.0] - 2026-07-28

### Added

- Cross-platform runtime support across NVIDIA CUDA, AMD ROCm, Apple Metal,
  Intel XPU, and CPU, without replacing ComfyUI's PyTorch (`4c200c4`).

## [2.0.1] - 2026-07-28

### Added

- Small VLM catalog and real-weight model validation evidence
  (see `MODEL_VALIDATION.md`) (`460b27a`).

## [2.0.0] - 2026-07-28

### Changed

- **Breaking:** node pack modernized with an explicit GPU lifecycle. Models
  now load lazily on first execution and register with ComfyUI's model manager
  so they participate in smart VRAM offloading, which addresses models
  remaining resident after generation (#137) (`b89f628`).
- **Breaking:** `forceInput` string hacks removed from node schemas. They
  corrupted the widget index during serialization and shifted inputs on saved
  workflows (#156). Use the native right-click "Convert to Input" instead.
- Import is now failure-isolated: a broken optional model cannot prevent
  unrelated nodes from loading (#94, #145).
- `numpy` is no longer pinned. The old `numpy<2.0.0` pin crashed startup on
  NumPy 2.x environments (#157).
- Model coverage moved to current releases, including Qwen 3 / 3.5 VL,
  SmolVLM2, InternVL, Granite Vision, and Gemma 3 (#148, #151). The
  unmaintained InternLM-XComposer2 nodes were dropped (#139).

### Removed

- **Breaking:** `llama-cpp-agent` dependency (see 2.2.0).
- **Breaking:** InternLM-XComposer2 nodes, which depended on an AutoGPTQ stack
  that pinned incompatible PyTorch versions (#139).

## 1.0.0 - 1.0.6 (2024-05-20 to 2024-11-03)

Initial packaged releases, predating changelog tracking. This line covered
LLaVA GGUF loaders and samplers, Moondream, Kosmos-2, JoyTag, UForm,
MiniCPM-V, PaLI-Gemma, Florence-2, Molmo, Qwen2-VL, the LLM prompt and
suggestion generators, AudioLDM2, and ChatMusician. See the
[commit history](https://github.com/gokayfem/ComfyUI_VLM_nodes/commits/main)
for detail.

Tagging began at 3.3.0. Earlier versions link to the commit that declared
them, because retroactively tagging them would run current CI against code
that predates it.

[3.4.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/compare/v3.3.1...v3.4.0
[3.3.1]: https://github.com/gokayfem/ComfyUI_VLM_nodes/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/releases/tag/v3.3.0
[3.2.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/commit/44fefcb
[3.1.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/commit/505b324
[3.0.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/commit/39fc116
[2.3.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/commit/239c904
[2.2.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/commit/0da5070
[2.1.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/commit/4c200c4
[2.0.1]: https://github.com/gokayfem/ComfyUI_VLM_nodes/commit/460b27a
[2.0.0]: https://github.com/gokayfem/ComfyUI_VLM_nodes/commit/b89f628
