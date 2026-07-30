# Robotics and VLA workflows

The robotics nodes make ComfyUI a policy-development, inspection, and
simulation surface. They do **not** send commands to motors, ROS, CAN, serial,
or a robot SDK.

The boundary is intentional:

```text
camera/state/task
      |
      v
VLA Observation Builder
      |
      v
isolated policy server  --->  raw action chunk
                                  |
                                  v
                         VLA Action Safety Gate
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
            inspect / plot / record      simulator or your own
                                         supervised controller bridge
```

A real controller bridge must independently enforce a deadman, watchdog,
emergency stop, collision/workspace limits, timestamps, command freshness, and
the manufacturer's limits. A `safe_for_handoff=true` workflow result only
means that the declared ComfyUI profile checks passed.

## Why policy runtimes are isolated

LeRobot, openpi, Isaac-GR00T, OpenVLA/OFT, and Octo use different PyTorch/JAX,
CUDA, Transformers, compiler, and operating-system combinations. Installing
all of those into ComfyUI would replace or constrain the working accelerator
stack and make Windows, macOS, ROCm, and XPU support worse.

The ComfyUI package therefore contains only:

- typed state/action/camera contracts;
- bounded image serialization;
- a dependency-light universal HTTPS/loopback HTTP client;
- exact clients for the official openpi MessagePack WebSocket and GR00T
  MessagePack/ZeroMQ protocols;
- action validation, horizon control, inspection, and plotting.

The heavyweight policy stays in its own process, container, WSL distribution,
Linux machine, Mac, or GPU server. This also allows ComfyUI to use AMD ROCm,
Apple Metal, Intel XPU, or CPU while a policy runs on an NVIDIA Linux server.

## Fast path: SmolVLA through the universal sidecar

Use a separate LeRobot environment. Current LeRobot documentation recommends
Python 3.12 and exposes policy-specific extras. On this computer, keep it on
the D drive:

```bash
# WSL
python3.12 -m venv /mnt/d/vla-runtime/lerobot-smolvla
source /mnt/d/vla-runtime/lerobot-smolvla/bin/activate
python -m pip install --upgrade pip
python -m pip install "lerobot[smolvla]"

export VLA_POLICY_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python /mnt/d/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI_VLM_nodes/examples/robotics/lerobot_policy_server.py \
  --policy-type smolvla \
  --policy-path YOUR_FINE_TUNED_SMOLVLA_CHECKPOINT \
  --device auto \
  --actions-per-chunk 16 \
  --idle-offload-seconds 300
```

Set the same `VLA_POLICY_TOKEN` in the environment that launches ComfyUI.
Never put it in a workflow. In `VLA Policy — Universal HTTP`, use
`http://127.0.0.1:8787`.

`lerobot/smolvla_base` is a base model. It is a useful fine-tuning starting
point, not a universal zero-shot controller. Use an embodiment-specific
checkpoint whose feature names, action dimensions, state dimensions,
normalization statistics, and camera keys match the workflow.

The sidecar:

- loads only the chosen policy and its serialized pre/post-processors;
- uses `predict_action_chunk` when provided and falls back to `select_action`;
- keeps the model resident by default for low latency;
- can move it to CPU after an idle interval and move it back on demand;
- accepts one request at a time per policy, preventing stateful policy races;
- uses bounded JSON/JPEG rather than pickle;
- never returns tracebacks, environment variables, request data, or
  authorization headers.

For a real local API acceptance run, start ComfyUI and the policy sidecar, put
an image in ComfyUI's `input` directory, then run:

```bash
python tests/manual_robotics_smoke.py \
  --comfy-url http://127.0.0.1:8188 \
  --policy-url http://127.0.0.1:8787 \
  --image robot_front.png
```

The script queues the graph through `POST /prompt`, waits on its history entry,
and prints the policy report, safety report, first action, and preview filename.

Install the relevant official LeRobot extra for another policy. Examples are
`lerobot[pi]` for π0/π0.5/π0-FAST and `lerobot[smolvla]` for SmolVLA. Some
newer policy integrations may require installing current LeRobot from source
with their documented extra.

## Native openpi server

Install the small ComfyUI client dependencies:

```bash
python -m pip install -r requirements-robotics-client.txt
```

Run the official openpi policy WebSocket server in its own supported
environment. The upstream runtime is currently tested on Ubuntu 22.04 and an
NVIDIA GPU with more than 8 GB for inference; use WSL/Docker or a remote Linux
server rather than forcing it into a macOS/Windows ComfyUI environment.

Use:

- `Flat keys (DROID / LIBERO)` for observations such as
  `observation/image`, `observation/wrist_image`, and `observation/state`;
- `Nested images (ALOHA)` for `state`, an `images` mapping such as
  `cam_high`/wrist cameras, and `prompt`.

The workflow supplies key names, but the policy's own transform still defines
the exact shapes and normalization. `OPENPI_API_KEY` is read only from the
ComfyUI server environment. Remote endpoints require WSS and explicit
`allow_remote=true`.

## Native Isaac-GR00T N1.7 server

Install the same lightweight robotics client requirements in ComfyUI. Run the
official GR00T `PolicyServer` beside an embodiment-compatible `Gr00tPolicy`.
The node sends the documented nested contract:

```text
video.<camera>     uint8  [batch=1, history, height, width, RGB=3]
state.state        float32[batch=1, history, state_dim]
language.task      string [batch=1, 1]
```

The official server returns one or more physical-unit action streams with
shape `[batch, horizon, dimension]`. The node flattens those streams while
preserving their named slices. `GROOT_API_TOKEN` remains in the ComfyUI
environment.

GR00T N1.7 currently targets NVIDIA CUDA/Jetson Linux and needs an
embodiment-compatible base or post-trained checkpoint. A ComfyUI client on
Windows, macOS, ROCm, or another machine may call that server over a trusted
network, but remote access must be explicitly enabled. Native GR00T ZMQ does
not encrypt traffic; use a private authenticated network/tunnel. Prefer the
universal HTTPS bridge when transport-layer encryption is required.

## Model catalog: what “available” means

`VLA Model Catalog` distinguishes these states:

| Family | Example checkpoint | Route | Important qualification |
| --- | --- | --- | --- |
| SmolVLA | `lerobot/smolvla_base` | LeRobot HTTP sidecar | 450M and the best small starting point; fine-tune for the robot |
| X-VLA | `lerobot/xvla-base` | LeRobot HTTP sidecar | 0.9B cross-embodiment base; use a matching domain checkpoint |
| π0 | `lerobot/pi0_base` | LeRobot or openpi | Base/fine-tuning model, not a universal drop-in controller |
| π0-FAST | `lerobot/pi0fast-base` | LeRobot or openpi | Faster tokenized action generation |
| π0.5 | `lerobot/pi05_base` | LeRobot or openpi | Open-world generalization; still embodiment-specific |
| GR00T N1.7 | `nvidia/GR00T-N1.7-3B` | GR00T ZMQ or LeRobot | Base has specific zero-shot tags; other robots need post-training |
| X-Square WALL-OSS | `x-square-robot/wall-oss-flow` | LeRobot HTTP sidecar | MoE research model; validate checkpoint terms and embodiment |
| MolmoAct2 | `lerobot/MolmoAct2-SO100_101-LeRobot` | LeRobot HTTP sidecar | Converted SO-100/SO-101 checkpoint |
| VLA-JEPA | `lerobot/VLA-JEPA-Pretrain` | LeRobot HTTP sidecar | DROID pretrain plus LIBERO/SimplerEnv checkpoints |
| LingBot-VA | `lerobot/lingbot_va_base` | LeRobot HTTP sidecar | Prefer its LIBERO-Long/RoboTwin post-train where applicable |
| FastWAM | released LIBERO checkpoint | LeRobot HTTP sidecar | Heavy world-action research runtime |
| EO-1 / EVO-1 | your trained checkpoint | LeRobot HTTP sidecar | Architecture support, not a universal ready-made controller |
| OpenVLA-OFT | compatible OFT fine-tune | dedicated sidecar | OFT is the faster multi-image/high-frequency OpenVLA route |
| Octo small | Octo small 27M | dedicated JAX sidecar | Lightweight legacy research baseline |

The catalog is a verified runtime/checkpoint map, not a promise that a base
checkpoint understands an arbitrary robot. Exact data transforms and
fine-tuning are part of the policy.

## Observation history and real-time use

Connect an `IMAGE` batch to a camera input to represent temporal history. All
camera batches must have the same length, although a one-frame camera may
broadcast. Use:

`Video Slice` → `VLM Adaptive Frame Sampler` or a live capture source →
`VLM Image Pixel Budget` → `VLA Observation Builder`

For closed-loop robotics, do not run an unbounded ComfyUI queue for each motor
tick. Use ComfyUI to prototype and inspect the observation/policy/safety
contract, and use the policy runtime's asynchronous control support for the
actual high-frequency loop. LeRobot supports asynchronous action chunks and
GR00T supports TensorRT deployment; both are better places for timing-critical
execution.

## Action safety semantics

The `VLA Action Safety Gate` checks:

- policy action dimension against the embodiment;
- NaN and infinity;
- minimum and maximum values;
- maximum change per action dimension and control step;
- the requested execution horizon.

Modes:

- `Block unsafe`: raise and stop the workflow on any violation.
- `Clamp safely`: replace non-finite values conservatively, then clamp bounds
  and sequential per-step deltas.
- `Hold position on unsafe`: replace the whole chunk with the explicitly
  supplied previous/current command.
- `Report only`: preserve the raw trajectory and set
  `safe_for_handoff=false`.

For delta-action policies, `previous_action_json` means the previous delta
command, not an absolute joint pose. Define the profile in the same units and
semantics as the policy output.

`VLA Actions From JSON` imports recorded/simulator trajectories without a
network policy. `VLA Action Chunk Replan` blends the unexecuted edge of an old
chunk into a new chunk to reduce discontinuities, then the result should pass
through the safety gate again. This deterministic blend is useful for workflow
experiments but does not replace LeRobot's asynchronous controller or a
policy-specific real-time chunking implementation.

## API example

`vla_http_policy_safety_api.json` is a ComfyUI API prompt graph. Put
`robot_front.png` in `ComfyUI/input`, start a compatible sidecar, then POST:

```json
{"prompt": {"...": "contents of vla_http_policy_safety_api.json"}}
```

It builds an observation, calls the policy, clamps it against the explicit
profile, renders the trajectory, and outputs both inference and safety JSON.

## Security checklist

- Keep all policy tokens in environment variables.
- Leave `allow_remote=false` for local servers.
- Remote universal endpoints must use HTTPS; remote openpi endpoints must use
  WSS.
- Never expose GR00T ZMQ directly to an untrusted network.
- Pin checkpoint revisions when reproducibility matters.
- Treat camera images, task language, and robot state as sensitive data.
- Do not connect action JSON directly to hardware without a separate
  supervised controller bridge and independent safety system.

Authoritative upstream documentation:

- [LeRobot installation](https://huggingface.co/docs/lerobot/main/en/installation)
- [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot asynchronous inference](https://huggingface.co/docs/lerobot/async)
- [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi)
- [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [OpenVLA and OFT](https://github.com/openvla/openvla)
- [Octo](https://github.com/octo-models/octo)
