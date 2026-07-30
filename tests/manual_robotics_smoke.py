"""Queue a real robotics policy workflow through ComfyUI's local API.

This is intentionally excluded from pytest: it requires a running ComfyUI
server, a running policy sidecar, a real image in ComfyUI's input directory,
and downloaded policy weights.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid


def _graph(image: str, policy_endpoint: str) -> dict:
    camera_names = [
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    ]
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "12": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "lanczos",
                "width": 256,
                "height": 256,
                "crop": "center",
            },
        },
        "2": {
            "class_type": "VLAEmbodimentProfile",
            "inputs": {
                "preset": "LeRobot SO-100 / SO-101 template",
                "control_hz": 30.0,
                "state_names_json": "",
                "action_names_json": "",
                "action_min_json": "",
                "action_max_json": "",
                "max_delta_json": "",
                "camera_names_json": json.dumps(camera_names),
                "action_mode_override": "",
            },
        },
        "3": {
            "class_type": "VLAObservationBuilder",
            "inputs": {
                "task": (
                    "Move the end effector toward the backpack and prepare to grasp it."
                ),
                "state_json": "[0, 0, 0, 0, 0, 0]",
                "primary_image": ["12", 0],
                "primary_camera": camera_names[0],
                "history_fps": 10.0,
                "timestamp": 0.0,
                "embodiment": ["2", 0],
                "wrist_image": ["12", 0],
                "wrist_camera": camera_names[1],
                "secondary_image": ["12", 0],
                "secondary_camera": camera_names[2],
            },
        },
        "4": {
            "class_type": "VLAHTTPPolicy",
            "inputs": {
                "observation": ["3", 0],
                "endpoint": policy_endpoint,
                "timeout_seconds": 120.0,
                "include_history": True,
                "allow_remote": False,
            },
        },
        "5": {
            "class_type": "VLAActionSafety",
            "inputs": {
                "actions": ["4", 0],
                "embodiment": ["2", 0],
                "mode": "Clamp safely",
                "execution_horizon": 4,
                "previous_action_json": "[0, 0, 0, 0, 0, 0]",
            },
        },
        "6": {
            "class_type": "VLATrajectoryPreview",
            "inputs": {
                "actions": ["5", 0],
                "width": 960,
                "height": 480,
                "embodiment": ["2", 0],
            },
        },
        "7": {
            "class_type": "VLAActionInspect",
            "inputs": {"actions": ["5", 0], "step_index": 0},
        },
        "8": {"class_type": "PreviewImage", "inputs": {"images": ["6", 0]}},
        "9": {"class_type": "ViewText", "inputs": {"text": ["4", 1]}},
        "10": {"class_type": "ViewText", "inputs": {"text": ["5", 1]}},
        "11": {"class_type": "ViewText", "inputs": {"text": ["7", 0]}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8787")
    parser.add_argument(
        "--image",
        required=True,
        help="Filename relative to the running ComfyUI instance's input directory.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    base = args.comfy_url.rstrip("/")
    body = json.dumps(
        {
            "prompt": _graph(args.image, args.policy_url),
            "client_id": str(uuid.uuid4()),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        queued = json.load(response)
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        with urllib.request.urlopen(
            f"{base}/history/{prompt_id}",
            timeout=10,
        ) as response:
            history = json.load(response)
        if prompt_id not in history:
            time.sleep(0.5)
            continue
        entry = history[prompt_id]
        result = {
            "prompt_id": prompt_id,
            "status": entry.get("status"),
            "outputs": entry.get("outputs"),
        }
        print(json.dumps(result, indent=2))
        if entry.get("status", {}).get("status_str") != "success":
            raise SystemExit(1)
        return
    raise SystemExit(f"Timed out waiting for prompt {prompt_id}.")


if __name__ == "__main__":
    main()
