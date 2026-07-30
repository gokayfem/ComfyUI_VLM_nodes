# Contributing

Thanks for helping out. This pack runs inside other people's ComfyUI installs
on five accelerator backends, so a few rules exist to keep it from breaking
them.

## The rules that matter most

**Importing the pack must never download a model, install a package, compile
anything, or allocate VRAM.** Models load on first execution. This is enforced
by `tests/test_nodes.py`, which asserts the source contains no `pip install`,
no `subprocess.run`, and no direct `torch.cuda.empty_cache`.

**Never reorder or insert widgets in an existing node's `INPUT_TYPES`.** Comfy
serializes widget values by position, so a reordered schema silently rebinds
every saved workflow. Add new inputs to `optional` at the end. The widget order
of the long-lived nodes is pinned by tests; if a test fails because you moved a
widget, the test is right.

**Never use `forceInput`.** It corrupts the widget index during serialization.
Users get the same result from the native right-click "Convert to Input".

**An optional dependency must fail only the node that needs it.** Use
`require_module()` from `nodes/runtime.py`, which raises an actionable error at
execution time rather than at import time.

**Do not install or replace `torch`.** ComfyUI's own installer picks the CUDA,
ROCm, XPU, Metal, or CPU build. The same applies to `numpy` and `Pillow`.

## Setting up

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gokayfem/ComfyUI_VLM_nodes.git
cd ComfyUI_VLM_nodes
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Use ComfyUI's Python. On ComfyUI Portable there is no `activate` script, so
call the interpreter directly:

```
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

## Running checks

```bash
PYTHONPATH=/path/to/custom_nodes:/path/to/ComfyUI python -m pytest -q
python -m ruff check .
```

`PYTHONPATH` needs the directory *containing* this checkout plus ComfyUI
itself, because the tests import `ComfyUI_VLM_nodes` as a package and the nodes
import ComfyUI's `folder_paths`.

CI additionally enforces a coverage floor on Linux/Python 3.13:

```bash
python -m pytest -q --cov=nodes --cov-fail-under=70
```

Real-weight tests are opt-in because they download multi-gigabyte checkpoints,
and are never run in CI:

```bash
python tests/manual_model_smoke.py --model "Qwen 3 VL 4B Instruct"
python tests/manual_specialized_smoke.py --backend florence-large
python tests/manual_llama_cpp_smoke.py --download
```

## Writing tests

Tests must pass without model weights, without a GPU, and without
`llama-cpp-python`. Stub the model boundary instead: see
`tests/test_suggest.py` and `tests/test_llavaloader.py` for the pattern of
faking `LlamaHandle` and `create_chat_completion` to assert what the node sends
to the backend.

`nodes/joytagger/` is vendored upstream code kept byte-compatible with its
source. It is excluded from lint; please don't reformat it.

## Adding a model

1. Prefer adding an entry to the catalog in `nodes/modern_vlm.py` over a new
   node. Most current VLMs work through the shared `transformers` path.
2. If it needs a bespoke loader, follow `nodes/minicpm.py` as the smallest
   complete example.
3. Register the module in the `node_list` in `__init__.py`.
4. Record what you actually ran in `MODEL_VALIDATION.md`. Catalog entries that
   were never executed against real weights must be marked as such.
5. Add the node to the reference table in `README.md`.

## Releasing

The Comfy Registry publishes from `pyproject.toml`, and only when `version`
changes. A fix merged without a version bump never reaches Registry users. So:

- bump `version` in `pyproject.toml`,
- add a `CHANGELOG.md` entry,
- tag the merge commit `vX.Y.Z`.

## Commit messages

Short imperative subject, one logical change per commit. Reference the issue it
closes in the body.
