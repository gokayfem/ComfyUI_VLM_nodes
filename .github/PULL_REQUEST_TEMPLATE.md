## What does this change?

<!-- One or two sentences. Link any issue it closes: "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New model support
- [ ] New node
- [ ] Refactor / maintenance
- [ ] Documentation

## Checklist

- [ ] `python -m pytest -q` passes.
- [ ] `python -m ruff check .` passes.
- [ ] Importing the pack still performs no network access, compilation, or
      package install.
- [ ] If a node schema changed, existing widget order is preserved (Comfy
      serializes widget values by position, so reordering breaks saved
      workflows).
- [ ] New optional dependencies fail only the node that needs them, with an
      actionable error.
- [ ] `pyproject.toml` `version` is bumped if this is user-visible, and
      `CHANGELOG.md` has an entry. Releases only publish on a version change.

## Testing

<!--
Which nodes did you run, on which backend (CUDA / ROCm / Metal / XPU / CPU),
and with which model? Real-weight checks are opt-in:
  python tests/manual_model_smoke.py --model "Qwen 3 VL 4B Instruct"
-->
