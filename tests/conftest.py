"""Make the source checkout importable on every supported test runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
for candidate in (
    REPOSITORY.parent,
    REPOSITORY.parent / "ComfyUI",
    REPOSITORY.parents[1],
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

# Git worktrees are often intentionally named after a feature branch rather
# than the import package.  Load this checkout explicitly so tests can never
# pass by silently importing a sibling clone with the canonical directory name.
if REPOSITORY.name != "ComfyUI_VLM_nodes":
    specification = importlib.util.spec_from_file_location(
        "ComfyUI_VLM_nodes",
        REPOSITORY / "__init__.py",
        submodule_search_locations=[str(REPOSITORY)],
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load package from {REPOSITORY}.")
    package = importlib.util.module_from_spec(specification)
    sys.modules["ComfyUI_VLM_nodes"] = package
    specification.loader.exec_module(package)
