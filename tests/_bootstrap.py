"""Make this checkout importable from the manual smoke scripts.

The manual scripts run as `python tests/manual_*.py`, outside pytest, so they
do not get `conftest.py`. Without this they only import when the checkout
directory happens to be named `ComfyUI_VLM_nodes`, which is true in a normal
ComfyUI install but not in a git worktree named after a feature branch.

Usage, before importing anything from the package:

    from _bootstrap import bootstrap

    bootstrap()
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE = "ComfyUI_VLM_nodes"
REPOSITORY = Path(__file__).resolve().parents[1]


def bootstrap() -> None:
    """Put the repository and ComfyUI on sys.path, then load this checkout."""

    for candidate in (
        REPOSITORY.parent,
        REPOSITORY.parent / "ComfyUI",
        REPOSITORY.parents[1],
    ):
        if candidate.exists():
            sys.path.insert(0, str(candidate))

    if PACKAGE in sys.modules or REPOSITORY.name == PACKAGE:
        return

    # Load this checkout explicitly so the script can never pass by silently
    # importing a sibling clone with the canonical directory name.
    specification = importlib.util.spec_from_file_location(
        PACKAGE,
        REPOSITORY / "__init__.py",
        submodule_search_locations=[str(REPOSITORY)],
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load package from {REPOSITORY}.")
    package = importlib.util.module_from_spec(specification)
    sys.modules[PACKAGE] = package
    specification.loader.exec_module(package)
