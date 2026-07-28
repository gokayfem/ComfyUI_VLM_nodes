"""Make the source checkout importable on every supported test runner."""

from __future__ import annotations

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
