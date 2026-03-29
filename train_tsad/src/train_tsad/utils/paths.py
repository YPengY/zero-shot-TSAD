"""Path helpers for config-relative and workspace-relative resolution."""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    """Resolve relative paths against a known base directory.

    Callers pass in the directory that defines path semantics explicitly, which
    avoids depending on the current working directory.
    """

    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
