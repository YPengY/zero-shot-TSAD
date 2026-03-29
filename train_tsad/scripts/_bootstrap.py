from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src_path() -> None:
    """Expose the local `src/` tree to legacy wrapper scripts."""

    script_path = Path(__file__).resolve()
    train_tsad_root = script_path.parents[1]
    src_root = train_tsad_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
