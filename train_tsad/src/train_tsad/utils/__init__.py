"""Cross-cutting utility exports shared across the Python codebase.

These helpers are intentionally small and side-effect-light: path resolution,
JSON I/O, logging setup, device selection, and RNG seeding. Business logic
should live in higher-level modules, not in this package.
"""

from .device import resolve_torch_device
from .io import iter_jsonl, read_first_jsonl_mapping, read_json_file, read_json_mapping, write_json_file
from .logging import configure_logging, get_logger
from .paths import resolve_path
from .seed import seed_everything

__all__ = [
    "configure_logging",
    "get_logger",
    "iter_jsonl",
    "read_first_jsonl_mapping",
    "read_json_file",
    "read_json_mapping",
    "resolve_path",
    "resolve_torch_device",
    "seed_everything",
    "write_json_file",
]
