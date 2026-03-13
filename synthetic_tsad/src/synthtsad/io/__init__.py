"""I/O helpers for synthetic dataset generation and packing."""

from .sharded import PackReport, SplitPackStats, discover_input_splits, pack_synthetic_corpus
from .writer import DatasetWriter

__all__ = [
    "DatasetWriter",
    "PackReport",
    "SplitPackStats",
    "discover_input_splits",
    "pack_synthetic_corpus",
]
