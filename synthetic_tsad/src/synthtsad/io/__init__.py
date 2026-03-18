"""I/O helpers for synthetic dataset generation and packing."""

from .sharded import (
    PackReport,
    SplitPackStats,
    discover_input_splits,
    pack_synthetic_corpus,
    pack_windows_from_packed_corpus,
    write_dataset_meta_for_existing_packed_corpus,
)
from .writer import DatasetWriter, PackedDatasetWriter, PackedWindowDatasetWriter

__all__ = [
    "DatasetWriter",
    "PackedDatasetWriter",
    "PackedWindowDatasetWriter",
    "PackReport",
    "SplitPackStats",
    "discover_input_splits",
    "pack_synthetic_corpus",
    "pack_windows_from_packed_corpus",
    "write_dataset_meta_for_existing_packed_corpus",
]
