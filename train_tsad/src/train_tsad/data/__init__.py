"""Public data-layer exports for training and inspection workflows.

The package is intentionally split into raw-sample loading, window materialization,
batch collation, and quality inspection. This export layer keeps those building
blocks discoverable without forcing callers to know the internal file layout.
"""

from .collate import ContextWindowCollator
from .factory import (
    RawDataset,
    ResolvedDatasetPaths,
    WindowDataset,
    auto_select_available_split,
    build_raw_dataset,
    build_evaluation_loader,
    build_window_loader,
    default_inspection_splits,
    infer_fixed_num_features,
    manifest_is_window_packed,
    resolve_dataset_paths,
)
from .quality import (
    DataQualityInspector,
    DataQualityThresholds,
    DatasetQualityReport,
    QualityIssue,
    SplitQualityReport,
)
from .raw_dataset import SyntheticTsadDataset
from .sampler import GroupedWindowSampler
from .sharded_dataset import ShardedSyntheticTsadDataset, WindowShardedTsadDataset
from .transforms import MaskingTargets, RandomPatchMaskingTransform
from .window_dataset import ContextWindowDataset
from .windowizer import SlidingContextWindowizer

__all__ = [
    "ContextWindowCollator",
    "ContextWindowDataset",
    "DataQualityInspector",
    "DataQualityThresholds",
    "DatasetQualityReport",
    "GroupedWindowSampler",
    "MaskingTargets",
    "RawDataset",
    "QualityIssue",
    "ResolvedDatasetPaths",
    "RandomPatchMaskingTransform",
    "ShardedSyntheticTsadDataset",
    "SlidingContextWindowizer",
    "SplitQualityReport",
    "SyntheticTsadDataset",
    "WindowDataset",
    "WindowShardedTsadDataset",
    "auto_select_available_split",
    "build_evaluation_loader",
    "build_raw_dataset",
    "build_window_loader",
    "default_inspection_splits",
    "infer_fixed_num_features",
    "manifest_is_window_packed",
    "resolve_dataset_paths",
]
