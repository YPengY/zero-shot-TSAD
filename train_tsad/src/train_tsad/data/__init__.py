"""Data loading, windowizing, batching, and data-quality diagnostics for synthetic corpora."""

from .collate import ContextWindowCollator
from .dataset import (
    ContextWindowDataset,
    ShardedSyntheticTsadDataset,
    SyntheticTsadDataset,
    WindowShardedTsadDataset,
)
from .quality import (
    DataQualityInspector,
    DataQualityThresholds,
    DatasetQualityReport,
    QualityIssue,
    SplitQualityReport,
)
from .sampler import GroupedWindowSampler
from .transforms import MaskingTargets, RandomPatchMaskingTransform
from .windowizer import SlidingContextWindowizer

__all__ = [
    "ContextWindowCollator",
    "ContextWindowDataset",
    "DataQualityInspector",
    "DataQualityThresholds",
    "DatasetQualityReport",
    "GroupedWindowSampler",
    "MaskingTargets",
    "QualityIssue",
    "RandomPatchMaskingTransform",
    "ShardedSyntheticTsadDataset",
    "SlidingContextWindowizer",
    "SplitQualityReport",
    "SyntheticTsadDataset",
    "WindowShardedTsadDataset",
]
