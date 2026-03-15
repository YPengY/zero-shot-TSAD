"""Data loading, windowizing, batching, and data-quality diagnostics for synthetic corpora."""

from .collate import ContextWindowCollator
from .dataset import ContextWindowDataset, ShardedSyntheticTsadDataset, SyntheticTsadDataset
from .quality import (
    DataQualityInspector,
    DataQualityThresholds,
    DatasetQualityReport,
    QualityIssue,
    SplitQualityReport,
)
from .transforms import MaskingTargets, RandomPatchMaskingTransform
from .windowizer import SlidingContextWindowizer

__all__ = [
    "ContextWindowCollator",
    "ContextWindowDataset",
    "DataQualityInspector",
    "DataQualityThresholds",
    "DatasetQualityReport",
    "MaskingTargets",
    "QualityIssue",
    "RandomPatchMaskingTransform",
    "ShardedSyntheticTsadDataset",
    "SlidingContextWindowizer",
    "SplitQualityReport",
    "SyntheticTsadDataset",
]
