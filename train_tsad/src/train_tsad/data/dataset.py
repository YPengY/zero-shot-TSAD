"""Compatibility layer for legacy imports from `train_tsad.data.dataset`."""

from .raw_dataset import SyntheticTsadDataset
from .sharded_dataset import ShardedSyntheticTsadDataset, WindowShardedTsadDataset
from .window_dataset import ContextWindowDataset

__all__ = [
    "ContextWindowDataset",
    "ShardedSyntheticTsadDataset",
    "SyntheticTsadDataset",
    "WindowShardedTsadDataset",
]
