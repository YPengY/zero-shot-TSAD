"""Factory helpers for constructing datasets and loaders from config.

This module is the assembly boundary for the data stack. It resolves paths,
chooses the appropriate dataset implementation, builds window views when
needed, and configures the collator/sampler pair expected by the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from ..config import DataConfig, LossConfig
from ..interfaces import Batch
from ..utils import read_first_jsonl_mapping, resolve_path
from .collate import ContextWindowCollator
from .raw_dataset import SyntheticTsadDataset
from .sampler import GroupedWindowSampler
from .sharded_dataset import ShardedSyntheticTsadDataset, WindowShardedTsadDataset
from .transforms import RandomPatchMaskingTransform
from .window_dataset import ContextWindowDataset
from .windowizer import SlidingContextWindowizer

RawDataset = SyntheticTsadDataset | ShardedSyntheticTsadDataset | WindowShardedTsadDataset
WindowDataset = ContextWindowDataset | WindowShardedTsadDataset


@dataclass(frozen=True, slots=True)
class ResolvedDatasetPaths:
    """Filesystem paths needed to build one split dataset."""

    dataset_root: Path
    manifest_path: Path | None


def manifest_is_window_packed(manifest_path: Path | None) -> bool:
    """Return whether a manifest describes pre-windowed training shards."""

    if manifest_path is None or not manifest_path.exists():
        return False

    first_row = read_first_jsonl_mapping(manifest_path)
    if first_row is None:
        return False

    return (
        "shard_npz_path" in first_row
        and ("window_index" in first_row or "sample_index" in first_row)
        and "context_start" in first_row
        and "context_end" in first_row
        and "valid_length" in first_row
    )


def resolve_dataset_paths(
    data_config: DataConfig,
    *,
    split: str,
    base_dir: Path,
) -> ResolvedDatasetPaths:
    """Resolve the dataset root and optional manifest for one split."""

    dataset_root = resolve_path(data_config.dataset_root, base_dir=base_dir)
    manifest_path = resolve_path(data_config.manifest_path(split), base_dir=base_dir)
    return ResolvedDatasetPaths(
        dataset_root=dataset_root,
        manifest_path=manifest_path if manifest_path.exists() else None,
    )


def build_raw_dataset(
    data_config: DataConfig,
    *,
    split: str,
    base_dir: Path,
) -> RawDataset:
    """Build the raw dataset object for one configured split."""

    dataset_paths = resolve_dataset_paths(data_config, split=split, base_dir=base_dir)

    if data_config.use_sharded_dataset:
        if manifest_is_window_packed(dataset_paths.manifest_path):
            return WindowShardedTsadDataset(
                root_dir=dataset_paths.dataset_root,
                split=split,
                manifest_path=dataset_paths.manifest_path,
                max_cached_shards=data_config.max_cached_shards,
            )
        return ShardedSyntheticTsadDataset(
            root_dir=dataset_paths.dataset_root,
            split=split,
            manifest_path=dataset_paths.manifest_path,
            max_cached_shards=data_config.max_cached_shards,
            load_normal_series=data_config.load_normal_series,
            load_metadata=data_config.load_metadata,
        )

    return SyntheticTsadDataset(
        root_dir=dataset_paths.dataset_root,
        split=split,
        manifest_path=dataset_paths.manifest_path,
        load_normal_series=data_config.load_normal_series,
        load_metadata=data_config.load_metadata,
    )


def build_window_loader(
    raw_dataset: RawDataset,
    *,
    data_config: DataConfig,
    loss_config: LossConfig,
    train_seed: int,
    split: str,
    shuffle: bool,
    batch_size: int,
) -> tuple[WindowDataset, DataLoader[Batch]]:
    """Build the model-facing dataset/loader pair for one split.

    Raw-sequence datasets are wrapped with a window view on demand, while
    pre-windowed shards can flow through directly. The loader returned here is
    therefore the first point where the training loop can treat both storage
    layouts identically.
    """

    is_prewindowed = bool(getattr(raw_dataset, "is_prewindowed", False))
    if is_prewindowed:
        window_dataset: WindowDataset = raw_dataset
    else:
        windowizer = SlidingContextWindowizer(
            context_size=data_config.context_size,
            patch_size=data_config.patch_size,
            stride=data_config.stride,
            pad_short_sequences=data_config.pad_short_sequences,
            include_tail=data_config.include_tail,
        )
        window_dataset = ContextWindowDataset(
            raw_dataset,
            windowizer,
            enable_direct_window_read=data_config.enable_direct_window_read,
        )

    masking_transform = None
    if loss_config.reconstruction_loss_weight > 0.0 and data_config.enable_patch_masking:
        split_seed_offset = 0 if split == data_config.split else 1
        masking_transform = RandomPatchMaskingTransform(
            patch_size=data_config.patch_size,
            mask_ratio=data_config.mask_ratio,
            seed=train_seed + split_seed_offset,
        )

    sampler = None
    effective_shuffle = False
    if shuffle:
        if data_config.shuffle_strategy == "global_window":
            effective_shuffle = True
        else:
            split_seed_offset = 0 if split == data_config.split else 1
            sampler = GroupedWindowSampler(
                window_dataset.grouped_blocks(data_config.shuffle_strategy),
                seed=train_seed + split_seed_offset,
            )

    loader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last if shuffle else False,
        collate_fn=ContextWindowCollator(
            include_reconstruction_targets=loss_config.reconstruction_loss_weight > 0.0,
            masking_transform=masking_transform,
            patch_size=data_config.patch_size,
            normalization_mode=data_config.normalization_mode,
            normalization_eps=data_config.normalization_eps,
        ),
    )
    return window_dataset, loader


def build_evaluation_loader(
    raw_dataset: RawDataset,
    *,
    data_config: DataConfig,
    batch_size: int,
) -> tuple[WindowDataset, DataLoader[Batch]]:
    """Build a deterministic evaluation dataloader."""

    is_prewindowed = bool(getattr(raw_dataset, "is_prewindowed", False))
    if is_prewindowed:
        window_dataset: WindowDataset = raw_dataset
    else:
        window_dataset = ContextWindowDataset(
            raw_dataset,
            SlidingContextWindowizer(
                context_size=data_config.context_size,
                patch_size=data_config.patch_size,
                stride=data_config.stride,
                pad_short_sequences=data_config.pad_short_sequences,
                include_tail=data_config.include_tail,
            ),
            enable_direct_window_read=data_config.enable_direct_window_read,
        )

    loader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=False,
        collate_fn=ContextWindowCollator(
            include_reconstruction_targets=False,
            patch_size=data_config.patch_size,
            normalization_mode=data_config.normalization_mode,
            normalization_eps=data_config.normalization_eps,
        ),
    )
    return window_dataset, loader


def infer_fixed_num_features(*datasets: RawDataset | None) -> int:
    """Infer one shared feature-channel count across all provided datasets."""

    feature_counts: set[int] = set()
    sample_counts_by_feature: dict[int, int] = {}
    example_ids_by_feature: dict[int, list[str]] = {}

    for dataset in datasets:
        if dataset is None:
            continue

        sample_num_features = getattr(dataset, "sample_num_features", None)
        sample_id_getter = getattr(dataset, "sample_id", None)
        for index in range(len(dataset)):
            if callable(sample_num_features):
                feature_count = int(sample_num_features(index))
                sample_id = (
                    str(sample_id_getter(index))
                    if callable(sample_id_getter)
                    else f"sample_{index:06d}"
                )
            else:
                sample = dataset[index]
                feature_count = int(sample.series.shape[1])
                sample_id = str(sample.sample_id)

            feature_counts.add(feature_count)
            sample_counts_by_feature[feature_count] = (
                sample_counts_by_feature.get(feature_count, 0) + 1
            )
            examples = example_ids_by_feature.setdefault(feature_count, [])
            if len(examples) < 3:
                examples.append(sample_id)

    if not feature_counts:
        raise FileNotFoundError("Could not infer the number of feature channels from the datasets.")
    if len(feature_counts) != 1:
        feature_summary = ", ".join(
            f"{feature_count}D(count={sample_counts_by_feature[feature_count]}, "
            f"examples={example_ids_by_feature[feature_count]})"
            for feature_count in sorted(feature_counts)
        )
        raise ValueError(
            "The current pipeline requires a fixed number of feature channels per run. "
            f"Found multiple values: {sorted(feature_counts)}. "
            f"Observed distribution: {feature_summary}."
        )
    return next(iter(feature_counts))


def default_inspection_splits(data_config: DataConfig) -> list[str]:
    """Return the configured split order without duplicates."""

    ordered = [data_config.split, data_config.validation_split, data_config.test_split]
    unique: list[str] = []
    for split in ordered:
        if split not in unique:
            unique.append(split)
    return unique


def auto_select_available_split(
    data_config: DataConfig,
    *,
    base_dir: Path,
) -> str:
    """Pick the first available evaluation split by preference order."""

    for split in (data_config.test_split, data_config.validation_split, data_config.split):
        try:
            build_raw_dataset(data_config, split=split, base_dir=base_dir)
            return split
        except FileNotFoundError:
            continue

    raise FileNotFoundError("Could not find any available split for evaluation.")


__all__ = [
    "RawDataset",
    "ResolvedDatasetPaths",
    "WindowDataset",
    "auto_select_available_split",
    "build_raw_dataset",
    "build_evaluation_loader",
    "build_window_loader",
    "default_inspection_splits",
    "infer_fixed_num_features",
    "manifest_is_window_packed",
    "resolve_dataset_paths",
]
