from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_PATH = Path(__file__).resolve()
TRAIN_TSAD_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SRC_ROOT = TRAIN_TSAD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from train_tsad.config import ExperimentConfig  # noqa: E402
from train_tsad.data import (  # noqa: E402
    ContextWindowCollator,
    ContextWindowDataset,
    DataQualityInspector,
    GroupedWindowSampler,
    RandomPatchMaskingTransform,
    ShardedSyntheticTsadDataset,
    SlidingContextWindowizer,
    SyntheticTsadDataset,
    WindowShardedTsadDataset,
)
from train_tsad.losses import TimeRCDMultiTaskLoss  # noqa: E402
from train_tsad.evaluation import PatchFeatureEvaluator, TimeRCDEvaluator  # noqa: E402
from train_tsad.models import TimeRCDModel  # noqa: E402
from train_tsad.training import (  # noqa: E402
    CheckpointManager,
    Trainer,
    build_optimizer,
    build_scheduler,
)


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    """Resolve relative paths against the project root.

    Args:
        path: User-provided path from config or CLI.
        base_dir: Base directory used when `path` is relative.

    Returns:
        Absolute resolved path that can be used by loaders/checkpoint writers.
    """

    return path if path.is_absolute() else (base_dir / path).resolve()


def _set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manifest_is_window_packed(manifest_path: Path | None) -> bool:
    """Heuristic detection for window-packed manifests."""

    if manifest_path is None or not manifest_path.exists():
        return False
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row_text = line.strip()
        if not row_text:
            continue
        row = json.loads(row_text)
        return (
            isinstance(row, dict)
            and "shard_npz_path" in row
            and ("window_index" in row or "sample_index" in row)
            and "context_start" in row
            and "context_end" in row
            and "valid_length" in row
        )
    return False


def _build_raw_dataset(config: ExperimentConfig, *, split: str):
    """Create the raw dataset object for one split.

    Workflow:
    1. Resolve dataset root and split manifest to absolute paths.
    2. Pick sharded/non-sharded dataset class from config.
    3. Build the dataset instance without materializing samples yet.
    """

    dataset_root = _resolve_path(config.data.dataset_root, base_dir=PROJECT_ROOT)
    manifest_path = config.data.manifest_path(split)
    manifest_path = _resolve_path(manifest_path, base_dir=PROJECT_ROOT)
    manifest_value: Path | None = manifest_path if manifest_path.exists() else None

    if config.data.use_sharded_dataset:
        if _manifest_is_window_packed(manifest_value):
            return WindowShardedTsadDataset(
                root_dir=dataset_root,
                split=split,
                manifest_path=manifest_value,
                max_cached_shards=config.data.max_cached_shards,
            )
        return ShardedSyntheticTsadDataset(
            root_dir=dataset_root,
            split=split,
            manifest_path=manifest_value,
            max_cached_shards=config.data.max_cached_shards,
            load_normal_series=config.data.load_normal_series,
            load_metadata=config.data.load_metadata,
        )

    return SyntheticTsadDataset(
        root_dir=dataset_root,
        split=split,
        manifest_path=manifest_value,
        load_normal_series=config.data.load_normal_series,
        load_metadata=config.data.load_metadata,
    )


def _build_window_loader(
    raw_dataset,
    *,
    config: ExperimentConfig,
    split: str,
    shuffle: bool,
    batch_size: int,
) -> tuple[ContextWindowDataset, DataLoader]:
    """Build window dataset + dataloader for train/validation.

    The function wires together:
    - Configurable context windowization (`SlidingContextWindowizer`)
    - Optional patch masking (only when reconstruction loss is enabled)
    - Batch collation with normalization/padding masks into `Batch` tensors
    """

    is_prewindowed = bool(getattr(raw_dataset, "is_prewindowed", False))
    if is_prewindowed:
        window_dataset = raw_dataset
    else:
        windowizer = SlidingContextWindowizer(
            context_size=config.data.context_size,
            patch_size=config.data.patch_size,
            stride=config.data.stride,
            pad_short_sequences=config.data.pad_short_sequences,
            include_tail=config.data.include_tail,
        )
        window_dataset = ContextWindowDataset(
            raw_dataset,
            windowizer,
            enable_direct_window_read=config.data.enable_direct_window_read,
        )
    masking_transform = None
    if (
        config.loss.reconstruction_loss_weight > 0.0
        and config.data.enable_patch_masking
    ):
        seed_offset = 0 if split == config.data.split else 1
        masking_transform = RandomPatchMaskingTransform(
            patch_size=config.data.patch_size,
            mask_ratio=config.data.mask_ratio,
            seed=config.train.seed + seed_offset,
        )
    sampler = None
    effective_shuffle = False
    if shuffle:
        if config.data.shuffle_strategy == "global_window":
            effective_shuffle = True
        else:
            blocks = window_dataset.grouped_blocks(config.data.shuffle_strategy)
            split_seed_offset = 0 if split == config.data.split else 1
            sampler = GroupedWindowSampler(
                blocks,
                seed=config.train.seed + split_seed_offset,
            )

    loader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last if shuffle else False,
        collate_fn=ContextWindowCollator(
            include_reconstruction_targets=config.loss.reconstruction_loss_weight > 0.0,
            masking_transform=masking_transform,
            patch_size=config.data.patch_size,
            normalization_mode=config.data.normalization_mode,
            normalization_eps=config.data.normalization_eps,
        ),
    )
    return window_dataset, loader


def _infer_fixed_num_features(*raw_datasets) -> int:
    """Infer a single feature-channel count across datasets.

    Returns:
        The shared D in `[T, D]`.

    Raises:
        FileNotFoundError: No samples are available.
        ValueError: Different samples contain different feature counts.
    """

    feature_counts: set[int] = set()
    sample_counts_by_feature: dict[int, int] = {}
    example_ids_by_feature: dict[int, list[str]] = {}
    for raw_dataset in raw_datasets:
        if raw_dataset is None:
            continue
        sample_num_features_getter = getattr(raw_dataset, "sample_num_features", None)
        sample_id_getter = getattr(raw_dataset, "sample_id", None)
        for index in range(len(raw_dataset)):
            if callable(sample_num_features_getter):
                feature_count = int(sample_num_features_getter(index))
                sample_id = (
                    str(sample_id_getter(index))
                    if callable(sample_id_getter)
                    else f"sample_{index:06d}"
                )
            else:
                sample = raw_dataset[index]
                feature_count = int(sample.series.shape[1])
                sample_id = str(sample.sample_id)
            feature_counts.add(feature_count)
            sample_counts_by_feature[feature_count] = sample_counts_by_feature.get(feature_count, 0) + 1
            example_ids = example_ids_by_feature.setdefault(feature_count, [])
            if len(example_ids) < 3:
                example_ids.append(sample_id)

    if not feature_counts:
        raise FileNotFoundError("Could not infer the number of feature channels from the datasets.")
    if len(feature_counts) != 1:
        feature_summary = ", ".join(
            f"{feature_count}D(count={sample_counts_by_feature[feature_count]}, "
            f"examples={example_ids_by_feature[feature_count]})"
            for feature_count in sorted(feature_counts)
        )
        raise ValueError(
            "The current training pipeline requires a fixed number of feature channels per run. "
            f"Found multiple values: {sorted(feature_counts)}. "
            f"Observed distribution: {feature_summary}. "
            "If you are using synthetic_tsad, fix `num_series` to one value "
            "(for example via `--num-series 6` or `num_series.min == num_series.max` in the raw config)."
        )
    return next(iter(feature_counts))


def _estimate_patch_feature_label_stats(raw_dataset, *, config: ExperimentConfig) -> dict[str, float]:
    """Estimate train-time patch-feature label balance from the raw training split."""

    cached = config.extras.get("_cached_train_label_stats")
    if isinstance(cached, dict) and "patch" in cached:
        return dict(cached["patch"])

    patch_stats, point_stats = _estimate_combined_label_stats(raw_dataset, config=config)
    config.extras["_cached_train_label_stats"] = {
        "patch": patch_stats,
        "point": point_stats,
    }
    return patch_stats


def _estimate_combined_label_stats(
    raw_dataset,
    *,
    config: ExperimentConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate patch-feature and point-feature label balance in a single pass."""

    patch_positive_units = 0
    patch_total_units = 0
    point_positive_units = 0
    point_total_units = 0
    num_windows = 0
    is_prewindowed = bool(getattr(raw_dataset, "is_prewindowed", False))
    if is_prewindowed:
        window_iterable = (raw_dataset[index] for index in range(len(raw_dataset)))
    else:
        windowizer = SlidingContextWindowizer(
            context_size=config.data.context_size,
            patch_size=config.data.patch_size,
            stride=config.data.stride,
            pad_short_sequences=config.data.pad_short_sequences,
            include_tail=config.data.include_tail,
        )
        window_iterable = (
            window
            for sample_index in range(len(raw_dataset))
            for window in windowizer.transform(raw_dataset[sample_index])
        )

    for window in window_iterable:
        patch_labels = np.asarray(window.patch_labels, dtype=np.uint8)
        patch_positive_units += int(patch_labels.sum())
        patch_total_units += int(patch_labels.size)

        point_mask = np.asarray(window.point_mask, dtype=np.uint8)
        if point_mask.ndim != 2:
            raise ValueError(
                "`window.point_mask` must have shape [W, D] for point anomaly statistics."
            )
        valid_length = max(0, int(window.context_end) - int(window.context_start))
        valid_point_mask = point_mask[:valid_length]
        point_positive_units += int(valid_point_mask.sum())
        point_total_units += int(valid_point_mask.size)
        num_windows += 1

    if patch_total_units <= 0:
        raise ValueError("Unable to estimate patch-feature statistics from an empty training split.")
    if point_total_units <= 0:
        raise ValueError("Unable to estimate point-feature statistics from an empty training split.")

    patch_negative_units = patch_total_units - patch_positive_units
    point_negative_units = point_total_units - point_positive_units
    patch_positive_rate = patch_positive_units / patch_total_units
    point_positive_rate = point_positive_units / point_total_units

    patch_stats = {
        "num_windows": float(num_windows),
        "num_patch_feature_units": float(patch_total_units),
        "num_positive_patch_feature_units": float(patch_positive_units),
        "num_negative_patch_feature_units": float(patch_negative_units),
        "patch_feature_positive_rate": float(patch_positive_rate),
    }
    point_stats = {
        "num_windows": float(num_windows),
        "num_point_feature_units": float(point_total_units),
        "num_positive_point_feature_units": float(point_positive_units),
        "num_negative_point_feature_units": float(point_negative_units),
        "point_feature_positive_rate": float(point_positive_rate),
    }
    return patch_stats, point_stats


def _estimate_point_feature_label_stats(raw_dataset, *, config: ExperimentConfig) -> dict[str, float]:
    """Estimate train-time point-feature label balance after windowization and padding trim."""

    cached = config.extras.get("_cached_train_label_stats")
    if isinstance(cached, dict) and "point" in cached:
        return dict(cached["point"])

    patch_stats, point_stats = _estimate_combined_label_stats(raw_dataset, config=config)
    config.extras["_cached_train_label_stats"] = {
        "patch": patch_stats,
        "point": point_stats,
    }
    return point_stats


def _resolve_anomaly_pos_weight(
    config: ExperimentConfig,
    *,
    train_raw_dataset,
) -> dict[str, float] | None:
    """Resolve anomaly pos_weight from config, optionally auto-estimating it from train labels."""

    configured = config.loss.anomaly_pos_weight
    if configured is None:
        print(
            "Anomaly pos_weight is disabled (`null`). "
            "Imbalanced patch-feature supervision may still collapse to all-negative predictions."
        )
        return None
    if isinstance(configured, str):
        if configured != "auto":
            raise ValueError(
                "`loss.anomaly_pos_weight` must be a positive float, null, or `auto`."
            )
        stats = _estimate_patch_feature_label_stats(train_raw_dataset, config=config)
        positive_units = stats["num_positive_patch_feature_units"]
        negative_units = stats["num_negative_patch_feature_units"]
        if positive_units <= 0:
            raise ValueError(
                "Cannot auto-estimate `loss.anomaly_pos_weight` because the train split "
                "contains zero positive patch-feature labels."
            )
        resolved = float(negative_units / positive_units)
        config.loss.anomaly_pos_weight = resolved
        config.extras["train_patch_feature_label_stats"] = stats
        config.extras["resolved_anomaly_pos_weight"] = resolved
        print(
            "Auto anomaly_pos_weight resolved from train patch-feature labels: "
            f"pos_weight={resolved:.6f}, positive_rate={stats['patch_feature_positive_rate']:.6f}, "
            f"positives={int(positive_units)}, negatives={int(negative_units)}, "
            f"windows={int(stats['num_windows'])}."
        )
        return stats

    resolved = float(configured)
    config.loss.anomaly_pos_weight = resolved
    config.extras["resolved_anomaly_pos_weight"] = resolved
    print(f"Using configured anomaly_pos_weight={resolved:.6f}.")
    return None


def _resolve_point_anomaly_pos_weight(
    config: ExperimentConfig,
    *,
    train_raw_dataset,
) -> dict[str, float] | None:
    """Resolve point anomaly pos_weight when point-level auxiliary supervision is enabled."""

    if config.loss.point_anomaly_loss_weight <= 0.0:
        return None

    configured = config.loss.point_anomaly_pos_weight
    if configured is None:
        print(
            "Point anomaly pos_weight is disabled (`null`). "
            "Imbalanced point-feature supervision may still collapse to all-negative predictions."
        )
        return None
    if isinstance(configured, str):
        if configured != "auto":
            raise ValueError(
                "`loss.point_anomaly_pos_weight` must be a positive float, null, or `auto`."
            )
        stats = _estimate_point_feature_label_stats(train_raw_dataset, config=config)
        positive_units = stats["num_positive_point_feature_units"]
        negative_units = stats["num_negative_point_feature_units"]
        if positive_units <= 0:
            raise ValueError(
                "Cannot auto-estimate `loss.point_anomaly_pos_weight` because the train split "
                "contains zero positive point-feature labels."
            )
        resolved = float(negative_units / positive_units)
        config.loss.point_anomaly_pos_weight = resolved
        config.extras["train_point_feature_label_stats"] = stats
        config.extras["resolved_point_anomaly_pos_weight"] = resolved
        print(
            "Auto point_anomaly_pos_weight resolved from train point-feature labels: "
            f"pos_weight={resolved:.6f}, positive_rate={stats['point_feature_positive_rate']:.6f}, "
            f"positives={int(positive_units)}, negatives={int(negative_units)}, "
            f"windows={int(stats['num_windows'])}."
        )
        return stats

    resolved = float(configured)
    config.loss.point_anomaly_pos_weight = resolved
    config.extras["resolved_point_anomaly_pos_weight"] = resolved
    print(f"Using configured point_anomaly_pos_weight={resolved:.6f}.")
    return None


def _resolve_device(requested: str) -> torch.device:
    """Resolve runtime device with a safe CUDA fallback."""

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _build_validation_evaluator(config: ExperimentConfig):
    """Build the validation-time evaluator used for checkpoint selection."""

    if config.eval.task == "patch_feature":
        return PatchFeatureEvaluator(
            patch_size=config.data.patch_size,
            patch_feature_score_aggregation=config.eval.patch_feature_score_aggregation,
            threshold=config.eval.threshold,
            threshold_search=config.eval.threshold_search,
            threshold_search_metric=config.eval.threshold_search_metric,
            report_per_feature=False,
            report_per_sample=False,
        )
    return TimeRCDEvaluator(
        patch_size=config.data.patch_size,
        score_reduction=config.eval.score_reduction,
        point_score_aggregation=config.eval.point_score_aggregation,
        threshold=config.eval.threshold,
        threshold_search=config.eval.threshold_search,
        threshold_search_metric=config.eval.threshold_search_metric,
    )



def _default_inspection_splits(config: ExperimentConfig) -> list[str]:
    """Return configured split order for optional data-quality inspection."""

    ordered = [config.data.split, config.data.validation_split, config.data.test_split]
    unique: list[str] = []
    for split in ordered:
        if split not in unique:
            unique.append(split)
    return unique


def _run_data_quality_inspection(
    config: ExperimentConfig,
    *,
    train_raw_dataset,
    val_raw_dataset,
    max_samples: int | None,
    output_path: Path | None,
) -> None:
    """Run optional split-level data-quality diagnostics before model training."""

    if bool(getattr(train_raw_dataset, "is_prewindowed", False)):
        print(
            "Skipping data quality inspection because the selected dataset is already window-packed. "
            "Inspection currently expects raw sequence samples."
        )
        return

    datasets_by_split: dict[str, object] = {
        config.data.split: train_raw_dataset,
    }
    missing_splits: list[str] = []

    if val_raw_dataset is not None:
        datasets_by_split[config.data.validation_split] = val_raw_dataset

    for split in _default_inspection_splits(config):
        if split in datasets_by_split:
            continue
        try:
            datasets_by_split[split] = _build_raw_dataset(config, split=split)
        except FileNotFoundError:
            missing_splits.append(split)

    windowizer = SlidingContextWindowizer(
        context_size=config.data.context_size,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        pad_short_sequences=config.data.pad_short_sequences,
        include_tail=config.data.include_tail,
    )
    inspector = DataQualityInspector(
        windowizer=windowizer,
        max_samples=max_samples,
    )
    report = inspector.inspect_many(
        datasets_by_split,
        expected_training_split=config.data.split,
        missing_splits=missing_splits,
    )

    payload = report.to_dict()
    output_json = json.dumps(payload, indent=2, ensure_ascii=False)

    report_path = output_path or (config.train.output_dir / "data_quality_report.json")
    report_path = _resolve_path(report_path, base_dir=PROJECT_ROOT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(output_json, encoding="utf-8")

    config.extras["data_quality_summary"] = payload["summary"]

    print("Data quality inspection completed. Summary:")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Data quality report written to: {report_path}")
    print(output_json)


def parse_args() -> argparse.Namespace:
    """Parse train entrypoint CLI arguments."""

    default_config = TRAIN_TSAD_ROOT / "configs" / "timercd_pretrain_paper_aligned.json"
    parser = argparse.ArgumentParser(description="Train the paper-aligned TimeRCD model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to a JSON or YAML experiment config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional max epoch override.",
    )
    parser.add_argument(
        "--inspect-data",
        action="store_true",
        help="Run split-level data quality inspection before training starts.",
    )
    parser.add_argument(
        "--inspect-max-samples",
        type=int,
        default=None,
        help="Optional cap on inspected samples per split during data inspection.",
    )
    parser.add_argument(
        "--inspect-output",
        type=Path,
        default=None,
        help="Optional path to save the data quality report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run end-to-end training from config loading to summary export.

    Workflow:
    1. Parse config and CLI overrides.
    2. Build datasets/loaders and infer fixed model feature dimension.
    3. Build model/loss/optimizer/scheduler/trainer.
    4. Execute fitting and persist history + summary artifacts.
    """

    args = parse_args()
    config = ExperimentConfig.from_file(args.config)
    if args.max_epochs is not None:
        config.train.max_epochs = args.max_epochs
    if args.device is not None:
        config.train.device = args.device

    config.data.dataset_root = _resolve_path(config.data.dataset_root, base_dir=PROJECT_ROOT)
    config.train.output_dir = _resolve_path(config.train.output_dir, base_dir=PROJECT_ROOT)
    config.train.output_dir.mkdir(parents=True, exist_ok=True)

    _set_global_seed(config.train.seed)

    # Training split is required.
    train_raw_dataset = _build_raw_dataset(config, split=config.data.split)

    # Validation split is optional. Missing validation keeps a train-only loop.
    val_raw_dataset = None
    try:
        val_raw_dataset = _build_raw_dataset(config, split=config.data.validation_split)
    except FileNotFoundError:
        print(f"Validation split `{config.data.validation_split}` not found. Training without validation.")

    if args.inspect_data:
        _run_data_quality_inspection(
            config,
            train_raw_dataset=train_raw_dataset,
            val_raw_dataset=val_raw_dataset,
            max_samples=args.inspect_max_samples,
            output_path=args.inspect_output,
        )

    _resolve_anomaly_pos_weight(
        config,
        train_raw_dataset=train_raw_dataset,
    )
    _resolve_point_anomaly_pos_weight(
        config,
        train_raw_dataset=train_raw_dataset,
    )

    # Current model path assumes a fixed feature count across all samples.
    num_features = _infer_fixed_num_features(train_raw_dataset, val_raw_dataset)

    _, train_loader = _build_window_loader(
        train_raw_dataset,
        config=config,
        split=config.data.split,
        shuffle=config.data.shuffle,
        batch_size=config.data.batch_size,
    )
    val_loader = None
    if val_raw_dataset is not None:
        _, val_loader = _build_window_loader(
            val_raw_dataset,
            config=config,
            split=config.data.validation_split,
            shuffle=False,
            batch_size=config.data.eval_batch_size,
        )

    device = _resolve_device(config.train.device)
    model = TimeRCDModel(
        patch_size=config.model.patch_size,
        d_model=config.model.d_model,
        d_proj=config.model.d_proj,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        max_patches=config.data.num_patches,
        max_features=max(num_features, 1),
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        attention_dropout=config.model.attention_dropout,
        activation=config.model.activation,
        use_learned_positional_encoding=config.model.use_learned_positional_encoding,
        use_shared_output_projection=config.model.use_shared_output_projection,
        use_observation_space_anomaly_head=config.model.use_observation_space_anomaly_head,
        anomaly_patch_aggregation=config.model.anomaly_patch_aggregation,
        use_reconstruction_head=config.loss.reconstruction_loss_weight > 0.0,
    )
    loss_fn = TimeRCDMultiTaskLoss.from_config(config.loss)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(
        optimizer,
        config.optimizer,
        max_epochs=config.train.max_epochs,
    )
    checkpoint_manager = CheckpointManager(config.train.output_dir)
    checkpoint_manager.save_json("config.json", config.to_dict())

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_manager=checkpoint_manager,
        gradient_clip_norm=config.train.gradient_clip_norm,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_every_n_steps=config.train.log_every_n_steps,
        mixed_precision=config.train.mixed_precision,
        progress_write_interval_seconds=config.train.progress_write_interval_seconds,
    )
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.train.max_epochs,
        validate_every_n_epochs=config.train.validate_every_n_epochs,
        early_stopping_patience=config.train.early_stopping_patience,
        config_payload=config.to_dict(),
        validation_evaluator_factory=(
            (lambda: _build_validation_evaluator(config))
            if val_loader is not None
            else None
        ),
        monitor_metric=config.eval.primary_metric if val_loader is not None else "total_loss",
        monitor_mode=config.eval.primary_metric_mode if val_loader is not None else "min",
    )

    checkpoint_manager.save_json("history.json", result.history)
    summary = {
        "best_epoch": result.best_epoch,
        "best_metric": result.best_metric,
        "latest_checkpoint": result.latest_checkpoint,
        "best_checkpoint": result.best_checkpoint,
    }
    checkpoint_manager.save_json("summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
