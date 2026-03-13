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
    RandomPatchMaskingTransform,
    ShardedSyntheticTsadDataset,
    SlidingContextWindowizer,
    SyntheticTsadDataset,
)
from train_tsad.losses import TimeRCDMultiTaskLoss  # noqa: E402
from train_tsad.models import TimeRCDModel  # noqa: E402
from train_tsad.training import (  # noqa: E402
    CheckpointManager,
    Trainer,
    build_optimizer,
    build_scheduler,
)


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_raw_dataset(config: ExperimentConfig, *, split: str):
    dataset_root = _resolve_path(config.data.dataset_root, base_dir=PROJECT_ROOT)
    manifest_path = config.data.manifest_path(split)
    manifest_path = _resolve_path(manifest_path, base_dir=PROJECT_ROOT)
    manifest_value: Path | None = manifest_path if manifest_path.exists() else None

    dataset_cls = ShardedSyntheticTsadDataset if config.data.use_sharded_dataset else SyntheticTsadDataset
    return dataset_cls(
        root_dir=dataset_root,
        split=split,
        manifest_path=manifest_value,
    )


def _build_window_loader(
    raw_dataset,
    *,
    config: ExperimentConfig,
    split: str,
    shuffle: bool,
    batch_size: int,
) -> tuple[ContextWindowDataset, DataLoader]:
    windowizer = SlidingContextWindowizer(
        context_size=config.data.context_size,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        pad_short_sequences=config.data.pad_short_sequences,
        include_tail=config.data.include_tail,
    )
    window_dataset = ContextWindowDataset(raw_dataset, windowizer)
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
    loader = DataLoader(
        window_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last if shuffle else False,
        collate_fn=ContextWindowCollator(
            include_reconstruction_targets=config.loss.reconstruction_loss_weight > 0.0,
            masking_transform=masking_transform,
        ),
    )
    return window_dataset, loader


def _infer_fixed_num_features(*raw_datasets) -> int:
    feature_counts: set[int] = set()
    for raw_dataset in raw_datasets:
        if raw_dataset is None:
            continue
        for index in range(len(raw_dataset)):
            feature_counts.add(int(raw_dataset[index].series.shape[1]))

    if not feature_counts:
        raise FileNotFoundError("Could not infer the number of feature channels from the datasets.")
    if len(feature_counts) != 1:
        raise ValueError(
            "The current training pipeline requires a fixed number of feature channels per run. "
            f"Found multiple values: {sorted(feature_counts)}."
        )
    return next(iter(feature_counts))


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    default_config = TRAIN_TSAD_ROOT / "configs" / "timercd_small.json"
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
    return parser.parse_args()


def main() -> None:
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

    train_raw_dataset = _build_raw_dataset(config, split=config.data.split)

    val_raw_dataset = None
    try:
        val_raw_dataset = _build_raw_dataset(config, split=config.data.validation_split)
    except FileNotFoundError:
        print(f"Validation split `{config.data.validation_split}` not found. Training without validation.")

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
    )
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.train.max_epochs,
        validate_every_n_epochs=config.train.validate_every_n_epochs,
        early_stopping_patience=config.train.early_stopping_patience,
        config_payload=config.to_dict(),
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
