"""High-level orchestration for end-to-end training runs.

This module is the main assembly layer for the training stack. It wires
config loading, dataset construction, preflight inspection, model/loss
instantiation, the trainer runtime, and artifact persistence into one
deterministic workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..data import build_raw_dataset, build_window_loader, infer_fixed_num_features
from ..evaluation import build_evaluator
from ..losses import TimeRCDMultiTaskLoss
from ..models import build_timercd_model
from ..training import (
    CheckpointManager,
    FitResult,
    ResolvedLossWeights,
    Trainer,
    build_optimizer,
    build_scheduler,
    resolve_loss_weights,
    run_data_quality_inspection,
)
from ..utils import resolve_path, resolve_torch_device, seed_everything


@dataclass(frozen=True, slots=True)
class TrainingOverrides:
    """CLI overrides applied on top of the experiment config."""

    device: str | None = None
    max_epochs: int | None = None
    inspect_data: bool = False
    inspect_max_samples: int | None = None
    inspect_output: Path | None = None


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Named files persisted by one completed training workflow."""

    output_dir: Path
    config_snapshot_path: Path
    history_path: Path
    summary_path: Path
    quality_report_path: Path | None = None


@dataclass(frozen=True, slots=True)
class TrainingWorkflowResult:
    """Structured result returned to CLI or service entrypoints."""

    config: ExperimentConfig
    fit_result: FitResult
    artifacts: TrainingArtifacts
    resolved_loss_weights: ResolvedLossWeights
    summary: dict[str, Any]


def prepare_training_config(
    config_path: Path,
    *,
    base_dir: Path,
    overrides: TrainingOverrides,
) -> ExperimentConfig:
    """Load one config file and apply runtime-only overrides.

    This keeps CLI concerns such as path resolution and temporary epoch/device
    overrides out of the immutable config files stored on disk.
    """

    config = ExperimentConfig.from_file(config_path).clone()
    if overrides.max_epochs is not None:
        config.train.max_epochs = overrides.max_epochs
    if overrides.device is not None:
        config.train.device = overrides.device

    config.data.dataset_root = resolve_path(config.data.dataset_root, base_dir=base_dir)
    config.train.output_dir = resolve_path(config.train.output_dir, base_dir=base_dir)
    config.train.output_dir.mkdir(parents=True, exist_ok=True)
    return config


def run_training(
    config_path: Path,
    *,
    base_dir: Path,
    overrides: TrainingOverrides,
    logger: logging.Logger,
) -> TrainingWorkflowResult:
    """Run one full training workflow from config loading to artifact writing."""

    config = prepare_training_config(
        config_path,
        base_dir=base_dir,
        overrides=overrides,
    )
    seed_everything(config.train.seed)

    train_raw_dataset = build_raw_dataset(
        config.data,
        split=config.data.split,
        base_dir=base_dir,
    )

    val_raw_dataset = None
    try:
        val_raw_dataset = build_raw_dataset(
            config.data,
            split=config.data.validation_split,
            base_dir=base_dir,
        )
    except FileNotFoundError:
        logger.warning(
            "Validation split `%s` was not found. Training will continue without validation.",
            config.data.validation_split,
        )

    quality_result = None
    if overrides.inspect_data:
        quality_result = run_data_quality_inspection(
            data_config=config.data,
            train_output_dir=config.train.output_dir,
            train_raw_dataset=train_raw_dataset,
            val_raw_dataset=val_raw_dataset,
            base_dir=base_dir,
            max_samples=overrides.inspect_max_samples,
            output_path=overrides.inspect_output,
            logger=logger,
        )

    resolved_loss_weights = resolve_loss_weights(
        config.loss,
        data_config=config.data,
        train_raw_dataset=train_raw_dataset,
        logger=logger,
    )
    config.loss.anomaly_pos_weight = resolved_loss_weights.anomaly_pos_weight
    config.loss.point_anomaly_pos_weight = resolved_loss_weights.point_anomaly_pos_weight

    num_features = infer_fixed_num_features(train_raw_dataset, val_raw_dataset)
    _, train_loader = build_window_loader(
        train_raw_dataset,
        data_config=config.data,
        loss_config=config.loss,
        train_seed=config.train.seed,
        split=config.data.split,
        shuffle=config.data.shuffle,
        batch_size=config.data.batch_size,
    )
    val_loader = None
    if val_raw_dataset is not None:
        _, val_loader = build_window_loader(
            val_raw_dataset,
            data_config=config.data,
            loss_config=config.loss,
            train_seed=config.train.seed,
            split=config.data.validation_split,
            shuffle=False,
            batch_size=config.data.eval_batch_size,
        )

    model = build_timercd_model(
        data_config=config.data,
        model_config=config.model,
        loss_config=config.loss,
        num_features=num_features,
    )
    loss_fn = TimeRCDMultiTaskLoss.from_config(config.loss)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(
        optimizer,
        config.optimizer,
        max_epochs=config.train.max_epochs,
    )

    checkpoint_manager = CheckpointManager(config.train.output_dir)
    config_snapshot_path = checkpoint_manager.save_json("config.json", config.to_dict())

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=resolve_torch_device(config.train.device, logger=logger),
        checkpoint_manager=checkpoint_manager,
        gradient_clip_norm=config.train.gradient_clip_norm,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_every_n_steps=config.train.log_every_n_steps,
        mixed_precision=config.train.mixed_precision,
        progress_write_interval_seconds=config.train.progress_write_interval_seconds,
        logger=logger,
    )
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.train.max_epochs,
        validate_every_n_epochs=config.train.validate_every_n_epochs,
        early_stopping_patience=config.train.early_stopping_patience,
        config_payload=config.to_dict(),
        validation_evaluator_factory=(
            (
                lambda: build_evaluator(
                    data_config=config.data,
                    eval_config=config.eval,
                    report_per_feature=False,
                    report_per_sample=False,
                )
            )
            if val_loader is not None
            else None
        ),
        monitor_metric=config.eval.primary_metric if val_loader is not None else "total_loss",
        monitor_mode=config.eval.primary_metric_mode if val_loader is not None else "min",
    )

    history_path = checkpoint_manager.save_json("history.json", fit_result.history)
    summary = {
        "best_epoch": fit_result.best_epoch,
        "best_metric": fit_result.best_metric,
        "latest_checkpoint": fit_result.latest_checkpoint,
        "best_checkpoint": fit_result.best_checkpoint,
        "resolved_anomaly_pos_weight": resolved_loss_weights.anomaly_pos_weight,
        "resolved_point_anomaly_pos_weight": resolved_loss_weights.point_anomaly_pos_weight,
        "data_quality_report": (
            str(quality_result.report_path) if quality_result is not None else None
        ),
    }
    summary_path = checkpoint_manager.save_json("summary.json", summary)
    logger.info(
        "Training finished. best_epoch=%s best_metric=%s output_dir=%s",
        fit_result.best_epoch,
        fit_result.best_metric,
        config.train.output_dir,
    )

    return TrainingWorkflowResult(
        config=config,
        fit_result=fit_result,
        artifacts=TrainingArtifacts(
            output_dir=config.train.output_dir,
            config_snapshot_path=config_snapshot_path,
            history_path=history_path,
            summary_path=summary_path,
            quality_report_path=(
                quality_result.report_path if quality_result is not None else None
            ),
        ),
        resolved_loss_weights=resolved_loss_weights,
        summary=summary,
    )
