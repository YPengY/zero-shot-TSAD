from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..data import auto_select_available_split, build_evaluation_loader, build_raw_dataset, infer_fixed_num_features
from ..evaluation import build_evaluator
from ..models import build_timercd_model
from ..utils import resolve_path, resolve_torch_device, write_json_file


@dataclass(frozen=True, slots=True)
class EvaluationOverrides:
    """CLI overrides applied during offline evaluation."""

    device: str | None = None
    split: str | None = None
    checkpoint: Path | None = None
    output: Path | None = None


@dataclass(frozen=True, slots=True)
class EvaluationWorkflowResult:
    """Metrics and artifacts produced by offline evaluation."""

    config: ExperimentConfig
    split: str
    checkpoint_path: Path
    metrics: dict[str, Any]
    output_path: Path | None = None


def prepare_evaluation_config(
    config_path: Path,
    *,
    base_dir: Path,
    overrides: EvaluationOverrides,
) -> ExperimentConfig:
    """Load the evaluation config and resolve path-like fields."""

    config = ExperimentConfig.from_file(config_path).clone()
    if overrides.device is not None:
        config.train.device = overrides.device

    config.data.dataset_root = resolve_path(config.data.dataset_root, base_dir=base_dir)
    config.train.output_dir = resolve_path(config.train.output_dir, base_dir=base_dir)
    return config


def run_evaluation(
    config_path: Path,
    *,
    base_dir: Path,
    overrides: EvaluationOverrides,
    logger: logging.Logger,
) -> EvaluationWorkflowResult:
    """Run end-to-end checkpoint evaluation."""

    config = prepare_evaluation_config(
        config_path,
        base_dir=base_dir,
        overrides=overrides,
    )
    split = overrides.split or auto_select_available_split(config.data, base_dir=base_dir)
    checkpoint_path = resolve_path(
        overrides.checkpoint or (config.train.output_dir / "best.pt"),
        base_dir=base_dir,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_dataset = build_raw_dataset(config.data, split=split, base_dir=base_dir)
    num_features = infer_fixed_num_features(raw_dataset)
    _, loader = build_evaluation_loader(
        raw_dataset,
        data_config=config.data,
        batch_size=config.data.eval_batch_size,
    )

    device = resolve_torch_device(config.train.device, logger=logger)

    model = build_timercd_model(
        data_config=config.data,
        model_config=config.model,
        loss_config=config.loss,
        num_features=num_features,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    evaluator = build_evaluator(
        data_config=config.data,
        eval_config=config.eval,
    )
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            evaluator.update(batch, model(batch))

    metrics = evaluator.compute()
    metrics.update(
        {
            "split": split,
            "checkpoint": str(checkpoint_path),
        }
    )

    output_path = None
    if overrides.output is not None:
        output_path = resolve_path(overrides.output, base_dir=base_dir)
        write_json_file(output_path, metrics)
        logger.info("Evaluation metrics written to %s.", output_path)

    return EvaluationWorkflowResult(
        config=config,
        split=split,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        output_path=output_path,
    )
