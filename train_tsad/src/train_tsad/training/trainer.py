from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..interfaces import Batch, LossProtocol, ModelProtocol
from .checkpoint import CheckpointManager


@dataclass(slots=True)
class EpochResult:
    """Aggregated metrics for a single epoch."""

    epoch: int
    split: str
    metrics: dict[str, float]


@dataclass(slots=True)
class FitResult:
    """Final training summary returned by the trainer."""

    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int | None = None
    best_metric: float | None = None
    latest_checkpoint: str | None = None
    best_checkpoint: str | None = None


class Trainer:
    """Minimal training loop for TimeRCD-style pretraining."""

    def __init__(
        self,
        *,
        model: ModelProtocol | nn.Module,
        loss_fn: LossProtocol | nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: LRScheduler | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        gradient_clip_norm: float | None = None,
        gradient_accumulation_steps: int = 1,
        log_every_n_steps: int = 50,
    ) -> None:
        if gradient_accumulation_steps <= 0:
            raise ValueError("`gradient_accumulation_steps` must be positive.")
        if log_every_n_steps <= 0:
            raise ValueError("`log_every_n_steps` must be positive.")

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_every_n_steps = log_every_n_steps

        if isinstance(self.model, nn.Module):
            self.model.to(self.device)

    def fit(
        self,
        *,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch] | None = None,
        max_epochs: int,
        validate_every_n_epochs: int = 1,
        early_stopping_patience: int = 0,
        config_payload: dict[str, Any] | None = None,
    ) -> FitResult:
        if max_epochs <= 0:
            raise ValueError("`max_epochs` must be positive.")
        if validate_every_n_epochs <= 0:
            raise ValueError("`validate_every_n_epochs` must be positive.")
        if early_stopping_patience < 0:
            raise ValueError("`early_stopping_patience` cannot be negative.")

        history: list[dict[str, Any]] = []
        best_epoch: int | None = None
        best_metric: float | None = None
        latest_checkpoint: str | None = None
        best_checkpoint: str | None = None
        bad_epoch_count = 0

        for epoch in range(1, max_epochs + 1):
            train_metrics = self._run_epoch(
                loader=train_loader,
                epoch=epoch,
                split="train",
                training=True,
            )

            val_metrics: dict[str, float] | None = None
            if val_loader is not None and epoch % validate_every_n_epochs == 0:
                val_metrics = self._run_epoch(
                    loader=val_loader,
                    epoch=epoch,
                    split="val",
                    training=False,
                )

            if self.scheduler is not None:
                self.scheduler.step()

            monitored_metrics = val_metrics or train_metrics
            monitored_value = monitored_metrics["total_loss"]
            is_best = best_metric is None or monitored_value < best_metric

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
            }
            if val_metrics is not None:
                epoch_record["val"] = val_metrics
            history.append(epoch_record)

            if self.checkpoint_manager is not None:
                latest_path = self.checkpoint_manager.save_latest(
                    epoch=epoch,
                    model=self._nn_module,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=epoch_record,
                    config=config_payload,
                )
                latest_checkpoint = str(latest_path)

                if is_best:
                    best_path = self.checkpoint_manager.save_best(
                        epoch=epoch,
                        model=self._nn_module,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        metrics=epoch_record,
                        config=config_payload,
                    )
                    best_checkpoint = str(best_path)

            if is_best:
                best_metric = monitored_value
                best_epoch = epoch
                bad_epoch_count = 0
            else:
                bad_epoch_count += 1

            self._print_epoch_summary(epoch, train_metrics, val_metrics)

            if early_stopping_patience > 0 and bad_epoch_count >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch={best_epoch}, best_total_loss={best_metric:.6f}."
                )
                break

        return FitResult(
            history=history,
            best_epoch=best_epoch,
            best_metric=best_metric,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=best_checkpoint,
        )

    @property
    def _nn_module(self) -> nn.Module:
        if not isinstance(self.model, nn.Module):
            raise TypeError("Trainer currently requires `model` to be an `nn.Module` instance.")
        return self.model

    def _run_epoch(
        self,
        *,
        loader: DataLoader[Batch],
        epoch: int,
        split: str,
        training: bool,
    ) -> dict[str, float]:
        module = self._nn_module
        module.train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        metric_sums: dict[str, float] = {}
        sample_count = 0
        num_batches = len(loader)

        for step, batch in enumerate(loader, start=1):
            batch = batch.to(self.device)
            batch_size = int(batch.inputs.shape[0])
            sample_count += batch_size

            with torch.set_grad_enabled(training):
                output = module(batch)
                loss_output = self.loss_fn(batch, output)
                raw_loss = loss_output.loss

                if training:
                    scaled_loss = raw_loss / self.gradient_accumulation_steps
                    scaled_loss.backward()

                    is_update_step = (
                        step % self.gradient_accumulation_steps == 0 or step == num_batches
                    )
                    if is_update_step:
                        if self.gradient_clip_norm is not None:
                            clip_grad_norm_(module.parameters(), self.gradient_clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

            batch_metrics = dict(loss_output.metrics)
            batch_metrics.setdefault("total_loss", float(raw_loss.detach().item()))
            for name, value in batch_metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * batch_size

            if training and (step % self.log_every_n_steps == 0 or step == num_batches):
                print(
                    f"[train] epoch={epoch} step={step}/{num_batches} "
                    f"loss={float(raw_loss.detach().item()):.6f}"
                )

        if sample_count == 0:
            raise ValueError(f"The `{split}` loader produced zero batches.")

        return {name: total / sample_count for name, total in metric_sums.items()}

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        train_loss = train_metrics.get("total_loss", float("nan"))
        message = f"[epoch {epoch}] train_total_loss={train_loss:.6f}"
        if val_metrics is not None:
            val_loss = val_metrics.get("total_loss", float("nan"))
            message += f" val_total_loss={val_loss:.6f}"
        print(message)


__all__ = ["EpochResult", "FitResult", "Trainer"]
