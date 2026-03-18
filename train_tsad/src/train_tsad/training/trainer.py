from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..interfaces import Batch, LossProtocol, ModelProtocol
from .checkpoint import CheckpointManager


class EpochEvaluatorProtocol(Protocol):
    """Runtime protocol for validation-time evaluators."""

    def update(self, batch: Batch, output: Any) -> None:
        ...

    def compute(self) -> dict[str, Any]:
        ...


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
        mixed_precision: bool = False,
        progress_write_interval_seconds: float = 2.0,
    ) -> None:
        """Configure the trainer runtime state.

        Args:
            model: Forward model that consumes `Batch` and returns `ModelOutput`.
            loss_fn: Loss callable that maps `(Batch, ModelOutput)` to `LossOutput`.
            optimizer: Optimizer used for parameter updates.
            device: Device where model and batches are placed.
            scheduler: Optional epoch-level scheduler stepped after each epoch.
            checkpoint_manager: Optional artifact writer for latest/best checkpoints.
            gradient_clip_norm: Optional global grad-norm clipping threshold.
            gradient_accumulation_steps: Number of mini-batches per optimizer step.
            log_every_n_steps: Training-step log interval.
            mixed_precision: Enable AMP autocast + scaler on CUDA.
            progress_write_interval_seconds: Minimum wall time between progress writes.
        """

        if gradient_accumulation_steps <= 0:
            raise ValueError("`gradient_accumulation_steps` must be positive.")
        if log_every_n_steps <= 0:
            raise ValueError("`log_every_n_steps` must be positive.")
        if progress_write_interval_seconds <= 0:
            raise ValueError("`progress_write_interval_seconds` must be positive.")

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_every_n_steps = log_every_n_steps
        self.mixed_precision = bool(mixed_precision)
        self.progress_write_interval_seconds = float(progress_write_interval_seconds)
        self._last_progress_write_time = 0.0

        if isinstance(self.model, nn.Module):
            self.model.to(self.device)

        self._amp_enabled = bool(self.mixed_precision and self.device.type == "cuda")
        self._grad_scaler = torch.amp.GradScaler("cuda") if self._amp_enabled else None

    def fit(
        self,
        *,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch] | None = None,
        max_epochs: int,
        validate_every_n_epochs: int = 1,
        early_stopping_patience: int = 0,
        config_payload: dict[str, Any] | None = None,
        validation_evaluator_factory: Callable[[], EpochEvaluatorProtocol] | None = None,
        monitor_metric: str = "total_loss",
        monitor_mode: str = "min",
    ) -> FitResult:
        """Run the full train/validation loop.

        Workflow:
        1. Execute one training epoch.
        2. Optionally execute validation on configured cadence.
        3. Step scheduler, checkpoint latest/best, track best loss.
        4. Stop early when no improvement exceeds patience.
        """

        if max_epochs <= 0:
            raise ValueError("`max_epochs` must be positive.")
        if validate_every_n_epochs <= 0:
            raise ValueError("`validate_every_n_epochs` must be positive.")
        if early_stopping_patience < 0:
            raise ValueError("`early_stopping_patience` cannot be negative.")
        if not monitor_metric.strip():
            raise ValueError("`monitor_metric` cannot be empty.")
        if monitor_mode not in {"min", "max"}:
            raise ValueError("`monitor_mode` must be one of: min, max.")

        history: list[dict[str, Any]] = []
        best_epoch: int | None = None
        best_metric: float | None = None
        latest_checkpoint: str | None = None
        best_checkpoint: str | None = None
        bad_epoch_count = 0
        fit_started_at = time.perf_counter()
        train_batches = len(train_loader)
        val_batches = len(val_loader) if val_loader is not None else 0
        planned_validation_epochs = sum(
            1 for epoch in range(1, max_epochs + 1) if val_loader is not None and epoch % validate_every_n_epochs == 0
        )
        planned_total_units = max(1, max_epochs * train_batches + planned_validation_epochs * val_batches)
        latest_train_metrics: dict[str, float] | None = None
        latest_val_metrics: dict[str, float] | None = None

        def completed_units_before_epoch(epoch: int) -> int:
            units = max(epoch - 1, 0) * train_batches
            if val_loader is None or val_batches <= 0:
                return units
            units += sum(val_batches for previous_epoch in range(1, epoch) if previous_epoch % validate_every_n_epochs == 0)
            return units

        def emit_progress(
            *,
            stage: str,
            status: str,
            epoch_current: int,
            progress_units: int,
            split: str | None = None,
            split_step_current: int | None = None,
            split_step_total: int | None = None,
            current_split_metrics: dict[str, float] | None = None,
            current_batch_metrics: dict[str, float] | None = None,
            latest_epoch_record: dict[str, Any] | None = None,
            stopped_early: bool = False,
            error: str | None = None,
            force_write: bool = False,
        ) -> None:
            progress_ratio = min(1.0, max(0.0, progress_units / planned_total_units))
            elapsed_seconds = max(0.0, time.perf_counter() - fit_started_at)
            eta_seconds = None
            if 0.0 < progress_ratio < 1.0:
                eta_seconds = elapsed_seconds * (1.0 - progress_ratio) / progress_ratio
            self._write_progress(
                stage=stage,
                status=status,
                epoch_current=epoch_current,
                epoch_total=max_epochs,
                epochs_completed=len(history),
                split=split,
                split_step_current=split_step_current,
                split_step_total=split_step_total,
                learning_rate=self._current_learning_rate(),
                elapsed_seconds=elapsed_seconds,
                eta_seconds=eta_seconds,
                overall_progress_ratio=progress_ratio,
                latest_train_metrics=latest_train_metrics,
                latest_val_metrics=latest_val_metrics,
                current_split_metrics=current_split_metrics,
                current_batch_metrics=current_batch_metrics,
                best_epoch=best_epoch,
                best_metric=best_metric,
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                latest_epoch_record=latest_epoch_record,
                train_batches_per_epoch=train_batches,
                val_batches_per_epoch=val_batches,
                stopped_early=stopped_early,
                error=error,
                force=force_write,
            )

        emit_progress(
            stage="starting",
            status="running",
            epoch_current=0,
            progress_units=0,
            force_write=True,
        )

        for epoch in range(1, max_epochs + 1):
            train_metrics = self._run_epoch(
                loader=train_loader,
                epoch=epoch,
                split="train",
                training=True,
                progress_callback=lambda update, epoch=epoch: emit_progress(
                    stage="training",
                    status="running",
                    epoch_current=epoch,
                    split="train",
                    split_step_current=update["step_current"],
                    split_step_total=update["step_total"],
                    current_split_metrics=update["running_metrics"],
                    current_batch_metrics=update["batch_metrics"],
                    progress_units=completed_units_before_epoch(epoch) + update["step_current"],
                ),
            )
            latest_train_metrics = train_metrics

            val_metrics: dict[str, float] | None = None
            if val_loader is not None and epoch % validate_every_n_epochs == 0:
                evaluator = (
                    validation_evaluator_factory()
                    if validation_evaluator_factory is not None
                    else None
                )
                val_metrics = self._run_epoch(
                    loader=val_loader,
                    epoch=epoch,
                    split="val",
                    training=False,
                    evaluator=evaluator,
                    progress_callback=lambda update, epoch=epoch: emit_progress(
                        stage="validation",
                        status="running",
                        epoch_current=epoch,
                        split="val",
                        split_step_current=update["step_current"],
                        split_step_total=update["step_total"],
                        current_split_metrics=update["running_metrics"],
                        current_batch_metrics=update["batch_metrics"],
                        progress_units=completed_units_before_epoch(epoch) + train_batches + update["step_current"],
                    ),
                )
                latest_val_metrics = val_metrics

            if self.scheduler is not None:
                self.scheduler.step()

            # Prefer validation loss when available; otherwise monitor train loss.
            monitored_metrics = val_metrics or train_metrics
            if monitor_metric not in monitored_metrics:
                available = ", ".join(sorted(monitored_metrics))
                raise KeyError(
                    f"Monitor metric `{monitor_metric}` is unavailable in epoch metrics. "
                    f"Available metrics: {available}."
                )
            monitored_value = monitored_metrics[monitor_metric]
            is_best = self._is_better(
                candidate=monitored_value,
                current_best=best_metric,
                mode=monitor_mode,
            )

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

            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_json("history.json", history)
                self.checkpoint_manager.save_json(
                    "summary.json",
                    {
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "latest_checkpoint": latest_checkpoint,
                        "best_checkpoint": best_checkpoint,
                    },
                )

            self._print_epoch_summary(epoch, train_metrics, val_metrics)
            emit_progress(
                stage="epoch_complete",
                status="running",
                epoch_current=epoch,
                progress_units=completed_units_before_epoch(epoch) + train_batches + (val_batches if val_metrics is not None else 0),
                latest_epoch_record=epoch_record,
                force_write=True,
            )

            if early_stopping_patience > 0 and bad_epoch_count >= early_stopping_patience:
                emit_progress(
                    stage="completed",
                    status="completed",
                    epoch_current=epoch,
                    progress_units=planned_total_units,
                    latest_epoch_record=epoch_record,
                    stopped_early=True,
                    force_write=True,
                )
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch={best_epoch}, best_{monitor_metric}={best_metric:.6f}."
                )
                break

        emit_progress(
            stage="completed",
            status="completed",
            epoch_current=len(history),
            progress_units=planned_total_units,
            latest_epoch_record=history[-1] if history else None,
            stopped_early=early_stopping_patience > 0 and len(history) < max_epochs,
            force_write=True,
        )

        return FitResult(
            history=history,
            best_epoch=best_epoch,
            best_metric=best_metric,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=best_checkpoint,
        )

    @property
    def _nn_module(self) -> nn.Module:
        """Return `model` as `nn.Module` and fail fast on protocol-only objects."""

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
        evaluator: EpochEvaluatorProtocol | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, float]:
        """Run one epoch and return sample-weighted mean metrics.

        Important behavior:
        - Uses gradient accumulation when `training=True`.
        - Applies optional gradient clipping before optimizer steps.
        - Aggregates each metric by batch size, not by batch count.
        """

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
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self._amp_enabled,
                ):
                    output = module(batch)
                    loss_output = self.loss_fn(batch, output)
                    raw_loss = loss_output.loss
                if evaluator is not None:
                    evaluator.update(batch, output)

                if training:
                    scaled_loss = raw_loss / self.gradient_accumulation_steps
                    if self._grad_scaler is not None:
                        self._grad_scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    # Update params only at accumulation boundaries (or final step).
                    is_update_step = (
                        step % self.gradient_accumulation_steps == 0 or step == num_batches
                    )
                    if is_update_step:
                        if self._grad_scaler is not None:
                            self._grad_scaler.unscale_(self.optimizer)
                        if self.gradient_clip_norm is not None:
                            clip_grad_norm_(module.parameters(), self.gradient_clip_norm)
                        if self._grad_scaler is not None:
                            self._grad_scaler.step(self.optimizer)
                            self._grad_scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

            batch_metrics = dict(loss_output.metrics)
            batch_metrics.setdefault("total_loss", float(raw_loss.detach().item()))
            for name, value in batch_metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * batch_size

            if progress_callback is not None:
                running_metrics = {name: total / sample_count for name, total in metric_sums.items()}
                progress_callback(
                    {
                        "step_current": step,
                        "step_total": num_batches,
                        "batch_metrics": batch_metrics,
                        "running_metrics": running_metrics,
                    }
                )

            if training and (step % self.log_every_n_steps == 0 or step == num_batches):
                print(
                    f"[train] epoch={epoch} step={step}/{num_batches} "
                    f"loss={float(raw_loss.detach().item()):.6f}"
                )

        if sample_count == 0:
            raise ValueError(f"The `{split}` loader produced zero batches.")

        metrics = {name: total / sample_count for name, total in metric_sums.items()}
        if evaluator is not None:
            for name, value in evaluator.compute().items():
                if isinstance(value, bool):
                    metrics[name] = float(value)
                elif isinstance(value, (int, float)):
                    metrics[name] = float(value)
        return metrics

    @staticmethod
    def _is_better(
        *,
        candidate: float,
        current_best: float | None,
        mode: str,
    ) -> bool:
        """Return whether `candidate` improves over `current_best` for the given mode."""

        if current_best is None:
            return True
        if mode == "min":
            return candidate < current_best
        return candidate > current_best

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        """Print compact train/val loss summary for one epoch."""

        train_loss = train_metrics.get("total_loss", float("nan"))
        message = f"[epoch {epoch}] train_total_loss={train_loss:.6f}"
        if val_metrics is not None:
            val_loss = val_metrics.get("total_loss", float("nan"))
            message += f" val_total_loss={val_loss:.6f}"
        print(message)

    def _current_learning_rate(self) -> float | None:
        if not self.optimizer.param_groups:
            return None
        value = self.optimizer.param_groups[0].get("lr")
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def _write_progress(
        self,
        *,
        stage: str,
        status: str,
        epoch_current: int,
        epoch_total: int,
        epochs_completed: int,
        split: str | None,
        split_step_current: int | None,
        split_step_total: int | None,
        learning_rate: float | None,
        elapsed_seconds: float | None,
        eta_seconds: float | None,
        overall_progress_ratio: float | None,
        latest_train_metrics: dict[str, float] | None,
        latest_val_metrics: dict[str, float] | None,
        current_split_metrics: dict[str, float] | None,
        current_batch_metrics: dict[str, float] | None,
        best_epoch: int | None,
        best_metric: float | None,
        monitor_metric: str,
        monitor_mode: str,
        latest_epoch_record: dict[str, Any] | None,
        train_batches_per_epoch: int,
        val_batches_per_epoch: int,
        stopped_early: bool,
        error: str | None,
        force: bool = False,
    ) -> None:
        if self.checkpoint_manager is None:
            return
        now = time.perf_counter()
        if not force and (now - self._last_progress_write_time) < self.progress_write_interval_seconds:
            return
        self._last_progress_write_time = now

        split_progress_ratio = None
        if split_step_current is not None and split_step_total and split_step_total > 0:
            split_progress_ratio = min(1.0, max(0.0, split_step_current / split_step_total))

        payload = {
            "status": status,
            "stage": stage,
            "epoch_current": int(epoch_current),
            "epoch_total": int(epoch_total),
            "epochs_completed": int(epochs_completed),
            "split": split,
            "split_step_current": int(split_step_current) if split_step_current is not None else None,
            "split_step_total": int(split_step_total) if split_step_total is not None else None,
            "split_progress_ratio": split_progress_ratio,
            "overall_progress_ratio": overall_progress_ratio,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_metric": best_metric,
            "monitor_metric": monitor_metric,
            "monitor_mode": monitor_mode,
            "latest_train_metrics": self._normalize_metrics(latest_train_metrics),
            "latest_val_metrics": self._normalize_metrics(latest_val_metrics),
            "current_split_metrics": self._normalize_metrics(current_split_metrics),
            "current_batch_metrics": self._normalize_metrics(current_batch_metrics),
            "latest_epoch_record": latest_epoch_record,
            "train_batches_per_epoch": int(train_batches_per_epoch),
            "val_batches_per_epoch": int(val_batches_per_epoch),
            "stopped_early": bool(stopped_early),
            "updated_at": time.time(),
            "error": error,
        }
        self.checkpoint_manager.save_json("progress.json", payload)

    @staticmethod
    def _normalize_metrics(metrics: dict[str, float] | None) -> dict[str, float | None] | None:
        if metrics is None:
            return None
        normalized: dict[str, float | None] = {}
        for name, value in metrics.items():
            if isinstance(value, bool):
                normalized[name] = float(value)
                continue
            if isinstance(value, (int, float)):
                numeric = float(value)
                normalized[name] = numeric if math.isfinite(numeric) else None
        return normalized


__all__ = ["EpochResult", "FitResult", "Trainer"]
