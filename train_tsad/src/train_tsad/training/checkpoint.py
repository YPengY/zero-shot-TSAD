from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(slots=True)
class CheckpointManager:
    """Persist latest/best checkpoints and auxiliary training artifacts."""

    output_dir: Path

    def __post_init__(self) -> None:
        """Ensure checkpoint output directory exists."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        *,
        filename: str,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        metrics: Mapping[str, Any],
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize one checkpoint bundle to disk."""

        checkpoint_path = self.output_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "config": config,
        }
        torch.save(state, checkpoint_path)
        return checkpoint_path

    def save_latest(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        metrics: Mapping[str, Any],
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Write/update the rolling latest checkpoint."""

        return self.save_checkpoint(
            filename="latest.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            config=config,
        )

    def save_best(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        metrics: Mapping[str, Any],
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Write/update the best-performing checkpoint."""

        return self.save_checkpoint(
            filename="best.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            config=config,
        )

    def save_json(self, filename: str, payload: Any) -> Path:
        """Persist auxiliary JSON artifact (history, summary, config snapshot)."""

        path = self.output_dir / filename
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        temp_path.write_text(serialized, encoding="utf-8")

        for attempt in range(6):
            try:
                temp_path.replace(path)
                return path
            except PermissionError:
                if attempt == 5:
                    break
                time.sleep(0.05 * (attempt + 1))

        # Windows may deny atomic replace while another process briefly reads the file.
        # Fall back to direct overwrite so progress reporting does not abort training.
        path.write_text(serialized, encoding="utf-8")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return path


__all__ = ["CheckpointManager"]
