from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..interfaces import Batch, LossOutput, ModelOutput


class PatchAnomalyLoss(nn.Module):
    """Binary patch-level anomaly loss for TimeRCD training."""

    def __init__(
        self,
        *,
        pos_weight: float | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if pos_weight is not None and pos_weight <= 0:
            raise ValueError("`pos_weight` must be positive when provided.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("`label_smoothing` must be in [0, 1).")

        self.label_smoothing = label_smoothing
        if pos_weight is None:
            self.register_buffer("_pos_weight", None, persistent=False)
        else:
            self.register_buffer(
                "_pos_weight",
                torch.tensor(float(pos_weight), dtype=torch.float32),
                persistent=False,
            )

    def _smoothed_targets(self, targets: Tensor) -> Tensor:
        if self.label_smoothing == 0.0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def forward(self, batch: Batch, output: ModelOutput) -> LossOutput:
        if batch.patch_labels is None:
            raise ValueError("`batch.patch_labels` is required for anomaly loss.")

        logits = output.logits
        targets = batch.patch_labels
        if logits.shape != targets.shape:
            raise ValueError(
                "`output.logits` must match `batch.patch_labels` shape. "
                f"Got {tuple(logits.shape)} vs {tuple(targets.shape)}."
            )

        targets = targets.to(dtype=logits.dtype)
        smoothed_targets = self._smoothed_targets(targets)
        pos_weight = cast(Tensor | None, self._pos_weight)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            smoothed_targets,
            pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
        )

        with torch.no_grad():
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= 0.5
            binary_targets = targets >= 0.5
            accuracy = (predictions == binary_targets).float().mean().item()
            positive_rate = binary_targets.float().mean().item()
            predicted_positive_rate = predictions.float().mean().item()

        return LossOutput(
            loss=loss,
            metrics={
                "anomaly_loss": float(loss.detach().item()),
                "patch_accuracy": accuracy,
                "target_positive_rate": positive_rate,
                "predicted_positive_rate": predicted_positive_rate,
            },
        )


__all__ = ["PatchAnomalyLoss"]
