"""Observation-space anomaly objective for the optional auxiliary head.

This loss is only meaningful when the model predicts point-level logits and
the batch still carries point masks after collation. It complements the main
patch head by directly supervising the reconstructed observation grid.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..interfaces import Batch, LossOutput, ModelOutput


class PointAnomalyLoss(nn.Module):
    """Binary point-feature loss for the optional observation-space head."""

    def __init__(
        self,
        *,
        pos_weight: float | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        """Configure BCE-with-logits point anomaly objective."""

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
        """Apply label smoothing to binary targets."""

        if self.label_smoothing == 0.0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def forward(self, batch: Batch, output: ModelOutput) -> LossOutput:
        """Compute point-level anomaly loss and diagnostics on valid points only."""

        if batch.point_masks is None:
            raise ValueError("`batch.point_masks` is required for point anomaly loss.")
        if output.point_logits is None:
            raise ValueError("`output.point_logits` is required for point anomaly loss.")

        logits = output.point_logits
        targets = batch.point_masks
        if logits.shape != targets.shape:
            raise ValueError(
                "`output.point_logits` must match `batch.point_masks` shape. "
                f"Got {tuple(logits.shape)} vs {tuple(targets.shape)}."
            )

        valid_mask = batch.point_valid_mask
        if valid_mask is not None:
            if valid_mask.shape != logits.shape:
                raise ValueError(
                    "`batch.point_valid_mask` must match point anomaly logits shape. "
                    f"Got {tuple(valid_mask.shape)} vs {tuple(logits.shape)}."
                )
            valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
            if not torch.any(valid_mask):
                raise ValueError(
                    "`batch.point_valid_mask` does not contain any valid point-feature units."
                )
            logits = logits[valid_mask]
            targets = targets[valid_mask]

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
                "point_anomaly_loss": float(loss.detach().item()),
                "point_accuracy": accuracy,
                "point_target_positive_rate": positive_rate,
                "point_predicted_positive_rate": predicted_positive_rate,
            },
        )


__all__ = ["PointAnomalyLoss"]
