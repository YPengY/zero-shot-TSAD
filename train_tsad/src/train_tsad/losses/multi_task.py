from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import LossConfig
from ..interfaces import Batch, LossOutput, ModelOutput
from .anomaly import PatchAnomalyLoss
from .reconstruction import MaskedReconstructionLoss


@dataclass(slots=True)
class MultiTaskLossComponents:
    """Detached scalar breakdown of the combined training objective."""

    anomaly_loss: float
    reconstruction_loss: float
    total_loss: float


class TimeRCDMultiTaskLoss(nn.Module):
    """Combine patch anomaly loss and reconstruction loss with fixed weights."""

    def __init__(
        self,
        *,
        anomaly_loss: PatchAnomalyLoss | None = None,
        reconstruction_loss: MaskedReconstructionLoss | None = None,
        anomaly_weight: float = 1.0,
        reconstruction_weight: float = 0.2,
    ) -> None:
        """Combine anomaly and reconstruction objectives with fixed weights."""

        super().__init__()

        if anomaly_weight <= 0:
            raise ValueError("`anomaly_weight` must be positive.")
        if reconstruction_weight < 0:
            raise ValueError("`reconstruction_weight` cannot be negative.")

        self.anomaly_loss = anomaly_loss or PatchAnomalyLoss()
        self.reconstruction_loss = reconstruction_loss or MaskedReconstructionLoss()
        self.anomaly_weight = anomaly_weight
        self.reconstruction_weight = reconstruction_weight

    @classmethod
    def from_config(cls, config: LossConfig) -> TimeRCDMultiTaskLoss:
        """Factory that builds loss modules from experiment config."""

        return cls(
            anomaly_loss=PatchAnomalyLoss(
                pos_weight=config.anomaly_pos_weight,
                label_smoothing=config.label_smoothing,
            ),
            reconstruction_loss=MaskedReconstructionLoss(use_mask_only=True),
            anomaly_weight=config.anomaly_loss_weight,
            reconstruction_weight=config.reconstruction_loss_weight,
        )

    def forward(self, batch: Batch, output: ModelOutput) -> LossOutput:
        """Compute weighted multi-task loss and merged metric dictionary.

        Behavior:
        - Anomaly term is always computed.
        - Reconstruction term is computed only when enabled and tensors exist.
        """

        anomaly_output = self.anomaly_loss(batch, output)
        total_loss = anomaly_output.loss * self.anomaly_weight

        reconstruction_scalar = 0.0
        metrics = dict(anomaly_output.metrics)

        if self.reconstruction_weight > 0.0:
            has_reconstruction = (
                batch.reconstruction_targets is not None and output.reconstruction is not None
            )
            if has_reconstruction:
                reconstruction_output = self.reconstruction_loss(batch, output)
                total_loss = total_loss + reconstruction_output.loss * self.reconstruction_weight
                reconstruction_scalar = float(reconstruction_output.loss.detach().item())
                metrics.update(reconstruction_output.metrics)
            else:
                metrics["reconstruction_loss"] = 0.0
                metrics["reconstruction_mask_fraction"] = 0.0
                metrics["reconstruction_used_mask"] = 0.0

        components = MultiTaskLossComponents(
            anomaly_loss=float(anomaly_output.loss.detach().item()),
            reconstruction_loss=reconstruction_scalar,
            total_loss=float(total_loss.detach().item()),
        )
        metrics.update(
            {
                "anomaly_weight": float(self.anomaly_weight),
                "reconstruction_weight": float(self.reconstruction_weight),
                "total_loss": components.total_loss,
            }
        )

        return LossOutput(loss=total_loss, metrics=metrics)


__all__ = ["MultiTaskLossComponents", "TimeRCDMultiTaskLoss"]
