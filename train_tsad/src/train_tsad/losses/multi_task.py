"""Composition of patch, point, and reconstruction training objectives.

This module is the only place where the three supervision paths are combined
into one scalar loss. Keeping the weighting logic centralized makes it easier
to reason about which heads are active for a given run.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from ..config import LossConfig
from ..interfaces import Batch, LossOutput, ModelOutput
from .anomaly import PatchAnomalyLoss, PatchAsymmetricLoss
from .point_anomaly import PointAnomalyLoss
from .reconstruction import MaskedReconstructionLoss


@dataclass(slots=True)
class MultiTaskLossComponents:
    """Detached scalar breakdown of the combined training objective."""

    anomaly_loss: float
    point_anomaly_loss: float
    reconstruction_loss: float
    total_loss: float


def _resolve_pos_weight_config(
    value: float | str | None,
    *,
    field_name: str,
) -> float | None:
    """Require `auto` weights to be resolved before loss module construction."""

    if isinstance(value, str):
        raise ValueError(
            f"`loss.{field_name}` must be resolved before constructing the loss module. "
            "Run the training entrypoint auto-resolution step or set a float/null explicitly."
        )
    return value


class TimeRCDMultiTaskLoss(nn.Module):
    """Combine all enabled supervision heads into one scalar objective.

    The patch anomaly head is always active. Point-level and reconstruction
    terms are optional and are only evaluated when both their weights and the
    required tensors are present.
    """

    def __init__(
        self,
        *,
        anomaly_loss: PatchAnomalyLoss | None = None,
        point_anomaly_loss: PointAnomalyLoss | None = None,
        reconstruction_loss: MaskedReconstructionLoss | None = None,
        anomaly_weight: float = 1.0,
        point_anomaly_weight: float = 0.0,
        reconstruction_weight: float = 0.2,
    ) -> None:
        """Combine anomaly and reconstruction objectives with fixed weights."""

        super().__init__()

        if anomaly_weight <= 0:
            raise ValueError("`anomaly_weight` must be positive.")
        if point_anomaly_weight < 0:
            raise ValueError("`point_anomaly_weight` cannot be negative.")
        if reconstruction_weight < 0:
            raise ValueError("`reconstruction_weight` cannot be negative.")

        self.anomaly_loss = anomaly_loss or PatchAnomalyLoss()
        self.point_anomaly_loss = point_anomaly_loss
        self.reconstruction_loss = reconstruction_loss or MaskedReconstructionLoss()
        self.anomaly_weight = anomaly_weight
        self.point_anomaly_weight = point_anomaly_weight
        self.reconstruction_weight = reconstruction_weight

    @classmethod
    def from_config(cls, config: LossConfig) -> TimeRCDMultiTaskLoss:
        """Factory that builds loss modules from experiment config."""

        point_anomaly_loss = None
        if config.point_anomaly_loss_weight > 0.0:
            point_anomaly_loss = PointAnomalyLoss(
                pos_weight=_resolve_pos_weight_config(
                    config.point_anomaly_pos_weight,
                    field_name="point_anomaly_pos_weight",
                ),
                label_smoothing=config.label_smoothing,
            )

        if config.anomaly_loss_type == "asl":
            anomaly_loss = PatchAsymmetricLoss(
                gamma_neg=config.anomaly_asl_gamma_neg,
                gamma_pos=config.anomaly_asl_gamma_pos,
                clip=config.anomaly_asl_clip,
                label_smoothing=config.label_smoothing,
            )
        else:
            anomaly_loss = PatchAnomalyLoss(
                pos_weight=_resolve_pos_weight_config(
                    config.anomaly_pos_weight,
                    field_name="anomaly_pos_weight",
                ),
                label_smoothing=config.label_smoothing,
            )

        return cls(
            anomaly_loss=anomaly_loss,
            point_anomaly_loss=point_anomaly_loss,
            reconstruction_loss=MaskedReconstructionLoss(use_mask_only=True),
            anomaly_weight=config.anomaly_loss_weight,
            point_anomaly_weight=config.point_anomaly_loss_weight,
            reconstruction_weight=config.reconstruction_loss_weight,
        )

    def forward(self, batch: Batch, output: ModelOutput) -> LossOutput:
        """Compute the weighted objective and expose detached diagnostics."""

        anomaly_output = self.anomaly_loss(batch, output)
        total_loss = anomaly_output.loss * self.anomaly_weight

        point_anomaly_scalar = 0.0
        reconstruction_scalar = 0.0
        metrics = dict(anomaly_output.metrics)

        if self.point_anomaly_weight > 0.0:
            if self.point_anomaly_loss is None:
                raise ValueError(
                    "`point_anomaly_weight > 0` requires `point_anomaly_loss` to be configured."
                )
            has_point_anomaly = batch.point_masks is not None and output.point_logits is not None
            if not has_point_anomaly:
                raise ValueError(
                    "`point_anomaly_weight > 0` requires `batch.point_masks` and `output.point_logits`."
                )
            point_anomaly_output = self.point_anomaly_loss(batch, output)
            total_loss = total_loss + point_anomaly_output.loss * self.point_anomaly_weight
            point_anomaly_scalar = float(point_anomaly_output.loss.detach().item())
            metrics.update(point_anomaly_output.metrics)

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
            point_anomaly_loss=point_anomaly_scalar,
            reconstruction_loss=reconstruction_scalar,
            total_loss=float(total_loss.detach().item()),
        )
        metrics.update(
            {
                "anomaly_weight": float(self.anomaly_weight),
                "point_anomaly_weight": float(self.point_anomaly_weight),
                "reconstruction_weight": float(self.reconstruction_weight),
                "total_loss": components.total_loss,
            }
        )

        return LossOutput(loss=total_loss, metrics=metrics)


__all__ = ["MultiTaskLossComponents", "TimeRCDMultiTaskLoss"]
