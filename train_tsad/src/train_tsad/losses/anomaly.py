"""Patch-level anomaly objectives for the main supervision head.

Both losses in this module operate on patch-feature logits with shape
`[B, N_patches, D]`. They differ only in how they handle sparse positives:
plain BCE uses an explicit positive-class weight, while ASL down-weights easy
negatives through asymmetric focusing.
"""

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
        """Configure BCE-with-logits anomaly objective.

        Args:
            pos_weight: Optional positive-class weight for class imbalance.
            label_smoothing: Smooth binary targets toward 0.5.
        """

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
        """Compute anomaly loss and monitoring metrics on patch grid.

        Inputs:
            `output.logits` and `batch.patch_labels` with shape `[B, N_patches, D]`.
        """

        if batch.patch_labels is None:
            raise ValueError("`batch.patch_labels` is required for anomaly loss.")

        logits = output.logits
        targets = batch.patch_labels
        if logits.shape != targets.shape:
            raise ValueError(
                "`output.logits` must match `batch.patch_labels` shape. "
                f"Got {tuple(logits.shape)} vs {tuple(targets.shape)}."
            )

        valid_mask = batch.patch_valid_mask
        if valid_mask is not None:
            if valid_mask.shape != logits.shape:
                raise ValueError(
                    "`batch.patch_valid_mask` must match anomaly logits shape. "
                    f"Got {tuple(valid_mask.shape)} vs {tuple(logits.shape)}."
                )
            valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
            if not torch.any(valid_mask):
                raise ValueError("`batch.patch_valid_mask` does not contain any valid anomaly units.")
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
            # Quick training diagnostics at threshold=0.5.
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


class PatchAsymmetricLoss(nn.Module):
    """Asymmetric patch-level anomaly loss for sparse multi-label supervision."""

    def __init__(
        self,
        *,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if gamma_neg < 0:
            raise ValueError("`gamma_neg` cannot be negative.")
        if gamma_pos < 0:
            raise ValueError("`gamma_pos` cannot be negative.")
        if not 0.0 <= clip < 1.0:
            raise ValueError("`clip` must be in [0, 1).")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("`label_smoothing` must be in [0, 1).")
        if eps <= 0:
            raise ValueError("`eps` must be positive.")

        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.label_smoothing = float(label_smoothing)
        self.eps = float(eps)

    def _smoothed_targets(self, targets: Tensor) -> Tensor:
        if self.label_smoothing == 0.0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def forward(self, batch: Batch, output: ModelOutput) -> LossOutput:
        """Compute ASL on valid patch-feature units only.

        This loss shares the same target semantics as `PatchAnomalyLoss` but
        replaces explicit class-weighting with asymmetric focusing over easy
        negatives, which is often more stable for extremely sparse anomaly
        labels.
        """

        if batch.patch_labels is None:
            raise ValueError("`batch.patch_labels` is required for anomaly loss.")

        logits = output.logits
        targets = batch.patch_labels
        if logits.shape != targets.shape:
            raise ValueError(
                "`output.logits` must match `batch.patch_labels` shape. "
                f"Got {tuple(logits.shape)} vs {tuple(targets.shape)}."
            )

        valid_mask = batch.patch_valid_mask
        if valid_mask is not None:
            if valid_mask.shape != logits.shape:
                raise ValueError(
                    "`batch.patch_valid_mask` must match anomaly logits shape. "
                    f"Got {tuple(valid_mask.shape)} vs {tuple(logits.shape)}."
                )
            valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
            if not torch.any(valid_mask):
                raise ValueError("`batch.patch_valid_mask` does not contain any valid anomaly units.")
            logits = logits[valid_mask]
            targets = targets[valid_mask]

        targets = targets.to(dtype=logits.dtype)
        smoothed_targets = self._smoothed_targets(targets)
        anti_targets = 1.0 - smoothed_targets

        probabilities = torch.sigmoid(logits)
        positive_probabilities = probabilities.clamp(min=self.eps, max=1.0 - self.eps)
        negative_probabilities = (1.0 - probabilities).clamp(min=self.eps, max=1.0)
        if self.clip > 0.0:
            negative_probabilities = (negative_probabilities + self.clip).clamp(max=1.0)
            negative_probabilities = negative_probabilities.clamp(min=self.eps)

        positive_loss = smoothed_targets * torch.log(positive_probabilities)
        negative_loss = anti_targets * torch.log(negative_probabilities)
        loss = positive_loss + negative_loss

        if self.gamma_neg > 0.0 or self.gamma_pos > 0.0:
            pt = positive_probabilities * smoothed_targets + negative_probabilities * anti_targets
            gamma = self.gamma_pos * smoothed_targets + self.gamma_neg * anti_targets
            loss = loss * torch.pow((1.0 - pt).clamp(min=0.0), gamma)

        loss = -loss.mean()

        with torch.no_grad():
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


__all__ = ["PatchAnomalyLoss", "PatchAsymmetricLoss"]
