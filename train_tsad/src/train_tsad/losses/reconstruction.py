from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..interfaces import Batch, LossOutput, ModelOutput


class MaskedReconstructionLoss(nn.Module):
    """Mean squared reconstruction loss for masked or full context recovery."""

    def __init__(self, *, use_mask_only: bool = True) -> None:
        """Configure reconstruction objective over full or masked positions."""

        super().__init__()
        self.use_mask_only = use_mask_only

    def _masked_mse(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute MSE only on selected masked positions."""

        masked_prediction = prediction[mask]
        masked_target = target[mask]
        if masked_prediction.numel() == 0:
            return prediction.sum() * 0.0
        return F.mse_loss(masked_prediction, masked_target)

    def _masked_full_mse(self, prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
        """Compute MSE only on valid non-padding positions."""

        valid_prediction = prediction[valid_mask]
        valid_target = target[valid_mask]
        if valid_prediction.numel() == 0:
            return prediction.sum() * 0.0
        return F.mse_loss(valid_prediction, valid_target)

    def forward(self, batch: Batch, output: ModelOutput) -> LossOutput:
        """Compute reconstruction loss and reconstruction-path diagnostics."""

        if batch.reconstruction_targets is None:
            raise ValueError("`batch.reconstruction_targets` is required for reconstruction loss.")
        if output.reconstruction is None:
            raise ValueError("`output.reconstruction` is required for reconstruction loss.")

        prediction = output.reconstruction
        target = batch.reconstruction_targets
        if prediction.shape != target.shape:
            raise ValueError(
                "`output.reconstruction` must match `batch.reconstruction_targets` shape. "
                f"Got {tuple(prediction.shape)} vs {tuple(target.shape)}."
            )

        valid_mask = batch.point_valid_mask
        if valid_mask is not None:
            if valid_mask.shape != prediction.shape:
                raise ValueError(
                    "`batch.point_valid_mask` must match reconstruction shape. "
                    f"Got {tuple(valid_mask.shape)} vs {tuple(prediction.shape)}."
                )
            valid_mask = valid_mask.to(device=prediction.device, dtype=torch.bool)

        used_mask = False
        if batch.mask_indices is not None and self.use_mask_only:
            mask = batch.mask_indices.bool()
            if mask.shape != prediction.shape:
                raise ValueError(
                    "`batch.mask_indices` must match reconstruction shape. "
                    f"Got {tuple(mask.shape)} vs {tuple(prediction.shape)}."
                )
            if valid_mask is not None:
                mask = torch.logical_and(mask, valid_mask)
            loss = self._masked_mse(prediction, target, mask)
            mask_fraction = mask.float().mean().item()
            used_mask = True
        else:
            if valid_mask is not None:
                loss = self._masked_full_mse(prediction, target, valid_mask)
                mask_fraction = valid_mask.float().mean().item()
            else:
                loss = F.mse_loss(prediction, target)
                mask_fraction = 1.0

        return LossOutput(
            loss=loss,
            metrics={
                "reconstruction_loss": float(loss.detach().item()),
                "reconstruction_mask_fraction": mask_fraction,
                "reconstruction_used_mask": 1.0 if used_mask else 0.0,
            },
        )


__all__ = ["MaskedReconstructionLoss"]
