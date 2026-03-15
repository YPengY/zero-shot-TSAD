from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(slots=True)
class MaskingTargets:
    """Outputs produced by a masking transform before model forward."""

    reconstruction_targets: Tensor
    mask_indices: Tensor
    metadata: dict[str, float] = field(default_factory=dict)


class RandomPatchMaskingTransform:
    """Randomly mask a fixed fraction of patch-feature tokens per sample.

    The transform operates on point-level inputs `[B, W, D]`, samples masked
    patch positions on the implicit `[N_patches, D]` grid, and then expands the
    mask back to point level so the model can zero the corresponding time spans
    before patch embedding.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        mask_ratio: float = 0.2,
        seed: int | None = None,
        min_masked_tokens: int = 1,
    ) -> None:
        """Configure random masking over the patch-feature token grid."""

        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("`mask_ratio` must be in [0, 1].")
        if min_masked_tokens < 0:
            raise ValueError("`min_masked_tokens` cannot be negative.")

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.min_masked_tokens = min_masked_tokens
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, inputs: Tensor) -> MaskingTargets:
        """Generate reconstruction targets and point-level mask indices.

        Args:
            inputs: Tensor shaped `[B, W, D]`.

        Returns:
            MaskingTargets containing:
            - `reconstruction_targets`: original unmasked inputs
            - `mask_indices`: bool mask `[B, W, D]` used to zero model inputs
            - `metadata`: configured vs realized mask ratios
        """

        if inputs.ndim != 3:
            raise ValueError(f"`inputs` must have shape [B, W, D], got ndim={inputs.ndim}.")

        batch_size, context_size, num_features = inputs.shape
        if context_size % self.patch_size != 0:
            raise ValueError(
                "`inputs.shape[1]` must be divisible by `patch_size`. "
                f"Got context_size={context_size}, patch_size={self.patch_size}."
            )

        reconstruction_targets = inputs.clone()
        num_patches = context_size // self.patch_size
        num_patch_tokens = num_patches * num_features

        if self.mask_ratio == 0.0 or num_patch_tokens == 0:
            patch_mask = torch.zeros(
                (batch_size, num_patches, num_features),
                dtype=torch.bool,
            )
        else:
            desired_masked = int(math.ceil(num_patch_tokens * self.mask_ratio))
            num_masked_tokens = min(
                num_patch_tokens,
                max(self.min_masked_tokens, desired_masked),
            )

            patch_mask = torch.zeros(
                (batch_size, num_patch_tokens),
                dtype=torch.bool,
            )
            for batch_index in range(batch_size):
                permutation = torch.randperm(num_patch_tokens, generator=self.generator)
                patch_mask[batch_index, permutation[:num_masked_tokens]] = True
            patch_mask = patch_mask.reshape(batch_size, num_patches, num_features)

        # Expand patch mask back to point level along the time axis.
        mask_indices = patch_mask.repeat_interleave(self.patch_size, dim=1)
        return MaskingTargets(
            reconstruction_targets=reconstruction_targets,
            mask_indices=mask_indices,
            metadata={
                "configured_mask_ratio": float(self.mask_ratio),
                "actual_mask_ratio": float(mask_indices.float().mean().item()),
                "actual_patch_mask_ratio": float(patch_mask.float().mean().item()),
            },
        )


__all__ = ["MaskingTargets", "RandomPatchMaskingTransform"]
