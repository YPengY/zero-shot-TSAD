from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class PatchEmbeddingOutput:
    """Patchified inputs and token embeddings for the encoder stack.

    Shape conventions:
    - `patches`: [B, N_patches, D, patch_size]
    - `patch_embeddings`: [B, N_patches, D, d_proj]
    - `tokens`: [B, N_patches * D, d_model]
    """

    patches: Tensor
    patch_embeddings: Tensor
    tokens: Tensor
    num_patches: int
    num_features: int


class PatchEmbedding(nn.Module):
    """Convert `[B, W, D]` context windows into patch tokens.

    The module follows the paper-aligned tokenization strategy used across the
    training pipeline:
    1. Split each feature channel along time into fixed-size patches.
    2. Project each per-feature patch into a latent patch embedding.
    3. Flatten the `(patch_index, feature_index)` grid into a token sequence for
       the Transformer encoder.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        d_model: int,
        d_proj: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        if d_proj <= 0:
            raise ValueError("`d_proj` must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("`dropout` must be in [0, 1).")

        self.patch_size = patch_size
        self.d_model = d_model
        self.d_proj = d_proj

        self.patch_projection = nn.Linear(patch_size, d_proj)
        self.patch_norm = nn.LayerNorm(d_proj)
        self.output_projection = nn.Linear(d_proj, d_model) if d_proj != d_model else nn.Identity()
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def patchify(self, inputs: Tensor) -> Tensor:
        """Split point-level inputs into non-overlapping time patches."""

        if inputs.ndim != 3:
            raise ValueError(f"`inputs` must have shape [B, W, D], got ndim={inputs.ndim}.")

        batch_size, context_size, num_features = inputs.shape
        if context_size % self.patch_size != 0:
            raise ValueError(
                "`inputs.shape[1]` must be divisible by `patch_size`. "
                f"Got context_size={context_size}, patch_size={self.patch_size}."
            )

        num_patches = context_size // self.patch_size
        # Rearrange to keep labels aligned with the `[N_patches, D]` training view.
        patches = inputs.transpose(1, 2).reshape(
            batch_size,
            num_features,
            num_patches,
            self.patch_size,
        )
        return patches.transpose(1, 2).contiguous()

    def forward(self, inputs: Tensor) -> PatchEmbeddingOutput:
        patches = self.patchify(inputs)
        batch_size, num_patches, num_features, _ = patches.shape

        patch_embeddings = self.patch_projection(patches)
        patch_embeddings = self.patch_norm(patch_embeddings)

        tokens = self.output_projection(patch_embeddings)
        tokens = self.output_norm(tokens)
        tokens = self.dropout(tokens)
        tokens = tokens.reshape(batch_size, num_patches * num_features, self.d_model)

        return PatchEmbeddingOutput(
            patches=patches,
            patch_embeddings=patch_embeddings,
            tokens=tokens,
            num_patches=num_patches,
            num_features=num_features,
        )


__all__ = ["PatchEmbedding", "PatchEmbeddingOutput"]
