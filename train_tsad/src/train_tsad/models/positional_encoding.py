from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _build_sinusoidal_table(length: int, d_model: int) -> Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )

    table = torch.zeros(length, d_model, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table


class GridPositionalEncoding(nn.Module):
    """Add factorized patch and feature positional signals to token sequences."""

    def __init__(
        self,
        *,
        d_model: int,
        max_patches: int,
        max_features: int,
        dropout: float = 0.1,
        learned: bool = True,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        if max_patches <= 0:
            raise ValueError("`max_patches` must be positive.")
        if max_features <= 0:
            raise ValueError("`max_features` must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("`dropout` must be in [0, 1).")

        self.d_model = d_model
        self.max_patches = max_patches
        self.max_features = max_features
        self.learned = learned
        self.dropout = nn.Dropout(dropout)

        if learned:
            self.patch_embedding = nn.Embedding(max_patches, d_model)
            self.feature_embedding = nn.Embedding(max_features, d_model)
        else:
            self.register_buffer(
                "patch_table",
                _build_sinusoidal_table(max_patches, d_model),
                persistent=False,
            )
            self.register_buffer(
                "feature_table",
                _build_sinusoidal_table(max_features, d_model),
                persistent=False,
            )

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must have shape [B, L, d_model], got ndim={tokens.ndim}.")
        if tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"`tokens.shape[-1]` must match d_model={self.d_model}, got {tokens.shape[-1]}."
            )
        if num_patches <= 0 or num_features <= 0:
            raise ValueError("`num_patches` and `num_features` must be positive.")
        if num_patches > self.max_patches:
            raise ValueError(
                f"`num_patches` exceeds configured max_patches={self.max_patches}: {num_patches}."
            )
        if num_features > self.max_features:
            raise ValueError(
                f"`num_features` exceeds configured max_features={self.max_features}: {num_features}."
            )
        if tokens.shape[1] != num_patches * num_features:
            raise ValueError(
                "`tokens.shape[1]` must equal `num_patches * num_features`. "
                f"Got {tokens.shape[1]} vs {num_patches * num_features}."
            )

        patch_ids = torch.arange(num_patches, device=tokens.device)
        feature_ids = torch.arange(num_features, device=tokens.device)

        if self.learned:
            patch_pos = self.patch_embedding(patch_ids)
            feature_pos = self.feature_embedding(feature_ids)
        else:
            patch_pos = self.patch_table[:num_patches].to(tokens.device)
            feature_pos = self.feature_table[:num_features].to(tokens.device)

        position_grid = patch_pos[:, None, :] + feature_pos[None, :, :]
        position_tokens = position_grid.reshape(1, num_patches * num_features, self.d_model)
        return self.dropout(tokens + position_tokens)


__all__ = ["GridPositionalEncoding"]
