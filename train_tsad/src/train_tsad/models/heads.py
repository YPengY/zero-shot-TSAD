from __future__ import annotations

from torch import Tensor, nn


def _reshape_token_grid(tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"`tokens` must have shape [B, L, H], got ndim={tokens.ndim}.")
    if tokens.shape[1] != num_patches * num_features:
        raise ValueError(
            "`tokens.shape[1]` must equal `num_patches * num_features`. "
            f"Got {tokens.shape[1]} vs {num_patches * num_features}."
        )
    return tokens.reshape(tokens.shape[0], num_patches, num_features, tokens.shape[-1])


class AnomalyHead(nn.Module):
    """Predict per-patch anomaly logits for each feature channel."""

    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        token_grid = _reshape_token_grid(self.norm(tokens), num_patches=num_patches, num_features=num_features)
        return self.projection(token_grid).squeeze(-1)


class ReconstructionHead(nn.Module):
    """Reconstruct point-level contexts from encoded patch tokens."""

    def __init__(self, *, d_model: int, patch_size: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, patch_size)

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        token_grid = _reshape_token_grid(self.norm(tokens), num_patches=num_patches, num_features=num_features)
        patch_values = self.projection(token_grid)
        sequence = patch_values.permute(0, 2, 1, 3).reshape(
            tokens.shape[0],
            num_features,
            num_patches * self.patch_size,
        )
        return sequence.transpose(1, 2).contiguous()


__all__ = ["AnomalyHead", "ReconstructionHead"]
