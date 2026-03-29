"""Prediction heads used by the TimeRCD encoder stack.

The heads keep task-specific decoding separate from the shared encoder so
training can mix patch-level anomaly detection, point-level anomaly decoding,
and reconstruction without entangling those choices in the encoder itself.
"""

from __future__ import annotations

from torch import Tensor, nn


def _reshape_token_grid(tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
    """Reshape flattened token sequence back to `[B, N_patches, D, H]`."""

    if tokens.ndim != 3:
        raise ValueError(f"`tokens` must have shape [B, L, H], got ndim={tokens.ndim}.")
    if tokens.shape[1] != num_patches * num_features:
        raise ValueError(
            "`tokens.shape[1]` must equal `num_patches * num_features`. "
            f"Got {tokens.shape[1]} vs {num_patches * num_features}."
        )
    return tokens.reshape(tokens.shape[0], num_patches, num_features, tokens.shape[-1])


class AnomalyHead(nn.Module):
    """Predict patch-level anomaly logits for each feature channel.

    Each token corresponds to one `(patch, feature)` pair, so the head only
    needs to reduce the embedding dimension and reshape back to `[B, N_patches, D]`.
    """

    def __init__(self, *, d_model: int) -> None:
        """Initialize per-token binary anomaly predictor."""

        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        """Map encoded tokens to patch-level logits `[B, N_patches, D]`."""

        token_grid = _reshape_token_grid(self.norm(tokens), num_patches=num_patches, num_features=num_features)
        return self.projection(token_grid).squeeze(-1)


class ReconstructionHead(nn.Module):
    """Decode patch tokens back to observation-space values.

    The reconstruction target lives at the point level, so each token expands
    back into `patch_size` values for its feature channel.
    """

    def __init__(self, *, d_model: int, patch_size: int) -> None:
        """Initialize patch-to-point reconstruction projector."""

        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, patch_size)

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        """Decode token grid back to point sequence `[B, W, D]`."""

        token_grid = _reshape_token_grid(self.norm(tokens), num_patches=num_patches, num_features=num_features)
        patch_values = self.projection(token_grid)
        sequence = patch_values.permute(0, 2, 1, 3).reshape(
            tokens.shape[0],
            num_features,
            num_patches * self.patch_size,
        )
        return sequence.transpose(1, 2).contiguous()


class SharedOutputProjection(nn.Module):
    """Shared projection applied before task-specific heads.

    This matches the paper-style setup where multiple heads consume a common
    projected representation instead of attaching directly to encoder outputs.
    """

    def __init__(self, *, d_model: int) -> None:
        """Initialize the shared projection in token embedding space."""

        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        """Project encoder outputs while preserving `[B, L, H]` shape."""

        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must have shape [B, L, H], got ndim={tokens.ndim}.")
        return self.projection(self.norm(tokens))


class ProjectedAnomalyHead(nn.Module):
    """Task-specific anomaly projection used after the shared output layer."""

    def __init__(self, *, d_model: int) -> None:
        """Initialize per-token anomaly projection without extra normalization."""

        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        self.projection = nn.Linear(d_model, 1)

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        """Map shared token embeddings to patch-level logits `[B, N_patches, D]`."""

        token_grid = _reshape_token_grid(tokens, num_patches=num_patches, num_features=num_features)
        return self.projection(token_grid).squeeze(-1)


class ProjectedReconstructionHead(nn.Module):
    """Task-specific reconstruction projection used after the shared output layer."""

    def __init__(self, *, d_model: int, patch_size: int) -> None:
        """Initialize point reconstruction projection without extra normalization."""

        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        self.patch_size = patch_size
        self.projection = nn.Linear(d_model, patch_size)

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        """Decode shared token embeddings back to point sequence `[B, W, D]`."""

        token_grid = _reshape_token_grid(tokens, num_patches=num_patches, num_features=num_features)
        patch_values = self.projection(token_grid)
        sequence = patch_values.permute(0, 2, 1, 3).reshape(
            tokens.shape[0],
            num_features,
            num_patches * self.patch_size,
        )
        return sequence.transpose(1, 2).contiguous()


class ObservationSpaceAnomalyHead(nn.Module):
    """Decode tokens into point-level anomaly logits.

    This head is useful when supervision is naturally defined per timestep and
    feature, while the transformer still operates over patch-feature tokens.
    """

    def __init__(self, *, d_model: int, patch_size: int) -> None:
        """Initialize point-level anomaly projection with token normalization."""

        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, patch_size)

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        """Decode token grid to point-level anomaly logits `[B, W, D]`."""

        token_grid = _reshape_token_grid(self.norm(tokens), num_patches=num_patches, num_features=num_features)
        patch_logits = self.projection(token_grid)
        sequence = patch_logits.permute(0, 2, 1, 3).reshape(
            tokens.shape[0],
            num_features,
            num_patches * self.patch_size,
        )
        return sequence.transpose(1, 2).contiguous()


class ProjectedObservationSpaceAnomalyHead(nn.Module):
    """Point-level anomaly head used after the shared output projection."""

    def __init__(self, *, d_model: int, patch_size: int) -> None:
        """Initialize point-level anomaly projection without extra normalization."""

        super().__init__()
        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        self.patch_size = patch_size
        self.projection = nn.Linear(d_model, patch_size)

    def forward(self, tokens: Tensor, *, num_patches: int, num_features: int) -> Tensor:
        """Decode shared token embeddings to point-level anomaly logits `[B, W, D]`."""

        token_grid = _reshape_token_grid(tokens, num_patches=num_patches, num_features=num_features)
        patch_logits = self.projection(token_grid)
        sequence = patch_logits.permute(0, 2, 1, 3).reshape(
            tokens.shape[0],
            num_features,
            num_patches * self.patch_size,
        )
        return sequence.transpose(1, 2).contiguous()


__all__ = [
    "AnomalyHead",
    "ObservationSpaceAnomalyHead",
    "ProjectedAnomalyHead",
    "ProjectedObservationSpaceAnomalyHead",
    "ProjectedReconstructionHead",
    "ReconstructionHead",
    "SharedOutputProjection",
]
