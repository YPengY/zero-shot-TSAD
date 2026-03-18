from __future__ import annotations

import torch
from torch import Tensor, nn

from ..interfaces import Batch, ModelOutput
from .encoder import TransformerEncoder
from .heads import (
    AnomalyHead,
    ObservationSpaceAnomalyHead,
    ProjectedAnomalyHead,
    ProjectedObservationSpaceAnomalyHead,
    ProjectedReconstructionHead,
    ReconstructionHead,
    SharedOutputProjection,
)
from .patch_embed import PatchEmbedding
from .positional_encoding import GridPositionalEncoding


class TimeRCDModel(nn.Module):
    """Paper-aligned dual-head model for synthetic TimeRCD pretraining."""

    def __init__(
        self,
        *,
        patch_size: int,
        d_model: int,
        d_proj: int,
        num_layers: int,
        num_heads: int,
        max_patches: int,
        max_features: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        use_learned_positional_encoding: bool = True,
        use_shared_output_projection: bool = False,
        use_observation_space_anomaly_head: bool = False,
        anomaly_patch_aggregation: str = "or",
        use_reconstruction_head: bool = True,
    ) -> None:
        """Build the TimeRCD model stack.

        Structure:
        1. Patch embedding over `[B, W, D]`.
        2. Grid positional encoding over `(patch, feature)` tokens.
        3. Transformer encoder.
        4. Optional shared output projection `W_s`.
        5. Optional anomaly projection back to observation space.
        6. Anomaly head (+ optional reconstruction head).
        """

        super().__init__()

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            d_proj=d_proj,
            dropout=dropout,
        )
        self.position_encoding = GridPositionalEncoding(
            d_model=d_model,
            max_patches=max_patches,
            max_features=max_features,
            dropout=dropout,
            learned=use_learned_positional_encoding,
        )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
        )
        self.shared_output_projection = (
            SharedOutputProjection(d_model=d_model) if use_shared_output_projection else None
        )
        self.patch_size = patch_size
        self.use_observation_space_anomaly_head = use_observation_space_anomaly_head
        self.anomaly_patch_aggregation = anomaly_patch_aggregation

        if use_shared_output_projection and use_observation_space_anomaly_head:
            self.anomaly_head = ProjectedObservationSpaceAnomalyHead(
                d_model=d_model,
                patch_size=patch_size,
            )
        elif use_shared_output_projection:
            self.anomaly_head = ProjectedAnomalyHead(d_model=d_model)
        elif use_observation_space_anomaly_head:
            self.anomaly_head = ObservationSpaceAnomalyHead(
                d_model=d_model,
                patch_size=patch_size,
            )
        else:
            self.anomaly_head = AnomalyHead(d_model=d_model)

        if use_shared_output_projection:
            self.reconstruction_head = (
                ProjectedReconstructionHead(d_model=d_model, patch_size=patch_size)
                if use_reconstruction_head
                else None
            )
        else:
            self.reconstruction_head = (
                ReconstructionHead(d_model=d_model, patch_size=patch_size)
                if use_reconstruction_head
                else None
            )

    def _apply_input_mask(self, inputs: Tensor, mask_indices: Tensor | None) -> Tensor:
        """Zero masked positions before patch embedding."""

        if mask_indices is None:
            return inputs
        if mask_indices.shape != inputs.shape:
            raise ValueError(
                "`mask_indices` must match `inputs` shape. "
                f"Got {mask_indices.shape} vs {inputs.shape}."
            )
        masked_inputs = inputs.clone()
        masked_inputs[mask_indices.bool()] = 0.0
        return masked_inputs

    def _aggregate_point_logits_to_patch_logits(
        self,
        point_logits: Tensor,
        *,
        num_patches: int,
        num_features: int,
        point_valid_mask: Tensor | None,
    ) -> Tensor:
        """Reduce point-level anomaly logits back to patch logits for supervision."""

        if point_logits.ndim != 3:
            raise ValueError(
                "`point_logits` must have shape [B, W, D]. "
                f"Got ndim={point_logits.ndim}."
            )
        if point_logits.shape[1] != num_patches * self.patch_size:
            raise ValueError(
                "`point_logits.shape[1]` must equal `num_patches * patch_size`. "
                f"Got {point_logits.shape[1]} vs {num_patches * self.patch_size}."
            )
        if point_logits.shape[2] != num_features:
            raise ValueError(
                "`point_logits.shape[2]` must equal `num_features`. "
                f"Got {point_logits.shape[2]} vs {num_features}."
            )

        point_probabilities = torch.sigmoid(point_logits)
        point_probability_grid = point_probabilities.reshape(
            point_logits.shape[0],
            num_patches,
            self.patch_size,
            num_features,
        )

        valid_mask_grid = None
        if point_valid_mask is not None:
            if point_valid_mask.shape != point_logits.shape:
                raise ValueError(
                    "`batch.point_valid_mask` must match `point_logits` shape. "
                    f"Got {tuple(point_valid_mask.shape)} vs {tuple(point_logits.shape)}."
                )
            valid_mask_grid = point_valid_mask.to(device=point_logits.device, dtype=torch.bool).reshape(
                point_logits.shape[0],
                num_patches,
                self.patch_size,
                num_features,
            )
            point_probability_grid = torch.where(
                valid_mask_grid,
                point_probability_grid,
                torch.zeros_like(point_probability_grid),
            )

        aggregation = self.anomaly_patch_aggregation
        if aggregation == "or":
            patch_probabilities = 1.0 - torch.prod(1.0 - point_probability_grid, dim=2)
        elif aggregation == "mean":
            if valid_mask_grid is None:
                patch_probabilities = point_probability_grid.mean(dim=2)
            else:
                valid_counts = valid_mask_grid.sum(dim=2).clamp_min(1)
                patch_probabilities = point_probability_grid.sum(dim=2) / valid_counts.to(
                    dtype=point_probability_grid.dtype
                )
        elif aggregation == "max":
            patch_probabilities = point_probability_grid.max(dim=2).values
        else:
            raise ValueError(
                f"Unsupported anomaly patch aggregation `{aggregation}`. "
                "Supported: or, mean, max."
            )

        patch_probabilities = patch_probabilities.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.logit(patch_probabilities)

    def forward(self, batch: Batch) -> ModelOutput:
        """Run one forward pass from batched windows to output heads.

        Inputs:
            `batch.inputs` in shape `[B, W, D]`.

        Returns:
            `ModelOutput` with patch logits `[B, N_patches, D]` and optional
            reconstruction aligned to `batch.reconstruction_targets`.
        """

        # Apply point-level masking used by the reconstruction pretext task.
        inputs = self._apply_input_mask(batch.inputs, batch.mask_indices)

        # Tokenize fixed windows into a `[B, N_patches * D, d_model]` sequence.
        patch_output = self.patch_embedding(inputs)
        tokens = self.position_encoding(
            patch_output.tokens,
            num_patches=patch_output.num_patches,
            num_features=patch_output.num_features,
        )
        encoded = self.encoder(tokens, padding_mask=batch.token_padding_mask)
        head_inputs = (
            self.shared_output_projection(encoded)
            if self.shared_output_projection is not None
            else encoded
        )

        # Anomaly logits are always produced; reconstruction is optional by config.
        point_logits = None
        if self.use_observation_space_anomaly_head:
            point_logits = self.anomaly_head(
                head_inputs,
                num_patches=patch_output.num_patches,
                num_features=patch_output.num_features,
            )
            logits = self._aggregate_point_logits_to_patch_logits(
                point_logits,
                num_patches=patch_output.num_patches,
                num_features=patch_output.num_features,
                point_valid_mask=batch.point_valid_mask,
            )
        else:
            logits = self.anomaly_head(
                head_inputs,
                num_patches=patch_output.num_patches,
                num_features=patch_output.num_features,
            )
        reconstruction = None
        if self.reconstruction_head is not None:
            reconstruction = self.reconstruction_head(
                head_inputs,
                num_patches=patch_output.num_patches,
                num_features=patch_output.num_features,
            )

        return ModelOutput(
            logits=logits,
            point_logits=point_logits,
            reconstruction=reconstruction,
        )


__all__ = ["TimeRCDModel"]
