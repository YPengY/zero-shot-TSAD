from __future__ import annotations

from torch import Tensor, nn

from ..interfaces import Batch, ModelOutput
from .encoder import TransformerEncoder
from .heads import AnomalyHead, ReconstructionHead
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
        use_reconstruction_head: bool = True,
    ) -> None:
        """Build the TimeRCD model stack.

        Structure:
        1. Patch embedding over `[B, W, D]`.
        2. Grid positional encoding over `(patch, feature)` tokens.
        3. Transformer encoder.
        4. Anomaly head (+ optional reconstruction head).
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
        self.anomaly_head = AnomalyHead(d_model=d_model)
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
        encoded = self.encoder(tokens)

        # Anomaly logits are always produced; reconstruction is optional by config.
        logits = self.anomaly_head(
            encoded,
            num_patches=patch_output.num_patches,
            num_features=patch_output.num_features,
        )
        reconstruction = None
        if self.reconstruction_head is not None:
            reconstruction = self.reconstruction_head(
                encoded,
                num_patches=patch_output.num_patches,
                num_features=patch_output.num_features,
            )

        return ModelOutput(logits=logits, reconstruction=reconstruction)


__all__ = ["TimeRCDModel"]
