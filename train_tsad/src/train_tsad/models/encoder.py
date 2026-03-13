from __future__ import annotations

import inspect

from torch import Tensor, nn


class TransformerEncoder(nn.Module):
    """Thin wrapper around the encoder stack used by the paper-aligned model."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("`d_model` must be positive.")
        if num_heads <= 0:
            raise ValueError("`num_heads` must be positive.")
        if num_layers <= 0:
            raise ValueError("`num_layers` must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("`d_model` must be divisible by `num_heads`.")
        if mlp_ratio <= 0:
            raise ValueError("`mlp_ratio` must be positive.")

        feedforward_dim = int(d_model * mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        encoder_kwargs = {"num_layers": num_layers}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            encoder_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(layer, **encoder_kwargs)
        self.norm = nn.LayerNorm(d_model)

        if attention_dropout != dropout:
            # PyTorch's stock encoder layer exposes a single dropout value for the
            # attention path. Keep the dedicated argument for config compatibility.
            for module in self.encoder.layers:
                module.self_attn.dropout = attention_dropout

    def forward(self, tokens: Tensor, *, padding_mask: Tensor | None = None) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must have shape [B, L, d_model], got ndim={tokens.ndim}.")
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.norm(encoded)


__all__ = ["TransformerEncoder"]
