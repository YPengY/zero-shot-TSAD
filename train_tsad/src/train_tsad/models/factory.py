"""Model construction helpers for workflow and CLI entry points."""

from __future__ import annotations

from ..config import DataConfig, LossConfig, ModelConfig
from .timercd import TimeRCDModel


def build_timercd_model(
    *,
    data_config: DataConfig,
    model_config: ModelConfig,
    loss_config: LossConfig,
    num_features: int,
) -> TimeRCDModel:
    """Build the default TimeRCD model from validated config sections.

    The factory is intentionally thin: it translates configuration fields into
    constructor arguments while keeping workflow code independent from the
    concrete model class layout.
    """

    return TimeRCDModel(
        patch_size=model_config.patch_size,
        d_model=model_config.d_model,
        d_proj=model_config.d_proj,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        max_patches=data_config.num_patches,
        max_features=max(num_features, 1),
        mlp_ratio=model_config.mlp_ratio,
        dropout=model_config.dropout,
        attention_dropout=model_config.attention_dropout,
        activation=model_config.activation,
        use_learned_positional_encoding=model_config.use_learned_positional_encoding,
        use_shared_output_projection=model_config.use_shared_output_projection,
        use_observation_space_anomaly_head=model_config.use_observation_space_anomaly_head,
        anomaly_patch_aggregation=model_config.anomaly_patch_aggregation,
        use_reconstruction_head=loss_config.reconstruction_loss_weight > 0.0,
    )
