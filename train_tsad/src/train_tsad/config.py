"""Typed experiment configuration for training, evaluation, and runtime I/O.

This module defines the stable config surface consumed by the training
workflows. Each dataclass owns validation for one concern so downstream
modules can assume already-normalized settings instead of re-validating them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _coerce_path(value: str | Path) -> Path:
    """Convert string-like path config values into `Path`."""

    return value if isinstance(value, Path) else Path(value)


def _as_plain_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Materialize optional mapping as a mutable plain dictionary."""

    return dict(value or {})


@dataclass(slots=True)
class DataConfig:
    """Dataset, windowing, and loader settings."""

    dataset_root: Path
    split: str = "train"
    validation_split: str = "val"
    test_split: str = "test"
    use_sharded_dataset: bool = True
    manifest_name_template: str = "manifest.{split}.jsonl"
    context_size: int = 1024
    patch_size: int = 16
    stride: int = 512
    batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 0
    max_cached_shards: int = 32
    shuffle: bool = True
    shuffle_strategy: str = "shard_block"
    pin_memory: bool = True
    include_tail: bool = True
    pad_short_sequences: bool = True
    drop_last: bool = False
    enable_patch_masking: bool = True
    mask_ratio: float = 0.2
    load_normal_series: bool = False
    load_metadata: bool = False
    enable_direct_window_read: bool = True
    normalization_mode: str = "none"
    normalization_eps: float = 1e-5

    def __post_init__(self) -> None:
        self.dataset_root = _coerce_path(self.dataset_root)
        if self.context_size <= 0:
            raise ValueError("`data.context_size` must be positive.")
        if self.patch_size <= 0:
            raise ValueError("`data.patch_size` must be positive.")
        if self.context_size % self.patch_size != 0:
            raise ValueError("`data.context_size` must be divisible by `data.patch_size`.")
        if self.stride <= 0:
            raise ValueError("`data.stride` must be positive.")
        if self.batch_size <= 0:
            raise ValueError("`data.batch_size` must be positive.")
        if self.eval_batch_size <= 0:
            raise ValueError("`data.eval_batch_size` must be positive.")
        if self.num_workers < 0:
            raise ValueError("`data.num_workers` cannot be negative.")
        if self.max_cached_shards <= 0:
            raise ValueError("`data.max_cached_shards` must be positive.")
        if self.shuffle_strategy not in {"global_window", "sample_block", "shard_block"}:
            raise ValueError(
                "`data.shuffle_strategy` must be one of: global_window, sample_block, shard_block."
            )
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("`data.mask_ratio` must be in [0, 1].")
        if self.normalization_mode not in {"none", "per_sample_per_feature_zscore"}:
            raise ValueError(
                "`data.normalization_mode` must be one of: none, per_sample_per_feature_zscore."
            )
        if self.normalization_eps <= 0:
            raise ValueError("`data.normalization_eps` must be positive.")

    @property
    def num_patches(self) -> int:
        """Number of patches per context window (`context_size / patch_size`)."""

        return self.context_size // self.patch_size

    def manifest_path(self, split: str | None = None) -> Path:
        """Build split-specific manifest path from template."""

        target_split = split or self.split
        return self.dataset_root / self.manifest_name_template.format(split=target_split)


@dataclass(slots=True)
class ModelConfig:
    """Transformer model settings for the paper-aligned TimeRCD path."""

    name: str = "timercd"
    patch_size: int = 16
    d_model: int = 512
    d_proj: int = 256
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    use_learned_positional_encoding: bool = True
    use_shared_output_projection: bool = False
    use_observation_space_anomaly_head: bool = False
    anomaly_patch_aggregation: str = "or"

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("`model.patch_size` must be positive.")
        if self.d_model <= 0:
            raise ValueError("`model.d_model` must be positive.")
        if self.d_proj <= 0:
            raise ValueError("`model.d_proj` must be positive.")
        if self.num_layers <= 0:
            raise ValueError("`model.num_layers` must be positive.")
        if self.num_heads <= 0:
            raise ValueError("`model.num_heads` must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("`model.d_model` must be divisible by `model.num_heads`.")
        if self.mlp_ratio <= 0:
            raise ValueError("`model.mlp_ratio` must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("`model.dropout` must be in [0, 1).")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("`model.attention_dropout` must be in [0, 1).")
        if self.anomaly_patch_aggregation not in {"or", "mean", "max"}:
            raise ValueError(
                "`model.anomaly_patch_aggregation` must be one of: or, mean, max."
            )


@dataclass(slots=True)
class LossConfig:
    """Loss weights and anomaly supervision settings."""

    anomaly_loss_weight: float = 1.0
    point_anomaly_loss_weight: float = 0.0
    reconstruction_loss_weight: float = 0.01
    anomaly_loss_type: str = "bce"
    anomaly_pos_weight: float | str | None = "auto"
    anomaly_asl_gamma_neg: float = 2.0
    anomaly_asl_gamma_pos: float = 0.0
    anomaly_asl_clip: float = 0.05
    point_anomaly_pos_weight: float | str | None = "auto"
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.anomaly_loss_weight <= 0:
            raise ValueError("`loss.anomaly_loss_weight` must be positive.")
        if self.point_anomaly_loss_weight < 0:
            raise ValueError("`loss.point_anomaly_loss_weight` cannot be negative.")
        if self.reconstruction_loss_weight < 0:
            raise ValueError("`loss.reconstruction_loss_weight` cannot be negative.")
        if self.anomaly_loss_type not in {"bce", "asl"}:
            raise ValueError("`loss.anomaly_loss_type` must be one of: bce, asl.")
        if isinstance(self.anomaly_pos_weight, str):
            if self.anomaly_pos_weight != "auto":
                raise ValueError("`loss.anomaly_pos_weight` must be a positive float, null, or `auto`.")
        elif self.anomaly_pos_weight is not None and self.anomaly_pos_weight <= 0:
            raise ValueError("`loss.anomaly_pos_weight` must be positive when provided.")
        if self.anomaly_asl_gamma_neg < 0:
            raise ValueError("`loss.anomaly_asl_gamma_neg` cannot be negative.")
        if self.anomaly_asl_gamma_pos < 0:
            raise ValueError("`loss.anomaly_asl_gamma_pos` cannot be negative.")
        if not 0.0 <= self.anomaly_asl_clip < 1.0:
            raise ValueError("`loss.anomaly_asl_clip` must be in [0, 1).")
        if isinstance(self.point_anomaly_pos_weight, str):
            if self.point_anomaly_pos_weight != "auto":
                raise ValueError(
                    "`loss.point_anomaly_pos_weight` must be a positive float, null, or `auto`."
                )
        elif self.point_anomaly_pos_weight is not None and self.point_anomaly_pos_weight <= 0:
            raise ValueError("`loss.point_anomaly_pos_weight` must be positive when provided.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("`loss.label_smoothing` must be in [0, 1).")


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer and scheduler settings."""

    name: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 0.0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("`optimizer.lr` must be positive.")
        if self.weight_decay < 0:
            raise ValueError("`optimizer.weight_decay` cannot be negative.")
        if len(self.betas) != 2:
            raise ValueError("`optimizer.betas` must contain two coefficients.")
        if not all(0.0 < beta < 1.0 for beta in self.betas):
            raise ValueError("`optimizer.betas` values must be in (0, 1).")
        if self.eps <= 0:
            raise ValueError("`optimizer.eps` must be positive.")
        if self.warmup_epochs < 0:
            raise ValueError("`optimizer.warmup_epochs` cannot be negative.")
        if self.min_lr < 0:
            raise ValueError("`optimizer.min_lr` cannot be negative.")


@dataclass(slots=True)
class TrainConfig:
    """Training loop and checkpoint behavior."""

    output_dir: Path = Path("runs/default")
    seed: int = 42
    device: str = "cuda"
    max_epochs: int = 50
    early_stopping_patience: int = 7
    gradient_clip_norm: float | None = 1.0
    gradient_accumulation_steps: int = 1
    log_every_n_steps: int = 50
    validate_every_n_epochs: int = 1
    save_best_only: bool = False
    mixed_precision: bool = False
    progress_write_interval_seconds: float = 2.0

    def __post_init__(self) -> None:
        self.output_dir = _coerce_path(self.output_dir)
        if self.max_epochs <= 0:
            raise ValueError("`train.max_epochs` must be positive.")
        if self.early_stopping_patience < 0:
            raise ValueError("`train.early_stopping_patience` cannot be negative.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("`train.gradient_clip_norm` must be positive when provided.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("`train.gradient_accumulation_steps` must be positive.")
        if self.log_every_n_steps <= 0:
            raise ValueError("`train.log_every_n_steps` must be positive.")
        if self.validate_every_n_epochs <= 0:
            raise ValueError("`train.validate_every_n_epochs` must be positive.")
        if self.progress_write_interval_seconds <= 0:
            raise ValueError("`train.progress_write_interval_seconds` must be positive.")


@dataclass(slots=True)
class EvalConfig:
    """Evaluation and score post-processing settings."""

    task: str = "patch_feature"
    primary_metric: str = "pr_auc"
    primary_metric_mode: str = "max"
    threshold: float = 0.5
    threshold_search: bool = False
    threshold_search_metric: str = "f1"
    patch_feature_score_aggregation: str = "mean"
    report_per_feature: bool = False
    report_per_sample: bool = False
    score_reduction: str = "mean"
    point_score_aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.task not in {"patch_feature", "point_any_feature_legacy"}:
            raise ValueError(
                "`eval.task` must be one of: patch_feature, point_any_feature_legacy."
            )
        if not self.primary_metric.strip():
            raise ValueError("`eval.primary_metric` cannot be empty.")
        if self.primary_metric_mode not in {"min", "max"}:
            raise ValueError("`eval.primary_metric_mode` must be one of: min, max.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("`eval.threshold` must be in [0, 1].")
        if self.patch_feature_score_aggregation not in {"mean", "max"}:
            raise ValueError(
                "`eval.patch_feature_score_aggregation` must be one of: mean, max."
            )


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level configuration shared by train and evaluation entrypoints."""

    experiment_name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tags: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("`experiment_name` cannot be empty.")
        if self.data.patch_size != self.model.patch_size:
            raise ValueError(
                "`data.patch_size` must match `model.patch_size` to keep patch labels aligned."
            )
        if self.loss.point_anomaly_loss_weight > 0.0 and not self.model.use_observation_space_anomaly_head:
            raise ValueError(
                "`loss.point_anomaly_loss_weight > 0` requires "
                "`model.use_observation_space_anomaly_head = true`."
            )

    def to_dict(self) -> dict[str, Any]:
        """Export config dataclasses as JSON-serializable dictionary."""

        payload = asdict(self)
        payload["data"]["dataset_root"] = str(self.data.dataset_root)
        payload["train"]["output_dir"] = str(self.train.output_dir)
        return payload

    def clone(self) -> ExperimentConfig:
        """Return a deep config clone suitable for runtime overrides."""

        return type(self).from_mapping(self.to_dict())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ExperimentConfig:
        """Build `ExperimentConfig` from mapping payload with extras passthrough."""

        payload = _as_plain_dict(mapping)
        if "data" not in payload:
            raise ValueError("Configuration mapping must include a `data` section.")

        known_keys = {
            "experiment_name",
            "data",
            "model",
            "loss",
            "optimizer",
            "train",
            "eval",
            "tags",
            "extras",
        }
        extras = {key: value for key, value in payload.items() if key not in known_keys}

        return cls(
            experiment_name=payload.get("experiment_name", "timercd-experiment"),
            data=DataConfig(**_as_plain_dict(payload.get("data"))),
            model=ModelConfig(**_as_plain_dict(payload.get("model"))),
            loss=LossConfig(**_as_plain_dict(payload.get("loss"))),
            optimizer=OptimizerConfig(**_as_plain_dict(payload.get("optimizer"))),
            train=TrainConfig(**_as_plain_dict(payload.get("train"))),
            eval=EvalConfig(**_as_plain_dict(payload.get("eval"))),
            tags=list(payload.get("tags", [])),
            extras={**_as_plain_dict(payload.get("extras")), **extras},
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        """Load experiment config from JSON/YAML file."""

        config_path = _coerce_path(path)
        suffix = config_path.suffix.lower()
        text = config_path.read_text(encoding="utf-8")

        if suffix == ".json":
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "YAML config loading requires PyYAML. Install it or use a JSON config file."
                )
            payload = yaml.safe_load(text)
        else:
            raise ValueError(
                f"Unsupported config format `{suffix}`. Expected one of: .json, .yaml, .yml."
            )

        if not isinstance(payload, Mapping):
            raise ValueError("Top-level config payload must be a mapping.")
        return cls.from_mapping(payload)


def build_timercd_base_config(
    *,
    dataset_root: str | Path,
    experiment_name: str = "timercd-base",
) -> ExperimentConfig:
    """Return a paper-aligned default configuration preset."""

    return ExperimentConfig(
        experiment_name=experiment_name,
        data=DataConfig(dataset_root=dataset_root),
        model=ModelConfig(),
        loss=LossConfig(),
        optimizer=OptimizerConfig(),
        train=TrainConfig(),
        eval=EvalConfig(),
    )


__all__ = [
    "DataConfig",
    "EvalConfig",
    "ExperimentConfig",
    "LossConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainConfig",
    "build_timercd_base_config",
]

