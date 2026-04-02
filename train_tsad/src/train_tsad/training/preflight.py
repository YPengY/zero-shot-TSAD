"""Training-time preparation steps that depend on the dataset contents.

This module contains the logic that must inspect actual training data before
the model is built: label-balance estimation, automatic positive-class weight
resolution, and pre-training data quality inspection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..config import DataConfig, LossConfig
from ..data import DataQualityInspector, default_inspection_splits
from ..data.factory import RawDataset, build_raw_dataset
from ..data.quality import DatasetQualityReport
from ..data.windowizer import SlidingContextWindowizer
from ..utils import resolve_path, write_json_file

LabelSpace = Literal["patch_feature", "point_feature"]


@dataclass(frozen=True, slots=True)
class LabelBalanceStatistics:
    """Label-balance summary for one supervised target space."""

    label_space: LabelSpace
    num_windows: int
    num_positive_units: int
    num_negative_units: int
    positive_rate: float

    @property
    def num_units(self) -> int:
        """Return the total number of supervised units."""

        return self.num_positive_units + self.num_negative_units

    @property
    def auto_pos_weight(self) -> float:
        """Return the inverse positive frequency used by BCE-like losses."""

        if self.num_positive_units <= 0:
            raise ValueError(
                f"Cannot estimate `{self.label_space}` loss weight because no positive labels were found."
            )
        return float(self.num_negative_units / self.num_positive_units)

    def to_dict(self) -> dict[str, float]:
        """Serialize the statistics as a JSON-friendly mapping."""

        return {
            "label_space": self.label_space,
            "num_windows": float(self.num_windows),
            "num_units": float(self.num_units),
            "num_positive_units": float(self.num_positive_units),
            "num_negative_units": float(self.num_negative_units),
            "positive_rate": float(self.positive_rate),
            "auto_pos_weight": float(self.auto_pos_weight)
            if self.num_positive_units > 0
            else float("nan"),
        }


@dataclass(frozen=True, slots=True)
class ResolvedLossWeights:
    """Runtime loss-weight resolution derived from the training split.

    This object separates user configuration from dataset-derived quantities
    such as automatically estimated positive-class weights.
    """

    anomaly_loss_type: str
    anomaly_pos_weight: float | None
    point_anomaly_pos_weight: float | None
    patch_feature_stats: LabelBalanceStatistics | None = None
    point_feature_stats: LabelBalanceStatistics | None = None


@dataclass(frozen=True, slots=True)
class DataQualityInspectionResult:
    """Persisted quality report emitted before an optional training run."""

    report: DatasetQualityReport
    report_path: Path


def _build_windowizer(data_config: DataConfig) -> SlidingContextWindowizer:
    """Create the shared windowizer used by inspection and label statistics."""

    return SlidingContextWindowizer(
        context_size=data_config.context_size,
        patch_size=data_config.patch_size,
        stride=data_config.stride,
        pad_short_sequences=data_config.pad_short_sequences,
        include_tail=data_config.include_tail,
    )


def _iter_windows(raw_dataset: RawDataset, *, data_config: DataConfig):
    """Yield window samples from either raw-sequence or pre-windowed datasets."""

    if bool(getattr(raw_dataset, "is_prewindowed", False)):
        for index in range(len(raw_dataset)):
            yield raw_dataset[index]
        return

    windowizer = _build_windowizer(data_config)
    for sample_index in range(len(raw_dataset)):
        yield from windowizer.transform(raw_dataset[sample_index])


def _compute_label_balance(
    raw_dataset: RawDataset,
    *,
    data_config: DataConfig,
    label_space: LabelSpace,
) -> LabelBalanceStatistics:
    """Compute label-balance statistics for one supervision space."""

    positive_units = 0
    total_units = 0
    num_windows = 0

    for window in _iter_windows(raw_dataset, data_config=data_config):
        if label_space == "patch_feature":
            labels = np.asarray(window.patch_labels, dtype=np.uint8)
            positive_units += int(labels.sum())
            total_units += int(labels.size)
        else:
            point_mask = np.asarray(window.point_mask, dtype=np.uint8)
            if point_mask.ndim != 2:
                raise ValueError(
                    "`window.point_mask` must have shape [W, D] when point anomaly weighting is enabled."
                )
            valid_length = max(0, int(window.context_end) - int(window.context_start))
            valid_point_mask = point_mask[:valid_length]
            positive_units += int(valid_point_mask.sum())
            total_units += int(valid_point_mask.size)
        num_windows += 1

    if total_units <= 0:
        raise ValueError(
            f"Unable to estimate `{label_space}` statistics from an empty training split."
        )

    negative_units = total_units - positive_units
    return LabelBalanceStatistics(
        label_space=label_space,
        num_windows=num_windows,
        num_positive_units=positive_units,
        num_negative_units=negative_units,
        positive_rate=float(positive_units / total_units),
    )


def estimate_patch_feature_balance(
    raw_dataset: RawDataset,
    *,
    data_config: DataConfig,
) -> LabelBalanceStatistics:
    """Estimate patch-feature label balance for the training split."""

    return _compute_label_balance(
        raw_dataset,
        data_config=data_config,
        label_space="patch_feature",
    )


def estimate_point_feature_balance(
    raw_dataset: RawDataset,
    *,
    data_config: DataConfig,
) -> LabelBalanceStatistics:
    """Estimate point-feature label balance for the training split."""

    return _compute_label_balance(
        raw_dataset,
        data_config=data_config,
        label_space="point_feature",
    )


def resolve_loss_weights(
    loss_config: LossConfig,
    *,
    data_config: DataConfig,
    train_raw_dataset: RawDataset,
    logger: logging.Logger,
) -> ResolvedLossWeights:
    """Resolve runtime loss weights without mutating unrelated config fields."""

    patch_feature_stats: LabelBalanceStatistics | None = None
    point_feature_stats: LabelBalanceStatistics | None = None

    if loss_config.anomaly_loss_type == "asl":
        anomaly_pos_weight = None
        logger.info("Patch anomaly loss uses ASL; `anomaly_pos_weight` is intentionally ignored.")
    else:
        configured_patch_weight = loss_config.anomaly_pos_weight
        if configured_patch_weight is None:
            anomaly_pos_weight = None
            logger.warning(
                "Patch anomaly loss does not use `anomaly_pos_weight`; highly imbalanced labels may destabilize training."
            )
        elif configured_patch_weight == "auto":
            patch_feature_stats = estimate_patch_feature_balance(
                train_raw_dataset,
                data_config=data_config,
            )
            anomaly_pos_weight = patch_feature_stats.auto_pos_weight
            logger.info(
                "Resolved patch anomaly pos_weight from training labels: %.6f (positive_rate=%.6f, windows=%d).",
                anomaly_pos_weight,
                patch_feature_stats.positive_rate,
                patch_feature_stats.num_windows,
            )
        else:
            anomaly_pos_weight = float(configured_patch_weight)
            logger.info("Using configured patch anomaly pos_weight: %.6f.", anomaly_pos_weight)

    if loss_config.point_anomaly_loss_weight <= 0.0:
        point_anomaly_pos_weight = None
    else:
        configured_point_weight = loss_config.point_anomaly_pos_weight
        if configured_point_weight is None:
            point_anomaly_pos_weight = None
            logger.warning(
                "Point anomaly auxiliary loss does not use `point_anomaly_pos_weight`; "
                "rare positives may collapse to trivial predictions."
            )
        elif configured_point_weight == "auto":
            point_feature_stats = estimate_point_feature_balance(
                train_raw_dataset,
                data_config=data_config,
            )
            point_anomaly_pos_weight = point_feature_stats.auto_pos_weight
            logger.info(
                "Resolved point anomaly pos_weight from training labels: %.6f (positive_rate=%.6f, windows=%d).",
                point_anomaly_pos_weight,
                point_feature_stats.positive_rate,
                point_feature_stats.num_windows,
            )
        else:
            point_anomaly_pos_weight = float(configured_point_weight)
            logger.info(
                "Using configured point anomaly pos_weight: %.6f.",
                point_anomaly_pos_weight,
            )

    return ResolvedLossWeights(
        anomaly_loss_type=loss_config.anomaly_loss_type,
        anomaly_pos_weight=anomaly_pos_weight,
        point_anomaly_pos_weight=point_anomaly_pos_weight,
        patch_feature_stats=patch_feature_stats,
        point_feature_stats=point_feature_stats,
    )


def run_data_quality_inspection(
    *,
    data_config: DataConfig,
    train_output_dir: Path,
    train_raw_dataset: RawDataset,
    val_raw_dataset: RawDataset | None,
    base_dir: Path,
    max_samples: int | None,
    output_path: Path | None,
    logger: logging.Logger,
) -> DataQualityInspectionResult | None:
    """Inspect raw-sequence data quality and persist the JSON report."""

    if bool(getattr(train_raw_dataset, "is_prewindowed", False)):
        logger.info(
            "Skipping data quality inspection because the selected dataset is already window-packed."
        )
        return None

    datasets_by_split: dict[str, RawDataset] = {
        data_config.split: train_raw_dataset,
    }
    missing_splits: list[str] = []
    if val_raw_dataset is not None:
        datasets_by_split[data_config.validation_split] = val_raw_dataset

    for split in default_inspection_splits(data_config):
        if split in datasets_by_split:
            continue
        try:
            datasets_by_split[split] = build_raw_dataset(
                data_config,
                split=split,
                base_dir=base_dir,
            )
        except FileNotFoundError:
            missing_splits.append(split)

    inspector = DataQualityInspector(
        windowizer=_build_windowizer(data_config),
        max_samples=max_samples,
    )
    report = inspector.inspect_many(
        datasets_by_split,
        expected_training_split=data_config.split,
        missing_splits=missing_splits,
    )

    report_payload = report.to_dict()
    report_path = resolve_path(
        output_path or (train_output_dir / "data_quality_report.json"),
        base_dir=base_dir,
    )
    write_json_file(report_path, report_payload)
    logger.info(
        "Data quality inspection completed for splits=%s. Overall grade=%s, score=%.2f. Report=%s",
        sorted(report.split_reports),
        report.overall_grade,
        report.overall_score,
        report_path,
    )
    return DataQualityInspectionResult(report=report, report_path=report_path)


__all__ = [
    "DataQualityInspectionResult",
    "LabelBalanceStatistics",
    "ResolvedLossWeights",
    "estimate_patch_feature_balance",
    "estimate_point_feature_balance",
    "resolve_loss_weights",
    "run_data_quality_inspection",
]
