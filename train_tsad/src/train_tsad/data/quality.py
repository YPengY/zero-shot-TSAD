from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..interfaces import DatasetProtocol
from .windowizer import SlidingContextWindowizer


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _distribution_stats(values: list[float] | list[int], *, prefix: str) -> dict[str, float | int | None]:
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_p50": float(np.percentile(array, 50)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_max": float(np.max(array)),
    }


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


@dataclass(slots=True)
class DataQualityThresholds:
    """Rule thresholds for pre-training data quality diagnostics."""

    min_samples: int = 100
    min_point_positive_rate: float = 0.001
    min_patch_positive_rate: float = 0.001
    max_zero_positive_sample_ratio: float = 0.5
    max_short_sequence_ratio: float = 0.3
    max_padded_point_ratio: float = 0.3
    max_nonfinite_value_ratio: float = 0.0
    max_point_mask_any_mismatch_ratio: float = 0.0
    min_positive_feature_coverage_ratio: float = 0.5
    max_feature_dim_variants: int = 1


@dataclass(slots=True)
class QualityIssue:
    """One rule result produced by quality diagnostics."""

    code: str
    severity: str
    message: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.metric is not None:
            payload["metric"] = self.metric
        if self.value is not None:
            payload["value"] = float(self.value)
        if self.threshold is not None:
            payload["threshold"] = float(self.threshold)
        return payload


@dataclass(slots=True)
class SplitQualityReport:
    """Detailed quality report for one split."""

    split: str
    expected_training_split: str | None
    stats: dict[str, Any]
    issues: list[QualityIssue] = field(default_factory=list)
    quality_score: float = 0.0
    quality_grade: str = "F"
    has_blocking_issues: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "expected_training_split": self.expected_training_split,
            "stats": self.stats,
            "issues": [issue.to_dict() for issue in self.issues],
            "quality_score": float(self.quality_score),
            "quality_grade": self.quality_grade,
            "has_blocking_issues": bool(self.has_blocking_issues),
        }


@dataclass(slots=True)
class DatasetQualityReport:
    """Aggregated report across one or more splits."""

    expected_training_split: str | None
    split_reports: dict[str, SplitQualityReport]
    missing_splits: list[str]
    overall_score: float
    overall_grade: str
    recommended_to_train: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "expected_training_split": self.expected_training_split,
                "inspected_splits": list(self.split_reports.keys()),
                "missing_splits": self.missing_splits,
                "overall_score": float(self.overall_score),
                "overall_grade": self.overall_grade,
                "recommended_to_train": bool(self.recommended_to_train),
            },
            "splits": {split: report.to_dict() for split, report in self.split_reports.items()},
        }


class DataQualityInspector:
    """Reusable interface for pre-training dataset quality diagnostics."""

    def __init__(
        self,
        *,
        windowizer: SlidingContextWindowizer,
        thresholds: DataQualityThresholds | None = None,
        max_samples: int | None = None,
    ) -> None:
        if max_samples is not None and max_samples <= 0:
            raise ValueError("`max_samples` must be positive when provided.")
        self.windowizer = windowizer
        self.thresholds = thresholds or DataQualityThresholds()
        self.max_samples = max_samples

    def inspect_many(
        self,
        datasets_by_split: Mapping[str, DatasetProtocol],
        *,
        expected_training_split: str | None = None,
        missing_splits: list[str] | None = None,
    ) -> DatasetQualityReport:
        split_reports: dict[str, SplitQualityReport] = {}
        for split, dataset in datasets_by_split.items():
            split_reports[split] = self.inspect_split(
                dataset,
                split=split,
                expected_training_split=expected_training_split,
            )

        if not split_reports:
            overall_score = 0.0
        else:
            overall_score = float(
                np.mean([report.quality_score for report in split_reports.values()], dtype=np.float64)
            )
        overall_grade = _score_to_grade(overall_score)

        recommended_to_train = True
        if expected_training_split is not None:
            train_report = split_reports.get(expected_training_split)
            if train_report is None:
                recommended_to_train = False
            else:
                recommended_to_train = (not train_report.has_blocking_issues) and train_report.quality_score >= 60
        elif any(report.has_blocking_issues for report in split_reports.values()):
            recommended_to_train = False

        return DatasetQualityReport(
            expected_training_split=expected_training_split,
            split_reports=split_reports,
            missing_splits=missing_splits or [],
            overall_score=overall_score,
            overall_grade=overall_grade,
            recommended_to_train=recommended_to_train,
        )

    def inspect_split(
        self,
        raw_dataset: DatasetProtocol,
        *,
        split: str,
        expected_training_split: str | None = None,
    ) -> SplitQualityReport:
        num_samples_total = int(len(raw_dataset))
        analyzed_samples = (
            num_samples_total
            if self.max_samples is None
            else min(num_samples_total, int(self.max_samples))
        )

        lengths: list[int] = []
        feature_counts: list[int] = []
        samples_mean: list[float] = []
        samples_std: list[float] = []
        samples_abs_max: list[float] = []
        windows_per_sample: list[int] = []
        positive_points_per_sample: list[int] = []

        point_positive = 0
        point_total = 0
        patch_positive = 0
        patch_total = 0
        num_windows = 0
        windows_with_positive = 0
        short_sequences = 0
        zero_positive_samples = 0
        missing_point_mask_samples = 0
        invalid_point_mask_samples = 0
        invalid_point_mask_any_samples = 0
        invalid_normal_series_samples = 0
        samples_with_nonfinite = 0
        near_constant_samples = 0
        normal_series_available_samples = 0
        sample_loading_errors = 0
        window_transform_errors = 0

        nan_values = 0
        inf_values = 0
        total_values = 0
        padded_point_units = 0
        total_point_units = 0
        padded_windows = 0
        point_mask_any_mismatch_points = 0
        point_mask_any_total_points = 0

        normal_series_abs_diff_mean: list[float] = []
        error_examples: list[str] = []

        point_positive_by_feature: np.ndarray | None = None
        point_total_by_feature: np.ndarray | None = None
        feature_rate_unavailable = False

        for index in range(analyzed_samples):
            try:
                sample = raw_dataset[index]
            except Exception as exc:  # pragma: no cover - defensive guard
                sample_loading_errors += 1
                if len(error_examples) < 5:
                    error_examples.append(f"load[{index}]: {type(exc).__name__}: {exc}")
                continue

            series = np.asarray(sample.series, dtype=np.float32)
            if series.ndim != 2 or series.shape[0] <= 0 or series.shape[1] <= 0:
                sample_loading_errors += 1
                if len(error_examples) < 5:
                    error_examples.append(
                        f"shape[{index}]: invalid series shape {tuple(series.shape)} for sample_id={sample.sample_id}"
                    )
                continue

            sample_length = int(series.shape[0])
            num_features = int(series.shape[1])
            lengths.append(sample_length)
            feature_counts.append(num_features)
            if sample_length < self.windowizer.context_size:
                short_sequences += 1

            finite_mask = np.isfinite(series)
            sample_total_values = int(series.size)
            total_values += sample_total_values
            sample_nan = int(np.isnan(series).sum())
            sample_inf = int(np.isinf(series).sum())
            nan_values += sample_nan
            inf_values += sample_inf
            if sample_nan > 0 or sample_inf > 0:
                samples_with_nonfinite += 1

            finite_values = series[finite_mask]
            if finite_values.size > 0:
                sample_mean_value = float(np.mean(finite_values))
                sample_std_value = float(np.std(finite_values))
                sample_abs_max_value = float(np.max(np.abs(finite_values)))
                samples_mean.append(sample_mean_value)
                samples_std.append(sample_std_value)
                samples_abs_max.append(sample_abs_max_value)
                if sample_std_value <= 1e-6:
                    near_constant_samples += 1

            point_mask_valid: np.ndarray | None = None
            point_mask_raw = sample.point_mask
            if point_mask_raw is None:
                missing_point_mask_samples += 1
            else:
                point_mask = np.asarray(point_mask_raw)
                if point_mask.ndim != 2 or point_mask.shape != series.shape:
                    invalid_point_mask_samples += 1
                else:
                    point_mask_valid = (point_mask > 0).astype(np.uint8, copy=False)
                    sample_positive_points = int(point_mask_valid.sum())
                    positive_points_per_sample.append(sample_positive_points)
                    point_positive += sample_positive_points
                    point_total += int(point_mask_valid.size)
                    if sample_positive_points == 0:
                        zero_positive_samples += 1

                    if point_positive_by_feature is None:
                        point_positive_by_feature = np.zeros((num_features,), dtype=np.int64)
                        point_total_by_feature = np.zeros((num_features,), dtype=np.int64)
                    elif point_positive_by_feature.shape[0] != num_features:
                        feature_rate_unavailable = True
                        point_positive_by_feature = None
                        point_total_by_feature = None
                    if point_positive_by_feature is not None and point_total_by_feature is not None:
                        point_positive_by_feature += np.asarray(point_mask_valid.sum(axis=0), dtype=np.int64)
                        point_total_by_feature += sample_length

            if sample.point_mask_any is not None:
                point_mask_any = np.asarray(sample.point_mask_any)
                if point_mask_any.ndim != 1 or point_mask_any.shape[0] != sample_length:
                    invalid_point_mask_any_samples += 1
                elif point_mask_valid is not None:
                    derived_point_mask_any = (point_mask_valid.sum(axis=1) > 0).astype(np.uint8, copy=False)
                    provided_point_mask_any = (point_mask_any > 0).astype(np.uint8, copy=False)
                    mismatches = int(np.not_equal(provided_point_mask_any, derived_point_mask_any).sum())
                    point_mask_any_mismatch_points += mismatches
                    point_mask_any_total_points += sample_length

            if sample.normal_series is not None:
                normal_series_available_samples += 1
                normal_series = np.asarray(sample.normal_series, dtype=np.float32)
                if normal_series.shape != series.shape:
                    invalid_normal_series_samples += 1
                else:
                    diff = np.abs(normal_series - series)
                    finite_diff = diff[np.isfinite(diff)]
                    if finite_diff.size > 0:
                        normal_series_abs_diff_mean.append(float(np.mean(finite_diff)))

            try:
                windows = self.windowizer.transform(sample)
            except Exception as exc:
                window_transform_errors += 1
                if len(error_examples) < 5:
                    error_examples.append(f"window[{index}]: {type(exc).__name__}: {exc}")
                windows_per_sample.append(0)
                continue

            sample_windows = int(len(windows))
            windows_per_sample.append(sample_windows)
            for window in windows:
                patch_labels = np.asarray(window.patch_labels, dtype=np.uint8)
                positive_patch_units = int((patch_labels > 0).sum())
                patch_positive += positive_patch_units
                patch_total += int(patch_labels.size)
                if positive_patch_units > 0:
                    windows_with_positive += 1

                num_windows += 1
                effective_length = int(window.context_end) - int(window.context_start)
                effective_length = max(0, min(effective_length, self.windowizer.context_size))
                padded_steps = self.windowizer.context_size - effective_length
                if padded_steps > 0:
                    padded_windows += 1
                padded_point_units += padded_steps * int(window.series.shape[1])
                total_point_units += self.windowizer.context_size * int(window.series.shape[1])

        unique_feature_counts = sorted(set(feature_counts))
        point_positive_rate = _rate(point_positive, point_total)
        patch_positive_rate = _rate(patch_positive, patch_total)
        suggested_anomaly_pos_weight = _rate(
            patch_total - patch_positive,
            patch_positive if patch_positive > 0 else 1,
        )

        positive_feature_coverage_ratio: float | None = None
        feature_point_positive_rate_stats: dict[str, float | None] = {
            "feature_point_positive_rate_min": None,
            "feature_point_positive_rate_p50": None,
            "feature_point_positive_rate_max": None,
        }
        if (
            point_positive_by_feature is not None
            and point_total_by_feature is not None
            and point_total_by_feature.size > 0
        ):
            feature_rates = point_positive_by_feature / np.maximum(point_total_by_feature, 1)
            positive_feature_coverage_ratio = _rate(int((feature_rates > 0).sum()), int(feature_rates.size))
            feature_point_positive_rate_stats = {
                "feature_point_positive_rate_min": float(np.min(feature_rates)),
                "feature_point_positive_rate_p50": float(np.percentile(feature_rates, 50)),
                "feature_point_positive_rate_max": float(np.max(feature_rates)),
            }

        stats: dict[str, Any] = {
            "split": split,
            "num_samples_total": num_samples_total,
            "num_samples_analyzed": analyzed_samples,
            "analyzed_fraction": _rate(analyzed_samples, max(num_samples_total, 1)),
            "num_windows": int(num_windows),
            "num_feature_values": ",".join(str(v) for v in unique_feature_counts),
            "num_feature_values_count": int(len(unique_feature_counts)),
            "is_training_split": bool(expected_training_split is not None and split == expected_training_split),
            "short_sequence_ratio": _rate(short_sequences, analyzed_samples),
            "zero_positive_sample_ratio": _rate(zero_positive_samples, analyzed_samples),
            "missing_point_mask_ratio": _rate(missing_point_mask_samples, analyzed_samples),
            "invalid_point_mask_ratio": _rate(invalid_point_mask_samples, analyzed_samples),
            "invalid_point_mask_any_ratio": _rate(invalid_point_mask_any_samples, analyzed_samples),
            "invalid_normal_series_ratio": _rate(invalid_normal_series_samples, analyzed_samples),
            "sample_loading_error_ratio": _rate(sample_loading_errors, analyzed_samples),
            "window_transform_error_ratio": _rate(window_transform_errors, analyzed_samples),
            "samples_with_nonfinite_ratio": _rate(samples_with_nonfinite, analyzed_samples),
            "nonfinite_value_ratio": _rate(nan_values + inf_values, total_values),
            "near_constant_sample_ratio": _rate(near_constant_samples, analyzed_samples),
            "point_mask_any_mismatch_ratio": _rate(point_mask_any_mismatch_points, point_mask_any_total_points),
            "normal_series_available_ratio": _rate(normal_series_available_samples, analyzed_samples),
            "point_positive_rate": point_positive_rate,
            "patch_positive_rate": patch_positive_rate,
            "windows_with_positive_ratio": _rate(windows_with_positive, num_windows),
            "padded_window_ratio": _rate(padded_windows, num_windows),
            "padded_point_ratio": _rate(padded_point_units, total_point_units),
            "suggested_anomaly_pos_weight": suggested_anomaly_pos_weight,
            "point_positive_units": int(point_positive),
            "point_total_units": int(point_total),
            "patch_positive_units": int(patch_positive),
            "patch_total_units": int(patch_total),
            "positive_feature_coverage_ratio": positive_feature_coverage_ratio,
            "feature_rate_unavailable_due_to_mixed_dims": bool(feature_rate_unavailable),
            "error_examples": error_examples,
        }
        stats.update(_distribution_stats(lengths, prefix="length"))
        stats.update(_distribution_stats(samples_mean, prefix="sample_mean"))
        stats.update(_distribution_stats(samples_std, prefix="sample_std"))
        stats.update(_distribution_stats(samples_abs_max, prefix="sample_abs_max"))
        stats.update(_distribution_stats(windows_per_sample, prefix="windows_per_sample"))
        stats.update(_distribution_stats(positive_points_per_sample, prefix="positive_points_per_sample"))
        stats.update(_distribution_stats(normal_series_abs_diff_mean, prefix="normal_series_abs_diff_mean"))
        stats.update(feature_point_positive_rate_stats)

        issues = self._derive_issues(stats, split=split, expected_training_split=expected_training_split)
        quality_score = self._score_from_issues(issues)
        quality_grade = _score_to_grade(quality_score)
        has_blocking_issues = any(issue.severity == "error" for issue in issues)

        return SplitQualityReport(
            split=split,
            expected_training_split=expected_training_split,
            stats=stats,
            issues=issues,
            quality_score=quality_score,
            quality_grade=quality_grade,
            has_blocking_issues=has_blocking_issues,
        )

    def _score_from_issues(self, issues: list[QualityIssue]) -> float:
        penalties = {
            "error": 25.0,
            "warning": 10.0,
            "info": 2.0,
        }
        score = 100.0
        for issue in issues:
            score -= penalties.get(issue.severity, 0.0)
        return float(max(0.0, score))

    def _derive_issues(
        self,
        stats: Mapping[str, Any],
        *,
        split: str,
        expected_training_split: str | None,
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        thresholds = self.thresholds

        def add_issue(
            *,
            code: str,
            severity: str,
            message: str,
            metric: str | None = None,
            value: float | None = None,
            threshold: float | None = None,
        ) -> None:
            issues.append(
                QualityIssue(
                    code=code,
                    severity=severity,
                    message=message,
                    metric=metric,
                    value=value,
                    threshold=threshold,
                )
            )

        if expected_training_split is not None and split != expected_training_split:
            add_issue(
                code="inspecting_non_training_split",
                severity="info",
                message=(
                    f"Inspecting split '{split}' while configured training split is "
                    f"'{expected_training_split}'."
                ),
            )

        analyzed_samples = int(stats.get("num_samples_analyzed", 0))
        if analyzed_samples <= 0:
            add_issue(
                code="no_samples_analyzed",
                severity="error",
                message="No samples were analyzed in this split.",
            )
            return issues

        if int(stats.get("num_samples_total", 0)) < thresholds.min_samples:
            add_issue(
                code="too_few_samples",
                severity="warning",
                message=(
                    f"Split has only {int(stats.get('num_samples_total', 0))} samples, which may "
                    "be too small for stable training."
                ),
                metric="num_samples_total",
                value=float(stats.get("num_samples_total", 0)),
                threshold=float(thresholds.min_samples),
            )

        if int(stats.get("num_feature_values_count", 0)) > thresholds.max_feature_dim_variants:
            add_issue(
                code="mixed_feature_dimensions",
                severity="error",
                message=(
                    "Multiple feature dimensions were found in one run. Current training path "
                    "expects a fixed D."
                ),
                metric="num_feature_values_count",
                value=float(stats.get("num_feature_values_count", 0)),
                threshold=float(thresholds.max_feature_dim_variants),
            )

        if float(stats.get("sample_loading_error_ratio", 0.0)) > 0.0:
            add_issue(
                code="sample_loading_errors",
                severity="error",
                message="Some samples could not be loaded or had invalid series shape.",
                metric="sample_loading_error_ratio",
                value=float(stats.get("sample_loading_error_ratio", 0.0)),
                threshold=0.0,
            )

        if float(stats.get("missing_point_mask_ratio", 0.0)) > 0.0:
            add_issue(
                code="missing_point_mask",
                severity="error",
                message="Some samples are missing point_mask, which is required for supervision.",
                metric="missing_point_mask_ratio",
                value=float(stats.get("missing_point_mask_ratio", 0.0)),
                threshold=0.0,
            )

        if float(stats.get("invalid_point_mask_ratio", 0.0)) > 0.0:
            add_issue(
                code="invalid_point_mask_shape",
                severity="error",
                message="Some point_mask tensors do not match the [T, D] series shape.",
                metric="invalid_point_mask_ratio",
                value=float(stats.get("invalid_point_mask_ratio", 0.0)),
                threshold=0.0,
            )

        if float(stats.get("window_transform_error_ratio", 0.0)) > 0.0:
            add_issue(
                code="window_transform_errors",
                severity="error",
                message="Some samples failed during windowization and cannot be used for training.",
                metric="window_transform_error_ratio",
                value=float(stats.get("window_transform_error_ratio", 0.0)),
                threshold=0.0,
            )

        nonfinite_ratio = float(stats.get("nonfinite_value_ratio", 0.0))
        if nonfinite_ratio > thresholds.max_nonfinite_value_ratio:
            add_issue(
                code="nonfinite_values",
                severity="error",
                message="NaN/Inf values were found in series data.",
                metric="nonfinite_value_ratio",
                value=nonfinite_ratio,
                threshold=thresholds.max_nonfinite_value_ratio,
            )

        point_mask_any_mismatch_ratio = float(stats.get("point_mask_any_mismatch_ratio", 0.0))
        if point_mask_any_mismatch_ratio > thresholds.max_point_mask_any_mismatch_ratio:
            add_issue(
                code="point_mask_any_mismatch",
                severity="warning",
                message="Provided point_mask_any disagrees with point_mask-derived labels.",
                metric="point_mask_any_mismatch_ratio",
                value=point_mask_any_mismatch_ratio,
                threshold=thresholds.max_point_mask_any_mismatch_ratio,
            )

        point_positive_rate = float(stats.get("point_positive_rate", 0.0))
        if point_positive_rate < thresholds.min_point_positive_rate:
            add_issue(
                code="low_point_positive_rate",
                severity="warning",
                message="Point-level anomaly positives are very sparse.",
                metric="point_positive_rate",
                value=point_positive_rate,
                threshold=thresholds.min_point_positive_rate,
            )

        patch_positive_rate = float(stats.get("patch_positive_rate", 0.0))
        if patch_positive_rate <= 0.0:
            add_issue(
                code="zero_patch_positives",
                severity="error",
                message="No positive patch labels detected. Supervised anomaly learning will fail.",
                metric="patch_positive_rate",
                value=patch_positive_rate,
                threshold=0.0,
            )
        elif patch_positive_rate < thresholds.min_patch_positive_rate:
            add_issue(
                code="low_patch_positive_rate",
                severity="warning",
                message="Patch-level positives are extremely sparse and may cause all-negative collapse.",
                metric="patch_positive_rate",
                value=patch_positive_rate,
                threshold=thresholds.min_patch_positive_rate,
            )

        zero_positive_ratio = float(stats.get("zero_positive_sample_ratio", 0.0))
        if zero_positive_ratio > thresholds.max_zero_positive_sample_ratio:
            add_issue(
                code="too_many_all_negative_samples",
                severity="warning",
                message="Too many samples contain no positive labels.",
                metric="zero_positive_sample_ratio",
                value=zero_positive_ratio,
                threshold=thresholds.max_zero_positive_sample_ratio,
            )

        short_sequence_ratio = float(stats.get("short_sequence_ratio", 0.0))
        if short_sequence_ratio > thresholds.max_short_sequence_ratio:
            add_issue(
                code="too_many_short_sequences",
                severity="warning",
                message="A large fraction of samples is shorter than context_size.",
                metric="short_sequence_ratio",
                value=short_sequence_ratio,
                threshold=thresholds.max_short_sequence_ratio,
            )

        padded_point_ratio = float(stats.get("padded_point_ratio", 0.0))
        if padded_point_ratio > thresholds.max_padded_point_ratio:
            add_issue(
                code="excessive_padding",
                severity="warning",
                message="Padding occupies a large fraction of model input points.",
                metric="padded_point_ratio",
                value=padded_point_ratio,
                threshold=thresholds.max_padded_point_ratio,
            )

        positive_feature_coverage_ratio = stats.get("positive_feature_coverage_ratio")
        if (
            isinstance(positive_feature_coverage_ratio, (float, int))
            and positive_feature_coverage_ratio < thresholds.min_positive_feature_coverage_ratio
        ):
            add_issue(
                code="low_positive_feature_coverage",
                severity="warning",
                message="Only a small fraction of feature channels contains positive anomalies.",
                metric="positive_feature_coverage_ratio",
                value=float(positive_feature_coverage_ratio),
                threshold=thresholds.min_positive_feature_coverage_ratio,
            )

        near_constant_sample_ratio = float(stats.get("near_constant_sample_ratio", 0.0))
        if near_constant_sample_ratio > 0.2:
            add_issue(
                code="too_many_near_constant_samples",
                severity="warning",
                message="Many samples are nearly constant and may carry weak learning signal.",
                metric="near_constant_sample_ratio",
                value=near_constant_sample_ratio,
                threshold=0.2,
            )

        return issues


__all__ = [
    "DataQualityInspector",
    "DataQualityThresholds",
    "DatasetQualityReport",
    "QualityIssue",
    "SplitQualityReport",
]
