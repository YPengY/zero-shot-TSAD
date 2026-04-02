from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..interfaces import DatasetProtocol, RawSample
from .windowizer import SlidingContextWindowizer


def _rate(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _distribution_stats(
    values: list[float] | list[int],
    *,
    prefix: str,
) -> dict[str, float | int | None]:
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


@dataclass(slots=True)
class _FeatureRateAccumulator:
    point_positive_by_feature: np.ndarray | None = None
    point_total_by_feature: np.ndarray | None = None
    unavailable_due_to_mixed_dims: bool = False

    def observe(self, point_mask_valid: np.ndarray) -> None:
        num_features = int(point_mask_valid.shape[1])
        if self.unavailable_due_to_mixed_dims:
            return
        if self.point_positive_by_feature is None or self.point_total_by_feature is None:
            self.point_positive_by_feature = np.zeros((num_features,), dtype=np.int64)
            self.point_total_by_feature = np.zeros((num_features,), dtype=np.int64)
        elif self.point_positive_by_feature.shape[0] != num_features:
            self.unavailable_due_to_mixed_dims = True
            self.point_positive_by_feature = None
            self.point_total_by_feature = None
            return

        self.point_positive_by_feature += np.asarray(point_mask_valid.sum(axis=0), dtype=np.int64)
        self.point_total_by_feature += int(point_mask_valid.shape[0])

    def summarize(self) -> tuple[float | None, dict[str, float | None], bool]:
        if (
            self.point_positive_by_feature is None
            or self.point_total_by_feature is None
            or self.point_total_by_feature.size == 0
        ):
            return (
                None,
                {
                    "feature_point_positive_rate_min": None,
                    "feature_point_positive_rate_p50": None,
                    "feature_point_positive_rate_max": None,
                },
                bool(self.unavailable_due_to_mixed_dims),
            )

        feature_rates = self.point_positive_by_feature / np.maximum(self.point_total_by_feature, 1)
        coverage_ratio = _rate(int((feature_rates > 0).sum()), int(feature_rates.size))
        return (
            coverage_ratio,
            {
                "feature_point_positive_rate_min": float(np.min(feature_rates)),
                "feature_point_positive_rate_p50": float(np.percentile(feature_rates, 50)),
                "feature_point_positive_rate_max": float(np.max(feature_rates)),
            },
            bool(self.unavailable_due_to_mixed_dims),
        )


@dataclass(slots=True)
class SplitStatisticsAccumulator:
    """Accumulate sample-level and window-level statistics for one split."""

    windowizer: SlidingContextWindowizer
    lengths: list[int] = field(default_factory=list)
    feature_counts: list[int] = field(default_factory=list)
    samples_mean: list[float] = field(default_factory=list)
    samples_std: list[float] = field(default_factory=list)
    samples_abs_max: list[float] = field(default_factory=list)
    windows_per_sample: list[int] = field(default_factory=list)
    positive_points_per_sample: list[int] = field(default_factory=list)
    normal_series_abs_diff_mean: list[float] = field(default_factory=list)
    error_examples: list[str] = field(default_factory=list)
    feature_rates: _FeatureRateAccumulator = field(default_factory=_FeatureRateAccumulator)
    point_positive: int = 0
    point_total: int = 0
    patch_positive: int = 0
    patch_total: int = 0
    num_windows: int = 0
    windows_with_positive: int = 0
    short_sequences: int = 0
    zero_positive_samples: int = 0
    missing_point_mask_samples: int = 0
    invalid_point_mask_samples: int = 0
    invalid_point_mask_any_samples: int = 0
    invalid_normal_series_samples: int = 0
    samples_with_nonfinite: int = 0
    near_constant_samples: int = 0
    normal_series_available_samples: int = 0
    sample_loading_errors: int = 0
    window_transform_errors: int = 0
    nan_values: int = 0
    inf_values: int = 0
    total_values: int = 0
    padded_point_units: int = 0
    total_point_units: int = 0
    padded_windows: int = 0
    point_mask_any_mismatch_points: int = 0
    point_mask_any_total_points: int = 0

    def remember_error(self, message: str) -> None:
        if len(self.error_examples) < 5:
            self.error_examples.append(message)

    def add_load_error(self, sample_index: int, exc: Exception) -> None:
        self.sample_loading_errors += 1
        self.remember_error(f"load[{sample_index}]: {type(exc).__name__}: {exc}")

    def add_invalid_series(self, sample_index: int, sample_id: str, shape: tuple[int, ...]) -> None:
        self.sample_loading_errors += 1
        self.remember_error(
            f"shape[{sample_index}]: invalid series shape {shape} for sample_id={sample_id}"
        )

    def observe_series(self, series: np.ndarray) -> tuple[int, int]:
        sample_length = int(series.shape[0])
        num_features = int(series.shape[1])
        self.lengths.append(sample_length)
        self.feature_counts.append(num_features)
        if sample_length < self.windowizer.context_size:
            self.short_sequences += 1

        finite_mask = np.isfinite(series)
        self.total_values += int(series.size)
        sample_nan = int(np.isnan(series).sum())
        sample_inf = int(np.isinf(series).sum())
        self.nan_values += sample_nan
        self.inf_values += sample_inf
        if sample_nan > 0 or sample_inf > 0:
            self.samples_with_nonfinite += 1

        finite_values = series[finite_mask]
        if finite_values.size > 0:
            sample_mean_value = float(np.mean(finite_values))
            sample_std_value = float(np.std(finite_values))
            sample_abs_max_value = float(np.max(np.abs(finite_values)))
            self.samples_mean.append(sample_mean_value)
            self.samples_std.append(sample_std_value)
            self.samples_abs_max.append(sample_abs_max_value)
            if sample_std_value <= 1e-6:
                self.near_constant_samples += 1

        return sample_length, num_features

    def observe_point_mask(
        self,
        point_mask_raw: np.ndarray | None,
        *,
        series: np.ndarray,
    ) -> np.ndarray | None:
        if point_mask_raw is None:
            self.missing_point_mask_samples += 1
            return None

        point_mask = np.asarray(point_mask_raw)
        if point_mask.ndim != 2 or point_mask.shape != series.shape:
            self.invalid_point_mask_samples += 1
            return None

        point_mask_valid = (point_mask > 0).astype(np.uint8, copy=False)
        sample_positive_points = int(point_mask_valid.sum())
        self.positive_points_per_sample.append(sample_positive_points)
        self.point_positive += sample_positive_points
        self.point_total += int(point_mask_valid.size)
        if sample_positive_points == 0:
            self.zero_positive_samples += 1
        self.feature_rates.observe(point_mask_valid)
        return point_mask_valid

    def observe_point_mask_any(
        self,
        point_mask_any_raw: np.ndarray | None,
        *,
        point_mask_valid: np.ndarray | None,
        sample_length: int,
    ) -> None:
        if point_mask_any_raw is None:
            return

        point_mask_any = np.asarray(point_mask_any_raw)
        if point_mask_any.ndim != 1 or point_mask_any.shape[0] != sample_length:
            self.invalid_point_mask_any_samples += 1
            return

        if point_mask_valid is not None:
            derived_point_mask_any = (point_mask_valid.sum(axis=1) > 0).astype(np.uint8, copy=False)
            provided_point_mask_any = (point_mask_any > 0).astype(np.uint8, copy=False)
            mismatches = int(np.not_equal(provided_point_mask_any, derived_point_mask_any).sum())
            self.point_mask_any_mismatch_points += mismatches
            self.point_mask_any_total_points += sample_length

    def observe_normal_series(
        self, normal_series_raw: np.ndarray | None, *, series: np.ndarray
    ) -> None:
        if normal_series_raw is None:
            return

        self.normal_series_available_samples += 1
        normal_series = np.asarray(normal_series_raw, dtype=np.float32)
        if normal_series.shape != series.shape:
            self.invalid_normal_series_samples += 1
            return

        diff = np.abs(normal_series - series)
        finite_diff = diff[np.isfinite(diff)]
        if finite_diff.size > 0:
            self.normal_series_abs_diff_mean.append(float(np.mean(finite_diff)))

    def observe_windows(self, sample: RawSample, sample_index: int) -> None:
        try:
            windows = self.windowizer.transform(sample)
        except Exception as exc:
            self.window_transform_errors += 1
            self.remember_error(f"window[{sample_index}]: {type(exc).__name__}: {exc}")
            self.windows_per_sample.append(0)
            return

        sample_windows = int(len(windows))
        self.windows_per_sample.append(sample_windows)
        for window in windows:
            patch_labels = np.asarray(window.patch_labels, dtype=np.uint8)
            positive_patch_units = int((patch_labels > 0).sum())
            self.patch_positive += positive_patch_units
            self.patch_total += int(patch_labels.size)
            if positive_patch_units > 0:
                self.windows_with_positive += 1

            self.num_windows += 1
            effective_length = int(window.context_end) - int(window.context_start)
            effective_length = max(0, min(effective_length, self.windowizer.context_size))
            padded_steps = self.windowizer.context_size - effective_length
            if padded_steps > 0:
                self.padded_windows += 1
            self.padded_point_units += padded_steps * int(window.series.shape[1])
            self.total_point_units += self.windowizer.context_size * int(window.series.shape[1])

    def to_stats(
        self,
        *,
        split: str,
        expected_training_split: str | None,
        num_samples_total: int,
        analyzed_samples: int,
    ) -> dict[str, Any]:
        unique_feature_counts = sorted(set(self.feature_counts))
        point_positive_rate = _rate(self.point_positive, self.point_total)
        patch_positive_rate = _rate(self.patch_positive, self.patch_total)
        suggested_anomaly_pos_weight = _rate(
            self.patch_total - self.patch_positive,
            self.patch_positive if self.patch_positive > 0 else 1,
        )

        (
            positive_feature_coverage_ratio,
            feature_point_positive_rate_stats,
            feature_rate_unavailable,
        ) = self.feature_rates.summarize()

        stats: dict[str, Any] = {
            "split": split,
            "num_samples_total": int(num_samples_total),
            "num_samples_analyzed": int(analyzed_samples),
            "analyzed_fraction": _rate(analyzed_samples, max(num_samples_total, 1)),
            "num_windows": int(self.num_windows),
            "num_feature_values": ",".join(str(value) for value in unique_feature_counts),
            "num_feature_values_count": int(len(unique_feature_counts)),
            "is_training_split": bool(
                expected_training_split is not None and split == expected_training_split
            ),
            "short_sequence_ratio": _rate(self.short_sequences, analyzed_samples),
            "zero_positive_sample_ratio": _rate(self.zero_positive_samples, analyzed_samples),
            "missing_point_mask_ratio": _rate(self.missing_point_mask_samples, analyzed_samples),
            "invalid_point_mask_ratio": _rate(self.invalid_point_mask_samples, analyzed_samples),
            "invalid_point_mask_any_ratio": _rate(
                self.invalid_point_mask_any_samples,
                analyzed_samples,
            ),
            "invalid_normal_series_ratio": _rate(
                self.invalid_normal_series_samples,
                analyzed_samples,
            ),
            "sample_loading_error_ratio": _rate(self.sample_loading_errors, analyzed_samples),
            "window_transform_error_ratio": _rate(self.window_transform_errors, analyzed_samples),
            "samples_with_nonfinite_ratio": _rate(self.samples_with_nonfinite, analyzed_samples),
            "nonfinite_value_ratio": _rate(self.nan_values + self.inf_values, self.total_values),
            "near_constant_sample_ratio": _rate(self.near_constant_samples, analyzed_samples),
            "point_mask_any_mismatch_ratio": _rate(
                self.point_mask_any_mismatch_points,
                self.point_mask_any_total_points,
            ),
            "normal_series_available_ratio": _rate(
                self.normal_series_available_samples,
                analyzed_samples,
            ),
            "point_positive_rate": point_positive_rate,
            "patch_positive_rate": patch_positive_rate,
            "windows_with_positive_ratio": _rate(self.windows_with_positive, self.num_windows),
            "padded_window_ratio": _rate(self.padded_windows, self.num_windows),
            "padded_point_ratio": _rate(self.padded_point_units, self.total_point_units),
            "suggested_anomaly_pos_weight": suggested_anomaly_pos_weight,
            "point_positive_units": int(self.point_positive),
            "point_total_units": int(self.point_total),
            "patch_positive_units": int(self.patch_positive),
            "patch_total_units": int(self.patch_total),
            "positive_feature_coverage_ratio": positive_feature_coverage_ratio,
            "feature_rate_unavailable_due_to_mixed_dims": bool(feature_rate_unavailable),
            "error_examples": list(self.error_examples),
        }
        stats.update(_distribution_stats(self.lengths, prefix="length"))
        stats.update(_distribution_stats(self.samples_mean, prefix="sample_mean"))
        stats.update(_distribution_stats(self.samples_std, prefix="sample_std"))
        stats.update(_distribution_stats(self.samples_abs_max, prefix="sample_abs_max"))
        stats.update(_distribution_stats(self.windows_per_sample, prefix="windows_per_sample"))
        stats.update(
            _distribution_stats(
                self.positive_points_per_sample,
                prefix="positive_points_per_sample",
            )
        )
        stats.update(
            _distribution_stats(
                self.normal_series_abs_diff_mean,
                prefix="normal_series_abs_diff_mean",
            )
        )
        stats.update(feature_point_positive_rate_stats)
        return stats


def collect_split_statistics(
    raw_dataset: DatasetProtocol,
    *,
    split: str,
    expected_training_split: str | None,
    windowizer: SlidingContextWindowizer,
    max_samples: int | None,
) -> dict[str, Any]:
    """Collect descriptive statistics for one raw split before training."""

    num_samples_total = int(len(raw_dataset))
    analyzed_samples = (
        num_samples_total if max_samples is None else min(num_samples_total, int(max_samples))
    )
    accumulator = SplitStatisticsAccumulator(windowizer=windowizer)

    for sample_index in range(analyzed_samples):
        try:
            sample = raw_dataset[sample_index]
        except Exception as exc:  # pragma: no cover - defensive guard
            accumulator.add_load_error(sample_index, exc)
            continue

        series = np.asarray(sample.series, dtype=np.float32)
        if series.ndim != 2 or series.shape[0] <= 0 or series.shape[1] <= 0:
            accumulator.add_invalid_series(sample_index, str(sample.sample_id), tuple(series.shape))
            continue

        sample_length, _ = accumulator.observe_series(series)
        point_mask_valid = accumulator.observe_point_mask(
            sample.point_mask,
            series=series,
        )
        accumulator.observe_point_mask_any(
            sample.point_mask_any,
            point_mask_valid=point_mask_valid,
            sample_length=sample_length,
        )
        accumulator.observe_normal_series(sample.normal_series, series=series)
        accumulator.observe_windows(sample, sample_index)

    return accumulator.to_stats(
        split=split,
        expected_training_split=expected_training_split,
        num_samples_total=num_samples_total,
        analyzed_samples=analyzed_samples,
    )


__all__ = ["collect_split_statistics"]
