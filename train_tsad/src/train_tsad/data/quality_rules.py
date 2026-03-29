from __future__ import annotations

from typing import Any, Mapping

from .quality_types import DataQualityThresholds, QualityIssue


def score_to_grade(score: float) -> str:
    """Convert one numeric quality score into a coarse human-readable grade."""

    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def score_from_issues(issues: list[QualityIssue]) -> float:
    """Assign a score penalty for each issue emitted by the rule engine."""

    penalties = {
        "error": 25.0,
        "warning": 10.0,
        "info": 2.0,
    }
    score = 100.0
    for issue in issues:
        score -= penalties.get(issue.severity, 0.0)
    return float(max(0.0, score))


def derive_quality_issues(
    stats: Mapping[str, Any],
    *,
    split: str,
    expected_training_split: str | None,
    thresholds: DataQualityThresholds,
) -> list[QualityIssue]:
    """Evaluate split statistics against explicit quality thresholds."""

    issues: list[QualityIssue] = []

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


__all__ = ["derive_quality_issues", "score_from_issues", "score_to_grade"]
