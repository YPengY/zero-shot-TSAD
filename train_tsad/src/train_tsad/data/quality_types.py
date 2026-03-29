from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


__all__ = [
    "DataQualityThresholds",
    "DatasetQualityReport",
    "QualityIssue",
    "SplitQualityReport",
]
