from __future__ import annotations

from typing import Mapping

import numpy as np

from ..interfaces import DatasetProtocol
from .quality_rules import derive_quality_issues, score_from_issues, score_to_grade
from .quality_stats import collect_split_statistics
from .quality_types import (
    DataQualityThresholds,
    DatasetQualityReport,
    QualityIssue,
    SplitQualityReport,
)
from .windowizer import SlidingContextWindowizer


class DataQualityInspector:
    """Inspect raw datasets and emit structured pre-training quality reports.

    The inspector intentionally acts as a thin orchestration layer:
    - `quality_stats` collects descriptive statistics from raw samples
    - `quality_rules` converts those statistics into explicit issues and scores
    - this module preserves the stable public API consumed by training workflows
    """

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
        """Inspect multiple splits and aggregate them into one dataset-level report."""

        split_reports = {
            split: self.inspect_split(
                dataset,
                split=split,
                expected_training_split=expected_training_split,
            )
            for split, dataset in datasets_by_split.items()
        }

        overall_score = self._compute_overall_score(split_reports)
        return DatasetQualityReport(
            expected_training_split=expected_training_split,
            split_reports=split_reports,
            missing_splits=missing_splits or [],
            overall_score=overall_score,
            overall_grade=score_to_grade(overall_score),
            recommended_to_train=self._recommend_training(
                split_reports,
                expected_training_split=expected_training_split,
            ),
        )

    def inspect_split(
        self,
        raw_dataset: DatasetProtocol,
        *,
        split: str,
        expected_training_split: str | None = None,
    ) -> SplitQualityReport:
        """Inspect one split and return descriptive statistics plus rule-derived issues."""

        stats = collect_split_statistics(
            raw_dataset,
            split=split,
            expected_training_split=expected_training_split,
            windowizer=self.windowizer,
            max_samples=self.max_samples,
        )
        issues = derive_quality_issues(
            stats,
            split=split,
            expected_training_split=expected_training_split,
            thresholds=self.thresholds,
        )
        quality_score = score_from_issues(issues)
        return SplitQualityReport(
            split=split,
            expected_training_split=expected_training_split,
            stats=stats,
            issues=issues,
            quality_score=quality_score,
            quality_grade=score_to_grade(quality_score),
            has_blocking_issues=any(issue.severity == "error" for issue in issues),
        )

    @staticmethod
    def _compute_overall_score(split_reports: Mapping[str, SplitQualityReport]) -> float:
        if not split_reports:
            return 0.0
        return float(np.mean([report.quality_score for report in split_reports.values()], dtype=np.float64))

    @staticmethod
    def _recommend_training(
        split_reports: Mapping[str, SplitQualityReport],
        *,
        expected_training_split: str | None,
    ) -> bool:
        if expected_training_split is not None:
            train_report = split_reports.get(expected_training_split)
            if train_report is None:
                return False
            return (not train_report.has_blocking_issues) and train_report.quality_score >= 60.0
        return not any(report.has_blocking_issues for report in split_reports.values())


__all__ = [
    "DataQualityInspector",
    "DataQualityThresholds",
    "DatasetQualityReport",
    "QualityIssue",
    "SplitQualityReport",
]
