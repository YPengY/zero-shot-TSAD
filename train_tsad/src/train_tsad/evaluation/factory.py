from __future__ import annotations

from ..config import DataConfig, EvalConfig
from .evaluator import PatchFeatureEvaluator, TimeRCDEvaluator

Evaluator = PatchFeatureEvaluator | TimeRCDEvaluator


def build_evaluator(
    *,
    data_config: DataConfig,
    eval_config: EvalConfig,
    report_per_feature: bool | None = None,
    report_per_sample: bool | None = None,
) -> Evaluator:
    """Build the configured evaluator for validation or offline evaluation."""

    if eval_config.task == "patch_feature":
        return PatchFeatureEvaluator(
            patch_size=data_config.patch_size,
            patch_feature_score_aggregation=eval_config.patch_feature_score_aggregation,
            threshold=eval_config.threshold,
            threshold_search=eval_config.threshold_search,
            threshold_search_metric=eval_config.threshold_search_metric,
            report_per_feature=(
                eval_config.report_per_feature if report_per_feature is None else report_per_feature
            ),
            report_per_sample=(
                eval_config.report_per_sample if report_per_sample is None else report_per_sample
            ),
        )

    return TimeRCDEvaluator(
        patch_size=data_config.patch_size,
        score_reduction=eval_config.score_reduction,
        point_score_aggregation=eval_config.point_score_aggregation,
        threshold=eval_config.threshold,
        threshold_search=eval_config.threshold_search,
        threshold_search_metric=eval_config.threshold_search_metric,
    )
