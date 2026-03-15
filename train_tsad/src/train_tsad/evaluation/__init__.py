"""Evaluation utilities for patch-feature and legacy point-level reporting."""

from .evaluator import PatchFeatureEvaluator, TimeRCDEvaluator
from .metrics import average_precision, compute_detection_metrics, find_best_threshold, precision_recall_f1
from .postprocess import (
    PatchFeatureAccumulator,
    PatchFeatureRecord,
    PointScoreAccumulator,
    patch_scores_to_point_scores,
    reduce_patch_scores,
)

__all__ = [
    "PatchFeatureAccumulator",
    "PatchFeatureEvaluator",
    "PatchFeatureRecord",
    "PointScoreAccumulator",
    "TimeRCDEvaluator",
    "average_precision",
    "compute_detection_metrics",
    "find_best_threshold",
    "patch_scores_to_point_scores",
    "precision_recall_f1",
    "reduce_patch_scores",
]
