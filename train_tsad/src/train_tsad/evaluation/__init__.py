"""Public evaluation exports used by CLI, workflows, and tests.

Evaluation is organized around three concerns: score reduction back to sequence
space, metric computation and threshold search, and evaluator objects that tie
those pieces together for a given model/output style.
"""

from .evaluator import PatchFeatureEvaluator, TimeRCDEvaluator
from .factory import Evaluator, build_evaluator
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
    "Evaluator",
    "average_precision",
    "build_evaluator",
    "compute_detection_metrics",
    "find_best_threshold",
    "patch_scores_to_point_scores",
    "precision_recall_f1",
    "reduce_patch_scores",
]
