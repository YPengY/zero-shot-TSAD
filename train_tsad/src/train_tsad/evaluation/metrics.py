"""Thresholded detection metrics and threshold-search helpers.

These functions are intentionally dependency-light so both scripts and runtime
evaluators can compute the same thresholded metrics without pulling in a full
metrics framework.
"""

from __future__ import annotations

import numpy as np


def _as_flat_numpy(values) -> np.ndarray:
    """Convert scores/targets to contiguous 1D NumPy array view."""

    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


def precision_recall_f1(scores, targets, *, threshold: float) -> dict[str, float]:
    """Compute binary detection metrics at a fixed threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("`threshold` must be in [0, 1].")

    score_array = _as_flat_numpy(scores).astype(np.float32)
    target_array = _as_flat_numpy(targets).astype(np.bool_)
    if score_array.shape != target_array.shape:
        raise ValueError(
            f"`scores` and `targets` must have the same shape, got {score_array.shape} vs {target_array.shape}."
        )

    prediction_array = score_array >= threshold
    tp = int(np.logical_and(prediction_array, target_array).sum())
    fp = int(np.logical_and(prediction_array, np.logical_not(target_array)).sum())
    fn = int(np.logical_and(np.logical_not(prediction_array), target_array).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def average_precision(scores, targets) -> float:
    """Compute average precision without external metric dependencies."""

    score_array = _as_flat_numpy(scores).astype(np.float32)
    target_array = _as_flat_numpy(targets).astype(np.bool_)
    if score_array.shape != target_array.shape:
        raise ValueError(
            f"`scores` and `targets` must have the same shape, got {score_array.shape} vs {target_array.shape}."
        )

    positives = int(target_array.sum())
    if positives == 0:
        return 0.0

    order = np.argsort(-score_array, kind="mergesort")
    sorted_targets = target_array[order].astype(np.int64)

    tp = np.cumsum(sorted_targets)
    fp = np.cumsum(1 - sorted_targets)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives

    recall_previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_previous) * precision))


def find_best_threshold(
    scores,
    targets,
    *,
    metric: str = "f1",
    num_scan_thresholds: int = 256,
) -> dict[str, float]:
    """Search the threshold that maximizes one thresholded metric.

    The scan is performed over observed score values when they are few enough,
    otherwise over a dense grid in `[0, 1]`. This keeps threshold search stable
    without assuming access to external metric libraries.
    """

    metric_name = metric.lower()
    if metric_name not in {"f1", "precision", "recall"}:
        raise ValueError(f"Unsupported threshold-search metric `{metric}`.")

    score_array = _as_flat_numpy(scores).astype(np.float32)
    unique_scores = np.unique(score_array)
    if unique_scores.size <= num_scan_thresholds:
        thresholds = unique_scores
    else:
        thresholds = np.linspace(0.0, 1.0, num=num_scan_thresholds, dtype=np.float32)

    if thresholds.size == 0:
        thresholds = np.array([0.5], dtype=np.float32)

    best_metrics: dict[str, float] | None = None
    best_value = float("-inf")
    for threshold in thresholds:
        metrics = precision_recall_f1(score_array, targets, threshold=float(threshold))
        value = metrics[metric_name]
        if value > best_value:
            best_value = value
            best_metrics = metrics

    assert best_metrics is not None
    return best_metrics


def compute_detection_metrics(
    scores,
    targets,
    *,
    threshold: float,
    threshold_search: bool = False,
    threshold_search_metric: str = "f1",
) -> dict[str, float]:
    """Compute the thresholded metric bundle used by evaluators.

    When `threshold_search=True`, the reported threshold-dependent metrics are
    taken at the threshold that optimizes `threshold_search_metric`; otherwise
    the provided fixed threshold is used as-is.
    """

    if threshold_search:
        metrics = find_best_threshold(scores, targets, metric=threshold_search_metric)
    else:
        metrics = precision_recall_f1(scores, targets, threshold=threshold)

    metrics["pr_auc"] = average_precision(scores, targets)
    metrics["num_points"] = float(_as_flat_numpy(targets).shape[0])
    metrics["num_positive_points"] = float(_as_flat_numpy(targets).astype(np.bool_).sum())
    return metrics


__all__ = [
    "average_precision",
    "compute_detection_metrics",
    "find_best_threshold",
    "precision_recall_f1",
]
