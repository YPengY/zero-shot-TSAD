from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..interfaces import Batch, ModelOutput
from .metrics import compute_detection_metrics
from .postprocess import (
    PatchFeatureAccumulator,
    PatchFeatureRecord,
    PointScoreAccumulator,
    patch_scores_to_point_scores,
    reduce_patch_scores,
)


def _compute_group_metrics(
    records: list[PatchFeatureRecord],
    *,
    threshold: float,
    threshold_search: bool,
    threshold_search_metric: str,
) -> dict[str, float]:
    """Compute binary metrics for one patch-feature record subset."""

    scores = np.asarray([record.score for record in records], dtype=np.float32)
    targets = np.asarray([record.target for record in records], dtype=np.bool_)
    metrics = compute_detection_metrics(
        scores,
        targets,
        threshold=threshold,
        threshold_search=threshold_search,
        threshold_search_metric=threshold_search_metric,
    )
    metrics.pop("num_points", None)
    metrics.pop("num_positive_points", None)
    metrics["num_patch_feature_units"] = float(len(records))
    metrics["num_positive_patch_feature_units"] = float(targets.sum())
    return metrics


@dataclass(slots=True)
class PatchFeatureEvaluator:
    """Evaluate absolute sample patch regions for each feature channel."""

    patch_size: int
    patch_feature_score_aggregation: str = "mean"
    threshold: float = 0.5
    threshold_search: bool = False
    threshold_search_metric: str = "f1"
    report_per_feature: bool = False
    report_per_sample: bool = False
    accumulator: PatchFeatureAccumulator = field(init=False)

    def __post_init__(self) -> None:
        """Build the internal absolute patch accumulator."""

        self.accumulator = PatchFeatureAccumulator(
            aggregation=self.patch_feature_score_aggregation,
        )

    def update(self, batch: Batch, output: ModelOutput) -> None:
        """Accumulate patch-feature predictions aligned to absolute sample regions."""

        if batch.patch_labels is None:
            raise ValueError("Patch-feature evaluation requires `batch.patch_labels`.")

        probabilities = torch.sigmoid(output.logits).detach().cpu().numpy()
        patch_targets = batch.patch_labels.detach().cpu().numpy().astype(np.uint8)
        starts = batch.context_start.detach().cpu().numpy()
        ends = batch.context_end.detach().cpu().numpy()

        if probabilities.shape != patch_targets.shape:
            raise ValueError(
                "Patch-feature evaluation requires predictions and labels to share the same shape. "
                f"Got {probabilities.shape} vs {patch_targets.shape}."
            )

        for index, sample_id in enumerate(batch.sample_ids):
            self.accumulator.update(
                sample_id=sample_id,
                context_start=int(starts[index]),
                context_end=int(ends[index]),
                patch_scores=probabilities[index],
                patch_targets=patch_targets[index],
                patch_size=self.patch_size,
            )

    def compute(self) -> dict[str, Any]:
        """Finalize all patch-feature units and compute global/grouped metrics."""

        records = list(self.accumulator.finalize())
        if not records:
            raise ValueError("No patch-feature records were accumulated before `compute()`.")

        metrics = _compute_group_metrics(
            records,
            threshold=self.threshold,
            threshold_search=self.threshold_search,
            threshold_search_metric=self.threshold_search_metric,
        )
        metrics["task"] = "patch_feature"
        metrics["patch_feature_score_aggregation"] = self.patch_feature_score_aggregation
        metrics["num_samples"] = float(len({record.sample_id for record in records}))

        if self.report_per_feature:
            per_feature: dict[str, dict[str, float]] = {}
            for feature_index in sorted({record.feature_index for record in records}):
                feature_records = [record for record in records if record.feature_index == feature_index]
                per_feature[str(feature_index)] = _compute_group_metrics(
                    feature_records,
                    threshold=self.threshold,
                    threshold_search=self.threshold_search,
                    threshold_search_metric=self.threshold_search_metric,
                )
            metrics["per_feature"] = per_feature

        if self.report_per_sample:
            per_sample: dict[str, dict[str, float]] = {}
            for sample_id in sorted({record.sample_id for record in records}):
                sample_records = [record for record in records if record.sample_id == sample_id]
                per_sample[sample_id] = _compute_group_metrics(
                    sample_records,
                    threshold=self.threshold,
                    threshold_search=self.threshold_search,
                    threshold_search_metric=self.threshold_search_metric,
                )
            metrics["per_sample"] = per_sample

        return metrics


@dataclass(slots=True)
class TimeRCDEvaluator:
    """Accumulate window predictions and compute point-level detection metrics."""

    patch_size: int
    score_reduction: str = "mean"
    point_score_aggregation: str = "mean"
    threshold: float = 0.5
    threshold_search: bool = False
    threshold_search_metric: str = "f1"
    sample_states: dict[str, PointScoreAccumulator] = field(default_factory=dict)

    def update(self, batch: Batch, output: ModelOutput) -> None:
        """Accumulate one batch of window predictions.

        Workflow per sample:
        1. Convert logits to probabilities.
        2. Reduce per-feature patch scores to scalar patch scores.
        3. Expand patch scores to point scores.
        4. Merge window scores back into sample timeline accumulator.
        """

        if batch.point_mask_any is None:
            if batch.point_masks is None:
                raise ValueError("Evaluator requires `batch.point_mask_any` or `batch.point_masks`.")
            point_targets = batch.point_masks.any(dim=-1)
        else:
            point_targets = batch.point_mask_any

        probabilities = torch.sigmoid(output.logits).detach().cpu().numpy()
        target_array = point_targets.detach().cpu().numpy().astype(np.uint8)
        starts = batch.context_start.detach().cpu().numpy()
        ends = batch.context_end.detach().cpu().numpy()

        for index, sample_id in enumerate(batch.sample_ids):
            patch_scores = reduce_patch_scores(
                probabilities[index],
                reduction=self.score_reduction,
            )
            point_scores = patch_scores_to_point_scores(patch_scores, patch_size=self.patch_size)

            state = self.sample_states.setdefault(
                sample_id,
                PointScoreAccumulator(aggregation=self.point_score_aggregation),
            )
            state.update(
                start=int(starts[index]),
                end=int(ends[index]),
                point_scores=point_scores,
                point_targets=target_array[index],
            )

    def compute(self) -> dict[str, float]:
        """Finalize all accumulated samples and compute detection metrics."""

        if not self.sample_states:
            raise ValueError("No evaluation samples were accumulated before `compute()`.")

        sample_ids = sorted(self.sample_states)
        point_scores = []
        point_targets = []
        for sample_id in sample_ids:
            scores, targets = self.sample_states[sample_id].finalize()
            point_scores.append(scores)
            point_targets.append(targets)

        concatenated_scores = np.concatenate(point_scores, axis=0)
        concatenated_targets = np.concatenate(point_targets, axis=0)
        metrics = compute_detection_metrics(
            concatenated_scores,
            concatenated_targets,
            threshold=self.threshold,
            threshold_search=self.threshold_search,
            threshold_search_metric=self.threshold_search_metric,
        )
        metrics["num_samples"] = float(len(sample_ids))
        return metrics

    def reset(self) -> None:
        """Clear all accumulated state so evaluator can be reused."""

        self.sample_states.clear()

__all__ = ["PatchFeatureEvaluator", "TimeRCDEvaluator"]
