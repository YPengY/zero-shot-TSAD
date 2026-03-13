from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..interfaces import Batch, ModelOutput
from .metrics import compute_detection_metrics
from .postprocess import PointScoreAccumulator, patch_scores_to_point_scores, reduce_patch_scores


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
        self.sample_states.clear()


__all__ = ["TimeRCDEvaluator"]
