from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def reduce_patch_scores(patch_scores: np.ndarray, *, reduction: str = "mean") -> np.ndarray:
    """Reduce per-feature patch scores `[N_patches, D]` to scalar patch scores."""

    scores = np.asarray(patch_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"`patch_scores` must have shape [N_patches, D], got {scores.shape}.")

    mode = reduction.lower()
    if mode == "mean":
        return scores.mean(axis=1)
    if mode == "max":
        return scores.max(axis=1)
    raise ValueError(f"Unsupported score reduction `{reduction}`. Supported: mean, max.")


def patch_scores_to_point_scores(patch_scores: np.ndarray, *, patch_size: int) -> np.ndarray:
    """Expand scalar patch scores `[N_patches]` to point scores `[W]`."""

    if patch_size <= 0:
        raise ValueError("`patch_size` must be positive.")
    scores = np.asarray(patch_scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(f"`patch_scores` must have shape [N_patches], got {scores.shape}.")
    return np.repeat(scores, patch_size)


@dataclass(slots=True)
class PointScoreAccumulator:
    """Aggregate overlapping window scores back to a full sample timeline."""

    aggregation: str = "mean"
    score_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32),
    )
    count_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32),
    )
    target_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.uint8),
    )
    observed_length: int = 0

    def update(
        self,
        *,
        start: int,
        end: int,
        point_scores: np.ndarray,
        point_targets: np.ndarray,
    ) -> None:
        if start < 0 or end < start:
            raise ValueError(f"Invalid context bounds: start={start}, end={end}.")

        valid_length = end - start
        scores = np.asarray(point_scores, dtype=np.float32)
        targets = np.asarray(point_targets, dtype=np.uint8)
        if scores.ndim != 1 or targets.ndim != 1:
            raise ValueError("`point_scores` and `point_targets` must be 1D arrays.")
        if scores.shape[0] < valid_length or targets.shape[0] < valid_length:
            raise ValueError(
                "Window arrays are shorter than the valid unpadded range. "
                f"Got scores={scores.shape[0]}, targets={targets.shape[0]}, valid_length={valid_length}."
            )

        self._ensure_length(end)
        window_scores = scores[:valid_length]
        window_targets = targets[:valid_length]

        if self.aggregation == "mean":
            self.score_buffer[start:end] += window_scores
            self.count_buffer[start:end] += 1
        elif self.aggregation == "max":
            self.score_buffer[start:end] = np.maximum(self.score_buffer[start:end], window_scores)
            self.count_buffer[start:end] = 1
        else:
            raise ValueError(
                f"Unsupported point score aggregation `{self.aggregation}`. Supported: mean, max."
            )

        self.target_buffer[start:end] = np.maximum(self.target_buffer[start:end], window_targets)
        self.observed_length = max(self.observed_length, end)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score_buffer[: self.observed_length].copy()
        counts = self.count_buffer[: self.observed_length]
        if self.aggregation == "mean":
            np.divide(scores, np.maximum(counts, 1), out=scores, where=np.maximum(counts, 1) > 0)
        targets = self.target_buffer[: self.observed_length].astype(np.bool_)
        return scores, targets

    def _ensure_length(self, length: int) -> None:
        if length <= self.score_buffer.shape[0]:
            return

        new_size = length
        new_scores = np.zeros((new_size,), dtype=np.float32)
        new_counts = np.zeros((new_size,), dtype=np.int32)
        new_targets = np.zeros((new_size,), dtype=np.uint8)

        old_size = self.score_buffer.shape[0]
        if old_size > 0:
            new_scores[:old_size] = self.score_buffer
            new_counts[:old_size] = self.count_buffer
            new_targets[:old_size] = self.target_buffer

        self.score_buffer = new_scores
        self.count_buffer = new_counts
        self.target_buffer = new_targets


__all__ = ["PointScoreAccumulator", "patch_scores_to_point_scores", "reduce_patch_scores"]
