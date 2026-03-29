"""Score reduction and overlap aggregation for evaluation-time predictions.

These helpers convert model outputs from window-local tensors back into
sample-level score streams. They define how feature channels are reduced,
how patch scores are expanded to points, and how overlapping windows are
merged before thresholded metrics are computed.
"""

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


def reduce_point_feature_scores(point_scores: np.ndarray, *, reduction: str = "mean") -> np.ndarray:
    """Reduce per-feature point scores `[W, D]` to scalar point scores `[W]`."""

    scores = np.asarray(point_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"`point_scores` must have shape [W, D], got {scores.shape}.")

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


@dataclass(frozen=True, slots=True)
class PatchFeatureRecord:
    """One evaluated patch-feature unit anchored to an absolute sample region."""

    sample_id: str
    patch_start: int
    patch_end: int
    feature_index: int
    score: float
    target: bool


@dataclass(slots=True)
class _PatchFeatureSampleState:
    """Per-sample patch-feature buffers keyed by absolute patch start."""

    aggregation: str
    patch_size: int
    num_features: int
    score_buffer: dict[int, np.ndarray] = field(default_factory=dict)
    count_buffer: dict[int, int] = field(default_factory=dict)
    target_buffer: dict[int, np.ndarray] = field(default_factory=dict)
    observed_end: int = 0

    def update(
        self,
        *,
        context_start: int,
        context_end: int,
        patch_scores: np.ndarray,
        patch_targets: np.ndarray,
    ) -> None:
        """Merge one window into this sample state."""

        if context_start < 0 or context_end < context_start:
            raise ValueError(
                f"Invalid context bounds for patch aggregation: start={context_start}, end={context_end}."
            )

        scores = np.asarray(patch_scores, dtype=np.float32)
        targets = np.asarray(patch_targets, dtype=np.uint8)
        if scores.ndim != 2 or targets.ndim != 2:
            raise ValueError("`patch_scores` and `patch_targets` must have shape [N_patches, D].")
        if scores.shape != targets.shape:
            raise ValueError(
                "`patch_scores` and `patch_targets` must share the same shape. "
                f"Got {scores.shape} vs {targets.shape}."
            )
        if scores.shape[1] != self.num_features:
            raise ValueError(
                "`patch_scores.shape[1]` must match state feature count. "
                f"Got {scores.shape[1]} vs {self.num_features}."
            )

        targets_binary = (targets > 0).astype(np.uint8, copy=False)
        for patch_index in range(scores.shape[0]):
            patch_start = context_start + patch_index * self.patch_size
            if patch_start >= context_end:
                break

            score_row = scores[patch_index]
            target_row = targets_binary[patch_index]

            cached_scores = self.score_buffer.get(patch_start)
            if cached_scores is None:
                self.score_buffer[patch_start] = score_row.astype(np.float32, copy=True)
                self.target_buffer[patch_start] = target_row.copy()
                self.count_buffer[patch_start] = 1
                continue

            if self.aggregation == "mean":
                cached_scores += score_row
                self.count_buffer[patch_start] = self.count_buffer.get(patch_start, 0) + 1
            elif self.aggregation == "max":
                np.maximum(cached_scores, score_row, out=cached_scores)
            else:
                raise ValueError(
                    f"Unsupported patch-feature aggregation `{self.aggregation}`. "
                    "Supported: mean, max."
                )

            np.maximum(
                self.target_buffer[patch_start],
                target_row,
                out=self.target_buffer[patch_start],
            )

        self.observed_end = max(self.observed_end, int(context_end))

    def finalized_patch_rows(self) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
        """Return sorted `(patch_start, score_row, target_row)` rows."""

        rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        for patch_start in sorted(self.score_buffer):
            score_row = self.score_buffer[patch_start]
            if self.aggregation == "mean":
                score_row = score_row / max(self.count_buffer.get(patch_start, 1), 1)
            target_row = self.target_buffer[patch_start]
            rows.append((patch_start, score_row, target_row))
        return tuple(rows)


@dataclass(slots=True)
class PatchFeatureAccumulator:
    """Aggregate window-local patch scores into absolute sample patch regions.

    The accumulator keeps patch-feature units in absolute sample coordinates,
    which is what allows the evaluator to report patch metrics consistently
    even when the same region is visited by multiple overlapping windows.
    """

    aggregation: str = "mean"
    sample_states: dict[str, _PatchFeatureSampleState] = field(default_factory=dict)

    def update(
        self,
        *,
        sample_id: str,
        context_start: int,
        context_end: int,
        patch_scores: np.ndarray,
        patch_targets: np.ndarray,
        patch_size: int,
    ) -> None:
        """Merge one window's patch-feature predictions into absolute sample regions."""

        if patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")

        scores = np.asarray(patch_scores, dtype=np.float32)
        targets = np.asarray(patch_targets, dtype=np.uint8)
        if scores.ndim != 2 or targets.ndim != 2:
            raise ValueError("`patch_scores` and `patch_targets` must have shape [N_patches, D].")
        if scores.shape != targets.shape:
            raise ValueError(
                "`patch_scores` and `patch_targets` must share the same shape. "
                f"Got {scores.shape} vs {targets.shape}."
            )

        state = self.sample_states.get(sample_id)
        if state is None:
            state = _PatchFeatureSampleState(
                aggregation=self.aggregation,
                patch_size=patch_size,
                num_features=int(scores.shape[1]),
            )
            self.sample_states[sample_id] = state
        elif state.patch_size != patch_size:
            raise ValueError(
                "Patch size changed across updates for one sample. "
                f"Got {patch_size} vs {state.patch_size}."
            )

        state.update(
            context_start=context_start,
            context_end=context_end,
            patch_scores=scores,
            patch_targets=targets,
        )

    def finalize_arrays(
        self,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return dense arrays for fast metric computation."""

        sample_ids = sorted(self.sample_states)
        if not sample_ids:
            return (
                [],
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.bool_),
            )

        total_units = 0
        for sample_id in sample_ids:
            state = self.sample_states[sample_id]
            total_units += len(state.score_buffer) * state.num_features

        sample_indices = np.zeros((total_units,), dtype=np.int32)
        feature_indices = np.zeros((total_units,), dtype=np.int32)
        scores = np.zeros((total_units,), dtype=np.float32)
        targets = np.zeros((total_units,), dtype=np.bool_)

        offset = 0
        for sample_index, sample_id in enumerate(sample_ids):
            state = self.sample_states[sample_id]
            feature_row = np.arange(state.num_features, dtype=np.int32)
            for _, score_row, target_row in state.finalized_patch_rows():
                next_offset = offset + state.num_features
                sample_indices[offset:next_offset] = sample_index
                feature_indices[offset:next_offset] = feature_row
                scores[offset:next_offset] = score_row.astype(np.float32, copy=False)
                targets[offset:next_offset] = target_row.astype(np.bool_, copy=False)
                offset = next_offset

        return sample_ids, sample_indices, feature_indices, scores, targets

    def finalize(self) -> tuple[PatchFeatureRecord, ...]:
        """Return finalized patch-feature records sorted by sample, region, and feature."""

        records: list[PatchFeatureRecord] = []
        for sample_id in sorted(self.sample_states):
            state = self.sample_states[sample_id]
            for patch_start, score_row, target_row in state.finalized_patch_rows():
                patch_end = min(state.observed_end, patch_start + state.patch_size)
                for feature_index in range(state.num_features):
                    records.append(
                        PatchFeatureRecord(
                            sample_id=sample_id,
                            patch_start=int(patch_start),
                            patch_end=int(patch_end),
                            feature_index=feature_index,
                            score=float(score_row[feature_index]),
                            target=bool(target_row[feature_index]),
                        )
                    )
        return tuple(records)


@dataclass(slots=True)
class PointScoreAccumulator:
    """Aggregate overlapping window scores back to one point timeline.

    This is the point-level analogue of `PatchFeatureAccumulator`: it merges
    overlapping windows into one score per absolute time step and one boolean
    target per point before metric computation.
    """

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
        """Merge one window's point scores/targets into global sample buffers."""

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
        """Return finalized full-length point scores and boolean targets."""

        scores = self.score_buffer[: self.observed_length].copy()
        counts = self.count_buffer[: self.observed_length]
        if self.aggregation == "mean":
            np.divide(scores, np.maximum(counts, 1), out=scores, where=np.maximum(counts, 1) > 0)
        targets = self.target_buffer[: self.observed_length].astype(np.bool_)
        return scores, targets

    def _ensure_length(self, length: int) -> None:
        """Grow internal buffers to cover at least `length` points."""

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


__all__ = [
    "PatchFeatureAccumulator",
    "PatchFeatureRecord",
    "PointScoreAccumulator",
    "patch_scores_to_point_scores",
    "reduce_point_feature_scores",
    "reduce_patch_scores",
]
