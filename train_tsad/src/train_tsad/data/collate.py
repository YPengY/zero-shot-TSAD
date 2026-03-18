from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..interfaces import Batch, CollatorProtocol, ContextWindowSample
from .transforms import RandomPatchMaskingTransform


def _stack_required(
    arrays: Sequence[np.ndarray],
    *,
    name: str,
    expected_ndim: int,
) -> np.ndarray:
    """Validate and stack required arrays into `[B, ...]`."""

    if not arrays:
        raise ValueError(f"`{name}` requires at least one array.")

    first_shape = arrays[0].shape
    for index, array in enumerate(arrays):
        if array.ndim != expected_ndim:
            raise ValueError(
                f"`{name}` arrays must have ndim={expected_ndim}, got "
                f"ndim={array.ndim} at index={index}."
            )
        if array.shape != first_shape:
            raise ValueError(
                f"`{name}` arrays must share the same shape. "
                f"Expected {first_shape}, got {array.shape} at index={index}."
            )

    return np.stack(arrays, axis=0)


def _stack_optional(
    arrays: Sequence[np.ndarray | None],
    *,
    name: str,
    expected_ndim: int,
) -> np.ndarray | None:
    """Stack optional arrays only when all samples provide them."""

    present = [array is not None for array in arrays]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"`{name}` must be present for every sample in the batch or none.")

    return _stack_required(
        [array for array in arrays if array is not None],
        name=name,
        expected_ndim=expected_ndim,
    )


def _valid_lengths_from_samples(samples: Sequence[ContextWindowSample]) -> np.ndarray:
    """Return per-sample valid point counts before right padding."""

    valid_lengths = np.asarray(
        [max(0, int(sample.context_end) - int(sample.context_start)) for sample in samples],
        dtype=np.int64,
    )
    if valid_lengths.size == 0:
        raise ValueError("`samples` must contain at least one context window.")
    return valid_lengths


def _build_point_valid_mask(
    valid_lengths: np.ndarray,
    *,
    context_size: int,
    num_features: int,
) -> np.ndarray:
    """Build point-level validity mask `[B, W, D]` from valid lengths."""

    point_valid_mask = np.zeros((valid_lengths.shape[0], context_size, num_features), dtype=np.bool_)
    for batch_index, valid_length in enumerate(valid_lengths.tolist()):
        if valid_length <= 0:
            continue
        point_valid_mask[batch_index, :valid_length, :] = True
    return point_valid_mask


def _build_patch_valid_mask(
    valid_lengths: np.ndarray,
    *,
    context_size: int,
    patch_size: int,
    num_features: int,
) -> np.ndarray:
    """Build patch-level validity mask `[B, N_patches, D]` from valid lengths."""

    if context_size % patch_size != 0:
        raise ValueError("`context_size` must be divisible by `patch_size`.")

    num_patches = context_size // patch_size
    patch_valid_mask = np.zeros((valid_lengths.shape[0], num_patches, num_features), dtype=np.bool_)
    for batch_index, valid_length in enumerate(valid_lengths.tolist()):
        valid_patches = int(np.ceil(valid_length / patch_size)) if valid_length > 0 else 0
        if valid_patches <= 0:
            continue
        patch_valid_mask[batch_index, :valid_patches, :] = True
    return patch_valid_mask


def _normalize_series_array(
    series_array: np.ndarray,
    *,
    valid_lengths: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-sample per-feature z-score normalization over valid points only."""

    normalized = np.empty_like(series_array, dtype=np.float32)
    means = np.zeros((series_array.shape[0], 1, series_array.shape[2]), dtype=np.float32)
    stds = np.ones((series_array.shape[0], 1, series_array.shape[2]), dtype=np.float32)

    for batch_index, valid_length in enumerate(valid_lengths.tolist()):
        if valid_length <= 0:
            normalized[batch_index] = 0.0
            continue

        valid_view = series_array[batch_index, :valid_length, :]
        mean = valid_view.mean(axis=0, keepdims=True, dtype=np.float32)
        std = valid_view.std(axis=0, keepdims=True, dtype=np.float32)
        std = np.maximum(std, np.float32(eps)).astype(np.float32, copy=False)

        normalized_sample = (series_array[batch_index] - mean) / std
        if valid_length < series_array.shape[1]:
            normalized_sample[valid_length:, :] = 0.0

        normalized[batch_index] = normalized_sample.astype(np.float32, copy=False)
        means[batch_index, 0, :] = mean
        stds[batch_index, 0, :] = std

    return normalized, means, stds


@dataclass(slots=True)
class ContextWindowCollator(CollatorProtocol):
    """Convert fixed-size context windows into a model-ready tensor batch.

    This collator assumes the upstream windowizer has already aligned every
    sample to the same `context_size` and `patch_size`. It only validates shape
    consistency, stacks arrays, and converts them into torch tensors.
    """

    include_reconstruction_targets: bool = True
    masking_transform: RandomPatchMaskingTransform | None = None
    patch_size: int = 16
    normalization_mode: str = "none"
    normalization_eps: float = 1e-5

    def __call__(self, samples: Sequence[ContextWindowSample]) -> Batch:
        """Collate window samples into one `Batch`.

        Workflow:
        1. Validate batch consistency (non-empty and single split).
        2. Stack numpy arrays and convert to torch tensors.
        3. Build optional reconstruction/masking targets.
        4. Attach context bounds and shape metadata.
        """

        if not samples:
            raise ValueError("`samples` must contain at least one context window.")

        splits = {sample.split for sample in samples}
        if len(splits) != 1:
            raise ValueError(f"Mixed dataset splits are not allowed in one batch: {sorted(splits)}.")

        series_array = _stack_required(
            [np.asarray(sample.series, dtype=np.float32) for sample in samples],
            name="series",
            expected_ndim=2,
        )
        patch_labels_array = _stack_required(
            [np.asarray(sample.patch_labels, dtype=np.float32) for sample in samples],
            name="patch_labels",
            expected_ndim=2,
        )
        point_masks_array = _stack_optional(
            [sample.point_mask for sample in samples],
            name="point_mask",
            expected_ndim=2,
        )
        point_mask_any_array = _stack_optional(
            [sample.point_mask_any for sample in samples],
            name="point_mask_any",
            expected_ndim=1,
        )
        valid_lengths_array = _valid_lengths_from_samples(samples)

        if self.normalization_mode not in {"none", "per_sample_per_feature_zscore"}:
            raise ValueError(
                "`normalization_mode` must be one of: none, per_sample_per_feature_zscore."
            )
        if self.normalization_eps <= 0:
            raise ValueError("`normalization_eps` must be positive.")

        normalization_means = None
        normalization_stds = None
        if self.normalization_mode == "per_sample_per_feature_zscore":
            series_array, normalization_means, normalization_stds = _normalize_series_array(
                series_array,
                valid_lengths=valid_lengths_array,
                eps=self.normalization_eps,
            )

        point_valid_mask_array = _build_point_valid_mask(
            valid_lengths_array,
            context_size=int(series_array.shape[1]),
            num_features=int(series_array.shape[2]),
        )
        patch_valid_mask_array = _build_patch_valid_mask(
            valid_lengths_array,
            context_size=int(series_array.shape[1]),
            patch_size=self.patch_size,
            num_features=int(series_array.shape[2]),
        )

        inputs = torch.from_numpy(series_array)
        patch_labels = torch.from_numpy(patch_labels_array)
        valid_lengths = torch.from_numpy(valid_lengths_array.astype(np.int64, copy=False))
        point_valid_mask = torch.from_numpy(point_valid_mask_array)
        patch_valid_mask = torch.from_numpy(patch_valid_mask_array)
        token_padding_mask = torch.from_numpy(
            np.logical_not(patch_valid_mask_array).reshape(
                patch_valid_mask_array.shape[0],
                patch_valid_mask_array.shape[1] * patch_valid_mask_array.shape[2],
            )
        )
        point_masks = (
            torch.from_numpy(point_masks_array.astype(np.bool_, copy=False))
            if point_masks_array is not None
            else None
        )
        point_mask_any = (
            torch.from_numpy(point_mask_any_array.astype(np.bool_, copy=False))
            if point_mask_any_array is not None
            else None
        )
        reconstruction_targets = inputs.clone() if self.include_reconstruction_targets else None
        mask_indices = None
        if self.masking_transform is not None:
            # Masking transform keeps original inputs as reconstruction targets.
            masking_targets = self.masking_transform(
                inputs,
                valid_token_mask=patch_valid_mask,
            )
            reconstruction_targets = masking_targets.reconstruction_targets
            mask_indices = masking_targets.mask_indices
        else:
            masking_targets = None

        context_start = torch.tensor(
            [sample.context_start for sample in samples],
            dtype=torch.long,
        )
        context_end = torch.tensor(
            [sample.context_end for sample in samples],
            dtype=torch.long,
        )

        metadata = {
            "split": samples[0].split,
            "context_size": int(series_array.shape[1]),
            "num_features": int(series_array.shape[2]),
            "num_patches": int(patch_labels_array.shape[1]),
            "normalization_mode": self.normalization_mode,
        }
        if masking_targets is not None:
            metadata.update(masking_targets.metadata)
        if normalization_means is not None and normalization_stds is not None:
            metadata["normalization_mean_abs"] = float(np.mean(np.abs(normalization_means)))
            metadata["normalization_std_mean"] = float(np.mean(normalization_stds))

        return Batch(
            sample_ids=[sample.sample_id for sample in samples],
            context_start=context_start,
            context_end=context_end,
            valid_lengths=valid_lengths,
            inputs=inputs,
            patch_labels=patch_labels,
            reconstruction_targets=reconstruction_targets,
            mask_indices=mask_indices,
            point_valid_mask=point_valid_mask,
            patch_valid_mask=patch_valid_mask,
            token_padding_mask=token_padding_mask,
            point_masks=point_masks,
            point_mask_any=point_mask_any,
            metadata=metadata,
        )


__all__ = ["ContextWindowCollator"]
