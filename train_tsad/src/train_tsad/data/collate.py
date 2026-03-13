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


@dataclass(slots=True)
class ContextWindowCollator(CollatorProtocol):
    """Convert fixed-size context windows into a model-ready tensor batch.

    This collator assumes the upstream windowizer has already aligned every
    sample to the same `context_size` and `patch_size`. It only validates shape
    consistency, stacks arrays, and converts them into torch tensors.
    """

    include_reconstruction_targets: bool = True
    masking_transform: RandomPatchMaskingTransform | None = None

    def __call__(self, samples: Sequence[ContextWindowSample]) -> Batch:
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

        inputs = torch.from_numpy(series_array)
        patch_labels = torch.from_numpy(patch_labels_array)
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
            masking_targets = self.masking_transform(inputs)
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
        }
        if masking_targets is not None:
            metadata.update(masking_targets.metadata)

        return Batch(
            sample_ids=[sample.sample_id for sample in samples],
            context_start=context_start,
            context_end=context_end,
            inputs=inputs,
            patch_labels=patch_labels,
            reconstruction_targets=reconstruction_targets,
            mask_indices=mask_indices,
            point_masks=point_masks,
            point_mask_any=point_mask_any,
            metadata=metadata,
        )


__all__ = ["ContextWindowCollator"]
