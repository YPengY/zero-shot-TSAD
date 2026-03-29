"""Core runtime data structures shared across data, model, loss, and eval.

The training stack passes only a small set of typed objects between stages:
raw samples, fixed windows, tensor batches, model outputs, and loss outputs.
Keeping these contracts explicit is what allows the rest of the project to
stay modular.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import torch


SplitName = Literal["train", "val", "test"]
Metadata = dict[str, Any]


@dataclass(slots=True)
class RawSample:
    """A full sequence loaded from the synthetic generator outputs.

    Shape conventions:
    - `series`: [T, D]
    - `point_mask`: [T, D] when available
    - `point_mask_any`: [T] when available
    - `normal_series`: [T, D] when available

    `metadata` is an opaque pass-through payload from `synthetic_tsad`.
    Training code should only rely on the stable core fields above and treat
    `metadata` as optional supplemental context.
    """

    sample_id: str
    split: SplitName
    series: np.ndarray
    point_mask: np.ndarray | None = None
    point_mask_any: np.ndarray | None = None
    normal_series: np.ndarray | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ContextWindowSample:
    """A fixed context crop prepared for patch-level RCD supervision.

    Shape conventions:
    - `series`: [W, D]
    - `patch_labels`: [N_patches, D]
    - `point_mask`: [W, D] when available
    - `point_mask_any`: [W] when available
    - `normal_series`: [W, D] when available

    The sample still stores point-level masks so later stages can recover
    point-level anomaly scores from patch-level predictions.
    """

    sample_id: str
    split: SplitName
    context_start: int
    context_end: int
    series: np.ndarray
    patch_labels: np.ndarray
    point_mask: np.ndarray | None = None
    point_mask_any: np.ndarray | None = None
    normal_series: np.ndarray | None = None


@dataclass(slots=True)
class Batch:
    """Tensorized batch consumed by the model and loss functions.

    Shape conventions:
    - `inputs`: [B, W, D]
    - `valid_lengths`: [B]
    - `patch_labels`: [B, N_patches, D] when available
    - `reconstruction_targets`: shape defined by the masking/reconstruction path
    - `mask_indices`: boolean mask aligned with the reconstruction path
    - `point_valid_mask`: [B, W, D] marks non-padding points
    - `patch_valid_mask`: [B, N_patches, D] marks non-padding patch-feature units
    - `token_padding_mask`: [B, N_patches * D] for Transformer attention masking
    - `point_masks`: [B, W, D] when available
    - `point_mask_any`: [B, W] when available
    """

    sample_ids: list[str]
    context_start: "torch.Tensor"
    context_end: "torch.Tensor"
    valid_lengths: "torch.Tensor"
    inputs: "torch.Tensor"
    patch_labels: "torch.Tensor | None" = None
    reconstruction_targets: "torch.Tensor | None" = None
    mask_indices: "torch.Tensor | None" = None
    point_valid_mask: "torch.Tensor | None" = None
    patch_valid_mask: "torch.Tensor | None" = None
    token_padding_mask: "torch.Tensor | None" = None
    point_masks: "torch.Tensor | None" = None
    point_mask_any: "torch.Tensor | None" = None
    metadata: Metadata = field(default_factory=dict)

    def to(
        self,
        device: "str | torch.device",
        *,
        non_blocking: bool = False,
    ) -> Batch:
        """Return a copy with all tensor fields moved to `device`."""

        return replace(
            self,
            context_start=self.context_start.to(device, non_blocking=non_blocking),
            context_end=self.context_end.to(device, non_blocking=non_blocking),
            valid_lengths=self.valid_lengths.to(device, non_blocking=non_blocking),
            inputs=self.inputs.to(device, non_blocking=non_blocking),
            patch_labels=(
                self.patch_labels.to(device, non_blocking=non_blocking)
                if self.patch_labels is not None
                else None
            ),
            reconstruction_targets=(
                self.reconstruction_targets.to(device, non_blocking=non_blocking)
                if self.reconstruction_targets is not None
                else None
            ),
            mask_indices=(
                self.mask_indices.to(device, non_blocking=non_blocking)
                if self.mask_indices is not None
                else None
            ),
            point_valid_mask=(
                self.point_valid_mask.to(device, non_blocking=non_blocking)
                if self.point_valid_mask is not None
                else None
            ),
            patch_valid_mask=(
                self.patch_valid_mask.to(device, non_blocking=non_blocking)
                if self.patch_valid_mask is not None
                else None
            ),
            token_padding_mask=(
                self.token_padding_mask.to(device, non_blocking=non_blocking)
                if self.token_padding_mask is not None
                else None
            ),
            point_masks=(
                self.point_masks.to(device, non_blocking=non_blocking)
                if self.point_masks is not None
                else None
            ),
            point_mask_any=(
                self.point_mask_any.to(device, non_blocking=non_blocking)
                if self.point_mask_any is not None
                else None
            ),
        )


@dataclass(slots=True)
class ModelOutput:
    """Standardized model outputs for the paper-aligned training path.

    Shape conventions:
    - `logits`: [B, N_patches, D]
    - `point_logits`: [B, W, D] when anomaly scores are projected to observation space
    - `reconstruction`: must align with `Batch.reconstruction_targets` when used
    """

    logits: "torch.Tensor"
    point_logits: "torch.Tensor | None" = None
    reconstruction: "torch.Tensor | None" = None


@dataclass(slots=True)
class LossOutput:
    """Unified loss payload returned by loss modules."""

    loss: "torch.Tensor"
    metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for datasets that load full synthetic samples."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> RawSample:
        ...


@runtime_checkable
class ContextWindowizerProtocol(Protocol):
    """Protocol for slicing full sequences into context windows."""

    def iter_context_bounds(self, sequence_length: int) -> Sequence[tuple[int, int]]:
        ...

    def transform(self, sample: RawSample) -> Sequence[ContextWindowSample]:
        ...

    def slice_window(
        self,
        sample: RawSample,
        *,
        start: int,
        end: int,
    ) -> ContextWindowSample:
        ...


@runtime_checkable
class CollatorProtocol(Protocol):
    """Protocol for batching context windows into model-ready tensors."""

    def __call__(self, samples: Sequence[ContextWindowSample]) -> Batch:
        ...


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol shared by baseline and TimeRCD-style models."""

    def forward(self, batch: Batch) -> ModelOutput:
        ...


@runtime_checkable
class LossProtocol(Protocol):
    """Protocol for anomaly-only or multi-task loss functions."""

    def __call__(self, batch: Batch, output: ModelOutput) -> LossOutput:
        ...


__all__ = [
    "Batch",
    "CollatorProtocol",
    "ContextWindowSample",
    "ContextWindowizerProtocol",
    "DatasetProtocol",
    "LossOutput",
    "LossProtocol",
    "Metadata",
    "ModelOutput",
    "ModelProtocol",
    "RawSample",
    "SplitName",
]
