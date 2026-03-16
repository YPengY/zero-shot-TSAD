from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import ContextWindowSample, ContextWindowizerProtocol, RawSample


def _as_float32_2d(array: np.ndarray, *, name: str) -> np.ndarray:
    """Validate `[T, D]` numeric input and cast to float32."""

    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"`{name}` must have shape [T, D], got ndim={value.ndim}.")
    if value.shape[0] == 0:
        raise ValueError(f"`{name}` must contain at least one time step.")
    if value.shape[1] == 0:
        raise ValueError(f"`{name}` must contain at least one feature channel.")
    return value


def _as_mask_2d(array: np.ndarray, *, name: str) -> np.ndarray:
    """Validate mask shape `[T, D]` and convert to binary uint8."""

    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError(f"`{name}` must have shape [T, D], got ndim={value.ndim}.")
    return (value > 0).astype(np.uint8, copy=False)


def _as_mask_1d(array: np.ndarray, *, name: str) -> np.ndarray:
    """Validate mask shape `[T]` and convert to binary uint8."""

    value = np.asarray(array)
    if value.ndim != 1:
        raise ValueError(f"`{name}` must have shape [T], got ndim={value.ndim}.")
    return (value > 0).astype(np.uint8, copy=False)


def _slice_or_pad_2d(
    array: np.ndarray,
    *,
    start: int,
    end: int,
    target_length: int,
    pad_value: float | int,
    dtype: np.dtype[np.generic] | type[np.generic],
) -> np.ndarray:
    """Slice `[T, D]` region and right-pad to a fixed window length."""

    window = np.asarray(array[start:end], dtype=dtype)
    if window.shape[0] > target_length:
        raise ValueError("Window length cannot exceed target length.")
    if window.shape[0] == target_length:
        return np.ascontiguousarray(window)

    padded = np.full((target_length, array.shape[1]), pad_value, dtype=dtype)
    padded[: window.shape[0]] = window
    return padded


def _slice_or_pad_1d(
    array: np.ndarray,
    *,
    start: int,
    end: int,
    target_length: int,
    pad_value: int,
) -> np.ndarray:
    """Slice `[T]` region and right-pad to a fixed window length."""

    window = np.asarray(array[start:end], dtype=np.uint8)
    if window.shape[0] > target_length:
        raise ValueError("Window length cannot exceed target length.")
    if window.shape[0] == target_length:
        return np.ascontiguousarray(window)

    padded = np.full((target_length,), pad_value, dtype=np.uint8)
    padded[: window.shape[0]] = window
    return padded


def _build_patch_labels(point_mask: np.ndarray, patch_size: int) -> np.ndarray:
    """Aggregate point anomaly mask `[W, D]` into patch labels `[N_patches, D]`.

    Each patch is labeled anomalous when any point inside that patch is anomalous.
    """

    if point_mask.ndim != 2:
        raise ValueError("`point_mask` must have shape [W, D].")
    if point_mask.shape[0] % patch_size != 0:
        raise ValueError("Context length must be divisible by `patch_size`.")

    num_patches = point_mask.shape[0] // patch_size
    reshaped = point_mask.reshape(num_patches, patch_size, point_mask.shape[1])
    return reshaped.max(axis=1).astype(np.uint8, copy=False)


@dataclass(slots=True)
class SlidingContextWindowizer(ContextWindowizerProtocol):
    """Slice full sequences into fixed-size non-overlapping context windows.

    For compatibility the legacy `stride`, `pad_short_sequences`, and
    `include_tail` fields are still accepted, but the active behavior is:
    - short samples are discarded
    - only full `context_size` windows are kept
    - tail remainders are dropped instead of padded
    """

    context_size: int
    patch_size: int
    stride: int | None = None
    pad_short_sequences: bool = True
    include_tail: bool = True

    def __post_init__(self) -> None:
        """Validate windowizer geometry and normalize to strict chunking."""

        if self.context_size <= 0:
            raise ValueError("`context_size` must be positive.")
        if self.patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        if self.context_size % self.patch_size != 0:
            raise ValueError("`context_size` must be divisible by `patch_size`.")

        if self.stride is not None and self.stride <= 0:
            raise ValueError("`stride` must be positive.")

        # Training now uses strict full-window chunking only.
        self.stride = self.context_size
        self.pad_short_sequences = False
        self.include_tail = False

    def transform(self, sample: RawSample) -> tuple[ContextWindowSample, ...]:
        """Convert one raw sequence into fixed-size context windows.

        Workflow:
        1. Validate source arrays and align optional masks.
        2. Iterate full-window bounds from `_iter_context_bounds`.
        3. Slice per-window tensors and build patch-level labels.
        4. Return immutable tuple of `ContextWindowSample`.
        """

        series = _as_float32_2d(sample.series, name="series")
        point_mask = sample.point_mask
        if point_mask is None:
            raise ValueError("`sample.point_mask` is required to build patch labels.")
        point_mask = _as_mask_2d(point_mask, name="point_mask")
        if point_mask.shape != series.shape:
            raise ValueError(
                "`point_mask` shape must match `series` shape. "
                f"Got {point_mask.shape} vs {series.shape}."
            )

        point_mask_any = sample.point_mask_any
        if point_mask_any is None:
            point_mask_any = point_mask.max(axis=1).astype(np.uint8, copy=False)
        else:
            point_mask_any = _as_mask_1d(point_mask_any, name="point_mask_any")
            if point_mask_any.shape[0] != series.shape[0]:
                raise ValueError(
                    "`point_mask_any` length must match `series` length. "
                    f"Got {point_mask_any.shape[0]} vs {series.shape[0]}."
                )

        normal_series = sample.normal_series
        if normal_series is not None:
            normal_series = _as_float32_2d(normal_series, name="normal_series")
            if normal_series.shape != series.shape:
                raise ValueError(
                    "`normal_series` shape must match `series` shape. "
                    f"Got {normal_series.shape} vs {series.shape}."
                )

        windows: list[ContextWindowSample] = []
        for start, end in self._iter_context_bounds(series.shape[0]):
            # Window tensors are always fixed to `context_size` so collate can stack safely.
            window_series = _slice_or_pad_2d(
                series,
                start=start,
                end=end,
                target_length=self.context_size,
                pad_value=0.0,
                dtype=np.float32,
            )
            window_point_mask = _slice_or_pad_2d(
                point_mask,
                start=start,
                end=end,
                target_length=self.context_size,
                pad_value=0,
                dtype=np.uint8,
            )
            window_point_mask_any = _slice_or_pad_1d(
                point_mask_any,
                start=start,
                end=end,
                target_length=self.context_size,
                pad_value=0,
            )
            window_normal_series = None
            if normal_series is not None:
                window_normal_series = _slice_or_pad_2d(
                    normal_series,
                    start=start,
                    end=end,
                    target_length=self.context_size,
                    pad_value=0.0,
                    dtype=np.float32,
                )

            windows.append(
                ContextWindowSample(
                    sample_id=sample.sample_id,
                    split=sample.split,
                    context_start=start,
                    context_end=end,
                    series=window_series,
                    patch_labels=_build_patch_labels(window_point_mask, self.patch_size),
                    point_mask=window_point_mask,
                    point_mask_any=window_point_mask_any,
                    normal_series=window_normal_series,
                )
            )

        return tuple(windows)

    def _iter_context_bounds(self, sequence_length: int) -> tuple[tuple[int, int], ...]:
        """Generate `(start, end)` windows over one sequence.

        Behavior:
        - Sequences shorter than `context_size` are discarded.
        - Only non-overlapping full windows are kept.
        - Tail remainders shorter than `context_size` are dropped.
        """

        if sequence_length <= 0:
            raise ValueError("`series` must contain at least one time step.")
        if sequence_length < self.context_size:
            return ()

        usable_length = (sequence_length // self.context_size) * self.context_size
        return tuple(
            (start, start + self.context_size)
            for start in range(0, usable_length, self.context_size)
        )


__all__ = ["SlidingContextWindowizer"]
