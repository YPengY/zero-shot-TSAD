"""Window slicing utilities that bridge raw sequences and model supervision.

The windowizer is the component that turns full sequences into fixed-size
training examples. It is also where point-level anomaly masks are converted
into patch-level labels, so this module defines the semantic boundary between
raw labels and model-facing supervision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import ContextWindowSample, ContextWindowizerProtocol, RawSample, SplitName


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
    """Slice full sequences into fixed-size, right-padded context windows.

    This class owns three related decisions:
    - how context bounds are enumerated from a full sequence
    - how short/tail fragments are preserved through right-padding
    - how point-level labels `[W, D]` are aggregated into patch labels

    It intentionally does not batch samples or read files; those concerns stay
    in the dataset and collator layers.
    """

    context_size: int
    patch_size: int
    stride: int | None = None
    pad_short_sequences: bool = True
    include_tail: bool = True

    def __post_init__(self) -> None:
        """Validate windowizer geometry and normalize the default stride."""

        if self.context_size <= 0:
            raise ValueError("`context_size` must be positive.")
        if self.patch_size <= 0:
            raise ValueError("`patch_size` must be positive.")
        if self.context_size % self.patch_size != 0:
            raise ValueError("`context_size` must be divisible by `patch_size`.")

        if self.stride is not None and self.stride <= 0:
            raise ValueError("`stride` must be positive.")

        if self.stride is None:
            self.stride = self.context_size

    def transform(self, sample: RawSample) -> tuple[ContextWindowSample, ...]:
        """Convert one raw sequence into fixed-size context windows."""

        return tuple(
            self.slice_window(sample, start=start, end=end)
            for start, end in self.iter_context_bounds(int(sample.series.shape[0]))
        )

    def slice_window(
        self,
        sample: RawSample,
        *,
        start: int,
        end: int,
    ) -> ContextWindowSample:
        """Slice one raw sample into a single fixed-size context window."""

        series, point_mask, point_mask_any, normal_series = self._prepare_sample(sample)
        if start < 0 or end <= start:
            raise ValueError(f"Invalid context bounds: start={start}, end={end}.")
        if end > series.shape[0]:
            raise ValueError(
                "`end` cannot exceed sequence length. "
                f"Got end={end} for sequence_length={series.shape[0]}."
            )
        if end - start > self.context_size:
            raise ValueError(
                "`end - start` cannot exceed `context_size`. "
                f"Got window_length={end - start}, context_size={self.context_size}."
            )

        return self.assemble_window(
            sample_id=sample.sample_id,
            split=sample.split,
            context_start=int(start),
            context_end=int(end),
            series_window=series[start:end],
            point_mask_window=point_mask[start:end],
            point_mask_any_window=point_mask_any[start:end],
            normal_series_window=(normal_series[start:end] if normal_series is not None else None),
        )

    def assemble_window(
        self,
        *,
        sample_id: str,
        split: SplitName,
        context_start: int,
        context_end: int,
        series_window: np.ndarray,
        point_mask_window: np.ndarray,
        point_mask_any_window: np.ndarray | None = None,
        normal_series_window: np.ndarray | None = None,
    ) -> ContextWindowSample:
        """Build one model-facing window from already-cropped arrays.

        The returned window always has padded shape `[context_size, D]`.
        `context_end - context_start` records the valid unpadded prefix, which
        later becomes `valid_lengths` and validity masks in the collator.
        """

        series = _as_float32_2d(series_window, name="series_window")
        point_mask = _as_mask_2d(point_mask_window, name="point_mask_window")
        if point_mask.shape != series.shape:
            raise ValueError(
                "`point_mask_window` shape must match `series_window` shape. "
                f"Got {point_mask.shape} vs {series.shape}."
            )
        if point_mask_any_window is None:
            point_mask_any = point_mask.max(axis=1).astype(np.uint8, copy=False)
        else:
            point_mask_any = _as_mask_1d(point_mask_any_window, name="point_mask_any_window")
            if point_mask_any.shape[0] != series.shape[0]:
                raise ValueError(
                    "`point_mask_any_window` length must match `series_window` length. "
                    f"Got {point_mask_any.shape[0]} vs {series.shape[0]}."
                )

        normal_series = None
        if normal_series_window is not None:
            normal_series = _as_float32_2d(normal_series_window, name="normal_series_window")
            if normal_series.shape != series.shape:
                raise ValueError(
                    "`normal_series_window` shape must match `series_window` shape. "
                    f"Got {normal_series.shape} vs {series.shape}."
                )

        window_length = int(series.shape[0])
        if window_length > self.context_size:
            raise ValueError(
                "`series_window` length cannot exceed `context_size`. "
                f"Got {window_length} vs {self.context_size}."
            )
        if int(context_end) - int(context_start) != window_length:
            raise ValueError(
                "`context_end - context_start` must match `series_window` length. "
                f"Got {context_end - context_start} vs {window_length}."
            )

        window_series = _slice_or_pad_2d(
            series,
            start=0,
            end=window_length,
            target_length=self.context_size,
            pad_value=0.0,
            dtype=np.float32,
        )
        window_point_mask = _slice_or_pad_2d(
            point_mask,
            start=0,
            end=window_length,
            target_length=self.context_size,
            pad_value=0,
            dtype=np.uint8,
        )
        window_point_mask_any = _slice_or_pad_1d(
            point_mask_any,
            start=0,
            end=window_length,
            target_length=self.context_size,
            pad_value=0,
        )
        window_normal_series = None
        if normal_series is not None:
            window_normal_series = _slice_or_pad_2d(
                normal_series,
                start=0,
                end=window_length,
                target_length=self.context_size,
                pad_value=0.0,
                dtype=np.float32,
            )

        return ContextWindowSample(
            sample_id=str(sample_id),
            split=split,
            context_start=int(context_start),
            context_end=int(context_end),
            series=window_series,
            patch_labels=_build_patch_labels(window_point_mask, self.patch_size),
            point_mask=window_point_mask,
            point_mask_any=window_point_mask_any,
            normal_series=window_normal_series,
        )

    def iter_context_bounds(self, sequence_length: int) -> tuple[tuple[int, int], ...]:
        """Generate `(start, end)` windows over one sequence.

        Behavior:
        - Full windows are emitted according to `stride`.
        - Short sequences are kept only when `pad_short_sequences=True`.
        - Tail remainders are kept only when `include_tail=True`.
        """

        if sequence_length <= 0:
            raise ValueError("`series` must contain at least one time step.")
        if sequence_length < self.context_size:
            if not self.pad_short_sequences:
                return ()
            return ((0, sequence_length),)

        assert self.stride is not None
        last_full_start = sequence_length - self.context_size
        starts = list(range(0, last_full_start + 1, self.stride))
        bounds = [(start, start + self.context_size) for start in starts]

        last_covered_end = bounds[-1][1] if bounds else 0
        if self.include_tail and last_covered_end < sequence_length:
            tail_start = starts[-1] + self.stride if starts else 0
            tail_start = min(tail_start, sequence_length - 1)
            tail_end = sequence_length
            if tail_start < tail_end:
                bounds.append((tail_start, tail_end))

        return tuple(bounds)

    def _prepare_sample(
        self,
        sample: RawSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Validate source arrays once before window slicing."""

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

        return series, point_mask, point_mask_any, normal_series


__all__ = ["SlidingContextWindowizer"]
