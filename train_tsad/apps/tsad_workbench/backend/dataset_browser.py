from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import resolve_packed_root


def iter_context_bounds(
    sequence_length: int,
    *,
    context_size: int,
    stride: int,
    include_tail: bool,
    pad_short_sequences: bool = True,
) -> list[tuple[int, int]]:
    """Return context-window bounds for one sequence."""

    if context_size <= 0:
        raise ValueError("`context_size` must be positive.")
    if stride <= 0:
        raise ValueError("`stride` must be positive.")
    if sequence_length <= 0:
        return []
    if sequence_length < context_size:
        return [(0, sequence_length)] if pad_short_sequences else []

    last_full_start = sequence_length - context_size
    starts = list(range(0, last_full_start + 1, stride))
    bounds = [(start, start + context_size) for start in starts]

    last_covered_end = bounds[-1][1] if bounds else 0
    if include_tail and last_covered_end < sequence_length:
        tail_start = starts[-1] + stride if starts else 0
        tail_start = min(tail_start, sequence_length - 1)
        if tail_start < sequence_length:
            bounds.append((tail_start, sequence_length))
    return bounds


def slice_or_pad_1d(
    array: np.ndarray,
    *,
    start: int,
    end: int,
    target_length: int,
    pad_value: float | int,
    dtype: np.dtype[np.generic] | type[np.generic],
) -> np.ndarray:
    """Slice one 1D array and right-pad it to a fixed length."""

    window = np.asarray(array[start:end], dtype=dtype)
    if window.shape[0] > target_length:
        raise ValueError("Window length cannot exceed target length.")
    if window.shape[0] == target_length:
        return np.ascontiguousarray(window)

    padded = np.full((target_length,), pad_value, dtype=dtype)
    padded[: window.shape[0]] = window
    return padded


def build_patch_labels_1d(point_mask: np.ndarray, *, patch_size: int) -> np.ndarray:
    """Aggregate a point mask into patch labels."""

    if patch_size <= 0:
        raise ValueError("`patch_size` must be positive.")
    trimmed_length = point_mask.shape[0] - (point_mask.shape[0] % patch_size)
    if trimmed_length <= 0:
        return np.zeros((0,), dtype=np.uint8)

    trimmed = np.asarray(point_mask[:trimmed_length], dtype=np.uint8)
    return (
        trimmed.reshape(trimmed_length // patch_size, patch_size)
        .max(axis=1)
        .astype(np.uint8, copy=False)
    )


def describe_window(
    *,
    start: int,
    end: int,
    sequence_length: int,
    context_size: int,
    patch_size: int,
) -> dict[str, int | bool]:
    """Describe the geometry of one context window."""

    effective_length = end - start
    padded_steps = max(0, context_size - effective_length)
    trimmed_length = context_size - (context_size % patch_size) if patch_size > 0 else 0
    total_patch_count = trimmed_length // patch_size if patch_size > 0 else 0
    valid_patch_count = min(effective_length, trimmed_length) // patch_size if patch_size > 0 else 0
    is_short_window = sequence_length < context_size
    is_tail_window = (
        sequence_length >= context_size
        and effective_length < context_size
        and end == sequence_length
    )
    return {
        "effective_length": int(effective_length),
        "padded_steps": int(padded_steps),
        "valid_patch_count": int(valid_patch_count),
        "total_patch_count": int(total_patch_count),
        "is_short_window": bool(is_short_window),
        "is_tail_window": bool(is_tail_window),
    }


def build_padding_mask(*, effective_length: int, context_size: int) -> np.ndarray:
    """Build a right-padding mask for one context window."""

    padding_mask = np.zeros((context_size,), dtype=np.uint8)
    if effective_length < context_size:
        padding_mask[effective_length:] = 1
    return padding_mask


def load_manifest_rows(packed_root: Path, split: str) -> list[dict[str, Any]]:
    """Load the split manifest rows from one packed dataset root."""

    manifest_path = packed_root / f"manifest.{split}.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_shard_metadata(shard_jsonl_path: Path) -> list[dict[str, Any]]:
    """Load optional shard-sidecar metadata rows."""

    if not shard_jsonl_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in shard_jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_sample_from_manifest_row(*, packed_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Load one packed sample or window-shard sample from a manifest row."""

    shard_npz_path = packed_root / str(row["shard_npz_path"])
    shard_jsonl_path_raw = row.get("shard_jsonl_path")
    debug_jsonl_path_raw = row.get("debug_jsonl_path")
    sample_index = int(row.get("sample_index", row.get("window_index", 0)))

    with np.load(shard_npz_path, allow_pickle=False) as npz:
        if "series_windows" in npz.files:
            series = np.asarray(npz["series_windows"][sample_index], dtype=np.float32)
            point_mask = np.asarray(npz["point_mask_windows"][sample_index], dtype=np.uint8)
            valid_lengths = np.asarray(npz["valid_lengths"], dtype=np.int32)
            if series.ndim != 2:
                raise ValueError(
                    f"Window shard sample must be 2D [T, D], got shape {series.shape}."
                )
            length = int(series.shape[0])
            dims = int(series.shape[1])
            valid_length = (
                int(valid_lengths[sample_index]) if sample_index < len(valid_lengths) else length
            )
            valid_length = max(0, min(valid_length, length))
            point_mask_any = np.zeros((length,), dtype=np.uint8)
            if valid_length > 0:
                point_mask_any[:valid_length] = (point_mask[:valid_length].sum(axis=1) > 0).astype(
                    np.uint8,
                    copy=False,
                )
            if valid_length < length:
                point_mask[valid_length:] = 0
            normal_series = np.zeros_like(series, dtype=np.float32)
        else:
            lengths = np.asarray(npz["lengths"], dtype=np.int32)
            num_series = np.asarray(npz["num_series"], dtype=np.int32)
            series_offsets = np.asarray(npz["series_offsets"], dtype=np.int64)
            time_offsets = np.asarray(npz["time_offsets"], dtype=np.int64)

            length = int(lengths[sample_index])
            dims = int(num_series[sample_index])
            flat_start = int(series_offsets[sample_index])
            flat_end = int(series_offsets[sample_index + 1])
            time_start = int(time_offsets[sample_index])
            time_end = int(time_offsets[sample_index + 1])

            series = np.asarray(
                npz["series_values"][flat_start:flat_end], dtype=np.float32
            ).reshape(length, dims)
            normal_series = np.asarray(
                npz["normal_series_values"][flat_start:flat_end], dtype=np.float32
            ).reshape(length, dims)
            point_mask = np.asarray(
                npz["point_mask_values"][flat_start:flat_end], dtype=np.uint8
            ).reshape(length, dims)
            point_mask_any = np.asarray(
                npz["point_mask_any_values"][time_start:time_end], dtype=np.uint8
            )

    metadata: dict[str, Any] = {}
    if debug_jsonl_path_raw is not None:
        debug_rows = load_shard_metadata(packed_root / str(debug_jsonl_path_raw))
        debug_row_index = int(row.get("debug_row_index", sample_index))
        if 0 <= debug_row_index < len(debug_rows):
            debug_row = debug_rows[debug_row_index]
            source_metadata = debug_row.get("source_metadata")
            metadata = source_metadata if isinstance(source_metadata, dict) else debug_row
    elif shard_jsonl_path_raw is not None:
        metadata_rows = load_shard_metadata(packed_root / str(shard_jsonl_path_raw))
        if sample_index < len(metadata_rows):
            maybe_metadata = metadata_rows[sample_index]
            if isinstance(maybe_metadata, dict):
                metadata = maybe_metadata

    return {
        "length": length,
        "num_series": dims,
        "series": series,
        "normal_series": normal_series,
        "point_mask": point_mask,
        "point_mask_any": point_mask_any,
        "metadata": metadata,
    }


def build_samples_payload(*, path_like: str, split: str, limit: int) -> dict[str, Any]:
    """Build the split sample list payload returned by `/api/samples`."""

    packed_root = resolve_packed_root(path_like)
    rows = load_manifest_rows(packed_root, split)[:limit]
    samples = [
        {
            "sample_id": str(row.get("sample_id")),
            "length": int(row.get("length", 0)),
            "num_series": int(row.get("num_series", 0)),
            "is_anomalous_sample": int(row.get("is_anomalous_sample", 0)),
            "num_events": int(row.get("num_events", 0)),
            "anomaly_point_ratio": float(row.get("anomaly_point_ratio", 0.0)),
        }
        for row in rows
    ]
    return {
        "packed_root": str(packed_root),
        "split": split,
        "num_samples": len(samples),
        "samples": samples,
    }


def build_sample_payload(
    *,
    path_like: str,
    split: str,
    sample_id: str,
    feature_index: int,
    slice_start: int,
    slice_end: int,
    context_size: int,
    stride: int,
    patch_size: int,
) -> dict[str, Any]:
    """Build the payload returned by `/api/sample`."""

    packed_root = resolve_packed_root(path_like)
    row = next(
        (
            entry
            for entry in load_manifest_rows(packed_root, split)
            if str(entry.get("sample_id")) == sample_id
        ),
        None,
    )
    if row is None:
        raise FileNotFoundError(f"sample_id not found in split `{split}`: {sample_id}")

    include_tail = True
    pad_short_sequences = True
    sample_payload = load_sample_from_manifest_row(packed_root=packed_root, row=row)
    length = int(sample_payload["length"])
    dims = int(sample_payload["num_series"])
    feature_index = max(0, min(feature_index, dims - 1))
    if slice_end <= 0 or slice_end > length:
        slice_end = length
    if slice_start >= slice_end:
        slice_start = 0

    series = sample_payload["series"]
    normal_series = sample_payload["normal_series"]
    point_mask = sample_payload["point_mask"]
    point_mask_any = sample_payload["point_mask_any"]
    feature_mask = np.asarray(point_mask[:, feature_index], dtype=np.uint8)

    windows = iter_context_bounds(
        length,
        context_size=context_size,
        stride=stride,
        include_tail=include_tail,
        pad_short_sequences=pad_short_sequences,
    )
    window_rows: list[dict[str, Any]] = []
    padded_window_count = 0
    short_window_count = 0
    tail_window_count = 0
    for index, (start, end) in enumerate(windows):
        geometry = describe_window(
            start=start,
            end=end,
            sequence_length=length,
            context_size=context_size,
            patch_size=patch_size,
        )
        if geometry["padded_steps"]:
            padded_window_count += 1
        if geometry["is_short_window"]:
            short_window_count += 1
        if geometry["is_tail_window"]:
            tail_window_count += 1
        if index >= 500:
            continue

        window_mask = slice_or_pad_1d(
            feature_mask,
            start=start,
            end=end,
            target_length=context_size,
            pad_value=0,
            dtype=np.uint8,
        )
        patch_labels = build_patch_labels_1d(window_mask, patch_size=patch_size)
        window_rows.append(
            {
                "start": int(start),
                "end": int(end),
                **geometry,
                "patch_positive_ratio": float(patch_labels.mean()) if patch_labels.size else 0.0,
            }
        )

    metadata = sample_payload.get("metadata", {})
    summary = metadata.get("summary", {}) if isinstance(metadata, dict) else {}
    return {
        "sample_id": sample_id,
        "split": split,
        "length": length,
        "num_series": dims,
        "feature_index": feature_index,
        "slice_start": slice_start,
        "slice_end": slice_end,
        "series_feature": series[slice_start:slice_end, feature_index].astype(float).tolist(),
        "normal_series_feature": normal_series[slice_start:slice_end, feature_index]
        .astype(float)
        .tolist(),
        "point_mask_feature": point_mask[slice_start:slice_end, feature_index].astype(int).tolist(),
        "point_mask_any": point_mask_any[slice_start:slice_end].astype(int).tolist(),
        "anomaly_ratio_feature": float(point_mask[:, feature_index].mean()),
        "windowing": {
            "context_size": context_size,
            "stride": stride,
            "patch_size": patch_size,
            "include_tail": include_tail,
            "pad_short_sequences": pad_short_sequences,
            "num_windows": len(windows),
            "padded_window_count": padded_window_count,
            "short_window_count": short_window_count,
            "tail_window_count": tail_window_count,
            "windows": window_rows,
        },
        "manifest_row": row,
        "summary": summary,
        "metadata": metadata,
    }


def build_window_payload(
    *,
    path_like: str,
    split: str,
    sample_id: str,
    feature_index: int,
    window_index: int,
    context_size: int,
    stride: int,
    patch_size: int,
) -> dict[str, Any]:
    """Build the payload returned by `/api/window`."""

    packed_root = resolve_packed_root(path_like)
    row = next(
        (
            entry
            for entry in load_manifest_rows(packed_root, split)
            if str(entry.get("sample_id")) == sample_id
        ),
        None,
    )
    if row is None:
        raise FileNotFoundError(f"sample_id not found in split `{split}`: {sample_id}")

    include_tail = True
    pad_short_sequences = True
    sample_payload = load_sample_from_manifest_row(packed_root=packed_root, row=row)
    length = int(sample_payload["length"])
    dims = int(sample_payload["num_series"])
    feature_index = max(0, min(feature_index, dims - 1))
    windows = iter_context_bounds(
        length,
        context_size=context_size,
        stride=stride,
        include_tail=include_tail,
        pad_short_sequences=pad_short_sequences,
    )
    if not windows:
        raise ValueError(
            "No training windows are available for the selected sample after applying the current "
            "context_size / stride settings."
        )
    if window_index >= len(windows):
        window_index = len(windows) - 1
    start, end = windows[window_index]
    geometry = describe_window(
        start=start,
        end=end,
        sequence_length=length,
        context_size=context_size,
        patch_size=patch_size,
    )

    series_window = slice_or_pad_1d(
        sample_payload["series"][:, feature_index],
        start=start,
        end=end,
        target_length=context_size,
        pad_value=0.0,
        dtype=np.float32,
    )
    normal_window = slice_or_pad_1d(
        sample_payload["normal_series"][:, feature_index],
        start=start,
        end=end,
        target_length=context_size,
        pad_value=0.0,
        dtype=np.float32,
    )
    mask_window = slice_or_pad_1d(
        sample_payload["point_mask"][:, feature_index],
        start=start,
        end=end,
        target_length=context_size,
        pad_value=0,
        dtype=np.uint8,
    )
    any_window = slice_or_pad_1d(
        sample_payload["point_mask_any"],
        start=start,
        end=end,
        target_length=context_size,
        pad_value=0,
        dtype=np.uint8,
    )
    padding_mask = build_padding_mask(
        effective_length=int(geometry["effective_length"]),
        context_size=context_size,
    )
    patch_labels = build_patch_labels_1d(mask_window, patch_size=patch_size).astype(int).tolist()

    return {
        "sample_id": sample_id,
        "split": split,
        "feature_index": feature_index,
        "window_index": window_index,
        "start": int(start),
        "end": int(end),
        **geometry,
        "context_size": int(context_size),
        "stride": int(stride),
        "patch_size": int(patch_size),
        "include_tail": include_tail,
        "pad_short_sequences": pad_short_sequences,
        "series_feature": series_window.astype(float).tolist(),
        "normal_series_feature": normal_window.astype(float).tolist(),
        "point_mask_feature": mask_window.astype(int).tolist(),
        "point_mask_any": any_window.astype(int).tolist(),
        "padding_mask": padding_mask.astype(int).tolist(),
        "patch_labels": patch_labels,
        "patch_alignment_warning": (
            None
            if context_size % patch_size == 0
            else "context_size is not divisible by patch_size"
        ),
    }
