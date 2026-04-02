from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SplitName = str


@dataclass(frozen=True, slots=True)
class SourceSampleRecord:
    sample_id: str
    npz_path: Path
    json_path: Path


@dataclass(frozen=True, slots=True)
class SplitPackStats:
    split: SplitName
    num_samples: int
    num_shards: int
    total_points: int
    anomalous_samples: int
    length_min: int
    length_max: int
    length_mean: float
    num_series_min: int
    num_series_max: int
    num_series_mean: float


@dataclass(frozen=True, slots=True)
class PackReport:
    dataset_name: str
    dataset_version: str
    output_root: Path
    splits: dict[SplitName, SplitPackStats]


def discover_input_splits(
    input_root: Path, split: SplitName | None = None
) -> dict[SplitName, Path]:
    if split is not None:
        return {split: input_root.resolve()}

    discovered: dict[SplitName, Path] = {}
    for name in ("train", "val", "test"):
        candidate = (input_root / name).resolve()
        if candidate.is_dir() and any(candidate.glob("sample_*.npz")):
            discovered[name] = candidate

    if discovered:
        return discovered

    return {"train": input_root.resolve()}


def pack_synthetic_corpus(
    input_root: Path,
    output_root: Path,
    *,
    split: SplitName | None = None,
    samples_per_shard: int = 512,
    overwrite: bool = False,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
) -> PackReport:
    if samples_per_shard <= 0:
        raise ValueError(f"samples_per_shard must be positive, got {samples_per_shard}")

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    split_dirs = discover_input_splits(input_root, split=split)

    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_dataset_name = dataset_name or output_root.parent.name or "synthetic_rcd"
    resolved_dataset_version = dataset_version or output_root.name or "v1"

    split_reports: dict[SplitName, SplitPackStats] = {}
    for split_name, split_dir in split_dirs.items():
        split_reports[split_name] = _pack_split(
            split_name=split_name,
            split_dir=split_dir,
            output_root=output_root,
            samples_per_shard=samples_per_shard,
        )

    _write_dataset_meta(
        output_root=output_root,
        dataset_name=resolved_dataset_name,
        dataset_version=resolved_dataset_version,
        samples_per_shard=samples_per_shard,
        split_reports=split_reports,
    )

    return PackReport(
        dataset_name=resolved_dataset_name,
        dataset_version=resolved_dataset_version,
        output_root=output_root,
        splits=split_reports,
    )


def write_dataset_meta_for_existing_packed_corpus(
    output_root: Path,
    *,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    samples_per_shard: int | None = None,
    storage_format_type: str = "npz_shards_with_jsonl_metadata",
    storage_format_extras: dict[str, Any] | None = None,
) -> PackReport:
    """Write `dataset_meta.json` from existing manifest files under a packed root."""

    output_root = output_root.resolve()
    split_reports: dict[SplitName, SplitPackStats] = {}
    inferred_samples_per_shard = 0

    for split_name in ("train", "val", "test"):
        manifest_path = output_root / f"manifest.{split_name}.jsonl"
        if not manifest_path.exists():
            continue
        split_report, split_max_samples_per_shard = _summarize_manifest_split(
            split_name=split_name,
            manifest_path=manifest_path,
        )
        split_reports[split_name] = split_report
        inferred_samples_per_shard = max(
            inferred_samples_per_shard,
            split_max_samples_per_shard,
        )

    if not split_reports:
        raise FileNotFoundError(f"No manifest files were found under packed root: {output_root}")

    resolved_samples_per_shard = (
        int(samples_per_shard)
        if samples_per_shard is not None
        else max(1, inferred_samples_per_shard)
    )
    if resolved_samples_per_shard <= 0:
        raise ValueError(
            f"`samples_per_shard` must be positive when provided, got {resolved_samples_per_shard}."
        )

    resolved_dataset_name = dataset_name or output_root.parent.name or "synthetic_rcd"
    resolved_dataset_version = dataset_version or output_root.name or "v1"

    _write_dataset_meta(
        output_root=output_root,
        dataset_name=resolved_dataset_name,
        dataset_version=resolved_dataset_version,
        samples_per_shard=resolved_samples_per_shard,
        split_reports=split_reports,
        storage_format_type=storage_format_type,
        storage_format_extras=storage_format_extras,
    )
    return PackReport(
        dataset_name=resolved_dataset_name,
        dataset_version=resolved_dataset_version,
        output_root=output_root,
        splits=split_reports,
    )


def _discover_packed_manifest_splits(
    input_root: Path,
    split: SplitName | None,
) -> dict[SplitName, Path]:
    if split is not None:
        manifest_path = (input_root / f"manifest.{split}.jsonl").resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing packed manifest for split `{split}`: {manifest_path}")
        return {split: manifest_path}

    discovered: dict[SplitName, Path] = {}
    for split_name in ("train", "val", "test"):
        manifest_path = (input_root / f"manifest.{split_name}.jsonl").resolve()
        if manifest_path.exists():
            discovered[split_name] = manifest_path
    if not discovered:
        raise FileNotFoundError(
            f"No packed manifests were found under {input_root}. "
            "Expected files like manifest.train.jsonl."
        )
    return discovered


def _iter_context_bounds(
    sequence_length: int,
    *,
    context_size: int,
    stride: int,
    include_tail: bool,
    pad_short_sequences: bool,
) -> tuple[tuple[int, int], ...]:
    if sequence_length <= 0:
        return ()
    if sequence_length < context_size:
        if not pad_short_sequences:
            return ()
        return ((0, sequence_length),)

    last_full_start = sequence_length - context_size
    starts = list(range(0, last_full_start + 1, stride))
    bounds = [(start, start + context_size) for start in starts]

    last_covered_end = bounds[-1][1] if bounds else 0
    if include_tail and last_covered_end < sequence_length:
        tail_start = starts[-1] + stride if starts else 0
        tail_start = min(tail_start, sequence_length - 1)
        tail_end = sequence_length
        if tail_start < tail_end:
            bounds.append((tail_start, tail_end))
    return tuple(bounds)


def _slice_or_pad_2d(
    array: np.ndarray,
    *,
    start: int,
    end: int,
    target_length: int,
    dtype: np.dtype[np.generic] | type[np.generic],
    pad_value: float | int,
) -> np.ndarray:
    window = np.asarray(array[start:end], dtype=dtype)
    if window.ndim != 2:
        raise ValueError(f"Expected 2D array slice, got shape {window.shape}.")
    if window.shape[0] > target_length:
        raise ValueError(f"Window length {window.shape[0]} exceeds target_length {target_length}.")
    if window.shape[0] == target_length:
        return np.ascontiguousarray(window)
    padded = np.full((target_length, window.shape[1]), pad_value, dtype=dtype)
    padded[: window.shape[0]] = window
    return padded


def _build_patch_labels(point_mask_window: np.ndarray, patch_size: int) -> np.ndarray:
    if point_mask_window.ndim != 2:
        raise ValueError(f"`point_mask_window` must be [W, D], got {point_mask_window.shape}.")
    if point_mask_window.shape[0] % patch_size != 0:
        raise ValueError(
            f"Window length {point_mask_window.shape[0]} must be divisible by patch_size={patch_size}."
        )
    num_patches = point_mask_window.shape[0] // patch_size
    reshaped = point_mask_window.reshape(num_patches, patch_size, point_mask_window.shape[1])
    return reshaped.max(axis=1).astype(np.uint8, copy=False)


def _passes_window_ratio_filter(
    *,
    patch_positive_ratio: float,
    anomaly_point_ratio: float,
    min_patch_positive_ratio: float | None,
    min_anomaly_point_ratio: float | None,
) -> bool:
    if min_patch_positive_ratio is not None and patch_positive_ratio < min_patch_positive_ratio:
        return False
    if min_anomaly_point_ratio is not None and anomaly_point_ratio < min_anomaly_point_ratio:
        return False
    return True


def pack_windows_from_packed_corpus(
    input_root: Path,
    output_root: Path,
    *,
    split: SplitName | None = None,
    context_size: int = 1024,
    patch_size: int = 16,
    stride: int | None = None,
    include_tail: bool = True,
    pad_short_sequences: bool = True,
    windows_per_shard: int = 4096,
    overwrite: bool = False,
    dataset_name: str | None = None,
    dataset_version: str | None = None,
    write_debug_sidecar: bool = True,
    min_patch_positive_ratio: float | None = None,
    min_anomaly_point_ratio: float | None = None,
) -> PackReport:
    """Convert sample-level packed shards into window-level packed shards for training."""

    if context_size <= 0:
        raise ValueError(f"context_size must be positive, got {context_size}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if context_size % patch_size != 0:
        raise ValueError("context_size must be divisible by patch_size")
    resolved_stride = context_size if stride is None else int(stride)
    if resolved_stride <= 0:
        raise ValueError(f"stride must be positive, got {resolved_stride}")
    if windows_per_shard <= 0:
        raise ValueError(f"windows_per_shard must be positive, got {windows_per_shard}")
    if min_patch_positive_ratio is not None and float(min_patch_positive_ratio) < 0.0:
        raise ValueError(
            f"`min_patch_positive_ratio` must be >= 0 when provided, got {min_patch_positive_ratio}"
        )
    if min_anomaly_point_ratio is not None and float(min_anomaly_point_ratio) < 0.0:
        raise ValueError(
            f"`min_anomaly_point_ratio` must be >= 0 when provided, got {min_anomaly_point_ratio}"
        )

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    split_manifests = _discover_packed_manifest_splits(input_root, split)

    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_dataset_name = dataset_name or output_root.parent.name or "synthetic_rcd_window"
    resolved_dataset_version = dataset_version or output_root.name or "v1"

    split_reports: dict[SplitName, SplitPackStats] = {}
    for split_name, manifest_path in split_manifests.items():
        split_reports[split_name] = _pack_window_split(
            split_name=split_name,
            manifest_path=manifest_path,
            output_root=output_root,
            context_size=context_size,
            patch_size=patch_size,
            stride=resolved_stride,
            include_tail=include_tail,
            pad_short_sequences=pad_short_sequences,
            windows_per_shard=windows_per_shard,
            write_debug_sidecar=write_debug_sidecar,
            min_patch_positive_ratio=(
                None if min_patch_positive_ratio is None else float(min_patch_positive_ratio)
            ),
            min_anomaly_point_ratio=(
                None if min_anomaly_point_ratio is None else float(min_anomaly_point_ratio)
            ),
        )

    _write_dataset_meta(
        output_root=output_root,
        dataset_name=resolved_dataset_name,
        dataset_version=resolved_dataset_version,
        samples_per_shard=windows_per_shard,
        split_reports=split_reports,
        storage_format_type="npz_window_shards_minimal_v1",
        storage_format_extras={
            "context_size": int(context_size),
            "patch_size": int(patch_size),
            "stride": int(resolved_stride),
            "include_tail": bool(include_tail),
            "pad_short_sequences": bool(pad_short_sequences),
            "windows_per_shard": int(windows_per_shard),
            "debug_sidecar": bool(write_debug_sidecar),
            "min_patch_positive_ratio": (
                None if min_patch_positive_ratio is None else float(min_patch_positive_ratio)
            ),
            "min_anomaly_point_ratio": (
                None if min_anomaly_point_ratio is None else float(min_anomaly_point_ratio)
            ),
            "required_fields": [
                "series_windows",
                "point_mask_windows",
                "patch_labels_windows",
                "valid_lengths",
            ],
        },
    )

    return PackReport(
        dataset_name=resolved_dataset_name,
        dataset_version=resolved_dataset_version,
        output_root=output_root,
        splits=split_reports,
    )


def _pack_window_split(
    *,
    split_name: SplitName,
    manifest_path: Path,
    output_root: Path,
    context_size: int,
    patch_size: int,
    stride: int,
    include_tail: bool,
    pad_short_sequences: bool,
    windows_per_shard: int,
    write_debug_sidecar: bool,
    min_patch_positive_ratio: float | None,
    min_anomaly_point_ratio: float | None,
) -> SplitPackStats:
    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_root / f"manifest.{split_name}.jsonl"
    debug_sidecar_path = output_root / f"debug.{split_name}.jsonl"

    source_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not source_rows:
        raise FileNotFoundError(f"No source rows found in manifest: {manifest_path}")

    shard_npz_cache: dict[Path, dict[str, np.ndarray]] = {}
    shard_jsonl_cache: dict[Path, list[dict[str, Any]]] = {}

    series_windows_buffer: list[np.ndarray] = []
    point_mask_windows_buffer: list[np.ndarray] = []
    patch_labels_windows_buffer: list[np.ndarray] = []
    valid_lengths_buffer: list[int] = []
    context_start_buffer: list[int] = []
    context_end_buffer: list[int] = []
    manifest_meta_buffer: list[dict[str, Any]] = []
    debug_meta_buffer: list[dict[str, Any] | None] = []

    total_points = 0
    anomalous_windows = 0
    lengths: list[int] = []
    num_series_values: list[int] = []
    shard_count = 0
    debug_row_index = 0

    def load_source_arrays(shard_npz_path: Path) -> dict[str, np.ndarray]:
        cached = shard_npz_cache.get(shard_npz_path)
        if cached is not None:
            return cached
        required = {
            "series_values",
            "point_mask_values",
            "lengths",
            "num_series",
            "series_offsets",
            "time_offsets",
        }
        with np.load(shard_npz_path, allow_pickle=False) as npz:
            missing = sorted(name for name in required if name not in npz.files)
            if missing:
                raise KeyError(
                    f"Missing required arrays in source shard {shard_npz_path}: {', '.join(missing)}"
                )
            cached = {name: np.asarray(npz[name]) for name in required}
        shard_npz_cache[shard_npz_path] = cached
        return cached

    def load_source_metadata_rows(shard_jsonl_path: Path) -> list[dict[str, Any]]:
        cached = shard_jsonl_cache.get(shard_jsonl_path)
        if cached is not None:
            return cached
        cached = [
            json.loads(line)
            for line in shard_jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        shard_jsonl_cache[shard_jsonl_path] = cached
        return cached

    def flush_window_shard(
        shard_index: int,
        *,
        manifest_handle,
        debug_handle,
    ) -> None:
        nonlocal debug_row_index
        if not series_windows_buffer:
            return

        shard_npz_path = split_output_dir / f"shard-{shard_index:05d}.npz"
        np.savez_compressed(
            shard_npz_path,
            series_windows=np.stack(series_windows_buffer, axis=0).astype(np.float32, copy=False),
            point_mask_windows=np.stack(point_mask_windows_buffer, axis=0).astype(
                np.uint8, copy=False
            ),
            patch_labels_windows=np.stack(patch_labels_windows_buffer, axis=0).astype(
                np.uint8, copy=False
            ),
            valid_lengths=np.asarray(valid_lengths_buffer, dtype=np.int32),
            context_start=np.asarray(context_start_buffer, dtype=np.int32),
            context_end=np.asarray(context_end_buffer, dtype=np.int32),
        )

        shard_npz_rel = str(shard_npz_path.relative_to(output_root))
        for window_index, row_meta in enumerate(manifest_meta_buffer):
            manifest_row = {
                "sample_id": row_meta["sample_id"],
                "source_sample_id": row_meta["source_sample_id"],
                "split": split_name,
                "shard_id": int(shard_index),
                "window_index": int(window_index),
                "sample_index": int(window_index),
                "shard_npz_path": shard_npz_rel,
                "context_start": int(row_meta["context_start"]),
                "context_end": int(row_meta["context_end"]),
                "valid_length": int(row_meta["valid_length"]),
                "length": int(context_size),
                "num_series": int(row_meta["num_series"]),
                "is_anomalous_sample": int(row_meta["is_anomalous_window"]),
                "anomaly_point_ratio": float(row_meta["anomaly_point_ratio"]),
                "patch_positive_ratio": float(row_meta["patch_positive_ratio"]),
            }
            debug_row = debug_meta_buffer[window_index]
            if write_debug_sidecar and debug_row is not None:
                manifest_row["debug_jsonl_path"] = str(debug_sidecar_path.relative_to(output_root))
                manifest_row["debug_row_index"] = int(debug_row_index)
                assert debug_handle is not None
                debug_handle.write(json.dumps(debug_row, ensure_ascii=False) + "\n")
                debug_row_index += 1

            manifest_handle.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

        series_windows_buffer.clear()
        point_mask_windows_buffer.clear()
        patch_labels_windows_buffer.clear()
        valid_lengths_buffer.clear()
        context_start_buffer.clear()
        context_end_buffer.clear()
        manifest_meta_buffer.clear()
        debug_meta_buffer.clear()

    with output_manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_handle:
        debug_handle = (
            debug_sidecar_path.open("w", encoding="utf-8", newline="\n")
            if write_debug_sidecar
            else None
        )
        try:
            for source_row in source_rows:
                shard_npz_path_raw = source_row.get("shard_npz_path")
                sample_index_raw = source_row.get("sample_index")
                if shard_npz_path_raw is None or sample_index_raw is None:
                    raise ValueError(
                        f"Source packed manifest row is missing `shard_npz_path` or `sample_index`: {source_row}"
                    )
                shard_npz_path = (manifest_path.parent / str(shard_npz_path_raw)).resolve()
                sample_index = int(sample_index_raw)
                shard_arrays = load_source_arrays(shard_npz_path)

                lengths_arr = shard_arrays["lengths"]
                num_series_arr = shard_arrays["num_series"]
                series_offsets = shard_arrays["series_offsets"]
                time_offsets = shard_arrays["time_offsets"]
                if sample_index < 0 or sample_index >= len(lengths_arr):
                    raise IndexError(
                        f"Sample index {sample_index} out of range for shard {shard_npz_path}"
                    )

                sample_length = int(lengths_arr[sample_index])
                num_series = int(num_series_arr[sample_index])
                flat_start = int(series_offsets[sample_index])
                flat_end = int(series_offsets[sample_index + 1])
                time_start = int(time_offsets[sample_index])
                time_end = int(time_offsets[sample_index + 1])
                expected_flat = int(sample_length * num_series)
                if flat_end - flat_start != expected_flat:
                    raise ValueError(
                        f"Corrupt source offsets in {shard_npz_path}: "
                        f"sample_index={sample_index}, expected={expected_flat}, actual={flat_end - flat_start}"
                    )
                if time_end - time_start != sample_length:
                    raise ValueError(
                        f"Corrupt source time offsets in {shard_npz_path}: "
                        f"sample_index={sample_index}, expected={sample_length}, actual={time_end - time_start}"
                    )

                sample_series = shard_arrays["series_values"][flat_start:flat_end].reshape(
                    sample_length,
                    num_series,
                )
                sample_point_mask = shard_arrays["point_mask_values"][flat_start:flat_end].reshape(
                    sample_length,
                    num_series,
                )

                source_sample_id = str(source_row.get("sample_id") or f"sample_{sample_index:06d}")
                source_metadata = None
                if write_debug_sidecar:
                    shard_jsonl_path_raw = source_row.get("shard_jsonl_path")
                    if shard_jsonl_path_raw is not None:
                        shard_jsonl_path = (
                            manifest_path.parent / str(shard_jsonl_path_raw)
                        ).resolve()
                        if shard_jsonl_path.exists():
                            metadata_rows = load_source_metadata_rows(shard_jsonl_path)
                            if 0 <= sample_index < len(metadata_rows):
                                source_metadata = metadata_rows[sample_index]

                bounds = _iter_context_bounds(
                    sample_length,
                    context_size=context_size,
                    stride=stride,
                    include_tail=include_tail,
                    pad_short_sequences=pad_short_sequences,
                )
                for context_start, context_end in bounds:
                    valid_length = int(context_end - context_start)
                    if valid_length <= 0:
                        continue

                    window_series = _slice_or_pad_2d(
                        sample_series,
                        start=context_start,
                        end=context_end,
                        target_length=context_size,
                        dtype=np.float32,
                        pad_value=0.0,
                    )
                    window_point_mask = _slice_or_pad_2d(
                        sample_point_mask,
                        start=context_start,
                        end=context_end,
                        target_length=context_size,
                        dtype=np.uint8,
                        pad_value=0,
                    )
                    patch_labels = _build_patch_labels(window_point_mask, patch_size=patch_size)
                    valid_point_mask = window_point_mask[:valid_length]
                    anomaly_point_ratio = (
                        float(valid_point_mask.mean()) if valid_point_mask.size else 0.0
                    )
                    patch_positive_ratio = float(patch_labels.mean()) if patch_labels.size else 0.0
                    if not _passes_window_ratio_filter(
                        patch_positive_ratio=patch_positive_ratio,
                        anomaly_point_ratio=anomaly_point_ratio,
                        min_patch_positive_ratio=min_patch_positive_ratio,
                        min_anomaly_point_ratio=min_anomaly_point_ratio,
                    ):
                        continue
                    is_anomalous_window = int(bool(valid_point_mask.any()))

                    window_id = f"{source_sample_id}__w{context_start:06d}_{context_end:06d}"
                    series_windows_buffer.append(window_series)
                    point_mask_windows_buffer.append(window_point_mask)
                    patch_labels_windows_buffer.append(patch_labels)
                    valid_lengths_buffer.append(valid_length)
                    context_start_buffer.append(int(context_start))
                    context_end_buffer.append(int(context_end))
                    manifest_meta_buffer.append(
                        {
                            "sample_id": window_id,
                            "source_sample_id": source_sample_id,
                            "context_start": int(context_start),
                            "context_end": int(context_end),
                            "valid_length": int(valid_length),
                            "num_series": int(num_series),
                            "is_anomalous_window": int(is_anomalous_window),
                            "anomaly_point_ratio": float(anomaly_point_ratio),
                            "patch_positive_ratio": float(patch_positive_ratio),
                        }
                    )
                    if write_debug_sidecar:
                        debug_meta_buffer.append(
                            {
                                "sample_id": window_id,
                                "source_sample_id": source_sample_id,
                                "split": split_name,
                                "context_start": int(context_start),
                                "context_end": int(context_end),
                                "source_shard_npz_path": str(shard_npz_path),
                                "source_sample_index": int(sample_index),
                                "source_metadata": source_metadata,
                            }
                        )
                    else:
                        debug_meta_buffer.append(None)

                    total_points += int(context_size * num_series)
                    anomalous_windows += int(is_anomalous_window)
                    lengths.append(int(context_size))
                    num_series_values.append(int(num_series))

                    if len(series_windows_buffer) >= windows_per_shard:
                        flush_window_shard(
                            shard_count, manifest_handle=manifest_handle, debug_handle=debug_handle
                        )
                        shard_count += 1

            if series_windows_buffer:
                flush_window_shard(
                    shard_count, manifest_handle=manifest_handle, debug_handle=debug_handle
                )
                shard_count += 1
        finally:
            if debug_handle is not None:
                debug_handle.close()

    if not lengths or not num_series_values:
        raise FileNotFoundError(
            f"No windows were produced for split `{split_name}` from manifest {manifest_path}"
        )

    return SplitPackStats(
        split=split_name,
        num_samples=len(lengths),
        num_shards=shard_count,
        total_points=total_points,
        anomalous_samples=anomalous_windows,
        length_min=min(lengths),
        length_max=max(lengths),
        length_mean=float(sum(lengths) / len(lengths)),
        num_series_min=min(num_series_values),
        num_series_max=max(num_series_values),
        num_series_mean=float(sum(num_series_values) / len(num_series_values)),
    )


def _pack_split(
    *,
    split_name: SplitName,
    split_dir: Path,
    output_root: Path,
    samples_per_shard: int,
) -> SplitPackStats:
    records = _scan_source_records(split_dir)
    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"manifest.{split_name}.jsonl"

    shard_count = 0
    total_points = 0
    anomalous_samples = 0
    lengths: list[int] = []
    num_series_values: list[int] = []

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_handle:
        for shard_index, chunk in enumerate(_chunked(records, samples_per_shard)):
            shard_count += 1
            shard_npz_path = split_output_dir / f"shard-{shard_index:05d}.npz"
            shard_jsonl_path = split_output_dir / f"shard-{shard_index:05d}.jsonl"

            shard_manifest_rows, shard_stats = _write_shard(
                split_name=split_name,
                shard_index=shard_index,
                records=chunk,
                shard_npz_path=shard_npz_path,
                shard_jsonl_path=shard_jsonl_path,
            )

            total_points += shard_stats["total_points"]
            anomalous_samples += shard_stats["anomalous_samples"]
            lengths.extend(shard_stats["lengths"])
            num_series_values.extend(shard_stats["num_series"])

            for row in shard_manifest_rows:
                manifest_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not lengths or not num_series_values:
        raise FileNotFoundError(f"No source samples found in {split_dir}")

    return SplitPackStats(
        split=split_name,
        num_samples=len(lengths),
        num_shards=shard_count,
        total_points=total_points,
        anomalous_samples=anomalous_samples,
        length_min=min(lengths),
        length_max=max(lengths),
        length_mean=float(sum(lengths) / len(lengths)),
        num_series_min=min(num_series_values),
        num_series_max=max(num_series_values),
        num_series_mean=float(sum(num_series_values) / len(num_series_values)),
    )


def _scan_source_records(split_dir: Path) -> list[SourceSampleRecord]:
    records: list[SourceSampleRecord] = []
    for npz_path in sorted(split_dir.glob("sample_*.npz")):
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Missing JSON pair for {npz_path}")
        records.append(
            SourceSampleRecord(
                sample_id=npz_path.stem,
                npz_path=npz_path,
                json_path=json_path,
            )
        )
    return records


def _chunked(items: list[SourceSampleRecord], size: int) -> Iterable[list[SourceSampleRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _write_shard(
    *,
    split_name: SplitName,
    shard_index: int,
    records: list[SourceSampleRecord],
    shard_npz_path: Path,
    shard_jsonl_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    series_values: list[np.ndarray] = []
    normal_series_values: list[np.ndarray] = []
    point_mask_values: list[np.ndarray] = []
    point_mask_any_values: list[np.ndarray] = []

    series_offsets = [0]
    time_offsets = [0]
    lengths: list[int] = []
    num_series_list: list[int] = []
    manifest_rows: list[dict[str, Any]] = []
    anomalous_samples = 0
    total_points = 0

    with shard_jsonl_path.open("w", encoding="utf-8", newline="\n") as shard_jsonl_handle:
        for sample_index, record in enumerate(records):
            with np.load(record.npz_path, allow_pickle=False) as npz:
                series = np.asarray(npz["series"], dtype=np.float32)
                normal_series = np.asarray(npz["normal_series"], dtype=np.float32)
                point_mask = np.asarray(npz["point_mask"], dtype=np.uint8)
                point_mask_any = np.asarray(npz["point_mask_any"], dtype=np.uint8)

            payload = json.loads(record.json_path.read_text(encoding="utf-8"))

            _validate_sample_shapes(
                sample_id=record.sample_id,
                series=series,
                normal_series=normal_series,
                point_mask=point_mask,
                point_mask_any=point_mask_any,
            )

            length = int(series.shape[0])
            num_series = int(series.shape[1])
            flat_size = int(series.size)

            series_values.append(series.reshape(-1))
            normal_series_values.append(normal_series.reshape(-1))
            point_mask_values.append(point_mask.reshape(-1))
            point_mask_any_values.append(point_mask_any.reshape(-1))

            series_offsets.append(series_offsets[-1] + flat_size)
            time_offsets.append(time_offsets[-1] + length)
            lengths.append(length)
            num_series_list.append(num_series)
            total_points += int(length * num_series)

            summary = payload.get("summary", {})
            is_anomalous_sample = int(summary.get("is_anomalous_sample", int(point_mask_any.any())))
            anomalous_samples += is_anomalous_sample
            num_events = len(payload.get("events", []))
            anomaly_point_ratio = float(point_mask_any.mean()) if point_mask_any.size else 0.0

            shard_line = dict(payload)
            shard_line["sample_id"] = record.sample_id
            shard_line["split"] = split_name
            shard_line["sample_index"] = sample_index
            shard_jsonl_handle.write(json.dumps(shard_line, ensure_ascii=False) + "\n")

            manifest_rows.append(
                {
                    "sample_id": record.sample_id,
                    "split": split_name,
                    "shard_id": shard_index,
                    "sample_index": sample_index,
                    "shard_npz_path": str(shard_npz_path.relative_to(shard_npz_path.parents[1])),
                    "shard_jsonl_path": str(
                        shard_jsonl_path.relative_to(shard_jsonl_path.parents[1])
                    ),
                    "length": length,
                    "num_series": num_series,
                    "is_anomalous_sample": is_anomalous_sample,
                    "num_events": num_events,
                    "anomaly_point_ratio": anomaly_point_ratio,
                }
            )

    np.savez_compressed(
        shard_npz_path,
        series_values=np.concatenate(series_values).astype(np.float32, copy=False),
        normal_series_values=np.concatenate(normal_series_values).astype(np.float32, copy=False),
        point_mask_values=np.concatenate(point_mask_values).astype(np.uint8, copy=False),
        point_mask_any_values=np.concatenate(point_mask_any_values).astype(np.uint8, copy=False),
        lengths=np.asarray(lengths, dtype=np.int32),
        num_series=np.asarray(num_series_list, dtype=np.int32),
        series_offsets=np.asarray(series_offsets, dtype=np.int64),
        time_offsets=np.asarray(time_offsets, dtype=np.int64),
    )

    return manifest_rows, {
        "total_points": total_points,
        "anomalous_samples": anomalous_samples,
        "lengths": lengths,
        "num_series": num_series_list,
    }


def _validate_sample_shapes(
    *,
    sample_id: str,
    series: np.ndarray,
    normal_series: np.ndarray,
    point_mask: np.ndarray,
    point_mask_any: np.ndarray,
) -> None:
    if series.ndim != 2:
        raise ValueError(f"{sample_id}: series must be 2D [T, D], got {series.shape}")
    if normal_series.shape != series.shape:
        raise ValueError(
            f"{sample_id}: normal_series shape mismatch, expected {series.shape}, got {normal_series.shape}"
        )
    if point_mask.shape != series.shape:
        raise ValueError(
            f"{sample_id}: point_mask shape mismatch, expected {series.shape}, got {point_mask.shape}"
        )
    if point_mask_any.ndim != 1 or point_mask_any.shape[0] != series.shape[0]:
        raise ValueError(f"{sample_id}: point_mask_any must be [T], got {point_mask_any.shape}")


def _summarize_manifest_split(
    *,
    split_name: SplitName,
    manifest_path: Path,
) -> tuple[SplitPackStats, int]:
    lengths: list[int] = []
    num_series_values: list[int] = []
    shard_counts: dict[int, int] = {}
    total_points = 0
    anomalous_samples = 0

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row_text = line.strip()
            if not row_text:
                continue
            try:
                row = json.loads(row_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row must be an object in {manifest_path}:{line_number}")

            try:
                length = int(row["length"])
                num_series = int(row["num_series"])
                shard_id = int(row["shard_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Manifest row missing required numeric fields in {manifest_path}:{line_number}"
                ) from exc

            if length <= 0 or num_series <= 0:
                raise ValueError(
                    f"Manifest row has non-positive dimensions in {manifest_path}:{line_number}"
                )

            lengths.append(length)
            num_series_values.append(num_series)
            total_points += int(length * num_series)
            anomalous_samples += int(bool(row.get("is_anomalous_sample", 0)))
            shard_counts[shard_id] = shard_counts.get(shard_id, 0) + 1

    if not lengths or not num_series_values:
        raise ValueError(f"Manifest has no valid rows: {manifest_path}")

    return (
        SplitPackStats(
            split=split_name,
            num_samples=len(lengths),
            num_shards=len(shard_counts),
            total_points=total_points,
            anomalous_samples=anomalous_samples,
            length_min=min(lengths),
            length_max=max(lengths),
            length_mean=float(sum(lengths) / len(lengths)),
            num_series_min=min(num_series_values),
            num_series_max=max(num_series_values),
            num_series_mean=float(sum(num_series_values) / len(num_series_values)),
        ),
        max(shard_counts.values()),
    )


def _write_dataset_meta(
    *,
    output_root: Path,
    dataset_name: str,
    dataset_version: str,
    samples_per_shard: int,
    split_reports: dict[SplitName, SplitPackStats],
    storage_format_type: str = "npz_shards_with_jsonl_metadata",
    storage_format_extras: dict[str, Any] | None = None,
) -> None:
    total_samples = sum(report.num_samples for report in split_reports.values())
    total_points = sum(report.total_points for report in split_reports.values())
    total_shards = sum(report.num_shards for report in split_reports.values())

    payload = {
        "dataset_name": dataset_name,
        "version": dataset_version,
        "storage_format": {
            "type": storage_format_type,
            "samples_per_shard": int(samples_per_shard),
            **(dict(storage_format_extras or {})),
        },
        "splits": {
            split: {
                "num_samples": report.num_samples,
                "num_shards": report.num_shards,
                "total_points": report.total_points,
                "anomalous_samples": report.anomalous_samples,
                "length": {
                    "min": report.length_min,
                    "max": report.length_max,
                    "mean": report.length_mean,
                },
                "num_series": {
                    "min": report.num_series_min,
                    "max": report.num_series_max,
                    "mean": report.num_series_mean,
                },
            }
            for split, report in split_reports.items()
        },
        "summary": {
            "total_samples": total_samples,
            "total_points": total_points,
            "total_shards": total_shards,
        },
    }
    (output_root / "dataset_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "PackReport",
    "SourceSampleRecord",
    "SplitPackStats",
    "discover_input_splits",
    "pack_windows_from_packed_corpus",
    "pack_synthetic_corpus",
    "write_dataset_meta_for_existing_packed_corpus",
]
