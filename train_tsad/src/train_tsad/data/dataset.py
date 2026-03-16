from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..interfaces import (
    ContextWindowSample,
    ContextWindowizerProtocol,
    DatasetProtocol,
    Metadata,
    RawSample,
    SplitName,
)


@dataclass(frozen=True, slots=True)
class _RawSampleRecord:
    sample_id: str
    npz_path: Path
    json_path: Path | None


@dataclass(frozen=True, slots=True)
class _ShardSampleRecord:
    sample_id: str
    shard_npz_path: Path
    shard_jsonl_path: Path
    sample_index: int


def _ensure_2d_array(name: str, value: np.ndarray) -> np.ndarray:
    """Validate array shape as `[T, D]`."""

    if value.ndim != 2:
        raise ValueError(f"{name} must be 2D [T, D], got shape {value.shape}")
    return value


def _ensure_1d_array(name: str, value: np.ndarray, expected_length: int) -> np.ndarray:
    """Validate array shape as `[T]` and enforce sequence length."""

    if value.ndim != 1:
        raise ValueError(f"{name} must be 1D [T], got shape {value.shape}")
    if value.shape[0] != expected_length:
        raise ValueError(
            f"{name} length mismatch: expected {expected_length}, got {value.shape[0]}"
        )
    return value


def _optional_json_path(npz_path: Path) -> Path | None:
    """Return sibling `.json` path if present."""

    json_path = npz_path.with_suffix(".json")
    return json_path if json_path.exists() else None


def _load_json_payload(json_path: Path | None) -> Metadata:
    """Load optional JSON metadata payload."""

    if json_path is None:
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _resolve_sample_id(npz_path: Path, payload: Metadata) -> str:
    """Resolve stable sample id from metadata when available."""

    summary = payload.get("summary")
    if isinstance(summary, dict) and "sample_id" in summary:
        return f"sample_{int(summary['sample_id']):06d}"
    return npz_path.stem


def _resolve_manifest_path(base_dir: Path, raw_path: str | None) -> Path | None:
    """Resolve relative manifest path against the manifest directory."""

    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _iter_raw_records_from_directory(base_dir: Path) -> Iterator[_RawSampleRecord]:
    """Yield sample records from `sample_*.npz` files in one directory."""

    for npz_path in sorted(base_dir.glob("sample_*.npz")):
        yield _RawSampleRecord(
            sample_id=npz_path.stem,
            npz_path=npz_path,
            json_path=_optional_json_path(npz_path),
        )


def _iter_raw_records_from_jsonl(manifest_path: Path) -> Iterator[_RawSampleRecord]:
    """Yield sample records described by a raw-sample JSONL manifest."""

    base_dir = manifest_path.parent
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        npz_path = _resolve_manifest_path(base_dir, row.get("npz_path") or row.get("npz"))
        if npz_path is None:
            raise ValueError(f"Missing npz path in {manifest_path}:{line_number}")
        json_path = _resolve_manifest_path(base_dir, row.get("json_path") or row.get("json"))
        sample_id = str(row.get("sample_id") or npz_path.stem)
        yield _RawSampleRecord(sample_id=sample_id, npz_path=npz_path, json_path=json_path)


def _iter_raw_records_from_csv(manifest_path: Path) -> Iterator[_RawSampleRecord]:
    """Yield sample records described by a raw-sample CSV manifest."""

    base_dir = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, 2):
            npz_path = _resolve_manifest_path(base_dir, row.get("npz_path") or row.get("npz"))
            if npz_path is None:
                raise ValueError(f"Missing npz path in {manifest_path}:{line_number}")
            json_path = _resolve_manifest_path(base_dir, row.get("json_path") or row.get("json"))
            sample_id = str(row.get("sample_id") or npz_path.stem)
            yield _RawSampleRecord(sample_id=sample_id, npz_path=npz_path, json_path=json_path)


def _iter_shard_records_from_jsonl(manifest_path: Path) -> Iterator[_ShardSampleRecord]:
    """Yield shard sample records from packed-dataset JSONL manifest."""

    base_dir = manifest_path.parent
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        shard_npz_path = _resolve_manifest_path(base_dir, row.get("shard_npz_path"))
        shard_jsonl_path = _resolve_manifest_path(base_dir, row.get("shard_jsonl_path"))
        sample_index = row.get("sample_index")
        if shard_npz_path is None or shard_jsonl_path is None or sample_index is None:
            raise ValueError(
                f"Shard manifest entry in {manifest_path}:{line_number} must include "
                "'shard_npz_path', 'shard_jsonl_path', and 'sample_index'"
            )
        yield _ShardSampleRecord(
            sample_id=str(row.get("sample_id") or f"sample_{int(sample_index):06d}"),
            shard_npz_path=shard_npz_path,
            shard_jsonl_path=shard_jsonl_path,
            sample_index=int(sample_index),
        )


def _load_raw_sample(
    *,
    sample_id: str,
    split: SplitName,
    npz_path: Path,
    json_path: Path | None,
) -> RawSample:
    """Load one sample from NPZ (+ optional JSON) into the canonical `RawSample`.

    Side effects:
        Reads sample files from disk. Does not cache by itself.
    """

    payload = _load_json_payload(json_path)

    with np.load(npz_path, allow_pickle=False) as npz:
        series = _ensure_2d_array("series", np.asarray(npz["series"], dtype=np.float32))

        point_mask = None
        if "point_mask" in npz.files:
            point_mask = _ensure_2d_array(
                "point_mask",
                np.asarray(npz["point_mask"], dtype=np.uint8),
            )
            if point_mask.shape != series.shape:
                raise ValueError(
                    f"point_mask shape mismatch for {npz_path}: expected {series.shape}, got {point_mask.shape}"
                )

        point_mask_any = None
        if "point_mask_any" in npz.files:
            point_mask_any = _ensure_1d_array(
                "point_mask_any",
                np.asarray(npz["point_mask_any"], dtype=np.uint8),
                expected_length=series.shape[0],
            )
        elif point_mask is not None:
            point_mask_any = (point_mask.sum(axis=1) > 0).astype(np.uint8)

        normal_series = None
        if "normal_series" in npz.files:
            normal_series = _ensure_2d_array(
                "normal_series",
                np.asarray(npz["normal_series"], dtype=np.float32),
            )
            if normal_series.shape != series.shape:
                raise ValueError(
                    f"normal_series shape mismatch for {npz_path}: "
                    f"expected {series.shape}, got {normal_series.shape}"
                )

    resolved_sample_id = sample_id or _resolve_sample_id(npz_path, payload)
    metadata = dict(payload)
    metadata["_source"] = {
        "format": "raw_files",
        "npz_path": str(npz_path),
        "json_path": str(json_path) if json_path is not None else None,
    }

    return RawSample(
        sample_id=resolved_sample_id,
        split=split,
        series=series,
        point_mask=point_mask,
        point_mask_any=point_mask_any,
        normal_series=normal_series,
        metadata=metadata,
    )


class SyntheticTsadDataset:
    """Load per-sample `sample_*.npz/json` files and expose them as `RawSample`."""

    def __init__(
        self,
        root_dir: str | Path,
        split: SplitName = "train",
        manifest_path: str | Path | None = None,
    ) -> None:
        """Build dataset index for per-sample NPZ/JSON files."""

        self.root_dir = Path(root_dir).resolve()
        self.split = split
        self.manifest_path = Path(manifest_path).resolve() if manifest_path is not None else None
        self._records = self._build_index()

    def __len__(self) -> int:
        """Return number of raw samples."""

        return len(self._records)

    def __getitem__(self, index: int) -> RawSample:
        """Load one raw sample by index."""

        record = self._records[index]
        return _load_raw_sample(
            sample_id=record.sample_id,
            split=self.split,
            npz_path=record.npz_path,
            json_path=record.json_path,
        )

    def _build_index(self) -> list[_RawSampleRecord]:
        """Create sample index from manifest or directory scan."""

        if self.manifest_path is not None:
            if self.manifest_path.suffix == ".jsonl":
                records = list(_iter_raw_records_from_jsonl(self.manifest_path))
            elif self.manifest_path.suffix == ".csv":
                records = list(_iter_raw_records_from_csv(self.manifest_path))
            else:
                raise ValueError(
                    f"Unsupported manifest format: {self.manifest_path}. Expected .jsonl or .csv."
                )
        else:
            records = list(_iter_raw_records_from_directory(self._resolve_base_dir()))

        if not records:
            location = (
                str(self.manifest_path)
                if self.manifest_path is not None
                else str(self._resolve_base_dir())
            )
            raise FileNotFoundError(f"No synthetic TSAD samples found in {location}")

        for record in records:
            if not record.npz_path.exists():
                raise FileNotFoundError(f"Missing NPZ file: {record.npz_path}")
            if record.json_path is not None and not record.json_path.exists():
                raise FileNotFoundError(f"Missing JSON file: {record.json_path}")

        return records

    def _resolve_base_dir(self) -> Path:
        """Resolve `<root>/<split>` when present, otherwise fall back to `<root>`."""

        split_dir = self.root_dir / self.split
        return split_dir if split_dir.is_dir() else self.root_dir


class ShardedSyntheticTsadDataset:
    """Load shard-packed corpora produced by `synthetic_tsad/scripts/pack_dataset.py`."""

    def __init__(
        self,
        root_dir: str | Path,
        split: SplitName = "train",
        manifest_path: str | Path | None = None,
        max_cached_shards: int = 2,
    ) -> None:
        """Initialize shard-backed dataset with a small LRU-like cache."""

        self.root_dir = Path(root_dir).resolve()
        self.split = split
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else (self.root_dir / f"manifest.{split}.jsonl").resolve()
        )
        self.max_cached_shards = max_cached_shards
        self._records = self._build_index()
        self._shard_array_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._shard_metadata_cache: OrderedDict[Path, list[Metadata]] = OrderedDict()

    def __len__(self) -> int:
        """Return number of samples indexed by the shard manifest."""

        return len(self._records)

    def __getitem__(self, index: int) -> RawSample:
        """Load one sample slice from a packed shard NPZ.

        Workflow:
        1. Load shard arrays/metadata from cache (or disk).
        2. Use offsets to slice flattened arrays into `[T, D]` tensors.
        3. Return a detached `RawSample` with source trace in metadata.
        """

        record = self._records[index]
        shard_arrays = self._get_cached_shard_arrays(record.shard_npz_path)
        shard_metadata = self._get_cached_shard_metadata(record.shard_jsonl_path)

        sample_index = record.sample_index
        lengths = shard_arrays["lengths"]
        num_series = shard_arrays["num_series"]
        series_offsets = shard_arrays["series_offsets"]
        time_offsets = shard_arrays["time_offsets"]

        if sample_index < 0 or sample_index >= len(lengths):
            raise IndexError(
                f"Sample index {sample_index} out of range for shard {record.shard_npz_path}"
            )
        if sample_index >= len(shard_metadata):
            raise IndexError(
                f"Metadata index {sample_index} out of range for shard {record.shard_jsonl_path}"
            )

        length = int(lengths[sample_index])
        num_dim = int(num_series[sample_index])
        flat_start = int(series_offsets[sample_index])
        flat_end = int(series_offsets[sample_index + 1])
        time_start = int(time_offsets[sample_index])
        time_end = int(time_offsets[sample_index + 1])

        # Data is stored flattened; offsets map each sample back to `[T, D]`.
        series = shard_arrays["series_values"][flat_start:flat_end].reshape(length, num_dim).copy()
        normal_series = (
            shard_arrays["normal_series_values"][flat_start:flat_end].reshape(length, num_dim).copy()
        )
        point_mask = (
            shard_arrays["point_mask_values"][flat_start:flat_end].reshape(length, num_dim).copy()
        )
        point_mask_any = shard_arrays["point_mask_any_values"][time_start:time_end].copy()

        metadata = dict(shard_metadata[sample_index])
        metadata["_source"] = {
            "format": "npz_shards_with_jsonl_metadata",
            "shard_npz_path": str(record.shard_npz_path),
            "shard_jsonl_path": str(record.shard_jsonl_path),
            "sample_index": sample_index,
        }

        return RawSample(
            sample_id=record.sample_id,
            split=self.split,
            series=series,
            point_mask=point_mask,
            point_mask_any=point_mask_any,
            normal_series=normal_series,
            metadata=metadata,
        )

    def _build_index(self) -> list[_ShardSampleRecord]:
        """Validate shard manifest and build random-access record list."""

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing shard manifest: {self.manifest_path}")

        records = list(_iter_shard_records_from_jsonl(self.manifest_path))
        if not records:
            raise FileNotFoundError(f"No shard records found in {self.manifest_path}")

        for record in records:
            if not record.shard_npz_path.exists():
                raise FileNotFoundError(f"Missing shard NPZ file: {record.shard_npz_path}")
            if not record.shard_jsonl_path.exists():
                raise FileNotFoundError(f"Missing shard JSONL file: {record.shard_jsonl_path}")

        return records

    def _get_cached_shard_arrays(self, shard_path: Path) -> dict[str, np.ndarray]:
        """Return shard arrays from cache and refresh recency."""

        cached = self._shard_array_cache.pop(shard_path, None)
        if cached is None:
            with np.load(shard_path, allow_pickle=False) as npz:
                cached = {name: np.asarray(npz[name]) for name in npz.files}
        self._shard_array_cache[shard_path] = cached
        self._evict_cache(self._shard_array_cache)
        return cached

    def _get_cached_shard_metadata(self, shard_jsonl_path: Path) -> list[Metadata]:
        """Return shard metadata rows from cache and refresh recency."""

        cached = self._shard_metadata_cache.pop(shard_jsonl_path, None)
        if cached is None:
            cached = [
                json.loads(line)
                for line in shard_jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self._shard_metadata_cache[shard_jsonl_path] = cached
        self._evict_cache(self._shard_metadata_cache)
        return cached

    def _evict_cache(self, cache: OrderedDict[Path, object]) -> None:
        """Evict oldest cache entries until the cache fits `max_cached_shards`."""

        while len(cache) > self.max_cached_shards:
            cache.popitem(last=False)


class ContextWindowDataset:
    """Adapt a raw-sample dataset into a flat window dataset.

    This version keeps only an index of `(raw_sample_index, window_index)` pairs.
    It avoids materializing all windows in memory, while still presenting a
    standard map-style dataset to the PyTorch DataLoader.
    """

    def __init__(
        self,
        raw_dataset: DatasetProtocol,
        windowizer: ContextWindowizerProtocol,
    ) -> None:
        """Build a flat `(sample_idx, window_idx)` index for window-level access."""

        self.raw_dataset = raw_dataset
        self.windowizer = windowizer
        self._index = self._build_index()

    def __len__(self) -> int:
        """Return number of windows across all raw samples."""

        return len(self._index)

    def __getitem__(self, index: int) -> ContextWindowSample:
        """Load one window by first loading its parent raw sample on demand."""

        raw_index, window_index = self._index[index]
        sample = self.raw_dataset[raw_index]
        windows = self.windowizer.transform(sample)
        return windows[window_index]

    def _build_index(self) -> list[tuple[int, int]]:
        """Precompute mapping from flat window index to raw-sample/window pair."""

        index: list[tuple[int, int]] = []
        for raw_index in range(len(self.raw_dataset)):
            sample = self.raw_dataset[raw_index]
            windows = self.windowizer.transform(sample)
            index.extend((raw_index, window_index) for window_index in range(len(windows)))

        if not index:
            raise FileNotFoundError(
                "No full context windows could be built from the provided dataset. "
                "Samples shorter than `context_size` and tail remainders are discarded."
            )

        return index


__all__ = ["ContextWindowDataset", "ShardedSyntheticTsadDataset", "SyntheticTsadDataset"]
