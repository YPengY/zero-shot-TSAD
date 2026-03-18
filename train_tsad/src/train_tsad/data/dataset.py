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
    length: int | None = None
    num_features: int | None = None


@dataclass(frozen=True, slots=True)
class _ShardSampleRecord:
    sample_id: str
    shard_npz_path: Path
    shard_jsonl_path: Path | None
    sample_index: int
    length: int | None = None
    num_features: int | None = None


@dataclass(frozen=True, slots=True)
class _ShardSampleGeometry:
    length: int
    num_dim: int
    sample_flat_start: int
    sample_flat_end: int
    sample_time_start: int
    sample_time_end: int


@dataclass(frozen=True, slots=True)
class _WindowRecord:
    raw_index: int
    context_start: int
    context_end: int


@dataclass(frozen=True, slots=True)
class _WindowShardRecord:
    sample_id: str
    source_sample_id: str
    shard_npz_path: Path
    window_index: int
    context_start: int
    context_end: int
    valid_length: int
    num_features: int | None = None


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


def _optional_int(value: object) -> int | None:
    """Return integer value when present and non-empty, otherwise `None`."""

    if value is None or value == "":
        return None
    return int(value)


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
        yield _RawSampleRecord(
            sample_id=sample_id,
            npz_path=npz_path,
            json_path=json_path,
            length=_optional_int(row.get("length")),
            num_features=_optional_int(row.get("num_features") or row.get("num_series")),
        )


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
            yield _RawSampleRecord(
                sample_id=sample_id,
                npz_path=npz_path,
                json_path=json_path,
                length=_optional_int(row.get("length")),
                num_features=_optional_int(row.get("num_features") or row.get("num_series")),
            )


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
        if shard_npz_path is None or sample_index is None:
            raise ValueError(
                f"Shard manifest entry in {manifest_path}:{line_number} must include "
                "'shard_npz_path' and 'sample_index'"
            )
        yield _ShardSampleRecord(
            sample_id=str(row.get("sample_id") or f"sample_{int(sample_index):06d}"),
            shard_npz_path=shard_npz_path,
            shard_jsonl_path=shard_jsonl_path,
            sample_index=int(sample_index),
            length=_optional_int(row.get("length")),
            num_features=_optional_int(row.get("num_features") or row.get("num_series")),
        )


def _iter_window_shard_records_from_jsonl(manifest_path: Path) -> Iterator[_WindowShardRecord]:
    """Yield window-level records from a window-packed dataset manifest."""

    base_dir = manifest_path.parent
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        shard_npz_path = _resolve_manifest_path(base_dir, row.get("shard_npz_path"))
        if shard_npz_path is None:
            raise ValueError(
                f"Window manifest entry in {manifest_path}:{line_number} must include `shard_npz_path`."
            )

        window_index_raw = row.get("window_index")
        if window_index_raw is None:
            window_index_raw = row.get("sample_index")
        if window_index_raw is None:
            raise ValueError(
                f"Window manifest entry in {manifest_path}:{line_number} must include "
                "`window_index` (or `sample_index`)."
            )
        window_index = int(window_index_raw)

        context_start_raw = row.get("context_start")
        context_end_raw = row.get("context_end")
        if context_start_raw is None or context_end_raw is None:
            raise ValueError(
                f"Window manifest entry in {manifest_path}:{line_number} must include "
                "`context_start` and `context_end`."
            )
        context_start = int(context_start_raw)
        context_end = int(context_end_raw)

        valid_length = _optional_int(row.get("valid_length"))
        if valid_length is None:
            valid_length = max(0, context_end - context_start)

        sample_id = str(
            row.get("sample_id")
            or row.get("window_id")
            or f"window_{line_number:08d}"
        )
        source_sample_id = str(row.get("source_sample_id") or sample_id)

        yield _WindowShardRecord(
            sample_id=sample_id,
            source_sample_id=source_sample_id,
            shard_npz_path=shard_npz_path,
            window_index=window_index,
            context_start=context_start,
            context_end=context_end,
            valid_length=int(valid_length),
            num_features=_optional_int(row.get("num_features") or row.get("num_series")),
        )


def _load_raw_sample(
    *,
    sample_id: str,
    split: SplitName,
    npz_path: Path,
    json_path: Path | None,
    load_normal_series: bool = True,
    load_metadata: bool = True,
) -> RawSample:
    """Load one sample from NPZ (+ optional JSON) into the canonical `RawSample`.

    Side effects:
        Reads sample files from disk. Does not cache by itself.
    """

    payload = _load_json_payload(json_path) if load_metadata else {}

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
        if load_normal_series and "normal_series" in npz.files:
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
        load_normal_series: bool = True,
        load_metadata: bool = True,
    ) -> None:
        """Build dataset index for per-sample NPZ/JSON files."""

        self.root_dir = Path(root_dir).resolve()
        self.split = split
        self.manifest_path = Path(manifest_path).resolve() if manifest_path is not None else None
        self.load_normal_series = load_normal_series
        self.load_metadata = load_metadata
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
            load_normal_series=self.load_normal_series,
            load_metadata=self.load_metadata,
        )

    def sample_id(self, index: int) -> str:
        """Return stable sample identifier without touching sample arrays."""

        return self._records[index].sample_id

    def sample_length(self, index: int) -> int:
        """Return sample sequence length, loading the sample only when manifest lacks it."""

        record = self._records[index]
        if record.length is not None:
            return int(record.length)
        return int(self[index].series.shape[0])

    def sample_num_features(self, index: int) -> int:
        """Return sample feature count, loading the sample only when manifest lacks it."""

        record = self._records[index]
        if record.num_features is not None:
            return int(record.num_features)
        return int(self[index].series.shape[1])

    def sample_shard_key(self, index: int) -> str | None:
        """Raw-file datasets do not expose shard locality."""

        _ = index
        return None

    def slice_window(
        self,
        index: int,
        *,
        start: int,
        end: int,
        windowizer: ContextWindowizerProtocol,
    ) -> ContextWindowSample:
        """Fallback fast-path API: load one sample then slice one window."""

        return windowizer.slice_window(self[index], start=start, end=end)

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
        load_normal_series: bool = True,
        load_metadata: bool = True,
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
        self.load_normal_series = load_normal_series
        self.load_metadata = load_metadata
        self._records = self._build_index()
        self.num_shards = len({record.shard_npz_path for record in self._records})
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
        sample_index = record.sample_index
        geometry = self._sample_geometry(
            shard_arrays=shard_arrays,
            sample_index=sample_index,
            shard_path=record.shard_npz_path,
        )

        # Data is stored flattened; offsets map each sample back to `[T, D]`.
        series = (
            shard_arrays["series_values"][geometry.sample_flat_start : geometry.sample_flat_end]
            .reshape(geometry.length, geometry.num_dim)
            .copy()
        )
        normal_series = None
        if self.load_normal_series and "normal_series_values" in shard_arrays:
            normal_series = (
                shard_arrays["normal_series_values"][
                    geometry.sample_flat_start : geometry.sample_flat_end
                ]
                .reshape(geometry.length, geometry.num_dim)
                .copy()
            )
        point_mask = (
            shard_arrays["point_mask_values"][
                geometry.sample_flat_start : geometry.sample_flat_end
            ]
            .reshape(geometry.length, geometry.num_dim)
            .copy()
        )
        if "point_mask_any_values" in shard_arrays:
            point_mask_any = shard_arrays["point_mask_any_values"][
                geometry.sample_time_start : geometry.sample_time_end
            ].copy()
        else:
            point_mask_any = (point_mask.max(axis=1) > 0).astype(np.uint8, copy=False)

        metadata: Metadata = {}
        if self.load_metadata:
            if record.shard_jsonl_path is None:
                raise FileNotFoundError(
                    "This dataset was packed without shard JSONL metadata, "
                    "but `load_metadata=True` was requested."
                )
            shard_metadata = self._get_cached_shard_metadata(record.shard_jsonl_path)
            if sample_index >= len(shard_metadata):
                raise IndexError(
                    f"Metadata index {sample_index} out of range for shard {record.shard_jsonl_path}"
                )
            metadata = dict(shard_metadata[sample_index])
        metadata["_source"] = {
            "format": "npz_shards_with_jsonl_metadata",
            "shard_npz_path": str(record.shard_npz_path),
            "shard_jsonl_path": str(record.shard_jsonl_path) if record.shard_jsonl_path else None,
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

    def slice_window(
        self,
        index: int,
        *,
        start: int,
        end: int,
        windowizer: ContextWindowizerProtocol,
    ) -> ContextWindowSample:
        """Read only one `(start, end)` region from shard arrays and build one window."""

        record = self._records[index]
        shard_arrays = self._get_cached_shard_arrays(record.shard_npz_path)
        geometry = self._sample_geometry(
            shard_arrays=shard_arrays,
            sample_index=record.sample_index,
            shard_path=record.shard_npz_path,
        )
        if start < 0 or end <= start or end > geometry.length:
            raise ValueError(
                f"Invalid window bounds [{start}, {end}) for sample length {geometry.length}."
            )

        window_length = end - start
        flat_start = geometry.sample_flat_start + (start * geometry.num_dim)
        flat_end = geometry.sample_flat_start + (end * geometry.num_dim)
        time_start = geometry.sample_time_start + start
        time_end = geometry.sample_time_start + end

        series_window = shard_arrays["series_values"][flat_start:flat_end].reshape(
            window_length,
            geometry.num_dim,
        )
        point_mask_window = shard_arrays["point_mask_values"][flat_start:flat_end].reshape(
            window_length,
            geometry.num_dim,
        )
        point_mask_any_window = (
            shard_arrays["point_mask_any_values"][time_start:time_end]
            if "point_mask_any_values" in shard_arrays
            else (point_mask_window.max(axis=1) > 0).astype(np.uint8, copy=False)
        )
        normal_series_window = None
        if self.load_normal_series and "normal_series_values" in shard_arrays:
            normal_series_window = shard_arrays["normal_series_values"][flat_start:flat_end].reshape(
                window_length,
                geometry.num_dim,
            )

        assemble_window = getattr(windowizer, "assemble_window", None)
        if callable(assemble_window):
            return assemble_window(
                sample_id=record.sample_id,
                split=self.split,
                context_start=start,
                context_end=end,
                series_window=series_window,
                point_mask_window=point_mask_window,
                point_mask_any_window=point_mask_any_window,
                normal_series_window=normal_series_window,
            )

        sample = RawSample(
            sample_id=record.sample_id,
            split=self.split,
            series=series_window,
            point_mask=point_mask_window,
            point_mask_any=point_mask_any_window,
            normal_series=normal_series_window,
            metadata={},
        )
        return windowizer.slice_window(sample, start=0, end=window_length)

    def sample_id(self, index: int) -> str:
        """Return manifest sample id without opening shard payloads."""

        return self._records[index].sample_id

    def sample_length(self, index: int) -> int:
        """Return sample sequence length from manifest."""

        record = self._records[index]
        if record.length is None:
            return int(self[index].series.shape[0])
        return int(record.length)

    def sample_num_features(self, index: int) -> int:
        """Return sample feature count from manifest."""

        record = self._records[index]
        if record.num_features is None:
            return int(self[index].series.shape[1])
        return int(record.num_features)

    def sample_shard_key(self, index: int) -> str | None:
        """Return shard path used to preserve shard locality during sampling."""

        return str(self._records[index].shard_npz_path)

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
            if self.load_metadata:
                if record.shard_jsonl_path is None:
                    raise FileNotFoundError(
                        "Manifest rows do not include `shard_jsonl_path`, "
                        "but `load_metadata=True` was requested."
                    )
                if not record.shard_jsonl_path.exists():
                    raise FileNotFoundError(f"Missing shard JSONL file: {record.shard_jsonl_path}")

        return records

    def _get_cached_shard_arrays(self, shard_path: Path) -> dict[str, np.ndarray]:
        """Return shard arrays from cache and refresh recency."""

        cached = self._shard_array_cache.pop(shard_path, None)
        if cached is None:
            required_names = {
                "series_values",
                "point_mask_values",
                "lengths",
                "num_series",
                "series_offsets",
                "time_offsets",
            }
            optional_names = {"point_mask_any_values"}
            if self.load_normal_series:
                required_names.add("normal_series_values")
            with np.load(shard_path, allow_pickle=False) as npz:
                missing = sorted(name for name in required_names if name not in npz.files)
                if missing:
                    raise KeyError(
                        f"Missing required arrays in shard {shard_path}: {', '.join(missing)}"
                    )
                cached = {
                    name: np.asarray(npz[name])
                    for name in required_names
                }
                for name in optional_names:
                    if name in npz.files:
                        cached[name] = np.asarray(npz[name])
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

    def _sample_geometry(
        self,
        *,
        shard_arrays: dict[str, np.ndarray],
        sample_index: int,
        shard_path: Path,
    ) -> _ShardSampleGeometry:
        lengths = shard_arrays["lengths"]
        num_series = shard_arrays["num_series"]
        series_offsets = shard_arrays["series_offsets"]
        time_offsets = shard_arrays["time_offsets"]

        if sample_index < 0 or sample_index >= len(lengths):
            raise IndexError(f"Sample index {sample_index} out of range for shard {shard_path}")

        length = int(lengths[sample_index])
        num_dim = int(num_series[sample_index])
        sample_flat_start = int(series_offsets[sample_index])
        sample_flat_end = int(series_offsets[sample_index + 1])
        sample_time_start = int(time_offsets[sample_index])
        sample_time_end = int(time_offsets[sample_index + 1])

        expected_flat_size = int(length * num_dim)
        if sample_flat_end - sample_flat_start != expected_flat_size:
            raise ValueError(
                f"Corrupt series offsets in shard {shard_path}: "
                f"sample_index={sample_index}, "
                f"expected_flat_size={expected_flat_size}, "
                f"actual={sample_flat_end - sample_flat_start}"
            )
        if sample_time_end - sample_time_start != length:
            raise ValueError(
                f"Corrupt time offsets in shard {shard_path}: "
                f"sample_index={sample_index}, expected_length={length}, "
                f"actual={sample_time_end - sample_time_start}"
            )

        return _ShardSampleGeometry(
            length=length,
            num_dim=num_dim,
            sample_flat_start=sample_flat_start,
            sample_flat_end=sample_flat_end,
            sample_time_start=sample_time_start,
            sample_time_end=sample_time_end,
        )

    def _evict_cache(self, cache: OrderedDict[Path, object]) -> None:
        """Evict oldest cache entries until the cache fits `max_cached_shards`."""

        while len(cache) > self.max_cached_shards:
            cache.popitem(last=False)


class WindowShardedTsadDataset:
    """Load window-packed shard corpora with direct `ContextWindowSample` outputs."""

    is_prewindowed = True

    def __init__(
        self,
        root_dir: str | Path,
        split: SplitName = "train",
        manifest_path: str | Path | None = None,
        max_cached_shards: int = 2,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.split = split
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else (self.root_dir / f"manifest.{split}.jsonl").resolve()
        )
        self.max_cached_shards = max_cached_shards
        self._records = self._build_index()
        self.num_shards = len({record.shard_npz_path for record in self._records})
        self._shard_array_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._sample_blocks, self._shard_blocks = self._build_blocks()

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> ContextWindowSample:
        record = self._records[index]
        shard_arrays = self._get_cached_shard_arrays(record.shard_npz_path)
        window_index = int(record.window_index)
        total_windows = int(shard_arrays["series_windows"].shape[0])
        if window_index < 0 or window_index >= total_windows:
            raise IndexError(
                f"Window index {window_index} out of range for shard {record.shard_npz_path}"
            )

        series = np.asarray(shard_arrays["series_windows"][window_index], dtype=np.float32).copy()
        point_mask = np.asarray(shard_arrays["point_mask_windows"][window_index], dtype=np.uint8).copy()
        patch_labels = np.asarray(shard_arrays["patch_labels_windows"][window_index], dtype=np.uint8).copy()
        valid_lengths = np.asarray(shard_arrays["valid_lengths"], dtype=np.int32)
        valid_length = int(valid_lengths[window_index])

        context_start = int(record.context_start)
        context_end = int(record.context_end)
        if "context_start" in shard_arrays:
            context_start = int(np.asarray(shard_arrays["context_start"], dtype=np.int32)[window_index])
        if "context_end" in shard_arrays:
            context_end = int(np.asarray(shard_arrays["context_end"], dtype=np.int32)[window_index])
        if context_end <= context_start:
            context_end = context_start + max(0, valid_length)

        max_window_length = int(series.shape[0])
        valid_length = max(0, min(valid_length, max_window_length))
        expected_valid_length = max(0, context_end - context_start)
        if expected_valid_length != valid_length:
            context_end = context_start + valid_length

        point_mask_any = np.zeros((max_window_length,), dtype=np.uint8)
        if valid_length > 0:
            point_mask_any[:valid_length] = (point_mask[:valid_length].sum(axis=1) > 0).astype(
                np.uint8,
                copy=False,
            )

        return ContextWindowSample(
            sample_id=record.sample_id,
            split=self.split,
            context_start=context_start,
            context_end=context_end,
            series=series,
            patch_labels=patch_labels,
            point_mask=point_mask,
            point_mask_any=point_mask_any,
            normal_series=None,
        )

    def sample_id(self, index: int) -> str:
        return self._records[index].sample_id

    def sample_num_features(self, index: int) -> int:
        record = self._records[index]
        if record.num_features is not None:
            return int(record.num_features)
        sample = self[index]
        return int(sample.series.shape[1])

    def sample_shard_key(self, index: int) -> str:
        return str(self._records[index].shard_npz_path)

    def grouped_blocks(self, strategy: str) -> tuple[tuple[int, ...], ...]:
        if strategy == "sample_block":
            return self._sample_blocks
        if strategy == "shard_block":
            return self._shard_blocks
        raise ValueError(
            f"Unsupported grouping strategy `{strategy}`. Supported: sample_block, shard_block."
        )

    def _build_index(self) -> list[_WindowShardRecord]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing window shard manifest: {self.manifest_path}")

        records = list(_iter_window_shard_records_from_jsonl(self.manifest_path))
        if not records:
            raise FileNotFoundError(f"No window shard records found in {self.manifest_path}")

        for record in records:
            if not record.shard_npz_path.exists():
                raise FileNotFoundError(f"Missing window shard NPZ file: {record.shard_npz_path}")
            if record.valid_length < 0:
                raise ValueError(
                    f"Invalid valid_length={record.valid_length} in {self.manifest_path}"
                )
        return records

    def _build_blocks(self) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        sample_blocks: list[tuple[int, ...]] = []
        shard_blocks: list[tuple[int, ...]] = []

        current_sample_id: str | None = None
        current_sample_block: list[int] = []
        current_shard_key: str | None = None
        current_shard_block: list[int] = []

        for index, record in enumerate(self._records):
            sample_id = str(record.source_sample_id)
            shard_key = str(record.shard_npz_path)

            if current_sample_id is None:
                current_sample_id = sample_id
            elif sample_id != current_sample_id:
                if current_sample_block:
                    sample_blocks.append(tuple(current_sample_block))
                current_sample_block = []
                current_sample_id = sample_id
            current_sample_block.append(index)

            if current_shard_key is None:
                current_shard_key = shard_key
            elif shard_key != current_shard_key:
                if current_shard_block:
                    shard_blocks.append(tuple(current_shard_block))
                current_shard_block = []
                current_shard_key = shard_key
            current_shard_block.append(index)

        if current_sample_block:
            sample_blocks.append(tuple(current_sample_block))
        if current_shard_block:
            shard_blocks.append(tuple(current_shard_block))

        return tuple(sample_blocks), tuple(shard_blocks)

    def _get_cached_shard_arrays(self, shard_path: Path) -> dict[str, np.ndarray]:
        cached = self._shard_array_cache.pop(shard_path, None)
        if cached is None:
            required_names = {
                "series_windows",
                "point_mask_windows",
                "patch_labels_windows",
                "valid_lengths",
            }
            optional_names = {"context_start", "context_end"}
            with np.load(shard_path, allow_pickle=False) as npz:
                missing = sorted(name for name in required_names if name not in npz.files)
                if missing:
                    raise KeyError(
                        f"Missing required arrays in window shard {shard_path}: {', '.join(missing)}"
                    )
                cached = {name: np.asarray(npz[name]) for name in required_names}
                for name in optional_names:
                    if name in npz.files:
                        cached[name] = np.asarray(npz[name])
        self._shard_array_cache[shard_path] = cached
        self._evict_cache(self._shard_array_cache)
        return cached

    def _evict_cache(self, cache: OrderedDict[Path, object]) -> None:
        while len(cache) > self.max_cached_shards:
            cache.popitem(last=False)


class ContextWindowDataset:
    """Adapt a raw-sample dataset into a flat window dataset.

    This version keeps only `(raw_sample_index, context_start, context_end)`
    records. When manifest lengths are available, it can build that index
    without loading the full samples first.
    """

    def __init__(
        self,
        raw_dataset: DatasetProtocol,
        windowizer: ContextWindowizerProtocol,
        *,
        enable_direct_window_read: bool = True,
    ) -> None:
        """Build a flat `(sample_idx, context_start, context_end)` window index."""

        self.raw_dataset = raw_dataset
        self.windowizer = windowizer
        self.enable_direct_window_read = bool(enable_direct_window_read)
        self._index, self._sample_blocks, self._shard_blocks = self._build_index()

    def __len__(self) -> int:
        """Return number of windows across all raw samples."""

        return len(self._index)

    def __getitem__(self, index: int) -> ContextWindowSample:
        """Load one window by slicing the already-indexed region on demand."""

        record = self._index[index]
        if self.enable_direct_window_read:
            dataset_slicer = getattr(self.raw_dataset, "slice_window", None)
            if callable(dataset_slicer):
                return dataset_slicer(
                    record.raw_index,
                    start=record.context_start,
                    end=record.context_end,
                    windowizer=self.windowizer,
                )
        sample = self.raw_dataset[record.raw_index]
        return self.windowizer.slice_window(
            sample,
            start=record.context_start,
            end=record.context_end,
        )

    def grouped_blocks(self, strategy: str) -> tuple[tuple[int, ...], ...]:
        """Return flat-index blocks used by locality-aware shuffling strategies."""

        if strategy == "sample_block":
            return self._sample_blocks
        if strategy == "shard_block":
            return self._shard_blocks
        raise ValueError(
            f"Unsupported grouping strategy `{strategy}`. Supported: sample_block, shard_block."
        )

    def _build_index(self) -> tuple[list[_WindowRecord], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        """Precompute flat window records and locality-preserving blocks."""

        index: list[_WindowRecord] = []
        sample_blocks: list[tuple[int, ...]] = []
        shard_blocks: list[tuple[int, ...]] = []
        current_shard_key: str | None = None
        current_shard_block: list[int] = []

        for raw_index in range(len(self.raw_dataset)):
            sample_length = self._sample_length(raw_index)
            sample_bounds = self.windowizer.iter_context_bounds(sample_length)
            if not sample_bounds:
                continue

            sample_window_indices: list[int] = []
            for context_start, context_end in sample_bounds:
                index.append(
                    _WindowRecord(
                        raw_index=raw_index,
                        context_start=context_start,
                        context_end=context_end,
                    )
                )
                sample_window_indices.append(len(index) - 1)

            sample_blocks.append(tuple(sample_window_indices))

            shard_key = self._sample_shard_key(raw_index) or f"sample:{self._sample_id(raw_index)}"
            if current_shard_key is None:
                current_shard_key = shard_key
            elif shard_key != current_shard_key:
                if current_shard_block:
                    shard_blocks.append(tuple(current_shard_block))
                current_shard_block = []
                current_shard_key = shard_key
            current_shard_block.extend(sample_window_indices)

        if current_shard_block:
            shard_blocks.append(tuple(current_shard_block))

        if not index:
            raise FileNotFoundError(
                "No full context windows could be built from the provided dataset. "
                "Samples shorter than `context_size` and tail remainders are discarded."
            )

        return index, tuple(sample_blocks), tuple(shard_blocks)

    def _sample_id(self, raw_index: int) -> str:
        getter = getattr(self.raw_dataset, "sample_id", None)
        if callable(getter):
            return str(getter(raw_index))
        return str(self.raw_dataset[raw_index].sample_id)

    def _sample_length(self, raw_index: int) -> int:
        getter = getattr(self.raw_dataset, "sample_length", None)
        if callable(getter):
            return int(getter(raw_index))
        return int(self.raw_dataset[raw_index].series.shape[0])

    def _sample_shard_key(self, raw_index: int) -> str | None:
        getter = getattr(self.raw_dataset, "sample_shard_key", None)
        if callable(getter):
            shard_key = getter(raw_index)
            return None if shard_key is None else str(shard_key)
        return None


__all__ = [
    "ContextWindowDataset",
    "ShardedSyntheticTsadDataset",
    "SyntheticTsadDataset",
    "WindowShardedTsadDataset",
]
