from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..interfaces import (
    ContextWindowizerProtocol,
    ContextWindowSample,
    Metadata,
    RawSample,
    SplitName,
)
from .manifest import _iter_shard_records_from_jsonl, _iter_window_shard_records_from_jsonl
from .records import _ShardSampleGeometry, _ShardSampleRecord, _WindowShardRecord


class ShardedSyntheticTsadDataset:
    """Load shard-packed corpora produced by the dataset packer."""

    def __init__(
        self,
        root_dir: str | Path,
        split: SplitName = "train",
        manifest_path: str | Path | None = None,
        max_cached_shards: int = 2,
        load_normal_series: bool = True,
        load_metadata: bool = True,
    ) -> None:
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
        return len(self._records)

    def __getitem__(self, index: int) -> RawSample:
        record = self._records[index]
        shard_arrays = self._get_cached_shard_arrays(record.shard_npz_path)
        sample_index = record.sample_index
        geometry = self._sample_geometry(
            shard_arrays=shard_arrays,
            sample_index=sample_index,
            shard_path=record.shard_npz_path,
        )

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
            shard_arrays["point_mask_values"][geometry.sample_flat_start : geometry.sample_flat_end]
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
            normal_series_window = shard_arrays["normal_series_values"][
                flat_start:flat_end
            ].reshape(
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
        return self._records[index].sample_id

    def sample_length(self, index: int) -> int:
        record = self._records[index]
        if record.length is None:
            return int(self[index].series.shape[0])
        return int(record.length)

    def sample_num_features(self, index: int) -> int:
        record = self._records[index]
        if record.num_features is None:
            return int(self[index].series.shape[1])
        return int(record.num_features)

    def sample_shard_key(self, index: int) -> str | None:
        return str(self._records[index].shard_npz_path)

    def _build_index(self) -> list[_ShardSampleRecord]:
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
                cached = {name: np.asarray(npz[name]) for name in required_names}
                for name in optional_names:
                    if name in npz.files:
                        cached[name] = np.asarray(npz[name])
        self._shard_array_cache[shard_path] = cached
        self._evict_cache(self._shard_array_cache)
        return cached

    def _get_cached_shard_metadata(self, shard_jsonl_path: Path) -> list[Metadata]:
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
        while len(cache) > self.max_cached_shards:
            cache.popitem(last=False)


class WindowShardedTsadDataset:
    """Load pre-windowed shard corpora and expose `ContextWindowSample` items."""

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
        point_mask = np.asarray(
            shard_arrays["point_mask_windows"][window_index], dtype=np.uint8
        ).copy()
        patch_labels = np.asarray(
            shard_arrays["patch_labels_windows"][window_index],
            dtype=np.uint8,
        ).copy()
        valid_lengths = np.asarray(shard_arrays["valid_lengths"], dtype=np.int32)
        valid_length = int(valid_lengths[window_index])

        context_start = int(record.context_start)
        context_end = int(record.context_end)
        if "context_start" in shard_arrays:
            context_start = int(
                np.asarray(shard_arrays["context_start"], dtype=np.int32)[window_index]
            )
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

    def _build_blocks(self) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
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


__all__ = ["ShardedSyntheticTsadDataset", "WindowShardedTsadDataset"]
