from __future__ import annotations

from pathlib import Path

from ..interfaces import (
    ContextWindowizerProtocol,
    ContextWindowSample,
    RawSample,
    SplitName,
)
from .manifest import (
    _iter_raw_records_from_csv,
    _iter_raw_records_from_directory,
    _iter_raw_records_from_jsonl,
)
from .raw_sample import load_raw_sample
from .records import _RawSampleRecord


class SyntheticTsadDataset:
    """Load per-sample `sample_*.npz/json` files as `RawSample` objects."""

    def __init__(
        self,
        root_dir: str | Path,
        split: SplitName = "train",
        manifest_path: str | Path | None = None,
        load_normal_series: bool = True,
        load_metadata: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.split = split
        self.manifest_path = Path(manifest_path).resolve() if manifest_path is not None else None
        self.load_normal_series = load_normal_series
        self.load_metadata = load_metadata
        self._records = self._build_index()

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> RawSample:
        record = self._records[index]
        return load_raw_sample(
            sample_id=record.sample_id,
            split=self.split,
            npz_path=record.npz_path,
            json_path=record.json_path,
            load_normal_series=self.load_normal_series,
            load_metadata=self.load_metadata,
        )

    def sample_id(self, index: int) -> str:
        return self._records[index].sample_id

    def sample_length(self, index: int) -> int:
        record = self._records[index]
        if record.length is not None:
            return int(record.length)
        return int(self[index].series.shape[0])

    def sample_num_features(self, index: int) -> int:
        record = self._records[index]
        if record.num_features is not None:
            return int(record.num_features)
        return int(self[index].series.shape[1])

    def sample_shard_key(self, index: int) -> str | None:
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
        return windowizer.slice_window(self[index], start=start, end=end)

    def _build_index(self) -> list[_RawSampleRecord]:
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
        split_dir = self.root_dir / self.split
        return split_dir if split_dir.is_dir() else self.root_dir


__all__ = ["SyntheticTsadDataset"]
