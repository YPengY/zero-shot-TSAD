from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from .records import _RawSampleRecord, _ShardSampleRecord, _WindowShardRecord


def _resolve_manifest_path(base_dir: Path, raw_path: str | None) -> Path | None:
    """Resolve one relative manifest path against its parent directory."""

    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _optional_int(value: object) -> int | None:
    """Return an integer when present and non-empty, otherwise `None`."""

    if value is None or value == "":
        return None
    return int(value)


def _iter_raw_records_from_directory(base_dir: Path) -> Iterator[_RawSampleRecord]:
    """Yield sample records from `sample_*.npz` files in one directory."""

    for npz_path in sorted(base_dir.glob("sample_*.npz")):
        json_path = npz_path.with_suffix(".json")
        yield _RawSampleRecord(
            sample_id=npz_path.stem,
            npz_path=npz_path,
            json_path=json_path if json_path.exists() else None,
        )


def _iter_raw_records_from_jsonl(manifest_path: Path) -> Iterator[_RawSampleRecord]:
    """Yield raw sample records described by a JSONL manifest."""

    base_dir = manifest_path.parent
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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
    """Yield raw sample records described by a CSV manifest."""

    base_dir = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
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
    """Yield shard-backed sample records described by a JSONL manifest."""

    base_dir = manifest_path.parent
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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
    """Yield pre-windowed shard records from a JSONL manifest."""

    base_dir = manifest_path.parent
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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

        context_start_raw = row.get("context_start")
        context_end_raw = row.get("context_end")
        if context_start_raw is None or context_end_raw is None:
            raise ValueError(
                f"Window manifest entry in {manifest_path}:{line_number} must include "
                "`context_start` and `context_end`."
            )
        valid_length = _optional_int(row.get("valid_length"))
        if valid_length is None:
            valid_length = max(0, int(context_end_raw) - int(context_start_raw))

        sample_id = str(row.get("sample_id") or row.get("window_id") or f"window_{line_number:08d}")
        source_sample_id = str(row.get("source_sample_id") or sample_id)
        yield _WindowShardRecord(
            sample_id=sample_id,
            source_sample_id=source_sample_id,
            shard_npz_path=shard_npz_path,
            window_index=int(window_index_raw),
            context_start=int(context_start_raw),
            context_end=int(context_end_raw),
            valid_length=int(valid_length),
            num_features=_optional_int(row.get("num_features") or row.get("num_series")),
        )
