from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
