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


def discover_input_splits(input_root: Path, split: SplitName | None = None) -> dict[SplitName, Path]:
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
                    "shard_jsonl_path": str(shard_jsonl_path.relative_to(shard_jsonl_path.parents[1])),
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
        raise ValueError(
            f"{sample_id}: point_mask_any must be [T], got {point_mask_any.shape}"
        )


def _write_dataset_meta(
    *,
    output_root: Path,
    dataset_name: str,
    dataset_version: str,
    samples_per_shard: int,
    split_reports: dict[SplitName, SplitPackStats],
) -> None:
    total_samples = sum(report.num_samples for report in split_reports.values())
    total_points = sum(report.total_points for report in split_reports.values())
    total_shards = sum(report.num_shards for report in split_reports.values())

    payload = {
        "dataset_name": dataset_name,
        "version": dataset_version,
        "storage_format": {
            "type": "npz_shards_with_jsonl_metadata",
            "samples_per_shard": int(samples_per_shard),
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
    "pack_synthetic_corpus",
]
