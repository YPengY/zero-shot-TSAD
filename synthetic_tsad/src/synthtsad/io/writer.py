from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces import GenerationMetadata, LabelPayload
from .sharded import _build_patch_labels, _iter_context_bounds, _slice_or_pad_2d


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _build_sample_payload(
    *,
    sample_id: int,
    normal_series: np.ndarray,
    observed_series: np.ndarray,
    labels: LabelPayload,
    graph: Any,
    metadata: GenerationMetadata,
) -> dict[str, Any]:
    """Build stable JSON payload shared by raw and packed writers."""

    return {
        "summary": {
            "sample_id": int(sample_id),
            "length": int(observed_series.shape[0]),
            "num_series": int(observed_series.shape[1]),
            "is_anomalous_sample": int(labels["is_anomalous_sample"]),
        },
        "labels": {
            "root_cause": labels["root_cause"],
            "affected_nodes": labels["affected_nodes"],
            "summary": labels["summary"],
        },
        "events": labels["events"],
        "graph": {
            "num_nodes": int(graph.num_nodes),
            "adjacency": graph.adjacency.tolist(),
            "topo_order": [int(v) for v in graph.topo_order],
            "parents": [[int(u) for u in nodes] for nodes in graph.parents],
        },
        "metadata": metadata,
    }


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


class DatasetWriter:
    """Persist generated samples and metadata.

    NPZ payload:
    - series: anomalous/observed sequence [T, D]
    - normal_series: pre-anomaly reference [T, D]
    - point_mask: per-node anomaly labels [T, D]
    - point_mask_any: sequence-level mask [T]

    JSON payload:
    - event list, causal graph, sampled parameters and summary statistics.
    """

    def __init__(self, output_dir: Path, *, compress_arrays: bool = False) -> None:
        self.output_dir = output_dir
        self.compress_arrays = compress_arrays

    def write_sample(
        self,
        sample_id: int,
        normal_series: np.ndarray,
        observed_series: np.ndarray,
        labels: LabelPayload,
        graph,
        metadata: GenerationMetadata,
    ) -> None:
        npz_path = self.output_dir / f"sample_{sample_id:06d}.npz"
        json_path = self.output_dir / f"sample_{sample_id:06d}.json"

        savez = np.savez_compressed if self.compress_arrays else np.savez
        savez(
            npz_path,
            series=observed_series,
            normal_series=normal_series,
            point_mask=labels["point_mask"],
            point_mask_any=labels["point_mask_any"],
        )

        payload = _build_sample_payload(
            sample_id=sample_id,
            normal_series=normal_series,
            observed_series=observed_series,
            labels=labels,
            graph=graph,
            metadata=metadata,
        )
        json_path.write_text(
            json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class PackedDatasetWriter:
    """Stream generated samples directly into shard-packed dataset artifacts."""

    def __init__(
        self,
        output_root: Path,
        *,
        split: str,
        samples_per_shard: int = 512,
    ) -> None:
        if samples_per_shard <= 0:
            raise ValueError(f"`samples_per_shard` must be positive, got {samples_per_shard}.")

        self.output_root = output_root.resolve()
        self.split = split
        self.samples_per_shard = int(samples_per_shard)
        self.split_output_dir = self.output_root / self.split
        self.split_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.output_root / f"manifest.{self.split}.jsonl"
        self._manifest_handle = self.manifest_path.open("w", encoding="utf-8", newline="\n")

        self._buffer: list[dict[str, Any]] = []
        self._next_shard_index = 0

    def write_sample(
        self,
        sample_id: int,
        normal_series: np.ndarray,
        observed_series: np.ndarray,
        labels: LabelPayload,
        graph,
        metadata: GenerationMetadata,
    ) -> None:
        payload = _build_sample_payload(
            sample_id=sample_id,
            normal_series=normal_series,
            observed_series=observed_series,
            labels=labels,
            graph=graph,
            metadata=metadata,
        )
        self._buffer.append(
            {
                "sample_id": f"sample_{int(sample_id):06d}",
                "series": np.asarray(observed_series, dtype=np.float32),
                "normal_series": np.asarray(normal_series, dtype=np.float32),
                "point_mask": np.asarray(labels["point_mask"], dtype=np.uint8),
                "point_mask_any": np.asarray(labels["point_mask_any"], dtype=np.uint8),
                "payload": payload,
            }
        )
        if len(self._buffer) >= self.samples_per_shard:
            self._flush_shard()

    def close(self) -> None:
        """Flush any remaining samples and close manifest handle."""

        if self._buffer:
            self._flush_shard()
        self._manifest_handle.close()

    def _flush_shard(self) -> None:
        shard_index = self._next_shard_index
        self._next_shard_index += 1

        shard_npz_path = self.split_output_dir / f"shard-{shard_index:05d}.npz"
        shard_jsonl_path = self.split_output_dir / f"shard-{shard_index:05d}.jsonl"

        series_values: list[np.ndarray] = []
        normal_series_values: list[np.ndarray] = []
        point_mask_values: list[np.ndarray] = []
        point_mask_any_values: list[np.ndarray] = []
        series_offsets = [0]
        time_offsets = [0]
        lengths: list[int] = []
        num_series_list: list[int] = []

        with shard_jsonl_path.open("w", encoding="utf-8", newline="\n") as shard_jsonl_handle:
            for sample_index, record in enumerate(self._buffer):
                series = np.asarray(record["series"], dtype=np.float32)
                normal_series = np.asarray(record["normal_series"], dtype=np.float32)
                point_mask = np.asarray(record["point_mask"], dtype=np.uint8)
                point_mask_any = np.asarray(record["point_mask_any"], dtype=np.uint8)
                payload = dict(record["payload"])

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

                shard_line = payload
                shard_line["sample_id"] = str(record["sample_id"])
                shard_line["split"] = self.split
                shard_line["sample_index"] = sample_index
                shard_jsonl_handle.write(
                    json.dumps(_to_jsonable(shard_line), ensure_ascii=False) + "\n"
                )

                manifest_row = {
                    "sample_id": str(record["sample_id"]),
                    "split": self.split,
                    "shard_id": shard_index,
                    "sample_index": sample_index,
                    "shard_npz_path": str(shard_npz_path.relative_to(self.output_root)),
                    "shard_jsonl_path": str(shard_jsonl_path.relative_to(self.output_root)),
                    "length": length,
                    "num_series": num_series,
                    "is_anomalous_sample": int(payload.get("summary", {}).get("is_anomalous_sample", int(point_mask_any.any()))),
                    "num_events": int(len(payload.get("events", []))),
                    "anomaly_point_ratio": float(point_mask_any.mean()) if point_mask_any.size else 0.0,
                }
                self._manifest_handle.write(
                    json.dumps(_to_jsonable(manifest_row), ensure_ascii=False) + "\n"
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
        self._buffer.clear()


class PackedWindowDatasetWriter:
    """Stream generated samples directly into window-level shard artifacts."""

    def __init__(
        self,
        output_root: Path,
        *,
        split: str,
        context_size: int = 1024,
        patch_size: int = 16,
        stride: int | None = None,
        include_tail: bool = True,
        pad_short_sequences: bool = True,
        windows_per_shard: int = 4096,
        write_debug_sidecar: bool = True,
        min_patch_positive_ratio: float | None = None,
        min_anomaly_point_ratio: float | None = None,
    ) -> None:
        if context_size <= 0:
            raise ValueError(f"`context_size` must be positive, got {context_size}.")
        if patch_size <= 0:
            raise ValueError(f"`patch_size` must be positive, got {patch_size}.")
        if context_size % patch_size != 0:
            raise ValueError("`context_size` must be divisible by `patch_size`.")
        resolved_stride = context_size if stride is None else int(stride)
        if resolved_stride <= 0:
            raise ValueError(f"`stride` must be positive, got {resolved_stride}.")
        if windows_per_shard <= 0:
            raise ValueError(f"`windows_per_shard` must be positive, got {windows_per_shard}.")
        if min_patch_positive_ratio is not None and float(min_patch_positive_ratio) < 0.0:
            raise ValueError(
                "`min_patch_positive_ratio` must be >= 0 when provided, "
                f"got {min_patch_positive_ratio}."
            )
        if min_anomaly_point_ratio is not None and float(min_anomaly_point_ratio) < 0.0:
            raise ValueError(
                "`min_anomaly_point_ratio` must be >= 0 when provided, "
                f"got {min_anomaly_point_ratio}."
            )

        self.output_root = output_root.resolve()
        self.split = split
        self.context_size = int(context_size)
        self.patch_size = int(patch_size)
        self.stride = int(resolved_stride)
        self.include_tail = bool(include_tail)
        self.pad_short_sequences = bool(pad_short_sequences)
        self.windows_per_shard = int(windows_per_shard)
        self.write_debug_sidecar = bool(write_debug_sidecar)
        self.min_patch_positive_ratio = (
            None if min_patch_positive_ratio is None else float(min_patch_positive_ratio)
        )
        self.min_anomaly_point_ratio = (
            None if min_anomaly_point_ratio is None else float(min_anomaly_point_ratio)
        )

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.split_output_dir = self.output_root / self.split
        self.split_output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.output_root / f"manifest.{self.split}.jsonl"
        self._manifest_handle = self.manifest_path.open("w", encoding="utf-8", newline="\n")

        self._debug_handle = None
        self._debug_sidecar_path = self.output_root / f"debug.{self.split}.jsonl"
        if self.write_debug_sidecar:
            self._debug_handle = self._debug_sidecar_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            )

        self._series_windows_buffer: list[np.ndarray] = []
        self._point_mask_windows_buffer: list[np.ndarray] = []
        self._patch_labels_windows_buffer: list[np.ndarray] = []
        self._valid_lengths_buffer: list[int] = []
        self._context_start_buffer: list[int] = []
        self._context_end_buffer: list[int] = []
        self._manifest_meta_buffer: list[dict[str, Any]] = []
        self._debug_meta_buffer: list[dict[str, Any] | None] = []
        self._next_shard_index = 0
        self._next_debug_row_index = 0

    def write_sample(
        self,
        sample_id: int,
        normal_series: np.ndarray,
        observed_series: np.ndarray,
        labels: LabelPayload,
        graph,
        metadata: GenerationMetadata,
    ) -> None:
        normal_series_arr = np.asarray(normal_series, dtype=np.float32)
        sample_series = np.asarray(observed_series, dtype=np.float32)
        sample_point_mask = np.asarray(labels["point_mask"], dtype=np.uint8)
        if sample_series.ndim != 2:
            raise ValueError(
                f"`observed_series` must be [T, D], got shape {sample_series.shape}."
            )
        if normal_series_arr.shape != sample_series.shape:
            raise ValueError(
                "`normal_series` shape mismatch: "
                f"expected {sample_series.shape}, got {normal_series_arr.shape}."
            )
        if sample_point_mask.shape != sample_series.shape:
            raise ValueError(
                "`labels['point_mask']` shape mismatch: "
                f"expected {sample_series.shape}, got {sample_point_mask.shape}."
            )

        source_sample_id = f"sample_{int(sample_id):06d}"
        sample_payload = _build_sample_payload(
            sample_id=sample_id,
            normal_series=normal_series_arr,
            observed_series=observed_series,
            labels=labels,
            graph=graph,
            metadata=metadata,
        )
        sequence_length = int(sample_series.shape[0])
        num_series = int(sample_series.shape[1])

        bounds = _iter_context_bounds(
            sequence_length,
            context_size=self.context_size,
            stride=self.stride,
            include_tail=self.include_tail,
            pad_short_sequences=self.pad_short_sequences,
        )
        for context_start, context_end in bounds:
            valid_length = int(context_end - context_start)
            if valid_length <= 0:
                continue

            window_series = _slice_or_pad_2d(
                sample_series,
                start=context_start,
                end=context_end,
                target_length=self.context_size,
                dtype=np.float32,
                pad_value=0.0,
            )
            window_point_mask = _slice_or_pad_2d(
                sample_point_mask,
                start=context_start,
                end=context_end,
                target_length=self.context_size,
                dtype=np.uint8,
                pad_value=0,
            )
            patch_labels = _build_patch_labels(window_point_mask, patch_size=self.patch_size)
            valid_point_mask = window_point_mask[:valid_length]
            is_anomalous_window = int(bool(valid_point_mask.any()))
            anomaly_point_ratio = float(valid_point_mask.mean()) if valid_point_mask.size else 0.0
            patch_positive_ratio = float(patch_labels.mean()) if patch_labels.size else 0.0
            if not _passes_window_ratio_filter(
                patch_positive_ratio=patch_positive_ratio,
                anomaly_point_ratio=anomaly_point_ratio,
                min_patch_positive_ratio=self.min_patch_positive_ratio,
                min_anomaly_point_ratio=self.min_anomaly_point_ratio,
            ):
                continue

            window_id = f"{source_sample_id}__w{context_start:06d}_{context_end:06d}"
            self._series_windows_buffer.append(window_series)
            self._point_mask_windows_buffer.append(window_point_mask)
            self._patch_labels_windows_buffer.append(patch_labels)
            self._valid_lengths_buffer.append(valid_length)
            self._context_start_buffer.append(int(context_start))
            self._context_end_buffer.append(int(context_end))
            self._manifest_meta_buffer.append(
                {
                    "sample_id": window_id,
                    "source_sample_id": source_sample_id,
                    "context_start": int(context_start),
                    "context_end": int(context_end),
                    "valid_length": int(valid_length),
                    "num_series": num_series,
                    "is_anomalous_window": is_anomalous_window,
                    "anomaly_point_ratio": float(anomaly_point_ratio),
                    "patch_positive_ratio": float(patch_positive_ratio),
                }
            )
            if self.write_debug_sidecar:
                self._debug_meta_buffer.append(
                    {
                        "sample_id": window_id,
                        "source_sample_id": source_sample_id,
                        "split": self.split,
                        "context_start": int(context_start),
                        "context_end": int(context_end),
                        "source_sample_index": int(sample_id),
                        "source_metadata": sample_payload,
                    }
                )
            else:
                self._debug_meta_buffer.append(None)

            if len(self._series_windows_buffer) >= self.windows_per_shard:
                self._flush_shard()

    def close(self) -> None:
        if self._series_windows_buffer:
            self._flush_shard()
        self._manifest_handle.close()
        if self._debug_handle is not None:
            self._debug_handle.close()

    def _flush_shard(self) -> None:
        if not self._series_windows_buffer:
            return

        shard_index = self._next_shard_index
        self._next_shard_index += 1
        shard_npz_path = self.split_output_dir / f"shard-{shard_index:05d}.npz"
        np.savez_compressed(
            shard_npz_path,
            series_windows=np.stack(self._series_windows_buffer, axis=0).astype(
                np.float32,
                copy=False,
            ),
            point_mask_windows=np.stack(self._point_mask_windows_buffer, axis=0).astype(
                np.uint8,
                copy=False,
            ),
            patch_labels_windows=np.stack(self._patch_labels_windows_buffer, axis=0).astype(
                np.uint8,
                copy=False,
            ),
            valid_lengths=np.asarray(self._valid_lengths_buffer, dtype=np.int32),
            context_start=np.asarray(self._context_start_buffer, dtype=np.int32),
            context_end=np.asarray(self._context_end_buffer, dtype=np.int32),
        )

        shard_npz_rel = str(shard_npz_path.relative_to(self.output_root))
        debug_sidecar_rel = str(self._debug_sidecar_path.relative_to(self.output_root))
        for window_index, row_meta in enumerate(self._manifest_meta_buffer):
            manifest_row = {
                "sample_id": row_meta["sample_id"],
                "source_sample_id": row_meta["source_sample_id"],
                "split": self.split,
                "shard_id": int(shard_index),
                "window_index": int(window_index),
                "sample_index": int(window_index),
                "shard_npz_path": shard_npz_rel,
                "context_start": int(row_meta["context_start"]),
                "context_end": int(row_meta["context_end"]),
                "valid_length": int(row_meta["valid_length"]),
                "length": int(self.context_size),
                "num_series": int(row_meta["num_series"]),
                "is_anomalous_sample": int(row_meta["is_anomalous_window"]),
                "anomaly_point_ratio": float(row_meta["anomaly_point_ratio"]),
                "patch_positive_ratio": float(row_meta["patch_positive_ratio"]),
            }
            debug_row = self._debug_meta_buffer[window_index]
            if self._debug_handle is not None and debug_row is not None:
                manifest_row["debug_jsonl_path"] = debug_sidecar_rel
                manifest_row["debug_row_index"] = int(self._next_debug_row_index)
                self._debug_handle.write(json.dumps(_to_jsonable(debug_row), ensure_ascii=False) + "\n")
                self._next_debug_row_index += 1
            self._manifest_handle.write(json.dumps(_to_jsonable(manifest_row), ensure_ascii=False) + "\n")

        self._series_windows_buffer.clear()
        self._point_mask_windows_buffer.clear()
        self._patch_labels_windows_buffer.clear()
        self._valid_lengths_buffer.clear()
        self._context_start_buffer.clear()
        self._context_end_buffer.clear()
        self._manifest_meta_buffer.clear()
        self._debug_meta_buffer.clear()
