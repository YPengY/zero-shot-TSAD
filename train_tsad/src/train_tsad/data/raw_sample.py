from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..interfaces import Metadata, RawSample, SplitName


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
        raise ValueError(f"{name} length mismatch: expected {expected_length}, got {value.shape[0]}")
    return value


def _load_json_payload(json_path: Path | None) -> Metadata:
    """Load optional JSON metadata payload."""

    if json_path is None:
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _resolve_sample_id(npz_path: Path, payload: Metadata) -> str:
    """Resolve a stable sample id from metadata when available."""

    summary = payload.get("summary")
    if isinstance(summary, dict) and "sample_id" in summary:
        return f"sample_{int(summary['sample_id']):06d}"
    return npz_path.stem


def load_raw_sample(
    *,
    sample_id: str,
    split: SplitName,
    npz_path: Path,
    json_path: Path | None,
    load_normal_series: bool = True,
    load_metadata: bool = True,
) -> RawSample:
    """Load one raw sample from NPZ plus optional JSON metadata."""

    payload = _load_json_payload(json_path) if load_metadata else {}
    with np.load(npz_path, allow_pickle=False) as npz:
        series = _ensure_2d_array("series", np.asarray(npz["series"], dtype=np.float32))

        point_mask = None
        if "point_mask" in npz.files:
            point_mask = _ensure_2d_array("point_mask", np.asarray(npz["point_mask"], dtype=np.uint8))
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
                    f"normal_series shape mismatch for {npz_path}: expected {series.shape}, got {normal_series.shape}"
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
