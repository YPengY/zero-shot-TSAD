from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
TRAIN_TSAD_ROOT = APP_ROOT.parents[1]
PROJECT_ROOT = TRAIN_TSAD_ROOT.parent
SYNTH_ROOT = PROJECT_ROOT / "synthetic_tsad"
STUDIO_APP_ROOT = SYNTH_ROOT / "apps" / "tsad_studio"

if str(STUDIO_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDIO_APP_ROOT))

from studio_core import get_bootstrap_payload, import_config_text, preview_sample, randomize_config

PREVIEW_CACHE_LIMIT = 12
JOB_LOG_LIMIT = 2500


@dataclass(slots=True)
class JobState:
    job_id: str
    kind: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    logs: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


JOBS: dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()
PREVIEW_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
PREVIEW_LOCK = threading.Lock()


def _json_clone(payload: Any) -> Any:
    return json.loads(json.dumps(payload))


def _resolve_python_executable() -> Path:
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        TRAIN_TSAD_ROOT / ".venv" / "Scripts" / "python.exe",
        SYNTH_ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path(sys.executable)


def _resolve_path_like(path_like: str) -> Path:
    raw_path = Path(path_like).expanduser()
    if not raw_path.is_absolute():
        raw_path = (PROJECT_ROOT / raw_path).resolve()
    else:
        raw_path = raw_path.resolve()
    return raw_path


def _job_to_dict(job: JobState) -> dict[str, Any]:
    progress = None
    progress_path = job.artifacts.get("progress_path")
    if isinstance(progress_path, str):
        progress = _read_json_artifact(Path(progress_path))
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "logs": job.logs,
        "artifacts": job.artifacts,
        "progress": progress,
        "result": job.result,
        "error": job.error,
    }


def _create_job(kind: str) -> JobState:
    job = JobState(job_id=uuid.uuid4().hex[:12], kind=kind)
    with JOBS_LOCK:
        JOBS[job.job_id] = job
    return job


def _append_log(job: JobState, line: str) -> None:
    text = line.rstrip()
    if not text:
        return
    with JOBS_LOCK:
        job.logs.append(text)
        if len(job.logs) > JOB_LOG_LIMIT:
            del job.logs[:500]


def _read_json_artifact(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _run_subprocess(cmd: list[str], *, cwd: Path, job: JobState) -> None:
    _append_log(job, f"$ {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _append_log(job, line)
    rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def _cache_preview(preview: dict[str, Any]) -> str:
    preview_id = uuid.uuid4().hex[:12]
    with PREVIEW_LOCK:
        PREVIEW_CACHE[preview_id] = preview
        PREVIEW_CACHE.move_to_end(preview_id)
        while len(PREVIEW_CACHE) > PREVIEW_CACHE_LIMIT:
            PREVIEW_CACHE.popitem(last=False)
    return preview_id


def _get_cached_preview(preview_id: str) -> dict[str, Any] | None:
    with PREVIEW_LOCK:
        preview = PREVIEW_CACHE.get(preview_id)
        if preview is None:
            return None
        PREVIEW_CACHE.move_to_end(preview_id)
        return preview


def _preview_with_seed_offset(raw_config: dict[str, Any], seed_offset: int = 0) -> dict[str, Any]:
    preview_config = _json_clone(raw_config)
    base_seed = int(preview_config.get("seed", 0) or 0)
    preview_config["seed"] = base_seed + int(seed_offset)
    return preview_sample(preview_config)


def _summarize_preview(preview_id: str, preview: dict[str, Any], *, seed: int) -> dict[str, Any]:
    summary = preview.get("summary", {}) if isinstance(preview, dict) else {}
    return {
        "preview_id": preview_id,
        "seed": seed,
        "length": int(summary.get("length", 0)),
        "num_series": int(summary.get("num_series", 0)),
        "is_anomalous_sample": int(summary.get("is_anomalous_sample", 0)),
        "num_events": int(summary.get("num_events", 0)),
        "num_local_events": int(summary.get("num_local_events", 0)),
        "num_seasonal_events": int(summary.get("num_seasonal_events", 0)),
        "num_endogenous_events": int(summary.get("num_endogenous_events", 0)),
    }


def _iter_context_bounds(
    sequence_length: int,
    *,
    context_size: int,
    stride: int,
    include_tail: bool,
) -> list[tuple[int, int]]:
    _ = stride
    _ = include_tail
    if sequence_length <= 0:
        return []
    if sequence_length < context_size:
        return []

    usable_length = (sequence_length // context_size) * context_size
    return [(start, start + context_size) for start in range(0, usable_length, context_size)]


def _resolve_packed_root(path_like: str) -> Path:
    raw_path = _resolve_path_like(path_like)
    if (raw_path / "manifest.train.jsonl").exists():
        return raw_path
    if (raw_path / "data_packed" / "manifest.train.jsonl").exists():
        return raw_path / "data_packed"
    raise FileNotFoundError(
        "Could not locate packed dataset root. Expected manifest.train.jsonl in "
        f"`{raw_path}` or `{raw_path / 'data_packed'}`."
    )


def _resolve_run_root(path_like: str) -> Path:
    raw_path = _resolve_path_like(path_like)
    if (raw_path / "data_packed" / "manifest.train.jsonl").exists():
        return raw_path
    if (raw_path / "manifest.train.jsonl").exists() and raw_path.name == "data_packed":
        return raw_path.parent
    if (raw_path / "manifest.train.jsonl").exists():
        return raw_path
    raise FileNotFoundError(
        "Could not locate run root. Expected a run directory with `data_packed/` or a packed dataset root."
    )


def _count_manifest_rows(manifest_path: Path) -> int:
    return sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_manifest_rows(packed_root: Path, split: str) -> list[dict[str, Any]]:
    manifest_path = packed_root / f"manifest.{split}.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_shard_metadata(shard_jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in shard_jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_sample_from_manifest_row(*, packed_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    shard_npz_path = packed_root / str(row["shard_npz_path"])
    shard_jsonl_path = packed_root / str(row["shard_jsonl_path"])
    sample_index = int(row["sample_index"])

    with np.load(shard_npz_path, allow_pickle=False) as npz:
        lengths = np.asarray(npz["lengths"], dtype=np.int32)
        num_series = np.asarray(npz["num_series"], dtype=np.int32)
        series_offsets = np.asarray(npz["series_offsets"], dtype=np.int64)
        time_offsets = np.asarray(npz["time_offsets"], dtype=np.int64)

        length = int(lengths[sample_index])
        dims = int(num_series[sample_index])
        flat_start = int(series_offsets[sample_index])
        flat_end = int(series_offsets[sample_index + 1])
        time_start = int(time_offsets[sample_index])
        time_end = int(time_offsets[sample_index + 1])

        series = np.asarray(npz["series_values"][flat_start:flat_end], dtype=np.float32).reshape(length, dims)
        normal_series = np.asarray(npz["normal_series_values"][flat_start:flat_end], dtype=np.float32).reshape(length, dims)
        point_mask = np.asarray(npz["point_mask_values"][flat_start:flat_end], dtype=np.uint8).reshape(length, dims)
        point_mask_any = np.asarray(npz["point_mask_any_values"][time_start:time_end], dtype=np.uint8)

    metadata_rows = _load_shard_metadata(shard_jsonl_path)
    metadata = metadata_rows[sample_index] if sample_index < len(metadata_rows) else {}
    return {
        "length": length,
        "num_series": dims,
        "series": series,
        "normal_series": normal_series,
        "point_mask": point_mask,
        "point_mask_any": point_mask_any,
        "metadata": metadata,
    }


def _build_run_info(path_like: str) -> dict[str, Any]:
    run_root = _resolve_run_root(path_like)
    packed_root = _resolve_packed_root(str(run_root))
    available_splits: list[str] = []
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        manifest_path = packed_root / f"manifest.{split}.jsonl"
        if manifest_path.exists():
            available_splits.append(split)
            split_counts[split] = _count_manifest_rows(manifest_path)

    config_dir = run_root / "configs"
    config_candidates = [
        config_dir / "train_generated.json",
        config_dir / "train_mini.json",
        config_dir / "train.json",
        config_dir / "train_config.json",
    ]
    if config_dir.exists():
        config_candidates.extend(sorted(config_dir.glob("*.json")))

    train_config_path = next((path for path in config_candidates if path.exists()), None)
    train_output_dir = run_root / "train_out"
    if train_config_path is not None:
        try:
            config_payload = json.loads(train_config_path.read_text(encoding="utf-8"))
            output_dir = config_payload.get("train", {}).get("output_dir")
            if output_dir:
                train_output_dir = _resolve_path_like(str(output_dir))
        except Exception:
            pass

    return {
        "run_root": str(run_root),
        "packed_root": str(packed_root),
        "available_splits": available_splits,
        "split_counts": split_counts,
        "train_config_path": str(train_config_path) if train_config_path is not None else None,
        "train_output_dir": str(train_output_dir),
        "data_quality_report": str(train_output_dir / "data_quality_report.json"),
        "history_path": str(train_output_dir / "history.json"),
        "summary_path": str(train_output_dir / "summary.json"),
        "progress_path": str(train_output_dir / "progress.json"),
    }


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _run_generation_job(payload: dict[str, Any], job: JobState) -> dict[str, Any]:
    python_exe = _resolve_python_executable()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = str(payload.get("run_name", f"workbench_run_{timestamp}")).strip() or f"workbench_run_{timestamp}"
    run_root_input = str(payload.get("run_root", "")).strip()
    run_root = _resolve_path_like(run_root_input) if run_root_input else (PROJECT_ROOT / "runs" / run_name)

    overwrite = bool(payload.get("overwrite_run", False))
    confirm_overwrite = bool(payload.get("confirm_overwrite", False))
    if run_root.exists() and overwrite:
        if not confirm_overwrite:
            raise FileExistsError(
                f"Run root already exists and overwrite was not confirmed: {run_root}"
            )
        shutil.rmtree(run_root)
    elif run_root.exists() and not overwrite:
        raise FileExistsError(f"Run root already exists: {run_root}")

    raw_root = run_root / "data_raw"
    packed_root = run_root / "data_packed"
    config_root = run_root / "configs"
    train_out = run_root / "train_out"
    eval_out = run_root / "eval"

    for split in ("train", "val", "test"):
        (raw_root / split).mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    train_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    synthetic_config = payload.get("config")
    if not isinstance(synthetic_config, dict):
        raise ValueError("Generation requires the full edited synthetic config.")

    synthetic_config_path = config_root / "synthetic_runtime_config.json"
    synthetic_config_path.write_text(
        json.dumps(synthetic_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split_counts = {
        "train": int(payload.get("train_samples", 10000)),
        "val": int(payload.get("val_samples", 1500)),
        "test": int(payload.get("test_samples", 1500)),
    }
    seed_base = int(payload.get("seed_base", synthetic_config.get("seed", 0) or 0))

    for split_index, split in enumerate(("train", "val", "test"), start=1):
        cmd = [
            str(python_exe),
            "-B",
            str(SYNTH_ROOT / "scripts" / "generate_dataset.py"),
            "--config",
            str(synthetic_config_path),
            "--output",
            str(raw_root / split),
            "--num-samples",
            str(split_counts[split]),
            "--seed",
            str(seed_base + split_index),
        ]
        _run_subprocess(cmd, cwd=PROJECT_ROOT, job=job)

    pack_cmd = [
        str(python_exe),
        "-B",
        str(SYNTH_ROOT / "scripts" / "pack_dataset.py"),
        "--input",
        str(raw_root),
        "--output",
        str(packed_root),
        "--samples-per-shard",
        str(int(payload.get("samples_per_shard", 128))),
        "--overwrite",
        "--dataset-name",
        str(payload.get("dataset_name", "workbench_tsad")),
        "--dataset-version",
        str(payload.get("dataset_version", "v1")),
    ]
    _run_subprocess(pack_cmd, cwd=PROJECT_ROOT, job=job)

    template_name = str(payload.get("train_template", "timercd_small.json"))
    train_template_path = TRAIN_TSAD_ROOT / "configs" / template_name
    if not train_template_path.exists():
        raise FileNotFoundError(f"Missing training template: {train_template_path}")
    train_config = json.loads(train_template_path.read_text(encoding="utf-8"))

    train_config["data"]["dataset_root"] = str(packed_root)
    train_config["train"]["output_dir"] = str(train_out)
    if payload.get("train_device"):
        train_config["train"]["device"] = str(payload["train_device"])
    if payload.get("train_max_epochs") is not None:
        train_config["train"]["max_epochs"] = int(payload["train_max_epochs"])
    if payload.get("train_batch_size") is not None:
        train_config["data"]["batch_size"] = int(payload["train_batch_size"])
    if payload.get("eval_batch_size") is not None:
        train_config["data"]["eval_batch_size"] = int(payload["eval_batch_size"])

    train_config_path = config_root / "train_generated.json"
    train_config_path.write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")

    result = _build_run_info(str(run_root))
    result.update(
        {
            "raw_root": str(raw_root),
            "synthetic_config_path": str(synthetic_config_path),
            "train_config_path": str(train_config_path),
            "eval_output_dir": str(eval_out),
        }
    )
    return result


def _run_train_job(payload: dict[str, Any], job: JobState) -> dict[str, Any]:
    python_exe = _resolve_python_executable()

    config_path_raw = str(payload.get("config_path", "")).strip()
    if not config_path_raw:
        raise ValueError("config_path is required")
    config_path = _resolve_path_like(config_path_raw)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = config_payload.get("train", {}).get("output_dir")
    output_dir_str = str(_resolve_path_like(str(output_dir))) if output_dir else None
    if output_dir_str:
        with JOBS_LOCK:
            job.artifacts = {
                "config_path": str(config_path),
                "output_dir": output_dir_str,
                "data_quality_report": str(Path(output_dir_str) / "data_quality_report.json"),
                "history_path": str(Path(output_dir_str) / "history.json"),
                "summary_path": str(Path(output_dir_str) / "summary.json"),
                "progress_path": str(Path(output_dir_str) / "progress.json"),
            }

    cmd = [
        str(python_exe),
        "-B",
        str(TRAIN_TSAD_ROOT / "scripts" / "train.py"),
        "--config",
        str(config_path),
    ]
    if payload.get("device"):
        cmd.extend(["--device", str(payload["device"])])
    if payload.get("max_epochs") is not None:
        cmd.extend(["--max-epochs", str(int(payload["max_epochs"]))])
    if bool(payload.get("inspect_data", True)):
        cmd.append("--inspect-data")
    if payload.get("inspect_max_samples") is not None:
        cmd.extend(["--inspect-max-samples", str(int(payload["inspect_max_samples"]))])
    if payload.get("inspect_output"):
        cmd.extend(["--inspect-output", str(payload["inspect_output"])])

    _run_subprocess(cmd, cwd=PROJECT_ROOT, job=job)

    return {
        "config_path": str(config_path),
        "output_dir": output_dir_str,
        "data_quality_report": str(Path(output_dir_str) / "data_quality_report.json") if output_dir_str else None,
        "history_path": str(Path(output_dir_str) / "history.json") if output_dir_str else None,
        "summary_path": str(Path(output_dir_str) / "summary.json") if output_dir_str else None,
        "progress_path": str(Path(output_dir_str) / "progress.json") if output_dir_str else None,
    }


def _run_job_wrapper(job: JobState, fn, payload: dict[str, Any]) -> None:
    with JOBS_LOCK:
        job.status = "running"
        job.started_at = time.time()
    try:
        result = fn(payload, job)
        with JOBS_LOCK:
            job.status = "completed"
            job.result = result
            job.finished_at = time.time()
    except Exception as exc:  # pragma: no cover
        _append_log(job, traceback.format_exc())
        with JOBS_LOCK:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = time.time()


def _resolve_train_output_dir(output_dir_raw: str | None, run_root_raw: str | None) -> Path:
    if output_dir_raw:
        output_dir = _resolve_path_like(output_dir_raw)
        if output_dir.exists() or output_dir.parent.exists():
            return output_dir
    if run_root_raw:
        run_info = _build_run_info(run_root_raw)
        return _resolve_path_like(run_info["train_output_dir"])
    raise FileNotFoundError("Provide output_dir or run_root to locate training artifacts.")


def _build_metric_series(history: list[dict[str, Any]]) -> dict[str, Any]:
    epochs = [int(entry.get("epoch", idx + 1)) for idx, entry in enumerate(history)]
    metric_names: list[str] = []
    for entry in history:
        for split in ("train", "val"):
            metrics = entry.get(split)
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    name = f"{split}.{key}"
                    if name not in metric_names:
                        metric_names.append(name)

    series: dict[str, list[float | None]] = {name: [] for name in metric_names}
    for entry in history:
        for name in metric_names:
            split, key = name.split(".", 1)
            metrics = entry.get(split, {})
            value = metrics.get(key) if isinstance(metrics, dict) else None
            series[name].append(float(value) if isinstance(value, (int, float)) else None)

    preferred_loss = next((name for name in metric_names if name.endswith("total_loss") and name.startswith("train.")), None)
    preferred_val = next((name for name in metric_names if name.startswith("val.")), None)
    chart_groups = _group_metric_names(metric_names)
    latest_entry = history[-1] if history else {}
    return {
        "epochs": epochs,
        "series": series,
        "metric_names": metric_names,
        "preferred_loss": preferred_loss,
        "preferred_quality": preferred_val,
        "chart_groups": chart_groups,
        "latest_epoch": int(latest_entry.get("epoch", 0)) if isinstance(latest_entry, dict) and latest_entry else None,
        "latest_train": latest_entry.get("train") if isinstance(latest_entry, dict) else None,
        "latest_val": latest_entry.get("val") if isinstance(latest_entry, dict) else None,
    }


def _group_metric_names(metric_names: list[str]) -> dict[str, list[str]]:
    groups = {
        "loss": [],
        "quality": [],
        "calibration": [],
    }
    for name in metric_names:
        _, key = name.split(".", 1)
        if key.startswith("num_") or key in {"tp", "fp", "fn", "anomaly_weight", "reconstruction_weight"}:
            continue
        if "loss" in key:
            groups["loss"].append(name)
            continue
        if key in {"precision", "recall", "f1", "pr_auc", "patch_accuracy"}:
            groups["quality"].append(name)
            continue
        if key in {"predicted_positive_rate", "target_positive_rate", "threshold", "reconstruction_mask_fraction", "reconstruction_used_mask"}:
            groups["calibration"].append(name)
    return {name: values for name, values in groups.items() if values}


def _build_training_kpis(
    history: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    progress: dict[str, Any] | None,
) -> dict[str, Any]:
    latest_entry = history[-1] if history else {}
    latest_train = latest_entry.get("train") if isinstance(latest_entry, dict) else None
    latest_val = latest_entry.get("val") if isinstance(latest_entry, dict) else None
    progress_train = progress.get("latest_train_metrics") if isinstance(progress, dict) else None
    progress_val = progress.get("latest_val_metrics") if isinstance(progress, dict) else None
    latest_train = progress_train if isinstance(progress_train, dict) and progress_train else latest_train
    latest_val = progress_val if isinstance(progress_val, dict) and progress_val else latest_val

    monitor_metric = None
    monitor_mode = None
    if isinstance(progress, dict):
        monitor_metric = progress.get("monitor_metric")
        monitor_mode = progress.get("monitor_mode")

    best_epoch = summary.get("best_epoch") if isinstance(summary, dict) else None
    best_metric = summary.get("best_metric") if isinstance(summary, dict) else None
    if best_epoch is None and isinstance(progress, dict):
        best_epoch = progress.get("best_epoch")
    if best_metric is None and isinstance(progress, dict):
        best_metric = progress.get("best_metric")
    latest_monitor_value = None
    if monitor_metric and isinstance(latest_val, dict) and monitor_metric in latest_val:
        latest_monitor_value = latest_val.get(monitor_metric)
    elif monitor_metric and isinstance(latest_train, dict):
        latest_monitor_value = latest_train.get(monitor_metric)

    return {
        "epoch_current": progress.get("epoch_current") if isinstance(progress, dict) else (int(latest_entry.get("epoch", 0)) if latest_entry else None),
        "epoch_total": progress.get("epoch_total") if isinstance(progress, dict) else None,
        "status": progress.get("status") if isinstance(progress, dict) else None,
        "stage": progress.get("stage") if isinstance(progress, dict) else None,
        "overall_progress_ratio": progress.get("overall_progress_ratio") if isinstance(progress, dict) else None,
        "learning_rate": progress.get("learning_rate") if isinstance(progress, dict) else None,
        "elapsed_seconds": progress.get("elapsed_seconds") if isinstance(progress, dict) else None,
        "eta_seconds": progress.get("eta_seconds") if isinstance(progress, dict) else None,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "latest_monitor_value": latest_monitor_value,
        "latest_train": latest_train,
        "latest_val": latest_val,
    }


class WorkbenchRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/bootstrap":
            self._write_json(HTTPStatus.OK, self._bootstrap_payload())
            return
        if path == "/api/job":
            self._handle_get_job(parsed)
            return
        if path == "/api/run":
            self._handle_run_info(parsed)
            return
        if path == "/api/samples":
            self._handle_samples(parsed)
            return
        if path == "/api/sample":
            self._handle_sample(parsed)
            return
        if path == "/api/window":
            self._handle_window(parsed)
            return
        if path == "/api/preview-item":
            self._handle_preview_item(parsed)
            return
        if path == "/api/train-metrics":
            self._handle_train_metrics(parsed)
            return
        if path in {"/", "/index.html"}:
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/randomize":
            self._handle_randomize()
            return
        if parsed.path == "/api/preview":
            self._handle_preview()
            return
        if parsed.path == "/api/preview-batch":
            self._handle_preview_batch()
            return
        if parsed.path == "/api/import-config":
            self._handle_import_config()
            return
        if parsed.path == "/api/generate":
            self._handle_generate()
            return
        if parsed.path == "/api/train":
            self._handle_train()
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {parsed.path}"})

    def _bootstrap_payload(self) -> dict[str, Any]:
        studio_payload = get_bootstrap_payload()
        templates = sorted(path.name for path in (TRAIN_TSAD_ROOT / "configs").glob("*.json"))
        studio_payload["workbench"] = {
            "workspace_root": str(PROJECT_ROOT),
            "python_executable": str(_resolve_python_executable()),
            "preview_cache_limit": PREVIEW_CACHE_LIMIT,
            "default_run_name": f"workbench_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "default_train_template": "timercd_small.json" if "timercd_small.json" in templates else (templates[0] if templates else None),
            "train_templates": templates,
            "generation_defaults": {
                "train_samples": 10000,
                "val_samples": 1500,
                "test_samples": 1500,
                "samples_per_shard": 128,
                "seed_base": 100,
                "train_device": "cuda",
                "train_max_epochs": 5,
            },
        }
        return studio_payload

    def _handle_get_job(self, parsed) -> None:
        query = parse_qs(parsed.query)
        job_id = str(query.get("job_id", [""])[0]).strip()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Job not found: {job_id}"})
                return
            payload = _job_to_dict(job)
        self._write_json(HTTPStatus.OK, payload)

    def _handle_run_info(self, parsed) -> None:
        try:
            query = parse_qs(parsed.query)
            path_like = str(query.get("path", [""])[0]).strip()
            if not path_like:
                raise ValueError("path is required")
            self._write_json(HTTPStatus.OK, _build_run_info(path_like))
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_randomize(self) -> None:
        try:
            body = self._read_json_body(optional=True)
            seed = None
            if isinstance(body, dict) and body.get("seed") is not None:
                seed = int(body["seed"])
            self._write_json(HTTPStatus.OK, {"config": randomize_config(seed=seed)})
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_preview(self) -> None:
        try:
            body = self._read_json_body(optional=False)
            if not isinstance(body, dict) or "config" not in body:
                raise ValueError("Expected JSON object with 'config'.")
            seed_offset = int(body.get("seed_offset", 0))
            preview = _preview_with_seed_offset(body["config"], seed_offset=seed_offset)
            preview_id = _cache_preview(preview) if body.get("cache", True) else None
            self._write_json(HTTPStatus.OK, {"preview": preview, "preview_id": preview_id})
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_preview_batch(self) -> None:
        try:
            body = self._read_json_body(optional=False)
            if not isinstance(body, dict) or "config" not in body:
                raise ValueError("Expected JSON object with 'config'.")
            count = max(1, min(int(body.get("count", 6)), PREVIEW_CACHE_LIMIT))
            seed_base = int(body.get("seed_base", body["config"].get("seed", 0) or 0))
            previews: list[dict[str, Any]] = []
            for seed_offset in range(count):
                preview = _preview_with_seed_offset(body["config"], seed_offset=seed_offset)
                preview_id = _cache_preview(preview)
                previews.append(_summarize_preview(preview_id, preview, seed=seed_base + seed_offset))
            self._write_json(HTTPStatus.OK, {"previews": previews})
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_preview_item(self, parsed) -> None:
        query = parse_qs(parsed.query)
        preview_id = str(query.get("preview_id", [""])[0]).strip()
        preview = _get_cached_preview(preview_id)
        if preview is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Preview not found: {preview_id}"})
            return
        self._write_json(HTTPStatus.OK, {"preview": preview, "preview_id": preview_id})

    def _handle_import_config(self) -> None:
        try:
            body = self._read_json_body(optional=False)
            if not isinstance(body, dict) or "text" not in body:
                raise ValueError("Expected JSON object with 'text'.")
            payload = import_config_text(str(body["text"]))
            self._write_json(HTTPStatus.OK, payload)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_generate(self) -> None:
        try:
            payload = self._read_json_body(optional=False)
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            job = _create_job("generate")
            thread = threading.Thread(target=_run_job_wrapper, args=(job, _run_generation_job, payload), daemon=True)
            thread.start()
            self._write_json(HTTPStatus.ACCEPTED, {"job_id": job.job_id})
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_train(self) -> None:
        try:
            payload = self._read_json_body(optional=False)
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            job = _create_job("train")
            thread = threading.Thread(target=_run_job_wrapper, args=(job, _run_train_job, payload), daemon=True)
            thread.start()
            self._write_json(HTTPStatus.ACCEPTED, {"job_id": job.job_id})
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_samples(self, parsed) -> None:
        try:
            query = parse_qs(parsed.query)
            packed_root = _resolve_packed_root(str(query.get("run_root", [""])[0]))
            split = str(query.get("split", ["train"])[0])
            limit = max(1, min(int(query.get("limit", [300])[0]), 2000))
            rows = _load_manifest_rows(packed_root, split)[:limit]
            samples = [
                {
                    "sample_id": str(row.get("sample_id")),
                    "length": int(row.get("length", 0)),
                    "num_series": int(row.get("num_series", 0)),
                    "is_anomalous_sample": int(row.get("is_anomalous_sample", 0)),
                    "num_events": int(row.get("num_events", 0)),
                    "anomaly_point_ratio": float(row.get("anomaly_point_ratio", 0.0)),
                }
                for row in rows
            ]
            self._write_json(
                HTTPStatus.OK,
                {
                    "packed_root": str(packed_root),
                    "split": split,
                    "num_samples": len(samples),
                    "samples": samples,
                },
            )
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_sample(self, parsed) -> None:
        try:
            query = parse_qs(parsed.query)
            packed_root = _resolve_packed_root(str(query.get("run_root", [""])[0]))
            split = str(query.get("split", ["train"])[0])
            sample_id = str(query.get("sample_id", [""])[0])
            feature_index = int(query.get("feature_index", [0])[0])
            slice_start = max(0, int(query.get("slice_start", [0])[0]))
            slice_end = int(query.get("slice_end", [0])[0])
            context_size = int(query.get("context_size", [512])[0])
            _requested_stride = int(query.get("stride", [256])[0])
            patch_size = int(query.get("patch_size", [16])[0])
            stride = context_size
            include_tail = False

            row = next(
                (entry for entry in _load_manifest_rows(packed_root, split) if str(entry.get("sample_id")) == sample_id),
                None,
            )
            if row is None:
                raise FileNotFoundError(f"sample_id not found in split `{split}`: {sample_id}")

            sample_payload = _load_sample_from_manifest_row(packed_root=packed_root, row=row)
            length = int(sample_payload["length"])
            dims = int(sample_payload["num_series"])
            feature_index = max(0, min(feature_index, dims - 1))

            if slice_end <= 0 or slice_end > length:
                slice_end = length
            if slice_start >= slice_end:
                slice_start = 0

            series = sample_payload["series"]
            normal_series = sample_payload["normal_series"]
            point_mask = sample_payload["point_mask"]
            point_mask_any = sample_payload["point_mask_any"]

            windows = _iter_context_bounds(
                length,
                context_size=context_size,
                stride=max(1, stride),
                include_tail=include_tail,
            )
            window_rows: list[dict[str, Any]] = []
            for start, end in windows[:500]:
                effective = end - start
                padded = 0
                feature_mask = point_mask[start:end, feature_index].astype(np.uint8, copy=False)
                patch_positive_ratio = 0.0
                if patch_size > 0 and context_size % patch_size == 0:
                    patch_labels = feature_mask.reshape(context_size // patch_size, patch_size).max(axis=1)
                    patch_positive_ratio = float(patch_labels.mean()) if patch_labels.size else 0.0
                window_rows.append(
                    {
                        "start": int(start),
                        "end": int(end),
                        "effective_length": int(effective),
                        "padded_steps": int(padded),
                        "patch_positive_ratio": patch_positive_ratio,
                    }
                )

            metadata = sample_payload.get("metadata", {})
            summary = metadata.get("summary", {}) if isinstance(metadata, dict) else {}
            response = {
                "sample_id": sample_id,
                "split": split,
                "length": length,
                "num_series": dims,
                "feature_index": feature_index,
                "slice_start": slice_start,
                "slice_end": slice_end,
                "series_feature": series[slice_start:slice_end, feature_index].astype(float).tolist(),
                "normal_series_feature": normal_series[slice_start:slice_end, feature_index].astype(float).tolist(),
                "point_mask_feature": point_mask[slice_start:slice_end, feature_index].astype(int).tolist(),
                "point_mask_any": point_mask_any[slice_start:slice_end].astype(int).tolist(),
                "anomaly_ratio_feature": float(point_mask[:, feature_index].mean()),
                "windowing": {
                    "context_size": context_size,
                    "stride": stride,
                    "patch_size": patch_size,
                    "include_tail": include_tail,
                    "num_windows": len(windows),
                    "windows": window_rows,
                },
                "manifest_row": row,
                "summary": summary,
                "metadata": metadata,
            }
            self._write_json(HTTPStatus.OK, response)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_window(self, parsed) -> None:
        try:
            query = parse_qs(parsed.query)
            packed_root = _resolve_packed_root(str(query.get("run_root", [""])[0]))
            split = str(query.get("split", ["train"])[0])
            sample_id = str(query.get("sample_id", [""])[0])
            feature_index = int(query.get("feature_index", [0])[0])
            window_index = max(0, int(query.get("window_index", [0])[0]))
            context_size = int(query.get("context_size", [512])[0])
            _requested_stride = max(1, int(query.get("stride", [256])[0]))
            patch_size = max(1, int(query.get("patch_size", [16])[0]))
            stride = context_size
            include_tail = False

            row = next(
                (entry for entry in _load_manifest_rows(packed_root, split) if str(entry.get("sample_id")) == sample_id),
                None,
            )
            if row is None:
                raise FileNotFoundError(f"sample_id not found in split `{split}`: {sample_id}")

            sample_payload = _load_sample_from_manifest_row(packed_root=packed_root, row=row)
            length = int(sample_payload["length"])
            dims = int(sample_payload["num_series"])
            feature_index = max(0, min(feature_index, dims - 1))
            windows = _iter_context_bounds(length, context_size=context_size, stride=stride, include_tail=include_tail)
            if not windows:
                raise ValueError(
                    "No full training windows are available for the selected sample. "
                    "Samples shorter than context_size and tail remainders are discarded."
                )
            if window_index >= len(windows):
                window_index = len(windows) - 1
            start, end = windows[window_index]
            effective_length = end - start

            series_window = np.asarray(sample_payload["series"][start:end, feature_index], dtype=np.float32)
            normal_window = np.asarray(sample_payload["normal_series"][start:end, feature_index], dtype=np.float32)
            mask_window = np.asarray(sample_payload["point_mask"][start:end, feature_index], dtype=np.uint8)
            any_window = np.asarray(sample_payload["point_mask_any"][start:end], dtype=np.uint8)
            padding_mask = np.zeros((context_size,), dtype=np.uint8)

            trimmed_length = context_size - (context_size % patch_size)
            patch_labels = []
            if trimmed_length > 0:
                patch_labels = (
                    mask_window[:trimmed_length]
                    .reshape(trimmed_length // patch_size, patch_size)
                    .max(axis=1)
                    .astype(int)
                    .tolist()
                )

            self._write_json(
                HTTPStatus.OK,
                {
                    "sample_id": sample_id,
                    "split": split,
                    "feature_index": feature_index,
                    "window_index": window_index,
                    "start": int(start),
                    "end": int(end),
                    "effective_length": int(effective_length),
                    "context_size": int(context_size),
                    "stride": int(stride),
                    "patch_size": int(patch_size),
                    "series_feature": series_window.astype(float).tolist(),
                    "normal_series_feature": normal_window.astype(float).tolist(),
                    "point_mask_feature": mask_window.astype(int).tolist(),
                    "point_mask_any": any_window.astype(int).tolist(),
                    "padding_mask": padding_mask.astype(int).tolist(),
                    "patch_labels": patch_labels,
                    "patch_alignment_warning": None if context_size % patch_size == 0 else "context_size is not divisible by patch_size",
                },
            )
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_train_metrics(self, parsed) -> None:
        try:
            query = parse_qs(parsed.query)
            output_dir_raw = str(query.get("output_dir", [""])[0]).strip() or None
            run_root_raw = str(query.get("run_root", [""])[0]).strip() or None
            output_dir = _resolve_train_output_dir(output_dir_raw, run_root_raw)
            history_path = output_dir / "history.json"
            summary_path = output_dir / "summary.json"
            quality_path = output_dir / "data_quality_report.json"
            progress_path = output_dir / "progress.json"

            history = []
            if history_path.exists():
                payload = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    history = payload

            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
            quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
            progress = _read_json_artifact(progress_path)
            history_view = _build_metric_series(history)
            self._write_json(
                HTTPStatus.OK,
                {
                    "output_dir": str(output_dir),
                    "history": history,
                    "history_view": history_view,
                    "summary": summary,
                    "progress": progress,
                    "kpis": _build_training_kpis(history, summary, progress),
                    "data_quality_report": quality,
                    "progress_path": str(progress_path),
                },
            )
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _read_json_body(self, optional: bool) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            if optional:
                return {}
            raise ValueError("Missing JSON request body")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), WorkbenchRequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="TSAD Workbench interactive frontend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8777, help="Port to bind")
    parser.add_argument("--open-browser", action="store_true", help="Open the browser after startup")
    args = parser.parse_args()

    server = make_server(args.host, int(args.port))
    url = f"http://{args.host}:{args.port}"
    print(f"TSAD Workbench listening on {url}")

    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
