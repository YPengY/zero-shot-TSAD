from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .job_store import read_json_artifact
from .runtime import build_run_info, resolve_path_like


def resolve_train_output_dir(output_dir_raw: str | None, run_root_raw: str | None) -> Path:
    """Resolve the training artifact directory from either explicit output_dir or run_root."""

    if output_dir_raw:
        output_dir = resolve_path_like(output_dir_raw)
        if output_dir.exists() or output_dir.parent.exists():
            return output_dir
    if run_root_raw:
        run_info = build_run_info(run_root_raw)
        return resolve_path_like(run_info["train_output_dir"])
    raise FileNotFoundError("Provide output_dir or run_root to locate training artifacts.")


def group_metric_names(metric_names: list[str]) -> dict[str, list[str]]:
    """Group metric series into chart-friendly categories."""

    groups = {
        "loss": [],
        "quality": [],
        "calibration": [],
    }
    for name in metric_names:
        _, key = name.split(".", 1)
        if key.startswith("num_") or key in {
            "tp",
            "fp",
            "fn",
            "anomaly_weight",
            "point_anomaly_weight",
            "reconstruction_weight",
        }:
            continue
        if "loss" in key:
            groups["loss"].append(name)
            continue
        if key in {"precision", "recall", "f1", "pr_auc", "patch_accuracy", "point_accuracy"}:
            groups["quality"].append(name)
            continue
        if key in {
            "predicted_positive_rate",
            "target_positive_rate",
            "point_predicted_positive_rate",
            "point_target_positive_rate",
            "threshold",
            "reconstruction_mask_fraction",
            "reconstruction_used_mask",
        }:
            groups["calibration"].append(name)
    return {name: values for name, values in groups.items() if values}


def build_metric_series(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Build chart-ready metric series from raw history rows."""

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

    preferred_loss = next(
        (name for name in metric_names if name.endswith("total_loss") and name.startswith("train.")),
        None,
    )
    preferred_val = next((name for name in metric_names if name.startswith("val.")), None)
    latest_entry = history[-1] if history else {}
    return {
        "epochs": epochs,
        "series": series,
        "metric_names": metric_names,
        "preferred_loss": preferred_loss,
        "preferred_quality": preferred_val,
        "chart_groups": group_metric_names(metric_names),
        "latest_epoch": int(latest_entry.get("epoch", 0)) if isinstance(latest_entry, dict) and latest_entry else None,
        "latest_train": latest_entry.get("train") if isinstance(latest_entry, dict) else None,
        "latest_val": latest_entry.get("val") if isinstance(latest_entry, dict) else None,
    }


def build_training_kpis(
    history: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    progress: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the summary KPI payload shown by the workbench."""

    latest_entry = history[-1] if history else {}
    latest_train = latest_entry.get("train") if isinstance(latest_entry, dict) else None
    latest_val = latest_entry.get("val") if isinstance(latest_entry, dict) else None
    progress_train = progress.get("latest_train_metrics") if isinstance(progress, dict) else None
    progress_val = progress.get("latest_val_metrics") if isinstance(progress, dict) else None
    latest_train = progress_train if isinstance(progress_train, dict) and progress_train else latest_train
    latest_val = progress_val if isinstance(progress_val, dict) and progress_val else latest_val

    monitor_metric = progress.get("monitor_metric") if isinstance(progress, dict) else None
    monitor_mode = progress.get("monitor_mode") if isinstance(progress, dict) else None
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
        "epoch_current": (
            progress.get("epoch_current")
            if isinstance(progress, dict)
            else (int(latest_entry.get("epoch", 0)) if latest_entry else None)
        ),
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


def build_train_metrics_payload(*, output_dir_raw: str | None, run_root_raw: str | None) -> dict[str, Any]:
    """Build the payload returned by `/api/train-metrics`."""

    output_dir = resolve_train_output_dir(output_dir_raw, run_root_raw)
    history_path = output_dir / "history.json"
    summary_path = output_dir / "summary.json"
    quality_path = output_dir / "data_quality_report.json"
    progress_path = output_dir / "progress.json"

    history: list[dict[str, Any]] = []
    if history_path.exists():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            history = payload

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
    progress = read_json_artifact(progress_path)
    return {
        "output_dir": str(output_dir),
        "history": history,
        "history_view": build_metric_series(history),
        "summary": summary,
        "progress": progress,
        "kpis": build_training_kpis(history, summary, progress),
        "data_quality_report": quality,
        "progress_path": str(progress_path),
    }
