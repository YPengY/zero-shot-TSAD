from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .environment import PROJECT_ROOT, SYNTH_ROOT, TRAIN_TSAD_ROOT
from .job_store import JobState, JobStore
from .processes import run_parallel_split_generation, run_subprocess
from .runtime import (
    build_run_info,
    build_workbench_train_config,
    count_manifest_rows,
    default_runs_root,
    ensure_train_config_for_run,
    find_training_config_path,
    generate_default_run_name,
    infer_run_root_from_config_path,
    is_training_config_payload,
    load_json_mapping,
    resolve_packed_root,
    resolve_path_like,
    resolve_python_executable,
    resolve_run_root,
    validate_generation_target_space,
)
from .studio_bridge import pack_windows_from_packed_corpus, write_dataset_meta_for_existing_packed_corpus


def run_generation_job(payload: dict[str, Any], job: JobState, *, job_store: JobStore) -> dict[str, Any]:
    """Run the asynchronous dataset generation workflow."""

    python_exe = resolve_python_executable()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(payload.get("run_name", generate_default_run_name())).strip() or f"workbench_run_{timestamp}"
    run_root_input = str(payload.get("run_root", "")).strip()
    run_root = resolve_path_like(run_root_input) if run_root_input else (default_runs_root() / run_name).resolve()

    overwrite = bool(payload.get("overwrite_run", False))
    confirm_overwrite = bool(payload.get("confirm_overwrite", False))
    if run_root.exists() and overwrite:
        if not confirm_overwrite:
            raise FileExistsError(f"Run root already exists and overwrite was not confirmed: {run_root}")
        shutil.rmtree(run_root)
    elif run_root.exists() and not overwrite:
        raise FileExistsError(f"Run root already exists: {run_root}")

    direct_pack = bool(payload.get("direct_pack", True))
    window_pack = bool(payload.get("window_pack", False))
    direct_window_pack = bool(payload.get("direct_window_pack", True)) and direct_pack and window_pack
    raw_root = run_root / "data_raw"
    packed_root = run_root / "data_packed"
    window_packed_root = run_root / "data_packed_windows"
    config_root = run_root / "configs"
    train_out = run_root / "train_out"
    eval_out = run_root / "eval"

    window_context_size = int(payload.get("window_context_size", 1024))
    window_patch_size = int(payload.get("window_patch_size", 16))
    window_stride = payload.get("window_stride")
    window_windows_per_shard = int(payload.get("window_windows_per_shard", 4096))
    window_include_tail = bool(payload.get("window_include_tail", True))
    window_pad_short_sequences = bool(payload.get("window_pad_short_sequences", True))
    window_debug_sidecar = bool(payload.get("window_debug_sidecar", True))
    window_train_min_patch_positive_ratio = (
        None
        if payload.get("window_train_min_patch_positive_ratio") is None
        else float(payload["window_train_min_patch_positive_ratio"])
    )
    window_train_min_anomaly_point_ratio = (
        None
        if payload.get("window_train_min_anomaly_point_ratio") is None
        else float(payload["window_train_min_anomaly_point_ratio"])
    )

    if direct_window_pack:
        window_packed_root.mkdir(parents=True, exist_ok=True)
        packed_root.mkdir(parents=True, exist_ok=True)
    elif direct_pack:
        packed_root.mkdir(parents=True, exist_ok=True)
    else:
        for split in ("train", "val", "test"):
            (raw_root / split).mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    train_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    synthetic_config = payload.get("config")
    if not isinstance(synthetic_config, dict):
        raise ValueError("Generation requires the full edited synthetic config.")

    train_template_name = str(payload.get("train_template", "timercd_small.json")).strip() or "timercd_small.json"
    generation_metadata = {
        "train_template": train_template_name,
        "train_device": str(payload["train_device"]) if payload.get("train_device") else None,
        "train_max_epochs": int(payload["train_max_epochs"]) if payload.get("train_max_epochs") is not None else None,
        "train_batch_size": int(payload["train_batch_size"]) if payload.get("train_batch_size") is not None else None,
        "eval_batch_size": int(payload["eval_batch_size"]) if payload.get("eval_batch_size") is not None else None,
    }
    synthetic_config_path = config_root / "synthetic_runtime_config.json"
    synthetic_config_path.write_text(json.dumps(synthetic_config, ensure_ascii=False, indent=2), encoding="utf-8")
    generation_metadata_path = config_root / "workbench_generation_metadata.json"
    generation_metadata_path.write_text(json.dumps(generation_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    split_counts = {
        "train": int(payload.get("train_samples", 10000)),
        "val": int(payload.get("val_samples", 1500)),
        "test": int(payload.get("test_samples", 1500)),
    }
    validate_generation_target_space(
        run_root=run_root,
        synthetic_config=synthetic_config,
        split_counts=split_counts,
        include_raw_stage=not direct_pack,
    )

    seed_base = int(payload.get("seed_base", synthetic_config.get("seed", 0) or 0))
    dataset_name = str(payload.get("dataset_name", "workbench_tsad"))
    dataset_version = str(payload.get("dataset_version", "v1"))
    samples_per_shard = int(payload.get("samples_per_shard", 128))

    split_commands: list[tuple[str, list[str]]] = []
    for split_index, split in enumerate(("train", "val", "test"), start=1):
        output_path = window_packed_root if direct_window_pack else (packed_root if direct_pack else (raw_root / split))
        cmd = [
            str(python_exe),
            "-B",
            str(SYNTH_ROOT / "scripts" / "generate_dataset.py"),
            "--config",
            str(synthetic_config_path),
            "--output",
            str(output_path),
            "--num-samples",
            str(split_counts[split]),
            "--seed",
            str(seed_base + split_index),
        ]
        if direct_window_pack:
            split_min_patch_positive_ratio = window_train_min_patch_positive_ratio if split == "train" else None
            split_min_anomaly_point_ratio = window_train_min_anomaly_point_ratio if split == "train" else None
            cmd.extend(
                [
                    "--direct-window-pack",
                    "--split",
                    split,
                    "--window-context-size",
                    str(window_context_size),
                    "--window-patch-size",
                    str(window_patch_size),
                    "--window-windows-per-shard",
                    str(window_windows_per_shard),
                ]
            )
            if split_min_patch_positive_ratio is not None:
                cmd.extend(["--window-min-patch-positive-ratio", str(split_min_patch_positive_ratio)])
            if split_min_anomaly_point_ratio is not None:
                cmd.extend(["--window-min-anomaly-point-ratio", str(split_min_anomaly_point_ratio)])
            if window_stride is not None:
                cmd.extend(["--window-stride", str(int(window_stride))])
            if not window_include_tail:
                cmd.append("--window-no-include-tail")
            if not window_pad_short_sequences:
                cmd.append("--window-no-pad-short-sequences")
            if not window_debug_sidecar:
                cmd.append("--window-no-debug-sidecar")
        elif direct_pack:
            cmd.extend(["--direct-pack", "--split", split, "--samples-per-shard", str(samples_per_shard)])
        split_commands.append((split, cmd))

    run_parallel_split_generation(split_commands, cwd=PROJECT_ROOT, job=job, job_store=job_store)
    if direct_window_pack:
        job_store.append_log(job, "All split generation processes finished in direct-window-pack mode.")
        resolved_window_stride = window_context_size if window_stride is None else int(window_stride)
        meta_report = write_dataset_meta_for_existing_packed_corpus(
            window_packed_root,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            samples_per_shard=window_windows_per_shard,
            storage_format_type="npz_window_shards_minimal_v1",
            storage_format_extras={
                "context_size": int(window_context_size),
                "patch_size": int(window_patch_size),
                "stride": int(resolved_window_stride),
                "include_tail": bool(window_include_tail),
                "pad_short_sequences": bool(window_pad_short_sequences),
                "windows_per_shard": int(window_windows_per_shard),
                "debug_sidecar": bool(window_debug_sidecar),
                "required_fields": ["series_windows", "point_mask_windows", "patch_labels_windows", "valid_lengths"],
            },
        )
        job_store.append_log(job, f"Wrote dataset_meta.json for window-packed manifests (splits={list(meta_report.splits.keys())}).")
        training_dataset_root = window_packed_root
    elif direct_pack:
        job_store.append_log(job, "All split generation processes finished in direct-pack mode.")
        meta_report = write_dataset_meta_for_existing_packed_corpus(
            packed_root,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            samples_per_shard=samples_per_shard,
        )
        job_store.append_log(job, f"Wrote dataset_meta.json from packed manifests (splits={list(meta_report.splits.keys())}).")
        training_dataset_root = packed_root
    else:
        job_store.append_log(job, "All split generation processes finished. Starting shard packing.")
        pack_cmd = [
            str(python_exe),
            "-B",
            str(SYNTH_ROOT / "scripts" / "pack_dataset.py"),
            "--input",
            str(raw_root),
            "--output",
            str(packed_root),
            "--samples-per-shard",
            str(samples_per_shard),
            "--overwrite",
            "--dataset-name",
            dataset_name,
            "--dataset-version",
            dataset_version,
        ]
        run_subprocess(pack_cmd, cwd=PROJECT_ROOT, job=job, job_store=job_store, log_prefix="pack")
        training_dataset_root = packed_root

    if window_pack and not direct_window_pack:
        job_store.append_log(job, "Starting window-level packing pass for training format.")
        window_report = pack_windows_from_packed_corpus(
            input_root=packed_root,
            output_root=window_packed_root,
            context_size=window_context_size,
            patch_size=window_patch_size,
            stride=(None if window_stride is None else int(window_stride)),
            include_tail=window_include_tail,
            pad_short_sequences=window_pad_short_sequences,
            windows_per_shard=window_windows_per_shard,
            overwrite=True,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            write_debug_sidecar=window_debug_sidecar,
            min_patch_positive_ratio=window_train_min_patch_positive_ratio,
            min_anomaly_point_ratio=window_train_min_anomaly_point_ratio,
        )
        total_windows = sum(report.num_samples for report in window_report.splits.values())
        job_store.append_log(job, f"Window-level packing finished: splits={list(window_report.splits.keys())}, total_windows={total_windows}.")
        training_dataset_root = window_packed_root

    train_config = build_workbench_train_config(
        packed_root=training_dataset_root,
        train_output_dir=train_out,
        metadata=generation_metadata,
        allow_template_fallback=False,
    )
    train_config_path = config_root / "train_generated.json"
    train_config_path.write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")

    result = build_run_info(str(run_root))
    generation_mode = (
        "direct_window_pack"
        if direct_window_pack
        else ("direct_pack+window_pack" if direct_pack and window_pack else ("direct_pack" if direct_pack else ("raw_then_pack+window_pack" if window_pack else "raw_then_pack")))
    )
    result.update(
        {
            "raw_root": str(raw_root) if raw_root.exists() else None,
            "generation_mode": generation_mode,
            "dataset_meta_path": str(training_dataset_root / "dataset_meta.json"),
            "training_dataset_root": str(training_dataset_root),
            "window_packed_root": str(window_packed_root) if window_pack else None,
            "synthetic_config_path": str(synthetic_config_path),
            "train_config_path": str(train_config_path),
            "eval_output_dir": str(eval_out),
        }
    )
    return result


def run_train_job(payload: dict[str, Any], job: JobState, *, job_store: JobStore) -> dict[str, Any]:
    """Run the asynchronous training workflow."""

    python_exe = resolve_python_executable()
    config_path_raw = str(payload.get("config_path", "")).strip()
    run_root_raw = str(payload.get("run_root", "")).strip()

    config_path: Path | None = None
    config_payload: dict[str, Any] | None = None
    if config_path_raw:
        config_path = resolve_path_like(config_path_raw)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing training config: {config_path}")
        config_payload = load_json_mapping(config_path)
        if not is_training_config_payload(config_payload):
            inferred_run_root = infer_run_root_from_config_path(config_path)
            if inferred_run_root is None and run_root_raw:
                inferred_run_root = resolve_run_root(run_root_raw)
            if inferred_run_root is None:
                raise ValueError(
                    "Provided config_path is not a valid training config. Expected a JSON file with `data` and `train` sections."
                )
            resolved_config_path, resolved_config_payload, repaired = ensure_train_config_for_run(inferred_run_root)
            if repaired or resolved_config_path != config_path:
                job_store.append_log(job, f"Resolved training config automatically: {config_path} -> {resolved_config_path}")
            config_path = resolved_config_path
            config_payload = resolved_config_payload
    elif run_root_raw:
        config_path, config_payload, repaired = ensure_train_config_for_run(resolve_run_root(run_root_raw))
        if repaired:
            job_store.append_log(job, f"Rebuilt missing training config at {config_path}.")
    else:
        raise ValueError("config_path or run_root is required")

    if config_path is None or not is_training_config_payload(config_payload):
        raise ValueError(f"Training config is invalid: {config_path}")

    output_dir = config_payload.get("train", {}).get("output_dir")
    output_dir_str = str(resolve_path_like(str(output_dir))) if output_dir else None
    if output_dir_str:
        job_store.set_artifacts(
            job,
            {
                "config_path": str(config_path),
                "output_dir": output_dir_str,
                "data_quality_report": str(Path(output_dir_str) / "data_quality_report.json"),
                "history_path": str(Path(output_dir_str) / "history.json"),
                "summary_path": str(Path(output_dir_str) / "summary.json"),
                "progress_path": str(Path(output_dir_str) / "progress.json"),
            },
        )

    cmd = [str(python_exe), "-B", str(TRAIN_TSAD_ROOT / "scripts" / "train.py"), "--config", str(config_path)]
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

    run_subprocess(cmd, cwd=PROJECT_ROOT, job=job, job_store=job_store)
    return {
        "config_path": str(config_path),
        "output_dir": output_dir_str,
        "data_quality_report": str(Path(output_dir_str) / "data_quality_report.json") if output_dir_str else None,
        "history_path": str(Path(output_dir_str) / "history.json") if output_dir_str else None,
        "summary_path": str(Path(output_dir_str) / "summary.json") if output_dir_str else None,
        "progress_path": str(Path(output_dir_str) / "progress.json") if output_dir_str else None,
    }
