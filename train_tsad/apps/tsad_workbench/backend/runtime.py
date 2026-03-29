from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .environment import (
    DEFAULT_FREE_SPACE_BUFFER_BYTES,
    DEFAULT_METADATA_BYTES_PER_SAMPLE,
    DEFAULT_NPZ_OVERHEAD_BYTES_PER_SAMPLE,
    DEFAULT_WORKBENCH_TRAIN_TEMPLATE,
    PROJECT_ROOT,
    SYNTH_ROOT,
    TRAIN_TSAD_ROOT,
    WORKBENCH_GENERATION_METADATA_FILENAME,
)


def json_clone(payload: Any) -> Any:
    """Return a JSON-roundtripped copy of one preview/config payload."""

    return json.loads(json.dumps(payload))


def resolve_python_executable() -> Path:
    """Return the preferred Python executable for subprocess work."""

    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        TRAIN_TSAD_ROOT / ".venv" / "Scripts" / "python.exe",
        SYNTH_ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path(sys.executable)


def resolve_path_like(path_like: str) -> Path:
    """Resolve a path-like string against the shared project root."""

    raw_path = Path(path_like).expanduser()
    return (PROJECT_ROOT / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()


def default_runs_root() -> Path:
    """Return the preferred default runs root for the current machine."""

    preferred_root = Path("D:/TSAD")
    if Path("D:/").exists():
        return (preferred_root / "runs").resolve()
    return (PROJECT_ROOT / "runs").resolve()


def generate_default_run_name() -> str:
    """Return the default run name shown in the workbench UI."""

    return f"workbench_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def nearest_existing_path(path: Path) -> Path:
    """Walk upward until an existing ancestor is found."""

    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            return candidate
        candidate = candidate.parent
    return candidate


def format_bytes(num_bytes: int) -> str:
    """Format a byte count with a compact binary-ish unit suffix."""

    value = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{int(value)} B"


def range_upper_bound(raw_value: Any, *, default: int) -> int:
    """Extract an upper-bound integer from either a scalar or `{min,max}` mapping."""

    if isinstance(raw_value, dict):
        if raw_value.get("max") is not None:
            return max(1, int(raw_value["max"]))
        if raw_value.get("min") is not None:
            return max(1, int(raw_value["min"]))
    if raw_value is None:
        return max(1, int(default))
    return max(1, int(raw_value))


def estimate_generation_required_bytes(
    synthetic_config: dict[str, Any],
    *,
    split_counts: dict[str, int],
    include_raw_stage: bool,
) -> int:
    """Estimate disk space required for one generation job."""

    sequence_length_max = range_upper_bound(synthetic_config.get("sequence_length"), default=1024)
    num_series_max = range_upper_bound(synthetic_config.get("num_series"), default=1)
    total_samples = sum(max(0, int(count)) for count in split_counts.values())
    float_bytes = np.dtype(np.float64).itemsize
    uint8_bytes = np.dtype(np.uint8).itemsize
    bytes_per_timestep = (2 * float_bytes * num_series_max) + (uint8_bytes * num_series_max) + uint8_bytes
    raw_array_bytes = total_samples * sequence_length_max * bytes_per_timestep
    metadata_bytes = total_samples * DEFAULT_METADATA_BYTES_PER_SAMPLE
    raw_npz_overhead = total_samples * DEFAULT_NPZ_OVERHEAD_BYTES_PER_SAMPLE
    packed_stage_bytes = int(raw_array_bytes * 0.45)
    if include_raw_stage:
        return (
            raw_array_bytes
            + metadata_bytes
            + raw_npz_overhead
            + packed_stage_bytes
            + DEFAULT_FREE_SPACE_BUFFER_BYTES
        )
    return packed_stage_bytes + metadata_bytes + DEFAULT_FREE_SPACE_BUFFER_BYTES


def validate_generation_target_space(
    *,
    run_root: Path,
    synthetic_config: dict[str, Any],
    split_counts: dict[str, int],
    include_raw_stage: bool,
) -> None:
    """Raise when the target disk has insufficient free space for generation."""

    target_base = nearest_existing_path(run_root.parent)
    free_bytes = shutil.disk_usage(target_base).free
    required_bytes = estimate_generation_required_bytes(
        synthetic_config,
        split_counts=split_counts,
        include_raw_stage=include_raw_stage,
    )
    if free_bytes >= required_bytes:
        return

    raise OSError(
        "Insufficient disk space for dataset generation. "
        f"Target root: {run_root}. "
        f"Available: {format_bytes(free_bytes)}. "
        f"Estimated required: {format_bytes(required_bytes)}. "
        "Use a run_root on a larger drive or reduce sample counts."
    )


def resolve_packed_root(path_like: str) -> Path:
    """Resolve a workbench path to a packed dataset root."""

    raw_path = resolve_path_like(path_like)
    if (raw_path / "dataset_meta.json").exists() and (raw_path / "manifest.train.jsonl").exists():
        return raw_path
    if (raw_path / "manifest.train.jsonl").exists():
        return raw_path
    if (raw_path / "data_packed" / "manifest.train.jsonl").exists():
        return raw_path / "data_packed"
    if (raw_path / "data_packed_windows" / "manifest.train.jsonl").exists():
        return raw_path / "data_packed_windows"
    raise FileNotFoundError(
        "Could not locate packed dataset root. Expected manifest.train.jsonl in "
        f"`{raw_path}` or `{raw_path / 'data_packed'}` or `{raw_path / 'data_packed_windows'}`."
    )


def resolve_run_root(path_like: str) -> Path:
    """Resolve a path to a workbench run root."""

    raw_path = resolve_path_like(path_like)
    if (raw_path / "data_packed" / "manifest.train.jsonl").exists():
        return raw_path
    if (raw_path / "data_packed_windows" / "manifest.train.jsonl").exists():
        return raw_path
    if (raw_path / "manifest.train.jsonl").exists() and raw_path.name in {"data_packed", "data_packed_windows"}:
        return raw_path.parent
    if (raw_path / "manifest.train.jsonl").exists():
        return raw_path
    raise FileNotFoundError(
        "Could not locate run root. Expected a run directory with `data_packed/` or "
        "`data_packed_windows/`, or a packed dataset root."
    )


def count_manifest_rows(manifest_path: Path) -> int:
    """Return the number of non-empty rows in one manifest file."""

    return sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_json_mapping(path: Path) -> dict[str, Any] | None:
    """Return one JSON object payload or `None` on parse/shape failure."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_training_config_payload(payload: Any) -> bool:
    """Return whether one JSON payload matches the expected train config shape."""

    if not isinstance(payload, dict):
        return False
    data_section = payload.get("data")
    train_section = payload.get("train")
    if not isinstance(data_section, dict) or not isinstance(train_section, dict):
        return False
    dataset_root = data_section.get("dataset_root")
    output_dir = train_section.get("output_dir")
    return (
        isinstance(dataset_root, str)
        and bool(dataset_root.strip())
        and isinstance(output_dir, str)
        and bool(output_dir.strip())
    )


def iter_training_config_candidates(config_dir: Path) -> list[Path]:
    """Return preferred training config candidate files inside one run."""

    candidates: list[Path] = []
    seen: set[Path] = set()
    for name in ("train_generated.json", "train_mini.json", "train.json", "train_config.json"):
        path = config_dir / name
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(path)

    if config_dir.exists():
        for path in sorted(config_dir.glob("*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(path)
    return candidates


def find_training_config_path(config_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """Return the first valid training config inside one config directory."""

    for path in iter_training_config_candidates(config_dir):
        payload = load_json_mapping(path)
        if is_training_config_payload(payload):
            return path, payload
    return None, None


def load_workbench_generation_metadata(config_dir: Path) -> dict[str, Any]:
    """Load generation metadata persisted by previous workbench runs."""

    payload = load_json_mapping(config_dir / WORKBENCH_GENERATION_METADATA_FILENAME)
    return payload if isinstance(payload, dict) else {}


def build_workbench_train_config(
    *,
    packed_root: Path,
    train_output_dir: Path,
    metadata: dict[str, Any] | None = None,
    allow_template_fallback: bool = True,
) -> dict[str, Any]:
    """Build the derived training config used by the workbench."""

    effective_metadata = dict(metadata or {})
    template_name = (
        str(effective_metadata.get("train_template") or DEFAULT_WORKBENCH_TRAIN_TEMPLATE).strip()
        or DEFAULT_WORKBENCH_TRAIN_TEMPLATE
    )
    template_path = TRAIN_TSAD_ROOT / "configs" / template_name
    if not template_path.exists() and allow_template_fallback and template_name != DEFAULT_WORKBENCH_TRAIN_TEMPLATE:
        template_name = DEFAULT_WORKBENCH_TRAIN_TEMPLATE
        template_path = TRAIN_TSAD_ROOT / "configs" / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Missing training template: {template_path}")

    train_config = load_json_mapping(template_path)
    if not is_training_config_payload(train_config):
        raise ValueError(f"Training template is not a valid training config: {template_path}")

    train_config["data"]["dataset_root"] = str(packed_root)
    train_config["train"]["output_dir"] = str(train_output_dir)
    if effective_metadata.get("train_device"):
        train_config["train"]["device"] = str(effective_metadata["train_device"])
    if effective_metadata.get("train_max_epochs") is not None:
        train_config["train"]["max_epochs"] = int(effective_metadata["train_max_epochs"])
    if effective_metadata.get("train_batch_size") is not None:
        train_config["data"]["batch_size"] = int(effective_metadata["train_batch_size"])
    if effective_metadata.get("eval_batch_size") is not None:
        train_config["data"]["eval_batch_size"] = int(effective_metadata["eval_batch_size"])
    return train_config


def ensure_train_config_for_run(run_root: Path) -> tuple[Path, dict[str, Any], bool]:
    """Return one valid train config for the run, rebuilding it when needed."""

    resolved_run_root = resolve_run_root(str(run_root))
    packed_root = resolve_packed_root(str(resolved_run_root))
    config_dir = resolved_run_root / "configs"
    train_output_dir = resolved_run_root / "train_out"
    config_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir.mkdir(parents=True, exist_ok=True)

    train_config_path, train_config = find_training_config_path(config_dir)
    if train_config_path is not None and train_config is not None:
        return train_config_path, train_config, False

    train_config = build_workbench_train_config(
        packed_root=packed_root,
        train_output_dir=train_output_dir,
        metadata=load_workbench_generation_metadata(config_dir),
        allow_template_fallback=True,
    )
    train_config_path = config_dir / "train_generated.json"
    train_config_path.write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_config_path, train_config, True


def infer_run_root_from_config_path(config_path: Path) -> Path | None:
    """Infer the enclosing run root for one config path."""

    config_dir = config_path.parent
    if config_dir.name != "configs":
        return None
    run_root = config_dir.parent
    if (run_root / "data_packed" / "manifest.train.jsonl").exists() or (
        run_root / "data_packed_windows" / "manifest.train.jsonl"
    ).exists():
        return run_root
    return None


def build_run_info(path_like: str) -> dict[str, Any]:
    """Build the workbench run summary payload for one run root."""

    run_root = resolve_run_root(path_like)
    packed_root = resolve_packed_root(str(run_root))
    available_splits: list[str] = []
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        manifest_path = packed_root / f"manifest.{split}.jsonl"
        if manifest_path.exists():
            available_splits.append(split)
            split_counts[split] = count_manifest_rows(manifest_path)

    config_dir = run_root / "configs"
    train_output_dir = run_root / "train_out"
    train_config_path: Path | None = None
    config_payload: dict[str, Any] | None = None
    try:
        train_config_path, config_payload, _ = ensure_train_config_for_run(run_root)
    except Exception:
        train_config_path, config_payload = find_training_config_path(config_dir)
    if config_payload is not None:
        output_dir = config_payload.get("train", {}).get("output_dir")
        if output_dir:
            train_output_dir = resolve_path_like(str(output_dir))

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
