from __future__ import annotations

from typing import Any

from .environment import PREVIEW_CACHE_LIMIT, PROJECT_ROOT, TRAIN_TSAD_ROOT
from .job_store import PreviewStore
from .runtime import default_runs_root, generate_default_run_name, json_clone, resolve_python_executable
from .studio_bridge import get_bootstrap_payload, preview_sample


def preview_with_seed_offset(raw_config: dict[str, Any], *, seed_offset: int = 0) -> dict[str, Any]:
    """Build one in-memory preview after offsetting the configured seed."""

    preview_config = json_clone(raw_config)
    base_seed = int(preview_config.get("seed", 0) or 0)
    preview_config["seed"] = base_seed + int(seed_offset)
    return preview_sample(preview_config)


def summarize_preview(preview_id: str, preview: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Build the compact preview summary shown in the gallery."""

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


def build_preview_payload(
    *,
    config: dict[str, Any],
    seed_offset: int = 0,
    cache: bool,
    preview_store: PreviewStore,
) -> dict[str, Any]:
    """Build the `/api/preview` response payload."""

    preview = preview_with_seed_offset(config, seed_offset=seed_offset)
    preview_id = preview_store.put(preview) if cache else None
    return {"preview": preview, "preview_id": preview_id}


def build_preview_batch_payload(
    *,
    config: dict[str, Any],
    count: int,
    seed_base: int,
    preview_store: PreviewStore,
) -> dict[str, Any]:
    """Build the `/api/preview-batch` response payload."""

    previews: list[dict[str, Any]] = []
    for seed_offset in range(count):
        preview = preview_with_seed_offset(config, seed_offset=seed_offset)
        preview_id = preview_store.put(preview)
        previews.append(summarize_preview(preview_id, preview, seed=seed_base + seed_offset))
    return {"previews": previews}


def build_bootstrap_payload() -> dict[str, Any]:
    """Build the combined studio + workbench bootstrap payload."""

    studio_payload = get_bootstrap_payload()
    templates = sorted(path.name for path in (TRAIN_TSAD_ROOT / "configs").glob("*.json"))
    studio_payload["workbench"] = {
        "workspace_root": str(PROJECT_ROOT),
        "python_executable": str(resolve_python_executable()),
        "preview_cache_limit": PREVIEW_CACHE_LIMIT,
        "default_runs_root": str(default_runs_root()),
        "default_run_name": generate_default_run_name(),
        "default_train_template": (
            "timercd_pretrain_paper_aligned.json"
            if "timercd_pretrain_paper_aligned.json" in templates
            else ("timercd_small.json" if "timercd_small.json" in templates else (templates[0] if templates else None))
        ),
        "train_templates": templates,
        "generation_defaults": {
            "train_samples": 10000,
            "val_samples": 1500,
            "test_samples": 1500,
            "direct_pack": True,
            "window_pack": True,
            "direct_window_pack": True,
            "window_context_size": 1024,
            "window_patch_size": 16,
            "window_stride": 1024,
            "window_windows_per_shard": 4096,
            "window_include_tail": True,
            "window_pad_short_sequences": True,
            "window_debug_sidecar": False,
            "window_train_min_patch_positive_ratio": 0.005208333333333333,
            "window_train_min_anomaly_point_ratio": None,
            "samples_per_shard": 128,
            "seed_base": 100,
            "train_device": "cuda",
            "train_max_epochs": 5,
        },
    }
    return studio_payload
