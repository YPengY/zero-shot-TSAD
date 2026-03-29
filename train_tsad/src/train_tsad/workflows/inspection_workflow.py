from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..data import DataQualityInspector, build_raw_dataset, default_inspection_splits
from ..data.windowizer import SlidingContextWindowizer
from ..utils import resolve_path, write_json_file


@dataclass(frozen=True, slots=True)
class InspectionOptions:
    """CLI options for data-quality inspection."""

    split: str | None = None
    inspect_all_splits: bool = False
    max_samples: int | None = None
    output: Path | None = None


@dataclass(frozen=True, slots=True)
class InspectionWorkflowResult:
    """Payload produced by data-quality inspection."""

    config: ExperimentConfig
    payload: dict[str, Any]
    output_path: Path | None = None


def prepare_inspection_config(config_path: Path, *, base_dir: Path) -> ExperimentConfig:
    """Load the config and resolve the dataset root."""

    config = ExperimentConfig.from_file(config_path).clone()
    config.data.dataset_root = resolve_path(config.data.dataset_root, base_dir=base_dir)
    return config


def run_inspection(
    config_path: Path,
    *,
    base_dir: Path,
    options: InspectionOptions,
    logger: logging.Logger,
) -> InspectionWorkflowResult:
    """Run split-level data-quality inspection and return the JSON payload."""

    config = prepare_inspection_config(config_path, base_dir=base_dir)
    if options.inspect_all_splits:
        target_splits = default_inspection_splits(config.data)
    else:
        target_splits = [options.split or config.data.split]

    datasets_by_split: dict[str, object] = {}
    missing_splits: list[str] = []
    for split in target_splits:
        try:
            datasets_by_split[split] = build_raw_dataset(
                config.data,
                split=split,
                base_dir=base_dir,
            )
        except FileNotFoundError:
            missing_splits.append(split)

    if not datasets_by_split:
        expected_manifests = {
            split: str(resolve_path(config.data.manifest_path(split), base_dir=base_dir))
            for split in target_splits
        }
        raise FileNotFoundError(
            "Could not load any requested split for inspection. "
            f"dataset_root={config.data.dataset_root}, expected_manifests={expected_manifests}, "
            f"missing_splits={missing_splits}"
        )

    if any(bool(getattr(dataset, "is_prewindowed", False)) for dataset in datasets_by_split.values()):
        payload: dict[str, Any] = {
            "summary": {
                "expected_training_split": config.data.split,
                "inspected_splits": list(datasets_by_split.keys()),
                "missing_splits": missing_splits,
                "mode": "window_packed_overview",
                "recommended_to_train": True,
            },
            "splits": {},
        }
        for split, dataset in datasets_by_split.items():
            num_windows = int(len(dataset))
            num_features = int(dataset.sample_num_features(0)) if num_windows > 0 else 0
            payload["splits"][split] = {
                "num_windows": num_windows,
                "num_features": num_features,
                "is_window_packed": True,
            }
    else:
        inspector = DataQualityInspector(
            windowizer=SlidingContextWindowizer(
                context_size=config.data.context_size,
                patch_size=config.data.patch_size,
                stride=config.data.stride,
                pad_short_sequences=config.data.pad_short_sequences,
                include_tail=config.data.include_tail,
            ),
            max_samples=options.max_samples,
        )
        payload = inspector.inspect_many(
            datasets_by_split,
            expected_training_split=config.data.split,
            missing_splits=missing_splits,
        ).to_dict()

    output_path = None
    if options.output is not None:
        output_path = resolve_path(options.output, base_dir=base_dir)
        write_json_file(output_path, payload)
        logger.info("Inspection report written to %s.", output_path)

    return InspectionWorkflowResult(
        config=config,
        payload=payload,
        output_path=output_path,
    )
