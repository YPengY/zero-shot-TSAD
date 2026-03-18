from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
TRAIN_TSAD_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SRC_ROOT = TRAIN_TSAD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from train_tsad.config import ExperimentConfig  # noqa: E402
from train_tsad.data import (  # noqa: E402
    DataQualityInspector,
    ShardedSyntheticTsadDataset,
    SlidingContextWindowizer,
    SyntheticTsadDataset,
    WindowShardedTsadDataset,
)


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def _manifest_is_window_packed(manifest_path: Path | None) -> bool:
    if manifest_path is None or not manifest_path.exists():
        return False
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row_text = line.strip()
        if not row_text:
            continue
        row = json.loads(row_text)
        return (
            isinstance(row, dict)
            and "shard_npz_path" in row
            and ("window_index" in row or "sample_index" in row)
            and "context_start" in row
            and "context_end" in row
            and "valid_length" in row
        )
    return False


def _build_raw_dataset(config: ExperimentConfig, *, split: str):
    dataset_root = _resolve_path(config.data.dataset_root, base_dir=PROJECT_ROOT)
    manifest_path = config.data.manifest_path(split)
    manifest_path = _resolve_path(manifest_path, base_dir=PROJECT_ROOT)
    manifest_value: Path | None = manifest_path if manifest_path.exists() else None

    if config.data.use_sharded_dataset:
        if _manifest_is_window_packed(manifest_value):
            return WindowShardedTsadDataset(
                root_dir=dataset_root,
                split=split,
                manifest_path=manifest_value,
                max_cached_shards=config.data.max_cached_shards,
            )
        return ShardedSyntheticTsadDataset(
            root_dir=dataset_root,
            split=split,
            manifest_path=manifest_value,
            max_cached_shards=config.data.max_cached_shards,
            load_normal_series=config.data.load_normal_series,
            load_metadata=config.data.load_metadata,
        )

    return SyntheticTsadDataset(
        root_dir=dataset_root,
        split=split,
        manifest_path=manifest_value,
        load_normal_series=config.data.load_normal_series,
        load_metadata=config.data.load_metadata,
    )


def _default_inspection_splits(config: ExperimentConfig) -> list[str]:
    ordered = [config.data.split, config.data.validation_split, config.data.test_split]
    unique: list[str] = []
    for split in ordered:
        if split not in unique:
            unique.append(split)
    return unique


def parse_args() -> argparse.Namespace:
    default_config = TRAIN_TSAD_ROOT / "configs" / "timercd_pretrain_paper_aligned.json"
    parser = argparse.ArgumentParser(description="Inspect data quality before TSAD training.")
    parser.add_argument("--config", type=Path, default=default_config, help="Path to experiment config.")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Inspect only one split. Defaults to training split unless --all-splits is enabled.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Inspect train/val/test splits when available.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on inspected samples per split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_file(args.config)
    config.data.dataset_root = _resolve_path(config.data.dataset_root, base_dir=PROJECT_ROOT)

    if args.all_splits:
        target_splits = _default_inspection_splits(config)
    else:
        target_splits = [args.split or config.data.split]

    datasets_by_split: dict[str, object] = {}
    missing_splits: list[str] = []
    for split in target_splits:
        try:
            datasets_by_split[split] = _build_raw_dataset(config, split=split)
        except FileNotFoundError:
            missing_splits.append(split)

    if not datasets_by_split:
        resolved_dataset_root = _resolve_path(config.data.dataset_root, base_dir=PROJECT_ROOT)
        expected_manifests = {
            split: str(_resolve_path(config.data.manifest_path(split), base_dir=PROJECT_ROOT))
            for split in target_splits
        }
        error_payload = {
            "requested_splits": target_splits,
            "missing_splits": missing_splits,
            "resolved_dataset_root": str(resolved_dataset_root),
            "expected_manifests": expected_manifests,
        }
        raise FileNotFoundError(
            "Could not load any requested split for inspection. "
            f"Details={json.dumps(error_payload, ensure_ascii=False)}"
        )

    if any(bool(getattr(dataset, "is_prewindowed", False)) for dataset in datasets_by_split.values()):
        summary = {
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
            summary["splits"][split] = {
                "num_windows": num_windows,
                "num_features": num_features,
                "is_window_packed": True,
            }
        text = json.dumps(summary, indent=2, ensure_ascii=False)
        if args.output is not None:
            output_path = _resolve_path(args.output, base_dir=PROJECT_ROOT)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
        print(text)
        return

    windowizer = SlidingContextWindowizer(
        context_size=config.data.context_size,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        pad_short_sequences=config.data.pad_short_sequences,
        include_tail=config.data.include_tail,
    )
    inspector = DataQualityInspector(windowizer=windowizer, max_samples=args.max_samples)
    report = inspector.inspect_many(
        datasets_by_split,
        expected_training_split=config.data.split,
        missing_splits=missing_splits,
    )

    payload = report.to_dict()
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.output is not None:
        output_path = _resolve_path(args.output, base_dir=PROJECT_ROOT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    print(text)


if __name__ == "__main__":
    main()
