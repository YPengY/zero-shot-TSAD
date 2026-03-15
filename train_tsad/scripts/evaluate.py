from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


SCRIPT_PATH = Path(__file__).resolve()
TRAIN_TSAD_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SRC_ROOT = TRAIN_TSAD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from train_tsad.config import ExperimentConfig  # noqa: E402
from train_tsad.data import (  # noqa: E402
    ContextWindowCollator,
    ContextWindowDataset,
    ShardedSyntheticTsadDataset,
    SlidingContextWindowizer,
    SyntheticTsadDataset,
)
from train_tsad.evaluation import TimeRCDEvaluator  # noqa: E402
from train_tsad.evaluation import PatchFeatureEvaluator  # noqa: E402
from train_tsad.models import TimeRCDModel  # noqa: E402


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    """Resolve relative paths against the project root."""

    return path if path.is_absolute() else (base_dir / path).resolve()


def _build_raw_dataset(config: ExperimentConfig, *, split: str):
    """Create the raw dataset for the target evaluation split."""

    dataset_root = _resolve_path(config.data.dataset_root, base_dir=PROJECT_ROOT)
    manifest_path = config.data.manifest_path(split)
    manifest_path = _resolve_path(manifest_path, base_dir=PROJECT_ROOT)
    manifest_value: Path | None = manifest_path if manifest_path.exists() else None

    dataset_cls = ShardedSyntheticTsadDataset if config.data.use_sharded_dataset else SyntheticTsadDataset
    return dataset_cls(
        root_dir=dataset_root,
        split=split,
        manifest_path=manifest_value,
    )


def _build_eval_loader(raw_dataset, *, config: ExperimentConfig) -> DataLoader:
    """Build a deterministic evaluation DataLoader.

    Notes:
    - Always disables shuffle/drop_last to keep metric computation stable.
    - Disables reconstruction-target construction in collate for eval speed.
    """

    windowizer = SlidingContextWindowizer(
        context_size=config.data.context_size,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        pad_short_sequences=config.data.pad_short_sequences,
        include_tail=config.data.include_tail,
    )
    window_dataset = ContextWindowDataset(raw_dataset, windowizer)
    return DataLoader(
        window_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
        collate_fn=ContextWindowCollator(include_reconstruction_targets=False),
    )


def _infer_fixed_num_features(raw_dataset) -> int:
    """Infer the unique feature-channel count used to build the model."""

    feature_counts = {int(raw_dataset[index].series.shape[1]) for index in range(len(raw_dataset))}
    if not feature_counts:
        raise FileNotFoundError("Could not infer the number of feature channels from the evaluation dataset.")
    if len(feature_counts) != 1:
        raise ValueError(
            "The current evaluation pipeline requires a fixed number of feature channels per run. "
            f"Found multiple values: {sorted(feature_counts)}."
        )
    return next(iter(feature_counts))


def _resolve_device(requested: str) -> torch.device:
    """Resolve runtime device with a safe CUDA fallback."""

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _auto_split(config: ExperimentConfig) -> str:
    """Pick the first available split in priority: test -> val -> train."""

    for split in (config.data.test_split, config.data.validation_split, config.data.split):
        try:
            _build_raw_dataset(config, split=split)
            return split
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Could not find any available split for evaluation.")


def parse_args() -> argparse.Namespace:
    """Parse evaluation entrypoint CLI arguments."""

    default_config = TRAIN_TSAD_ROOT / "configs" / "timercd_small.json"
    parser = argparse.ArgumentParser(description="Evaluate a trained TimeRCD checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to a JSON or YAML experiment config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to `<output_dir>/best.pt`.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate. Defaults to test, then val, then train.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the evaluation summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run end-to-end evaluation for a trained checkpoint.

    Workflow:
    1. Load config and select split/checkpoint.
    2. Build dataset/loader and instantiate model with matching feature size.
    3. Stream batches through evaluator and report detection metrics.
    """

    args = parse_args()
    config = ExperimentConfig.from_file(args.config)
    if args.device is not None:
        config.train.device = args.device

    config.data.dataset_root = _resolve_path(config.data.dataset_root, base_dir=PROJECT_ROOT)
    config.train.output_dir = _resolve_path(config.train.output_dir, base_dir=PROJECT_ROOT)

    # If split is unspecified, auto-discover a usable split by preference order.
    split = args.split or _auto_split(config)
    checkpoint_path = args.checkpoint or (config.train.output_dir / "best.pt")
    checkpoint_path = _resolve_path(checkpoint_path, base_dir=PROJECT_ROOT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_dataset = _build_raw_dataset(config, split=split)
    num_features = _infer_fixed_num_features(raw_dataset)
    loader = _build_eval_loader(raw_dataset, config=config)

    device = _resolve_device(config.train.device)
    model = TimeRCDModel(
        patch_size=config.model.patch_size,
        d_model=config.model.d_model,
        d_proj=config.model.d_proj,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        max_patches=config.data.num_patches,
        max_features=max(num_features, 1),
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        attention_dropout=config.model.attention_dropout,
        activation=config.model.activation,
        use_learned_positional_encoding=config.model.use_learned_positional_encoding,
        use_reconstruction_head=config.loss.reconstruction_loss_weight > 0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if config.eval.task == "patch_feature":
        evaluator = PatchFeatureEvaluator(
            patch_size=config.data.patch_size,
            patch_feature_score_aggregation=config.eval.patch_feature_score_aggregation,
            threshold=config.eval.threshold,
            threshold_search=config.eval.threshold_search,
            threshold_search_metric=config.eval.threshold_search_metric,
            report_per_feature=config.eval.report_per_feature,
            report_per_sample=config.eval.report_per_sample,
        )
    else:
        evaluator = TimeRCDEvaluator(
            patch_size=config.data.patch_size,
            score_reduction=config.eval.score_reduction,
            point_score_aggregation=config.eval.point_score_aggregation,
            threshold=config.eval.threshold,
            threshold_search=config.eval.threshold_search,
            threshold_search_metric=config.eval.threshold_search_metric,
        )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            evaluator.update(batch, output)

    metrics = evaluator.compute()
    metrics.update(
        {
            "split": split,
            "checkpoint": str(checkpoint_path),
        }
    )

    if args.output is not None:
        output_path = _resolve_path(args.output, base_dir=PROJECT_ROOT)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
