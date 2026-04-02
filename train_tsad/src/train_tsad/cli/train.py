from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger
from ..workflows import TrainingOverrides, run_training

TRAIN_TSAD_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = TRAIN_TSAD_ROOT.parent


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model training."""

    parser = argparse.ArgumentParser(description="Train the paper-aligned TimeRCD model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=TRAIN_TSAD_ROOT / "configs" / "timercd_pretrain_paper_aligned.json",
        help="Path to a JSON or YAML experiment config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional max epoch override.",
    )
    parser.add_argument(
        "--inspect-data",
        action="store_true",
        help="Run split-level data quality inspection before training starts.",
    )
    parser.add_argument(
        "--inspect-max-samples",
        type=int,
        default=None,
        help="Optional cap on inspected samples per split during data inspection.",
    )
    parser.add_argument(
        "--inspect-output",
        type=Path,
        default=None,
        help="Optional path to save the data quality report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI training workflow."""

    configure_logging()
    logger = get_logger("train_tsad.cli.train")
    args = parse_args()
    result = run_training(
        args.config,
        base_dir=PROJECT_ROOT,
        overrides=TrainingOverrides(
            device=args.device,
            max_epochs=args.max_epochs,
            inspect_data=bool(args.inspect_data),
            inspect_max_samples=args.inspect_max_samples,
            inspect_output=args.inspect_output,
        ),
        logger=logger,
    )
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
