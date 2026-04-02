from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger
from ..workflows import InspectionOptions, run_inspection

TRAIN_TSAD_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = TRAIN_TSAD_ROOT.parent


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data-quality inspection."""

    parser = argparse.ArgumentParser(description="Inspect data quality before TSAD training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=TRAIN_TSAD_ROOT / "configs" / "timercd_pretrain_paper_aligned.json",
        help="Path to experiment config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Inspect only one split. Defaults to the training split unless `--all-splits` is set.",
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
    """Run the CLI inspection workflow."""

    configure_logging()
    logger = get_logger("train_tsad.cli.inspect_data")
    args = parse_args()
    result = run_inspection(
        args.config,
        base_dir=PROJECT_ROOT,
        options=InspectionOptions(
            split=args.split,
            inspect_all_splits=bool(args.all_splits),
            max_samples=args.max_samples,
            output=args.output,
        ),
        logger=logger,
    )
    print(json.dumps(result.payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
