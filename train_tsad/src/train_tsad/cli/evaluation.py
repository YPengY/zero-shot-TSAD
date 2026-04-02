from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger
from ..workflows import EvaluationOverrides, run_evaluation

TRAIN_TSAD_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = TRAIN_TSAD_ROOT.parent


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline checkpoint evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a trained TimeRCD checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=TRAIN_TSAD_ROOT / "configs" / "timercd_pretrain_paper_aligned.json",
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
    """Run the CLI evaluation workflow."""

    configure_logging()
    logger = get_logger("train_tsad.cli.evaluation")
    args = parse_args()
    result = run_evaluation(
        args.config,
        base_dir=PROJECT_ROOT,
        overrides=EvaluationOverrides(
            device=args.device,
            split=args.split,
            checkpoint=args.checkpoint,
            output=args.output,
        ),
        logger=logger,
    )
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
