from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthtsad.io import pack_synthetic_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack per-sample synthetic outputs into shard-based training corpora"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input sample directory or split root")
    parser.add_argument("--output", type=Path, required=True, help="Output packed dataset root")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Treat input as a single split directory; otherwise auto-detect train/val/test subdirs",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=512,
        help="Maximum number of samples packed into one shard",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name recorded in dataset_meta.json",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Dataset version recorded in dataset_meta.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before writing packed shards",
    )
    args = parser.parse_args()

    report = pack_synthetic_corpus(
        input_root=args.input,
        output_root=args.output,
        split=args.split,
        samples_per_shard=args.samples_per_shard,
        overwrite=args.overwrite,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
    )

    print(f"Packed dataset: {report.dataset_name}/{report.dataset_version}")
    print(f"Output root: {report.output_root}")
    for split_name, split_report in report.splits.items():
        print(
            f"[{split_name}] samples={split_report.num_samples} "
            f"shards={split_report.num_shards} "
            f"total_points={split_report.total_points}"
        )


if __name__ == "__main__":
    main()
