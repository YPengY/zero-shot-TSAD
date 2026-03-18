from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthtsad.io import pack_synthetic_corpus, pack_windows_from_packed_corpus


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
    parser.add_argument(
        "--window-level",
        action="store_true",
        help="Convert sample-packed shards into window-packed training shards.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=1024,
        help="Context window length used by --window-level mode.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size used by --window-level mode.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Window stride used by --window-level mode (defaults to context_size).",
    )
    parser.add_argument(
        "--windows-per-shard",
        type=int,
        default=4096,
        help="Maximum number of windows per output shard in --window-level mode.",
    )
    parser.add_argument(
        "--no-include-tail",
        action="store_true",
        help="Disable tail window retention in --window-level mode.",
    )
    parser.add_argument(
        "--no-pad-short-sequences",
        action="store_true",
        help="Drop sequences shorter than context_size in --window-level mode.",
    )
    parser.add_argument(
        "--no-debug-sidecar",
        action="store_true",
        help="Do not write debug.<split>.jsonl sidecar in --window-level mode.",
    )
    parser.add_argument(
        "--min-patch-positive-ratio",
        type=float,
        default=None,
        help="Drop windows whose patch_positive_ratio is below this value in --window-level mode.",
    )
    parser.add_argument(
        "--min-anomaly-point-ratio",
        type=float,
        default=None,
        help="Drop windows whose anomaly_point_ratio is below this value in --window-level mode.",
    )
    args = parser.parse_args()

    if args.window_level:
        report = pack_windows_from_packed_corpus(
            input_root=args.input,
            output_root=args.output,
            split=args.split,
            context_size=args.context_size,
            patch_size=args.patch_size,
            stride=args.stride,
            include_tail=not args.no_include_tail,
            pad_short_sequences=not args.no_pad_short_sequences,
            windows_per_shard=args.windows_per_shard,
            overwrite=args.overwrite,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            write_debug_sidecar=not args.no_debug_sidecar,
            min_patch_positive_ratio=args.min_patch_positive_ratio,
            min_anomaly_point_ratio=args.min_anomaly_point_ratio,
        )
    else:
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
