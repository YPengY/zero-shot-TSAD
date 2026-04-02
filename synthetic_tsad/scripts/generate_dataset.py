from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthtsad.config import load_config, load_config_from_raw
from synthtsad.pipeline import SyntheticGeneratorPipeline


def _deep_merge_dict(
    base: Mapping[str, object], override: Mapping[str, object]
) -> dict[str, object]:
    """Recursively merge `override` into `base` and return a new dictionary."""

    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(base=base_value, override=value)
        else:
            merged[key] = value
    return merged


def _load_raw_overrides(path: Path) -> dict[str, object]:
    """Load raw config overrides from a JSON or YAML file."""

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YAML override loading requires PyYAML. Install it or use a JSON overrides file."
            ) from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(
            f"Unsupported overrides format `{suffix}`. Expected one of: .json, .yaml, .yml."
        )

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Overrides payload must be a mapping.")
    return dict(payload)


def _cfg_from_overrides(
    path: Path,
    num_samples: int | None,
    seed: int | None,
    num_series: int | None,
    sequence_length: int | None,
    raw_overrides_path: Path | None,
    disable_trend: bool,
    disable_seasonality: bool,
    disable_noise: bool,
    disable_causal: bool,
    disable_local_anomaly: bool,
    disable_seasonal_anomaly: bool,
):
    cfg = load_config(path)
    has_toggle = any(
        [
            disable_trend,
            disable_seasonality,
            disable_noise,
            disable_causal,
            disable_local_anomaly,
            disable_seasonal_anomaly,
        ]
    )
    if (
        num_samples is None
        and seed is None
        and num_series is None
        and sequence_length is None
        and raw_overrides_path is None
        and not has_toggle
    ):
        return cfg

    raw = dict(cfg.raw)
    if raw_overrides_path is not None:
        raw = _deep_merge_dict(base=raw, override=_load_raw_overrides(raw_overrides_path))
    if num_samples is not None:
        raw["num_samples"] = int(num_samples)
    if seed is not None:
        raw["seed"] = int(seed)
    if num_series is not None:
        raw["num_series"] = {"min": int(num_series), "max": int(num_series)}
    if sequence_length is not None:
        raw["sequence_length"] = {
            "min": int(sequence_length),
            "max": int(sequence_length),
        }

    debug = dict(raw.get("debug", {}))
    if disable_trend:
        debug["enable_trend"] = False
    if disable_seasonality:
        debug["enable_seasonality"] = False
    if disable_noise:
        debug["enable_noise"] = False
    if disable_causal:
        debug["enable_causal"] = False
    if disable_local_anomaly:
        debug["enable_local_anomaly"] = False
    if disable_seasonal_anomaly:
        debug["enable_seasonal_anomaly"] = False
    if debug:
        raw["debug"] = debug

    return load_config_from_raw(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic TSAD corpus")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--num-samples", type=int, default=None, help="Override config.num_samples")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed")
    parser.add_argument(
        "--num-series", type=int, default=None, help="Override sample series count (fixed min=max)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Override sample sequence length (fixed min=max).",
    )
    parser.add_argument(
        "--raw-overrides",
        type=Path,
        default=None,
        help="Path to JSON/YAML raw config overrides merged before scalar CLI overrides.",
    )
    parser.add_argument("--disable-trend", action="store_true", help="Disable trend component")
    parser.add_argument(
        "--disable-seasonality", action="store_true", help="Disable seasonality component"
    )
    parser.add_argument("--disable-noise", action="store_true", help="Disable noise component")
    parser.add_argument("--disable-causal", action="store_true", help="Disable causal ARX stage")
    parser.add_argument(
        "--disable-local-anomaly", action="store_true", help="Disable local anomaly injector"
    )
    parser.add_argument(
        "--disable-seasonal-anomaly", action="store_true", help="Disable seasonal anomaly injector"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print final merged config and exit"
    )
    parser.add_argument(
        "--compress-output",
        action="store_true",
        help="Compress per-sample raw NPZ outputs before shard packing.",
    )
    parser.add_argument(
        "--direct-pack",
        action="store_true",
        help="Write shard-packed outputs directly and skip raw sample files.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Split name used by direct-pack mode. Defaults to train when omitted.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=512,
        help="Maximum number of samples per shard in direct-pack mode.",
    )
    parser.add_argument(
        "--direct-window-pack",
        action="store_true",
        help="Write window-level shard outputs directly (skip sample-packed shards).",
    )
    parser.add_argument(
        "--window-context-size",
        type=int,
        default=1024,
        help="Context size used by --direct-window-pack.",
    )
    parser.add_argument(
        "--window-patch-size",
        type=int,
        default=16,
        help="Patch size used by --direct-window-pack.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Window stride used by --direct-window-pack (defaults to context size).",
    )
    parser.add_argument(
        "--window-windows-per-shard",
        type=int,
        default=4096,
        help="Maximum windows per shard in --direct-window-pack mode.",
    )
    parser.add_argument(
        "--window-no-include-tail",
        action="store_true",
        help="Disable tail window retention in --direct-window-pack mode.",
    )
    parser.add_argument(
        "--window-no-pad-short-sequences",
        action="store_true",
        help="Drop short sequences in --direct-window-pack mode.",
    )
    parser.add_argument(
        "--window-no-debug-sidecar",
        action="store_true",
        help="Disable debug.<split>.jsonl sidecar in --direct-window-pack mode.",
    )
    parser.add_argument(
        "--window-min-patch-positive-ratio",
        type=float,
        default=None,
        help="Drop windows whose patch_positive_ratio is below this value in --direct-window-pack mode.",
    )
    parser.add_argument(
        "--window-min-anomaly-point-ratio",
        type=float,
        default=None,
        help="Drop windows whose anomaly_point_ratio is below this value in --direct-window-pack mode.",
    )
    args = parser.parse_args()

    cfg = _cfg_from_overrides(
        path=args.config,
        num_samples=args.num_samples,
        seed=args.seed,
        num_series=args.num_series,
        sequence_length=args.sequence_length,
        raw_overrides_path=args.raw_overrides,
        disable_trend=args.disable_trend,
        disable_seasonality=args.disable_seasonality,
        disable_noise=args.disable_noise,
        disable_causal=args.disable_causal,
        disable_local_anomaly=args.disable_local_anomaly,
        disable_seasonal_anomaly=args.disable_seasonal_anomaly,
    )
    if args.print_config:
        print(json.dumps(cfg.raw, ensure_ascii=False, indent=2))
        return

    pipeline = SyntheticGeneratorPipeline(cfg)
    pipeline.run(
        args.output,
        compress_output=args.compress_output,
        direct_pack=args.direct_pack,
        direct_window_pack=args.direct_window_pack,
        split=args.split,
        samples_per_shard=args.samples_per_shard,
        window_context_size=args.window_context_size,
        window_patch_size=args.window_patch_size,
        window_stride=args.window_stride,
        window_include_tail=not args.window_no_include_tail,
        window_pad_short_sequences=not args.window_no_pad_short_sequences,
        window_windows_per_shard=args.window_windows_per_shard,
        window_debug_sidecar=not args.window_no_debug_sidecar,
        window_min_patch_positive_ratio=args.window_min_patch_positive_ratio,
        window_min_anomaly_point_ratio=args.window_min_anomaly_point_ratio,
    )


if __name__ == "__main__":
    main()
