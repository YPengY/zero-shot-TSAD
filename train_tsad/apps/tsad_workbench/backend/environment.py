from __future__ import annotations

import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_ROOT / "static"
TRAIN_TSAD_ROOT = APP_ROOT.parents[1]
PROJECT_ROOT = TRAIN_TSAD_ROOT.parent
SYNTH_ROOT = PROJECT_ROOT / "synthetic_tsad"
STUDIO_APP_ROOT = SYNTH_ROOT / "apps" / "tsad_studio"
SYNTH_SRC_ROOT = SYNTH_ROOT / "src"

PREVIEW_CACHE_LIMIT = 12
JOB_LOG_LIMIT = 2500
DEFAULT_FREE_SPACE_BUFFER_BYTES = 512 * 1024 * 1024
DEFAULT_METADATA_BYTES_PER_SAMPLE = 24 * 1024
DEFAULT_NPZ_OVERHEAD_BYTES_PER_SAMPLE = 8 * 1024
DEFAULT_WORKBENCH_TRAIN_TEMPLATE = "timercd_small.json"
WORKBENCH_GENERATION_METADATA_FILENAME = "workbench_generation_metadata.json"


def bootstrap_external_paths() -> None:
    """Expose synthetic-tsad studio modules to the workbench backend."""

    if str(STUDIO_APP_ROOT) not in sys.path:
        sys.path.insert(0, str(STUDIO_APP_ROOT))
    if str(SYNTH_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SYNTH_SRC_ROOT))
