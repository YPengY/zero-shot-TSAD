"""Random-seed helpers for reproducible training runs."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs.

    This helper covers the libraries used by the current training stack. It
    does not promise full determinism across all CUDA kernels.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
