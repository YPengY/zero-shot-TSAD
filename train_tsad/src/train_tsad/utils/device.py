"""Device-resolution helpers for CLI and workflow code."""

from __future__ import annotations

import logging

import torch


def resolve_torch_device(requested: str, *, logger: logging.Logger) -> torch.device:
    """Resolve a requested device string with a safe CUDA fallback.

    Requesting CUDA on a machine without CUDA support is treated as a recoverable
    configuration mismatch rather than a hard failure.
    """

    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)
