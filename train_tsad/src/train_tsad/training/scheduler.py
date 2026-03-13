from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, SequentialLR

from ..config import OptimizerConfig


def build_scheduler(
    optimizer: Optimizer,
    config: OptimizerConfig,
    *,
    max_epochs: int,
) -> LRScheduler | None:
    """Build an epoch-level learning-rate scheduler."""

    name = config.scheduler.lower()
    if name in {"none", "constant"}:
        return None

    if name != "cosine":
        raise ValueError(
            f"Unsupported scheduler `{config.scheduler}`. Supported: cosine, constant, none."
        )

    if max_epochs <= 0:
        raise ValueError("`max_epochs` must be positive.")

    cosine_epochs = max(1, max_epochs - config.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=config.min_lr,
    )

    if config.warmup_epochs <= 0:
        return cosine

    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config.warmup_epochs],
    )


__all__ = ["build_scheduler"]
