from __future__ import annotations

from collections.abc import Iterable

from torch import nn
from torch.optim import AdamW, Optimizer

from ..config import OptimizerConfig


def _trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Yield only parameters that require gradients."""

    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Build the optimizer for the current training run."""

    name = config.name.lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer `{config.name}`. Only `adamw` is implemented.")

    return AdamW(
        _trainable_parameters(model),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )


__all__ = ["build_optimizer"]
