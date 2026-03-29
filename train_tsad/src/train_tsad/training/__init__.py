"""Public training exports for workflow orchestration.

The training package separates pre-training inspection, optimizer/scheduler
construction, checkpoint management, and the stateful training loop so each
layer can be reasoned about independently.
"""

from .checkpoint import CheckpointManager
from .optimizer import build_optimizer
from .preflight import (
    DataQualityInspectionResult,
    LabelBalanceStatistics,
    ResolvedLossWeights,
    estimate_patch_feature_balance,
    estimate_point_feature_balance,
    resolve_loss_weights,
    run_data_quality_inspection,
)
from .scheduler import build_scheduler
from .trainer import EpochResult, FitResult, Trainer

__all__ = [
    "CheckpointManager",
    "DataQualityInspectionResult",
    "EpochResult",
    "FitResult",
    "LabelBalanceStatistics",
    "ResolvedLossWeights",
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "estimate_patch_feature_balance",
    "estimate_point_feature_balance",
    "resolve_loss_weights",
    "run_data_quality_inspection",
]
