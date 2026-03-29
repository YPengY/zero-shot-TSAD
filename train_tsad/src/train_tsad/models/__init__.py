"""Public model exports for TimeRCD-style training.

The package is layered from patch/token encoders up to task heads and the
assembled `TimeRCDModel`, with a thin factory for workflow-facing construction.
"""

from .encoder import TransformerEncoder
from .factory import build_timercd_model
from .heads import (
    AnomalyHead,
    ObservationSpaceAnomalyHead,
    ProjectedAnomalyHead,
    ProjectedObservationSpaceAnomalyHead,
    ProjectedReconstructionHead,
    ReconstructionHead,
    SharedOutputProjection,
)
from .patch_embed import PatchEmbedding, PatchEmbeddingOutput
from .positional_encoding import GridPositionalEncoding
from .timercd import TimeRCDModel

__all__ = [
    "AnomalyHead",
    "GridPositionalEncoding",
    "ObservationSpaceAnomalyHead",
    "PatchEmbedding",
    "PatchEmbeddingOutput",
    "build_timercd_model",
    "ProjectedAnomalyHead",
    "ProjectedObservationSpaceAnomalyHead",
    "ProjectedReconstructionHead",
    "ReconstructionHead",
    "SharedOutputProjection",
    "TimeRCDModel",
    "TransformerEncoder",
]
