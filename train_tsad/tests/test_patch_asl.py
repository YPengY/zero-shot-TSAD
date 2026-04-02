from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_tsad.config import LossConfig
from train_tsad.interfaces import Batch, LossOutput, ModelOutput
from train_tsad.losses import PatchAsymmetricLoss, TimeRCDMultiTaskLoss


class PatchAsymmetricLossTests(unittest.TestCase):
    def test_patch_asymmetric_loss_returns_scalar_and_metrics(self) -> None:
        loss_fn = PatchAsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)
        batch = Batch(
            sample_ids=["sample-0"],
            context_start=torch.tensor([0]),
            context_end=torch.tensor([4]),
            valid_lengths=torch.tensor([4]),
            inputs=torch.zeros((1, 4, 2), dtype=torch.float32),
            patch_labels=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
            patch_valid_mask=torch.ones((1, 2, 2), dtype=torch.bool),
        )
        output = ModelOutput(
            logits=torch.tensor([[[2.5, -1.0], [-0.5, 1.5]]], dtype=torch.float32),
        )

        result = loss_fn(batch, output)

        self.assertIsInstance(result, LossOutput)
        self.assertGreater(result.loss.item(), 0.0)
        self.assertIn("anomaly_loss", result.metrics)
        self.assertIn("predicted_positive_rate", result.metrics)

    def test_multitask_loss_uses_patch_asymmetric_loss_when_configured(self) -> None:
        config = LossConfig(
            anomaly_loss_type="asl",
            anomaly_pos_weight=None,
            anomaly_asl_gamma_neg=2.0,
            anomaly_asl_gamma_pos=0.0,
            anomaly_asl_clip=0.05,
        )

        loss_module = TimeRCDMultiTaskLoss.from_config(config)

        self.assertIsInstance(loss_module.anomaly_loss, PatchAsymmetricLoss)


if __name__ == "__main__":
    unittest.main()
