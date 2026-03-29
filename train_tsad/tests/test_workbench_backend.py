from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKBENCH_ROOT = Path(__file__).resolve().parents[1] / "apps" / "tsad_workbench"
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from backend.dataset_browser import iter_context_bounds
from backend.training_metrics import build_metric_series, build_training_kpis


class WorkbenchBackendTests(unittest.TestCase):
    def test_iter_context_bounds_keeps_tail_window(self) -> None:
        bounds = iter_context_bounds(
            sequence_length=10,
            context_size=4,
            stride=4,
            include_tail=True,
            pad_short_sequences=True,
        )

        self.assertEqual(bounds, [(0, 4), (4, 8), (8, 10)])

    def test_build_metric_series_and_kpis_from_history(self) -> None:
        history = [
            {
                "epoch": 1,
                "train": {"total_loss": 1.5, "predicted_positive_rate": 0.1},
                "val": {"pr_auc": 0.7},
            },
            {
                "epoch": 2,
                "train": {"total_loss": 1.0, "predicted_positive_rate": 0.2},
                "val": {"pr_auc": 0.8},
            },
        ]
        summary = {"best_epoch": 2, "best_metric": 0.8}
        progress = {
            "epoch_current": 2,
            "epoch_total": 5,
            "status": "running",
            "stage": "training",
            "overall_progress_ratio": 0.4,
            "learning_rate": 1e-4,
            "monitor_metric": "pr_auc",
            "monitor_mode": "max",
            "latest_train_metrics": {"total_loss": 1.0},
            "latest_val_metrics": {"pr_auc": 0.8},
        }

        history_view = build_metric_series(history)
        kpis = build_training_kpis(history, summary, progress)

        self.assertEqual(history_view["epochs"], [1, 2])
        self.assertIn("train.total_loss", history_view["series"])
        self.assertEqual(history_view["preferred_loss"], "train.total_loss")
        self.assertEqual(kpis["best_epoch"], 2)
        self.assertEqual(kpis["latest_monitor_value"], 0.8)


if __name__ == "__main__":
    unittest.main()
