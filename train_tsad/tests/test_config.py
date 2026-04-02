from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_tsad.config import DataConfig, ExperimentConfig


class ConfigTests(unittest.TestCase):
    def test_data_config_rejects_incompatible_patch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            DataConfig(dataset_root=Path("data"), context_size=63, patch_size=16)

    def test_experiment_config_collects_unknown_keys_into_extras(self) -> None:
        config = ExperimentConfig.from_mapping(
            {
                "experiment_name": "smoke",
                "data": {"dataset_root": "data/packed", "patch_size": 16, "context_size": 1024},
                "unexpected_flag": True,
            }
        )

        self.assertTrue(config.extras["unexpected_flag"])
        self.assertEqual(config.data.dataset_root, Path("data/packed"))


if __name__ == "__main__":
    unittest.main()
