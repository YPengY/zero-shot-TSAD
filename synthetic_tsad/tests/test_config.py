from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from synthtsad.config import DEFAULT_CONFIG, load_config_from_raw


class ConfigTests(unittest.TestCase):
    def test_load_config_from_raw_uses_defaults_for_partial_payload(self) -> None:
        config = load_config_from_raw({"num_samples": 3})

        self.assertEqual(config.num_samples, 3)
        self.assertEqual(config.sequence_length.min, DEFAULT_CONFIG["sequence_length"]["min"])

    def test_load_config_from_raw_rejects_unknown_top_level_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            load_config_from_raw({"unknown_key": 1})


if __name__ == "__main__":
    unittest.main()
