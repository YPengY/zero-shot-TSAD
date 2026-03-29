from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_tsad.config import DataConfig, ExperimentConfig, LossConfig
from train_tsad.data import SyntheticTsadDataset, manifest_is_window_packed
from train_tsad.data.dataset import SyntheticTsadDataset as LegacySyntheticTsadDataset
from train_tsad.interfaces import RawSample
from train_tsad.training import resolve_loss_weights


class _InMemoryRawDataset:
    def __init__(self, samples: list[RawSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> RawSample:
        return self._samples[index]


class RuntimeRefactorTests(unittest.TestCase):
    def test_manifest_is_window_packed_detects_window_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            manifest_path.write_text(
                (
                    '{"shard_npz_path":"shard_0001.npz","window_index":0,'
                    '"context_start":0,"context_end":16,"valid_length":16}\n'
                ),
                encoding="utf-8",
            )

            self.assertTrue(manifest_is_window_packed(manifest_path))

    def test_experiment_config_clone_is_independent(self) -> None:
        original = ExperimentConfig.from_mapping(
            {
                "experiment_name": "clone-check",
                "data": {"dataset_root": "data/packed"},
                "tags": ["baseline"],
            }
        )

        cloned = original.clone()
        cloned.tags.append("mutated")
        cloned.train.max_epochs = 3

        self.assertEqual(original.tags, ["baseline"])
        self.assertEqual(original.train.max_epochs, 50)

    def test_resolve_loss_weights_auto_computes_patch_and_point_balances(self) -> None:
        sample = RawSample(
            sample_id="sample-0",
            split="train",
            series=np.zeros((4, 2), dtype=np.float32),
            point_mask=np.asarray(
                [
                    [1, 0],
                    [0, 0],
                    [0, 1],
                    [0, 0],
                ],
                dtype=np.uint8,
            ),
            point_mask_any=np.asarray([1, 0, 1, 0], dtype=np.uint8),
        )
        dataset = _InMemoryRawDataset([sample])
        data_config = DataConfig(
            dataset_root=Path("data"),
            context_size=4,
            patch_size=2,
            stride=4,
        )
        loss_config = LossConfig(
            anomaly_loss_type="bce",
            anomaly_pos_weight="auto",
            point_anomaly_loss_weight=1.0,
            point_anomaly_pos_weight="auto",
        )

        resolved = resolve_loss_weights(
            loss_config,
            data_config=data_config,
            train_raw_dataset=dataset,
            logger=logging.getLogger("train_tsad.tests"),
        )

        self.assertEqual(resolved.anomaly_pos_weight, 1.0)
        self.assertEqual(resolved.point_anomaly_pos_weight, 3.0)
        self.assertIsNotNone(resolved.patch_feature_stats)
        self.assertIsNotNone(resolved.point_feature_stats)

    def test_dataset_module_split_keeps_raw_dataset_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                split_dir / "sample_000001.npz",
                series=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                point_mask=np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
                point_mask_any=np.asarray([1, 0], dtype=np.uint8),
            )
            (split_dir / "sample_000001.json").write_text(
                '{"summary":{"sample_id":1},"debug":{"source":"unit-test"}}',
                encoding="utf-8",
            )

            dataset = SyntheticTsadDataset(tmpdir, split="train")
            sample = dataset[0]

            self.assertIs(LegacySyntheticTsadDataset, SyntheticTsadDataset)
            self.assertEqual(dataset.sample_id(0), "sample_000001")
            self.assertEqual(sample.sample_id, "sample_000001")
            self.assertEqual(sample.series.shape, (2, 2))
            self.assertEqual(sample.point_mask_any.tolist(), [1, 0])
            self.assertEqual(sample.metadata["debug"]["source"], "unit-test")


if __name__ == "__main__":
    unittest.main()
