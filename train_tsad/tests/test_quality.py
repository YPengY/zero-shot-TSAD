from __future__ import annotations

import unittest

import numpy as np

from train_tsad.data import DataQualityInspector, SlidingContextWindowizer
from train_tsad.interfaces import RawSample


class _InMemoryRawDataset:
    def __init__(self, samples: list[RawSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> RawSample:
        return self._samples[index]


class DataQualityInspectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.windowizer = SlidingContextWindowizer(
            context_size=4,
            patch_size=2,
            stride=4,
        )

    def test_inspect_many_allows_training_for_non_blocking_warnings(self) -> None:
        sample = RawSample(
            sample_id="train-0",
            split="train",
            series=np.asarray(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                ],
                dtype=np.float32,
            ),
            point_mask=np.asarray(
                [
                    [1, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                dtype=np.uint8,
            ),
            point_mask_any=np.asarray([0, 0, 0, 0], dtype=np.uint8),
        )
        dataset = _InMemoryRawDataset([sample])
        inspector = DataQualityInspector(windowizer=self.windowizer)

        report = inspector.inspect_many(
            {"train": dataset},
            expected_training_split="train",
        )
        train_report = report.split_reports["train"]
        issue_codes = {issue.code for issue in train_report.issues}

        self.assertIn("too_few_samples", issue_codes)
        self.assertIn("point_mask_any_mismatch", issue_codes)
        self.assertFalse(train_report.has_blocking_issues)
        self.assertGreater(train_report.stats["patch_positive_rate"], 0.0)
        self.assertTrue(report.recommended_to_train)

    def test_inspect_many_blocks_training_when_patch_labels_are_all_negative(self) -> None:
        sample = RawSample(
            sample_id="train-0",
            split="train",
            series=np.asarray(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0],
                ],
                dtype=np.float32,
            ),
            point_mask=np.zeros((4, 2), dtype=np.uint8),
            point_mask_any=np.zeros((4,), dtype=np.uint8),
        )
        dataset = _InMemoryRawDataset([sample])
        inspector = DataQualityInspector(windowizer=self.windowizer)

        report = inspector.inspect_many(
            {"train": dataset},
            expected_training_split="train",
        )
        train_report = report.split_reports["train"]
        issue_codes = {issue.code for issue in train_report.issues}

        self.assertIn("zero_patch_positives", issue_codes)
        self.assertTrue(train_report.has_blocking_issues)
        self.assertEqual(train_report.stats["patch_positive_rate"], 0.0)
        self.assertFalse(report.recommended_to_train)


if __name__ == "__main__":
    unittest.main()
