from __future__ import annotations

from ..interfaces import ContextWindowSample, ContextWindowizerProtocol, DatasetProtocol
from .records import _WindowRecord


class ContextWindowDataset:
    """Adapt a raw-sample dataset into a flat context-window dataset."""

    def __init__(
        self,
        raw_dataset: DatasetProtocol,
        windowizer: ContextWindowizerProtocol,
        *,
        enable_direct_window_read: bool = True,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.windowizer = windowizer
        self.enable_direct_window_read = bool(enable_direct_window_read)
        self._index, self._sample_blocks, self._shard_blocks = self._build_index()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> ContextWindowSample:
        record = self._index[index]
        if self.enable_direct_window_read:
            dataset_slicer = getattr(self.raw_dataset, "slice_window", None)
            if callable(dataset_slicer):
                return dataset_slicer(
                    record.raw_index,
                    start=record.context_start,
                    end=record.context_end,
                    windowizer=self.windowizer,
                )
        sample = self.raw_dataset[record.raw_index]
        return self.windowizer.slice_window(
            sample,
            start=record.context_start,
            end=record.context_end,
        )

    def grouped_blocks(self, strategy: str) -> tuple[tuple[int, ...], ...]:
        if strategy == "sample_block":
            return self._sample_blocks
        if strategy == "shard_block":
            return self._shard_blocks
        raise ValueError(
            f"Unsupported grouping strategy `{strategy}`. Supported: sample_block, shard_block."
        )

    def _build_index(
        self,
    ) -> tuple[list[_WindowRecord], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        index: list[_WindowRecord] = []
        sample_blocks: list[tuple[int, ...]] = []
        shard_blocks: list[tuple[int, ...]] = []
        current_shard_key: str | None = None
        current_shard_block: list[int] = []

        for raw_index in range(len(self.raw_dataset)):
            sample_length = self._sample_length(raw_index)
            sample_bounds = self.windowizer.iter_context_bounds(sample_length)
            if not sample_bounds:
                continue

            sample_window_indices: list[int] = []
            for context_start, context_end in sample_bounds:
                index.append(
                    _WindowRecord(
                        raw_index=raw_index,
                        context_start=context_start,
                        context_end=context_end,
                    )
                )
                sample_window_indices.append(len(index) - 1)

            sample_blocks.append(tuple(sample_window_indices))

            shard_key = self._sample_shard_key(raw_index) or f"sample:{self._sample_id(raw_index)}"
            if current_shard_key is None:
                current_shard_key = shard_key
            elif shard_key != current_shard_key:
                if current_shard_block:
                    shard_blocks.append(tuple(current_shard_block))
                current_shard_block = []
                current_shard_key = shard_key
            current_shard_block.extend(sample_window_indices)

        if current_shard_block:
            shard_blocks.append(tuple(current_shard_block))

        if not index:
            raise FileNotFoundError(
                "No full context windows could be built from the provided dataset. "
                "Samples shorter than `context_size` and tail remainders are discarded."
            )

        return index, tuple(sample_blocks), tuple(shard_blocks)

    def _sample_id(self, raw_index: int) -> str:
        getter = getattr(self.raw_dataset, "sample_id", None)
        if callable(getter):
            return str(getter(raw_index))
        return str(self.raw_dataset[raw_index].sample_id)

    def _sample_length(self, raw_index: int) -> int:
        getter = getattr(self.raw_dataset, "sample_length", None)
        if callable(getter):
            return int(getter(raw_index))
        return int(self.raw_dataset[raw_index].series.shape[0])

    def _sample_shard_key(self, raw_index: int) -> str | None:
        getter = getattr(self.raw_dataset, "sample_shard_key", None)
        if callable(getter):
            shard_key = getter(raw_index)
            return None if shard_key is None else str(shard_key)
        return None


__all__ = ["ContextWindowDataset"]
