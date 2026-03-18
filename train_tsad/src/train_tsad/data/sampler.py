from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class GroupedWindowSampler(Sampler[int]):
    """Shuffle precomputed index blocks while preserving in-block locality."""

    def __init__(
        self,
        blocks: Sequence[Sequence[int]],
        *,
        seed: int = 0,
    ) -> None:
        self.blocks = tuple(tuple(int(index) for index in block) for block in blocks if block)
        self.seed = int(seed)
        self._iteration = 0
        self._length = sum(len(block) for block in self.blocks)

    def __iter__(self) -> Iterator[int]:
        """Yield indices by shuffled block order and stable in-block order."""

        order = list(range(len(self.blocks)))
        if len(order) > 1:
            rng = random.Random(self.seed + self._iteration)
            rng.shuffle(order)
        self._iteration += 1

        for block_index in order:
            yield from self.blocks[block_index]

    def __len__(self) -> int:
        """Return the number of flat sample indices produced per epoch."""

        return self._length


__all__ = ["GroupedWindowSampler"]
