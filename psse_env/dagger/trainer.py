from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, Mapping

from .replay_buffer import BalancedReplayBuffer


class DaggerTrainer:
    """Small adapter that trains from a balanced view of the full aggregate."""

    def __init__(
        self,
        train_fn: Callable[[Any, list[dict[str, Any]]], Any],
        *,
        sample_size: int | None = None,
        seed: int = 0,
        class_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.train_fn = train_fn
        self.sample_size = sample_size
        self.rng = random.Random(seed)
        self.class_weights = class_weights

    def train(self, policy: Any, aggregate_dataset: list[dict[str, Any]]) -> Any:
        buffer = BalancedReplayBuffer(aggregate_dataset, class_weights=self.class_weights)
        size = self.sample_size if self.sample_size is not None else len(aggregate_dataset)
        return self.train_fn(policy, buffer.sample(size, rng=self.rng))

    def __call__(self, policy: Any, aggregate_dataset: list[dict[str, Any]]) -> Any:
        return self.train(policy, aggregate_dataset)
