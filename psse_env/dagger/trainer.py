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
        unknown_class_policy: str = "error",
        unknown_class_weight: float = 0.05,
        max_duplicate_count: int = 2,
        max_rows_per_root: int | None = None,
        late_iteration_model_fraction: float = 0.25,
        require_late_iteration_model_quota: bool = False,
    ) -> None:
        self.train_fn = train_fn
        self.sample_size = sample_size
        self.rng = random.Random(seed)
        self.class_weights = class_weights
        self.unknown_class_policy = unknown_class_policy
        self.unknown_class_weight = unknown_class_weight
        self.max_duplicate_count = max_duplicate_count
        self.max_rows_per_root = max_rows_per_root
        self.late_iteration_model_fraction = late_iteration_model_fraction
        self.require_late_iteration_model_quota = require_late_iteration_model_quota
        self.last_replay_report: dict[str, Any] | None = None

    def train(self, policy: Any, aggregate_dataset: list[dict[str, Any]]) -> Any:
        buffer = BalancedReplayBuffer(
            aggregate_dataset,
            class_weights=self.class_weights,
            unknown_class_policy=self.unknown_class_policy,
            unknown_class_weight=self.unknown_class_weight,
            max_duplicate_count=self.max_duplicate_count,
            max_rows_per_root=self.max_rows_per_root,
            late_iteration_model_fraction=self.late_iteration_model_fraction,
            require_late_iteration_model_quota=(
                self.require_late_iteration_model_quota
            ),
        )
        size = self.sample_size if self.sample_size is not None else len(aggregate_dataset)
        sample = buffer.sample(size, rng=self.rng)
        self.last_replay_report = buffer.sample_report()
        return self.train_fn(policy, sample)

    def __call__(self, policy: Any, aggregate_dataset: list[dict[str, Any]]) -> Any:
        return self.train(policy, aggregate_dataset)
