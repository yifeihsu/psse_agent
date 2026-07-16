from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable, Mapping


DEFAULT_REPLAY_WEIGHTS: dict[str, float] = {
    "clean_successful": 0.30,
    "rejected_candidate_recovery": 0.25,
    "accepted_partial_continuation": 0.20,
    "accepted_partial_commit": 0.10,
    "accepted_final_commit": 0.10,
    "invalid_precondition_recovery": 0.10,
    "terminal_decision": 0.10,
    "loop_repetition": 0.05,
}


class BalancedReplayBuffer:
    """Aggregate all DAgger rows while sampling recovery classes deliberately."""

    def __init__(
        self,
        examples: Iterable[Mapping[str, Any]] | None = None,
        *,
        class_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.class_weights = dict(class_weights or DEFAULT_REPLAY_WEIGHTS)
        if any(weight < 0 for weight in self.class_weights.values()) or sum(self.class_weights.values()) <= 0:
            raise ValueError("Replay weights must be non-negative and contain positive mass.")
        self._examples: list[dict[str, Any]] = [dict(row) for row in (examples or [])]

    def add(self, example: Mapping[str, Any]) -> None:
        self._examples.append(dict(example))

    def extend(self, examples: Iterable[Mapping[str, Any]]) -> None:
        self._examples.extend(dict(row) for row in examples)

    def all_examples(self) -> list[dict[str, Any]]:
        return list(self._examples)

    def sample(self, size: int, *, rng: random.Random | None = None) -> list[dict[str, Any]]:
        if size < 0:
            raise ValueError("sample size must be non-negative")
        if size == 0:
            return []
        if not self._examples:
            raise ValueError("cannot sample an empty replay buffer")
        generator = rng or random.Random()
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._examples:
            state_class = str(row.get("state_class") or (row.get("labels") or {}).get("state_class") or "clean_successful")
            buckets[state_class].append(row)
        available_weights = {
            key: weight for key, weight in self.class_weights.items() if weight > 0 and buckets.get(key)
        }
        for key in buckets:
            if key not in available_weights:
                available_weights[key] = 0.0
        if not any(available_weights.values()):
            available_weights = {key: 1.0 for key in buckets}
        total_weight = sum(available_weights.values())
        exact = {key: size * weight / total_weight for key, weight in available_weights.items()}
        counts = {key: int(value) for key, value in exact.items()}
        remainder = size - sum(counts.values())
        order = sorted(exact, key=lambda key: (exact[key] - counts[key], available_weights[key], key), reverse=True)
        for key in order[:remainder]:
            counts[key] += 1

        result: list[dict[str, Any]] = []
        for key in sorted(counts):
            bucket = buckets[key]
            count = counts[key]
            if count <= len(bucket):
                result.extend(generator.sample(bucket, count))
            else:
                result.extend(generator.sample(bucket, len(bucket)))
                result.extend(generator.choice(bucket) for _ in range(count - len(bucket)))
        generator.shuffle(result)
        return [dict(row) for row in result]

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for row in self._examples:
            state_class = str(row.get("state_class") or (row.get("labels") or {}).get("state_class") or "clean_successful")
            counts[state_class] += 1
        return dict(counts)


def balanced_replay_sample(
    examples: Iterable[Mapping[str, Any]],
    size: int,
    *,
    seed: int = 0,
    class_weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    return BalancedReplayBuffer(examples, class_weights=class_weights).sample(size, rng=random.Random(seed))
