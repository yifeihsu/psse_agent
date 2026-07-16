from __future__ import annotations

import random
from typing import Iterable

from .error_injectors import InjectedAction


def sample_branches(
    branches: Iterable[InjectedAction],
    *,
    max_branches: int,
    seed: int = 0,
) -> list[InjectedAction]:
    candidates = list(branches)
    if max_branches < 0:
        raise ValueError("max_branches must be non-negative")
    if len(candidates) <= max_branches:
        return candidates
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(candidates)), max_branches))
    return [candidates[index] for index in selected_indices]
