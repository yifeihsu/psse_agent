"""Group-relative advantages over the K sampled episodes of one root.

GRPO: the advantage of an episode is its reward minus the group mean, divided
by the group standard deviation when ``scale`` is ``"group"``.  RLOO: the
baseline is the mean of the *other* episodes of the group (leave one out),
which is unbiased for the policy gradient.  Either way a group whose episodes
all scored the same carries no signal and is dropped before training, which
is also the measurement that says whether a draw is hard enough for RL to
learn from (plan section 6, step 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

ADVANTAGE_METHODS = ("grpo", "rloo")
SCALE_MODES = ("group", "none")


@dataclass(frozen=True)
class GroupAdvantages:
    root: str
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    mean: float
    std: float
    informative: bool


def group_advantages(
    rewards: Sequence[float],
    *,
    method: str = "grpo",
    scale: str = "group",
    epsilon: float = 1e-6,
    root: str = "",
) -> GroupAdvantages:
    if method not in ADVANTAGE_METHODS:
        raise ValueError(f"method must be one of {ADVANTAGE_METHODS}, got {method!r}")
    if scale not in SCALE_MODES:
        raise ValueError(f"scale must be one of {SCALE_MODES}, got {scale!r}")
    values = [float(value) for value in rewards]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("rewards must be finite")
    count = len(values)
    if count == 0:
        raise ValueError("a group needs at least one reward")
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    std = math.sqrt(variance)
    informative = count >= 2 and std > epsilon
    if not informative:
        return GroupAdvantages(root=root, rewards=tuple(values), advantages=tuple(0.0 for _ in values),
                               mean=mean, std=std, informative=False)
    if method == "grpo":
        centered = [value - mean for value in values]
    else:
        centered = [value - (sum(values) - value) / (count - 1) for value in values]
    if scale == "group":
        centered = [value / (std + epsilon) for value in centered]
    return GroupAdvantages(root=root, rewards=tuple(values), advantages=tuple(centered),
                           mean=mean, std=std, informative=True)


def summarize_groups(groups: Mapping[str, GroupAdvantages]) -> dict[str, float | int]:
    """How much signal the sampled batch carries, for the iteration record."""
    total = len(groups)
    informative = sum(1 for group in groups.values() if group.informative)
    means = [group.mean for group in groups.values()]
    return {
        "groups": total,
        "informative_groups": informative,
        "informative_fraction": (informative / total) if total else 0.0,
        "mean_reward": (sum(means) / total) if total else 0.0,
        "min_group_mean": min(means) if means else 0.0,
        "max_group_mean": max(means) if means else 0.0,
    }


__all__ = ["ADVANTAGE_METHODS", "SCALE_MODES", "GroupAdvantages", "group_advantages", "summarize_groups"]
