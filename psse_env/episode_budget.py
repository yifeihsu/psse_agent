"""Shared episode action horizon for collection, evaluation, and environments."""
from __future__ import annotations

from numbers import Integral
from typing import Any


DEFAULT_EPISODE_ACTION_LIMIT = 40


def validate_episode_action_limit(max_steps: int) -> int:
    """Validate an explicit horizon without rounding or accepting booleans.

    Small positive limits remain useful for tests and deliberate experiments;
    forty is the common default, not a restriction on explicit overrides.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 1:
        raise ValueError("episode action limit must be a positive integer")
    return int(max_steps)


def bind_env_action_limit(env: Any, max_steps: int) -> int:
    """Bind the runner's horizon to a budget-aware environment before reset.

    Runners must call this before every episode reset so the policy observes
    the same action horizon that the outer loop enforces. Minimal fake or
    external environments without ``max_steps`` are left untouched; their
    runner still enforces the returned limit. An environment that advertises
    this budget but cannot accept it fails explicitly instead of silently
    exposing a contradictory remaining budget.
    """
    limit = validate_episode_action_limit(max_steps)
    if hasattr(env, "max_steps"):
        if getattr(env, "max_steps") != limit:
            try:
                env.max_steps = limit
            except (AttributeError, TypeError) as exc:
                raise ValueError("environment advertises max_steps but its action limit is not writable") from exc
        if getattr(env, "max_steps") != limit:
            raise ValueError("environment did not retain the runner's episode action limit")
    return limit


__all__ = ["DEFAULT_EPISODE_ACTION_LIMIT", "validate_episode_action_limit", "bind_env_action_limit"]
