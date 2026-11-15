"""Grouped on-policy rollouts of the minimal environment, in the record the RL update trains on.

One case is one group: K sampled episodes of the same scenario, each run by
``run_episode`` and scored by ``psse_env.minimal.scoring``.  The records are
the RL series' own (``psse_env.rl.rollout``: ``Trajectory``, ``StepRecord``,
``TrajectoryStore``), so the group advantages, the row builder and the update
of ``psse_env.rl`` take them as they are.  What differs from the old rollouts
is what an episode is and how it is rewarded:

* the episode is the minimal environment's, with the model choosing every
  action;
* the reward is ``scoring.episode_reward``: the repair outcome, less actual
  damage, a false claim of resolution and a small cost per action;
* there is no lifecycle decision to label (``Trajectory.lifecycle`` stays
  empty), because a correction is kept or undone by the model's own later
  action and credited by the episode's outcome.

Truth-built sequences are never mixed into a group: a group holds only what
the policy sampled.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from psse_env.minimal.env import MinimalDiagnosisEnv
from psse_env.minimal.episode import OUTCOME_PROMPT_LIMIT, run_episode
from psse_env.minimal.scoring import RewardWeights
from psse_env.rl.rollout import (
    RecordingPolicy,
    Trajectory,
    TrajectoryStore,
    pair_records_with_trace,
    sample_seed,
    scenario_family,
    scenario_root,
)

ROLLOUT_CONTRACT = "minimal_grouped_rollout_v1"


def _trace(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The episode's steps in the shape the RL records pair with: one entry per call the policy made."""
    trace: list[dict[str, Any]] = []
    for step in episode["steps"]:
        result = step.get("result") or {}
        refused = step.get("action") is None  # the policy raised before a call existed (its prompt did not fit)
        trace.append({
            "step": int(step["step"]),
            "action": copy.deepcopy(step["action"]) if not refused else {"tool": None, "arguments": {}},
            "execution_status": "success" if result.get("status") == "ok" else "failure",
            "error_code": OUTCOME_PROMPT_LIMIT if refused else result.get("error_code"),
            "policy_error": step.get("policy_error"),
        })
    return trace


def rollout_group(scenario: Mapping[str, Any], *, samples: int, base_seed: int,
                  env_factory: Callable[[], MinimalDiagnosisEnv], policy_factory: Callable[[int], Any],
                  weights: RewardWeights | None = None) -> list[Trajectory]:
    """K sampled episodes of one case.  ``policy_factory(seed)`` returns a policy whose decoding is seeded with ``seed``."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    root = scenario_root(scenario)
    family = scenario_family(scenario)
    trajectories: list[Trajectory] = []
    for index in range(int(samples)):
        seed = sample_seed(base_seed, root, index)
        recorder = RecordingPolicy(policy_factory(seed))
        episode = run_episode(env_factory(), scenario, recorder, weights=weights, keep_observations=False)
        trace = _trace(episode)
        steps, paired, note = pair_records_with_trace(recorder.records, trace)
        score = episode["score"]
        identity = recorder.research_policy_identity
        trajectories.append(Trajectory(
            root=root, family=family, sample_index=index, episode_seed=seed, steps=steps,
            reward={**episode["reward"], "calls": int(score["actions"])},
            evaluation={
                # The RL summaries read this key for an episode's success.
                "truth_audited_task_success": bool(score["success"]),
                "terminal_outcome": episode["outcome"],
                "trace": trace,
                "score": score,
                "seconds": episode["seconds"],
            },
            pairing_ok=paired, pairing_note=note,
            policy_identity=dict(identity) if isinstance(identity, Mapping) else {},
            lifecycle=[], contract=ROLLOUT_CONTRACT,
        ))
    return trajectories


def rollout_roots(scenarios: Sequence[Mapping[str, Any]], *, store_dir: str | Path, samples: int, base_seed: int,
                  env_factory: Callable[[], MinimalDiagnosisEnv], policy_factory: Callable[[int], Any],
                  weights: RewardWeights | None = None,
                  progress: Callable[[Mapping[str, Any]], None] | None = None) -> TrajectoryStore:
    """Every case's group, written one file per case; a case whose group is complete is not run again."""
    store = TrajectoryStore(store_dir)
    for number, scenario in enumerate(scenarios, start=1):
        root = scenario_root(scenario)
        if store.has_group(root, int(samples)):
            continue
        group = rollout_group(scenario, samples=samples, base_seed=base_seed, env_factory=env_factory,
                              policy_factory=policy_factory, weights=weights)
        store.write_group(root, group)
        if progress is not None:
            progress({"root": root, "done": number, "of": len(scenarios),
                      "rewards": [trajectory.total_reward for trajectory in group],
                      "solved": [bool(trajectory.evaluation["score"]["solved"]) for trajectory in group]})
    return store


__all__ = ["ROLLOUT_CONTRACT", "rollout_group", "rollout_roots"]
