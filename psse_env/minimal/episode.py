"""Run a policy through one episode of the minimal environment and score it.

The loop is the whole controller: read the observation, ask the policy for one
action, execute it.  A policy is anything with ``act(observation) -> action``;
it receives the observation the model would read and nothing else.  The truth
enters only after the last action, in ``score_env``.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from psse_env.minimal.env import MinimalDiagnosisEnv
from psse_env.minimal.scoring import RewardWeights, episode_reward, score_env

EPISODE_CONTRACT = "minimal_episode_v1"
#: The policy's prompt outgrew its input; the episode ends with nothing declared.
OUTCOME_PROMPT_LIMIT = "prompt_limit"


#: What a model policy reports about its last generation (``last_action_metrics``), kept per step.
POLICY_METRIC_KEYS = ("prompt_tokens", "original_prompt_tokens", "generated_tokens", "generation_seconds",
                      "hit_max_new_tokens", "prompt_limit_refused", "prompt_limit_reason", "memory_retries")


def _policy_metrics(policy: Any) -> dict[str, Any]:
    metrics = getattr(policy, "last_action_metrics", None)
    if not isinstance(metrics, Mapping):
        return {}
    kept = {key: metrics[key] for key in POLICY_METRIC_KEYS if key in metrics}
    return {"policy_metrics": kept} if kept else {}


def run_episode(env: MinimalDiagnosisEnv, root: Mapping[str, Any], policy: Any, *,
                weights: RewardWeights | None = None, keep_observations: bool = True) -> dict[str, Any]:
    """One episode of ``policy`` on ``root``: its steps, its score and its reward.

    A policy that raises ends its own turn with an unreadable call, which the
    environment records as a tool error and charges like any other action.
    """
    started = time.perf_counter()
    observation = env.reset(root)
    steps: list[dict[str, Any]] = []
    policy_seconds = 0.0
    while not env.done:
        asked = time.perf_counter()
        try:
            action = policy.act(observation)
            failure = None
        except Exception as exc:  # a policy failure is the policy's action, not the evaluator's crash
            action, failure = None, f"{type(exc).__name__}: {exc}"
            if type(exc).__name__ == "PromptLimitError":
                # The prompt no longer fits the model: nothing later in the episode can fit either.
                policy_seconds += time.perf_counter() - asked
                env.stop(OUTCOME_PROMPT_LIMIT)
                steps.append({"step": env.steps + 1, "action": None, "result": None, "policy_error": failure,
                              **_policy_metrics(policy)})
                break
        policy_seconds += time.perf_counter() - asked
        before = observation
        observation, result = env.step(action)
        step: dict[str, Any] = {"step": env.steps, "action": env.history()[-1]["action"], "result": result,
                                **_policy_metrics(policy)}
        if failure is not None:
            step["policy_error"] = failure
        if isinstance(action, str):
            # The model's words where they were not one valid call: what an "unparseable_action" was.
            step["raw_output"] = action[:600]
        if keep_observations:
            step["observation"] = before
        steps.append(step)
    score = score_env(root, env)
    return {
        "contract": EPISODE_CONTRACT,
        "scenario_id": env.scenario_id,
        "steps": steps,
        "outcome": env.outcome,
        "score": score,
        "reward": episode_reward(score, weights),
        "seconds": {"episode": round(time.perf_counter() - started, 4), "policy": round(policy_seconds, 4)},
    }


def compact(episode: Mapping[str, Any]) -> dict[str, Any]:
    """An episode without its observations: what a results file keeps."""
    return {**{key: value for key, value in episode.items() if key != "steps"},
            "steps": [{key: value for key, value in step.items() if key != "observation"} for step in episode["steps"]]}


class ScriptedPolicy:
    """A fixed list of actions, then ``finish`` with the given outcome: for replays and tests."""

    def __init__(self, actions: Any, *, outcome: str = "resolved") -> None:
        self._actions = list(actions)
        self._outcome = outcome

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        del observation
        if self._actions:
            return self._actions.pop(0)
        return {"tool": "finish", "arguments": {"outcome": self._outcome}}


__all__ = ["EPISODE_CONTRACT", "OUTCOME_PROMPT_LIMIT", "ScriptedPolicy", "compact", "run_episode"]
