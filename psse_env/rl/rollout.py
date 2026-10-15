"""Grouped on-policy rollouts through the closed-loop evaluator.

One root is one group: K sampled episodes of the same scenario, each run by
``evaluate_rollout_suites`` with its own seed so the audit, the counters and
the trace are the ones every cell reports.  A recording wrapper around the
policy keeps, per step, the model observation the policy rendered and the
exact completion tokens it sampled; the trajectory pairs them with the
evaluator's trace and the reward of :mod:`psse_env.rl.reward`.

The store is one JSONL file per root under ``groups/``, so a preempted
rollout resumes at the first root without a complete group.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from psse_env.rl.reward import RewardBreakdown, RewardWeights, episode_reward

ROLLOUT_CONTRACT = "rl_grouped_rollout_v1"


@dataclass
class StepRecord:
    step: int
    model_observation: dict[str, Any]
    prompt_tokens: int
    original_prompt_tokens: int
    completion_token_ids: list[int]
    text: str
    action: dict[str, Any]
    tool: str
    execution_status: str | None
    error_code: str | None
    #: Probability of each sampled token under the distribution it was drawn
    #: from (empty for greedy or expert steps).
    token_probabilities: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def min_token_probability(self) -> float | None:
        return min(self.token_probabilities) if self.token_probabilities else None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StepRecord":
        return cls(
            step=int(value["step"]),
            model_observation=dict(value.get("model_observation") or {}),
            prompt_tokens=int(value.get("prompt_tokens") or 0),
            original_prompt_tokens=int(value.get("original_prompt_tokens") or 0),
            completion_token_ids=[int(item) for item in value.get("completion_token_ids") or []],
            text=str(value.get("text") or ""),
            action=dict(value.get("action") or {}),
            tool=str(value.get("tool") or ""),
            execution_status=value.get("execution_status"),
            error_code=value.get("error_code"),
            token_probabilities=[float(item) for item in value.get("token_probabilities") or []],
        )


@dataclass
class Trajectory:
    root: str
    family: str
    sample_index: int
    episode_seed: int
    steps: list[StepRecord]
    reward: dict[str, Any]
    evaluation: dict[str, Any]
    pairing_ok: bool
    pairing_note: str | None = None
    policy_identity: dict[str, Any] = field(default_factory=dict)
    contract: str = ROLLOUT_CONTRACT

    @property
    def total_reward(self) -> float:
        return float(self.reward["total"])

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.as_dict() for step in self.steps]
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Trajectory":
        return cls(
            root=str(value["root"]),
            family=str(value.get("family") or "unknown"),
            sample_index=int(value.get("sample_index") or 0),
            episode_seed=int(value.get("episode_seed") or 0),
            steps=[StepRecord.from_dict(step) for step in value.get("steps") or []],
            reward=dict(value.get("reward") or {}),
            evaluation=dict(value.get("evaluation") or {}),
            pairing_ok=bool(value.get("pairing_ok")),
            pairing_note=value.get("pairing_note"),
            policy_identity=dict(value.get("policy_identity") or {}),
            contract=str(value.get("contract") or ROLLOUT_CONTRACT),
        )


class RecordingPolicy:
    """Forward ``act`` to a policy and keep what it generated at every step.

    The wrapped policy exposes ``last_generation`` (the research policy
    does); a policy without it still records the action it returned, which
    is enough for the reward but not for training rows.
    """

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self.records: list[dict[str, Any]] = []

    @property
    def last_action_metrics(self) -> Any:
        return getattr(self._policy, "last_action_metrics", None)

    @property
    def research_policy_identity(self) -> Any:
        return getattr(self._policy, "research_policy_identity", None)

    def act(self, observation: Mapping[str, Any]) -> Any:
        action = self._policy.act(observation)
        generation = getattr(self._policy, "last_generation", None)
        record: dict[str, Any] = {"action": copy.deepcopy(action)}
        if isinstance(generation, Mapping):
            record.update(copy.deepcopy(dict(generation)))
        self.records.append(record)
        return action


def sample_seed(base_seed: int, root: str, sample_index: int) -> int:
    digest = hashlib.sha256(
        json.dumps([int(base_seed), str(root), int(sample_index)], separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def scenario_root(scenario: Mapping[str, Any]) -> str:
    execution = scenario.get("execution") if isinstance(scenario.get("execution"), Mapping) else {}
    grouping = scenario.get("grouping") if isinstance(scenario.get("grouping"), Mapping) else {}
    for value in (scenario.get("scenario_id"), grouping.get("scenario_id"), execution.get("scenario_id"), execution.get("id")):
        if value is not None:
            return str(value)
    raise ValueError("scenario carries no scenario_id")


def scenario_family(scenario: Mapping[str, Any]) -> str:
    grouping = scenario.get("grouping") if isinstance(scenario.get("grouping"), Mapping) else {}
    return str(grouping.get("scenario_family") or scenario.get("scenario_family") or "unknown")


def _policy_trace_steps(trace: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [step for step in trace if isinstance(step, Mapping) and not step.get("intervention")]


def pair_records_with_trace(
    records: Sequence[Mapping[str, Any]], trace: Sequence[Mapping[str, Any]]
) -> tuple[list[StepRecord], bool, str | None]:
    """Pair the policy's generation records with the evaluator's policy steps, in order."""

    policy_steps = _policy_trace_steps(trace)
    if len(policy_steps) != len(records):
        note = f"{len(records)} generation records against {len(policy_steps)} policy trace steps"
        return [], False, note
    steps: list[StepRecord] = []
    for index, (record, trace_step) in enumerate(zip(records, policy_steps)):
        action = trace_step.get("action") if isinstance(trace_step.get("action"), Mapping) else record.get("action") or {}
        steps.append(
            StepRecord(
                step=int(trace_step.get("step", index)),
                model_observation=dict(record.get("model_observation") or {}),
                prompt_tokens=int(record.get("prompt_tokens") or 0),
                original_prompt_tokens=int(record.get("original_prompt_tokens") or 0),
                completion_token_ids=[int(item) for item in record.get("completion_token_ids") or []],
                text=str(record.get("text") or ""),
                action=copy.deepcopy(dict(action)),
                tool=str(action.get("tool") if isinstance(action, Mapping) else ""),
                execution_status=trace_step.get("execution_status"),
                error_code=trace_step.get("error_code"),
                token_probabilities=[float(item) for item in record.get("completion_token_probabilities") or []],
            )
        )
    missing = sum(1 for step in steps if not step.model_observation or not step.completion_token_ids)
    if missing:
        return steps, False, f"{missing} steps without a recorded generation"
    return steps, True, None


def distinct_trajectories(group: Sequence[Trajectory]) -> int:
    """How many different completion-token sequences a group of sampled episodes produced."""
    return len({tuple(tuple(step.completion_token_ids) for step in trajectory.steps) for trajectory in group})


def reward_spread(group: Sequence[Trajectory]) -> float:
    rewards = [trajectory.total_reward for trajectory in group]
    return (max(rewards) - min(rewards)) if rewards else 0.0


def store_summary(store: "TrajectoryStore") -> dict[str, Any]:
    """What a whole iteration's rollouts contain: signal, success, cost and how peaked the sampling was."""
    groups = informative = distinct_groups = episodes = successes = calls = unpaired = 0
    steps = peaked_steps = 0
    for _root, group in store.iter_groups():
        groups += 1
        informative += int(reward_spread(group) > 1e-9)
        distinct_groups += int(distinct_trajectories(group) > 1)
        for trajectory in group:
            episodes += 1
            successes += int(bool(trajectory.evaluation.get("truth_audited_task_success")))
            calls += int(trajectory.reward.get("calls") or 0)
            unpaired += int(not trajectory.pairing_ok)
            for step in trajectory.steps:
                if step.token_probabilities:
                    steps += 1
                    peaked_steps += int(min(step.token_probabilities) >= 0.99)
    return {
        "groups": groups,
        "groups_with_reward_spread": informative,
        "groups_with_distinct_samples": distinct_groups,
        "episodes": episodes,
        "success_rate": (successes / episodes) if episodes else None,
        "mean_calls": (calls / episodes) if episodes else None,
        "unpaired_trajectories": unpaired,
        "sampled_steps_with_probabilities": steps,
        "fraction_of_steps_with_every_token_above_0_99": (peaked_steps / steps) if steps else None,
    }


def rollout_group(
    scenario: Mapping[str, Any],
    *,
    samples: int,
    base_seed: int,
    env_factory: Callable[..., Any],
    policy_factory: Callable[[int], Any],
    max_steps: int,
    case_loader: Callable[[Any], Any] | None,
    weights: RewardWeights,
    expert_reference_calls: int | None = None,
    evaluator: Callable[..., Any] | None = None,
) -> list[Trajectory]:
    """K sampled episodes of one root, each audited by the evaluator.

    ``policy_factory(seed)`` returns a policy whose decoding is seeded with
    ``seed``; the wrapper records its generations.  ``evaluator`` defaults to
    :func:`psse_env.dagger.evaluator.evaluate_rollout_suites`.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if evaluator is None:
        from psse_env.dagger.evaluator import evaluate_rollout_suites

        evaluator = evaluate_rollout_suites
    from scripts.run_dagger_research import PAIRED_EVALUATION_CONTRACT

    root = scenario_root(scenario)
    family = scenario_family(scenario)
    trajectories: list[Trajectory] = []
    for index in range(int(samples)):
        seed = sample_seed(base_seed, root, index)
        recorder = RecordingPolicy(policy_factory(seed))
        result = evaluator(
            {"standard_success": [copy.deepcopy(dict(scenario))]},
            env_factory=env_factory,
            policy_factory=lambda **_kwargs: recorder,
            max_steps=int(max_steps),
            seed=seed,
            case_loader=case_loader,
            **PAIRED_EVALUATION_CONTRACT,
        )
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        episodes = list(payload["suite_metrics"]["episodes"])
        if len(episodes) != 1:
            raise RuntimeError(f"expected one evaluated episode for {root}, got {len(episodes)}")
        episode = dict(episodes[0])
        if not episode.get("family") or episode.get("family") == "None":
            episode["family"] = family
        reward = episode_reward(episode, family=family, expert_reference_calls=expert_reference_calls, weights=weights)
        steps, ok, note = pair_records_with_trace(recorder.records, episode.get("trace") or [])
        identity = recorder.research_policy_identity
        trajectories.append(
            Trajectory(
                root=root,
                family=family,
                sample_index=index,
                episode_seed=seed,
                steps=steps,
                reward=reward.as_dict(),
                evaluation=episode,
                pairing_ok=ok,
                pairing_note=note,
                policy_identity=dict(identity) if isinstance(identity, Mapping) else {},
            )
        )
    return trajectories


class TrajectoryStore:
    """One JSONL file per root group; complete groups are never rewritten."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.groups_dir = self.directory / "groups"
        self.groups_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, root: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in root)
        return self.groups_dir / f"{safe}.jsonl"

    def has_group(self, root: str, samples: int) -> bool:
        path = self._path(root)
        if not path.is_file():
            return False
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    count += 1
        return count >= int(samples)

    def write_group(self, root: str, trajectories: Sequence[Trajectory]) -> Path:
        path = self._path(root)
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for trajectory in trajectories:
                stream.write(json.dumps(trajectory.as_dict(), sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
        return path

    def read_group(self, root: str) -> list[Trajectory]:
        path = self._path(root)
        trajectories: list[Trajectory] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    trajectories.append(Trajectory.from_dict(json.loads(line)))
        return trajectories

    def roots(self) -> list[str]:
        roots: list[str] = []
        for path in sorted(self.groups_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        roots.append(str(json.loads(line)["root"]))
                        break
        return roots

    def iter_groups(self) -> Iterator[tuple[str, list[Trajectory]]]:
        for root in self.roots():
            yield root, self.read_group(root)


def rollout_roots(
    scenarios: Iterable[Mapping[str, Any]],
    *,
    store: TrajectoryStore,
    samples: int,
    base_seed: int,
    env_factory: Callable[..., Any],
    policy_factory: Callable[[int], Any],
    max_steps: int,
    case_loader: Callable[[Any], Any] | None,
    weights: RewardWeights,
    expert_reference: Mapping[str, int] | None = None,
    evaluator: Callable[..., Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Roll out every root without a complete group; returns the pass's summary."""

    started = time.perf_counter()
    completed = skipped = 0
    rewards: list[float] = []
    for scenario in scenarios:
        root = scenario_root(scenario)
        if store.has_group(root, samples):
            skipped += 1
            continue
        reference = None if expert_reference is None else expert_reference.get(root)
        group = rollout_group(
            scenario,
            samples=samples,
            base_seed=base_seed,
            env_factory=env_factory,
            policy_factory=policy_factory,
            max_steps=max_steps,
            case_loader=case_loader,
            weights=weights,
            expert_reference_calls=reference,
            evaluator=evaluator,
        )
        store.write_group(root, group)
        completed += 1
        rewards.extend(trajectory.total_reward for trajectory in group)
        if progress is not None:
            progress({
                "root": root,
                "family": scenario_family(scenario),
                "rewards": [trajectory.total_reward for trajectory in group],
                "successes": sum(1 for trajectory in group if trajectory.evaluation.get("truth_audited_task_success")),
                "distinct_trajectories": distinct_trajectories(group),
                "pairing_ok": all(trajectory.pairing_ok for trajectory in group),
                "completed": completed,
                "elapsed_seconds": time.perf_counter() - started,
            })
    return {
        "contract": ROLLOUT_CONTRACT,
        "completed_groups": completed,
        "skipped_complete_groups": skipped,
        "samples_per_root": int(samples),
        "mean_reward_new": (sum(rewards) / len(rewards)) if rewards else None,
        "seconds": time.perf_counter() - started,
    }


__all__ = [
    "ROLLOUT_CONTRACT",
    "RecordingPolicy",
    "StepRecord",
    "Trajectory",
    "TrajectoryStore",
    "distinct_trajectories",
    "pair_records_with_trace",
    "reward_spread",
    "store_summary",
    "rollout_group",
    "rollout_roots",
    "sample_seed",
    "scenario_family",
    "scenario_root",
]
