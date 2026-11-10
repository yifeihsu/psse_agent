"""Branching an episode at a lifecycle decision (workplan of 2026-11-10, stages 2 and 4).

At an accepted candidate the policy may commit it or roll it back (decline
it); the environment allows both (``lifecycle_choice``).  What each choice
leads to is measured by running the episode again from that state: the
recorded actions before the decision are replayed through the real
environment, the choice is forced, and a policy continues to the end of the
episode under the budget the episode had left.

* :class:`DecisionState` is one such state, read from a stored trajectory:
  the actions before it, the observations the policy had at each of them,
  and the audit's label of the decision (for stratifying and reporting only).
* :class:`ForkPolicy` replays, forces and hands over.  At every replayed
  step it checks that the policy would be shown what it was shown in the
  recorded episode (the model-visible view, which names states by alias), so
  a branch whose state has drifted from the recorded one is marked and never
  used.  It binds the recorded actions' state references to the running
  episode's states through those aliases.
* :func:`run_branch` runs one branch through the closed-loop evaluator and
  scores it with the episode reward, so a branch's return is the return an
  ordinary episode would have had: every action of the prefix counts against
  the budget and the cost.

The forced step is an intervention of the harness, not a choice of the
policy: a branch records where it is (``forced_position``) and the steps
after it are never training rows.  Only :mod:`psse_env.rl.decision_credit`
reads branch returns, as values of the two choices at a state the policy
itself reached and decided at.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from psse_env.actions import COMMIT_STATE, ROLLBACK_STATE
from psse_env.dagger.observation_modes import (
    BOUNDED_HISTORY,
    COMPLETE_HISTORY,
    OBSERVABLE_HISTORY_KEY,
    observable_history_from_trace,
    validate_observation_mode,
    with_observable_history,
)
from psse_env.rl.lifecycle import (
    CORRECTION_TOOLS,
    lifecycle_decisions,
    policy_trace_steps,
    scenario_truth,
    summarize_decisions,
    target_is_true,
)
from psse_env.rl.reward import RewardWeights, episode_reward
from psse_env.rl.rollout import RecordingPolicy, Trajectory, scenario_family, scenario_root

FORK_CONTRACT = "rl_decision_fork_v1"
CHOICES = (COMMIT_STATE, ROLLBACK_STATE)
STATE_REFERENCE_KEYS = ("state_id", "candidate_state_id", "parent_state_id")
ACCEPTED = frozenset({"ACCEPT_FINAL", "ACCEPT_PARTIAL"})


def other_choice(tool: str) -> str:
    if tool not in CHOICES:
        raise ValueError(f"a lifecycle choice is one of {CHOICES}, got {tool!r}")
    return ROLLBACK_STATE if tool == COMMIT_STATE else COMMIT_STATE


def _bounded_part(observation: Mapping[str, Any]) -> dict[str, Any]:
    """An observation without the complete-history record: what both observation modes share."""

    return {key: value for key, value in observation.items() if key != OBSERVABLE_HISTORY_KEY}


def view_signature(observation: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """A digest of what the policy is shown of a state, and the aliases the view names its states by.

    The digest is of the bounded model-visible view, in which controller
    state ids are replaced by local aliases, so two episodes that reached
    the same state under different ids give the same digest, and it is the
    same whichever observation mode the episode runs in (the complete record
    adds earlier transitions, which a matching prefix already implies).

    The observation's keys are sorted first.  The bounded view numbers its
    aliases, and cuts an over-long record, in the order a mapping's keys
    come in, and a record read back from a store has them sorted while a
    running environment has them in its own order: without the sort the
    same state gives two digests (measured 2026-11-10: 142 of 395 steps of
    28 teacher episodes).
    """

    from psse_env.dagger.dataset_builder import _key_sorted, prepare_model_policy_observation

    view, control = prepare_model_policy_observation(_key_sorted(_bounded_part(observation)))
    digest = hashlib.sha256(json.dumps(view, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest, {str(alias): str(identifier) for alias, identifier in control["state_aliases"].items()}


def rebind_action(action: Mapping[str, Any], recorded_aliases: Mapping[str, str], live_aliases: Mapping[str, str]) -> dict[str, Any]:
    """A recorded action with its state references pointed at the running episode's states.

    A reference is translated through its alias: the recorded id's alias in
    the recorded view, then that alias's id in the running view.  A
    reference the recorded view has no alias for is left as it is.
    """

    bound = copy.deepcopy(dict(action))
    arguments = dict(bound.get("arguments") or {})
    alias_of = {identifier: alias for alias, identifier in recorded_aliases.items()}
    for key in STATE_REFERENCE_KEYS:
        value = arguments.get(key)
        if value is None:
            continue
        alias = alias_of.get(str(value))
        if alias is not None and alias in live_aliases:
            arguments[key] = live_aliases[alias]
    bound["arguments"] = arguments
    return bound


@dataclass
class DecisionState:
    """A lifecycle decision of a stored episode, with what is needed to reach its state again."""

    root: str
    sample_index: int
    #: The decision's position among the episode's policy steps (the trajectory's step records).
    position: int
    #: The tool the policy sampled there.
    sampled_tool: str
    #: The audit's record of the decision (:func:`psse_env.rl.lifecycle.lifecycle_decisions`); never shown to a policy.
    decision: dict[str, Any]
    #: The policy's actions before the decision, as the controller executed them.
    prefix_actions: list[dict[str, Any]]
    #: The observation the policy had at each prefix step and at the decision (``position + 1`` entries),
    #: as the trace stores them.
    observations: list[dict[str, Any]]
    #: Interventions of the evaluation contract that ran before the policy's first step.
    interventions: int = 0
    #: The digest of the prompt the policy decoded the decision from (``None`` for records without one).
    prompt_digest: str | None = None
    #: The episode's own return and outcome, a sample of what the sampled choice leads to.
    source_reward: float | None = None
    source_success: bool | None = None
    #: The index of the decision among the episode's decisions on accepted candidates, and the declines before it.
    ordinal: int = 0
    earlier_declines: int = 0
    #: Every transition of the episode before the decision, interventions included, as the evaluator's
    #: history held them: what a complete-history observation of the state carries.
    transitions: list[dict[str, Any]] = field(default_factory=list)
    #: The text the policy generated at the decision (the sampled tool call), when the episode recorded it.
    sampled_text: str | None = None
    #: The model-visible view the policy decoded the decision from, as the policy itself recorded it.
    sampled_view: dict[str, Any] | None = None
    contract: str = FORK_CONTRACT

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.root, int(self.sample_index), int(self.position))

    @property
    def name(self) -> str:
        return f"{''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in self.root)}__s{int(self.sample_index)}__p{int(self.position)}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionState":
        return cls(
            root=str(value["root"]), sample_index=int(value["sample_index"]), position=int(value["position"]),
            sampled_tool=str(value["sampled_tool"]), decision=dict(value.get("decision") or {}),
            prefix_actions=[dict(item) for item in value.get("prefix_actions") or []],
            observations=[dict(item) for item in value.get("observations") or []],
            interventions=int(value.get("interventions") or 0),
            prompt_digest=(str(value["prompt_digest"]) if value.get("prompt_digest") else None),
            source_reward=(float(value["source_reward"]) if value.get("source_reward") is not None else None),
            source_success=(bool(value["source_success"]) if value.get("source_success") is not None else None),
            ordinal=int(value.get("ordinal") or 0), earlier_declines=int(value.get("earlier_declines") or 0),
            transitions=[dict(item) for item in value.get("transitions") or []],
            sampled_text=(str(value["sampled_text"]) if value.get("sampled_text") else None),
            sampled_view=(dict(value["sampled_view"]) if isinstance(value.get("sampled_view"), Mapping) else None),
            contract=str(value.get("contract") or FORK_CONTRACT),
        )

    def observation(self, observation_mode: str = BOUNDED_HISTORY) -> dict[str, Any]:
        """The observation the policy has at the decision under ``observation_mode``.

        The two modes show the same state; the complete one adds the
        transitions before it, which the state keeps.
        """

        bounded = _bounded_part(self.observations[-1])
        if validate_observation_mode(observation_mode) == COMPLETE_HISTORY:
            return with_observable_history(bounded, self.transitions)
        return copy.deepcopy(bounded)


def decision_states(trajectory: Trajectory) -> list[DecisionState]:
    """Every labelled decision of a stored episode on an accepted candidate, as a state a branch can start from.

    A decision is left out when the episode's trace does not hold the
    observation of every step up to it (an evaluation stored without its
    trace), or when the episode began with interventions other than the
    evaluator's own (which a replay through the same contract repeats).
    """

    trace = trajectory.evaluation.get("trace") or []
    steps = policy_trace_steps(trace)
    interventions = sum(1 for row in trace if isinstance(row, Mapping) and row.get("intervention"))
    trace_index = [index for index, row in enumerate(trace) if isinstance(row, Mapping) and not row.get("intervention")]
    states: list[DecisionState] = []
    declines = 0
    for ordinal, decision in enumerate(trajectory.lifecycle):
        position = decision.get("position")
        usable = (
            decision.get("agrees") is not None
            and isinstance(position, int) and not isinstance(position, bool)
            and 0 <= position < len(steps)
            and decision.get("tool") in CHOICES
            and decision.get("verdict") in ACCEPTED
        )
        if usable:
            observations = [row.get("policy_observation") for row in steps[: position + 1]]
            action = steps[position].get("action") if isinstance(steps[position].get("action"), Mapping) else {}
            if all(isinstance(item, Mapping) for item in observations) and str(action.get("tool")) == decision.get("tool"):
                record = trajectory.steps[position] if position < len(trajectory.steps) else None
                states.append(DecisionState(
                    root=trajectory.root, sample_index=int(trajectory.sample_index), position=int(position),
                    sampled_tool=str(decision["tool"]), decision=copy.deepcopy(dict(decision)),
                    prefix_actions=[copy.deepcopy(dict(row.get("action") or {})) for row in steps[:position]],
                    observations=[copy.deepcopy(dict(item)) for item in observations],
                    interventions=interventions,
                    prompt_digest=(record.prompt_digest if record is not None else None),
                    source_reward=float(trajectory.total_reward) if trajectory.reward.get("total") is not None else None,
                    source_success=bool(trajectory.evaluation.get("truth_audited_task_success")),
                    ordinal=ordinal, earlier_declines=declines,
                    transitions=observable_history_from_trace(trace, trace_index[position]),
                    sampled_text=(record.text if record is not None and record.text else None),
                    sampled_view=(copy.deepcopy(record.model_observation) if record is not None and record.model_observation else None),
                ))
        declines += int(bool(decision.get("declined")))
    return states


class ForkPolicy:
    """Replay a recorded prefix, force a lifecycle choice, then let ``continuation`` act.

    ``continuation`` is a policy (``act(observation)``); what it generates
    after the forced step is kept as a :class:`RecordingPolicy` keeps it.
    ``mismatches`` lists the replayed steps at which the policy would have
    been shown something else than in the recorded episode.
    """

    def __init__(self, state: DecisionState, forced_tool: str, continuation: Any) -> None:
        if forced_tool not in CHOICES:
            raise ValueError(f"the forced choice is one of {CHOICES}, got {forced_tool!r}")
        if len(state.observations) != state.position + 1 or len(state.prefix_actions) != state.position:
            raise ValueError("the decision state does not hold an observation for every step up to the decision")
        self.state = state
        self.forced_tool = forced_tool
        self._recorder = RecordingPolicy(continuation)
        self._recorded = [view_signature(observation) for observation in state.observations]
        self.calls = 0
        self.mismatches: list[dict[str, Any]] = []
        self.fork_observation: dict[str, Any] | None = None

    @property
    def records(self) -> list[dict[str, Any]]:
        """The continuation's generation records, one per step after the forced one."""
        return self._recorder.records

    @property
    def last_action_metrics(self) -> Any:
        return self._recorder.last_action_metrics

    @property
    def research_policy_identity(self) -> Any:
        return self._recorder.research_policy_identity

    @property
    def release_policy_identity(self) -> Any:
        return getattr(self._recorder._policy, "release_policy_identity", None)

    def act(self, observation: Mapping[str, Any]) -> Any:
        index = self.calls
        self.calls += 1
        if index > self.state.position:
            return self._recorder.act(observation)
        digest, live = view_signature(observation)
        recorded_digest, recorded = self._recorded[index]
        if digest != recorded_digest:
            self.mismatches.append({"position": index, "recorded": recorded_digest, "replayed": digest})
        if index < self.state.position:
            return rebind_action(self.state.prefix_actions[index], recorded, live)
        self.fork_observation = copy.deepcopy(dict(observation))
        candidate = observation.get("candidate_state_id")
        return {"tool": self.forced_tool, "arguments": {"candidate_state_id": candidate}}


class CaptureEnded(RuntimeError):
    """Raised by :class:`CapturePolicy` once it holds the decision's observation: the episode has nothing more to give."""


class CapturePolicy:
    """Replay a recorded prefix and keep the observation the policy is shown at the decision, then stop acting.

    Used to read what a policy would see at a recorded state under an
    observation mode the episode was not recorded in: the running
    environment hands over its own observation, so the view rendered from it
    is the one a policy in that mode decodes from, key order and all.
    """

    def __init__(self, state: DecisionState) -> None:
        self.state = state
        self._recorded = [view_signature(observation) for observation in state.observations]
        self.calls = 0
        self.mismatches: list[dict[str, Any]] = []
        self.observation: dict[str, Any] | None = None

    def act(self, observation: Mapping[str, Any]) -> Any:
        index = self.calls
        self.calls += 1
        if index > self.state.position:
            raise CaptureEnded("the capture ended at the decision")
        digest, live = view_signature(observation)
        recorded_digest, recorded = self._recorded[index]
        if digest != recorded_digest:
            self.mismatches.append({"position": index, "recorded": recorded_digest, "replayed": digest})
        if index < self.state.position:
            return rebind_action(self.state.prefix_actions[index], recorded, live)
        self.observation = copy.deepcopy(dict(observation))
        raise CaptureEnded("the capture ended at the decision")


def capture_fork_view(
    scenario: Mapping[str, Any],
    state: DecisionState,
    *,
    observation_mode: str,
    env_factory: Callable[..., Any],
    max_steps: int,
    case_loader: Callable[[Any], Any] | None,
    evaluator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """The model-visible view of a decision state under ``observation_mode``, from a replay through the environment.

    ``matched`` is whether the replay showed the policy what the recorded
    episode showed it at every step up to the decision.  In the mode the
    episode was recorded in, ``equals_sampled_view`` says whether the view
    is the very one the policy recorded when it decided.
    """

    from psse_env.dagger.dataset_builder import prepare_model_policy_observation

    mode = validate_observation_mode(observation_mode)
    if evaluator is None:
        from psse_env.dagger.evaluator import evaluate_rollout_suites

        evaluator = evaluate_rollout_suites
    from scripts.run_dagger_research import PAIRED_EVALUATION_CONTRACT

    policy = CapturePolicy(state)
    options = {} if mode == BOUNDED_HISTORY else {"observation_mode": mode}
    evaluator({"standard_success": [copy.deepcopy(dict(scenario))]}, env_factory=env_factory, policy_factory=lambda **_kwargs: policy,
              max_steps=int(max_steps), seed=0, case_loader=case_loader, **PAIRED_EVALUATION_CONTRACT, **options)
    if policy.observation is None:
        return {"observation_mode": mode, "matched": False, "reached": False, "mismatches": list(policy.mismatches), "view": None,
                "equals_sampled_view": None}
    view, _control = prepare_model_policy_observation(policy.observation)
    return {
        "observation_mode": mode, "matched": not policy.mismatches, "reached": True, "mismatches": list(policy.mismatches),
        "view": view,
        "equals_sampled_view": None if state.sampled_view is None else view == state.sampled_view,
    }


def recovery_chain(trace: Sequence[Mapping[str, Any]], *, forced_position: int, truth: Mapping[str, Any]) -> dict[str, Any]:
    """What followed a forced choice: the next correction, whether a true target was on offer, and how it ended.

    Read from the branch's trace against the scenario's truth (an audit
    reading, after the episode).  ``productive_target_offered`` is whether
    any context the policy fetched after the fork advertised a correction on
    a truly faulted component; ``next_correction`` is the first successful
    correction after it, with the audit's label of its target, the
    verifier's verdict on its candidate and what the policy did with it.
    """

    steps = policy_trace_steps(trace)
    after = steps[forced_position + 1:]
    offered = False
    next_correction: dict[str, Any] | None = None
    for index, step in enumerate(after):
        action = step.get("action") if isinstance(step.get("action"), Mapping) else {}
        tool = str(action.get("tool") or "")
        output = step.get("policy_tool_output") if isinstance(step.get("policy_tool_output"), Mapping) else {}
        metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
        if next_correction is None:
            for supported in metrics.get("supported_corrections") or []:
                if not isinstance(supported, Mapping):
                    continue
                arguments = {key: value for key, value in (supported.get("arguments") or {}).items() if key != "state_id"}
                if target_is_true(str(supported.get("tool")), arguments, truth) is True:
                    offered = True
        if tool in CORRECTION_TOOLS and step.get("execution_status") == "success" and next_correction is None:
            arguments = {key: value for key, value in (action.get("arguments") or {}).items() if key != "state_id"}
            next_correction = {"position": forced_position + 1 + index, "tool": tool, "arguments": arguments,
                               "target_is_true": target_is_true(tool, arguments, truth), "verdict": None, "lifecycle": None}
        elif next_correction is not None and next_correction["lifecycle"] is None and tool in CHOICES \
                and step.get("execution_status") == "success":
            next_correction["verdict"] = step.get("candidate_disposition_offline")
            next_correction["lifecycle"] = tool
    return {
        "actions_after_fork": len(after),
        "productive_target_offered": offered,
        "next_correction": next_correction,
        "next_target_is_true": None if next_correction is None else next_correction["target_is_true"],
        "next_correction_accepted": None if next_correction is None else (next_correction["verdict"] in ACCEPTED),
        "next_correction_committed": None if next_correction is None else (next_correction["lifecycle"] == COMMIT_STATE),
    }


def branch_seed(base_seed: int, state: DecisionState, continuation: int) -> int:
    """The seed of a branch's continuation: one per decision state and continuation index, shared by the two choices
    and by the observation modes, so the branches of a state differ by the choice and the mode alone as far as the
    sampler's noise is concerned."""

    from psse_env.rl.rollout import sample_seed

    return sample_seed(int(base_seed), f"{state.root}:{int(state.sample_index)}:{int(state.position)}", int(continuation))


def run_branch(
    scenario: Mapping[str, Any],
    state: DecisionState,
    forced_tool: str,
    *,
    continuation: Any,
    continuation_index: int,
    seed: int,
    env_factory: Callable[..., Any],
    max_steps: int,
    case_loader: Callable[[Any], Any] | None,
    weights: RewardWeights,
    observation_mode: str = BOUNDED_HISTORY,
    evaluator: Callable[..., Any] | None = None,
    keep_trace: bool = False,
) -> dict[str, Any]:
    """One branch of a decision state: the prefix replayed, ``forced_tool`` forced, ``continuation`` to the end.

    The record is valid when the replay showed the policy what the recorded
    episode showed it at every step and the environment executed the forced
    choice; only valid branches carry a return anyone should use.
    """

    if scenario_root(scenario) != state.root:
        raise ValueError(f"the scenario is root {scenario_root(scenario)}, the decision state is of {state.root}")
    mode = validate_observation_mode(observation_mode)
    if evaluator is None:
        from psse_env.dagger.evaluator import evaluate_rollout_suites

        evaluator = evaluate_rollout_suites
    from scripts.run_dagger_research import PAIRED_EVALUATION_CONTRACT

    policy = ForkPolicy(state, forced_tool, continuation)
    options = {} if mode == BOUNDED_HISTORY else {"observation_mode": mode}
    result = evaluator(
        {"standard_success": [copy.deepcopy(dict(scenario))]},
        env_factory=env_factory,
        policy_factory=lambda **_kwargs: policy,
        max_steps=int(max_steps),
        seed=int(seed),
        case_loader=case_loader,
        **PAIRED_EVALUATION_CONTRACT,
        **options,
    )
    payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    episodes = list(payload["suite_metrics"]["episodes"])
    if len(episodes) != 1:
        raise RuntimeError(f"expected one evaluated episode for {state.root}, got {len(episodes)}")
    episode = dict(episodes[0])
    family = scenario_family(scenario)
    if not episode.get("family") or episode.get("family") == "None":
        episode["family"] = family
    trace = episode.get("trace") or []
    steps = policy_trace_steps(trace)
    truth = scenario_truth(scenario)
    decisions = lifecycle_decisions(trace, truth)
    reward = episode_reward(episode, family=family, expert_reference_calls=None, weights=weights,
                            kept_wrong_targets=summarize_decisions(decisions)["wrong_target_committed"])
    forced_step = steps[state.position] if state.position < len(steps) else None
    forced_action = forced_step.get("action") if isinstance(forced_step, Mapping) and isinstance(forced_step.get("action"), Mapping) else {}
    forced_executed = bool(
        forced_step is not None and str(forced_action.get("tool")) == forced_tool and forced_step.get("execution_status") == "success"
    )
    reached = policy.calls > state.position
    valid = bool(reached and not policy.mismatches and forced_executed)
    record: dict[str, Any] = {
        "contract": FORK_CONTRACT,
        "root": state.root,
        "sample_index": int(state.sample_index),
        "position": int(state.position),
        "forced_tool": forced_tool,
        "forced_is_sampled_choice": forced_tool == state.sampled_tool,
        "continuation_index": int(continuation_index),
        "seed": int(seed),
        "observation_mode": mode,
        "valid": valid,
        "prefix_matched": not policy.mismatches,
        "mismatches": list(policy.mismatches),
        "fork_reached": reached,
        "forced_executed": forced_executed,
        "forced_error_code": None if forced_step is None else forced_step.get("error_code"),
        "forced_position": int(state.position),
        "success": bool(episode.get("truth_audited_task_success")),
        "reward": reward.as_dict(),
        "steps": episode.get("steps"),
        "policy_steps": episode.get("policy_steps"),
        "continuation_steps": max(0, len(steps) - state.position - 1),
        "continuation_records": len(policy.records),
        "tools": [str((step.get("action") or {}).get("tool")) for step in trace if isinstance(step, Mapping)],
        "lifecycle": decisions,
        "false_commit_count": episode.get("false_commit_count"),
        "invalid_action_count": episode.get("invalid_action_count"),
        "loop_detected": bool(episode.get("loop_detected")),
        "healthy_components_preserved": episode.get("healthy_components_preserved"),
        "chain": recovery_chain(trace, forced_position=state.position, truth=truth),
    }
    if keep_trace:
        record["trace"] = trace
    return record


__all__ = [
    "CHOICES",
    "CaptureEnded",
    "CapturePolicy",
    "DecisionState",
    "FORK_CONTRACT",
    "ForkPolicy",
    "branch_seed",
    "capture_fork_view",
    "decision_states",
    "other_choice",
    "rebind_action",
    "recovery_chain",
    "run_branch",
    "view_signature",
]
