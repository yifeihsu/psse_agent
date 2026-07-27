from __future__ import annotations

import copy
import random
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any, Mapping

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    invalid_action,
    safe_normalize_action,
)
from psse_env.dagger.dataset_builder import validate_policy_payload
from psse_env.dagger.replay_buffer import BalancedReplayBuffer
from psse_env.state_store import OracleState, PolicyObservation, policy_safe_copy


ALL_ADMISSIBLE_SUPERVISION = "all_admissible"
BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION = "bc0_observable_sequential_v1"
SUPPORTED_SUPERVISION_POLICIES = frozenset(
    {
        ALL_ADMISSIBLE_SUPERVISION,
        BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
    }
)


def classify_state_example(
    observation: Mapping[str, Any],
    transition_label: Mapping[str, Any] | None = None,
    *,
    preferred_action: Mapping[str, Any] | str | None = None,
    candidate_assessment: Mapping[str, Any] | None = None,
    target_candidate_disposition: str | None = None,
) -> str:
    """Assign a target-aware replay class from the recovery taxonomy."""
    label = dict(transition_label or {})
    preferred = safe_normalize_action(preferred_action) if preferred_action is not None else None
    preferred_tool = preferred["tool"] if preferred is not None else None
    assessment = dict(candidate_assessment or {})
    target_disposition = (
        target_candidate_disposition
        or assessment.get("disposition")
        or observation.get("candidate_disposition")
    )
    if target_disposition is None and preferred is None:
        target_disposition = label.get("candidate_disposition")
    disposition = (
        str(getattr(target_disposition, "value", target_disposition))
        if target_disposition
        else None
    )

    # The supervision target defines the operational class.  These rules are
    # intentionally ahead of transition outcomes so a malformed learner
    # action cannot relabel a terminal or recovery teacher target as success.
    if preferred_tool == "finalize_diagnosis":
        return "terminal_resolved"
    if (
        preferred_tool == ASK_FOR_MORE_EVIDENCE
        and preferred is not None
        and preferred["arguments"].get("request")
        in {
            HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
            RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
        }
    ):
        return "terminal_operator_escalation"
    if preferred_tool == "rollback_state":
        return "rejected_candidate_recovery"
    if preferred_tool == "commit_state":
        if disposition == "ACCEPT_FINAL":
            return "accepted_final_commit"
        if disposition == "ACCEPT_PARTIAL":
            return "accepted_partial_commit"
        return "invalid_precondition_recovery"

    if label.get("process_valid") is False:
        return "invalid_precondition_recovery"
    last_output = observation.get("last_tool_output")
    if observation.get("last_tool_status") == "failure" or (
        isinstance(last_output, Mapping)
        and (
            last_output.get("execution_status") == "failure"
            or last_output.get("error_code") is not None
        )
    ):
        return "invalid_precondition_recovery"
    if disposition == "REJECT":
        return "rejected_candidate_recovery"
    accepted = observation.get("accepted_corrections") or []
    if disposition == "ACCEPT_PARTIAL" or (
        accepted and not observation.get("no_material_anomaly_remaining")
    ):
        return "accepted_partial_continuation"
    if observation.get("no_material_anomaly_remaining") or disposition == "ACCEPT_FINAL":
        return "terminal_resolved"
    signatures = observation.get("tried_action_signatures") or []
    if len(signatures) != len(set(signatures)):
        return "loop_repetition"
    return "clean_successful"


def audit_target_aware_state_classes(
    examples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute replay classes and report target-semantic violations."""
    rows = list(examples)
    counts: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        observation = row.get("policy_observation") or row.get("state_summary") or {}
        labels = row.get("labels") if isinstance(row.get("labels"), Mapping) else {}
        disposition = (
            labels.get("target_candidate_disposition")
            if "target_candidate_disposition" in labels
            else labels.get("candidate_disposition")
        )
        expected = classify_state_example(
            observation if isinstance(observation, Mapping) else {},
            row.get("transition_label")
            if isinstance(row.get("transition_label"), Mapping)
            else {},
            preferred_action=row.get("preferred_action"),
            candidate_assessment=(
                labels.get("candidate_assessment")
                if isinstance(labels.get("candidate_assessment"), Mapping)
                else None
            ),
            target_candidate_disposition=str(disposition) if disposition else None,
        )
        actual = str(row.get("state_class") or labels.get("state_class") or "")
        counts[actual] += 1
        if actual != expected:
            mismatches.append(
                {
                    "index": index,
                    "example_id": row.get("example_id"),
                    "actual": actual,
                    "expected": expected,
                }
            )
        action = safe_normalize_action(row.get("preferred_action")) if row.get(
            "preferred_action"
        ) is not None else None
        tool = action["tool"] if action else None
        required = {
            "finalize_diagnosis": "terminal_resolved",
            "rollback_state": "rejected_candidate_recovery",
        }.get(tool)
        if (
            tool == ASK_FOR_MORE_EVIDENCE
            and action is not None
            and action["arguments"].get("request")
            in {
                HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
                RECOVERY_BUDGET_EXHAUSTED_REQUEST,
                RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            }
        ):
            required = "terminal_operator_escalation"
        if required is not None and actual != required:
            violations.append(
                {
                    "index": index,
                    "example_id": row.get("example_id"),
                    "preferred_tool": tool,
                    "actual": actual,
                    "required": required,
                }
            )
    return {
        "total_rows": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "mismatches": mismatches,
        "semantic_violations": violations,
        "passed": not mismatches and not violations,
    }


class DaggerRolloutCollector:
    """Collect expert labels at every state visited by the mixture policy."""

    def __init__(
        self,
        *,
        env: Any,
        policy: Any,
        expert_oracle: Any,
        rng: random.Random | None = None,
        supervision_policy: str = ALL_ADMISSIBLE_SUPERVISION,
    ) -> None:
        if supervision_policy not in SUPPORTED_SUPERVISION_POLICIES:
            raise ValueError(
                "supervision_policy must be one of "
                f"{sorted(SUPPORTED_SUPERVISION_POLICIES)}, "
                f"got {supervision_policy!r}"
            )
        if (
            supervision_policy == BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION
            and not bool(getattr(env, "production_dataset_mode", False))
        ):
            raise ValueError(
                f"{BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION} requires "
                "production_dataset_mode=True"
            )
        self.env = env
        self.policy = policy
        self.expert_oracle = expert_oracle
        self.rng = rng or random.Random()
        self.supervision_policy = supervision_policy

    def collect_iteration(
        self,
        *,
        scenarios: Iterable[Mapping[str, Any]],
        iteration: int,
        beta: float,
        max_steps: int,
    ) -> list[dict[str, Any]]:
        if self.supervision_policy == BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION and (
            int(iteration) != 0 or float(beta) != 1.0
        ):
            raise ValueError(
                f"{BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION} is the expert-only "
                "round-0 contract and requires iteration=0, beta=1.0"
            )
        examples: list[dict[str, Any]] = []
        scenario_list = list(scenarios)
        for scenario_index, scenario in enumerate(scenario_list):
            self.env.reset(scenario)
            history: list[dict[str, Any]] = []
            scenario_id = str(scenario.get("scenario_id", scenario.get("id", f"scenario_{scenario_index}")))
            root_scenario_id = str(scenario.get("root_scenario_id", scenario_id))
            physical_root_fingerprint = scenario.get("physical_root_fingerprint")
            scenario_family = str(scenario.get("scenario_family") or "unknown")
            network_case = str(
                scenario.get("network_case") or scenario.get("case_id") or scenario.get("case") or "unknown"
            )
            source_tier = str(scenario.get("source_tier") or "unknown")
            try:
                error_cardinality = int(scenario.get("error_cardinality", 0))
            except (TypeError, ValueError, OverflowError):
                error_cardinality = 0
            state_visited_by = "initial"

            for step in range(max_steps):
                policy_observation = self._policy_observation(history)
                observation_dict = policy_observation.as_dict()
                validate_policy_payload(observation_dict)
                oracle_state = self._oracle_state(history, policy_observation)
                expert_actions = [
                    safe_normalize_action(action)
                    for action in self.expert_oracle.next_actions(oracle_state, history)
                ]
                expert_actions = [
                    action for action in expert_actions if action["tool"] != "__invalid_action__"
                ]
                preferred_action = expert_actions[0] if expert_actions else None
                if (
                    self.supervision_policy
                    == BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION
                ):
                    # BC0 clones a deterministic, deployment-observable expert
                    # protocol rather than a set-valued process-validity
                    # relation.  Only the current rank-one protocol action is
                    # executable now; the ordered remainder is deferred until
                    # a later transition (for example, after a verified
                    # rejection).  Keeping the deferred inventory outside
                    # ``valid_next_actions`` prevents it from being mistaken
                    # for simultaneous single-label supervision while retaining
                    # a non-model audit trail.
                    supervision_actions = (
                        [copy.deepcopy(preferred_action)]
                        if preferred_action is not None
                        else []
                    )
                    deferred_expert_actions = copy.deepcopy(expert_actions[1:])
                else:
                    supervision_actions = copy.deepcopy(expert_actions)
                    deferred_expert_actions = []
                target_candidate_disposition = None
                target_candidate_assessment: dict[str, Any] = {}
                if isinstance(oracle_state, OracleState):
                    target_candidate_disposition = oracle_state.candidate_disposition
                    target_candidate_assessment = copy.deepcopy(
                        dict(oracle_state.candidate_assessment or {})
                    )
                if preferred_action is not None and hasattr(
                    self.env, "assert_training_decision_evidence"
                ):
                    try:
                        self.env.assert_training_decision_evidence(preferred_action)
                    except ValueError as exc:
                        raise ValueError(
                            "Training-decision evidence failed for "
                            f"scenario={scenario_id}, step={step}, "
                            f"preferred_tool={preferred_action.get('tool')}: {exc}"
                        ) from exc
                model_action = self._policy_action(observation_dict)

                if preferred_action is not None and self.rng.random() < float(beta):
                    # Ordinary DAgger labels the visited state with the
                    # rank-one expert action.  Executing a different proposal
                    # here would roll out a different expert policy than the
                    # one represented by ``preferred_action`` in the SFT row.
                    # Deliberate exploration belongs in the counterfactual or
                    # ranking pipelines, where the executed target is recorded
                    # explicitly.
                    executed_action = copy.deepcopy(preferred_action)
                    executed_by = "expert"
                else:
                    executed_action = copy.deepcopy(model_action)
                    executed_by = "model"

                pre_state = self.env.current_state()
                next_state, tool_output = self.env.step(executed_action)
                provisional_transition = {
                    "state_id": observation_dict.get("active_state_id"),
                    "action": policy_safe_copy(executed_action),
                    "tool_output": policy_safe_copy(tool_output),
                }
                provisional_history = history + [provisional_transition]
                next_policy_observation = self._policy_observation(provisional_history)
                next_oracle_state = self._oracle_state(provisional_history, next_policy_observation)
                transition_label = self.expert_oracle.label_transition(
                    state=oracle_state,
                    action=executed_action,
                    tool_output=tool_output,
                    next_state=next_oracle_state,
                    history=provisional_history,
                    store=getattr(self.env, "store", None),
                    hidden_truth=oracle_state.truth_dict() if isinstance(oracle_state, OracleState) else None,
                )
                transition_record = {
                    **provisional_transition,
                    "transition_label": policy_safe_copy(transition_label),
                }
                next_history = history + [transition_record]
                final_next_policy_observation = self._policy_observation(next_history)
                next_oracle_state = self._oracle_state(next_history, final_next_policy_observation)
                next_valid_actions = []
                if not self.env.is_terminal(next_state):
                    next_valid_actions = [
                        safe_normalize_action(action)
                        for action in self.expert_oracle.next_actions(next_oracle_state, next_history)
                    ]
                    next_valid_actions = [
                        action for action in next_valid_actions if action["tool"] != "__invalid_action__"
                    ]
                if not transition_label.get("valid_next_actions"):
                    transition_label["valid_next_actions"] = next_valid_actions

                state_class = classify_state_example(
                    observation_dict,
                    transition_label,
                    preferred_action=preferred_action,
                    candidate_assessment=target_candidate_assessment,
                    target_candidate_disposition=target_candidate_disposition,
                )
                output_metrics = tool_output.get("tool_metrics")
                terminal_outcome = (
                    output_metrics.get("terminal_outcome")
                    if isinstance(output_metrics, Mapping)
                    else None
                )
                dataset_mode = (
                    "production"
                    if bool(getattr(self.env, "production_dataset_mode", False))
                    else "synthetic_pilot"
                )
                example = {
                    "example_id": f"dagger_iter{iteration}_{scenario_id}_step{step}",
                    "scenario_id": scenario_id,
                    "root_scenario_id": root_scenario_id,
                    "physical_root_fingerprint": physical_root_fingerprint,
                    "scenario_family": scenario_family,
                    "error_cardinality": error_cardinality,
                    "network_case": network_case,
                    "source_tier": source_tier,
                    "episode_id": observation_dict.get("episode_id"),
                    "iteration": iteration,
                    "step": step,
                    "dataset_mode": dataset_mode,
                    "policy_observation": observation_dict,
                    "parent_state_summary": observation_dict,
                    "state_summary": observation_dict,
                    "history_window": policy_safe_copy(observation_dict.get("history_window", [])),
                    "valid_next_actions": supervision_actions,
                    "deferred_expert_actions": policy_safe_copy(
                        deferred_expert_actions
                    ),
                    "supervision_policy": self.supervision_policy,
                    "preferred_action": preferred_action,
                    "model_action": policy_safe_copy(model_action),
                    "executed_action": policy_safe_copy(executed_action),
                    "executed_by": executed_by,
                    "state_visited_by": state_visited_by,
                    "tool_output": policy_safe_copy(tool_output),
                    "next_state_summary": final_next_policy_observation.as_dict(),
                    "candidate_state_summary": (
                        final_next_policy_observation.as_dict()
                        if final_next_policy_observation.candidate_state_id
                        else {}
                    ),
                    "transition_label": policy_safe_copy(transition_label),
                    "next_valid_actions": policy_safe_copy(next_valid_actions),
                    "state_class": state_class,
                    "terminal_outcome": terminal_outcome,
                    "labels": {
                        "candidate_disposition": transition_label.get("candidate_disposition"),
                        "target_candidate_disposition": target_candidate_disposition,
                        "candidate_assessment": policy_safe_copy(
                            target_candidate_assessment
                        ),
                        "progress_class": transition_label.get("progress_class"),
                        "process_valid": transition_label.get("process_valid"),
                        "state_class": state_class,
                        "dataset_mode": dataset_mode,
                        "terminal_outcome": terminal_outcome,
                        "scenario_family": scenario_family,
                        "error_cardinality": error_cardinality,
                        "network_case": network_case,
                        "source_tier": source_tier,
                        "supervision_policy": self.supervision_policy,
                        "deferred_expert_action_count": len(
                            deferred_expert_actions
                        ),
                    },
                }
                validate_policy_payload(
                    {
                        "policy_observation": example["policy_observation"],
                        "history_window": example["history_window"],
                    }
                )
                examples.append(example)
                history = next_history
                state_visited_by = executed_by
                if self.env.is_terminal(next_state):
                    break
        return examples

    def _policy_observation(self, history: list[Mapping[str, Any]]) -> PolicyObservation:
        if hasattr(self.env, "get_policy_observation"):
            observation = self.env.get_policy_observation(history)
            if isinstance(observation, PolicyObservation):
                return observation
            if isinstance(observation, Mapping):
                return PolicyObservation(**dict(observation))
        state = self.env.current_state()
        return PolicyObservation(
            active_state_id=str(state["active_state_id"]),
            history_window=policy_safe_copy(history),
            remaining_budget=int(state.get("remaining_budget") or 0),
        )

    def _oracle_state(
        self,
        history: list[Mapping[str, Any]],
        policy_observation: PolicyObservation,
    ) -> OracleState | Mapping[str, Any]:
        if hasattr(self.env, "get_oracle_state"):
            return self.env.get_oracle_state(history)
        return policy_observation.as_dict()

    def _policy_action(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        try:
            if hasattr(self.policy, "act"):
                raw = self.policy.act(copy.deepcopy(dict(observation)))
            elif callable(self.policy):
                raw = self.policy(copy.deepcopy(dict(observation)))
            else:
                raise TypeError("policy must be callable or expose .act(obs)")
        except Exception as exc:  # collection must retain arbitrary learner failures
            return invalid_action("policy_exception", f"{type(exc).__name__}: {exc}")
        return safe_normalize_action(raw)


def _validation_score(result: Any) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, Mapping):
        for key in ("score", "recovery_score", "validation_score"):
            if result.get(key) is not None:
                return float(result[key])
    if hasattr(result, "score"):
        return float(result.score)
    raise TypeError("evaluate_fn must return a number, a mapping with score, or an object with .score")


def _requires_explicit_snapshot(
    policy: Any, *, _seen: set[int] | None = None, _depth: int = 0
) -> bool:
    if policy is None or _depth > 3:
        return False
    seen = _seen if _seen is not None else set()
    identity = id(policy)
    if identity in seen:
        return False
    seen.add(identity)
    policy_type = type(policy)
    type_path = f"{policy_type.__module__}.{policy_type.__qualname__}".lower()
    mro_paths = " ".join(
        f"{item.__module__}.{item.__qualname__}".lower()
        for item in getattr(policy_type, "__mro__", ())
    )
    framework_policy = any(
        marker in f"{type_path} {mro_paths}"
        for marker in ("transformers.", "peft.", "accelerate.")
    )
    sharded_or_quantized = bool(getattr(policy, "hf_device_map", None)) or any(
        bool(getattr(policy, name, False))
        for name in ("is_loaded_in_4bit", "is_loaded_in_8bit")
    )
    peft_policy = getattr(policy, "peft_config", None) is not None
    pretrained_policy = all(
        hasattr(policy, name) for name in ("save_pretrained", "config", "state_dict")
    )
    if framework_policy or sharded_or_quantized or peft_policy or pretrained_policy:
        return True
    for attribute in ("policy", "model", "module"):
        try:
            nested = getattr(policy, attribute, None)
        except Exception:
            continue
        if nested is not None and nested is not policy and _requires_explicit_snapshot(
            nested, _seen=seen, _depth=_depth + 1
        ):
            return True
    return False


def _snapshot_policy(policy: Any, snapshot_policy_fn: Callable[[Any], Any] | None) -> Any:
    if snapshot_policy_fn is not None:
        return snapshot_policy_fn(policy)
    if _requires_explicit_snapshot(policy):
        raise TypeError(
            "evaluate_fn checkpoint selection requires snapshot_policy_fn for "
            "Transformers, PEFT, quantized, or sharded policies; generic deepcopy "
            "is not a reliable 31B checkpoint snapshot."
        )
    try:
        return copy.deepcopy(policy)
    except Exception as exc:
        raise TypeError(
            "Policy is not deepcopy-able; provide snapshot_policy_fn so best-checkpoint selection is reliable."
        ) from exc


def run_dagger(
    *,
    policy: Any,
    expert_oracle: Any,
    env: Any,
    scenarios_by_iteration: Callable[[int], Iterable[Mapping[str, Any]]] | Iterable[Mapping[str, Any]],
    initial_dataset: list[dict[str, Any]] | None = None,
    num_iterations: int = 8,
    beta_schedule: list[float] | None = None,
    max_steps: int = 24,
    train_policy_fn: Callable[[Any, list[dict[str, Any]]], Any] | None = None,
    evaluate_fn: Callable[[Any, Any, Any], Any] | None = None,
    snapshot_policy_fn: Callable[[Any], Any] | None = None,
    training_dataset_fn: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]] | None = None,
    balanced_replay: bool = True,
    replay_sample_size: int | None = None,
    replay_unknown_class_policy: str = "error",
    replay_unknown_class_weight: float = 0.05,
    replay_max_duplicate_count: int = 2,
    replay_max_rows_per_root: int | None = None,
    replay_late_iteration_model_fraction: float = 0.25,
    replay_require_late_iteration_model_quota: bool = True,
    replay_report_fn: Callable[[Mapping[str, Any], int], None] | None = None,
    rng: random.Random | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    dataset = list(initial_dataset or [])
    betas = beta_schedule or [1.0, 0.5, 0.25, 0.1, 0.05, 0.0, 0.0, 0.0]
    current_policy = policy
    best_policy: Any | None = None
    best_score = float("-inf")
    shared_rng = rng or random.Random()

    if evaluate_fn is not None:
        best_score = _validation_score(evaluate_fn(current_policy, env, expert_oracle))
        best_policy = _snapshot_policy(current_policy, snapshot_policy_fn)

    materialized_scenarios: list[Mapping[str, Any]] | None = None
    if not callable(scenarios_by_iteration):
        materialized_scenarios = list(scenarios_by_iteration)
        if not materialized_scenarios:
            raise ValueError("scenarios_by_iteration is empty.")

    for iteration in range(num_iterations):
        beta = betas[min(iteration, len(betas) - 1)]
        if callable(scenarios_by_iteration):
            scenarios = list(scenarios_by_iteration(iteration))
            if not scenarios:
                raise ValueError(f"Scenario provider returned no scenarios for iteration {iteration}.")
        else:
            scenarios = list(materialized_scenarios or [])
        collector = DaggerRolloutCollector(
            env=env,
            policy=current_policy,
            expert_oracle=expert_oracle,
            rng=shared_rng,
        )
        dataset.extend(
            collector.collect_iteration(
                scenarios=scenarios,
                iteration=iteration,
                beta=beta,
                max_steps=max_steps,
            )
        )
        if training_dataset_fn is not None:
            training_dataset = training_dataset_fn(dataset, iteration)
        elif balanced_replay and train_policy_fn is not None:
            sample_size = replay_sample_size if replay_sample_size is not None else len(dataset)
            replay_buffer = BalancedReplayBuffer(
                dataset,
                unknown_class_policy=replay_unknown_class_policy,
                unknown_class_weight=replay_unknown_class_weight,
                max_duplicate_count=replay_max_duplicate_count,
                max_rows_per_root=replay_max_rows_per_root,
                late_iteration_model_fraction=replay_late_iteration_model_fraction,
                require_late_iteration_model_quota=(
                    replay_require_late_iteration_model_quota
                ),
            )
            training_dataset = replay_buffer.sample(sample_size, rng=shared_rng)
            if replay_report_fn is not None:
                replay_report_fn(replay_buffer.sample_report() or {}, iteration)
        else:
            training_dataset = dataset
        if train_policy_fn is not None:
            current_policy = train_policy_fn(current_policy, list(training_dataset))
        if evaluate_fn is not None:
            score = _validation_score(evaluate_fn(current_policy, env, expert_oracle))
            if score > best_score:
                best_score = score
                best_policy = _snapshot_policy(current_policy, snapshot_policy_fn)

    selected_policy = best_policy if evaluate_fn is not None and best_policy is not None else current_policy
    return selected_policy, dataset
