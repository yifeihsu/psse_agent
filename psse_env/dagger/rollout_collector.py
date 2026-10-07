from __future__ import annotations

import copy
import math
import random
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import Executor
from dataclasses import replace
from typing import Any, Mapping

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    POST_CORRECTION_CONFIRMATION_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    invalid_action,
    safe_normalize_action,
)
from psse_env.dagger.dataset_builder import validate_policy_payload
from psse_env.episode_budget import bind_env_action_limit
from psse_env.dagger.offline_teacher_target_audit import (
    offline_teacher_target_audit,
)
from psse_env.state_store import OracleState, PolicyObservation, policy_safe_copy


ALL_ADMISSIBLE_SUPERVISION = "all_admissible"
BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION = (
    "bc0_observable_sequential_handoff_v2"
)
DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION = (
    "dagger1_observable_recovery_handoff_v2"
)
SUPPORTED_SUPERVISION_POLICIES = frozenset(
    {
        ALL_ADMISSIBLE_SUPERVISION,
        BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
        DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    }
)
RECOMMENDED_DAGGER1_RECOVERY_STRATA = frozenset(
    {
        "multi_measurement_safe_handoff",
        "post_failure_no_candidate",
        "premature_commit_recovery",
        "premature_escalation_recovery",
        "sequential_measurement_parameter_recovery",
        "unsupported_correction_recovery",
    }
)
_DAGGER1_PRODUCTION_RECOVERY_STRATA = frozenset(
    {
        *RECOMMENDED_DAGGER1_RECOVERY_STRATA,
        "invalid_precondition_repair",
        "loop_escape",
        "rejected_candidate_rollback",
    }
)


def _observable_last_action_tool(observation: Mapping[str, Any]) -> str | None:
    last_tool = str(observation.get("last_tool") or "").strip()
    if last_tool:
        return last_tool
    history = observation.get("history_window")
    if not isinstance(history, (list, tuple)) or not history:
        return None
    event = history[-1]
    if not isinstance(event, Mapping):
        return None
    action = event.get("action")
    if not isinstance(action, Mapping):
        return None
    tool = str(action.get("tool") or "").strip()
    return tool or None


def _observable_last_failure(observation: Mapping[str, Any]) -> tuple[bool, str | None]:
    output = observation.get("last_tool_output")
    output = output if isinstance(output, Mapping) else {}
    failed = observation.get("last_tool_status") == "failure" or (
        output.get("execution_status") == "failure"
        or output.get("error_code") is not None
    )
    error_code = str(output.get("error_code") or "").strip()
    return bool(failed), error_code or None


def observable_candidate_verified(observation: Mapping[str, Any]) -> bool:
    """Policy-visible test for a verified, committable candidate.

    Scope note: this is currently a state-class helper aligned with the
    observable candidate-lifecycle contract, called only from
    ``observable_commit_class``.  It is deliberately *not* yet the single
    shared implementation behind expert commit/rollback reconstruction and the
    rank-one target proof; unifying those paths should reuse the existing
    observable candidate-disposition implementation rather than add a third
    independent approximation of it.
    """
    lifecycle = str(observation.get("candidate_lifecycle") or "").strip().upper()
    status = str(observation.get("candidate_status") or "").strip().lower()
    return bool(
        observation.get("has_verified_candidate")
        or status == "verified"
        or lifecycle == "VERIFIED_CANDIDATE"
    )


def observable_commit_class(
    observation: Mapping[str, Any],
    *,
    declared_disposition: Any = None,
) -> str:
    """Replay class for a ``commit_state`` target, from observable evidence.

    The candidate disposition is frequently absent from the policy-visible
    state.  Recomputing the DAgger-1 round-2 collection shows 215 affected rows
    — every ``commit_state`` row in the run — each holding a
    ``VERIFIED_CANDIDATE`` with no disposition from any source, which the
    previous catch-all misfiled as ``invalid_precondition_recovery``.  (The
    initial diagnosis reported 159, which is the narrower learner-visited,
    previously unclassified subset of the same population.)  The same
    recomputation over the D0 aggregate changes 0 of its 256 ``commit_state``
    rows, so D0 row semantics are unaffected.

    Deriving the class from the same lifecycle evidence the expert uses to
    build the commit target keeps the taxonomy aligned with the teacher without
    exposing private candidate disposition.
    """
    if declared_disposition:
        token = str(
            getattr(declared_disposition, "value", declared_disposition)
        ).strip().upper()
        if token == "ACCEPT_FINAL":
            return "accepted_final_commit"
        if token == "ACCEPT_PARTIAL":
            return "accepted_partial_commit"
        if token == "REJECT":
            return "rejected_candidate_recovery"
    if not observable_candidate_verified(observation):
        return "invalid_precondition_recovery"
    if observation.get("no_material_anomaly_remaining"):
        return "accepted_final_commit"
    return "accepted_partial_commit"


def _action_family(action: Mapping[str, Any] | str | None) -> str | None:
    normalized = safe_normalize_action(action) if action is not None else None
    tool = normalized["tool"] if normalized is not None else None
    if tool in {"get_measurement_context", "correct_measurements", "correct_measurements_from_path"}:
        return "measurement"
    if tool in {"get_parameter_context", "correct_parameters", "correct_parameters_from_path"}:
        return "parameter"
    return None


def _observable_prior_action_families(observation: Mapping[str, Any]) -> set[str]:
    families: set[str] = set()
    history = observation.get("history_window")
    if isinstance(history, (list, tuple)):
        for event in history:
            if not isinstance(event, Mapping):
                continue
            family = _action_family(
                event.get("action") if isinstance(event.get("action"), Mapping) else None
            )
            if family is not None:
                families.add(family)
    accepted = observation.get("accepted_corrections")
    if isinstance(accepted, (list, tuple)):
        for record in accepted:
            if not isinstance(record, Mapping):
                continue
            action = record.get("source_action") or record.get("action")
            family = _action_family(action if isinstance(action, Mapping) else None)
            if family is not None:
                families.add(family)
    evidence = observation.get("fresh_context_evidence")
    if isinstance(evidence, Mapping):
        for family in ("measurement", "parameter"):
            if isinstance(evidence.get(family), Mapping):
                families.add(family)
    return families


def classify_dagger1_recovery_stratum(
    observation: Mapping[str, Any],
    *,
    preferred_action: Mapping[str, Any] | str | None,
    state_class: str,
    scenario_family: str,
    error_cardinality: int,
) -> str | None:
    """Classify an observable learner-recovery state into an audit stratum.

    This classifier deliberately uses only the policy observation, public
    scenario grouping, and the expert's current rank-one target.  It must not
    inspect hidden truth or the action that the expert will execute next.
    """
    del state_class  # Current-transition outcomes must not admit the input row.
    preferred = (
        safe_normalize_action(preferred_action)
        if preferred_action is not None
        else None
    )
    preferred_tool = preferred["tool"] if preferred is not None else None
    previous_tool = _observable_last_action_tool(observation)
    previous_failed, error_code = _observable_last_failure(observation)
    has_candidate = bool(
        observation.get("has_open_candidate")
        or observation.get("candidate_state_id")
    )

    unsupported_codes = {
        "correction_not_supported_by_current_context",
        "correction_route_not_actionable",
        "parameter_scans_missing",
        "post_correction_confirmation_required",
    }
    if (
        previous_failed
        and previous_tool
        in {
            "correct_measurements",
            "correct_measurements_from_path",
            "correct_parameters",
            "correct_parameters_from_path",
            "correct_topology",
            "correct_topology_from_path",
        }
        and error_code in unsupported_codes
    ):
        return "unsupported_correction_recovery"
    if (
        previous_failed
        and previous_tool == "commit_state"
        and (error_code == "candidate_lifecycle_violation" or not has_candidate)
    ):
        return "premature_commit_recovery"
    if (
        previous_failed
        and previous_tool == ASK_FOR_MORE_EVIDENCE
        and error_code == "operator_escalation_precondition_not_met"
    ):
        return "premature_escalation_recovery"
    if preferred_tool == "rollback_state" and has_candidate:
        return "rejected_candidate_rollback"

    if (
        str(scenario_family) == "multi_measurement"
        and int(error_cardinality) >= 2
        and preferred_tool == ASK_FOR_MORE_EVIDENCE
        and preferred is not None
        and preferred["arguments"].get("request")
        in {
            HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
            RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            POST_CORRECTION_CONFIRMATION_REQUEST,
        }
    ):
        return "multi_measurement_safe_handoff"

    preferred_family = _action_family(preferred)
    if (
        str(scenario_family) == "measurement+parameter"
        and preferred_family in {"measurement", "parameter"}
        and ({"measurement", "parameter"} - {preferred_family})
        & _observable_prior_action_families(observation)
    ):
        return "sequential_measurement_parameter_recovery"

    if previous_failed and not has_candidate:
        return "post_failure_no_candidate"
    signatures = observation.get("tried_action_signatures")
    if (
        isinstance(signatures, (list, tuple))
        and len(signatures) != len(set(str(item) for item in signatures))
    ):
        return "loop_escape"
    if previous_failed:
        return "invalid_precondition_repair"
    return None


def observable_rank_one_target_proof(
    observation: Mapping[str, Any],
    *,
    preferred_action: Mapping[str, Any] | str | None,
    expert_actions: Iterable[Mapping[str, Any] | str],
) -> dict[str, Any]:
    """Prove one deterministic target from policy-visible ranked evidence."""
    actions = [safe_normalize_action(action) for action in expert_actions]
    preferred = (
        safe_normalize_action(preferred_action)
        if preferred_action is not None
        else None
    )
    if preferred is None or not actions or actions[0] != preferred:
        return {"contract": "observable_rank_one_target_v1", "passed": False, "reason": "preferred_target_missing_or_not_first"}
    if len(actions) == 1:
        return {
            "contract": "observable_rank_one_target_v1",
            "passed": True,
            "basis": "singleton_expert_target",
            "expert_action_count": 1,
        }
    correction_tools = {
        "correct_measurements",
        "correct_parameters",
        "correct_parameters_from_path",
        "correct_topology",
    }
    if preferred["tool"] not in correction_tools:
        # A context fetch, a request, a solve, a commit, or a handoff is a
        # deterministic ladder choice: the first proposal is the target, and
        # the runners-up are the same ladder's later rungs, not competing
        # physical hypotheses.  Rank-one proof is a property of correction
        # targets only.
        return {
            "contract": "observable_rank_one_target_v1",
            "passed": True,
            "basis": "deterministic_non_correction_target",
            "expert_action_count": len(actions),
        }

    parameter_tools = {"correct_parameters", "correct_parameters_from_path"}
    if preferred["tool"] not in parameter_tools or any(
        action["tool"] not in parameter_tools for action in actions
    ):
        return {
            "contract": "observable_rank_one_target_v1",
            "passed": False,
            "reason": "multiple_targets_without_supported_parameter_ranking",
            "expert_action_count": len(actions),
        }
    evidence_by_family = observation.get("fresh_context_evidence")
    evidence = (
        evidence_by_family.get("parameter")
        if isinstance(evidence_by_family, Mapping)
        else None
    )
    if not isinstance(evidence, Mapping):
        return {"contract": "observable_rank_one_target_v1", "passed": False, "reason": "parameter_evidence_missing", "expert_action_count": len(actions)}
    active_state_id = str(observation.get("active_state_id") or "")
    binding = str(evidence.get("context_binding") or "")
    binding_valid = binding == "direct_context" or (
        binding == "branch_route_screening.parameter"
        and evidence.get("bundled_by_context_tool") == "get_measurement_context"
    )
    evidence_source = str(evidence.get("evidence_source") or "").lower()
    source_valid = bool(
        evidence_source
        and not any(
            token in evidence_source
            for token in ("hidden", "oracle", "truth", "synthetic", "fallback")
        )
        and evidence_source.startswith(
            ("deployment", "observable", "sensor", "wls", "configured_provider")
        )
    )
    if not (
        observation.get("has_fresh_parameter_context") is True
        and str(observation.get("parameter_context_state_id") or "")
        == active_state_id
        and evidence.get("context_tool") == "get_parameter_context"
        and binding_valid
        and source_valid
        and evidence.get("route_status") == "actionable"
        and str(evidence.get("state_id") or "") == active_state_id
        and str(evidence.get("state_hash") or "").strip()
        and evidence.get("parameter_ranking_contract")
        == "distinct_line_abs_lambda_dominance_v1"
    ):
        return {"contract": "observable_rank_one_target_v1", "passed": False, "reason": "parameter_evidence_not_actionable_or_state_bound", "expert_action_count": len(actions)}
    ranking = evidence.get("parameter_ranking_distinct_lines")
    if not isinstance(ranking, (list, tuple)) or len(ranking) < 2:
        return {"contract": "observable_rank_one_target_v1", "passed": False, "reason": "multi_target_parameter_ranking_incomplete", "expert_action_count": len(actions)}
    raw_supported = evidence.get("supported_corrections")
    if not isinstance(raw_supported, (list, tuple)):
        return {"contract": "observable_rank_one_target_v1", "passed": False, "reason": "supported_parameter_inventory_missing", "expert_action_count": len(actions)}
    supported = [safe_normalize_action(action) for action in raw_supported]
    if supported != actions:
        return {"contract": "observable_rank_one_target_v1", "passed": False, "reason": "expert_targets_do_not_match_supported_inventory", "expert_action_count": len(actions)}

    def finite(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None

    top = ranking[0] if isinstance(ranking[0], Mapping) else {}
    runner = ranking[1] if isinstance(ranking[1], Mapping) else {}
    top_score = finite(top.get("abs_lambda_score"))
    runner_score = finite(runner.get("abs_lambda_score"))
    recorded_top = finite(evidence.get("parameter_ranking_top_abs_lambda"))
    recorded_runner = finite(
        evidence.get("parameter_ranking_runner_up_abs_lambda")
    )
    ratio = finite(evidence.get("parameter_ranking_dominance_ratio"))
    threshold = finite(evidence.get("parameter_ranking_dominance_threshold"))
    try:
        top_line = int(top.get("line_index1"))
        preferred_line = int(preferred["arguments"].get("line_index"))
    except (TypeError, ValueError, OverflowError):
        top_line = preferred_line = -1
    scores_consistent = bool(
        top_score is not None
        and runner_score is not None
        and recorded_top is not None
        and recorded_runner is not None
        and ratio is not None
        and threshold is not None
        and math.isclose(top_score, recorded_top, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(
            runner_score, recorded_runner, rel_tol=1e-12, abs_tol=1e-12
        )
        and runner_score > 0.0
        and math.isclose(
            ratio, top_score / runner_score, rel_tol=1e-12, abs_tol=1e-12
        )
    )
    action_state_ids = {
        str(action["arguments"].get("state_id") or "") for action in actions
    }
    # The proof asks for a strict ranking, not for dominance at the route's
    # threshold: the top-ranked line is a deterministic first target whether
    # the ranking clears the dominance threshold (the route is actionable
    # outright) or falls inside the ambiguity band (the top candidates are
    # tested in rank order under verification).  Ties remain unprovable.
    passed = bool(
        scores_consistent
        and top_score > runner_score
        and ratio > 1.0
        and preferred_line == top_line > 0
        and action_state_ids == {active_state_id}
    )
    return {
        "contract": "observable_rank_one_target_v1",
        "passed": passed,
        "basis": (
            None
            if not passed
            else "strict_observable_parameter_ranking"
            if evidence.get("parameter_ranking_dominant") is True
            else "ranked_ambiguity_candidate"
        ),
        "reason": None if passed else "parameter_ranking_not_strict_or_target_mismatch",
        "expert_action_count": len(actions),
        "active_state_id": active_state_id,
        "top_line_index1": top_line if top_line > 0 else None,
        "preferred_line_index1": preferred_line if preferred_line > 0 else None,
        "top_abs_lambda": top_score,
        "runner_up_abs_lambda": runner_score,
        "dominance_ratio": ratio,
        "dominance_threshold": threshold,
    }


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
            POST_CORRECTION_CONFIRMATION_REQUEST,
        }
    ):
        return "terminal_operator_escalation"
    if preferred_tool == "rollback_state":
        return "rejected_candidate_recovery"
    if preferred_tool == "commit_state":
        return observable_commit_class(
            observation, declared_disposition=disposition
        )

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
                POST_CORRECTION_CONFIRMATION_REQUEST,
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
    """Collect expert labels at every state visited by the mixture policy.

    ``policy_executor`` is caller-owned and, when supplied, must be one
    persistent single-worker executor.  The collector never shuts it down so a
    top-level collection schedule can reuse the same worker across episodes.
    """

    def __init__(
        self,
        *,
        env: Any,
        policy: Any,
        expert_oracle: Any,
        rng: random.Random | None = None,
        supervision_policy: str = ALL_ADMISSIBLE_SUPERVISION,
        forbidden_physical_roots: Iterable[str] | None = None,
        policy_executor: Executor | None = None,
        stop_episode_on_unverified_expert_label: bool = False,
    ) -> None:
        if supervision_policy not in SUPPORTED_SUPERVISION_POLICIES:
            raise ValueError(
                "supervision_policy must be one of "
                f"{sorted(SUPPORTED_SUPERVISION_POLICIES)}, "
                f"got {supervision_policy!r}"
            )
        if (
            supervision_policy
            in {
                BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
                DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
            }
            and not bool(getattr(env, "production_dataset_mode", False))
        ):
            raise ValueError(
                f"{supervision_policy} requires "
                "production_dataset_mode=True"
            )
        self.env = env
        self.policy = policy
        self.expert_oracle = expert_oracle
        self.rng = rng or random.Random()
        # When the expert proposes a label the environment cannot justify from
        # observable evidence, the default is to abort: a release corpus must
        # not be silently shortened.  A research run may instead end that one
        # episode, keep the verified rows already collected, and record the
        # disagreement for inspection.
        self.stop_episode_on_unverified_expert_label = bool(
            stop_episode_on_unverified_expert_label
        )
        self.unverified_expert_labels: list[dict[str, Any]] = []
        self.supervision_policy = supervision_policy
        self.policy_executor = policy_executor
        self.forbidden_physical_roots = frozenset(
            str(root).strip()
            for root in (forbidden_physical_roots or [])
            if str(root).strip()
        )
        if (
            supervision_policy == DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
            and not self.forbidden_physical_roots
        ):
            raise ValueError(
                f"{DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION} requires a "
                "non-empty forbidden_physical_roots holdout set"
            )

    def collect_iteration(
        self,
        *,
        scenarios: Iterable[Mapping[str, Any]],
        iteration: int,
        beta: float,
        max_steps: int,
        collection_role: str | None = None,
    ) -> list[dict[str, Any]]:
        max_steps = bind_env_action_limit(self.env, max_steps)
        if self.supervision_policy == BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION and (
            int(iteration) != 0 or float(beta) != 1.0
        ):
            raise ValueError(
                f"{BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION} is the expert-only "
                "round-0 contract and requires iteration=0, beta=1.0"
            )
        if self.supervision_policy == DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION:
            if int(iteration) < 1:
                raise ValueError(
                    f"{DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION} requires "
                    "iteration>=1"
                )
            if collection_role == "diagnostic":
                valid_beta = float(beta) == 0.0
            elif collection_role == "training":
                valid_beta = 0.25 <= float(beta) <= 0.5
            else:
                raise ValueError(
                    f"{DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION} requires an "
                    "explicit collection_role of diagnostic or training"
                )
            if not valid_beta:
                raise ValueError(
                    f"{collection_role} DAgger-1 collection has an invalid "
                    f"beta={float(beta)}"
                )
        examples: list[dict[str, Any]] = []
        scenario_list = list(scenarios)
        for scenario_index, scenario in enumerate(scenario_list):
            runtime_scenario: Mapping[str, Any] = scenario
            grouping: Mapping[str, Any] = scenario
            offline_audit_scenario: dict[str, Any] = copy.deepcopy(dict(scenario))
            if (
                self.supervision_policy
                == DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                and all(key in scenario for key in ("execution", "audit", "grouping"))
            ):
                raw_execution = scenario.get("execution")
                raw_audit = scenario.get("audit")
                raw_grouping = scenario.get("grouping")
                if not all(
                    isinstance(value, Mapping)
                    for value in (raw_execution, raw_audit, raw_grouping)
                ):
                    raise ValueError("DAgger-1 scenario envelope is malformed")
                runtime_scenario = dict(raw_execution)
                private_truth = raw_audit.get("truth")
                if not isinstance(private_truth, Mapping):
                    raise ValueError(
                        "DAgger-1 scenario envelope lacks private audit truth"
                    )
                # The environment stores these fields exclusively in its
                # OracleState payload.  They are required for expert labels,
                # but get_policy_observation/validate_policy_payload keep them
                # out of both learner input and exported SFT rows.
                normalized_private_truth = dict(private_truth)
                clean_state = normalized_private_truth.get("clean_state")
                if isinstance(clean_state, Mapping):
                    for nested_key, flat_key in (
                        ("case", "clean_case"),
                        ("measurements", "clean_measurements"),
                    ):
                        if nested_key not in clean_state:
                            continue
                        nested_value = clean_state[nested_key]
                        if (
                            flat_key in normalized_private_truth
                            and normalized_private_truth[flat_key] != nested_value
                        ):
                            raise ValueError(
                                f"DAgger-1 private truth conflicts on {flat_key}"
                            )
                        normalized_private_truth[flat_key] = nested_value
                for key, value in normalized_private_truth.items():
                    if key in {"truth_complete", "clean_state"}:
                        continue
                    runtime_scenario[str(key)] = copy.deepcopy(value)
                grouping = raw_grouping
                offline_audit_scenario = copy.deepcopy(dict(runtime_scenario))
                offline_audit_scenario["truth_complete"] = (
                    normalized_private_truth.get("truth_complete") is True
                )
                release_audit = raw_audit.get("release_audit")
                if isinstance(release_audit, Mapping):
                    # This profile is private physical-audit input.  The
                    # environment retains it only in OracleState so commit-time
                    # truth retirement uses the same declared tolerance as the
                    # strict offline release audit.
                    runtime_scenario["release_audit"] = copy.deepcopy(
                        dict(release_audit)
                    )
                    offline_audit_scenario["release_audit"] = copy.deepcopy(
                        dict(release_audit)
                    )
                for key in (
                    "root_scenario_id",
                    "physical_root_fingerprint",
                    "scenario_family",
                    "error_cardinality",
                ):
                    if raw_grouping.get(key) is not None:
                        offline_audit_scenario[key] = copy.deepcopy(
                            raw_grouping[key]
                        )
            if self.supervision_policy == DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION:
                split = str(
                    grouping.get("dataset_split") or grouping.get("split") or ""
                ).strip()
                if split not in {"train", "dagger_train"}:
                    raise ValueError(
                        f"{DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION} accepts only "
                        "explicit train/dagger_train scenarios; got "
                        f"{split or 'missing'}"
                    )
                if not str(
                    grouping.get("physical_root_fingerprint") or ""
                ).strip():
                    raise ValueError(
                        f"{DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION} requires an "
                        "explicit physical_root_fingerprint"
                    )
                physical_root = str(
                    grouping["physical_root_fingerprint"]
                ).strip()
                if physical_root in self.forbidden_physical_roots:
                    raise ValueError(
                        "DAgger-1 training scenario overlaps a protected "
                        f"D0/evaluation holdout: {physical_root}"
                    )
            self.env.reset(runtime_scenario)
            history: list[dict[str, Any]] = []
            scenario_id = str(runtime_scenario.get("scenario_id", runtime_scenario.get("id", f"scenario_{scenario_index}")))
            root_scenario_id = str(grouping.get("root_scenario_id", scenario_id))
            physical_root_fingerprint = grouping.get("physical_root_fingerprint")
            scenario_family = str(grouping.get("scenario_family") or "unknown")
            network_case = str(
                grouping.get("network_case") or grouping.get("case_id") or runtime_scenario.get("case") or "unknown"
            )
            source_tier = str(grouping.get("source_tier") or "unknown")
            try:
                error_cardinality = int(grouping.get("error_cardinality", 0))
            except (TypeError, ValueError, OverflowError):
                error_cardinality = 0
            state_visited_by = "initial"

            for step in range(max_steps):
                policy_observation = replace(
                    self._policy_observation(history), remaining_budget=max_steps - step
                )
                observation_dict = policy_observation.as_dict()
                validate_policy_payload(observation_dict)
                policy_action_future = None
                if self.policy_executor is not None:
                    # The overlap boundary is deliberately narrow.  Submit one
                    # policy call only after the immutable learner payload has
                    # passed validation, and isolate the worker from every
                    # object the main-thread audits below.
                    policy_action_future = self.policy_executor.submit(
                        self._policy_action,
                        copy.deepcopy(observation_dict),
                    )
                oracle_state: OracleState | Mapping[str, Any] | None = None
                if (
                    self.supervision_policy
                    != DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                ):
                    oracle_state = self._oracle_state(
                        history, policy_observation
                    )
                expert_actions = self._select_expert_actions(
                    policy_observation=policy_observation,
                    oracle_state=oracle_state,
                    history=history,
                )
                preferred_action = expert_actions[0] if expert_actions else None
                rank_one_target_proof = observable_rank_one_target_proof(
                    observation_dict,
                    preferred_action=preferred_action,
                    expert_actions=expert_actions,
                )
                if (
                    self.supervision_policy
                    == DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                ):
                    # The D1 teacher target and its observable proof are fixed
                    # before private state is constructed.  OracleState is
                    # available only to the post-target audit and transition
                    # truth label below.
                    oracle_state = self._oracle_state(
                        history, policy_observation
                    )
                if self.supervision_policy in {
                    BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
                    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
                }:
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
                if (
                    self.supervision_policy
                    != DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                    and isinstance(oracle_state, OracleState)
                ):
                    target_candidate_disposition = oracle_state.candidate_disposition
                    target_candidate_assessment = copy.deepcopy(
                        dict(oracle_state.candidate_assessment or {})
                    )
                training_decision_evidence_verified = False
                if preferred_action is not None and hasattr(
                    self.env, "assert_training_decision_evidence"
                ):
                    try:
                        self.env.assert_training_decision_evidence(preferred_action)
                    except ValueError as exc:
                        if not self.stop_episode_on_unverified_expert_label:
                            raise ValueError(
                                "Training-decision evidence failed for "
                                f"scenario={scenario_id}, step={step}, "
                                f"preferred_tool={preferred_action.get('tool')}: {exc}"
                            ) from exc
                        # Capture the environment's own escalation ledger next
                        # to the expert's proposal.  The disagreement is between
                        # these two views of "exhausted", so recording only the
                        # error text leaves the actual cause unobservable.
                        diagnostic: dict[str, Any] = {
                            "scenario_id": str(scenario_id),
                            "step": int(step),
                            "preferred_tool": str(preferred_action.get("tool")),
                            "preferred_action": copy.deepcopy(dict(preferred_action)),
                            "reason": str(exc),
                        }
                        audit_fn = getattr(
                            self.env, "_operator_escalation_audit", None
                        )
                        if callable(audit_fn):
                            try:
                                audit = audit_fn(preferred_action)
                                diagnostic["escalation_audit"] = {
                                    "missing": list(audit.get("missing") or []),
                                    "ledger": copy.deepcopy(
                                        dict(audit.get("ledger") or {})
                                    ),
                                }
                            except Exception as audit_exc:  # noqa: BLE001
                                diagnostic["escalation_audit_error"] = (
                                    f"{type(audit_exc).__name__}: {audit_exc}"
                                )
                        try:
                            summary_state = self.env.current_state()
                            diagnostic["state"] = {
                                "accepted_corrections": copy.deepcopy(
                                    list(summary_state.get("accepted_corrections") or [])
                                ),
                                "rejected_hypotheses": copy.deepcopy(
                                    list(summary_state.get("rejected_hypotheses") or [])
                                ),
                                "tried_action_signatures": copy.deepcopy(
                                    list(
                                        summary_state.get("tried_action_signatures")
                                        or []
                                    )
                                ),
                                "unresolved_signatures": copy.deepcopy(
                                    list(
                                        summary_state.get("unresolved_signatures") or []
                                    )
                                ),
                                "remaining_budget": summary_state.get(
                                    "remaining_budget"
                                ),
                            }
                        except Exception as state_exc:  # noqa: BLE001
                            diagnostic["state_error"] = (
                                f"{type(state_exc).__name__}: {state_exc}"
                            )
                        self.unverified_expert_labels.append(diagnostic)
                        break
                    training_decision_evidence_verified = True
                offline_target_audit: dict[str, Any] | None = None
                if (
                    self.supervision_policy
                    == DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                ):
                    # The target is already fixed from PolicyObservation.  Only
                    # now may the collector inspect private physical truth, and
                    # the lossy result is used solely to admit or quarantine the
                    # row; it cannot feed back into target selection or policy
                    # input.
                    offline_target_audit = offline_teacher_target_audit(
                        preferred_action=preferred_action,
                        oracle_state=oracle_state,
                        policy_observation=policy_observation,
                        scenario=offline_audit_scenario,
                        env=self.env,
                        observable_evidence_passed=(
                            training_decision_evidence_verified
                        ),
                    )
                # This is the concurrency barrier.  Policy completion (or its
                # normalized policy-exception action) is observed before the
                # beta RNG advances and before the transition boundary
                # (current_state/step) can proceed.  Read-only oracle/audit
                # state access is permitted inside the overlap.  With no
                # executor, preserve the original sequential path byte-for-byte.
                model_action = (
                    self._policy_action(observation_dict)
                    if policy_action_future is None
                    else policy_action_future.result()
                )

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
                next_policy_observation = replace(
                    self._policy_observation(provisional_history), remaining_budget=max_steps - step - 1
                )
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
                final_next_policy_observation = replace(
                    self._policy_observation(next_history), remaining_budget=max_steps - step - 1
                )
                final_next_oracle_state: OracleState | Mapping[str, Any] | None = None
                if (
                    self.supervision_policy
                    != DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                ):
                    final_next_oracle_state = self._oracle_state(
                        next_history, final_next_policy_observation
                    )
                next_valid_actions = []
                if not self.env.is_terminal(next_state):
                    next_valid_actions = self._select_expert_actions(
                        policy_observation=final_next_policy_observation,
                        oracle_state=final_next_oracle_state,
                        history=next_history,
                    )
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
                state_origin = (
                    "learner_policy"
                    if state_visited_by == "model"
                    else "expert_policy"
                    if state_visited_by == "expert"
                    else "initial"
                )
                dataset_source = "dagger_rollout"
                dagger1_production_eligible: bool | None = None
                dagger1_ineligibility_reason: str | None = None
                recovery_stratum: str | None = None
                if (
                    self.supervision_policy
                    == DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                ):
                    recovery_stratum = classify_dagger1_recovery_stratum(
                        observation_dict,
                        preferred_action=preferred_action,
                        state_class=state_class,
                        scenario_family=scenario_family,
                        error_cardinality=error_cardinality,
                    )
                    if collection_role == "diagnostic":
                        dagger1_ineligibility_reason = (
                            "diagnostic_beta_zero_not_training_eligible"
                        )
                    elif state_origin != "learner_policy":
                        dagger1_ineligibility_reason = "not_learner_visited_state"
                    elif (
                        recovery_stratum
                        not in _DAGGER1_PRODUCTION_RECOVERY_STRATA
                    ):
                        dagger1_ineligibility_reason = "not_recovery_state"
                    elif preferred_action is None:
                        dagger1_ineligibility_reason = "missing_expert_target"
                    elif not training_decision_evidence_verified:
                        dagger1_ineligibility_reason = (
                            "training_decision_evidence_not_verified"
                        )
                    elif rank_one_target_proof.get("passed") is not True:
                        dagger1_ineligibility_reason = (
                            "expert_target_not_observably_rank_one"
                        )
                    elif not isinstance(offline_target_audit, Mapping) or (
                        offline_target_audit.get("passed") is not True
                    ):
                        dagger1_ineligibility_reason = (
                            "offline_teacher_target_audit_failed"
                        )
                    else:
                        dagger1_production_eligible = True
                    if dagger1_production_eligible is not True:
                        dagger1_production_eligible = False
                example = {
                    "example_id": f"dagger_iter{iteration}_{scenario_id}_step{step}",
                    "scenario_id": scenario_id,
                    "root_scenario_id": root_scenario_id,
                    "physical_root_fingerprint": physical_root_fingerprint,
                    "scenario_family": scenario_family,
                    "error_cardinality": error_cardinality,
                    "parameter_scans_available": grouping.get(
                        "parameter_scans_available"
                    ),
                    "network_case": network_case,
                    "source_tier": source_tier,
                    "dataset_split": (
                        grouping.get("dataset_split") or grouping.get("split")
                    ),
                    "episode_id": observation_dict.get("episode_id"),
                    "iteration": iteration,
                    "collection_beta": float(beta),
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
                    "observable_rank_one_target_proof": policy_safe_copy(
                        rank_one_target_proof
                    ),
                    "model_action": policy_safe_copy(model_action),
                    "executed_action": policy_safe_copy(executed_action),
                    "executed_by": executed_by,
                    "state_visited_by": state_visited_by,
                    "state_origin": state_origin,
                    "dataset_source": dataset_source,
                    "collection_role": collection_role,
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
                        "state_origin": state_origin,
                        "dataset_source": dataset_source,
                        "collection_role": collection_role,
                        "collection_beta": float(beta),
                        "supervision_policy": self.supervision_policy,
                        "training_decision_evidence_verified": (
                            training_decision_evidence_verified
                        ),
                        "observable_rank_one_target_proof": policy_safe_copy(
                            rank_one_target_proof
                        ),
                        "deferred_expert_action_count": len(
                            deferred_expert_actions
                        ),
                    },
                }
                if offline_target_audit is not None:
                    # Root-level collection metadata is deliberately not part
                    # of ``labels`` or any model-visible observation/message.
                    example["offline_teacher_target_audit"] = copy.deepcopy(
                        offline_target_audit
                    )
                if dagger1_production_eligible is not None:
                    example["production_label_eligible"] = (
                        dagger1_production_eligible
                    )
                    example["recovery_label_contract"] = (
                        "observable_rank_one_learner_state_v1"
                    )
                    example["labels"]["production_label_eligible"] = (
                        dagger1_production_eligible
                    )
                    example["labels"]["recovery_label_contract"] = (
                        "observable_rank_one_learner_state_v1"
                    )
                    example["recovery_stratum"] = recovery_stratum
                    example["labels"]["recovery_stratum"] = recovery_stratum
                    if dagger1_ineligibility_reason is not None:
                        example["production_label_ineligibility_reason"] = (
                            dagger1_ineligibility_reason
                        )
                        example["labels"][
                            "production_label_ineligibility_reason"
                        ] = dagger1_ineligibility_reason
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
            oracle = self.env.get_oracle_state(history)
            if isinstance(oracle, OracleState):
                return replace(oracle, policy_observation=policy_observation)
            if isinstance(oracle, Mapping):
                oracle = copy.deepcopy(dict(oracle))
                if "policy_observation" in oracle:
                    oracle["policy_observation"] = policy_observation.as_dict()
                else:
                    oracle["remaining_budget"] = policy_observation.remaining_budget
            return oracle
        return policy_observation.as_dict()

    def _select_expert_actions(
        self,
        *,
        policy_observation: PolicyObservation,
        oracle_state: OracleState | Mapping[str, Any] | None,
        history: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.supervision_policy == DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION:
            # D1 labels must be a function of exactly what the learner sees.
            # In particular, never pass the collector's longer private
            # transition history into either teacher path: transition labels
            # are produced after an OracleState exists and may contain fields
            # outside the bounded PolicyObservation history window.
            # One shared selector serves the release policy wrapper, this
            # collector, and the recovery-probe generator.  Keeping three
            # copies let the probe path drift onto the raw rule expert with an
            # unbounded history, which stalled at verified candidates and gave
            # the expert evidence the learner could not see.
            from psse_env.dagger.release_factories import (
                select_observable_expert_actions,
            )

            selection = select_observable_expert_actions(
                policy_observation=policy_observation.as_dict(),
                expert_oracle=self.expert_oracle,
            )
            return [copy.deepcopy(action) for action in selection.actions]
        else:
            if oracle_state is None:
                raise RuntimeError("non-D1 expert selection requires oracle state")
            raw_actions = self.expert_oracle.next_actions(oracle_state, history)
        normalized = [safe_normalize_action(action) for action in raw_actions]
        return [
            action for action in normalized if action["tool"] != "__invalid_action__"
        ]

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
