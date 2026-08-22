"""Reproducible, truth-isolated closed-loop recovery evaluation.

The evaluator deliberately has two phases at every step:

1. build and validate a policy-only observation, then ask the policy to act;
2. after the action is fixed, inspect oracle state for offline safety scoring.

Consequently, scenario truth, candidate disposition, cost labels, and physical
audit callbacks are never passed to the policy.  The same evaluator can be
used for a rule policy, a base model, or a trained checkpoint by changing only
``policy_factory``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    CORRECTION_TOOLS,
    DIAGNOSTIC_TOOLS,
    FINALIZE_DIAGNOSIS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    INVALID_ACTION,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_WLS,
    VERIFY_CANDIDATE,
    action_signature,
    invalid_action,
    safe_normalize_action,
)
from psse_env.dagger.dataset_builder import (
    TOOL_JSON_SCHEMAS,
    find_forbidden_policy_paths,
    find_forbidden_provenance_paths,
    validate_policy_payload,
)
from psse_env.dagger.release_audit import (
    ACCEPTED_TARGET_NONREGRESSION_CHECK,
    ACCEPTED_TARGETS_CHECK,
    DIAGNOSTIC_FAMILY_CHECK,
    FINAL_CASE_CHECK,
    FINAL_MEASUREMENTS_CHECK,
    HEALTHY_CASE_CHECK,
    HEALTHY_MEASUREMENTS_CHECK,
    REMAINING_FAULTS_CHECK,
    audit_post_correction_controller_handoff,
    audit_episode_against_truth as strict_audit_episode_against_truth,
    validate_post_correction_handoff_assessment,
)
from psse_env.state_store import OracleState, PolicyObservation, policy_safe_copy
from psse_env.sft.provenance import file_sha256, git_source_state, stable_json_sha256
from psse_env.sft.release_hardware import normalize_accelerator_class


STUDY_DEVELOPMENT_HOLDOUT_PROVENANCE_CONTRACT = (
    "dagger1_development_holdout_study_provenance_v1"
)
STUDY_OBJECTIVE_EPISODE_EVIDENCE_CONTRACT = (
    "dagger_study_objective_episode_evidence_v1"
)
STUDY_OBJECTIVE_ACTION_ASSESSMENT_CONTRACT = (
    "dagger_study_objective_action_assessment_v1"
)
STUDY_OBJECTIVE_TOOL_EVIDENCE_CONTRACT = (
    "dagger_study_objective_tool_evidence_v1"
)
STUDY_EVALUATION_SCHEMA_VERSION = 4
STUDY_POLICY_HISTORY_WINDOW = 4


def study_objective_episode_evidence_marker() -> dict[str, Any]:
    """Return the exact schema-v4 objective-evidence capability marker."""

    return {
        "contract": STUDY_OBJECTIVE_EPISODE_EVIDENCE_CONTRACT,
        "policy_observations_persisted": True,
        "objective_action_assessments_persisted": True,
        "objective_tool_evidence_persisted": True,
        "policy_tool_outputs_persisted": True,
    }


@dataclass(frozen=True)
class RecoveryMetrics:
    """Aggregate rates used for checkpoint selection.

    Existing fields retain their original meaning.  The additional fields make
    terminal handoff, physical correctness, and operational efficiency visible
    without requiring callers to unpack ``suite_metrics``.
    """

    final_physical_success: float = 0.0
    false_finalization: float = 0.0
    healthy_component_corruption: float = 0.0
    forced_error_recovery: float = 0.0
    tool_regret: float = 0.0
    partial_success_retention: float = 0.0
    false_rollback: float = 0.0
    false_commit: float = 0.0
    loop_rate: float = 0.0
    final_physical_correctness: float = 0.0
    terminal_rate: float = 0.0
    resolution_rate: float = 0.0
    operator_escalation_rate: float = 0.0
    audited_post_correction_handoff: float = 0.0
    audited_completion: float = 0.0
    unqualified_operator_escalation: float = 0.0
    healthy_component_preservation: float = 0.0
    invalid_action_recovery: float = 0.0
    mean_wls_calls: float = 0.0
    mean_specialized_tool_calls: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "final_physical_success": 4.0,
    "false_finalization": -5.0,
    "healthy_component_corruption": -5.0,
    "forced_error_recovery": 2.0,
    "tool_regret": -0.25,
    "partial_success_retention": 2.0,
    "false_rollback": -2.0,
    "false_commit": -4.0,
    "loop_rate": -2.0,
}


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    metrics: RecoveryMetrics
    suite_metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "metrics": self.metrics.as_dict(),
            "suite_metrics": copy.deepcopy(self.suite_metrics),
        }


@dataclass(frozen=True)
class EpisodeEvaluation:
    """Serializable record for one closed-loop physical root."""

    episode_key: str
    scenario_id: str
    suite: str
    family: str
    cardinality: int | str
    case: str
    split: str
    source_tier: str
    physical_root: str
    seed: int
    steps: int
    policy_steps: int
    terminal: bool
    terminal_outcome: str | None
    final_physical_correct: bool
    physical_correctness_known: bool
    final_physical_success: bool
    healthy_components_preserved: bool
    healthy_preservation_known: bool
    false_commit_count: int
    false_rollback_count: int
    false_finalization_count: int
    partial_fix_count: int
    retained_partial_fix_count: int
    invalid_action_count: int
    recovered_invalid_action_count: int
    loop_detected: bool
    wls_calls: int
    specialized_tool_calls: int
    tool_counts: dict[str, int]
    specialized_tool_counts: dict[str, int]
    tool_regret_total: float
    tool_regret_samples: int
    evaluation_intervention: dict[str, Any]
    release_environment_attestation: dict[str, Any] = field(default_factory=dict)
    policy_identity_attestation: dict[str, Any] = field(default_factory=dict)
    objective_evidence: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    evaluator_error: str | None = None
    control_quarantine: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def recovery_score(
    metrics: RecoveryMetrics | Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    values = metrics.as_dict() if isinstance(metrics, RecoveryMetrics) else dict(metrics)
    score_weights = dict(weights or DEFAULT_SCORE_WEIGHTS)
    return sum(
        float(values.get(key, 0.0) or 0.0) * weight
        for key, weight in score_weights.items()
    )


def make_evaluation_result(
    metrics: RecoveryMetrics | Mapping[str, Any],
    *,
    suite_metrics: Mapping[str, Any] | None = None,
    weights: Mapping[str, float] | None = None,
) -> EvaluationResult:
    typed = (
        metrics
        if isinstance(metrics, RecoveryMetrics)
        else RecoveryMetrics(
            **{
                key: float(value)
                for key, value in metrics.items()
                if key in RecoveryMetrics.__dataclass_fields__
            }
        )
    )
    return EvaluationResult(
        score=recovery_score(typed, weights=weights),
        metrics=typed,
        suite_metrics=dict(suite_metrics or {}),
    )


EVALUATION_SUITES = (
    "standard_success",
    "forced_error_recovery",
    "partial_success_retention",
    "invalid_action_recovery",
    "efficiency",
)
_DIAGNOSTIC_DEVELOPMENT_SUITE = "dagger1_development"


PhysicalAudit = Callable[[Mapping[str, Any]], Mapping[str, Any] | bool]
ToolCostResolver = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
CaseLoader = Callable[[Any], Any]
ProgressCallback = Callable[[Mapping[str, Any]], None]


_OFFLINE_EXECUTION_KEYS = frozenset(
    {
        "action_cost",
        "action_costs",
        "admissible_actions",
        "best_cost",
        "candidate_assessment",
        "candidate_disposition",
        "chosen_cost",
        "cost_margin",
        "cost_to_go",
        "costs",
        "cost_label",
        "cost_labels",
        "data_source_tier",
        "dataset_split",
        "detected",
        "detected_top1",
        "detected_top3",
        "evaluation_labels",
        "executed_cost",
        "expected_final_state",
        "expert_cost",
        "final_physical_state",
        "final_state",
        "final_states",
        "hidden_truth",
        "initial_states",
        "ground_truth",
        "label",
        "labels",
        "minimum_cost",
        "min_cost",
        "oracle_action_hints",
        "oracle_action",
        "optimal_cost",
        "oracle_cost_to_go",
        "preferred_action",
        "progress_class",
        "q_cost",
        "q_costs",
        "release_audit",
        "suggested_actions",
        "recommended_action",
        "ranked_actions",
        "target_fixed",
        "target_action",
        "teacher_action",
        "expert_action",
        "gold_action",
        "tool_cost",
        "tool_costs",
        "tool_cost_labels",
        "valid_actions",
        "valid_action",
        "valid_next_actions",
        "valid_next_action",
        "remaining_true_faults",
        "remaining_true_fault_count",
        "remaining_fault_count",
        "final_remaining",
        "truth_complete",
        "truth",
        "scenario_family",
        "error_family",
        "error_family_combination",
        "error_cardinality",
        "cardinality",
        "network_case",
        "case_id",
        "physical_root",
        "physical_root_fingerprint",
        "root_scenario_id",
        "split",
        "source_tier",
    }
)
_SCENARIO_SCHEMA_VERSION = 1
_INTERVENTION_SCHEMA_VERSION = 1
_SCENARIO_PARTITION_KEYS = frozenset(
    {"scenario_schema_version", "execution", "audit", "grouping"}
)
_SCENARIO_PARTITION_MARKERS = _SCENARIO_PARTITION_KEYS
_EXECUTION_METADATA_KEYS = frozenset(
    {
        "semantic_field_provenance",
        "unresolved_signatures",
        "remaining_anomaly_score",
        "no_material_anomaly_remaining",
        "requires_measurement_context",
        "measurement_covariance",
        "slack_bus",
        "pristine_model_dir",
        "faulted_model_dir",
        "load_scale",
        "parameter_scans",
        "harmonic_measurements",
        "harmonic_orders",
        "nlm_diagnostic",
        "hif_runtime",
        "hif_scan_window",
        "three_phase_voltages",
    }
)
_PHYSICAL_AUDIT_OVERRIDE_KEYS = frozenset(
    {
        "final_physical_correct",
        "physical_correctness_known",
        "healthy_components_preserved",
        "healthy_preservation_known",
        "partial_fixes_retained",
    }
)
_EXECUTION_SCENARIO_KEYS = frozenset(
    {
        "scenario_id",
        "id",
        "episode_id",
        "case",
        "case_path",
        "measurements",
        "z_obs",
        "metadata",
        "semantic_field_provenance",
        "unresolved_signatures",
        "remaining_anomaly_score",
        "no_material_anomaly_remaining",
        "requires_measurement_context",
        # Explicit test/development adapters may initialize their own state and
        # scripted transitions, but these fields contain no audit reference.
        "initial_physical_state",
        "script",
    }
)
_GROUPING_SCENARIO_KEYS = frozenset(
    {
        "root_scenario_id",
        "physical_root_fingerprint",
        "scenario_family",
        "error_cardinality",
        "case_id",
        "split",
        "source_tier",
    }
)
_REQUIRED_EXECUTION_SCENARIO_KEYS = frozenset(
    {"scenario_id", "case", "measurements"}
)
_REQUIRED_GROUPING_SCENARIO_KEYS = frozenset(
    {
        "physical_root_fingerprint",
        "scenario_family",
        "error_cardinality",
        "case_id",
        "split",
        "source_tier",
    }
)
_REQUIRED_RELEASE_ENVIRONMENT = {
    "production_dataset_mode": True,
    "candidate_quality_oracle_mode": "deployment",
}

_EXPECTED_INTERVENTION_KIND = {
    "standard_success": "none",
    "forced_error_recovery": "pre_policy_failure",
    "partial_success_retention": "committed_partial_correction",
    "invalid_action_recovery": "pre_policy_failure",
    "efficiency": "efficiency_budget",
}
_EFFICIENCY_LIMIT_FIELDS = frozenset(
    {
        "maximum_policy_steps",
        "maximum_wls_calls",
        "maximum_specialized_tool_calls",
    }
)
_REPEATED_DIAGNOSTIC_CIRCUIT_BREAKER = (
    "evaluation_repeated_nonadvancing_diagnostic"
)
_SPECIALIZED_TOOL_BUDGET_CIRCUIT_BREAKER = (
    "evaluation_specialized_tool_budget_exhausted"
)
_REPEATED_NONADVANCING_FAILURE_CIRCUIT_BREAKER = (
    "evaluation_repeated_nonadvancing_failure"
)
# Once an exact target failure or a family-wide state-contract failure has
# occurred without an observable state advance, another action in that same
# deterministic scope cannot help.  The repeated attempt remains represented
# in the trace and policy metrics, but the environment/provider is called at
# most once per scope.
_MAX_DETERMINISTIC_FAILURE_EXECUTIONS_PER_SCOPE = 1
_REJECTED_ESCALATION_ERROR_CODES = frozenset(
    {
        "candidate_lifecycle_violation",
        "operator_escalation_precondition_not_met",
        "operator_escalation_request_unsupported",
        "recovery_evidence_inventory_incomplete",
    }
)
_REJECTED_COMMIT_ERROR_CODES = frozenset(
    {
        "candidate_lifecycle_violation",
        "state_reference_mismatch",
    }
)
_DETERMINISTIC_CORRECTION_FAILURE_KINDS = {
    "correction_not_supported_by_current_context": "unsupported_correction",
    "correction_route_not_actionable": "correction_route_not_actionable",
    "parameter_scans_missing": "parameter_scans_missing",
    "post_correction_confirmation_required": "unsupported_correction",
    "topology_correction_unsupported": "unsupported_correction",
}
_FAMILY_WIDE_CORRECTION_FAILURE_CODES = frozenset(
    {
        "correction_route_not_actionable",
        "parameter_scans_missing",
        "post_correction_confirmation_required",
        "topology_correction_unsupported",
    }
)


def _deterministic_nonadvancing_failure_kind(
    *,
    tool: str,
    execution_status: str,
    error_code: Any,
) -> str | None:
    """Classify only failures whose identical retry cannot change the result."""

    if execution_status != "failure":
        return None
    normalized_error = str(error_code or "").strip()
    if tool == INVALID_ACTION:
        return "schema_invalid_action"
    if (
        tool == ASK_FOR_MORE_EVIDENCE
        and normalized_error in _REJECTED_ESCALATION_ERROR_CODES
    ):
        return "rejected_operator_escalation"
    if tool == COMMIT_STATE and normalized_error in _REJECTED_COMMIT_ERROR_CODES:
        return "rejected_commit"
    if tool in CORRECTION_TOOLS:
        return _DETERMINISTIC_CORRECTION_FAILURE_KINDS.get(normalized_error)
    return None


def _deterministic_failure_storage_key(
    *,
    tool: str,
    signature: str,
    error_code: Any,
) -> str:
    """Scope state-contract failures to a family and target failures exactly."""

    normalized_error = str(error_code or "").strip()
    if (
        tool in CORRECTION_TOOLS
        and normalized_error in _FAMILY_WIDE_CORRECTION_FAILURE_CODES
    ):
        return f"{tool}:family_wide_nonadvancing_failure"
    return signature


def _deterministic_failure_lookup_keys(
    *,
    tool: str,
    signature: str,
) -> tuple[str, ...]:
    if tool in CORRECTION_TOOLS:
        # A family-wide executor/route failure dominates any older
        # target-specific record for the same correction family.
        return (f"{tool}:family_wide_nonadvancing_failure", signature)
    return (signature,)


def _normalized_key(key: Any) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip())
    return re.sub(r"[^a-zA-Z0-9]+", "_", separated).strip("_").lower()


def _offline_execution_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    return bool(
        normalized in _OFFLINE_EXECUTION_KEYS
        or normalized.startswith(
            (
                "true_",
                "clean_",
                "expected_",
                "gold_final_",
                "ground_truth_",
                "oracle_action_",
                "expert_action_",
                "teacher_action_",
                "target_action_",
                "release_audit_",
                "action_cost",
                "q_value",
                "reference_solution",
            )
        )
        or normalized.startswith(
            (
                "final_",
                "recommended_action",
                "oracle_action",
                "expert_action",
                "teacher_action",
                "target_action",
                "gold_action",
                "valid_action",
                "correct_action",
                "optimal_action",
            )
        )
        or normalized.endswith(
            (
                "_clean",
                "_truth",
                "_ground_truth",
                "_label",
                "_labels",
                "_cost",
                "_costs",
                "_cost_label",
                "_cost_labels",
            )
        )
    )


def _has_partition_marker(scenario: Mapping[str, Any]) -> bool:
    return bool(
        {_normalized_key(key) for key in scenario} & _SCENARIO_PARTITION_MARKERS
    )


def _normalized_mapping_value(
    source: Mapping[str, Any], canonical_key: str, *, label: str
) -> tuple[bool, Any]:
    matches = [
        (str(key), value)
        for key, value in source.items()
        if _normalized_key(key) == canonical_key
    ]
    if not matches:
        return False, None
    value = copy.deepcopy(matches[0][1])
    for alias, candidate in matches[1:]:
        if candidate != value:
            raise ValueError(
                f"conflicting normalized field {canonical_key!r} in {label}: "
                + ", ".join(name for name, _ in matches)
            )
    return True, value


def privileged_execution_paths(value: Any, *, path: str = "$") -> list[str]:
    """Return recursively discovered audit-only fields in an execution payload."""

    if isinstance(value, Mapping):
        found: list[str] = []
        for key, item in value.items():
            child = f"{path}.{key}"
            if _offline_execution_key(key):
                found.append(child)
            found.extend(privileged_execution_paths(item, path=child))
        return found
    if isinstance(value, (list, tuple)):
        return [
            leaked
            for index, item in enumerate(value)
            for leaked in privileged_execution_paths(item, path=f"{path}[{index}]")
        ]
    return []


def policy_payload_leakage_paths(value: Any, *, path: str = "$") -> list[str]:
    """Return every privileged field/provenance path in a policy payload.

    ``validate_policy_payload`` remains the live fail-closed boundary.  This
    companion produces deterministic, countable study evidence so ingestion
    recomputes leakage from the persisted payload instead of trusting a
    reported zero count.
    """

    # The shared live boundary normalizes case and separators, so this count
    # uses the exact same denylist semantics as policy execution.
    found = list(find_forbidden_policy_paths(value, prefix=path))

    def embedded_provenance(item: Any, *, prefix: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                child_path = f"{prefix}.{key}"
                if "provenance" in str(key).lower():
                    found.extend(
                        find_forbidden_provenance_paths(
                            child, prefix=child_path
                        )
                    )
                else:
                    embedded_provenance(child, prefix=child_path)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                embedded_provenance(child, prefix=f"{prefix}[{index}]")

    embedded_provenance(value, prefix=path)
    return sorted(set(found))


_OBJECTIVE_RECOVERY_STRATUM_ALIASES = {
    "post_failure_no_candidate": "post_failure_no_candidate",
    "unsupported_correction_recovery": "unsupported_correction_recovery",
    "premature_commit_recovery": "premature_commit_recovery",
    "premature_escalation_recovery": "premature_escalation_recovery",
    "rejected_candidate_rollback": "rejected_candidate_rollback",
    "sequential_measurement_parameter_recovery": (
        "measurement_parameter_sequential_handoff"
    ),
}
_OBJECTIVE_OPERATOR_HANDOFF_REQUESTS = frozenset(
    {
        HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
        RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    }
)


def objective_recovery_action_assessment(
    observation: Mapping[str, Any],
    *,
    scenario_family: str,
    error_cardinality: Any,
    partial_success_opportunity: bool = False,
) -> dict[str, Any]:
    """Independently score one policy-visible state with the canonical expert.

    The assessment deliberately runs only on the exact observation already
    fixed for the learner.  It neither receives the environment nor consults
    scenario/oracle truth.  Persisting the observation beside this assessment
    lets study ingestion reproduce both every opportunity denominator and the
    exact expected action.
    """

    payload = copy.deepcopy(dict(observation))
    leakage = policy_payload_leakage_paths(payload)
    if leakage:
        return {
            "contract": STUDY_OBJECTIVE_ACTION_ASSESSMENT_CONTRACT,
            "evidence_available": False,
            "evidence_failure": "policy_payload_contains_privileged_evidence",
            "policy_payload_leakage_paths": leakage,
            "canonical_selector": "observable_expert_selection_v1",
            "selector_basis": None,
            "canonical_action_count": None,
            "expected_action": None,
            "recovery_stratum": None,
            "operator_handoff_opportunity": None,
        }
    try:
        parsed_cardinality = int(error_cardinality)
    except (TypeError, ValueError, OverflowError):
        parsed_cardinality = -1
    if parsed_cardinality < 0:
        return {
            "contract": STUDY_OBJECTIVE_ACTION_ASSESSMENT_CONTRACT,
            "evidence_available": False,
            "evidence_failure": "public_error_cardinality_unavailable",
            "policy_payload_leakage_paths": [],
            "canonical_selector": "observable_expert_selection_v1",
            "selector_basis": None,
            "canonical_action_count": None,
            "expected_action": None,
            "recovery_stratum": None,
            "operator_handoff_opportunity": None,
        }
    validate_policy_payload(payload)

    # Local imports avoid making the release-factory module depend on the
    # evaluator while still reusing its single reviewed observable selector.
    from psse_env.dagger.release_factories import (
        select_observable_expert_actions,
    )
    from psse_env.dagger.rollout_collector import (
        classify_dagger1_recovery_stratum,
    )
    from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle

    expert = ExpertPolicyOracle(
        process_oracle=ProcessValidityOracle(
            executor_hydrated_corrections=True
        )
    )
    selection = select_observable_expert_actions(
        policy_observation=payload,
        expert_oracle=expert,
    )
    expected = (
        copy.deepcopy(selection.preferred_action)
        if selection.preferred_action is not None
        else None
    )
    classified = classify_dagger1_recovery_stratum(
        payload,
        preferred_action=expected,
        state_class="study_objective_audit",
        scenario_family=str(scenario_family),
        error_cardinality=parsed_cardinality,
    )
    recovery_stratum = (
        "safe_continuation_after_partial_success"
        if partial_success_opportunity
        else _OBJECTIVE_RECOVERY_STRATUM_ALIASES.get(str(classified))
    )
    expected_arguments = (
        expected.get("arguments") if isinstance(expected, Mapping) else None
    )
    handoff_opportunity = bool(
        isinstance(expected, Mapping)
        and expected.get("tool") == ASK_FOR_MORE_EVIDENCE
        and isinstance(expected_arguments, Mapping)
        and expected_arguments.get("request")
        in _OBJECTIVE_OPERATOR_HANDOFF_REQUESTS
    )
    evidence_available = expected is not None
    return {
        "contract": STUDY_OBJECTIVE_ACTION_ASSESSMENT_CONTRACT,
        "evidence_available": evidence_available,
        "evidence_failure": (
            None
            if evidence_available
            else "canonical_observable_expert_returned_no_action"
        ),
        "policy_payload_leakage_paths": [],
        "canonical_selector": "observable_expert_selection_v1",
        "selector_basis": selection.selection_basis,
        "canonical_action_count": len(selection.actions),
        "expected_action": expected,
        "recovery_stratum": recovery_stratum,
        "operator_handoff_opportunity": handoff_opportunity,
    }


_OBJECTIVE_TOOL_METRIC_FIELDS = (
    "state_id",
    "state_hash",
    "evidence_source",
    "chi_square_statistic",
    "chi_square_threshold",
    "max_normalized_residual",
    "no_material_anomaly_remaining",
    "globally_resolved",
    "physical_constraints_ok",
    "physical_evidence_scope",
    "physical_evidence_complete",
    "physical_bound_violations",
    "steady_state_physical_evidence",
    "power_flow_converged",
    "topology_feasible",
)


def objective_tool_evidence(
    action: Mapping[str, Any], output: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Persist the narrow observable WLS/verification evidence study metrics use."""

    tool = str(action.get("tool") or "")
    if tool not in {RUN_WLS, VERIFY_CANDIDATE}:
        return None
    metrics = output.get("tool_metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    return {
        "contract": STUDY_OBJECTIVE_TOOL_EVIDENCE_CONTRACT,
        "tool": tool,
        **{
            field_name: copy.deepcopy(metrics.get(field_name))
            for field_name in _OBJECTIVE_TOOL_METRIC_FIELDS
        },
    }


def _partitioned_scenario_parts(
    scenario: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(scenario) != _SCENARIO_PARTITION_KEYS:
        missing = sorted(_SCENARIO_PARTITION_KEYS - set(scenario))
        extra = sorted(set(scenario) - _SCENARIO_PARTITION_KEYS)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(str(item) for item in extra))
        raise ValueError(
            "partitioned scenario must contain exactly scenario_schema_version, "
            "execution, audit, and grouping: " + "; ".join(details)
        )
    version = scenario.get("scenario_schema_version")
    if type(version) is not int or version != _SCENARIO_SCHEMA_VERSION:
        raise ValueError(
            f"partitioned scenario_schema_version must be {_SCENARIO_SCHEMA_VERSION}"
        )
    parts: list[dict[str, Any]] = []
    for name in ("execution", "audit", "grouping"):
        value = scenario.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"partitioned scenario {name} must be a mapping")
        parts.append(copy.deepcopy(dict(value)))
    execution, audit, grouping = parts
    unexpected_execution = sorted(set(execution) - _EXECUTION_SCENARIO_KEYS)
    if unexpected_execution:
        raise ValueError(
            "partitioned scenario execution contains unsupported fields: "
            + ", ".join(str(item) for item in unexpected_execution)
        )
    unexpected_grouping = sorted(set(grouping) - _GROUPING_SCENARIO_KEYS)
    if unexpected_grouping:
        raise ValueError(
            "partitioned scenario grouping contains unsupported fields: "
            + ", ".join(str(item) for item in unexpected_grouping)
        )
    leaked = privileged_execution_paths(execution)
    if leaked:
        raise ValueError(
            "partitioned scenario execution contains audit-only fields: "
            + ", ".join(leaked)
        )
    metadata = execution.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("partitioned scenario execution.metadata must be a mapping")
        unexpected_metadata = sorted(set(metadata) - _EXECUTION_METADATA_KEYS)
        if unexpected_metadata:
            raise ValueError(
                "partitioned scenario execution.metadata contains unsupported fields: "
                + ", ".join(str(item) for item in unexpected_metadata)
            )
    _, truth = _normalized_mapping_value(audit, "truth", label="audit")
    if truth is not None and not isinstance(truth, Mapping):
        raise ValueError("partitioned scenario audit.truth must be a mapping")
    execution_keys = {_normalized_key(key) for key in execution}
    grouping_keys = {_normalized_key(key) for key in grouping}
    audit_keys = {
        _normalized_key(key)
        for key in audit
        if _normalized_key(key) != "truth"
    }
    truth_keys = (
        {_normalized_key(key) for key in truth}
        if isinstance(truth, Mapping)
        else set()
    )
    collisions = sorted(
        (audit_keys & (execution_keys | grouping_keys))
        | (truth_keys & (execution_keys | grouping_keys | audit_keys))
    )
    if collisions:
        raise ValueError(
            "partitioned scenario audit fields collide with execution/grouping: "
            + ", ".join(str(item) for item in collisions)
        )
    missing_execution = sorted(_REQUIRED_EXECUTION_SCENARIO_KEYS - set(execution))
    if missing_execution:
        raise ValueError(
            "partitioned scenario execution is missing required fields: "
            + ", ".join(missing_execution)
        )
    missing_grouping = sorted(_REQUIRED_GROUPING_SCENARIO_KEYS - set(grouping))
    if missing_grouping:
        raise ValueError(
            "partitioned scenario grouping is missing required fields: "
            + ", ".join(missing_grouping)
        )
    if not str(execution.get("scenario_id") or "").strip():
        raise ValueError("partitioned scenario execution.scenario_id must be non-empty")
    measurements = execution.get("measurements")
    if not isinstance(measurements, Sequence) or isinstance(measurements, (str, bytes)):
        raise ValueError("partitioned scenario execution.measurements must be a sequence")
    cardinality = grouping.get("error_cardinality")
    if isinstance(cardinality, bool) or not isinstance(cardinality, int) or cardinality < 0:
        raise ValueError(
            "partitioned scenario grouping.error_cardinality must be a non-negative integer"
        )
    for key in (
        "physical_root_fingerprint",
        "scenario_family",
        "case_id",
        "split",
        "source_tier",
    ):
        if not isinstance(grouping.get(key), str) or not grouping[key].strip():
            raise ValueError(f"partitioned scenario grouping.{key} must be non-empty")
    return execution, audit, grouping


def evaluation_intervention_contract(
    suite: str,
    scenario: Mapping[str, Any],
    *,
    required: bool = True,
    allow_diagnostic_development: bool = False,
) -> dict[str, Any] | None:
    """Return the canonical policy-hidden intervention for one suite episode.

    Interventions live only in the audit partition.  They are therefore
    included in the frozen suite identity but never enter ``env.reset`` or a
    policy observation.  Their observable effects (a prior failed transition
    or a genuinely committed partial correction) are introduced by the
    evaluator before the first policy call.
    """

    normalized_suite = str(suite).strip()
    expected_kind = _EXPECTED_INTERVENTION_KIND.get(normalized_suite)
    if (
        expected_kind is None
        and allow_diagnostic_development
        and normalized_suite == _DIAGNOSTIC_DEVELOPMENT_SUITE
    ):
        expected_kind = "none"
    if expected_kind is None:
        raise ValueError(f"unsupported evaluation suite {normalized_suite!r}")
    if not _has_partition_marker(scenario):
        if required:
            raise ValueError("evaluation intervention requires a partitioned scenario")
        return None
    _, audit, _ = _partitioned_scenario_parts(scenario)
    aliases = [
        str(key)
        for key in audit
        if _normalized_key(key) == "evaluation_intervention"
    ]
    if aliases != ["evaluation_intervention"]:
        if not aliases and not required:
            return None
        raise ValueError(
            "scenario audit must contain exactly the canonical "
            "evaluation_intervention field"
        )
    raw = audit.get("evaluation_intervention")
    if not isinstance(raw, Mapping):
        raise ValueError("audit.evaluation_intervention must be a mapping")
    intervention = copy.deepcopy(dict(raw))
    if (
        type(intervention.get("intervention_schema_version")) is not int
        or intervention.get("intervention_schema_version")
        != _INTERVENTION_SCHEMA_VERSION
    ):
        raise ValueError("evaluation intervention schema version must be exactly 1")
    if intervention.get("kind") != expected_kind:
        raise ValueError(
            f"suite {normalized_suite!r} requires intervention kind {expected_kind!r}"
        )

    if expected_kind == "none":
        expected_fields = {"intervention_schema_version", "kind"}
    elif expected_kind == "pre_policy_failure":
        expected_fields = {
            "intervention_schema_version",
            "kind",
            "failure_mode",
            "error_code",
        }
        expected_mode = (
            "well_formed"
            if normalized_suite == "forced_error_recovery"
            else "malformed"
        )
        expected_code = (
            "injected_transient_tool_failure"
            if expected_mode == "well_formed"
            else "injected_invalid_action"
        )
        if intervention.get("failure_mode") != expected_mode:
            raise ValueError(
                f"suite {normalized_suite!r} requires failure_mode {expected_mode!r}"
            )
        if intervention.get("error_code") != expected_code:
            raise ValueError(
                f"suite {normalized_suite!r} requires error_code {expected_code!r}"
            )
    elif expected_kind == "committed_partial_correction":
        expected_fields = {
            "intervention_schema_version",
            "kind",
            "setup_actions",
            "retention_required",
        }
        if intervention.get("retention_required") is not True:
            raise ValueError("partial correction intervention must require retention")
        raw_actions = intervention.get("setup_actions")
        if not isinstance(raw_actions, list) or len(raw_actions) != 4:
            raise ValueError(
                "partial correction intervention requires exactly context, correction, "
                "verification, and commit actions"
            )
        actions: list[dict[str, Any]] = []
        for index, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, Mapping):
                raise ValueError(f"partial setup_actions[{index}] must be a mapping")
            if set(raw_action) != {"tool", "arguments"} or not isinstance(
                raw_action.get("arguments"), Mapping
            ):
                raise ValueError(
                    f"partial setup_actions[{index}] must use canonical tool/arguments fields"
                )
            normalized = safe_normalize_action(raw_action)
            if normalized.get("tool") == INVALID_ACTION:
                raise ValueError(
                    f"partial setup_actions[{index}] must be a canonical valid action"
                )
            actions.append(normalized)
        matching_pairs = {
            GET_MEASUREMENT_CONTEXT: CORRECT_MEASUREMENTS,
            GET_PARAMETER_CONTEXT: CORRECT_PARAMETERS,
            GET_TOPOLOGY_CONTEXT: CORRECT_TOPOLOGY,
        }
        if matching_pairs.get(actions[0]["tool"]) != actions[1]["tool"]:
            raise ValueError(
                "partial setup context action must match its correction action"
            )
        if actions[2]["tool"] not in {RUN_WLS, VERIFY_CANDIDATE}:
            raise ValueError("partial setup third action must verify the candidate")
        if actions[3]["tool"] != COMMIT_STATE:
            raise ValueError("partial setup must end with commit_state")
        for index, action in enumerate(actions):
            arguments = action["arguments"]
            for field, value in arguments.items():
                if isinstance(value, str) and value in {
                    "$active",
                    "$candidate",
                } and field not in {
                    "state_id",
                    "candidate_state_id",
                }:
                    raise ValueError(
                        f"partial setup alias is not permitted in {field!r}"
                    )
            expected_alias = "$active" if index < 2 else "$candidate"
            reference_field = "candidate_state_id" if index == 3 else "state_id"
            other_reference = (
                "state_id" if reference_field == "candidate_state_id" else "candidate_state_id"
            )
            if (
                arguments.get(reference_field) != expected_alias
                or other_reference in arguments
            ):
                raise ValueError(
                    f"partial setup_actions[{index}] must target {expected_alias} "
                    f"only through {reference_field}"
                )
    else:
        expected_fields = {
            "intervention_schema_version",
            "kind",
            "limits",
        }
        limits = intervention.get("limits")
        if not isinstance(limits, Mapping) or set(limits) != _EFFICIENCY_LIMIT_FIELDS:
            raise ValueError(
                "efficiency intervention limits must contain exactly: "
                + ", ".join(sorted(_EFFICIENCY_LIMIT_FIELDS))
            )
        normalized_limits = copy.deepcopy(dict(limits))
        for name in (
            "maximum_policy_steps",
            "maximum_wls_calls",
            "maximum_specialized_tool_calls",
        ):
            value = normalized_limits.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"efficiency limit {name} must be a non-negative integer")
        if normalized_limits["maximum_policy_steps"] < 1:
            raise ValueError("efficiency maximum_policy_steps must be positive")
        intervention["limits"] = normalized_limits

    if set(intervention) != expected_fields:
        raise ValueError(
            f"evaluation intervention for {normalized_suite!r} must contain exactly: "
            + ", ".join(sorted(expected_fields))
        )
    return intervention


def strip_offline_truth(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return an execution-only copy with privileged audit fields removed.

    Stripping is recursive because ``TransactionalPSSEEnv.reset`` also reads
    truth and action hints from nested metadata.  The input is never mutated.
    """

    if not isinstance(scenario, Mapping):
        raise TypeError("scenario must be a mapping")

    if _has_partition_marker(scenario):
        execution, _, _ = _partitioned_scenario_parts(scenario)
        return execution

    def strip(value: Any, *, depth: int = 0) -> Any:
        if isinstance(value, Mapping):
            return {
                copy.deepcopy(key): strip(item, depth=depth + 1)
                for key, item in value.items()
                if not _offline_execution_key(key)
                and not (depth == 0 and _normalized_key(key) == "family")
            }
        if isinstance(value, list):
            return [strip(item, depth=depth + 1) for item in value]
        if isinstance(value, tuple):
            return tuple(strip(item, depth=depth + 1) for item in value)
        return copy.deepcopy(value)

    return strip(scenario)


def _release_environment_attestation(env: Any) -> dict[str, Any]:
    """Read the deployment contract from the actual per-episode environment."""

    try:
        production_mode = getattr(env, "production_dataset_mode")
    except Exception:
        production_mode = None
    try:
        candidate_oracle = getattr(env, "candidate_quality_oracle")
        oracle_mode = getattr(candidate_oracle, "mode")
    except Exception:
        oracle_mode = None

    failures: list[str] = []
    if production_mode is not True:
        failures.append("production_dataset_mode is not exactly true")
    if oracle_mode != "deployment":
        failures.append("candidate_quality_oracle.mode is not 'deployment'")
    return {
        "passed": not failures,
        "production_dataset_mode": production_mode is True,
        "candidate_quality_oracle_mode": (
            str(oracle_mode) if oracle_mode is not None else None
        ),
        "failures": failures,
    }


def _summarize_release_environment_attestations(
    attestations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(row)) for row in attestations]
    observed_by_hash: dict[str, dict[str, Any]] = {}
    failures: set[str] = set()
    for row in rows:
        contract = {
            "production_dataset_mode": row.get("production_dataset_mode"),
            "candidate_quality_oracle_mode": row.get(
                "candidate_quality_oracle_mode"
            ),
        }
        observed_by_hash[_stable_hash(contract)] = contract
        failures.update(str(item) for item in row.get("failures") or [])
    return {
        "passed": bool(rows) and all(row.get("passed") is True for row in rows),
        "episodes_checked": len(rows),
        "required": copy.deepcopy(_REQUIRED_RELEASE_ENVIRONMENT),
        "observed": [observed_by_hash[key] for key in sorted(observed_by_hash)],
        "failures": sorted(failures),
    }


def _normalize_release_policy_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "explicit_policy_identity": (
            str(value.get("explicit_policy_identity") or "").strip() or None
        ),
        "model_id": str(value.get("model_id") or "").strip() or None,
        "model_revision": str(value.get("model_revision") or "").strip() or None,
    }
    explicit = identity["explicit_policy_identity"]
    model_id = identity["model_id"]
    revision = identity["model_revision"]
    if explicit and (model_id or revision):
        raise ValueError("release policy identity cannot mix explicit and model identities")
    if bool(model_id) != bool(revision):
        raise ValueError("release model identity requires both model_id and model_revision")
    if revision and _IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise ValueError("release model revision must be an immutable digest")
    if not explicit and not model_id:
        raise ValueError("release policy identity is empty")
    return identity


def _policy_identity_attestation(
    policy: Any, expected: Mapping[str, Any] | None
) -> dict[str, Any]:
    required = copy.deepcopy(dict(expected)) if isinstance(expected, Mapping) else None
    failures: list[str] = []
    try:
        exposed = getattr(policy, "release_policy_identity")
    except Exception:
        exposed = None
    actual: dict[str, Any] | None = None
    if not isinstance(exposed, Mapping):
        failures.append("policy does not expose release_policy_identity mapping")
    else:
        try:
            actual = _normalize_release_policy_identity(exposed)
        except ValueError as exc:
            failures.append(str(exc))
    if required is None:
        failures.append("no expected release policy identity was configured")
    elif actual != required:
        failures.append("instantiated policy identity does not match the required identity")
    return {
        "passed": not failures,
        "required": required,
        "actual": actual,
        "failures": failures,
    }


def _summarize_policy_identity_attestations(
    attestations: Sequence[Mapping[str, Any]],
    *,
    required: Mapping[str, Any] | None,
) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(row)) for row in attestations]
    observed_by_hash: dict[str, Any] = {}
    failures: set[str] = set()
    for row in rows:
        actual = row.get("actual")
        observed_by_hash[_stable_hash(actual)] = copy.deepcopy(actual)
        failures.update(str(item) for item in row.get("failures") or [])
    return {
        "passed": bool(rows) and all(row.get("passed") is True for row in rows),
        "episodes_checked": len(rows),
        "required": copy.deepcopy(dict(required)) if isinstance(required, Mapping) else None,
        "observed": [observed_by_hash[key] for key in sorted(observed_by_hash)],
        "failures": sorted(failures),
    }


class ClosedLoopRolloutEvaluator:
    """Execute fixed scenario suites against freshly constructed policies.

    ``env_factory`` and ``policy_factory`` are called once per episode.  They
    may accept a keyword-only ``seed`` and/or ``rng`` argument, but never
    receive the scenario or its truth.  Development-only
    ``physical_audit_fn`` and ``tool_cost_resolver`` callbacks may inspect
    copied offline truth, but never receive the live environment.  Release
    evaluation forbids both callbacks so trajectory and audit evidence come
    only from the pinned evaluator implementation.

    A physical audit callback can override the conservative built-in audit by
    returning any of these keys: ``final_physical_correct``,
    ``physical_correctness_known``, ``healthy_components_preserved``,
    ``healthy_preservation_known``, and ``partial_fixes_retained``.
    """

    def __init__(
        self,
        *,
        env_factory: Callable[..., Any],
        policy_factory: Callable[..., Any],
        max_steps: int = 24,
        seed: int = 0,
        weights: Mapping[str, float] | None = None,
        physical_audit_fn: PhysicalAudit | None = None,
        tool_cost_resolver: ToolCostResolver | None = None,
        case_loader: CaseLoader | None = None,
        required_suites: Iterable[str] | None = None,
        minimum_suites: int = 1,
        minimum_episodes_per_suite: int = 1,
        minimum_roots_per_suite: int = 1,
        require_release_environment: bool = False,
        expected_policy_identity: Mapping[str, Any] | None = None,
        require_policy_identity: bool = False,
        development_holdout_mode: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if not callable(env_factory) or not callable(policy_factory):
            raise TypeError("env_factory and policy_factory must be callable.")
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive.")
        if case_loader is not None and not callable(case_loader):
            raise TypeError("case_loader must be callable when supplied.")
        if physical_audit_fn is not None and not callable(physical_audit_fn):
            raise TypeError("physical_audit_fn must be callable when supplied.")
        if tool_cost_resolver is not None and not callable(tool_cost_resolver):
            raise TypeError("tool_cost_resolver must be callable when supplied.")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable when supplied.")
        if (
            require_release_environment
            or require_policy_identity
            or isinstance(expected_policy_identity, Mapping)
        ) and (
            physical_audit_fn is not None or tool_cost_resolver is not None
        ):
            raise ValueError(
                "release evaluation forbids custom physical-audit and tool-cost callbacks"
            )
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.weights = dict(weights) if weights is not None else None
        self.physical_audit_fn = physical_audit_fn
        self.tool_cost_resolver = tool_cost_resolver
        self.case_loader = case_loader
        self.required_suites = _normalize_required_suites(required_suites)
        self.minimum_suites = _positive_integer(
            minimum_suites, field="minimum_suites"
        )
        self.minimum_episodes_per_suite = _positive_integer(
            minimum_episodes_per_suite, field="minimum_episodes_per_suite"
        )
        self.minimum_roots_per_suite = _positive_integer(
            minimum_roots_per_suite, field="minimum_roots_per_suite"
        )
        self.require_release_environment = bool(require_release_environment)
        self.expected_policy_identity = (
            _normalize_release_policy_identity(expected_policy_identity)
            if isinstance(expected_policy_identity, Mapping)
            else None
        )
        self.require_policy_identity = bool(require_policy_identity)
        self.development_holdout_mode = bool(development_holdout_mode)
        self.progress_callback = progress_callback
        if self.require_policy_identity and self.expected_policy_identity is None:
            raise ValueError(
                "require_policy_identity needs an explicit expected policy identity"
            )

    def _emit_progress(self, event: str, **payload: Any) -> None:
        """Emit policy-safe runtime telemetry without affecting evaluation."""

        if self.progress_callback is None:
            return
        try:
            record = {"event": str(event), **copy.deepcopy(payload)}
            self.progress_callback(record)
        except Exception:
            # Progress reporting is diagnostic only. A closed pipe or logging
            # backend/serialization failure must not change the trajectory.
            return

    def evaluate(
        self,
        scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
        | Iterable[Mapping[str, Any]],
    ) -> EvaluationResult:
        suites = _normalize_suites(scenario_suites)
        if self.development_holdout_mode and (
            set(suites) != {_DIAGNOSTIC_DEVELOPMENT_SUITE}
            or self.required_suites != (_DIAGNOSTIC_DEVELOPMENT_SUITE,)
        ):
            raise ValueError(
                "development_holdout_mode requires exactly the canonical "
                "dagger1_development suite and required-suite contract"
            )
        try:
            validate_release_scenario_suites(
                suites,
                allow_diagnostic_development=self.development_holdout_mode,
            )
        except ValueError as exc:
            release_scenario_schema_validation = {
                "passed": False,
                "scenario_schema_version": _SCENARIO_SCHEMA_VERSION,
                "failures": [str(exc)],
            }
            if self.require_release_environment:
                raise
        else:
            release_scenario_schema_validation = {
                "passed": True,
                "scenario_schema_version": _SCENARIO_SCHEMA_VERSION,
            }
        suite_manifest = fingerprint_evaluation_suites(
            suites,
            seed=self.seed,
            required_suites=self.required_suites,
            minimum_suites=self.minimum_suites,
            minimum_episodes_per_suite=self.minimum_episodes_per_suite,
            minimum_roots_per_suite=self.minimum_roots_per_suite,
            allow_diagnostic_development=self.development_holdout_mode,
        )
        episodes: list[EpisodeEvaluation] = []
        total_episodes = sum(len(rows) for rows in suites.values())
        for suite_name in sorted(suites):
            ordered = sorted(
                enumerate(suites[suite_name]),
                key=lambda item: (
                    _scenario_id(item[1], item[0]),
                    _stable_hash(strip_offline_truth(item[1])),
                ),
            )
            occurrence_by_id: Counter[str] = Counter()
            for original_index, scenario in ordered:
                scenario_id = _scenario_id(scenario, original_index)
                occurrence = occurrence_by_id[scenario_id]
                occurrence_by_id[scenario_id] += 1
                episode_seed = _episode_seed(
                    self.seed,
                    suite_name,
                    scenario_id,
                    occurrence,
                )
                episode_ordinal = len(episodes) + 1
                episode_key = f"{suite_name}:{scenario_id}:{occurrence}"
                self._emit_progress(
                    "episode_start",
                    episode_key=episode_key,
                    episode_ordinal=episode_ordinal,
                    total_episodes=total_episodes,
                    suite=suite_name,
                    scenario_id=scenario_id,
                )
                episode_started = time.perf_counter()
                episode = self._run_episode(
                    suite=suite_name,
                    scenario=scenario,
                    scenario_index=occurrence,
                    episode_seed=episode_seed,
                )
                episodes.append(episode)
                self._emit_progress(
                    "episode_complete",
                    episode_key=episode.episode_key,
                    episode_ordinal=episode_ordinal,
                    total_episodes=total_episodes,
                    elapsed_seconds=time.perf_counter() - episode_started,
                    policy_steps=episode.policy_steps,
                    terminal=episode.terminal,
                    terminal_outcome=episode.terminal_outcome,
                    evaluator_error=episode.evaluator_error,
                )

        overall = summarize_episode_evaluations(episodes)
        release_environment_validation = _summarize_release_environment_attestations(
            [episode.release_environment_attestation for episode in episodes]
        )
        policy_identity_validation = _summarize_policy_identity_attestations(
            [episode.policy_identity_attestation for episode in episodes],
            required=self.expected_policy_identity,
        )
        grouped = {
            dimension: _group_episodes(episodes, attribute)
            for dimension, attribute in (
                ("suite", "suite"),
                ("family", "family"),
                ("cardinality", "cardinality"),
                ("case", "case"),
                ("split", "split"),
                ("source_tier", "source_tier"),
                ("physical_root", "physical_root"),
            )
        }
        report = {
            "schema_version": STUDY_EVALUATION_SCHEMA_VERSION,
            "configuration": {
                "seed": self.seed,
                "max_steps": self.max_steps,
                "suite_names": sorted(suites),
                "required_suites": list(self.required_suites),
                "minimum_suites": self.minimum_suites,
                "minimum_episodes_per_suite": self.minimum_episodes_per_suite,
                "minimum_roots_per_suite": self.minimum_roots_per_suite,
                "release_scenario_schema_validation": (
                    release_scenario_schema_validation
                ),
                "release_environment_validation": release_environment_validation,
                "policy_identity_validation": policy_identity_validation,
                "custom_callback_validation": {
                    "passed": self.physical_audit_fn is None
                    and self.tool_cost_resolver is None,
                    "physical_audit_callback": self.physical_audit_fn is not None,
                    "tool_cost_callback": self.tool_cost_resolver is not None,
                },
                **suite_manifest,
            },
            "overall": overall,
            "suites": grouped["suite"],
            "groups": grouped,
            # Named aliases keep JSON reports convenient for downstream jobs.
            **{f"by_{name}": values for name, values in grouped.items()},
            "episodes": [episode.as_dict() for episode in episodes],
        }
        metrics = _recovery_metrics(overall)
        return make_evaluation_result(
            metrics,
            suite_metrics=report,
            weights=self.weights,
        )

    def evaluate_suites(
        self,
        scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
        | Iterable[Mapping[str, Any]],
    ) -> EvaluationResult:
        """Alias retained for call sites that prefer an explicit suite verb."""

        return self.evaluate(scenario_suites)

    def _run_episode(
        self,
        *,
        suite: str,
        scenario: Mapping[str, Any],
        scenario_index: int,
        episode_seed: int,
    ) -> EpisodeEvaluation:
        # Keep the immutable, privileged scenario outside execution.  The
        # environment receives only a separately copied observable scenario;
        # the full record is consulted after the trajectory has terminated.
        audit_scenario = copy.deepcopy(dict(scenario))
        execution_scenario = strip_offline_truth(audit_scenario)
        progress_scenario_id = _scenario_id(audit_scenario, scenario_index)
        scenario_groups = _scenario_groups(audit_scenario)
        progress_episode_key = (
            f"{suite}:{progress_scenario_id}:{scenario_index}"
        )
        env = _call_factory(self.env_factory, episode_seed)
        environment_attestation = _release_environment_attestation(env)
        if self.require_release_environment and not environment_attestation["passed"]:
            raise ValueError(
                "release environment validation failed: "
                + "; ".join(environment_attestation["failures"])
            )
        policy = _call_factory(
            self.policy_factory,
            episode_seed,
            policy_identity=self.expected_policy_identity,
        )
        policy_attestation = _policy_identity_attestation(
            policy, self.expected_policy_identity
        )
        if self.require_policy_identity and not policy_attestation["passed"]:
            raise ValueError(
                "release policy identity validation failed: "
                + "; ".join(policy_attestation["failures"])
            )
        env.reset(copy.deepcopy(execution_scenario))
        initial_state = _current_state(env)
        intervention_contract = evaluation_intervention_contract(
            suite,
            audit_scenario,
            required=False,
            allow_diagnostic_development=self.development_holdout_mode,
        )
        efficiency_specialized_tool_limit: int | None = None
        if (
            intervention_contract is not None
            and intervention_contract["kind"] == "efficiency_budget"
        ):
            efficiency_specialized_tool_limit = int(
                intervention_contract["limits"]["maximum_specialized_tool_calls"]
            )
        history: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        tool_counts: Counter[str] = Counter()
        specialized_counts: Counter[str] = Counter()
        nonadvancing_signatures: set[str] = set()
        deterministic_nonadvancing_failures: dict[str, dict[str, Any]] = {}
        loop_detected = False
        invalid_indices: list[int] = []
        advancing_indices: list[int] = []
        false_commits = 0
        false_rollbacks = 0
        false_finalizations = 0
        deferred_finalization_audits: list[dict[str, Any]] = []
        partial_candidate_ids: list[str] = []
        partial_action_signatures: list[str] = []
        collateral_commit_seen = False
        regret_total = 0.0
        regret_samples = 0
        evaluator_error: str | None = None
        last_transition_label: dict[str, Any] | None = None
        policy_steps = 0
        intervention_evidence: dict[str, Any] = {
            "contract": copy.deepcopy(intervention_contract),
            "applied": intervention_contract is not None,
            "pre_policy_step_count": 0,
            "injected_failure_count": 0,
            "injected_invalid_action_count": 0,
            "recovered_failure_count": 0,
            "retention_opportunity_count": 0,
            "retained_opportunity_count": 0,
        }
        control_quarantine: dict[str, Any] = {
            "quarantined": False,
            "breaker_error_code": None,
            "failure_kind": None,
            "trigger_error_code": None,
            "action_tool": None,
            "action_signature_sha256": None,
            "executed_failure_count": 0,
            "attempted_failure_count": 0,
        }

        if intervention_contract is not None:
            intervention_kind = intervention_contract["kind"]
            if intervention_kind == "pre_policy_failure":
                failure_mode = intervention_contract["failure_mode"]
                injected_state = _current_state(env)
                if failure_mode == "well_formed":
                    active_id = str(injected_state.get("active_state_id") or "active")
                    injected_action = {
                        "tool": RUN_WLS,
                        "arguments": {"state_id": active_id},
                    }
                else:
                    injected_action = invalid_action(
                        intervention_contract["error_code"]
                    )
                injected_output = {
                    "execution_status": "failure",
                    "error_code": intervention_contract["error_code"],
                    "state_mutated": False,
                }
                transition = {
                    "state_id": injected_state.get("active_state_id"),
                    "candidate_state_id": injected_state.get("candidate_state_id"),
                    "action": policy_safe_copy(injected_action),
                    "tool_output": policy_safe_copy(injected_output),
                }
                history.append(transition)
                trace.append(
                    {
                        "step": 0,
                        "intervention": True,
                        "observation_hash": None,
                        "policy_observation": None,
                        "objective_action_assessment": None,
                        "policy_tool_output": policy_safe_copy(injected_output),
                        "action": policy_safe_copy(injected_action),
                        "execution_status": "failure",
                        "advanced": False,
                        "error_code": intervention_contract["error_code"],
                        "candidate_disposition_offline": None,
                        "tool_regret": None,
                        "runtime_state_hash": _output_runtime_state_hash(
                            injected_output
                        ),
                        "objective_tool_evidence": objective_tool_evidence(
                            injected_action, injected_output
                        ),
                        "terminal_outcome": None,
                        **trace_progress_evidence(
                            before=injected_state,
                            after=injected_state,
                            output=injected_output,
                            terminal=False,
                        ),
                    }
                )
                intervention_evidence["pre_policy_step_count"] = 1
                intervention_evidence["injected_failure_count"] = 1
                intervention_evidence["injected_invalid_action_count"] = int(
                    failure_mode == "malformed"
                )
            elif intervention_kind == "committed_partial_correction":
                setup_actions = intervention_contract["setup_actions"]
                committed_candidate_id: str | None = None
                committed_signature: str | None = None
                for setup_index, raw_setup_action in enumerate(setup_actions):
                    current = _current_state(env)
                    active_id = current.get("active_state_id")
                    candidate_id = current.get("candidate_state_id")
                    setup_action = safe_normalize_action(raw_setup_action)
                    resolved_arguments = copy.deepcopy(setup_action["arguments"])
                    for field, value in list(resolved_arguments.items()):
                        if value == "$active":
                            if active_id is None:
                                raise ValueError(
                                    "partial setup could not resolve $active"
                                )
                            resolved_arguments[field] = active_id
                        elif value == "$candidate":
                            if candidate_id is None:
                                raise ValueError(
                                    "partial setup could not resolve $candidate"
                                )
                            resolved_arguments[field] = candidate_id
                    setup_action = {
                        "tool": setup_action["tool"],
                        "arguments": resolved_arguments,
                    }
                    pre_oracle = _oracle_state(env, history)
                    disposition = _candidate_disposition(pre_oracle)
                    if setup_index == len(setup_actions) - 1:
                        if disposition != "ACCEPT_PARTIAL":
                            raise ValueError(
                                "partial setup commit requires ACCEPT_PARTIAL oracle disposition: "
                                f"scenario={_scenario_id(audit_scenario, scenario_index)}, "
                                f"observed={disposition}"
                            )
                        committed_candidate_id = str(candidate_id or "") or None
                        if committed_candidate_id is None:
                            raise ValueError(
                                "partial setup commit has no current candidate"
                            )
                    try:
                        audited_setup = getattr(
                            env,
                            "apply_audited_evaluation_setup_correction",
                            None,
                        )
                        if (
                            setup_action["tool"] == CORRECT_MEASUREMENTS
                            and "measurement_updates"
                            in setup_action["arguments"]
                            and callable(audited_setup)
                        ):
                            next_state, raw_output = audited_setup(
                                copy.deepcopy(setup_action)
                            )
                        else:
                            next_state, raw_output = env.step(
                                copy.deepcopy(setup_action)
                            )
                    except Exception as exc:
                        raise ValueError(
                            "partial setup action raised "
                            f"{type(exc).__name__} at index {setup_index}"
                        ) from exc
                    if not isinstance(raw_output, Mapping):
                        raise ValueError("partial setup output must be a mapping")
                    output = copy.deepcopy(dict(raw_output))
                    if output.get("execution_status") != "success":
                        raise ValueError(
                            "partial setup action failed at index "
                            f"{setup_index}: {output.get('error_code')!r}"
                        )
                    if setup_index in {1, 3} and output.get("state_mutated") is not True:
                        raise ValueError(
                            "partial setup correction/commit must report a real state mutation"
                        )
                    setup_terminal = _is_terminal(env, next_state)
                    setup_advanced = _successful_action_advanced(
                        before=current,
                        after=next_state,
                        output=output,
                        terminal=setup_terminal,
                    )
                    transition = {
                        "state_id": current.get("active_state_id"),
                        "candidate_state_id": current.get("candidate_state_id"),
                        "action": policy_safe_copy(setup_action),
                        "tool_output": policy_safe_copy(output),
                    }
                    history.append(transition)
                    trace.append(
                        {
                            "step": len(trace),
                            "intervention": True,
                            "observation_hash": None,
                            "policy_observation": None,
                            "objective_action_assessment": None,
                            "policy_tool_output": policy_safe_copy(output),
                            "action": policy_safe_copy(setup_action),
                            "execution_status": "success",
                            "advanced": setup_advanced,
                            "error_code": None,
                            "candidate_disposition_offline": disposition,
                            "tool_regret": None,
                            "runtime_state_hash": _output_runtime_state_hash(
                                output
                            ),
                            "objective_tool_evidence": objective_tool_evidence(
                                setup_action, output
                            ),
                            "terminal_outcome": _output_terminal_outcome(output),
                            **trace_progress_evidence(
                                before=current,
                                after=next_state,
                                output=output,
                                terminal=setup_terminal,
                            ),
                        }
                    )
                    if setup_index == 1:
                        created_candidate = (
                            output.get("candidate_state_id")
                            or _current_state(env).get("candidate_state_id")
                        )
                        if created_candidate is None:
                            raise ValueError(
                                "partial setup correction did not create a candidate"
                            )
                        committed_signature = action_signature(setup_action)
                    if setup_terminal:
                        raise ValueError(
                            "partial setup terminated the episode before policy evaluation"
                        )
                accepted_after_setup = _accepted_corrections(_current_state(env))
                if not any(
                    str(item.get("candidate_state_id")) == committed_candidate_id
                    for item in accepted_after_setup
                ):
                    raise ValueError(
                        "partial setup commit was not recorded in accepted corrections"
                    )
                assert committed_candidate_id is not None
                assert committed_signature is not None
                partial_candidate_ids.append(committed_candidate_id)
                partial_action_signatures.append(committed_signature)
                intervention_evidence["pre_policy_step_count"] = len(setup_actions)
                intervention_evidence["retention_opportunity_count"] = 1

        for policy_step in range(self.max_steps):
            step = len(trace)
            policy_steps += 1
            false_finalization_this_step = False
            state_before_action = _current_state(env)
            observation = _policy_observation(env, history)
            observation = _canonical_study_policy_observation(
                observation,
                state_before=state_before_action,
                history=history,
            )
            # This check is repeated even for PolicyObservation implementations
            # so custom environments cannot accidentally expand the boundary.
            validate_policy_payload(observation)
            policy_started = time.perf_counter()
            policy_exception: str | None = None
            try:
                raw_action = _policy_action(policy, observation)
            except Exception as exc:  # malformed learner behavior is measurable
                policy_exception = type(exc).__name__
                raw_action = invalid_action(
                    "policy_exception", f"{type(exc).__name__}: {exc}"
                )
            policy_seconds = time.perf_counter() - policy_started
            action = safe_normalize_action(raw_action)
            objective_action_assessment = (
                objective_recovery_action_assessment(
                    observation,
                    scenario_family=scenario_groups["family"],
                    error_cardinality=scenario_groups["cardinality"],
                    partial_success_opportunity=bool(
                        policy_step == 0
                        and intervention_evidence[
                            "retention_opportunity_count"
                        ]
                    ),
                )
            )
            self._emit_progress(
                "policy_action",
                episode_key=progress_episode_key,
                policy_step=policy_step,
                trace_step=step,
                policy_seconds=policy_seconds,
                policy_exception=policy_exception,
                action=policy_safe_copy(action),
                policy_metrics=_policy_action_metrics(policy),
            )

            # Offline state is intentionally obtained only after the policy has
            # committed to its action.
            pre_oracle = _oracle_state(env, history)
            disposition = _candidate_disposition(pre_oracle)
            pre_remaining = _remaining_fault_count(pre_oracle)
            independent_process_label = _independent_handoff_process_label(
                env,
                state_before_action,
                action,
            )
            tool = action["tool"]
            signature = action_signature(action)
            circuit_breaker_error: str | None = None
            if tool in DIAGNOSTIC_TOOLS:
                attempted_specialized_calls = (
                    sum(specialized_counts.values()) + 1
                )
                if (
                    efficiency_specialized_tool_limit is not None
                    and attempted_specialized_calls
                    > efficiency_specialized_tool_limit
                ):
                    circuit_breaker_error = (
                        _SPECIALIZED_TOOL_BUDGET_CIRCUIT_BREAKER
                    )
                elif signature in nonadvancing_signatures:
                    # The current state has not advanced since this exact
                    # read-only diagnostic was last attempted.  Executing it
                    # again cannot add state evidence, and specialized tools
                    # can launch costly external physics solvers.  Record the
                    # attempted action as a failed transition so loop and
                    # efficiency scoring remain trace-derived, but do not call
                    # the provider a second time.
                    circuit_breaker_error = (
                        _REPEATED_DIAGNOSTIC_CIRCUIT_BREAKER
                    )
            prior_deterministic_failure = next(
                (
                    deterministic_nonadvancing_failures[key]
                    for key in _deterministic_failure_lookup_keys(
                        tool=tool, signature=signature
                    )
                    if key in deterministic_nonadvancing_failures
                ),
                None,
            )
            if (
                circuit_breaker_error is None
                and prior_deterministic_failure is not None
                and int(prior_deterministic_failure.get("failure_count", 0))
                >= _MAX_DETERMINISTIC_FAILURE_EXECUTIONS_PER_SCOPE
            ):
                circuit_breaker_error = (
                    _REPEATED_NONADVANCING_FAILURE_CIRCUIT_BREAKER
                )
                prior_count = int(
                    prior_deterministic_failure.get("failure_count", 0)
                )
                control_quarantine = {
                    "quarantined": True,
                    "breaker_error_code": circuit_breaker_error,
                    "failure_kind": prior_deterministic_failure["failure_kind"],
                    "trigger_error_code": prior_deterministic_failure["error_code"],
                    "action_tool": tool,
                    "action_signature_sha256": _stable_hash(signature),
                    "executed_failure_count": prior_count,
                    "attempted_failure_count": prior_count + 1,
                }
                # Family-wide deterministic failures can be followed by a
                # different target signature.  They are still a control loop
                # because the active state and failed family did not advance.
                loop_detected = True
            tool_counts[tool] += 1
            if tool in DIAGNOSTIC_TOOLS:
                specialized_counts[tool] += 1

            if tool == COMMIT_STATE:
                if disposition not in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}:
                    false_commits += 1
                assessment = _candidate_assessment(pre_oracle)
            elif tool == ROLLBACK_STATE:
                if disposition in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}:
                    false_rollbacks += 1
            elif tool == FINALIZE_DIAGNOSIS:
                known_false = pre_remaining is not None and pre_remaining > 0
                if known_false:
                    false_finalizations += 1
                    false_finalization_this_step = True
                else:
                    deferred_finalization_audits.append(
                        {
                            "remaining_true_fault_count": pre_remaining,
                            "explained_anomalies": copy.deepcopy(
                                observation.get("explained_anomalies") or []
                            ),
                            "counted": False,
                        }
                    )

            tool_started = time.perf_counter()
            if circuit_breaker_error is not None:
                output = {
                    "execution_status": "failure",
                    "error_code": circuit_breaker_error,
                    "state_mutated": False,
                }
                next_state = copy.deepcopy(state_before_action)
            else:
                try:
                    next_state, tool_output = env.step(copy.deepcopy(action))
                    if not isinstance(tool_output, Mapping):
                        raise TypeError("env.step() tool output must be a mapping")
                    output = copy.deepcopy(dict(tool_output))
                except Exception as exc:
                    evaluator_error = f"env_step:{type(exc).__name__}"
                    output = {
                        "execution_status": "failure",
                        "error_code": "evaluator_env_step_exception",
                        "error_detail": type(exc).__name__,
                        "state_mutated": False,
                    }
                    next_state = _current_state(env)
            tool_seconds = time.perf_counter() - tool_started

            status = str(output.get("execution_status") or "failure")
            output["execution_status"] = status
            if (
                status == "failure"
                and tool == INVALID_ACTION
                and not str(output.get("error_code") or "").strip()
            ):
                # The sentinel action is produced by the evaluator's own
                # canonical normalization boundary.  Some environments report
                # only a generic failed transition for that sentinel; retain
                # the authoritative normalization error so the persisted
                # trace remains schema-complete and independently ingestible.
                action_arguments = action.get("arguments")
                action_arguments = (
                    action_arguments
                    if isinstance(action_arguments, Mapping)
                    else {}
                )
                output["error_code"] = str(
                    action_arguments.get("error_code") or "invalid_action"
                )
            last_transition_label = (
                {
                    **independent_process_label,
                    "execution_status": status,
                }
                if independent_process_label is not None
                else None
            )
            if (
                tool == FINALIZE_DIAGNOSIS
                and status != "success"
                and not false_finalization_this_step
            ):
                false_finalizations += 1
                false_finalization_this_step = True
                if deferred_finalization_audits:
                    deferred_finalization_audits[-1]["counted"] = True
            if tool == COMMIT_STATE and status == "success":
                if disposition == "ACCEPT_PARTIAL":
                    candidate_id = _candidate_id(pre_oracle, observation)
                    if candidate_id:
                        partial_candidate_ids.append(candidate_id)
                    partial_action_signatures.append(signature)
                if assessment.get("collateral_damage") is True or assessment.get(
                    "healthy_component_modified"
                ) is True:
                    collateral_commit_seen = True
            invalid = tool == INVALID_ACTION or status != "success"
            if invalid:
                invalid_indices.append(step)
            terminal_after = _is_terminal(env, next_state)
            advanced = bool(
                not invalid
                and _successful_action_advanced(
                    before=state_before_action,
                    after=next_state,
                    output=output,
                    terminal=terminal_after,
                )
            )
            if advanced:
                advancing_indices.append(step)
                # A real state/lifecycle advance starts a new control epoch.
                # Reusing a read-only action after that advance (for example,
                # verifying a candidate and later re-running WLS after it is
                # committed) is not a loop.
                nonadvancing_signatures.clear()
                deterministic_nonadvancing_failures.clear()
            else:
                if signature in nonadvancing_signatures:
                    loop_detected = True
                nonadvancing_signatures.add(signature)
                deterministic_failure_kind = (
                    _deterministic_nonadvancing_failure_kind(
                        tool=tool,
                        execution_status=status,
                        error_code=output.get("error_code"),
                    )
                )
                if (
                    circuit_breaker_error is None
                    and deterministic_failure_kind is not None
                ):
                    deterministic_failure_key = (
                        _deterministic_failure_storage_key(
                            tool=tool,
                            signature=signature,
                            error_code=output.get("error_code"),
                        )
                    )
                    previous = deterministic_nonadvancing_failures.get(
                        deterministic_failure_key
                    )
                    previous_count = (
                        int(previous.get("failure_count", 0))
                        if previous is not None
                        and previous.get("failure_kind")
                        == deterministic_failure_kind
                        and previous.get("error_code") == output.get("error_code")
                        else 0
                    )
                    deterministic_nonadvancing_failures[
                        deterministic_failure_key
                    ] = {
                        "failure_kind": deterministic_failure_kind,
                        "error_code": output.get("error_code"),
                        "failure_count": min(
                            previous_count + 1,
                            _MAX_DETERMINISTIC_FAILURE_EXECUTIONS_PER_SCOPE,
                        ),
                    }

            label = _resolve_cost_label(
                self.tool_cost_resolver,
                # Offline labels belong only in a copied scorer context, never
                # in env.reset, the policy payload, or a live environment handle.
                scenario=audit_scenario,
                suite=suite,
                step=policy_step,
                observation=observation,
                action=action,
                tool_output=output,
                oracle_state=pre_oracle,
            )
            regret = _tool_regret(label, action)
            if regret is not None:
                regret_total += regret
                regret_samples += 1

            persisted_tool_output = policy_safe_copy(output)
            transition = {
                "state_id": observation.get("active_state_id"),
                "candidate_state_id": observation.get("candidate_state_id"),
                "action": policy_safe_copy(action),
                "tool_output": copy.deepcopy(persisted_tool_output),
            }
            history.append(transition)
            trace.append(
                {
                    "step": step,
                    "intervention": False,
                    "observation_hash": _stable_hash(observation),
                    "policy_observation": policy_safe_copy(observation),
                    "objective_action_assessment": copy.deepcopy(
                        objective_action_assessment
                    ),
                    "policy_tool_output": copy.deepcopy(persisted_tool_output),
                    "action": policy_safe_copy(action),
                    "execution_status": status,
                    "advanced": advanced,
                    "error_code": output.get("error_code"),
                    "candidate_disposition_offline": disposition,
                    "tool_regret": regret,
                    "runtime_state_hash": _output_runtime_state_hash(output),
                    "objective_tool_evidence": objective_tool_evidence(
                        action, output
                    ),
                    "terminal_outcome": _output_terminal_outcome(output),
                    **trace_progress_evidence(
                        before=state_before_action,
                        after=next_state,
                        output=output,
                        terminal=terminal_after,
                    ),
                }
            )
            self._emit_progress(
                "step_complete",
                episode_key=progress_episode_key,
                policy_step=policy_step,
                trace_step=step,
                tool=tool,
                tool_seconds=tool_seconds,
                execution_status=status,
                error_code=output.get("error_code"),
                advanced=advanced,
                terminal=terminal_after,
                circuit_breaker=circuit_breaker_error,
            )
            if evaluator_error or circuit_breaker_error or terminal_after:
                break

        final_state = _current_state(env)
        final_oracle = _oracle_state(env, history)
        terminal = _is_terminal(env, final_state)
        outcome = _terminal_outcome(env, trace)
        # A release trace has one authoritative terminal marker: the final
        # transition if and only if the episode is terminal.  Tool outputs can
        # report the outcome in different fields, so normalize them only after
        # deriving the environment-level outcome above.
        for row in trace:
            row["terminal_outcome"] = None
        if terminal and outcome is not None and trace:
            trace[-1]["terminal_outcome"] = outcome
        active_physical_state = _active_physical_state(env, final_state)
        default_audit = _default_physical_audit(
            scenario=audit_scenario,
            initial_state=initial_state,
            final_state=final_state,
            final_oracle=final_oracle,
            history=history,
            collateral_commit_seen=collateral_commit_seen,
            terminal=terminal,
            terminal_outcome=outcome,
            active_physical_state=active_physical_state,
            case_loader=self.case_loader,
        )
        audit = dict(default_audit)
        audit["control_quarantine"] = copy.deepcopy(control_quarantine)
        if self.physical_audit_fn is not None:
            supplied = self.physical_audit_fn(
                {
                    "scenario": copy.deepcopy(audit_scenario),
                    "suite": suite,
                    "initial_state": copy.deepcopy(initial_state),
                    "final_state": copy.deepcopy(final_state),
                    "final_oracle_state": copy.deepcopy(final_oracle),
                    "history": copy.deepcopy(history),
                    "terminal": terminal,
                    "terminal_outcome": outcome,
                    "active_physical_state": copy.deepcopy(active_physical_state),
                    "default_audit": copy.deepcopy(default_audit),
                }
            )
            if isinstance(supplied, bool):
                supplied = {
                    "final_physical_correct": supplied,
                    "physical_correctness_known": True,
                }
            if not isinstance(supplied, Mapping):
                raise TypeError("physical_audit_fn must return a mapping or bool.")
            unexpected = sorted(set(supplied) - _PHYSICAL_AUDIT_OVERRIDE_KEYS)
            if unexpected:
                raise ValueError(
                    "physical_audit_fn returned unsupported override fields: "
                    + ", ".join(str(item) for item in unexpected)
                )
            audit.update(copy.deepcopy(dict(supplied)))

        handoff_audit_state = _audit_bound_terminal_state(
            final_state,
            history,
            independent_transition_label=last_transition_label,
        )
        post_correction_handoff_assessment = (
            audit_post_correction_controller_handoff(
                _strict_audit_scenario(audit_scenario),
                handoff_audit_state,
                terminal=terminal,
                terminal_outcome=outcome,
                active_physical_state=active_physical_state,
                remaining_truth=_complete_remaining_truth(final_oracle),
                case_loader=self.case_loader,
            )
        )
        # This is a sibling offline assessment, not a replacement for the
        # actual production outcome or the strict physical audit.  Keeping it
        # under ``audit`` also prevents privileged truth from entering the
        # policy observation or trajectory target.
        audit["post_correction_handoff_assessment"] = copy.deepcopy(
            post_correction_handoff_assessment
        )

        physical_known = bool(audit.get("physical_correctness_known", False))
        physical_correct = physical_known and bool(
            audit.get("final_physical_correct", False)
        )
        healthy_known = bool(audit.get("healthy_preservation_known", False))
        healthy_preserved = healthy_known and bool(
            audit.get("healthy_components_preserved", False)
        )
        audit_truth = _scenario_truth(audit_scenario)
        for pending in deferred_finalization_audits:
            if pending["counted"]:
                continue
            diagnostic = _diagnostic_truth_audit(
                audit_truth, pending["explained_anomalies"]
            )
            unknown_and_physically_wrong = (
                pending["remaining_true_fault_count"] is None
                and physical_known
                and not physical_correct
            )
            if (
                not diagnostic["diagnostic_truth_matched"]
                or unknown_and_physically_wrong
            ):
                false_finalizations += 1

        accepted = _accepted_corrections(final_state)
        accepted_ids = {
            str(item.get("candidate_state_id"))
            for item in accepted
            if item.get("candidate_state_id") is not None
        }
        accepted_signatures = {
            action_signature(_correction_action(item))
            for item in accepted
            if _correction_action(item).get("tool")
        }
        retained_partial = sum(
            candidate_id in accepted_ids for candidate_id in partial_candidate_ids
        )
        if len(partial_candidate_ids) < len(partial_action_signatures):
            retained_partial += sum(
                signature in accepted_signatures
                for signature in partial_action_signatures[len(partial_candidate_ids) :]
            )
        if audit.get("partial_fixes_retained") is not None:
            override = audit["partial_fixes_retained"]
            if isinstance(override, bool):
                retained_partial = len(partial_action_signatures) if override else 0
            else:
                retained_partial = max(
                    0, min(int(override), len(partial_action_signatures))
                )
        intervention_evidence["retained_opportunity_count"] = min(
            intervention_evidence["retention_opportunity_count"], retained_partial
        )

        groups = scenario_groups
        scenario_id = _scenario_id(audit_scenario, scenario_index)
        episode_key = f"{suite}:{scenario_id}:{scenario_index}"
        final_success = bool(
            terminal and outcome == "resolved" and physical_correct
        )
        audited_handoff = _is_audited_post_correction_handoff(
            terminal=terminal,
            terminal_outcome=outcome,
            assessment=post_correction_handoff_assessment,
            scenario_id=scenario_id,
            physical_root_fingerprint=groups["physical_root"],
            scenario_family=groups["family"],
            false_commit_count=false_commits,
            false_rollback_count=false_rollbacks,
            false_finalization_count=false_finalizations,
            loop_detected=loop_detected,
            evaluator_error=evaluator_error,
        )
        # A generic operator escalation remains a valid fail-closed terminal
        # outcome, but it is not evidence that an injected/invalid failure was
        # recovered.  Only strict resolution or the audited post-correction
        # completion contract can close those recovery numerators.
        recovery_terminal = bool(final_success or audited_handoff)
        if intervention_evidence["injected_failure_count"]:
            intervention_evidence["recovered_failure_count"] = int(
                recovery_terminal and bool(advancing_indices)
            )
        recovered_invalid = (
            sum(
                any(
                    advancing_index > invalid_index
                    for advancing_index in advancing_indices
                )
                for invalid_index in invalid_indices
            )
            if recovery_terminal
            else 0
        )
        return EpisodeEvaluation(
            episode_key=episode_key,
            scenario_id=scenario_id,
            suite=suite,
            family=groups["family"],
            cardinality=groups["cardinality"],
            case=groups["case"],
            split=groups["split"],
            source_tier=groups["source_tier"],
            physical_root=groups["physical_root"],
            seed=episode_seed,
            steps=len(trace),
            policy_steps=policy_steps,
            terminal=terminal,
            terminal_outcome=outcome,
            final_physical_correct=physical_correct,
            physical_correctness_known=physical_known,
            final_physical_success=final_success,
            healthy_components_preserved=healthy_preserved,
            healthy_preservation_known=healthy_known,
            false_commit_count=false_commits,
            false_rollback_count=false_rollbacks,
            false_finalization_count=false_finalizations,
            partial_fix_count=len(partial_action_signatures),
            retained_partial_fix_count=retained_partial,
            invalid_action_count=len(invalid_indices),
            recovered_invalid_action_count=recovered_invalid,
            loop_detected=loop_detected,
            wls_calls=sum(tool_counts[tool] for tool in (RUN_WLS, VERIFY_CANDIDATE)),
            specialized_tool_calls=sum(specialized_counts.values()),
            tool_counts=dict(sorted(tool_counts.items())),
            specialized_tool_counts=dict(sorted(specialized_counts.items())),
            tool_regret_total=regret_total,
            tool_regret_samples=regret_samples,
            evaluation_intervention=copy.deepcopy(intervention_evidence),
            control_quarantine=copy.deepcopy(control_quarantine),
            release_environment_attestation=copy.deepcopy(environment_attestation),
            policy_identity_attestation=copy.deepcopy(policy_attestation),
            objective_evidence=study_objective_episode_evidence_marker(),
            audit=copy.deepcopy(audit),
            trace=trace,
            evaluator_error=evaluator_error,
        )


def evaluate_rollout_suites(
    scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
    | Iterable[Mapping[str, Any]],
    *,
    env_factory: Callable[..., Any],
    policy_factory: Callable[..., Any],
    max_steps: int = 24,
    seed: int = 0,
    weights: Mapping[str, float] | None = None,
    physical_audit_fn: PhysicalAudit | None = None,
    tool_cost_resolver: ToolCostResolver | None = None,
    case_loader: CaseLoader | None = None,
    required_suites: Iterable[str] | None = None,
    minimum_suites: int = 1,
    minimum_episodes_per_suite: int = 1,
    minimum_roots_per_suite: int = 1,
    require_release_environment: bool = False,
    expected_policy_identity: Mapping[str, Any] | None = None,
    require_policy_identity: bool = False,
    development_holdout_mode: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> EvaluationResult:
    """Functional entry point for closed-loop suite evaluation."""

    return ClosedLoopRolloutEvaluator(
        env_factory=env_factory,
        policy_factory=policy_factory,
        max_steps=max_steps,
        seed=seed,
        weights=weights,
        physical_audit_fn=physical_audit_fn,
        tool_cost_resolver=tool_cost_resolver,
        case_loader=case_loader,
        required_suites=required_suites,
        minimum_suites=minimum_suites,
        minimum_episodes_per_suite=minimum_episodes_per_suite,
        minimum_roots_per_suite=minimum_roots_per_suite,
        require_release_environment=require_release_environment,
        expected_policy_identity=expected_policy_identity,
        require_policy_identity=require_policy_identity,
        development_holdout_mode=development_holdout_mode,
        progress_callback=progress_callback,
    ).evaluate(scenario_suites)


# A descriptive alias makes the entry point easy to discover without breaking
# callers that use the shorter name above.
ClosedLoopEvaluator = ClosedLoopRolloutEvaluator
evaluate_closed_loop_rollouts = evaluate_rollout_suites
evaluate_closed_loop = evaluate_rollout_suites


_RELEASE_SOURCE_FAILURE = (
    "source worktree is not a clean tracked commit; use --allow-dirty-source "
    "only for non-release development evidence"
)
_RELEASE_ENVIRONMENT_FAILURE = (
    "executed environment contract is not release-safe: "
    "production_dataset_mode=true and candidate_quality_oracle.mode='deployment' "
    "are required"
)
_RELEASE_SCENARIO_SCHEMA_FAILURE = (
    "input suite is not canonical release scenario schema version 1"
)
_RELEASE_POLICY_IDENTITY_FAILURE = (
    "instantiated policy identity did not match the release provenance identity"
)
_RELEASE_CALLBACK_FAILURE = (
    "custom physical-audit or tool-cost callbacks are forbidden in release evaluation"
)
_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
    )


def _source_path_descriptor(
    path: str | os.PathLike[str], *, repo_root: Path
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        displayed = str(resolved.relative_to(repo_root))
        location = "repository"
    except ValueError:
        displayed = str(resolved)
        location = "external"
    return {
        "path": displayed,
        "location": location,
        "sha256": file_sha256(resolved) if resolved.is_file() else None,
    }


def _callable_descriptor(
    spec: str,
    value: Callable[..., Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    normalized_spec = str(spec).strip()
    resolved = _load_import_spec(normalized_spec, field="factory import spec")
    if resolved is not value:
        raise ValueError(
            "factory import spec does not resolve to the supplied callable: "
            f"{normalized_spec}"
        )
    source_path: str | None = None
    try:
        source_path = inspect.getsourcefile(value) or inspect.getfile(value)
    except (OSError, TypeError):
        source_path = None
    return {
        "import_spec": normalized_spec,
        "module": getattr(value, "__module__", None),
        "qualname": getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__qualname__),
        ),
        "source": (
            _source_path_descriptor(source_path, repo_root=repo_root)
            if source_path is not None
            else None
        ),
    }


def _protocol_registry_descriptor(protocol: str) -> dict[str, Any]:
    normalized = str(protocol).strip().lower()
    if normalized == "canonical":
        from psse_env.dagger.protocol_bridge import unified_tool_schemas

        registry = unified_tool_schemas()
    elif normalized == "controller":
        registry = copy.deepcopy(TOOL_JSON_SCHEMAS)
    else:
        raise ValueError("protocol must be canonical or controller")
    names = [str(row["function"]["name"]) for row in registry]
    return {
        "protocol": normalized,
        "registry_sha256": stable_json_sha256(registry),
        "registered_tool_count": len(names),
        "registered_tools": names,
    }


def _nvidia_driver_version() -> str | None:
    """Read the NVIDIA driver version without making provenance collection fatal."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    versions = sorted(
        {
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        }
    )
    if not versions:
        return None
    return ",".join(versions)


def _accelerator_environment_descriptor() -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "backend": "cpu",
        "cuda_available": False,
        "torch_cuda_version": None,
        "driver_version": None,
        "device_count": 0,
        "bf16_supported": False,
        "devices": [],
    }
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError):
        return descriptor

    torch_version = getattr(torch, "version", None)
    cuda_runtime = getattr(torch_version, "cuda", None)
    descriptor["torch_cuda_version"] = (
        str(cuda_runtime).strip() if cuda_runtime is not None else None
    )
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return descriptor
    try:
        cuda_available = bool(cuda.is_available())
    except (AttributeError, RuntimeError):
        return descriptor
    descriptor["cuda_available"] = cuda_available
    if not cuda_available:
        return descriptor

    try:
        device_count = int(cuda.device_count())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        device_count = 0
    try:
        bf16_supported = bool(cuda.is_bf16_supported()) if device_count else False
    except (AttributeError, RuntimeError):
        bf16_supported = False
    devices: list[dict[str, Any]] = []
    for index in range(max(0, device_count)):
        try:
            properties = cuda.get_device_properties(index)
        except (AttributeError, RuntimeError):
            continue
        name = str(getattr(properties, "name", "") or "").strip()
        total_memory = getattr(properties, "total_memory", None)
        normalized_memory = (
            int(total_memory)
            if type(total_memory) is int and total_memory > 0
            else 0
        )
        try:
            raw_capability = cuda.get_device_capability(index)
            capability = [int(raw_capability[0]), int(raw_capability[1])]
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            capability = None
        devices.append(
            {
                "index": index,
                "name": name,
                "total_memory_bytes": normalized_memory or None,
                "compute_capability": capability,
                "accelerator_class": (
                    normalize_accelerator_class(name, normalized_memory)
                    if name and normalized_memory
                    else None
                ),
            }
        )
    descriptor.update(
        {
            "backend": "cuda",
            "driver_version": _nvidia_driver_version(),
            "device_count": device_count,
            "bf16_supported": bf16_supported,
            "devices": devices,
        }
    )
    return descriptor


def _runtime_environment_descriptor() -> dict[str, Any]:
    distributions = (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "trl",
        "sentencepiece",
        "Pillow",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "accelerator": _accelerator_environment_descriptor(),
    }


def _evaluation_provenance_failures(provenance: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(provenance, Mapping):
        return ["evaluation identity provenance is missing"]

    failures: list[str] = []
    provenance_schema_version = provenance.get("provenance_schema_version")
    if type(provenance_schema_version) is not int or provenance_schema_version != 1:
        failures.append("provenance_schema_version is not exactly integer 1")
    source_state = provenance.get("source_state")
    if not isinstance(source_state, Mapping) or source_state.get(
        "release_eligible_source"
    ) is not True:
        failures.append(_RELEASE_SOURCE_FAILURE)

    input_suite = provenance.get("input_suite")
    if not isinstance(input_suite, Mapping):
        failures.append("input suite path and hash are missing")
    else:
        if not str(input_suite.get("resolved_path") or "").strip():
            failures.append("input suite resolved path is missing")
        if not _is_sha256(input_suite.get("sha256")):
            failures.append("input suite SHA-256 is missing or invalid")

    factories = provenance.get("factories")
    factories = factories if isinstance(factories, Mapping) else {}
    for factory_role in ("environment", "policy"):
        descriptor = factories.get(factory_role)
        if not isinstance(descriptor, Mapping) or not str(
            descriptor.get("import_spec") or ""
        ).strip():
            failures.append(f"{factory_role} factory import spec is missing")
            continue
        source = descriptor.get("source")
        if not isinstance(source, Mapping) or not _is_sha256(source.get("sha256")):
            failures.append(
                f"{factory_role} factory source fingerprint is missing"
            )
    case_loader = factories.get("case_loader")
    if case_loader is not None:
        if not isinstance(case_loader, Mapping) or not str(
            case_loader.get("import_spec") or ""
        ).strip():
            failures.append("case-loader import spec is missing")
        else:
            source = case_loader.get("source")
            if not isinstance(source, Mapping) or not _is_sha256(
                source.get("sha256")
            ):
                failures.append("case-loader source fingerprint is missing")

    runtime_environment = provenance.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping) or not isinstance(
        runtime_environment.get("packages"), Mapping
    ):
        failures.append("runtime environment package versions are missing")
    elif not isinstance(runtime_environment.get("accelerator"), Mapping):
        failures.append("runtime environment accelerator identity is missing")

    policy_identity = provenance.get("policy_identity")
    if not isinstance(policy_identity, Mapping):
        failures.append("policy identity is missing")
    else:
        explicit = str(policy_identity.get("explicit_policy_identity") or "").strip()
        model_id = str(policy_identity.get("model_id") or "").strip()
        revision = str(policy_identity.get("model_revision") or "").strip()
        if bool(model_id) != bool(revision):
            failures.append("model ID and immutable model revision must be supplied together")
        if revision and _IMMUTABLE_REVISION.fullmatch(revision) is None:
            failures.append("model revision is not an immutable 40- or 64-hex digest")
        if not explicit and not (model_id and revision):
            failures.append(
                "policy identity requires an explicit identity or a model ID/revision pair"
            )

    protocol_registry = provenance.get("protocol_registry")
    if not isinstance(protocol_registry, Mapping):
        failures.append("model-visible protocol registry identity is missing")
    else:
        if protocol_registry.get("protocol") not in {"canonical", "controller"}:
            failures.append("model-visible protocol is missing or unsupported")
        if not _is_sha256(protocol_registry.get("registry_sha256")):
            failures.append("model-visible protocol registry SHA-256 is missing")

    evaluator_source = provenance.get("evaluator_source")
    if not isinstance(evaluator_source, Mapping) or not _is_sha256(
        evaluator_source.get("sha256")
    ):
        failures.append("evaluator source fingerprint is missing")
    return failures


def _load_unlinked_regular_json_value(
    path: str | os.PathLike[str],
    *,
    field: str,
) -> tuple[Any, Path, str]:
    """Read one immutable JSON value without following a linked path.

    Study inputs and checkpoint receipts are security-sensitive identity
    objects. Rejecting symbolic-link path components and multiply linked files
    prevents an evaluation from naming bytes outside the reviewed publication.
    The before/after descriptor check also fails if the file changes during
    the single read. The returned digest always names those exact captured
    bytes, rather than a later path lookup.
    """

    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{field} path contains a symbolic link: {current}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ValueError(f"{field} cannot be opened as a regular file: {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{field} must be one unlinked regular file: {absolute}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ):
            raise ValueError(f"{field} changed while it was being read: {absolute}")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid UTF-8 JSON: {absolute}") from exc
    return payload, absolute, hashlib.sha256(raw).hexdigest()


def _load_unlinked_regular_json_object(
    path: str | os.PathLike[str],
    *,
    field: str,
) -> tuple[dict[str, Any], Path]:
    """Read one immutable JSON object and retain the mapping-only API."""

    payload, absolute, _ = _load_unlinked_regular_json_value(path, field=field)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must contain one JSON object")
    return dict(payload), absolute


def _development_evaluation_contract(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = manifest.get("bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    contract = bindings.get("development_evaluation")
    if not isinstance(contract, Mapping):
        raise ValueError(
            "study manifest does not pin bindings.development_evaluation"
        )
    return copy.deepcopy(dict(contract))


def _validate_development_holdout_for_study(
    *,
    holdout_path: str | os.PathLike[str],
    holdout_manifest_path: str | os.PathLike[str],
    generator_report_path: str | os.PathLike[str],
    study_manifest: Mapping[str, Any],
    reviewed_source_commit: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Bind one generated 30-root holdout without trusting caller hashes."""

    from psse_env.dagger.build_dagger1_development_holdout import (
        APPROVED_DAGGER1_DEVELOPMENT_ROOT_COUNT,
        DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
        DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD,
        DAGGER1_DEVELOPMENT_SPLIT,
        DAGGER1_DEVELOPMENT_SUITE_NAME,
        DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
        _source_bindings as current_development_source_bindings,
    )
    from psse_env.providers.matpower import PARAMETER_RANKING_CONTRACT

    payload, holdout, holdout_sha256 = _load_unlinked_regular_json_value(
        holdout_path, field="development holdout"
    )
    provenance, holdout_manifest, holdout_manifest_sha256 = (
        _load_unlinked_regular_json_value(
        holdout_manifest_path, field="development holdout manifest"
        )
    )
    generator, generator_report, generator_report_sha256 = (
        _load_unlinked_regular_json_value(
        generator_report_path, field="development holdout generator report"
        )
    )
    if not all(
        isinstance(value, Mapping)
        for value in (payload, provenance, generator)
    ):
        raise ValueError(
            "development holdout inputs must each contain one JSON object"
        )
    payload = dict(payload)
    provenance = dict(provenance)
    generator = dict(generator)
    if len({holdout, holdout_manifest, generator_report}) != 3:
        raise ValueError(
            "development holdout, manifest, and generator report must be distinct"
        )
    if set(payload) != {DAGGER1_DEVELOPMENT_SUITE_NAME}:
        raise ValueError("development holdout must contain exactly its canonical suite")
    rows = payload[DAGGER1_DEVELOPMENT_SUITE_NAME]
    if not isinstance(rows, list) or not rows:
        raise ValueError("development holdout suite must be a non-empty array")
    roots: set[str] = set()
    family_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or type(row.get("scenario_schema_version")) is not int
            or row.get("scenario_schema_version") != 1
        ):
            raise ValueError(f"development holdout row {index} is not schema-v1")
        if set(row) != {"scenario_schema_version", "execution", "audit", "grouping"}:
            raise ValueError(
                f"development holdout row {index} envelope fields are not exact"
            )
        if not isinstance(row.get("execution"), Mapping) or not isinstance(
            row.get("audit"), Mapping
        ):
            raise ValueError(f"development holdout row {index} has a malformed envelope")
        grouping = row.get("grouping")
        if not isinstance(grouping, Mapping):
            raise ValueError(f"development holdout row {index} grouping is malformed")
        if grouping.get("split") != DAGGER1_DEVELOPMENT_SPLIT:
            raise ValueError(f"development holdout row {index} has the wrong split")
        root = str(grouping.get("physical_root_fingerprint") or "").strip()
        family = str(grouping.get("scenario_family") or "").strip()
        if not root or not family:
            raise ValueError(f"development holdout row {index} lacks root/family identity")
        if root in roots:
            raise ValueError("development holdout repeats a physical root")
        roots.add(root)
        family_counts[family] += 1

    bindings = study_manifest.get("bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    frozen = bindings.get("evaluation")
    frozen = frozen if isinstance(frozen, Mapping) else {}
    source = provenance.get("source_state")
    source = source if isinstance(source, Mapping) else {}
    source_bindings = provenance.get("source_bindings")
    report_partition = generator.get("source_partition")
    report_partition = report_partition if isinstance(report_partition, Mapping) else {}
    report_admission = generator.get("parameter_ranking_admission")
    report_admission = report_admission if isinstance(report_admission, Mapping) else {}
    plan = dict(sorted(DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN.items()))
    root_hash = stable_json_sha256(sorted(roots))
    declared_hashes = provenance.get("root_set_sha256")
    declared_hashes = declared_hashes if isinstance(declared_hashes, Mapping) else {}
    current_bindings = current_development_source_bindings(repo_root)
    overlap_fields = (
        (
            "pairwise_input_overlap",
            {"d0_frozen", "d0_d1_training", "frozen_d1_training"},
        ),
        (
            "training_development_reserved_boundary_overlap",
            {"d0", "frozen", "d1_training"},
        ),
        (
            "development_protected_overlap",
            {"d0", "frozen", "d1_training"},
        ),
    )
    overlaps_empty = True
    for field_name, expected_fields in overlap_fields:
        overlap = provenance.get(field_name)
        if (
            not isinstance(overlap, Mapping)
            or set(overlap) != expected_fields
            or any(value != [] for value in overlap.values())
        ):
            overlaps_empty = False
            break
    declared_plan = provenance.get("plan")
    declared_counts = provenance.get("selected_count_by_family")
    exact_plan = (
        isinstance(declared_plan, Mapping)
        and all(type(value) is int for value in declared_plan.values())
        and dict(declared_plan) == plan
    )
    exact_selected_counts = (
        isinstance(declared_counts, Mapping)
        and all(type(value) is int for value in declared_counts.values())
        and dict(declared_counts) == dict(sorted(family_counts.items()))
    )
    root_counts = provenance.get("root_counts")
    exact_development_root_count = (
        isinstance(root_counts, Mapping)
        and type(root_counts.get("development")) is int
        and root_counts.get("development") == len(roots)
    )
    contract = _development_evaluation_contract(study_manifest)
    checks = {
        "schema_version": type(provenance.get("schema_version")) is int
        and provenance.get("schema_version") == 1,
        "scenario_schema_version": type(
            provenance.get("scenario_schema_version")
        )
        is int
        and provenance.get("scenario_schema_version") == 1,
        "artifact_type": provenance.get("artifact_type")
        == "dagger1_development_holdout_suite",
        "builder_contract": provenance.get("builder_contract")
        == DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
        "suite_name": provenance.get("suite_name")
        == DAGGER1_DEVELOPMENT_SUITE_NAME,
        "suite_format": provenance.get("suite_format")
        == "evaluation_suite_mapping_v1",
        "split": provenance.get("split") == DAGGER1_DEVELOPMENT_SPLIT,
        "source_partition": provenance.get("source_partition") == "train",
        "parameter_threshold": provenance.get(
            "parameter_ranking_dominance_threshold"
        )
        == DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD,
        "seed": type(provenance.get("seed")) is int
        and provenance.get("seed") == contract.get("evaluator_seed"),
        "plan": exact_plan,
        "selected_counts": exact_selected_counts,
        "source_commit": source.get("release_eligible_source") is True
        and source.get("source_commit") == reviewed_source_commit,
        "source_bindings": source_bindings == current_bindings,
        "generator_partition": report_partition.get("enabled") is True
        and report_partition.get("selected") == "train",
        "generator_admission": report_admission.get("contract")
        == PARAMETER_RANKING_CONTRACT
        and report_admission.get("enforced") is True
        and report_admission.get("threshold")
        == DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD,
        "output_sha256": provenance.get("output_sha256") == holdout_sha256,
        "generator_report_sha256": provenance.get("generator_report_sha256")
        == generator_report_sha256,
        "scenario_count": type(provenance.get("scenario_count")) is int
        and provenance.get("scenario_count") == len(rows),
        "physical_root_count": type(provenance.get("physical_root_count")) is int
        and provenance.get("physical_root_count") == len(roots),
        "root_counts": exact_development_root_count,
        "approved_root_count": len(roots)
        == APPROVED_DAGGER1_DEVELOPMENT_ROOT_COUNT,
        "root_set_sha256": declared_hashes.get("development") == root_hash,
        "frozen_suite_sha256": provenance.get("frozen_suite_sha256")
        == frozen.get("suite_sha256"),
        "evaluation_policy_sha256": provenance.get("evaluation_policy_sha256")
        == frozen.get("policy_sha256"),
        "training_eligible": provenance.get("training_eligible") is False,
        "training_collection_eligible": provenance.get(
            "training_collection_eligible"
        )
        is False,
        "release_evidence_eligible": provenance.get("release_evidence_eligible")
        is False,
        "promotion_evidence_eligible": provenance.get(
            "promotion_evidence_eligible"
        )
        is False,
        "model_selection_eligible": provenance.get(
            "diagnostic_closed_loop_model_selection_eligible"
        )
        is True,
        "overlap_declarations": overlaps_empty,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "development holdout study binding failed: " + ", ".join(failed)
        )
    descriptor = {
        "contract": STUDY_DEVELOPMENT_HOLDOUT_PROVENANCE_CONTRACT,
        "holdout_sha256": holdout_sha256,
        "holdout_manifest_sha256": holdout_manifest_sha256,
        "generator_report_sha256": generator_report_sha256,
        "reviewed_source_commit": reviewed_source_commit,
        "builder_contract": DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
        "suite_name": DAGGER1_DEVELOPMENT_SUITE_NAME,
        "seed": provenance["seed"],
        "physical_roots": len(roots),
        "root_set_sha256": root_hash,
    }
    return {
        "development_holdout_sha256": descriptor["holdout_sha256"],
        "development_holdout_provenance_id": stable_json_sha256(descriptor),
        "development_holdout_root_set_sha256": root_hash,
        "development_holdout_physical_roots": len(roots),
        "provenance_descriptor": descriptor,
        "suite_name": DAGGER1_DEVELOPMENT_SUITE_NAME,
    }


def build_study_evaluation_binding(
    *,
    study_manifest_path: str | os.PathLike[str],
    variant_id: str,
    reviewed_source_commit: str,
    model_id: str,
    model_revision: str,
    input_suite_path: str | os.PathLike[str],
    diagnostic_only: bool,
    evaluator_seed: int,
    max_steps: int,
    required_suites: Sequence[str],
    minimum_suites: int,
    minimum_episodes_per_suite: int,
    minimum_roots_per_suite: int,
    protocol: str,
    training_seed: int | None = None,
    checkpoint_receipt_path: str | os.PathLike[str] | None = None,
    development_holdout_manifest_path: str | os.PathLike[str] | None = None,
    development_holdout_generator_report_path: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Derive every study field from pinned bytes and the live model tree.

    No hash, receipt ID, adapter revision, or development contract digest is a
    caller-provided assertion.  Base evaluations use the sole canonical null
    seed/receipt binding.  Adapted evaluations revalidate the complete receipt
    and independently inspect the exact local adapter tree before any rollout.
    """

    from psse_env.dagger.release_factories import inspect_release_checkpoint
    from psse_env.dagger.study_manifest import (
        DEVELOPMENT_EVALUATION_PROTOCOL_CONTRACT,
        EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256,
        PINNED_BASE_MODEL_ID,
        PINNED_BASE_MODEL_REVISION,
        TRAINED_VARIANT_IDS,
        load_study_manifest,
        validate_study_artifact_binding,
    )

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    reviewed = str(reviewed_source_commit).strip()
    if re.fullmatch(r"[0-9a-f]{40}", reviewed) is None:
        raise ValueError("reviewed_source_commit must be lowercase 40-hex")
    source = git_source_state(root)
    if (
        source.get("release_eligible_source") is not True
        or source.get("source_commit") != reviewed
    ):
        raise ValueError(
            "study evaluation requires the exact reviewed clean source commit"
        )
    manifest = load_study_manifest(study_manifest_path, repo_root=root)
    normalized_variant = str(variant_id).strip()
    normalized_model_id = str(model_id).strip()
    normalized_revision = str(model_revision).strip().lower()
    receipt_id: str | None = None
    checkpoint_tree: str | None = None
    normalized_seed: int | None = None

    if normalized_variant == "base":
        if training_seed is not None or checkpoint_receipt_path is not None:
            raise ValueError(
                "base study evaluation requires canonical null seed and receipt"
            )
        if (
            normalized_model_id != PINNED_BASE_MODEL_ID
            or normalized_revision != PINNED_BASE_MODEL_REVISION
        ):
            raise ValueError("base study evaluation must use the pinned base model")
    elif normalized_variant in TRAINED_VARIANT_IDS:
        if (
            isinstance(training_seed, bool)
            or not isinstance(training_seed, int)
            or checkpoint_receipt_path is None
        ):
            raise ValueError(
                "trained study evaluation requires an integer seed and checkpoint receipt"
            )
        requested_checkpoint = Path(normalized_model_id).expanduser()
        if not requested_checkpoint.is_absolute():
            raise ValueError("trained study model_id must be an absolute adapter path")
        receipt, _ = _load_unlinked_regular_json_object(
            checkpoint_receipt_path,
            field="checkpoint receipt",
        )
        validate_study_artifact_binding(
            manifest,
            receipt,
            variant_id=normalized_variant,
            artifact_role="checkpoint",
            expected_source_commit=reviewed,
            expected_training_seed=training_seed,
        )
        inspection = inspect_release_checkpoint(requested_checkpoint)
        inspected_path = Path(str(inspection.get("path") or ""))
        if not inspected_path.is_absolute():
            raise ValueError("checkpoint inspection returned a non-absolute path")
        receipt_adapter = Path(str(receipt.get("adapter_path") or ""))
        try:
            receipt_adapter = receipt_adapter.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("checkpoint receipt adapter_path is not live") from exc
        if receipt_adapter != inspected_path:
            raise ValueError(
                "checkpoint receipt adapter_path differs from the live evaluated tree"
            )
        checkpoint_tree = str(inspection.get("tree_sha256") or "").lower()
        if not _is_sha256(checkpoint_tree):
            raise ValueError("checkpoint inspection returned no canonical tree hash")
        if (
            normalized_revision != checkpoint_tree
            or receipt.get("adapter_tree_sha256") != checkpoint_tree
        ):
            raise ValueError(
                "model revision, receipt tree, and live adapter tree must be identical"
            )
        normalized_model_id = str(inspected_path)
        receipt_id = str(receipt.get("checkpoint_receipt_id") or "")
        normalized_seed = training_seed
    else:
        raise ValueError(f"unknown study variant: {normalized_variant!r}")

    common: dict[str, Any] = {
        "artifact_role": (
            "development_evaluation" if diagnostic_only else "evaluation"
        ),
        "variant_id": normalized_variant,
        "study_manifest_sha256": manifest["manifest_sha256"],
        "reviewed_source_commit": reviewed,
        "model_id": normalized_model_id,
        "model_revision": normalized_revision,
        "checkpoint_receipt_id": receipt_id,
        "checkpoint_adapter_tree_sha256": checkpoint_tree,
        "training_seed": normalized_seed,
    }
    suite_path = Path(input_suite_path).expanduser().resolve(strict=True)
    bindings = manifest.get("bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    frozen = bindings.get("evaluation")
    frozen = frozen if isinstance(frozen, Mapping) else {}
    if diagnostic_only:
        if (
            development_holdout_manifest_path is None
            or development_holdout_generator_report_path is None
        ):
            raise ValueError(
                "development study evaluation requires holdout manifest and generator report"
            )
        holdout = _validate_development_holdout_for_study(
            holdout_path=input_suite_path,
            holdout_manifest_path=development_holdout_manifest_path,
            generator_report_path=development_holdout_generator_report_path,
            study_manifest=manifest,
            reviewed_source_commit=reviewed,
            repo_root=root,
        )
        expected_contract = _development_evaluation_contract(manifest)
        if stable_json_sha256(expected_contract) != (
            EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
        ):
            raise ValueError(
                "study manifest development evaluator digest is not canonical"
            )
        actual_contract = {
            "contract": DEVELOPMENT_EVALUATION_PROTOCOL_CONTRACT,
            "evaluation_protocol": "diagnostic_model_selection_only",
            "diagnostic_only": True,
            "input_suite_name": holdout["suite_name"],
            "evaluator_seed": evaluator_seed,
            "max_steps": max_steps,
            "required_suites": list(required_suites),
            "minimum_suites": minimum_suites,
            "minimum_episodes_per_suite": minimum_episodes_per_suite,
            "minimum_roots_per_suite": minimum_roots_per_suite,
            "exact_physical_roots": holdout[
                "development_holdout_physical_roots"
            ],
            "protocol": protocol,
            "release_qualification_allowed": False,
        }
        if stable_json_sha256(actual_contract) != stable_json_sha256(
            expected_contract
        ):
            raise ValueError(
                "development evaluator configuration differs from the exact "
                "preregistered contract"
            )
        common.update(
            {
                key: value
                for key, value in holdout.items()
                if key.startswith("development_holdout_")
            }
        )
        common.update(
            {
                "development_evaluation_contract_sha256": (
                    EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
                ),
                "evaluation_protocol": "diagnostic_model_selection_only",
            }
        )
    else:
        if (
            development_holdout_manifest_path is not None
            or development_holdout_generator_report_path is not None
        ):
            raise ValueError(
                "frozen-suite evaluation cannot claim development holdout inputs"
            )
        if file_sha256(suite_path) != frozen.get("suite_sha256"):
            raise ValueError("frozen study evaluation suite bytes differ from manifest")
        if evaluator_seed != frozen.get("evaluator_seed"):
            raise ValueError("frozen evaluator seed differs from study manifest")
        if max_steps != frozen.get("max_steps"):
            raise ValueError("frozen evaluator max_steps differs from study manifest")
        if protocol != "canonical":
            raise ValueError("frozen study evaluation requires canonical protocol")
        common.update(
            {
                "frozen_suite_sha256": frozen["suite_sha256"],
                "evaluation_policy_sha256": frozen["policy_sha256"],
            }
        )

    validate_study_artifact_binding(
        manifest,
        common,
        variant_id=normalized_variant,
        artifact_role=common["artifact_role"],
        expected_source_commit=reviewed,
        expected_training_seed=normalized_seed,
    )
    return copy.deepcopy(common)


def build_evaluation_provenance(
    *,
    input_suite_path: str | os.PathLike[str],
    environment_factory_spec: str,
    environment_factory: Callable[..., Any],
    policy_factory_spec: str,
    policy_factory: Callable[..., Any],
    case_loader_spec: str | None = None,
    case_loader: Callable[..., Any] | None = None,
    policy_identity: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    protocol: str = "canonical",
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build the immutable identity envelope for one closed-loop evaluation.

    Library callers may pass this envelope to :func:`write_evaluation_artifact`.
    The CLI always does so and refuses release execution when the envelope is
    incomplete or the source tree is dirty.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    suite_path = Path(input_suite_path).expanduser().resolve(strict=True)
    if bool(case_loader_spec) != (case_loader is not None):
        raise ValueError("case_loader_spec and case_loader must be supplied together")
    explicit_identity = str(policy_identity or "").strip() or None
    normalized_model_id = str(model_id or "").strip() or None
    normalized_revision = str(model_revision or "").strip() or None
    core: dict[str, Any] = {
        "provenance_schema_version": 1,
        "source_state": git_source_state(root),
        "input_suite": {
            "provided_path": str(Path(input_suite_path).expanduser()),
            "resolved_path": str(suite_path),
            "sha256": file_sha256(suite_path),
            "size_bytes": suite_path.stat().st_size,
        },
        "factories": {
            "environment": _callable_descriptor(
                environment_factory_spec,
                environment_factory,
                repo_root=root,
            ),
            "policy": _callable_descriptor(
                policy_factory_spec,
                policy_factory,
                repo_root=root,
            ),
            "case_loader": (
                _callable_descriptor(case_loader_spec, case_loader, repo_root=root)
                if case_loader_spec is not None and case_loader is not None
                else None
            ),
        },
        "policy_identity": {
            "explicit_policy_identity": explicit_identity,
            "model_id": normalized_model_id,
            "model_revision": normalized_revision,
        },
        "protocol_registry": _protocol_registry_descriptor(protocol),
        "runtime_environment": _runtime_environment_descriptor(),
        "evaluator_source": _source_path_descriptor(__file__, repo_root=root),
    }
    core["identity_sha256"] = stable_json_sha256(core)
    failures = _evaluation_provenance_failures(core)
    return {
        **core,
        "release_eligible": not failures,
        "release_failures": failures,
    }


def write_evaluation_artifact(
    result: EvaluationResult,
    output_path: str | os.PathLike[str],
    *,
    provenance: Mapping[str, Any] | None = None,
    diagnostic_only: bool = False,
    study_binding: Mapping[str, Any] | None = None,
    study_manifest_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Atomically persist a deterministic closed-loop evaluation report.

    The original two-argument library call remains valid.  Such an artifact is
    explicitly non-release because a bare :class:`EvaluationResult` cannot
    identify the executed policy, factories, source tree, or input suite.
    ``diagnostic_only=True`` is irreversible for the emitted artifact: it uses
    a distinct artifact type and is explicitly ineligible for release and
    training even when every runtime/provenance check otherwise passes.  A
    study binding is accepted only together with the byte-pinned manifest and
    is revalidated before its fields enter the content-addressed artifact.
    """

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    output = Path(output_path).expanduser()
    if not output.name:
        raise ValueError("output_path must name a JSON artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    recorded_provenance = (
        copy.deepcopy(dict(provenance)) if isinstance(provenance, Mapping) else None
    )
    release_failures = _evaluation_provenance_failures(recorded_provenance)
    configuration = result.suite_metrics.get("configuration")
    release_environment = (
        configuration.get("release_environment_validation")
        if isinstance(configuration, Mapping)
        else None
    )
    if (
        not isinstance(release_environment, Mapping)
        or release_environment.get("passed") is not True
    ):
        release_failures.append(_RELEASE_ENVIRONMENT_FAILURE)
    release_scenario_schema = (
        configuration.get("release_scenario_schema_validation")
        if isinstance(configuration, Mapping)
        else None
    )
    if (
        not isinstance(release_scenario_schema, Mapping)
        or set(release_scenario_schema) != {"passed", "scenario_schema_version"}
        or release_scenario_schema.get("passed") is not True
        or type(release_scenario_schema.get("scenario_schema_version")) is not int
        or release_scenario_schema.get("scenario_schema_version")
        != _SCENARIO_SCHEMA_VERSION
    ):
        release_failures.append(_RELEASE_SCENARIO_SCHEMA_FAILURE)
    policy_identity_validation = (
        configuration.get("policy_identity_validation")
        if isinstance(configuration, Mapping)
        else None
    )
    if (
        not isinstance(policy_identity_validation, Mapping)
        or policy_identity_validation.get("passed") is not True
    ):
        release_failures.append(_RELEASE_POLICY_IDENTITY_FAILURE)
    callback_validation = (
        configuration.get("custom_callback_validation")
        if isinstance(configuration, Mapping)
        else None
    )
    if callback_validation != {
        "passed": True,
        "physical_audit_callback": False,
        "tool_cost_callback": False,
    }:
        release_failures.append(_RELEASE_CALLBACK_FAILURE)
    if diagnostic_only:
        release_failures.append(
            "diagnostic-only evaluation artifacts are not release evidence"
        )
    # Keep failure ordering deterministic when a caller supplied provenance
    # that already carried one of the structural failures above.
    release_failures = list(dict.fromkeys(release_failures))
    if recorded_provenance is not None:
        recorded_provenance["release_eligible"] = not release_failures
        recorded_provenance["release_failures"] = release_failures
    payload: dict[str, Any] = {
        "artifact_schema_version": STUDY_EVALUATION_SCHEMA_VERSION,
        "artifact_type": (
            "closed_loop_diagnostic_evaluation"
            if diagnostic_only
            else "closed_loop_release_evaluation"
        ),
        "release_eligible": not release_failures,
        "release_failures": release_failures,
        "provenance": recorded_provenance,
        "evaluation": result.as_dict(),
    }
    if diagnostic_only:
        payload.update(
            {
                "diagnostic_only": True,
                "release_evidence_eligible": False,
                "training_eligible": False,
            }
        )
    if (study_binding is None) != (study_manifest_path is None):
        raise ValueError(
            "study_binding and study_manifest_path must be supplied together"
        )
    if study_binding is not None:
        from psse_env.dagger.study_manifest import (
            load_study_manifest,
            validate_study_artifact_binding,
        )

        binding = copy.deepcopy(dict(study_binding))
        reserved = set(payload) | {"content_sha256"}
        overlap = sorted(reserved & set(binding))
        if overlap:
            raise ValueError(
                "study binding cannot replace evaluator-owned fields: "
                + ", ".join(overlap)
            )
        expected_role = (
            "development_evaluation" if diagnostic_only else "evaluation"
        )
        if binding.get("artifact_role") != expected_role:
            raise ValueError(
                "study artifact role is inconsistent with diagnostic_only"
            )
        if not isinstance(recorded_provenance, Mapping):
            raise ValueError("study evaluation requires complete provenance")
        source_state = recorded_provenance.get("source_state")
        source_state = source_state if isinstance(source_state, Mapping) else {}
        if (
            source_state.get("release_eligible_source") is not True
            or source_state.get("source_commit")
            != binding.get("reviewed_source_commit")
        ):
            raise ValueError(
                "study binding source differs from executed clean provenance"
            )
        policy_identity = recorded_provenance.get("policy_identity")
        policy_identity = (
            policy_identity if isinstance(policy_identity, Mapping) else {}
        )
        for field_name in ("model_id", "model_revision"):
            if policy_identity.get(field_name) != binding.get(field_name):
                raise ValueError(
                    f"study binding {field_name} differs from executed policy identity"
                )
        input_suite = recorded_provenance.get("input_suite")
        input_suite = input_suite if isinstance(input_suite, Mapping) else {}
        expected_suite_hash = binding.get(
            "development_holdout_sha256"
            if diagnostic_only
            else "frozen_suite_sha256"
        )
        if input_suite.get("sha256") != expected_suite_hash:
            raise ValueError(
                "study binding suite hash differs from executed input provenance"
            )
        captured_suite, _, captured_suite_sha256 = (
            _load_unlinked_regular_json_value(
                str(input_suite.get("resolved_path") or ""),
                field="study evaluation input suite",
            )
        )
        if not isinstance(captured_suite, (list, dict)):
            raise ValueError(
                "study evaluation input suite must contain a list or object"
            )
        if captured_suite_sha256 != expected_suite_hash:
            raise ValueError(
                "study evaluation input bytes changed after binding"
            )
        if not isinstance(configuration, Mapping):
            raise ValueError("study evaluation result has no configuration")
        expected_fingerprint = fingerprint_evaluation_suites(
            captured_suite,
            seed=configuration.get("seed"),
            required_suites=configuration.get("required_suites"),
            minimum_suites=configuration.get("minimum_suites"),
            minimum_episodes_per_suite=configuration.get(
                "minimum_episodes_per_suite"
            ),
            minimum_roots_per_suite=configuration.get(
                "minimum_roots_per_suite"
            ),
            allow_diagnostic_development=diagnostic_only,
        )
        mismatched_fingerprint_fields = sorted(
            field_name
            for field_name, expected_value in expected_fingerprint.items()
            if configuration.get(field_name) != expected_value
        )
        if mismatched_fingerprint_fields:
            raise ValueError(
                "study evaluation result was not produced from the bound suite: "
                + ", ".join(mismatched_fingerprint_fields)
            )
        manifest = load_study_manifest(study_manifest_path)
        if diagnostic_only:
            from psse_env.dagger.study_manifest import (
                EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256,
                canonical_development_evaluation_contract,
            )

            canonical_development = (
                canonical_development_evaluation_contract()
            )
            suite_names = configuration.get("suite_names")
            actual_development = {
                "contract": canonical_development["contract"],
                "evaluation_protocol": binding.get("evaluation_protocol"),
                "diagnostic_only": True,
                "input_suite_name": (
                    suite_names[0]
                    if isinstance(suite_names, list) and len(suite_names) == 1
                    else None
                ),
                "evaluator_seed": configuration.get("seed"),
                "max_steps": configuration.get("max_steps"),
                "required_suites": configuration.get("required_suites"),
                "minimum_suites": configuration.get("minimum_suites"),
                "minimum_episodes_per_suite": configuration.get(
                    "minimum_episodes_per_suite"
                ),
                "minimum_roots_per_suite": configuration.get(
                    "minimum_roots_per_suite"
                ),
                "exact_physical_roots": binding.get(
                    "development_holdout_physical_roots"
                ),
                "protocol": (
                    recorded_provenance.get("protocol_registry", {}).get(
                        "protocol"
                    )
                    if isinstance(
                        recorded_provenance.get("protocol_registry"), Mapping
                    )
                    else None
                ),
                "release_qualification_allowed": False,
            }
            if (
                actual_development != canonical_development
                or stable_json_sha256(actual_development)
                != EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
            ):
                raise ValueError(
                    "executed development evaluator configuration differs "
                    "from the exact preregistered contract"
                )
        variant = str(binding.get("variant_id") or "")
        expected_seed = (
            None if variant == "base" else binding.get("training_seed")
        )
        validate_study_artifact_binding(
            manifest,
            {**payload, **binding},
            variant_id=variant,
            artifact_role=expected_role,
            expected_source_commit=str(
                binding.get("reviewed_source_commit") or ""
            ),
            expected_training_seed=expected_seed,
        )
        payload.update(binding)
    payload["content_sha256"] = _stable_hash(payload)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if study_binding is not None:
            # Study evidence is write-once.  A hard-link publication cannot
            # replace an existing file or dangling symlink, unlike os.replace.
            os.link(temporary_name, output)
            if os.name != "nt":
                directory_descriptor = os.open(
                    output.parent,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        else:
            os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return copy.deepcopy(payload)


def _load_import_spec(spec: str, *, field: str) -> Callable[..., Any]:
    normalized = str(spec).strip()
    module_name, separator, attribute_path = normalized.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"{field} must use MODULE:ATTRIBUTE syntax")
    value: Any = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"{field} must resolve to a callable")
    return value


def _load_scenario_suite_file(path: str | os.PathLike[str]) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, (list, dict)):
        raise ValueError("scenario suite JSON must contain a list or object")
    return payload


def _stderr_progress(record: Mapping[str, Any]) -> None:
    """Write one immediately visible, machine-readable release progress row."""

    print(
        "BC0_EVAL_PROGRESS "
        + json.dumps(
            policy_safe_copy(dict(record)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run closed-loop suites from JSON and persist a release artifact."""

    parser = argparse.ArgumentParser(
        description="Evaluate frozen closed-loop scenario suites and write JSON evidence."
    )
    parser.add_argument("--input", required=True, help="JSON suite mapping or scenario list")
    parser.add_argument("--output", required=True, help="Destination release JSON artifact")
    parser.add_argument(
        "--env-factory", required=True, help="Environment factory as MODULE:ATTRIBUTE"
    )
    parser.add_argument(
        "--policy-factory", required=True, help="Policy factory as MODULE:ATTRIBUTE"
    )
    parser.add_argument(
        "--case-loader", help="Optional physical-case loader as MODULE:ATTRIBUTE"
    )
    parser.add_argument(
        "--policy-identity",
        help=(
            "Explicit immutable policy identity for non-model policies, such as "
            "rule-expert-v3. Required unless --model-id and --model-revision are set."
        ),
    )
    parser.add_argument(
        "--model-id",
        help="Model/checkpoint repository or immutable checkpoint name.",
    )
    parser.add_argument(
        "--model-revision",
        help="Pinned 40-character Git or 64-character content digest.",
    )
    parser.add_argument(
        "--protocol",
        choices=("canonical", "controller"),
        default="canonical",
        help="Exact model-visible tool registry used by the evaluated policy.",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Development only: run from a dirty/untracked source tree, while "
            "marking the persisted artifact release-ineligible."
        ),
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "Persist a non-release diagnostic evaluation artifact. This can "
            "exercise a temporary failure-replay suite but can never satisfy "
            "a release or training-evidence gate."
        ),
    )
    parser.add_argument(
        "--study-manifest",
        help="Byte-pinned preregistered study manifest for bound model evaluation.",
    )
    parser.add_argument(
        "--study-variant",
        choices=("base", "bc0", "natural_dagger", "natural_dagger_probes"),
        help="Exact preregistered model variant represented by this evaluation.",
    )
    parser.add_argument(
        "--reviewed-source-commit",
        help="Externally reviewed lowercase 40-hex source commit for the study.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        help="Preregistered training seed; forbidden for the untrained base.",
    )
    parser.add_argument(
        "--checkpoint-receipt",
        help="Write-once checkpoint receipt required for every trained variant.",
    )
    parser.add_argument(
        "--development-holdout-manifest",
        help="Immutable generation manifest for a diagnostic development holdout.",
    )
    parser.add_argument(
        "--development-holdout-generator-report",
        help="Generator report byte-bound by the development holdout manifest.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--required-suite", action="append", default=None)
    parser.add_argument("--minimum-suites", type=int, default=1)
    parser.add_argument("--minimum-episodes-per-suite", type=int, default=1)
    parser.add_argument("--minimum-roots-per-suite", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    explicit_policy_identity = str(args.policy_identity or "").strip()
    model_id = str(args.model_id or "").strip()
    model_revision = str(args.model_revision or "").strip()
    if bool(model_id) != bool(model_revision):
        parser.error("--model-id and --model-revision must be supplied together")
    if model_revision and _IMMUTABLE_REVISION.fullmatch(model_revision) is None:
        parser.error(
            "--model-revision must be an immutable 40- or 64-character hex digest"
        )
    if not explicit_policy_identity and not (model_id and model_revision):
        parser.error(
            "release evaluation requires --policy-identity or both --model-id "
            "and --model-revision"
        )
    study_values = (
        args.study_variant,
        args.reviewed_source_commit,
        args.training_seed,
        args.checkpoint_receipt,
        args.development_holdout_manifest,
        args.development_holdout_generator_report,
    )
    if args.study_manifest is None:
        if any(value is not None for value in study_values):
            parser.error(
                "study binding arguments require --study-manifest"
            )
    else:
        if explicit_policy_identity:
            parser.error("study evaluations require an exact model identity")
        if args.study_variant is None or args.reviewed_source_commit is None:
            parser.error(
                "--study-manifest requires --study-variant and "
                "--reviewed-source-commit"
            )
        if args.allow_dirty_source:
            parser.error("study evaluations never permit --allow-dirty-source")
        if args.study_variant == "base":
            if args.training_seed is not None or args.checkpoint_receipt is not None:
                parser.error(
                    "base study evaluation requires null training seed and receipt"
                )
        elif args.training_seed is None or args.checkpoint_receipt is None:
            parser.error(
                "trained study evaluation requires --training-seed and "
                "--checkpoint-receipt"
            )
        development_inputs = (
            args.development_holdout_manifest,
            args.development_holdout_generator_report,
        )
        if args.diagnostic_only and not all(
            value is not None for value in development_inputs
        ):
            parser.error(
                "diagnostic study evaluation requires the development holdout "
                "manifest and generator report"
            )
        if not args.diagnostic_only and any(
            value is not None for value in development_inputs
        ):
            parser.error(
                "development holdout bindings require --diagnostic-only"
            )
    expected_policy_identity = _normalize_release_policy_identity(
        {
            "explicit_policy_identity": explicit_policy_identity or None,
            "model_id": model_id or None,
            "model_revision": model_revision or None,
        }
    )

    required_suites = (
        tuple(args.required_suite)
        if args.required_suite is not None
        else EVALUATION_SUITES
    )
    environment_factory = _load_import_spec(
        args.env_factory, field="env_factory"
    )
    policy_factory = _load_import_spec(
        args.policy_factory, field="policy_factory"
    )
    case_loader = (
        _load_import_spec(args.case_loader, field="case_loader")
        if args.case_loader
        else None
    )
    captured_study_suite: Any | None = None
    captured_study_suite_sha256: str | None = None
    if args.study_manifest is not None:
        (
            captured_study_suite,
            _,
            captured_study_suite_sha256,
        ) = _load_unlinked_regular_json_value(
            args.input,
            field="study evaluation input suite",
        )
        if not isinstance(captured_study_suite, (list, dict)):
            raise ValueError(
                "study evaluation input suite must contain a list or object"
            )
    provenance = build_evaluation_provenance(
        input_suite_path=args.input,
        environment_factory_spec=args.env_factory,
        environment_factory=environment_factory,
        policy_factory_spec=args.policy_factory,
        policy_factory=policy_factory,
        case_loader_spec=args.case_loader,
        case_loader=case_loader,
        policy_identity=explicit_policy_identity,
        model_id=model_id,
        model_revision=model_revision,
        protocol=args.protocol,
    )
    if (
        captured_study_suite_sha256 is not None
        and provenance.get("input_suite", {}).get("sha256")
        != captured_study_suite_sha256
    ):
        raise RuntimeError(
            "study evaluation input changed while provenance was being built"
        )
    provenance_failures = list(provenance["release_failures"])
    blocking_failures = [
        failure
        for failure in provenance_failures
        if failure != _RELEASE_SOURCE_FAILURE or not args.allow_dirty_source
    ]
    if blocking_failures:
        raise RuntimeError(
            "Closed-loop release identity gate failed: "
            + "; ".join(blocking_failures)
        )

    study_binding = None
    if args.study_manifest is not None:
        study_binding = build_study_evaluation_binding(
            study_manifest_path=args.study_manifest,
            variant_id=args.study_variant,
            reviewed_source_commit=args.reviewed_source_commit,
            model_id=model_id,
            model_revision=model_revision,
            input_suite_path=args.input,
            diagnostic_only=args.diagnostic_only,
            evaluator_seed=args.seed,
            max_steps=args.max_steps,
            required_suites=required_suites,
            minimum_suites=args.minimum_suites,
            minimum_episodes_per_suite=args.minimum_episodes_per_suite,
            minimum_roots_per_suite=args.minimum_roots_per_suite,
            protocol=args.protocol,
            training_seed=args.training_seed,
            checkpoint_receipt_path=args.checkpoint_receipt,
            development_holdout_manifest_path=(
                args.development_holdout_manifest
            ),
            development_holdout_generator_report_path=(
                args.development_holdout_generator_report
            ),
        )
        bound_suite_sha256 = study_binding.get(
            "development_holdout_sha256"
            if args.diagnostic_only
            else "frozen_suite_sha256"
        )
        if bound_suite_sha256 != captured_study_suite_sha256:
            raise RuntimeError(
                "study evaluation input changed while its binding was built"
            )

    result = evaluate_rollout_suites(
        (
            captured_study_suite
            if captured_study_suite is not None
            else _load_scenario_suite_file(args.input)
        ),
        env_factory=environment_factory,
        policy_factory=policy_factory,
        max_steps=args.max_steps,
        seed=args.seed,
        case_loader=case_loader,
        required_suites=required_suites,
        minimum_suites=args.minimum_suites,
        minimum_episodes_per_suite=args.minimum_episodes_per_suite,
        minimum_roots_per_suite=args.minimum_roots_per_suite,
        require_release_environment=True,
        expected_policy_identity=expected_policy_identity,
        require_policy_identity=True,
        development_holdout_mode=bool(
            args.study_manifest is not None and args.diagnostic_only
        ),
        progress_callback=_stderr_progress,
    )
    artifact = write_evaluation_artifact(
        result,
        args.output,
        provenance=provenance,
        diagnostic_only=args.diagnostic_only,
        study_binding=study_binding,
        study_manifest_path=args.study_manifest,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser()),
                "score": result.score,
                "content_sha256": artifact["content_sha256"],
                "release_eligible": artifact["release_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


def _is_audited_post_correction_handoff(
    *,
    terminal: bool,
    terminal_outcome: str | None,
    assessment: Any,
    scenario_id: str,
    physical_root_fingerprint: str,
    scenario_family: str,
    false_commit_count: int,
    false_rollback_count: int,
    false_finalization_count: int,
    loop_detected: bool,
    evaluator_error: str | None,
) -> bool:
    """Recognize only the versioned, safety-clean completion assessment."""

    assessment_valid, _ = validate_post_correction_handoff_assessment(
        assessment,
        scenario_id,
        physical_root_fingerprint,
        scenario_family,
    )
    return bool(
        terminal
        and terminal_outcome == "operator_escalation"
        and assessment_valid
        and false_commit_count == 0
        and false_rollback_count == 0
        and false_finalization_count == 0
        and not loop_detected
        and evaluator_error is None
    )


def _episode_has_audited_post_correction_handoff(
    episode: EpisodeEvaluation,
) -> bool:
    audit = episode.audit if isinstance(episode.audit, Mapping) else {}
    return _is_audited_post_correction_handoff(
        terminal=episode.terminal,
        terminal_outcome=episode.terminal_outcome,
        assessment=audit.get("post_correction_handoff_assessment"),
        scenario_id=episode.scenario_id,
        physical_root_fingerprint=episode.physical_root,
        scenario_family=episode.family,
        false_commit_count=episode.false_commit_count,
        false_rollback_count=episode.false_rollback_count,
        false_finalization_count=episode.false_finalization_count,
        loop_detected=episode.loop_detected,
        evaluator_error=episode.evaluator_error,
    )


def summarize_episode_evaluations(
    episodes: Iterable[EpisodeEvaluation],
) -> dict[str, Any]:
    rows = list(episodes)
    total = len(rows)
    terminal = sum(row.terminal for row in rows)
    # A resolved label is release evidence only when the strict physical audit
    # also succeeds.  Counting outcome strings alone made evaluator summaries
    # disagree with the fail-closed release gate for false finalizations.
    resolved = sum(row.final_physical_success for row in rows)
    escalated = sum(
        row.terminal and row.terminal_outcome == "operator_escalation"
        for row in rows
    )
    audited_handoff_flags = [
        _episode_has_audited_post_correction_handoff(row) for row in rows
    ]
    audited_handoffs = sum(audited_handoff_flags)
    audited_completions = sum(
        row.final_physical_success or audited_handoff
        for row, audited_handoff in zip(rows, audited_handoff_flags)
    )
    unqualified_escalations = sum(
        row.terminal
        and row.terminal_outcome == "operator_escalation"
        and not audited_handoff
        for row, audited_handoff in zip(rows, audited_handoff_flags)
    )
    physical_known = sum(row.physical_correctness_known for row in rows)
    physical_correct = sum(row.final_physical_correct for row in rows)
    physical_success = sum(row.final_physical_success for row in rows)
    healthy_known = sum(row.healthy_preservation_known for row in rows)
    healthy_preserved = sum(row.healthy_components_preserved for row in rows)
    false_commit_count = sum(row.false_commit_count for row in rows)
    false_rollback_count = sum(row.false_rollback_count for row in rows)
    false_finalization_count = sum(row.false_finalization_count for row in rows)
    partial_count = sum(row.partial_fix_count for row in rows)
    retained_partial = sum(row.retained_partial_fix_count for row in rows)
    invalid_count = sum(row.invalid_action_count for row in rows)
    recovered_invalid = sum(row.recovered_invalid_action_count for row in rows)
    injected_failures = sum(
        int(row.evaluation_intervention.get("injected_failure_count", 0))
        for row in rows
    )
    recovered_injected = sum(
        int(row.evaluation_intervention.get("recovered_failure_count", 0))
        for row in rows
    )
    control_quarantined = [
        row for row in rows if row.control_quarantine.get("quarantined") is True
    ]
    control_quarantine_reasons: Counter[str] = Counter(
        str(row.control_quarantine.get("failure_kind") or "unknown")
        for row in control_quarantined
    )
    regret_samples = sum(row.tool_regret_samples for row in rows)
    regret_total = sum(row.tool_regret_total for row in rows)
    tool_counts: Counter[str] = Counter()
    specialized_counts: Counter[str] = Counter()
    for row in rows:
        tool_counts.update(row.tool_counts)
        specialized_counts.update(row.specialized_tool_counts)
    return {
        "episodes": total,
        "steps": sum(row.steps for row in rows),
        "mean_steps": _rate(sum(row.steps for row in rows), total),
        "terminal_episodes": terminal,
        "terminal_rate": _rate(terminal, total),
        "resolved_episodes": resolved,
        "resolution_rate": _rate(resolved, total),
        "operator_escalation_episodes": escalated,
        "operator_escalation_rate": _rate(escalated, total),
        "audited_post_correction_handoff_episodes": audited_handoffs,
        "audited_post_correction_handoff_rate": _rate(audited_handoffs, total),
        "audited_completion_episodes": audited_completions,
        "audited_completion_rate": _rate(audited_completions, total),
        "unqualified_operator_escalation_episodes": unqualified_escalations,
        "unqualified_operator_escalation_rate": _rate(
            unqualified_escalations, total
        ),
        "unknown_terminal_outcome_episodes": sum(
            row.terminal and row.terminal_outcome not in {"resolved", "operator_escalation"}
            for row in rows
        ),
        "physical_correctness_known_episodes": physical_known,
        "physical_correct_episodes": physical_correct,
        # The all-episode rate fails closed when truth/audit evidence is absent.
        "final_physical_correctness_rate": _rate(physical_correct, total),
        "final_physical_correctness_known_rate": _rate(
            physical_correct, physical_known
        ),
        "final_physical_success_episodes": physical_success,
        "final_physical_success_rate": _rate(physical_success, total),
        "healthy_preservation_known_episodes": healthy_known,
        "healthy_component_preservation_episodes": healthy_preserved,
        "healthy_component_preservation_rate": _rate(healthy_preserved, total),
        "healthy_component_preservation_known_rate": _rate(
            healthy_preserved, healthy_known
        ),
        "healthy_component_corruption_episodes": sum(
            row.healthy_preservation_known
            and not row.healthy_components_preserved
            for row in rows
        ),
        "healthy_component_corruption_rate": _rate(
            sum(
                row.healthy_preservation_known
                and not row.healthy_components_preserved
                for row in rows
            ),
            total,
        ),
        "false_commit_count": false_commit_count,
        "false_commit_episodes": sum(row.false_commit_count > 0 for row in rows),
        "false_commit_rate": _rate(
            sum(row.false_commit_count > 0 for row in rows), total
        ),
        "false_rollback_count": false_rollback_count,
        "false_rollback_episodes": sum(row.false_rollback_count > 0 for row in rows),
        "false_rollback_rate": _rate(
            sum(row.false_rollback_count > 0 for row in rows), total
        ),
        "false_finalization_count": false_finalization_count,
        "false_finalization_episodes": sum(
            row.false_finalization_count > 0 for row in rows
        ),
        "false_finalization_rate": _rate(
            sum(row.false_finalization_count > 0 for row in rows), total
        ),
        "partial_fix_opportunities": partial_count,
        "retained_partial_fixes": retained_partial,
        "partial_fix_retention_rate": _rate(retained_partial, partial_count),
        "invalid_action_count": invalid_count,
        "recovered_invalid_actions": recovered_invalid,
        "invalid_action_recovery_rate": _rate(recovered_invalid, invalid_count),
        "injected_failure_count": injected_failures,
        "recovered_injected_failures": recovered_injected,
        "injected_failure_recovery_rate": _rate(
            recovered_injected, injected_failures
        ),
        "episodes_with_injected_failures": sum(
            int(row.evaluation_intervention.get("injected_failure_count", 0)) > 0
            for row in rows
        ),
        "episodes_with_invalid_actions": sum(
            row.invalid_action_count > 0 for row in rows
        ),
        "loop_episodes": sum(row.loop_detected for row in rows),
        "loop_rate": _rate(sum(row.loop_detected for row in rows), total),
        "control_quarantined_episodes": len(control_quarantined),
        "control_quarantine_rate": _rate(len(control_quarantined), total),
        "control_quarantine_reason_counts": dict(
            sorted(control_quarantine_reasons.items())
        ),
        "repeated_nonadvancing_failure_breaker_episodes": sum(
            row.control_quarantine.get("breaker_error_code")
            == _REPEATED_NONADVANCING_FAILURE_CIRCUIT_BREAKER
            for row in rows
        ),
        "wls_calls": sum(row.wls_calls for row in rows),
        "mean_wls_calls": _rate(sum(row.wls_calls for row in rows), total),
        "specialized_tool_calls": sum(row.specialized_tool_calls for row in rows),
        "mean_specialized_tool_calls": _rate(
            sum(row.specialized_tool_calls for row in rows), total
        ),
        "tool_counts": dict(sorted(tool_counts.items())),
        "specialized_tool_counts": dict(sorted(specialized_counts.items())),
        "tool_regret_samples": regret_samples,
        "tool_regret_total": regret_total,
        "mean_tool_regret": (
            regret_total / regret_samples if regret_samples else None
        ),
        "tool_regret_coverage": _rate(
            regret_samples, sum(row.steps for row in rows)
        ),
        "evaluator_error_episodes": sum(row.evaluator_error is not None for row in rows),
    }


def _recovery_metrics(summary: Mapping[str, Any]) -> RecoveryMetrics:
    return RecoveryMetrics(
        final_physical_success=float(summary["final_physical_success_rate"]),
        false_finalization=float(summary["false_finalization_rate"]),
        healthy_component_corruption=float(
            summary["healthy_component_corruption_rate"]
        ),
        forced_error_recovery=float(summary["injected_failure_recovery_rate"]),
        tool_regret=float(summary["mean_tool_regret"] or 0.0),
        partial_success_retention=float(summary["partial_fix_retention_rate"]),
        false_rollback=float(summary["false_rollback_rate"]),
        false_commit=float(summary["false_commit_rate"]),
        loop_rate=float(summary["loop_rate"]),
        final_physical_correctness=float(
            summary["final_physical_correctness_rate"]
        ),
        terminal_rate=float(summary["terminal_rate"]),
        resolution_rate=float(summary["resolution_rate"]),
        operator_escalation_rate=float(summary["operator_escalation_rate"]),
        audited_post_correction_handoff=float(
            summary["audited_post_correction_handoff_rate"]
        ),
        audited_completion=float(summary["audited_completion_rate"]),
        unqualified_operator_escalation=float(
            summary["unqualified_operator_escalation_rate"]
        ),
        healthy_component_preservation=float(
            summary["healthy_component_preservation_rate"]
        ),
        invalid_action_recovery=float(summary["invalid_action_recovery_rate"]),
        mean_wls_calls=float(summary["mean_wls_calls"]),
        mean_specialized_tool_calls=float(summary["mean_specialized_tool_calls"]),
    )


def _group_episodes(
    episodes: Sequence[EpisodeEvaluation], attribute: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EpisodeEvaluation]] = {}
    for episode in episodes:
        key = str(getattr(episode, attribute))
        grouped.setdefault(key, []).append(episode)
    return {
        key: summarize_episode_evaluations(grouped[key])
        for key in sorted(grouped)
    }


def _normalize_suites(
    scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
    | Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    raw = (
        dict(scenario_suites)
        if isinstance(scenario_suites, Mapping)
        else {"standard_success": scenario_suites}
    )
    if not raw:
        raise ValueError("evaluation requires at least one non-empty suite")
    suites: dict[str, list[dict[str, Any]]] = {}
    for name, scenarios in raw.items():
        suite_name = str(name).strip()
        if not suite_name:
            raise ValueError("evaluation suite names must be non-empty")
        if suite_name in suites:
            raise ValueError(f"duplicate normalized evaluation suite name: {suite_name}")
        rows: list[dict[str, Any]] = []
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, Mapping):
                raise TypeError(
                    f"Scenario {suite_name}[{index}] must be a mapping."
                )
            rows.append(copy.deepcopy(dict(scenario)))
        if not rows:
            raise ValueError(f"evaluation suite {suite_name!r} is empty")
        suites[suite_name] = rows
    return suites


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed < 1 or not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _normalize_required_suites(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("required_suites must be an iterable of suite names")
    names = [str(item).strip() for item in value]
    if any(not name for name in names):
        raise ValueError("required_suites cannot contain an empty name")
    if len(set(names)) != len(names):
        raise ValueError("required_suites cannot contain duplicates")
    return tuple(sorted(names))


def _validate_and_fingerprint_suites(
    suites: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_suites: Sequence[str],
    minimum_suites: int,
    minimum_episodes_per_suite: int,
    minimum_roots_per_suite: int | Mapping[str, int],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(suites) < minimum_suites:
        errors.append(
            f"suite_count={len(suites)} < minimum_suites={minimum_suites}"
        )
    missing = sorted(set(required_suites) - set(suites))
    if missing:
        errors.append("missing_required_suites=" + ",".join(missing))

    manifest: dict[str, dict[str, Any]] = {}
    all_roots: set[str] = set()
    if isinstance(minimum_roots_per_suite, Mapping):
        root_minimums = {
            str(name): _positive_integer(value, field=f"{name}.minimum_roots")
            for name, value in minimum_roots_per_suite.items()
        }
        if set(root_minimums) != set(required_suites):
            raise ValueError(
                "minimum roots mapping must contain exactly the required suites"
            )
    else:
        shared_minimum = _positive_integer(
            minimum_roots_per_suite, field="minimum_roots_per_suite"
        )
        root_minimums = {name: shared_minimum for name in suites}
    for suite_name in sorted(suites):
        rows = list(suites[suite_name])
        ordered = sorted(
            enumerate(rows),
            key=lambda item: (
                _scenario_id(item[1], item[0]),
                _stable_hash(item[1]),
            ),
        )
        ordered_payload = [row for _, row in ordered]
        roots = sorted(
            {_scenario_groups(row)["physical_root"] for row in ordered_payload}
        )
        all_roots.update(roots)
        if len(rows) < minimum_episodes_per_suite:
            errors.append(
                f"{suite_name}: episodes={len(rows)} < "
                f"minimum_episodes_per_suite={minimum_episodes_per_suite}"
            )
        required_root_count = root_minimums.get(suite_name, 1)
        if len(roots) < required_root_count:
            errors.append(
                f"{suite_name}: distinct_roots={len(roots)} < "
                f"minimum_roots_per_suite={required_root_count}"
            )
        manifest[suite_name] = {
            "episodes": len(rows),
            "distinct_physical_roots": len(roots),
            "content_sha256": _stable_hash(ordered_payload),
            "root_set_sha256": _stable_hash(roots),
        }
    if errors:
        raise ValueError("evaluation suite coverage failed: " + "; ".join(errors))

    content_hashes = {
        name: details["content_sha256"] for name, details in manifest.items()
    }
    root_hashes = {
        name: details["root_set_sha256"] for name, details in manifest.items()
    }
    return {
        "suite_manifest": manifest,
        "suite_content_hashes": content_hashes,
        "suite_root_set_hashes": root_hashes,
        "suite_content_sha256": _stable_hash(content_hashes),
        "root_set_sha256": _stable_hash(sorted(all_roots)),
        "suite_coverage_validation": {
            "passed": True,
            "suite_count": len(suites),
            "distinct_physical_roots": len(all_roots),
        },
    }


def fingerprint_evaluation_suites(
    scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
    | Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    required_suites: Iterable[str] | None = None,
    minimum_suites: int = 1,
    minimum_episodes_per_suite: int = 1,
    minimum_roots_per_suite: int | Mapping[str, int] = 1,
    allow_diagnostic_development: bool = False,
) -> dict[str, Any]:
    """Return the canonical semantic identity used by evaluator and gate."""

    suites = _normalize_suites(scenario_suites)
    normalized_required = _normalize_required_suites(required_suites)
    fingerprint = _validate_and_fingerprint_suites(
        suites,
        required_suites=normalized_required,
        minimum_suites=_positive_integer(minimum_suites, field="minimum_suites"),
        minimum_episodes_per_suite=_positive_integer(
            minimum_episodes_per_suite, field="minimum_episodes_per_suite"
        ),
        minimum_roots_per_suite=minimum_roots_per_suite,
    )
    episode_manifest: list[dict[str, Any]] = []
    for suite_name in sorted(suites):
        ordered = sorted(
            enumerate(suites[suite_name]),
            key=lambda item: (
                _scenario_id(item[1], item[0]),
                _stable_hash(strip_offline_truth(item[1])),
            ),
        )
        occurrence_by_id: Counter[str] = Counter()
        for original_index, scenario in ordered:
            scenario_id = _scenario_id(scenario, original_index)
            occurrence = occurrence_by_id[scenario_id]
            occurrence_by_id[scenario_id] += 1
            groups = _scenario_groups(scenario)
            episode_manifest.append(
                {
                    "episode_key": f"{suite_name}:{scenario_id}:{occurrence}",
                    "scenario_id": scenario_id,
                    "scenario_index": occurrence,
                    "suite": suite_name,
                    "family": groups["family"],
                    "cardinality": groups["cardinality"],
                    "case": groups["case"],
                    "split": groups["split"],
                    "source_tier": groups["source_tier"],
                    "physical_root": groups["physical_root"],
                    "seed": _episode_seed(
                        int(seed), suite_name, scenario_id, occurrence
                    ),
                    "evaluation_intervention": evaluation_intervention_contract(
                        suite_name,
                        scenario,
                        required=False,
                        allow_diagnostic_development=(
                            allow_diagnostic_development
                        ),
                    ),
                }
            )
    return {
        **fingerprint,
        "suite_names": sorted(suites),
        "episode_order": [row["episode_key"] for row in episode_manifest],
        "episode_manifest": episode_manifest,
        "episode_manifest_sha256": _stable_hash(episode_manifest),
    }


def load_evaluation_suites(path: str | os.PathLike[str]) -> dict[str, list[dict[str, Any]]]:
    """Load and normalize a JSON evaluation-suite file."""

    payload = _load_scenario_suite_file(path)
    return _normalize_suites(payload)


def validate_release_scenario_suites(
    scenario_suites: Mapping[str, Iterable[Mapping[str, Any]]]
    | Iterable[Mapping[str, Any]],
    *,
    allow_diagnostic_development: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Require the canonical versioned execution/audit/grouping release schema."""

    suites = _normalize_suites(scenario_suites)
    problems: list[str] = []
    for suite_name in sorted(suites):
        for index, scenario in enumerate(suites[suite_name]):
            if not _has_partition_marker(scenario):
                problems.append(
                    f"{suite_name}[{index}] is not a versioned partitioned scenario"
                )
                continue
            try:
                execution, _, _ = _partitioned_scenario_parts(scenario)
                evaluation_intervention_contract(
                    suite_name,
                    scenario,
                    allow_diagnostic_development=(
                        allow_diagnostic_development
                    ),
                )
                development_fields = sorted(
                    set(execution) & {"initial_physical_state", "script"}
                )
                if development_fields:
                    raise ValueError(
                        "release execution contains development-only fields: "
                        + ", ".join(development_fields)
                    )
            except (TypeError, ValueError) as exc:
                problems.append(f"{suite_name}[{index}]: {exc}")
    if problems:
        raise ValueError("release scenario schema validation failed: " + "; ".join(problems))
    return suites


def _scenario_id(scenario: Mapping[str, Any], index: int) -> str:
    execution = scenario.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    grouping = scenario.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    explicit = scenario.get(
        "scenario_id",
        grouping.get(
            "scenario_id", execution.get("scenario_id", execution.get("id"))
        ),
    )
    if explicit is not None:
        return str(explicit)
    # A content-derived fallback keeps evaluation invariant to input ordering
    # without allowing hidden audit truth to alter the execution seed.
    return f"scenario_{_stable_hash(strip_offline_truth(scenario))[:12]}"


def _episode_seed(base_seed: int, suite: str, scenario_id: str, index: int) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [int(base_seed), str(suite), str(scenario_id), int(index)],
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _call_factory(
    factory: Callable[..., Any],
    seed: int,
    *,
    policy_identity: Mapping[str, Any] | None = None,
) -> Any:
    """Call a factory without ever supplying scenario data."""

    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if "seed" in parameters or accepts_kwargs:
        kwargs["seed"] = int(seed)
    if "rng" in parameters or accepts_kwargs:
        kwargs["rng"] = random.Random(seed)
    # Identity-bearing policy factories receive only explicitly declared
    # identity parameters.  The instantiated object must independently expose
    # the same values through ``release_policy_identity``.
    if isinstance(policy_identity, Mapping):
        explicit = policy_identity.get("explicit_policy_identity")
        model_id = policy_identity.get("model_id")
        model_revision = policy_identity.get("model_revision")
        if "policy_identity" in parameters and explicit is not None:
            kwargs["policy_identity"] = explicit
        if "model_id" in parameters and model_id is not None:
            kwargs["model_id"] = model_id
        if "model_revision" in parameters and model_revision is not None:
            kwargs["model_revision"] = model_revision
    return factory(**kwargs)


def _policy_action(policy: Any, observation: Mapping[str, Any]) -> Any:
    safe_observation = copy.deepcopy(dict(observation))
    if hasattr(policy, "act"):
        return policy.act(safe_observation)
    if hasattr(policy, "next_actions"):
        # ExpertPolicyOracle supports policy-observation input directly.  Do
        # not substitute an OracleState here: the expert comparator must obey
        # the same observation boundary as learned policies.
        actions = policy.next_actions(
            safe_observation,
            copy.deepcopy(list(safe_observation.get("history_window") or [])),
        )
        return actions[0] if actions else invalid_action("policy_returned_no_action")
    if callable(policy):
        return policy(safe_observation)
    raise TypeError(
        "Policy must be callable or expose .act(observation) / .next_actions(observation)."
    )


def _policy_action_metrics(policy: Any) -> dict[str, Any]:
    """Read an optional policy-safe timing/token telemetry mapping."""

    try:
        metrics = getattr(policy, "last_action_metrics", None)
        if callable(metrics):
            metrics = metrics()
        if not isinstance(metrics, Mapping):
            return {}
        allowed = {
            "prompt_tokens",
            "generated_tokens",
            "generation_seconds",
            "hit_max_new_tokens",
            "last_token_id",
        }
        sanitized: dict[str, Any] = {}
        for key in sorted(allowed & set(metrics)):
            value = metrics[key]
            if value is None or isinstance(value, bool):
                sanitized[key] = value
            elif isinstance(value, int):
                sanitized[key] = int(value)
            elif isinstance(value, float) and math.isfinite(value):
                sanitized[key] = float(value)
            else:
                # Telemetry is deliberately scalar-only. In particular, never
                # deep-copy arbitrary model objects while evaluating a policy.
                return {}
        return sanitized
    except Exception:
        return {}


def _policy_observation(
    env: Any, history: list[Mapping[str, Any]]
) -> dict[str, Any]:
    if hasattr(env, "get_policy_observation"):
        raw = _call_with_optional_argument(env.get_policy_observation, history)
        if isinstance(raw, PolicyObservation):
            return raw.as_dict()
        if isinstance(raw, Mapping):
            return copy.deepcopy(dict(raw))
        if hasattr(raw, "as_dict"):
            payload = raw.as_dict()
            if isinstance(payload, Mapping):
                return copy.deepcopy(dict(payload))
        raise TypeError("get_policy_observation() must return a mapping.")
    state = _current_state(env)
    return PolicyObservation(
        active_state_id=str(state.get("active_state_id") or "active"),
        candidate_state_id=state.get("candidate_state_id"),
        remaining_budget=int(state.get("remaining_budget") or 0),
        history_window=policy_safe_copy(history),
    ).as_dict()


def _oracle_state(env: Any, history: list[Mapping[str, Any]]) -> Any:
    if not hasattr(env, "get_oracle_state"):
        return None
    return copy.deepcopy(_call_with_optional_argument(env.get_oracle_state, history))


def _current_state(env: Any) -> dict[str, Any]:
    if not hasattr(env, "current_state"):
        return {}
    state = env.current_state()
    if not isinstance(state, Mapping):
        raise TypeError("current_state() must return a mapping.")
    return copy.deepcopy(dict(state))


def _independent_handoff_process_label(
    env: Any,
    state_before: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Independently certify the one terminal action used by the handoff audit.

    The returned label is audit-only: it is never appended to the rollout
    history supplied to the policy.  ``None`` means the action is not the
    reviewed handoff or that a test double has no process oracle; an empty
    mapping means an available checker failed to produce a canonical label and
    must not be replaced by an environment self-report.
    """

    normalized = safe_normalize_action(action)
    if not (
        normalized["tool"] == ASK_FOR_MORE_EVIDENCE
        and normalized["arguments"].get("request")
        == RECOVERY_OPTIONS_EXHAUSTED_REQUEST
    ):
        return None

    checker = getattr(getattr(env, "process_oracle", None), "check", None)
    if not callable(checker):
        return None

    validity_state = copy.deepcopy(dict(state_before))
    validity_state["require_context_supported_corrections"] = bool(
        getattr(env, "production_dataset_mode", False)
    )
    validity_state["audited_evaluation_setup_correction"] = bool(
        getattr(env, "_audited_evaluation_setup_correction", False)
    )
    try:
        raw_label = checker(
            validity_state,
            copy.deepcopy(normalized),
            store=getattr(env, "store", None),
        )
    except Exception:
        return {}

    required_fields = {
        "process_valid",
        "reason",
        "error_code",
        "error_detail",
        "valid_next_actions",
    }
    if not isinstance(raw_label, Mapping) or not required_fields <= set(raw_label):
        return {}
    if not isinstance(raw_label["process_valid"], bool):
        return {}
    if any(
        raw_label[field] is not None and not isinstance(raw_label[field], str)
        for field in ("reason", "error_code", "error_detail")
    ):
        return {}
    if not isinstance(raw_label["valid_next_actions"], list):
        return {}
    if raw_label["process_valid"] and (
        any(
            raw_label[field] is not None
            for field in ("reason", "error_code", "error_detail")
        )
        or raw_label["valid_next_actions"] != []
    ):
        return {}
    return policy_safe_copy(
        {field: raw_label[field] for field in sorted(required_fields)}
    )


def _audit_bound_terminal_state(
    final_state: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    independent_transition_label: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind the handoff certificate to the evaluator-owned final transition.

    Production ``current_state()`` intentionally omits history.  Reconstruct
    only the terminal audit view from the exact evaluator transition.  A test
    double may supply its own label only when every non-label transition field
    exactly matches the evaluator record and no independent checker existed.
    """

    bound = copy.deepcopy(dict(final_state))
    if not history or not isinstance(history[-1], Mapping):
        return bound

    terminal_transition = copy.deepcopy(dict(history[-1]))
    label: dict[str, Any] | None = None
    if independent_transition_label is not None:
        # This includes an empty mapping.  A failed independent check must not
        # fall back to a forged or stale environment-provided valid label.
        label = copy.deepcopy(dict(independent_transition_label))
    else:
        state_history = bound.get("history_window")
        existing = (
            state_history[-1]
            if isinstance(state_history, list)
            and state_history
            and isinstance(state_history[-1], Mapping)
            else None
        )
        if existing is not None and all(
            existing.get(field) == terminal_transition.get(field)
            for field in (
                "state_id",
                "candidate_state_id",
                "action",
                "tool_output",
            )
        ):
            existing_label = existing.get("transition_label")
            if isinstance(existing_label, Mapping):
                label = copy.deepcopy(dict(existing_label))

    if label is not None:
        terminal_transition["transition_label"] = label
    bound["history_window"] = [terminal_transition]
    return bound


def _active_physical_state(
    env: Any, final_state: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return the final store payload without treating a policy summary as physics."""

    store = getattr(env, "store", None)
    getter = getattr(store, "get_state_for_audit", None)
    if not callable(getter):
        getter = getattr(store, "get_state", None)
    state_id = final_state.get("active_state_id")
    if state_id is None and store is not None:
        state_id = getattr(store, "active_state_id", None)
    if callable(getter) and state_id is not None:
        try:
            payload = getter(str(state_id))
        except (KeyError, TypeError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            return copy.deepcopy(dict(payload))
    if _physical_state_available(final_state):
        return copy.deepcopy(dict(final_state))
    return None


def _successful_action_advanced(
    *,
    before: Mapping[str, Any],
    after: Any,
    output: Mapping[str, Any],
    terminal: bool,
) -> bool:
    """Recognize observable state/decision progress after a successful action."""

    if terminal or output.get("state_mutated") is True:
        return True
    if not isinstance(after, Mapping):
        return False
    for key in ("active_state_id", "candidate_state_id", "phase"):
        if before.get(key) != after.get(key):
            return True
    for key in ("accepted_corrections", "explained_anomalies"):
        before_rows = before.get(key)
        after_rows = after.get(key)
        before_count = (
            len(before_rows)
            if isinstance(before_rows, Sequence)
            and not isinstance(before_rows, (str, bytes))
            else 0
        )
        after_count = (
            len(after_rows)
            if isinstance(after_rows, Sequence)
            and not isinstance(after_rows, (str, bytes))
            else 0
        )
        if after_count > before_count:
            return True
    return False


_TRACE_STATE_FIELDS = frozenset(
    {
        "active_state_id",
        "candidate_state_id",
        "phase",
        "accepted_correction_count",
        "explained_anomaly_count",
    }
)
_TRACE_PROGRESS_FIELDS = frozenset(
    {
        "state_before",
        "state_after",
        "state_before_sha256",
        "state_after_sha256",
        "state_mutated",
        "terminal_after",
    }
)


def _trace_state_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the observable lifecycle fields used to classify progress."""

    def optional_text(value: Any) -> str | None:
        if value is None:
            return None
        return str(getattr(value, "value", value))

    def sequence_count(value: Any) -> int:
        return (
            len(value)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            else 0
        )

    return {
        "active_state_id": optional_text(state.get("active_state_id")),
        "candidate_state_id": optional_text(state.get("candidate_state_id")),
        "phase": optional_text(state.get("phase")),
        "accepted_correction_count": sequence_count(
            state.get("accepted_corrections")
        ),
        "explained_anomaly_count": sequence_count(
            state.get("explained_anomalies")
        ),
    }


def trace_progress_evidence(
    *,
    before: Mapping[str, Any],
    after: Any,
    output: Mapping[str, Any],
    terminal: bool,
) -> dict[str, Any]:
    """Build stable, chainable evidence for one trace row's progress flag.

    The hashes cover the exact persisted lifecycle snapshots used by
    :func:`_successful_action_advanced`. This makes the binding independently
    recomputable: a consumer never has to trust an opaque hash of runtime
    fields that were not retained in the artifact.
    """

    before_state = copy.deepcopy(dict(before))
    after_state = (
        copy.deepcopy(dict(after)) if isinstance(after, Mapping) else before_state
    )
    before_snapshot = _trace_state_snapshot(before_state)
    after_snapshot = _trace_state_snapshot(after_state)
    return {
        "state_before": before_snapshot,
        "state_after": after_snapshot,
        "state_before_sha256": _stable_hash(before_snapshot),
        "state_after_sha256": _stable_hash(after_snapshot),
        "state_mutated": output.get("state_mutated") is True,
        "terminal_after": bool(terminal),
    }


def trace_progress_advanced(evidence: Mapping[str, Any]) -> bool:
    """Validate trace progress evidence and derive its advancement decision."""

    if not isinstance(evidence, Mapping) or not _TRACE_PROGRESS_FIELDS.issubset(
        evidence
    ):
        raise ValueError("trace progress evidence is incomplete")
    before = evidence.get("state_before")
    after = evidence.get("state_after")
    if (
        not isinstance(before, Mapping)
        or set(before) != _TRACE_STATE_FIELDS
        or not isinstance(after, Mapping)
        or set(after) != _TRACE_STATE_FIELDS
    ):
        raise ValueError("trace progress state snapshot has an invalid schema")
    for label, snapshot in (("before", before), ("after", after)):
        for key in ("active_state_id", "candidate_state_id", "phase"):
            value = snapshot.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"trace progress {label}.{key} must be a string or null"
                )
        for key in ("accepted_correction_count", "explained_anomaly_count"):
            value = snapshot.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"trace progress {label}.{key} must be a non-negative integer"
                )
    before_hash = evidence.get("state_before_sha256")
    after_hash = evidence.get("state_after_sha256")
    if (
        not isinstance(before_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", before_hash) is None
        or not isinstance(after_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", after_hash) is None
    ):
        raise ValueError("trace progress state hashes must be lowercase SHA-256")
    if before_hash != _stable_hash(dict(before)):
        raise ValueError(
            "trace progress state_before_sha256 is not bound to state_before"
        )
    if after_hash != _stable_hash(dict(after)):
        raise ValueError(
            "trace progress state_after_sha256 is not bound to state_after"
        )
    state_mutated = evidence.get("state_mutated")
    terminal_after = evidence.get("terminal_after")
    if not isinstance(state_mutated, bool) or not isinstance(terminal_after, bool):
        raise ValueError(
            "trace progress state_mutated and terminal_after must be booleans"
        )
    if before != after and before_hash == after_hash:
        raise ValueError("trace progress changed lifecycle state without changing hash")
    if terminal_after or state_mutated:
        return True
    for key in ("active_state_id", "candidate_state_id", "phase"):
        if before.get(key) != after.get(key):
            return True
    for key in ("accepted_correction_count", "explained_anomaly_count"):
        if int(after[key]) > int(before[key]):
            return True
    return False


def _is_terminal(env: Any, state: Mapping[str, Any] | None) -> bool:
    if hasattr(env, "is_terminal"):
        return bool(_call_with_optional_argument(env.is_terminal, state))
    return bool(getattr(env, "terminal", False))


def _call_with_optional_argument(method: Callable[..., Any], value: Any) -> Any:
    """Invoke bound environment methods that support either zero or one input."""

    try:
        parameters = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return method(value)
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return method(value)
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if positional:
        return method(value)
    keyword_only = [
        parameter
        for parameter in parameters
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    if len(keyword_only) == 1:
        return method(**{keyword_only[0].name: value})
    return method()


def _terminal_outcome(env: Any, trace: Sequence[Mapping[str, Any]]) -> str | None:
    outcome = getattr(env, "terminal_outcome", None)
    if outcome:
        return str(outcome)
    for row in reversed(trace):
        candidate = row.get("terminal_outcome")
        if candidate:
            return str(candidate)
    return None


def _output_terminal_outcome(output: Mapping[str, Any]) -> str | None:
    direct = output.get("terminal_outcome")
    metrics = output.get("tool_metrics")
    nested = metrics.get("terminal_outcome") if isinstance(metrics, Mapping) else None
    value = direct if direct is not None else nested
    return str(value) if value is not None else None


def _output_runtime_state_hash(output: Mapping[str, Any]) -> str | None:
    """Persist the tool-reported physical-state hash for offline binding."""

    metrics = output.get("tool_metrics")
    value = metrics.get("state_hash") if isinstance(metrics, Mapping) else None
    return str(value) if value is not None else None


def _canonical_study_policy_observation(
    observation: Mapping[str, Any],
    *,
    state_before: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the policy input to evaluator-owned state and transition history.

    A custom environment may supply the observable state summary, but it does
    not get to choose or fabricate the history used by schema-v4 objective
    metrics.  The evaluator replaces that field with the canonical bounded
    suffix and rejects state identifiers that contradict its runtime snapshot.
    """

    bound = copy.deepcopy(dict(observation))
    snapshot = _trace_state_snapshot(state_before)
    if not isinstance(bound.get("active_state_id"), str) or not str(
        bound.get("active_state_id")
    ).strip():
        raise ValueError("policy observation lacks a nonempty active_state_id")
    for name in ("active_state_id", "candidate_state_id"):
        if name not in bound or bound.get(name) != snapshot.get(name):
            raise ValueError(
                f"policy observation {name} contradicts the runtime state"
            )
    if (
        snapshot.get("phase") is not None
        and "phase" in bound
        and bound.get("phase") != snapshot.get("phase")
    ):
        raise ValueError("policy observation phase contradicts the runtime state")
    canonical_history = policy_safe_copy(
        list(history)[-STUDY_POLICY_HISTORY_WINDOW:]
    )
    bound["history_window"] = canonical_history
    last_transition = canonical_history[-1] if canonical_history else None
    last_action = (
        last_transition.get("action")
        if isinstance(last_transition, Mapping)
        else None
    )
    last_output = (
        last_transition.get("tool_output")
        if isinstance(last_transition, Mapping)
        else None
    )
    bound["last_tool"] = (
        last_action.get("tool") if isinstance(last_action, Mapping) else None
    )
    bound["last_tool_status"] = (
        last_output.get("execution_status")
        if isinstance(last_output, Mapping)
        else None
    )
    bound["last_tool_output"] = (
        copy.deepcopy(dict(last_output))
        if isinstance(last_output, Mapping)
        else {}
    )
    return bound


def _canonical_trace_transition(
    *,
    state_before: Mapping[str, Any],
    action: Mapping[str, Any],
    policy_tool_output: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "state_id": state_before.get("active_state_id"),
        "candidate_state_id": state_before.get("candidate_state_id"),
        "action": copy.deepcopy(dict(action)),
        "tool_output": copy.deepcopy(dict(policy_tool_output)),
    }


def _validate_objective_tool_binding(
    *,
    action: Mapping[str, Any],
    execution_status: Any,
    runtime_state_hash: Any,
    policy_tool_output: Mapping[str, Any],
    reported_evidence: Any,
    label: str,
) -> dict[str, Any] | None:
    expected = objective_tool_evidence(action, policy_tool_output)
    if reported_evidence != expected:
        raise ValueError(
            f"{label}.objective_tool_evidence is not reproducible from "
            "policy_tool_output"
        )
    if expected is None:
        return None

    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    state_id = expected.get("state_id")
    state_hash = expected.get("state_hash")
    computed_runtime_hash = _output_runtime_state_hash(policy_tool_output)
    if runtime_state_hash != computed_runtime_hash:
        raise ValueError(
            f"{label}.runtime_state_hash is not reproducible from "
            "policy_tool_output"
        )
    if runtime_state_hash is not None and (
        not isinstance(runtime_state_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_state_hash) is None
    ):
        raise ValueError(
            f"{label}.runtime_state_hash must be null or lowercase SHA-256"
        )
    if execution_status == "success":
        if (
            not isinstance(state_id, str)
            or not state_id.strip()
            or state_id != arguments.get("state_id")
        ):
            raise ValueError(
                f"{label}.objective_tool_evidence lacks exact action-state binding"
            )
        if (
            not isinstance(state_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", state_hash) is None
            or runtime_state_hash != state_hash
        ):
            raise ValueError(
                f"{label}.objective_tool_evidence lacks exact non-null state-hash binding"
            )
    else:
        if state_id is not None and (
            not isinstance(state_id, str)
            or not state_id.strip()
            or state_id != arguments.get("state_id")
        ):
            raise ValueError(
                f"{label}.objective_tool_evidence state_id is inconsistent"
            )
        if state_hash is not None and (
            not isinstance(state_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", state_hash) is None
            or runtime_state_hash != state_hash
        ):
            raise ValueError(
                f"{label}.objective_tool_evidence state_hash is inconsistent"
            )

    for name in (
        "chi_square_statistic",
        "chi_square_threshold",
        "max_normalized_residual",
    ):
        value = expected.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label}.objective_tool_evidence.{name} is invalid")
    for name in (
        "no_material_anomaly_remaining",
        "globally_resolved",
        "physical_constraints_ok",
        "physical_evidence_complete",
        "power_flow_converged",
        "topology_feasible",
    ):
        value = expected.get(name)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{label}.objective_tool_evidence.{name} is invalid")
    for name in ("evidence_source", "physical_evidence_scope"):
        value = expected.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{label}.objective_tool_evidence.{name} is invalid")
    if expected.get("physical_bound_violations") is not None and not isinstance(
        expected.get("physical_bound_violations"), list
    ):
        raise ValueError(
            f"{label}.objective_tool_evidence.physical_bound_violations is invalid"
        )
    if expected.get("steady_state_physical_evidence") is not None and not isinstance(
        expected.get("steady_state_physical_evidence"), Mapping
    ):
        raise ValueError(
            f"{label}.objective_tool_evidence.steady_state_physical_evidence is invalid"
        )
    return copy.deepcopy(expected)


def validate_study_objective_episode_evidence(
    episode: Mapping[str, Any],
    *,
    scenario_family: Any | None = None,
    error_cardinality: Any | None = None,
    label: str = "episode",
) -> dict[str, Any]:
    """Recompute schema-v4 objective evidence from canonical trace inputs.

    This pure validator is the shared release-gate and study-ingestion
    boundary.  It ignores aggregate objective claims and derives policy
    history, action assessments, leakage, and narrow tool certificates from
    the persisted transition rows.  The only non-trace inputs are the
    evaluator-owned initial/final state identities in the privileged offline
    audit, which anchor the trace endpoints.
    """

    if not isinstance(episode, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if episode.get("objective_evidence") != study_objective_episode_evidence_marker():
        raise ValueError(f"{label}.objective_evidence does not satisfy schema v4")
    trace = episode.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"{label}.trace must be a nonempty array")
    audit = episode.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError(f"{label}.audit must be a mapping")
    initial_active_state_id = audit.get("initial_active_state_id")
    final_active_state_id = audit.get("final_active_state_id")
    for anchor_name, anchor_value in (
        ("initial_active_state_id", initial_active_state_id),
        ("final_active_state_id", final_active_state_id),
    ):
        if not isinstance(anchor_value, str) or not anchor_value.strip():
            raise ValueError(
                f"{label}.audit.{anchor_name} must be a nonempty evaluator-owned state identity"
            )
    family = episode.get("family") if scenario_family is None else scenario_family
    cardinality = (
        episode.get("cardinality")
        if error_cardinality is None
        else error_cardinality
    )
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"{label}.family is unavailable")
    if (
        isinstance(cardinality, bool)
        or not isinstance(cardinality, int)
        or cardinality < 0
    ):
        raise ValueError(f"{label}.cardinality is unavailable")
    intervention_evidence = episode.get("evaluation_intervention")
    if not isinstance(intervention_evidence, Mapping):
        raise ValueError(f"{label}.evaluation_intervention must be a mapping")
    partial_count = intervention_evidence.get("retention_opportunity_count", 0)
    if (
        isinstance(partial_count, bool)
        or not isinstance(partial_count, int)
        or partial_count not in {0, 1}
    ):
        raise ValueError(
            f"{label}.evaluation_intervention.retention_opportunity_count is invalid"
        )
    if partial_count and episode.get("suite") != "partial_success_retention":
        raise ValueError(f"{label} has a noncanonical partial-success opportunity")

    history: list[dict[str, Any]] = []
    derived_rows: list[dict[str, Any]] = []
    leakage_paths: list[str] = []
    policy_ordinal = 0
    previous_state_after: Mapping[str, Any] | None = None
    previous_state_after_sha256: str | None = None
    for index, raw_row in enumerate(trace):
        row_label = f"{label}.trace[{index}]"
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{row_label} must be a mapping")
        action = raw_row.get("action")
        if (
            not isinstance(action, Mapping)
            or set(action) != {"tool", "arguments"}
            or not isinstance(action.get("arguments"), Mapping)
        ):
            raise ValueError(f"{row_label}.action is not canonical")
        status = raw_row.get("execution_status")
        if status not in {"success", "failure"}:
            raise ValueError(f"{row_label}.execution_status is invalid")
        policy_tool_output = raw_row.get("policy_tool_output")
        if not isinstance(policy_tool_output, Mapping):
            raise ValueError(f"{row_label}.policy_tool_output must be a mapping")
        persisted_output = copy.deepcopy(dict(policy_tool_output))
        if persisted_output.get("execution_status") != status:
            raise ValueError(
                f"{row_label}.policy_tool_output execution status is inconsistent"
            )
        if persisted_output.get("error_code") != raw_row.get("error_code"):
            raise ValueError(
                f"{row_label}.policy_tool_output error code is inconsistent"
            )
        if (persisted_output.get("state_mutated") is True) is not (
            raw_row.get("state_mutated") is True
        ):
            raise ValueError(
                f"{row_label}.policy_tool_output mutation flag is inconsistent"
            )
        try:
            derived_advanced = trace_progress_advanced(raw_row)
        except ValueError as exc:
            raise ValueError(
                f"{row_label} progress evidence is invalid: {exc}"
            ) from exc
        if raw_row.get("advanced") is not derived_advanced:
            raise ValueError(f"{row_label}.advanced is not reproducible")
        state_before = raw_row.get("state_before")
        state_after = raw_row.get("state_after")
        if not isinstance(state_before, Mapping) or not isinstance(
            state_after, Mapping
        ):
            raise ValueError(f"{row_label} state snapshots are invalid")
        if index == 0 and (
            state_before.get("active_state_id") != initial_active_state_id
        ):
            raise ValueError(
                f"{row_label}.state_before is not bound to the evaluator-owned initial state identity"
            )
        if previous_state_after is not None and (
            state_before != previous_state_after
            or raw_row.get("state_before_sha256")
            != previous_state_after_sha256
        ):
            raise ValueError(f"{row_label}.state_before breaks the trace state chain")
        transition = _canonical_trace_transition(
            state_before=state_before,
            action=action,
            policy_tool_output=persisted_output,
        )
        transition_leakage = policy_payload_leakage_paths(transition)
        leakage_paths.extend(
            f"trace[{index}].policy_tool_output{path[len('$.tool_output'):] }"
            if path.startswith("$.tool_output")
            else f"trace[{index}].policy_transition{path[1:]}"
            for path in transition_leakage
        )
        tool_evidence = _validate_objective_tool_binding(
            action=action,
            execution_status=status,
            runtime_state_hash=raw_row.get("runtime_state_hash"),
            policy_tool_output=persisted_output,
            reported_evidence=raw_row.get("objective_tool_evidence"),
            label=row_label,
        )
        intervention = raw_row.get("intervention") is True
        observation: dict[str, Any] | None = None
        assessment: dict[str, Any] | None = None
        if intervention:
            if (
                raw_row.get("observation_hash") is not None
                or raw_row.get("policy_observation") is not None
                or raw_row.get("objective_action_assessment") is not None
            ):
                raise ValueError(f"{row_label} intervention carries policy evidence")
        else:
            raw_observation = raw_row.get("policy_observation")
            if not isinstance(raw_observation, Mapping):
                raise ValueError(f"{row_label}.policy_observation must be a mapping")
            observation = copy.deepcopy(dict(raw_observation))
            expected_history = policy_safe_copy(
                history[-STUDY_POLICY_HISTORY_WINDOW:]
            )
            if observation.get("history_window") != expected_history:
                raise ValueError(
                    f"{row_label}.policy_observation history is not derived from trace"
                )
            last_transition = expected_history[-1] if expected_history else None
            last_action = (
                last_transition.get("action")
                if isinstance(last_transition, Mapping)
                else None
            )
            last_output = (
                last_transition.get("tool_output")
                if isinstance(last_transition, Mapping)
                else None
            )
            expected_last_tool = (
                last_action.get("tool")
                if isinstance(last_action, Mapping)
                else None
            )
            expected_last_status = (
                last_output.get("execution_status")
                if isinstance(last_output, Mapping)
                else None
            )
            expected_last_output = (
                copy.deepcopy(dict(last_output))
                if isinstance(last_output, Mapping)
                else {}
            )
            if (
                observation.get("last_tool") != expected_last_tool
                or observation.get("last_tool_status") != expected_last_status
                or observation.get("last_tool_output") != expected_last_output
            ):
                raise ValueError(
                    f"{row_label}.policy_observation last-tool state is not derived from trace"
                )
            if (
                not isinstance(observation.get("active_state_id"), str)
                or not str(observation.get("active_state_id")).strip()
                or observation.get("active_state_id")
                != state_before.get("active_state_id")
                or "candidate_state_id" not in observation
                or observation.get("candidate_state_id")
                != state_before.get("candidate_state_id")
            ):
                raise ValueError(
                    f"{row_label}.policy_observation contradicts state_before"
                )
            if (
                state_before.get("phase") is not None
                and "phase" in observation
                and observation.get("phase") != state_before.get("phase")
            ):
                raise ValueError(
                    f"{row_label}.policy_observation phase contradicts state_before"
                )
            observed_hash = raw_row.get("observation_hash")
            if (
                not isinstance(observed_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", observed_hash) is None
                or _stable_hash(observation) != observed_hash
            ):
                raise ValueError(f"{row_label}.policy_observation hash is forged")
            observation_leakage = policy_payload_leakage_paths(observation)
            leakage_paths.extend(
                f"trace[{index}]{path[1:]}" if path.startswith("$") else path
                for path in observation_leakage
            )
            expected_assessment = objective_recovery_action_assessment(
                observation,
                scenario_family=family,
                error_cardinality=cardinality,
                partial_success_opportunity=bool(
                    partial_count and policy_ordinal == 0
                ),
            )
            if raw_row.get("objective_action_assessment") != expected_assessment:
                raise ValueError(
                    f"{row_label}.objective_action_assessment is not reproducible"
                )
            assessment = copy.deepcopy(expected_assessment)
            policy_ordinal += 1
        history.append(transition)
        previous_state_after = copy.deepcopy(dict(state_after))
        previous_state_after_sha256 = str(raw_row["state_after_sha256"])
        derived_rows.append(
            {
                "index": index,
                "intervention": intervention,
                "policy_observation": observation,
                "objective_action_assessment": assessment,
                "objective_tool_evidence": tool_evidence,
                "policy_transition": transition,
            }
        )
    assert previous_state_after is not None
    if previous_state_after.get("active_state_id") != final_active_state_id:
        raise ValueError(
            f"{label}.trace[-1].state_after is not bound to the evaluator-owned final state identity"
        )
    return {
        "rows": derived_rows,
        "policy_observation_count": policy_ordinal,
        "hidden_truth_leakage_paths": sorted(set(leakage_paths)),
    }


def _candidate_disposition(oracle_state: Any) -> str | None:
    value = (
        oracle_state.candidate_disposition
        if isinstance(oracle_state, OracleState)
        else oracle_state.get("candidate_disposition")
        if isinstance(oracle_state, Mapping)
        else getattr(oracle_state, "candidate_disposition", None)
    )
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _candidate_assessment(oracle_state: Any) -> dict[str, Any]:
    value = (
        oracle_state.candidate_assessment
        if isinstance(oracle_state, OracleState)
        else oracle_state.get("candidate_assessment")
        if isinstance(oracle_state, Mapping)
        else getattr(oracle_state, "candidate_assessment", None)
    )
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _candidate_id(
    oracle_state: Any, observation: Mapping[str, Any]
) -> str | None:
    policy_observation = (
        oracle_state.policy_observation
        if isinstance(oracle_state, OracleState)
        else None
    )
    value = (
        policy_observation.candidate_state_id
        if isinstance(policy_observation, PolicyObservation)
        else observation.get("candidate_state_id")
    )
    return str(value) if value is not None else None


def _oracle_truth(oracle_state: Any) -> dict[str, Any]:
    if isinstance(oracle_state, OracleState):
        return oracle_state.truth_dict()
    if isinstance(oracle_state, Mapping):
        return _scenario_truth(oracle_state)
    return {}


def _remaining_fault_count(oracle_state: Any) -> int | None:
    truth = _oracle_truth(oracle_state)
    for key in ("remaining_true_fault_count", "remaining_fault_count"):
        if truth.get(key) is not None:
            try:
                return int(truth[key])
            except (TypeError, ValueError):
                return None
    remaining = truth.get("remaining_true_faults")
    if isinstance(remaining, Sequence) and not isinstance(remaining, (str, bytes)):
        return len(remaining)
    return None


def _complete_remaining_truth(oracle_state: Any) -> dict[str, Any] | None:
    truth = _oracle_truth(oracle_state)
    if truth.get("truth_complete") is not True:
        return None
    remaining_rows = truth.get("remaining_true_faults")
    if remaining_rows is not None and (
        not isinstance(remaining_rows, Sequence)
        or isinstance(remaining_rows, (str, bytes))
    ):
        return None
    raw_count = truth.get("remaining_true_fault_count")
    if raw_count is None:
        if remaining_rows is None:
            return None
        count = len(remaining_rows)
        truth["remaining_true_fault_count"] = count
    else:
        try:
            count = int(raw_count)
            numeric_count = float(raw_count)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            count < 0
            or isinstance(raw_count, bool)
            or not math.isfinite(numeric_count)
            or numeric_count != count
        ):
            return None
        if remaining_rows is not None and len(remaining_rows) != count:
            return None
    return truth


def _scenario_truth(scenario: Mapping[str, Any]) -> dict[str, Any]:
    if _has_partition_marker(scenario):
        _, audit_partition, _ = _partitioned_scenario_parts(scenario)
    else:
        audit_partition = copy.deepcopy(dict(scenario))
    has_nested_truth, nested_truth_value = _normalized_mapping_value(
        audit_partition, "truth", label="scenario/audit"
    )
    if has_nested_truth and not isinstance(nested_truth_value, Mapping):
        raise ValueError("offline truth container must be a mapping")
    nested_truth = (
        dict(nested_truth_value) if isinstance(nested_truth_value, Mapping) else {}
    )
    _, metadata = _normalized_mapping_value(
        audit_partition, "metadata", label="scenario/audit"
    )
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    truth: dict[str, Any] = {}

    def merge(source: Any, *, label: str, all_fields: bool = False) -> None:
        if not isinstance(source, Mapping):
            return
        for raw_key, value in source.items():
            key = _normalized_key(raw_key)
            if key == "remaining_fault_count":
                key = "remaining_true_fault_count"
            if key in {"truth", "hidden_truth", "ground_truth"}:
                continue
            if not all_fields and not (
                key.startswith("true_")
                or key
                in {
                    "truth_complete",
                    "clean_case",
                    "clean_measurements",
                    "clean_state",
                    "remaining_true_faults",
                    "remaining_true_fault_count",
                    "remaining_fault_count",
                }
            ):
                continue
            copied = copy.deepcopy(value)
            if key in truth and truth[key] != copied:
                raise ValueError(
                    f"conflicting offline truth field {key!r} from {label}"
                )
            truth[key] = copied

    def merge_container(
        value: Any, *, label: str, seen: set[int] | None = None
    ) -> None:
        if value is None:
            return
        if not isinstance(value, Mapping):
            raise ValueError(f"offline {label} container must be a mapping")
        seen = set() if seen is None else seen
        if id(value) in seen:
            raise ValueError(f"offline {label} container is cyclic")
        seen.add(id(value))
        for container_key in ("truth", "hidden_truth", "ground_truth"):
            found, nested = _normalized_mapping_value(
                value, container_key, label=label
            )
            if found:
                merge_container(
                    nested,
                    label=f"{label}.{container_key}",
                    seen=seen,
                )
        merge(value, label=label, all_fields=True)
        seen.remove(id(value))

    for parent, parent_label in (
        (audit_partition, "scenario/audit"),
        (metadata, "metadata"),
    ):
        for container_key in ("truth", "hidden_truth", "ground_truth"):
            found, value = _normalized_mapping_value(
                parent, container_key, label=parent_label
            )
            if found:
                merge_container(value, label=f"{parent_label}.{container_key}")
    merge(metadata, label="metadata")
    merge(nested_truth, label="truth")
    merge(audit_partition, label="scenario/audit")

    clean_state = truth.get("clean_state")
    if clean_state is not None and not isinstance(clean_state, Mapping):
        raise ValueError("offline truth field 'clean_state' must be a mapping")
    if isinstance(clean_state, Mapping):
        for nested_key, canonical_key in (
            ("case", "clean_case"),
            ("measurements", "clean_measurements"),
        ):
            if nested_key not in clean_state:
                continue
            nested_value = copy.deepcopy(clean_state[nested_key])
            if canonical_key in truth and truth[canonical_key] != nested_value:
                raise ValueError(
                    f"conflicting offline truth field {canonical_key!r} from clean_state"
                )
            truth[canonical_key] = nested_value

    if "truth_complete" in truth and not isinstance(truth["truth_complete"], bool):
        raise ValueError("offline truth field 'truth_complete' must be a boolean")
    if "remaining_true_fault_count" in truth:
        remaining_count = truth["remaining_true_fault_count"]
        if (
            isinstance(remaining_count, bool)
            or not isinstance(remaining_count, int)
            or remaining_count < 0
        ):
            raise ValueError(
                "offline truth field 'remaining_true_fault_count' must be a non-negative integer"
            )
        remaining_rows = truth.get("remaining_true_faults")
        if remaining_rows is not None and (
            not isinstance(remaining_rows, Sequence)
            or isinstance(remaining_rows, (str, bytes))
            or len(remaining_rows) != remaining_count
        ):
            raise ValueError(
                "offline remaining_true_faults must match remaining_true_fault_count"
            )
    if "truth_complete" not in truth:
        truth["truth_complete"] = bool(
            truth
            or any(str(key).startswith("true_") for key in truth)
            or "clean_case" in truth
            or "clean_measurements" in truth
        )
    return truth


_STRICT_PHYSICAL_EVIDENCE_GAPS = frozenset(
    {
        "accepted_measurement_nonregression_evidence_missing_or_malformed",
        "accepted_parameter_nonregression_evidence_missing_or_malformed",
        "accepted_topology_nonregression_evidence_missing_or_malformed",
        "accepted_target_nonregression_target_evidence_invalid",
        "healthy_measurement_preservation_evidence_missing_or_malformed",
        "healthy_case_preservation_evidence_missing_or_unloadable",
        "final_clean_measurement_evidence_missing_or_malformed",
        "final_clean_case_evidence_missing_or_unloadable",
        "true_measurement_targets_malformed",
        "true_measurement_target_out_of_range",
    }
)


def _scenario_truth_available(
    scenario: Mapping[str, Any], truth: Mapping[str, Any]
) -> bool:
    del scenario
    return bool(
        truth.get("truth_complete") is True
        and (
            any(str(key).startswith("true_") for key in truth)
            or any(
                key in truth
                for key in ("clean_case", "clean_measurements", "clean_state")
            )
        )
    )


def _physical_state_available(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and any(value.get(key) is not None for key in ("case", "measurements"))
    )


def _case_evidence_comparable(
    scenario: Mapping[str, Any],
    active_physical_state: Mapping[str, Any],
    *,
    case_loader: CaseLoader | None,
) -> bool:
    clean_state = scenario.get("clean_state")
    clean_state = clean_state if isinstance(clean_state, Mapping) else {}
    expected = clean_state.get("case", scenario.get("clean_case"))
    observed = active_physical_state.get("case")
    if expected is None or observed is None or case_loader is not None:
        return True
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        return True
    return expected == observed


def _strict_audit_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Promote offline correction truth aliases into the strict audit contract."""

    if _has_partition_marker(scenario):
        execution, audit, grouping = _partitioned_scenario_parts(scenario)
        normalized = execution
        normalized.update(grouping)
        truth = audit.pop("truth", None)
        normalized.update(audit)
        if isinstance(truth, Mapping):
            normalized.update(copy.deepcopy(dict(truth)))
    else:
        normalized = copy.deepcopy(dict(scenario))
    canonical_truth = _scenario_truth(normalized)
    for key, value in canonical_truth.items():
        if key in normalized and normalized[key] != value:
            raise ValueError(f"conflicting offline truth field {key!r}")
        normalized[key] = copy.deepcopy(value)
    return normalized


def _strict_check_status(checks: Mapping[str, Any], name: str) -> str | None:
    check = checks.get(name)
    return str(check.get("status")) if isinstance(check, Mapping) else None


def _strict_physical_evidence_complete(problems: Sequence[str]) -> bool:
    return not any(
        problem in _STRICT_PHYSICAL_EVIDENCE_GAPS
        or problem.startswith("true_") and problem.endswith("_malformed")
        for problem in problems
    )


def _default_physical_audit(
    *,
    scenario: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    final_oracle: Any,
    history: Sequence[Mapping[str, Any]],
    collateral_commit_seen: bool,
    terminal: bool,
    terminal_outcome: str | None,
    active_physical_state: Mapping[str, Any] | None,
    case_loader: CaseLoader | None,
) -> dict[str, Any]:
    audit_scenario = _strict_audit_scenario(scenario)
    original_truth = _scenario_truth(audit_scenario)
    accepted = _accepted_corrections(final_state)
    target_audit = _accepted_target_audit(original_truth, accepted)
    explanations = list(final_state.get("explained_anomalies") or [])
    diagnostic = _diagnostic_truth_audit(original_truth, explanations)
    base = {
        "physical_correctness_known": False,
        "final_physical_correct": False,
        "healthy_preservation_known": False,
        "healthy_components_preserved": False,
        "remaining_true_fault_count": None,
        "accepted_target_audit": target_audit,
        "diagnostic_truth_audit": diagnostic,
        "initial_active_state_id": initial_state.get("active_state_id"),
        "final_active_state_id": final_state.get("active_state_id"),
        "strict_release_audit": None,
        "audit_mode": "insufficient_evidence",
        "evidence_complete": False,
    }
    evidence_problems: list[str] = []
    if not _scenario_truth_available(audit_scenario, original_truth):
        evidence_problems.append("scenario_truth_unavailable")
    if not _physical_state_available(active_physical_state):
        evidence_problems.append("active_physical_state_unavailable")
    elif not _case_evidence_comparable(
        audit_scenario, active_physical_state, case_loader=case_loader
    ):
        evidence_problems.append("case_loader_required_for_physical_comparison")
    if evidence_problems:
        return {
            **base,
            "problems": evidence_problems,
            "quarantined": True,
        }

    remaining_truth = _complete_remaining_truth(final_oracle)
    strict = strict_audit_episode_against_truth(
        audit_scenario,
        final_state,
        terminal=terminal,
        terminal_outcome=terminal_outcome,
        active_physical_state=active_physical_state,
        remaining_truth=remaining_truth,
        case_loader=case_loader,
    )
    strict_problems = [str(item) for item in strict.get("problems") or []]
    evidence_complete = _strict_physical_evidence_complete(strict_problems)
    resolved = bool(terminal and terminal_outcome == "resolved")
    physical_known = bool(resolved and evidence_complete)
    physical_correct = bool(
        physical_known
        and not strict.get("quarantined", True)
        and not collateral_commit_seen
    )

    checks = strict.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    healthy_statuses = [
        _strict_check_status(checks, HEALTHY_MEASUREMENTS_CHECK),
        _strict_check_status(checks, HEALTHY_CASE_CHECK),
    ]
    # Healthy-component preservation is a terminal safety invariant, not a
    # resolution claim.  A truth-audited operator handoff can therefore know
    # that the untouched system was preserved even while true faults remain.
    healthy_known = bool(
        terminal
        and terminal_outcome in {"resolved", "operator_escalation"}
        and evidence_complete
        and all(status in {"passed", "failed"} for status in healthy_statuses)
    )
    healthy_preserved = bool(
        healthy_known
        and all(status == "passed" for status in healthy_statuses)
        and target_audit["healthy_targets_preserved"]
        and not collateral_commit_seen
    )
    remaining_check = checks.get(REMAINING_FAULTS_CHECK)
    remaining = (
        remaining_check.get("derived_remaining_fault_count")
        if isinstance(remaining_check, Mapping)
        else None
    )
    return {
        **base,
        "physical_correctness_known": physical_known,
        "final_physical_correct": physical_correct,
        "healthy_preservation_known": healthy_known,
        "healthy_components_preserved": healthy_preserved,
        "remaining_true_fault_count": remaining,
        "strict_release_audit": copy.deepcopy(strict),
        "audit_mode": "strict_release_audit",
        "evidence_complete": evidence_complete,
        "problems": strict_problems,
        "quarantined": bool(strict.get("quarantined", True)),
        "strict_checks_used": [
            ACCEPTED_TARGET_NONREGRESSION_CHECK,
            ACCEPTED_TARGETS_CHECK,
            DIAGNOSTIC_FAMILY_CHECK,
            HEALTHY_MEASUREMENTS_CHECK,
            HEALTHY_CASE_CHECK,
            REMAINING_FAULTS_CHECK,
            FINAL_MEASUREMENTS_CHECK,
            FINAL_CASE_CHECK,
        ],
        "history_steps_audited": len(history),
    }


def _accepted_corrections(final_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(item))
        for item in final_state.get("accepted_corrections") or []
        if isinstance(item, Mapping)
    ]


def _correction_action(item: Mapping[str, Any]) -> dict[str, Any]:
    raw = item.get("source_action") or item.get("action") or item
    return safe_normalize_action(raw) if isinstance(raw, Mapping) else invalid_action(
        "accepted_correction_action_missing"
    )


def _measurement_targets(action: Mapping[str, Any]) -> set[int]:
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    targets: set[int] = set()
    group = arguments.get("suspect_group")
    if isinstance(group, Sequence) and not isinstance(group, (str, bytes)):
        for value in group:
            try:
                targets.add(int(value))
            except (TypeError, ValueError):
                continue
    updates = arguments.get("measurement_updates")
    if isinstance(updates, Mapping):
        for value in updates:
            try:
                targets.add(int(value))
            except (TypeError, ValueError):
                continue
    for key in ("measurement_index", "index", "index0", "target"):
        if arguments.get(key) is not None:
            try:
                targets.add(int(arguments[key]))
            except (TypeError, ValueError):
                continue
    return targets


def _branch_row0(value: Mapping[str, Any]) -> int | None:
    for key, offset in (("branch_row0", 0), ("line_index1", -1), ("line_index", -1)):
        if value.get(key) is not None:
            try:
                row = int(value[key]) + offset
            except (TypeError, ValueError):
                return None
            return row if row >= 0 else None
    return None


def _truth_measurement_targets(truth: Mapping[str, Any]) -> set[int]:
    targets: set[int] = set()
    for fault in truth.get("true_measurement_errors") or []:
        if not isinstance(fault, Mapping):
            continue
        for key in ("index", "index0", "measurement_index"):
            if fault.get(key) is not None:
                try:
                    targets.add(int(fault[key]))
                except (TypeError, ValueError):
                    pass
                break
    return targets


def _truth_branch_targets(truth: Mapping[str, Any], family: str) -> set[int]:
    return {
        row
        for item in truth.get(f"true_{family}_errors") or []
        if isinstance(item, Mapping)
        for row in [_branch_row0(item)]
        if row is not None
    }


def _accepted_target_audit(
    truth: Mapping[str, Any], accepted: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    true_targets: dict[str, set[int]] = {
        "measurement": _truth_measurement_targets(truth),
        "parameter": _truth_branch_targets(truth, "parameter"),
        "topology": _truth_branch_targets(truth, "topology"),
    }
    accepted_targets: dict[str, set[int]] = {
        "measurement": set(),
        "parameter": set(),
        "topology": set(),
    }
    problems: list[str] = []
    for item in accepted:
        action = _correction_action(item)
        tool = action["tool"]
        if tool == CORRECT_MEASUREMENTS:
            family = "measurement"
            targets = _measurement_targets(action)
        elif tool in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
            family = "parameter" if tool == CORRECT_PARAMETERS else "topology"
            row = _branch_row0(action["arguments"])
            targets = {row} if row is not None else set()
        else:
            continue
        accepted_targets[family].update(targets)
        if not targets:
            problems.append(f"{family}_accepted_target_missing")
        elif not targets.issubset(true_targets[family]):
            healthy = sorted(targets - true_targets[family])
            problems.append(f"{family}_healthy_targets_modified:{healthy}")
    uncovered = sum(
        len(true_targets[family] - accepted_targets[family])
        for family in true_targets
    )
    return {
        "true_targets": {
            key: sorted(value) for key, value in true_targets.items()
        },
        "accepted_targets": {
            key: sorted(value) for key, value in accepted_targets.items()
        },
        "healthy_targets_preserved": not problems,
        "uncovered_standard_faults": uncovered,
        "problems": problems,
    }


def _diagnostic_truth_audit(
    truth: Mapping[str, Any], explanations: Sequence[Any]
) -> dict[str, Any]:
    records = [dict(item) for item in explanations if isinstance(item, Mapping)]
    problems: list[str] = []
    checked = 0

    harmonic_truth = [
        item
        for item in truth.get("true_harmonic_errors") or []
        if isinstance(item, Mapping)
    ]
    for item in harmonic_truth:
        checked += 1
        true_bus = item.get("bus_1based", item.get("source_bus"))
        matches = [record for record in records if record.get("family") == "harmonic"]
        if true_bus is not None:
            matches = [
                record
                for record in matches
                if isinstance(record.get("detail"), Mapping)
                and record["detail"].get("bus_1based") is not None
                and int(record["detail"]["bus_1based"]) == int(true_bus)
            ]
        if not matches:
            problems.append("harmonic_localization_mismatch")

    hif_truth = [
        item
        for item in truth.get("true_hif_errors") or []
        if isinstance(item, Mapping)
    ]
    for item in hif_truth:
        checked += 1
        true_row = _branch_row0(item)
        true_phase = item.get("phase")
        matches = [record for record in records if record.get("family") == "hif"]
        if true_row is not None:
            matches = [
                record
                for record in matches
                if isinstance(record.get("detail"), Mapping)
                and record["detail"].get("candidate_branch_row0") is not None
                and int(record["detail"]["candidate_branch_row0"]) == true_row
            ]
        if true_phase is not None:
            phase_matches = []
            for record in matches:
                detail = record.get("detail") or {}
                estimated = detail.get("estimated")
                estimated = estimated if isinstance(estimated, Mapping) else {}
                phase = estimated.get("phase", detail.get("phase"))
                if phase is None or str(phase).upper() == str(true_phase).upper():
                    phase_matches.append(record)
            matches = phase_matches
        if not matches:
            problems.append("hif_localization_mismatch")

    unbalance_keys = (
        "true_three_phase_unbalance_errors",
        "true_unbalance_errors",
        "true_imbalance_errors",
    )
    for key in unbalance_keys:
        for item in truth.get(key) or []:
            checked += 1
            if not any(
                record.get("family") == "three_phase_unbalance"
                for record in records
            ):
                problems.append("three_phase_unbalance_explanation_missing")

    return {
        "checked_diagnostic_faults": checked,
        "diagnostic_truth_matched": not problems,
        "problems": problems,
    }


def _scenario_groups(scenario: Mapping[str, Any]) -> dict[str, Any]:
    execution = scenario.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    grouping = scenario.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    metadata = execution.get("metadata", scenario.get("metadata"))
    metadata = metadata if isinstance(metadata, Mapping) else {}

    def first(*keys: str, default: Any = "unknown") -> Any:
        for key in keys:
            if grouping.get(key) is not None:
                return grouping[key]
            if scenario.get(key) is not None:
                return scenario[key]
            if execution.get(key) is not None:
                return execution[key]
            if metadata.get(key) is not None:
                return metadata[key]
        return default

    cardinality = first("error_cardinality", "cardinality", default=None)
    if cardinality is None:
        truth = _scenario_truth(scenario)
        counted_keys = [
            key
            for key, value in truth.items()
            if key.startswith("true_")
            and key.endswith("_errors")
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ]
        cardinality = sum(len(truth[key]) for key in counted_keys)
        if not counted_keys and not truth.get("truth_complete"):
            cardinality = "unknown"
    case = first("case_id", "network_case", "case", "case_path")
    if isinstance(case, Mapping):
        case = case.get("case_id", case.get("case_path", _stable_hash(case)[:12]))
    scenario_id = _scenario_id(scenario, 0)
    return {
        "family": str(first("scenario_family", "error_family", "family")),
        "cardinality": int(cardinality)
        if isinstance(cardinality, (int, float)) and not isinstance(cardinality, bool)
        else str(cardinality),
        "case": str(case),
        "split": str(first("split", "dataset_split")),
        "source_tier": str(first("source_tier", "data_source_tier")),
        "physical_root": str(
            first(
                "physical_root_fingerprint",
                "physical_root",
                "root_scenario_id",
                default=scenario_id,
            )
        ),
    }


def _resolve_cost_label(
    resolver: ToolCostResolver | None,
    *,
    scenario: Mapping[str, Any],
    suite: str,
    step: int,
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    tool_output: Mapping[str, Any],
    oracle_state: Any,
) -> Mapping[str, Any] | None:
    context = {
        "scenario": copy.deepcopy(dict(scenario)),
        "suite": suite,
        "step": step,
        "observation": copy.deepcopy(dict(observation)),
        "action": copy.deepcopy(dict(action)),
        "tool_output": copy.deepcopy(dict(tool_output)),
        "oracle_state": copy.deepcopy(oracle_state),
    }
    if resolver is not None:
        label = resolver(context)
        if label is not None and not isinstance(label, Mapping):
            raise TypeError("tool_cost_resolver must return a mapping or None.")
        return _canonical_cost_label(label) if isinstance(label, Mapping) else None

    label_scenario: Mapping[str, Any] = scenario
    if _has_partition_marker(scenario):
        _, audit_partition, _ = _partitioned_scenario_parts(scenario)
        label_scenario = audit_partition
    has_source, source = _normalized_mapping_value(
        label_scenario, "evaluation_labels", label="audit cost labels"
    )
    if not has_source:
        has_source, source = _normalized_mapping_value(
            label_scenario, "tool_cost_labels", label="audit cost labels"
        )
    _, label_metadata = _normalized_mapping_value(
        label_scenario, "metadata", label="audit cost labels"
    )
    if not has_source and isinstance(label_metadata, Mapping):
        has_source, source = _normalized_mapping_value(
            label_metadata, "evaluation_labels", label="audit metadata cost labels"
        )
        if not has_source:
            has_source, source = _normalized_mapping_value(
                label_metadata,
                "tool_cost_labels",
                label="audit metadata cost labels",
            )
    direct_label_keys = {
        "action_costs",
        "costs",
        "chosen_cost",
        "action_cost",
        "executed_cost",
        "best_cost",
        "minimum_cost",
        "min_cost",
        "optimal_cost",
        "expert_cost",
        "preferred_action",
    }
    if not has_source and any(
        _normalized_key(key) in direct_label_keys for key in label_scenario
    ):
        source = label_scenario
        has_source = True
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        label = source[step] if step < len(source) else None
    elif isinstance(source, Mapping):
        label = source.get(step, source.get(str(step)))
        if label is None and any(
            _normalized_key(key) in direct_label_keys for key in source
        ):
            label = source
    else:
        label = None
    return _canonical_cost_label(label) if isinstance(label, Mapping) else None


def _canonical_cost_label(label: Mapping[str, Any]) -> dict[str, Any]:
    canonical_keys = {
        "action_costs",
        "costs",
        "chosen_cost",
        "action_cost",
        "executed_cost",
        "cost",
        "best_cost",
        "minimum_cost",
        "min_cost",
        "optimal_cost",
        "expert_cost",
        "preferred_action",
    }
    normalized: dict[str, Any] = {}
    for raw_key, value in label.items():
        candidate = _normalized_key(raw_key)
        key = candidate if candidate in canonical_keys else str(raw_key)
        copied = copy.deepcopy(value)
        if key in normalized and normalized[key] != copied:
            raise ValueError(f"conflicting cost-label field {key!r}")
        normalized[key] = copied
    return normalized


def _tool_regret(
    label: Mapping[str, Any] | None, action: Mapping[str, Any]
) -> float | None:
    if not isinstance(label, Mapping):
        return None

    chosen = _first_number(
        label.get("chosen_cost"),
        label.get("action_cost"),
        label.get("executed_cost"),
        label.get("cost"),
    )
    best = _first_number(
        label.get("best_cost"),
        label.get("minimum_cost"),
        label.get("min_cost"),
        label.get("optimal_cost"),
        label.get("expert_cost"),
    )
    costs = label.get("action_costs", label.get("costs"))
    if isinstance(costs, Mapping):
        numeric_costs = {
            str(key): float(value)
            for key, value in costs.items()
            if _is_finite_number(value)
        }
        signature = action_signature(action)
        tool = str(action.get("tool"))
        chosen = numeric_costs.get(signature, numeric_costs.get(tool, chosen))
        if numeric_costs:
            best = min(numeric_costs.values()) if best is None else best

    if chosen is not None and best is not None:
        return max(0.0, chosen - best)

    preferred = label.get("preferred_action")
    margin = _first_number(label.get("cost_margin"), label.get("margin"))
    if preferred is not None and margin is not None:
        preferred_action = safe_normalize_action(preferred)
        return 0.0 if preferred_action == dict(action) else max(0.0, margin)
    return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        if _is_finite_number(value):
            return float(value)
    return None


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


__all__ = [
    "ClosedLoopEvaluator",
    "ClosedLoopRolloutEvaluator",
    "DEFAULT_SCORE_WEIGHTS",
    "EVALUATION_SUITES",
    "EpisodeEvaluation",
    "EvaluationResult",
    "RecoveryMetrics",
    "STUDY_EVALUATION_SCHEMA_VERSION",
    "STUDY_OBJECTIVE_ACTION_ASSESSMENT_CONTRACT",
    "STUDY_OBJECTIVE_EPISODE_EVIDENCE_CONTRACT",
    "STUDY_OBJECTIVE_TOOL_EVIDENCE_CONTRACT",
    "STUDY_POLICY_HISTORY_WINDOW",
    "build_evaluation_provenance",
    "build_study_evaluation_binding",
    "evaluate_closed_loop",
    "evaluate_closed_loop_rollouts",
    "evaluate_rollout_suites",
    "fingerprint_evaluation_suites",
    "load_evaluation_suites",
    "main",
    "make_evaluation_result",
    "objective_recovery_action_assessment",
    "objective_tool_evidence",
    "policy_payload_leakage_paths",
    "recovery_score",
    "privileged_execution_paths",
    "strip_offline_truth",
    "summarize_episode_evaluations",
    "trace_progress_advanced",
    "trace_progress_evidence",
    "study_objective_episode_evidence_marker",
    "validate_study_objective_episode_evidence",
    "validate_release_scenario_suites",
    "write_evaluation_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
