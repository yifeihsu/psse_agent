"""Fail-closed metrics and paired decision rules for the DAgger study.

The accepted inputs are content-addressed schema-v3 or schema-v4 closed-loop
evaluations: release artifacts for the frozen suite and irreversibly
diagnostic-only artifacts for the development holdout. Episode outcomes are
recomputed from root-level evidence; aggregate scores are never trusted as
study results.

Schema v3 remains accepted for historical diagnosis, but objective fields that
require opportunity, residual, feasibility, or policy-boundary evidence remain
explicitly unevaluable. Schema v4 persists the exact policy-visible inputs and
narrow state-bound tool certificates needed to recompute those metrics.

The comparison is paired by physical root.  Bootstrap resampling treats a
physical root as the sampling unit and retains every training/evaluation seed
observation for that root, which avoids pretending repeated evaluations of the
same physical case are independent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CONTEXT_TOOLS,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    CORRECTION_TOOLS,
    DIAGNOSTIC_TOOLS,
    FINALIZE_DIAGNOSIS,
    INVALID_ACTION,
    ROLLBACK_STATE,
    RUN_ALTERNATIVE_TEST,
    RUN_WLS,
    VERIFY_CANDIDATE,
    action_signature,
)
from psse_env.dagger.evaluation_gate import _trace_action_schema_failure
from psse_env.dagger.evaluator import (
    STUDY_EVALUATION_SCHEMA_VERSION,
    STUDY_OBJECTIVE_ACTION_ASSESSMENT_CONTRACT,
    STUDY_OBJECTIVE_TOOL_EVIDENCE_CONTRACT,
    objective_recovery_action_assessment,
    policy_payload_leakage_paths,
    study_objective_episode_evidence_marker,
    trace_progress_advanced,
    validate_study_objective_episode_evidence,
)
from psse_env.dagger.study_manifest import (
    DEFAULT_STUDY_MANIFEST as VERSIONED_STUDY_MANIFEST,
    EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256,
    EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256,
    PRODUCTION_D1_QUARANTINE_AUDIT_REPORT_NAME,
    PRODUCTION_D1_QUARANTINE_BINDING_CONTRACT,
    REQUIRED_VARIANT_IDS,
    TRAINED_VARIANT_IDS,
    StudyManifestError,
    canonical_development_evaluation_contract,
    canonical_production_d1_quarantine_binding,
    canonical_recovery_stress_evaluation_contract,
    load_study_manifest,
    validate_study_artifact_binding,
)
from psse_env.sft.provenance import file_sha256, stable_json_sha256


METRIC_CONTRACT = "dagger_closed_loop_study_metrics_v1"
REPORT_SCHEMA_VERSION = 1
STANDARD_TARGET_FAMILIES = ("measurement", "parameter", "topology")
DEFAULT_BOOTSTRAP_RESAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED = 20_260_821
DEFAULT_STUDY_MANIFEST = VERSIONED_STUDY_MANIFEST
PRIMARY_EVALUATION_SCOPES = ("development_holdout", "frozen_suite")
RECOVERY_STRESS_SCOPE = "recovery_stress"
EVALUATION_SCOPES = (*PRIMARY_EVALUATION_SCOPES, RECOVERY_STRESS_SCOPE)
_SCOPE_ARTIFACT_ROLE = {
    "development_holdout": "development_evaluation",
    "frozen_suite": "evaluation",
    RECOVERY_STRESS_SCOPE: "recovery_stress_evaluation",
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
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
_IDENTITY_FIELDS = (
    "episode_key",
    "scenario_id",
    "suite",
    "family",
    "cardinality",
    "case",
    "split",
    "source_tier",
    "physical_root",
    "episode_seed",
    "true_targets",
)
_ACTIVE_BOUND_TOOLS = frozenset({*CORRECTION_TOOLS, *CONTEXT_TOOLS})
_CURRENT_BOUND_TOOLS = frozenset(
    {
        RUN_WLS,
        ASK_FOR_MORE_EVIDENCE,
        RUN_ALTERNATIVE_TEST,
        *DIAGNOSTIC_TOOLS,
    }
)
_CANDIDATE_BOUND_TOOLS = frozenset({VERIFY_CANDIDATE, COMMIT_STATE, ROLLBACK_STATE})
_DIAGNOSTIC_ONLY_FAILURE = (
    "diagnostic-only evaluation artifacts are not release evidence"
)
_RECOVERY_ACTION_STRATA = (
    "post_failure_no_candidate",
    "unsupported_correction_recovery",
    "premature_commit_recovery",
    "premature_escalation_recovery",
    "rejected_candidate_rollback",
    "safe_continuation_after_partial_success",
    "measurement_parameter_sequential_handoff",
)
_OBJECTIVE_TOOL_EVIDENCE_FIELDS = {
    "contract",
    "tool",
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
}
_OBJECTIVE_ACTION_ASSESSMENT_FIELDS = {
    "contract",
    "evidence_available",
    "evidence_failure",
    "policy_payload_leakage_paths",
    "canonical_selector",
    "selector_basis",
    "canonical_action_count",
    "expected_action",
    "recovery_stratum",
    "operator_handoff_opportunity",
}


class StudyEvidenceError(ValueError):
    """Raised when a metric would depend on missing or ambiguous evidence."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyEvidenceError(f"{field} must be a JSON object")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise StudyEvidenceError(f"{field} must be a JSON array")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyEvidenceError(f"{field} must be a non-empty string")
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise StudyEvidenceError(f"{field} must be an explicit boolean")
    return value


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StudyEvidenceError(f"{field} must be a non-negative integer")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    parsed = _nonnegative_integer(value, field=field)
    if parsed == 0:
        raise StudyEvidenceError(f"{field} must be positive")
    return parsed


def _hash(value: Any, *, field: str) -> str:
    parsed = _text(value, field=field).lower()
    if _SHA256.fullmatch(parsed) is None:
        raise StudyEvidenceError(f"{field} must be a lowercase SHA-256")
    return parsed


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _artifact_content_sha256(payload: Mapping[str, Any]) -> str:
    """Match evaluator._stable_hash for already decoded JSON values."""

    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _load_payload(
    artifact: str | os.PathLike[str] | Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if isinstance(artifact, Mapping):
        payload = copy.deepcopy(dict(artifact))
        source_path = None
    else:
        path = Path(artifact).expanduser().resolve(strict=True)
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StudyEvidenceError(
                f"evaluation artifact is not valid JSON: {path}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise StudyEvidenceError(
                f"evaluation artifact must be a JSON object: {path}"
            )
        payload = copy.deepcopy(dict(decoded))
        source_path = str(path)
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StudyEvidenceError(
            "evaluation artifact must contain finite JSON values"
        ) from exc
    return payload, source_path


def _target_set(value: Any, *, field: str) -> tuple[int, ...]:
    rows = _list(value, field=field)
    targets: list[int] = []
    for index, item in enumerate(rows):
        targets.append(_nonnegative_integer(item, field=f"{field}[{index}]"))
    if len(targets) != len(set(targets)):
        raise StudyEvidenceError(f"{field} contains duplicate targets")
    if targets != sorted(targets):
        raise StudyEvidenceError(f"{field} must be sorted")
    return tuple(targets)


def _accepted_target_evidence(
    episode: Mapping[str, Any], *, label: str
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], list[str]]:
    audit = _mapping(episode.get("audit"), field=f"{label}.audit")
    target_audit = _mapping(
        audit.get("accepted_target_audit"),
        field=f"{label}.audit.accepted_target_audit",
    )
    expected_fields = {
        "true_targets",
        "accepted_targets",
        "healthy_targets_preserved",
        "uncovered_standard_faults",
        "problems",
    }
    if set(target_audit) != expected_fields:
        raise StudyEvidenceError(
            f"{label}.audit.accepted_target_audit has a noncanonical schema"
        )
    true_raw = _mapping(
        target_audit.get("true_targets"),
        field=f"{label}.audit.accepted_target_audit.true_targets",
    )
    accepted_raw = _mapping(
        target_audit.get("accepted_targets"),
        field=f"{label}.audit.accepted_target_audit.accepted_targets",
    )
    if set(true_raw) != set(STANDARD_TARGET_FAMILIES) or set(accepted_raw) != set(
        STANDARD_TARGET_FAMILIES
    ):
        raise StudyEvidenceError(
            f"{label}.audit.accepted_target_audit must cover exactly "
            + ", ".join(STANDARD_TARGET_FAMILIES)
        )
    true_targets = {
        family: _target_set(
            true_raw[family],
            field=(f"{label}.audit.accepted_target_audit.true_targets.{family}"),
        )
        for family in STANDARD_TARGET_FAMILIES
    }
    accepted_targets = {
        family: _target_set(
            accepted_raw[family],
            field=(f"{label}.audit.accepted_target_audit.accepted_targets.{family}"),
        )
        for family in STANDARD_TARGET_FAMILIES
    }
    problems = _list(
        target_audit.get("problems"),
        field=f"{label}.audit.accepted_target_audit.problems",
    )
    if any(not isinstance(item, str) or not item for item in problems):
        raise StudyEvidenceError(
            f"{label}.audit.accepted_target_audit.problems must contain strings"
        )
    if len(problems) != len(set(problems)):
        raise StudyEvidenceError(
            f"{label}.audit.accepted_target_audit.problems contains duplicates"
        )
    if any(item.endswith("_accepted_target_missing") for item in problems):
        raise StudyEvidenceError(
            f"{label} has an accepted correction whose target is unknowable"
        )
    uncovered = sum(
        len(set(true_targets[family]) - set(accepted_targets[family]))
        for family in STANDARD_TARGET_FAMILIES
    )
    if (
        _nonnegative_integer(
            target_audit.get("uncovered_standard_faults"),
            field=(f"{label}.audit.accepted_target_audit.uncovered_standard_faults"),
        )
        != uncovered
    ):
        raise StudyEvidenceError(
            f"{label}.audit.accepted_target_audit uncovered count is inconsistent"
        )
    healthy_targets_preserved = _boolean(
        target_audit.get("healthy_targets_preserved"),
        field=(f"{label}.audit.accepted_target_audit.healthy_targets_preserved"),
    )
    if healthy_targets_preserved is not (not problems):
        raise StudyEvidenceError(
            f"{label}.audit.accepted_target_audit preservation flag is inconsistent"
        )
    false_targets = {
        family: sorted(set(accepted_targets[family]) - set(true_targets[family]))
        for family in STANDARD_TARGET_FAMILIES
    }
    if any(false_targets.values()) and healthy_targets_preserved:
        raise StudyEvidenceError(
            f"{label}.audit.accepted_target_audit hides healthy-target changes"
        )
    return true_targets, accepted_targets, list(problems)


def _trace_state_binding(
    *, action: Mapping[str, Any], state_before: Mapping[str, Any]
) -> tuple[bool, bool] | None:
    """Return (evaluable, valid), or None for a non-state-bound action.

    Only an explicit controller reference and the persisted pre-action state
    make this proxy evaluable.  Missing references are counted as unevaluable,
    never silently treated as valid.
    """

    tool = str(action.get("tool") or "")
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        return (False, False)
    if tool in {FINALIZE_DIAGNOSIS, INVALID_ACTION}:
        return None
    if tool in {COMMIT_STATE, ROLLBACK_STATE}:
        field = "candidate_state_id"
        expected = state_before.get("candidate_state_id")
    elif tool == VERIFY_CANDIDATE:
        field = "state_id"
        expected = state_before.get("candidate_state_id")
    elif tool in _ACTIVE_BOUND_TOOLS:
        field = "state_id"
        expected = state_before.get("active_state_id")
    elif tool in _CURRENT_BOUND_TOOLS:
        field = "state_id"
        expected = state_before.get("candidate_state_id") or state_before.get(
            "active_state_id"
        )
    else:
        return None
    observed = arguments.get(field)
    if expected is None or observed is None:
        return (False, False)
    if not isinstance(observed, str) or not observed:
        return (False, False)
    return (True, observed == expected)


def _optional_finite_number(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyEvidenceError(f"{field} must be a finite number or null")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise StudyEvidenceError(f"{field} must be a finite number or null")
    return parsed


def _optional_boolean(value: Any, *, field: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, field=field)


def _observable_objective_source(value: Any) -> bool:
    source = str(value or "").strip().lower()
    if not source or any(
        token in source
        for token in (
            "hidden",
            "oracle",
            "truth",
            "synthetic",
            "placeholder",
            "fallback",
        )
    ):
        return False
    return source.startswith(("observable", "deployment", "sensor", "wls"))


def _validated_objective_tool_evidence(
    value: Any,
    *,
    action: Mapping[str, Any],
    execution_status: str,
    runtime_state_hash: Any,
    field: str,
) -> dict[str, Any] | None:
    action_tool = str(action.get("tool") or "")
    if action_tool not in {RUN_WLS, VERIFY_CANDIDATE}:
        if value is not None:
            raise StudyEvidenceError(f"{field} must be null for a non-WLS action")
        return None
    evidence = _mapping(value, field=field)
    if set(evidence) != _OBJECTIVE_TOOL_EVIDENCE_FIELDS:
        raise StudyEvidenceError(f"{field} has a noncanonical schema")
    if (
        evidence.get("contract") != STUDY_OBJECTIVE_TOOL_EVIDENCE_CONTRACT
        or evidence.get("tool") != action_tool
    ):
        raise StudyEvidenceError(f"{field} contract/tool binding is invalid")
    state_id = evidence.get("state_id")
    if state_id is not None and (not isinstance(state_id, str) or not state_id.strip()):
        raise StudyEvidenceError(f"{field}.state_id is invalid")
    action_arguments = action.get("arguments")
    action_arguments = action_arguments if isinstance(action_arguments, Mapping) else {}
    if state_id is not None and state_id != action_arguments.get("state_id"):
        raise StudyEvidenceError(
            f"{field}.state_id is not bound to the executed action"
        )
    state_hash = evidence.get("state_hash")
    if state_hash is not None:
        state_hash = _hash(state_hash, field=f"{field}.state_hash")
        if runtime_state_hash != state_hash:
            raise StudyEvidenceError(
                f"{field}.state_hash differs from trace runtime_state_hash"
            )
    elif runtime_state_hash is not None:
        raise StudyEvidenceError(
            f"{field} omits the persisted trace runtime_state_hash"
        )
    if execution_status == "success" and (
        not isinstance(state_id, str)
        or not state_id.strip()
        or state_id != action_arguments.get("state_id")
        or state_hash is None
        or runtime_state_hash is None
    ):
        raise StudyEvidenceError(
            f"{field} lacks exact non-null state binding for a successful tool call"
        )
    for name in (
        "chi_square_statistic",
        "chi_square_threshold",
        "max_normalized_residual",
    ):
        _optional_finite_number(evidence.get(name), field=f"{field}.{name}")
    for name in (
        "no_material_anomaly_remaining",
        "globally_resolved",
        "physical_constraints_ok",
        "physical_evidence_complete",
        "power_flow_converged",
        "topology_feasible",
    ):
        _optional_boolean(evidence.get(name), field=f"{field}.{name}")
    source = evidence.get("evidence_source")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise StudyEvidenceError(f"{field}.evidence_source is invalid")
    scope = evidence.get("physical_evidence_scope")
    if scope is not None and (not isinstance(scope, str) or not scope.strip()):
        raise StudyEvidenceError(f"{field}.physical_evidence_scope is invalid")
    violations = evidence.get("physical_bound_violations")
    if violations is not None and not isinstance(violations, list):
        raise StudyEvidenceError(
            f"{field}.physical_bound_violations must be an array or null"
        )
    steady = evidence.get("steady_state_physical_evidence")
    if steady is not None and not isinstance(steady, Mapping):
        raise StudyEvidenceError(
            f"{field}.steady_state_physical_evidence must be an object or null"
        )
    return copy.deepcopy(dict(evidence))


def _residual_certificate(
    evidence: Mapping[str, Any] | None,
) -> tuple[bool, bool | None, str | None]:
    if evidence is None:
        return False, None, "bound_final_wls_evidence_missing"
    if not _observable_objective_source(evidence.get("evidence_source")):
        return False, None, "final_wls_evidence_source_not_observable"
    statistic = _optional_finite_number(
        evidence.get("chi_square_statistic"),
        field="objective_tool_evidence.chi_square_statistic",
    )
    threshold = _optional_finite_number(
        evidence.get("chi_square_threshold"),
        field="objective_tool_evidence.chi_square_threshold",
    )
    max_residual = _optional_finite_number(
        evidence.get("max_normalized_residual"),
        field="objective_tool_evidence.max_normalized_residual",
    )
    no_anomaly = evidence.get("no_material_anomaly_remaining")
    resolved = evidence.get("globally_resolved")
    if (
        statistic is None
        or threshold is None
        or threshold <= 0.0
        or statistic < 0.0
        or max_residual is None
        or max_residual < 0.0
        or not isinstance(no_anomaly, bool)
        or not isinstance(resolved, bool)
    ):
        return False, None, "final_residual_chi_square_fields_incomplete"
    expected = statistic < threshold
    if no_anomaly is not expected or resolved is not expected:
        raise StudyEvidenceError(
            "final residual/chi-square flags disagree with statistic and threshold"
        )
    return True, expected, None


def _physical_feasibility_certificate(
    evidence: Mapping[str, Any] | None,
) -> tuple[bool, bool | None, str | None]:
    if evidence is None:
        return False, None, "bound_candidate_verification_missing"
    if not _observable_objective_source(evidence.get("evidence_source")):
        return False, None, "candidate_feasibility_source_not_observable"
    complete = evidence.get("physical_evidence_complete")
    physical_ok = evidence.get("physical_constraints_ok")
    violations = evidence.get("physical_bound_violations")
    steady = evidence.get("steady_state_physical_evidence")
    if not isinstance(complete, bool) or not isinstance(violations, list):
        return False, None, "candidate_feasibility_fields_incomplete"
    if not isinstance(steady, Mapping):
        return False, None, "candidate_physical_certificate_missing"
    if (
        evidence.get("physical_evidence_scope")
        != "observed_snapshot_topology_vm_rate_a"
        or steady.get("scope") != "observed_snapshot_topology_vm_rate_a"
        or steady.get("method") != "matpower_case_limits_with_observed_wls_telemetry"
    ):
        return False, None, "candidate_physical_certificate_scope_unrecognized"
    if steady.get("complete") is not complete:
        raise StudyEvidenceError(
            "candidate physical certificate complete flag is inconsistent"
        )
    violation_count = steady.get("violation_count")
    if (
        isinstance(violation_count, bool)
        or not isinstance(violation_count, int)
        or violation_count < 0
        or violation_count != len(violations)
    ):
        raise StudyEvidenceError(
            "candidate physical certificate violation count is inconsistent"
        )
    input_errors = steady.get("input_errors")
    if not isinstance(input_errors, list) or any(
        not isinstance(item, str) or not item for item in input_errors
    ):
        raise StudyEvidenceError(
            "candidate physical certificate input_errors is invalid"
        )
    if not complete or input_errors:
        if physical_ok is not None:
            raise StudyEvidenceError(
                "incomplete candidate physical evidence claims a disposition"
            )
        return False, None, "candidate_physical_certificate_incomplete"
    if not isinstance(physical_ok, bool):
        raise StudyEvidenceError(
            "complete candidate physical evidence lacks a boolean disposition"
        )
    checks = {
        "topology_connectivity": (
            "connected",
            "topology_disconnected",
        ),
        "bus_voltage_bounds": (
            "within_bounds",
            "bus_voltage_out_of_bounds",
        ),
        "active_branch_rate_a_bounds": (
            "within_defined_rate_a_bounds",
            "active_branch_rate_a_exceeded",
        ),
    }
    violation_types: set[str] = set()
    for index, violation in enumerate(violations):
        if not isinstance(violation, Mapping):
            raise StudyEvidenceError(
                f"candidate physical violation[{index}] is not an object"
            )
        violation_type = violation.get("type")
        if not isinstance(violation_type, str) or not violation_type:
            raise StudyEvidenceError(
                f"candidate physical violation[{index}] lacks a type"
            )
        violation_types.add(violation_type)
    for name, (pass_field, violation_type) in checks.items():
        check = steady.get(name)
        if not isinstance(check, Mapping) or check.get("checked") is not True:
            return False, None, f"candidate_physical_{name}_not_checked"
        passed_check = check.get(pass_field)
        if not isinstance(passed_check, bool):
            return False, None, f"candidate_physical_{name}_result_missing"
        if passed_check is not (violation_type not in violation_types):
            raise StudyEvidenceError(
                f"candidate physical {name} result disagrees with violations"
            )
    expected = not violations
    if physical_ok is not expected:
        raise StudyEvidenceError(
            "candidate physical disposition disagrees with bound violations"
        )
    for optional_name in ("power_flow_converged", "topology_feasible"):
        optional_value = evidence.get(optional_name)
        if optional_value is False:
            expected = False
    return True, expected, None


def _trace_evidence(
    episode: Mapping[str, Any],
    *,
    label: str,
    artifact_schema_version: int,
    scenario_family: str,
    error_cardinality: int,
) -> dict[str, Any]:
    canonical_objective_evidence: dict[str, Any] | None = None
    if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION:
        try:
            canonical_objective_evidence = (
                validate_study_objective_episode_evidence(
                    episode,
                    scenario_family=scenario_family,
                    error_cardinality=error_cardinality,
                    label=label,
                )
            )
        except ValueError as exc:
            raise StudyEvidenceError(str(exc)) from exc
    trace = _list(episode.get("trace"), field=f"{label}.trace")
    steps = _nonnegative_integer(episode.get("steps"), field=f"{label}.steps")
    policy_steps = _nonnegative_integer(
        episode.get("policy_steps"), field=f"{label}.policy_steps"
    )
    if len(trace) != steps:
        raise StudyEvidenceError(f"{label}.trace does not match steps")

    policy_rows: list[tuple[Mapping[str, Any], bool]] = []
    previous_after: Mapping[str, Any] | None = None
    previous_after_hash: str | None = None
    intervention_finished = False
    terminal_marker_indices: list[int] = []
    terminal_after_indices: list[int] = []
    derived_invalid = 0
    tool_counts: Counter[str] = Counter()
    state_bound_evaluable = 0
    state_bound_valid = 0
    state_bound_invalid = 0
    state_bound_unevaluable = 0
    schema_valid_tool_calls = 0
    successful_correction_families: list[str] = []
    objective_action_evidence_complete = artifact_schema_version == (
        STUDY_EVALUATION_SCHEMA_VERSION
    )
    policy_observation_count = 0
    hidden_truth_leakage_paths: list[str] = (
        list(canonical_objective_evidence["hidden_truth_leakage_paths"])
        if canonical_objective_evidence is not None
        else []
    )
    recovery_opportunities: Counter[str] = Counter()
    correct_recovery_actions: Counter[str] = Counter()
    operator_handoff_opportunities = 0
    correct_operator_handoffs = 0
    objective_tool_rows: list[dict[str, Any]] = []
    policy_ordinal = 0
    intervention_evidence = episode.get("evaluation_intervention")
    intervention_evidence = (
        intervention_evidence if isinstance(intervention_evidence, Mapping) else {}
    )
    partial_opportunity_count = (
        _nonnegative_integer(
            intervention_evidence.get("retention_opportunity_count"),
            field=(f"{label}.evaluation_intervention.retention_opportunity_count"),
        )
        if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
        else 0
    )
    if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION and (
        partial_opportunity_count not in {0, 1}
        or (
            partial_opportunity_count
            and episode.get("suite") != "partial_success_retention"
        )
    ):
        raise StudyEvidenceError(
            f"{label} has a noncanonical partial-success opportunity count"
        )

    for index, raw_row in enumerate(trace):
        row = _mapping(raw_row, field=f"{label}.trace[{index}]")
        if row.get("step") != index:
            raise StudyEvidenceError(
                f"{label}.trace[{index}] has a noncanonical step index"
            )
        intervention = _boolean(
            row.get("intervention"),
            field=f"{label}.trace[{index}].intervention",
        )
        if intervention and intervention_finished:
            raise StudyEvidenceError(
                f"{label}.trace intervention rows must be a prefix"
            )
        if not intervention:
            intervention_finished = True
        action = _mapping(row.get("action"), field=f"{label}.trace[{index}].action")
        if set(action) != {"tool", "arguments"}:
            raise StudyEvidenceError(
                f"{label}.trace[{index}].action has a noncanonical schema"
            )
        tool = _text(action.get("tool"), field=f"{label}.trace[{index}].action.tool")
        _mapping(
            action.get("arguments"),
            field=f"{label}.trace[{index}].action.arguments",
        )
        status = row.get("execution_status")
        if status not in {"success", "failure"}:
            raise StudyEvidenceError(
                f"{label}.trace[{index}].execution_status is invalid"
            )
        error_code = row.get("error_code")
        if status == "failure" and (not isinstance(error_code, str) or not error_code):
            raise StudyEvidenceError(
                f"{label}.trace[{index}] failed action lacks an error code"
            )
        if status == "success" and error_code is not None:
            raise StudyEvidenceError(
                f"{label}.trace[{index}] successful action carries an error code"
            )
        reported_advanced = _boolean(
            row.get("advanced"),
            field=f"{label}.trace[{index}].advanced",
        )
        progress = {field: row.get(field) for field in _TRACE_PROGRESS_FIELDS}
        try:
            progress_advanced = trace_progress_advanced(progress)
        except ValueError as exc:
            raise StudyEvidenceError(
                f"{label}.trace[{index}] progress evidence is invalid: {exc}"
            ) from exc
        effective_advanced = bool(
            tool != INVALID_ACTION and status == "success" and progress_advanced
        )
        if reported_advanced != effective_advanced:
            raise StudyEvidenceError(
                f"{label}.trace[{index}] advanced flag is inconsistent"
            )
        before = _mapping(
            progress.get("state_before"),
            field=f"{label}.trace[{index}].state_before",
        )
        before_hash = progress.get("state_before_sha256")
        if index and (before != previous_after or before_hash != previous_after_hash):
            raise StudyEvidenceError(
                f"{label}.trace[{index}] state evidence is not continuous"
            )
        previous_after = _mapping(
            progress.get("state_after"),
            field=f"{label}.trace[{index}].state_after",
        )
        previous_after_hash = str(progress.get("state_after_sha256") or "")
        if progress.get("terminal_after") is True:
            terminal_after_indices.append(index)
        if row.get("terminal_outcome") is not None:
            terminal_marker_indices.append(index)
        objective_tool: dict[str, Any] | None = None
        if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION:
            objective_tool = _validated_objective_tool_evidence(
                row.get("objective_tool_evidence"),
                action=action,
                execution_status=status,
                runtime_state_hash=row.get("runtime_state_hash"),
                field=f"{label}.trace[{index}].objective_tool_evidence",
            )
            if intervention:
                if (
                    row.get("observation_hash") is not None
                    or row.get("policy_observation") is not None
                    or row.get("objective_action_assessment") is not None
                ):
                    raise StudyEvidenceError(
                        f"{label}.trace[{index}] intervention carries policy evidence"
                    )
            else:
                policy_observation = _mapping(
                    row.get("policy_observation"),
                    field=f"{label}.trace[{index}].policy_observation",
                )
                policy_observation_count += 1
                observed_hash = _hash(
                    row.get("observation_hash"),
                    field=f"{label}.trace[{index}].observation_hash",
                )
                if _artifact_content_sha256(policy_observation) != observed_hash:
                    raise StudyEvidenceError(
                        f"{label}.trace[{index}] policy observation hash is forged"
                    )
                leakage = policy_payload_leakage_paths(policy_observation)
                hidden_truth_leakage_paths.extend(
                    f"trace[{index}]{path[1:]}" if path.startswith("$") else path
                    for path in leakage
                )
                recomputed_assessment = objective_recovery_action_assessment(
                    policy_observation,
                    scenario_family=scenario_family,
                    error_cardinality=error_cardinality,
                    partial_success_opportunity=bool(
                        partial_opportunity_count and policy_ordinal == 0
                    ),
                )
                reported_assessment = _mapping(
                    row.get("objective_action_assessment"),
                    field=(f"{label}.trace[{index}].objective_action_assessment"),
                )
                if set(reported_assessment) != (_OBJECTIVE_ACTION_ASSESSMENT_FIELDS):
                    raise StudyEvidenceError(
                        f"{label}.trace[{index}] action assessment schema is noncanonical"
                    )
                if reported_assessment != recomputed_assessment:
                    raise StudyEvidenceError(
                        f"{label}.trace[{index}] action assessment is not reproducible"
                    )
                if (
                    reported_assessment.get("contract")
                    != STUDY_OBJECTIVE_ACTION_ASSESSMENT_CONTRACT
                ):
                    raise StudyEvidenceError(
                        f"{label}.trace[{index}] action assessment contract is invalid"
                    )
                assessment_available = reported_assessment.get("evidence_available")
                if not isinstance(assessment_available, bool):
                    raise StudyEvidenceError(
                        f"{label}.trace[{index}] action assessment availability is invalid"
                    )
                objective_action_evidence_complete = bool(
                    objective_action_evidence_complete and assessment_available
                )
                stratum = reported_assessment.get("recovery_stratum")
                if stratum is not None:
                    if stratum not in _RECOVERY_ACTION_STRATA:
                        raise StudyEvidenceError(
                            f"{label}.trace[{index}] has an unknown recovery stratum"
                        )
                    recovery_opportunities[str(stratum)] += 1
                    if (
                        assessment_available
                        and reported_assessment.get("expected_action") == action
                    ):
                        correct_recovery_actions[str(stratum)] += 1
                handoff_opportunity = reported_assessment.get(
                    "operator_handoff_opportunity"
                )
                if handoff_opportunity not in {True, False, None}:
                    raise StudyEvidenceError(
                        f"{label}.trace[{index}] handoff opportunity is invalid"
                    )
                if handoff_opportunity is True:
                    operator_handoff_opportunities += 1
                    if (
                        assessment_available
                        and reported_assessment.get("expected_action") == action
                        and status == "success"
                        and progress.get("terminal_after") is True
                        and row.get("terminal_outcome") == "operator_escalation"
                    ):
                        correct_operator_handoffs += 1
                policy_ordinal += 1
        objective_tool_rows.append(
            {
                "index": index,
                "intervention": intervention,
                "tool": tool,
                "action": copy.deepcopy(dict(action)),
                "execution_status": status,
                "state_before": copy.deepcopy(dict(before)),
                "state_after": copy.deepcopy(dict(previous_after)),
                "objective_tool_evidence": objective_tool,
            }
        )
        if intervention:
            continue

        policy_rows.append((action, effective_advanced))
        tool_counts[tool] += 1
        schema_failure = _trace_action_schema_failure(action, index=index)
        if schema_failure is not None:
            raise StudyEvidenceError(
                f"{label}.trace[{index}] action evidence is invalid: {schema_failure}"
            )
        if tool != INVALID_ACTION:
            schema_valid_tool_calls += 1
        if status == "success" and tool in CORRECTION_TOOLS:
            successful_correction_families.append(
                {
                    CORRECT_MEASUREMENTS: "measurement",
                    CORRECT_PARAMETERS: "parameter",
                    CORRECT_TOPOLOGY: "topology",
                }[tool]
            )
        if tool == INVALID_ACTION or status != "success":
            derived_invalid += 1
        binding = _trace_state_binding(action=action, state_before=before)
        if binding is not None:
            evaluable, valid = binding
            if evaluable:
                state_bound_evaluable += 1
                if valid:
                    state_bound_valid += 1
                else:
                    state_bound_invalid += 1
            else:
                state_bound_unevaluable += 1

    if len(policy_rows) != policy_steps:
        raise StudyEvidenceError(
            f"{label}.policy_steps does not match non-intervention trace rows"
        )
    recorded_invalid = _nonnegative_integer(
        episode.get("invalid_action_count"),
        field=f"{label}.invalid_action_count",
    )
    if recorded_invalid != derived_invalid:
        raise StudyEvidenceError(
            f"{label}.invalid_action_count does not match the policy trace"
        )
    raw_tool_counts = _mapping(episode.get("tool_counts"), field=f"{label}.tool_counts")
    recorded_tool_counts = {
        _text(tool, field=f"{label}.tool_counts key"): _nonnegative_integer(
            count, field=f"{label}.tool_counts.{tool}"
        )
        for tool, count in raw_tool_counts.items()
    }
    if recorded_tool_counts != dict(sorted(tool_counts.items())):
        raise StudyEvidenceError(f"{label}.tool_counts does not match the policy trace")

    redundant_actions = 0
    nonadvancing_signatures: set[str] = set()
    for action, advanced in policy_rows:
        if advanced:
            nonadvancing_signatures.clear()
            continue
        signature = action_signature(action)
        if signature in nonadvancing_signatures:
            redundant_actions += 1
        nonadvancing_signatures.add(signature)
    loop_detected = _boolean(
        episode.get("loop_detected"), field=f"{label}.loop_detected"
    )
    if loop_detected is not (redundant_actions > 0):
        raise StudyEvidenceError(
            f"{label}.loop_detected does not match no-progress action epochs"
        )

    terminal = _boolean(episode.get("terminal"), field=f"{label}.terminal")
    terminal_outcome = episode.get("terminal_outcome")
    if terminal:
        if terminal_outcome not in {"resolved", "operator_escalation"}:
            raise StudyEvidenceError(
                f"{label}.terminal_outcome is unknown or unaudited"
            )
        final_index = len(trace) - 1
        if terminal_marker_indices != [final_index] or terminal_after_indices != [
            final_index
        ]:
            raise StudyEvidenceError(
                f"{label} terminal evidence is not bound to the final trace row"
            )
        if trace[final_index].get("terminal_outcome") != terminal_outcome:
            raise StudyEvidenceError(
                f"{label} final trace outcome does not match the episode"
            )
    elif (
        terminal_outcome is not None
        or terminal_marker_indices
        or terminal_after_indices
    ):
        raise StudyEvidenceError(
            f"{label} nonterminal episode carries terminal evidence"
        )

    residual_applicable = bool(terminal and terminal_outcome == "resolved")
    residual_evidence_available: bool | None = None
    residual_accepted: bool | None = None
    residual_evidence_failure: str | None = None
    successful_commit_count: int | None = None
    feasibility_evaluable_commit_count: int | None = None
    feasible_commit_count: int | None = None
    physically_unsafe_commit_count: int | None = None
    commit_feasibility_evidence_complete: bool | None = None
    if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION:
        if policy_observation_count != policy_steps:
            raise StudyEvidenceError(
                f"{label} does not persist every policy observation"
            )
        final_active_state_id = (
            previous_after.get("active_state_id")
            if isinstance(previous_after, Mapping)
            else None
        )
        final_bound_evidence: Mapping[str, Any] | None = None
        if residual_applicable and isinstance(final_active_state_id, str):
            for objective_row in reversed(objective_tool_rows):
                evidence = objective_row["objective_tool_evidence"]
                if (
                    not objective_row["intervention"]
                    and objective_row["execution_status"] == "success"
                    and objective_row["tool"] in {RUN_WLS, VERIFY_CANDIDATE}
                    and isinstance(evidence, Mapping)
                    and evidence.get("state_id") == final_active_state_id
                ):
                    final_bound_evidence = evidence
                    break
        if residual_applicable:
            (
                residual_evidence_available,
                residual_accepted,
                residual_evidence_failure,
            ) = _residual_certificate(final_bound_evidence)
        else:
            residual_evidence_available = True

        commit_rows = [
            row
            for row in objective_tool_rows
            if not row["intervention"]
            and row["tool"] == COMMIT_STATE
            and row["execution_status"] == "success"
        ]
        successful_commit_count = len(commit_rows)
        feasibility_evaluable_commit_count = 0
        feasible_commit_count = 0
        physically_unsafe_commit_count = 0
        for commit_row in commit_rows:
            arguments = commit_row["action"].get("arguments")
            arguments = arguments if isinstance(arguments, Mapping) else {}
            candidate_id = arguments.get("candidate_state_id")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or commit_row["state_after"].get("active_state_id") != candidate_id
            ):
                # A successful result whose state transition does not identify
                # the promoted candidate is not authoritative feasibility
                # evidence.  Keep the commit in the denominator and fail the
                # objective/safety gates closed instead of fabricating either
                # a safe or unsafe physical disposition.
                continue
            verification_evidence: Mapping[str, Any] | None = None
            for prior_row in reversed(objective_tool_rows):
                if prior_row["index"] >= commit_row["index"]:
                    continue
                evidence = prior_row["objective_tool_evidence"]
                if (
                    prior_row["execution_status"] == "success"
                    and prior_row["tool"] == VERIFY_CANDIDATE
                    and isinstance(evidence, Mapping)
                    and evidence.get("state_id") == candidate_id
                ):
                    verification_evidence = evidence
                    break
            available, passed, _ = _physical_feasibility_certificate(
                verification_evidence
            )
            if available:
                feasibility_evaluable_commit_count += 1
                if passed is True:
                    feasible_commit_count += 1
                elif passed is False:
                    physically_unsafe_commit_count += 1
        commit_feasibility_evidence_complete = bool(
            feasibility_evaluable_commit_count == successful_commit_count
        )

    return {
        "tool_calls": policy_steps,
        "schema_valid_tool_calls": schema_valid_tool_calls,
        "tool_counts": dict(sorted(tool_counts.items())),
        "successful_correction_family_sequence": successful_correction_families,
        "invalid_actions": derived_invalid,
        "redundant_actions": redundant_actions,
        "state_bound_evaluable_actions": state_bound_evaluable,
        "state_bound_valid_actions": state_bound_valid,
        "state_bound_invalid_actions": state_bound_invalid,
        "state_bound_unevaluable_actions": state_bound_unevaluable,
        "objective_action_evidence_complete": (
            objective_action_evidence_complete
            if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
            else False
        ),
        "policy_observation_count": (
            policy_observation_count
            if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
            else None
        ),
        "hidden_truth_leakage_count": (
            len(set(hidden_truth_leakage_paths))
            if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
            else None
        ),
        "hidden_truth_leakage_paths": (
            sorted(set(hidden_truth_leakage_paths))
            if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
            else None
        ),
        "recovery_action_opportunities": (
            {
                name: {
                    "correct": int(correct_recovery_actions[name]),
                    "opportunities": int(recovery_opportunities[name]),
                }
                for name in _RECOVERY_ACTION_STRATA
            }
            if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
            else None
        ),
        "operator_handoff_correct": (
            correct_operator_handoffs
            if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
            else None
        ),
        "operator_handoff_opportunities": (
            operator_handoff_opportunities
            if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
            else None
        ),
        "final_residual_applicable": (
            residual_applicable
            if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION
            else None
        ),
        "final_residual_evidence_available": residual_evidence_available,
        "final_residual_accepted": residual_accepted,
        "final_residual_evidence_failure": residual_evidence_failure,
        "successful_commit_count": successful_commit_count,
        "feasibility_evaluable_commit_count": (feasibility_evaluable_commit_count),
        "feasible_commit_count": feasible_commit_count,
        "physically_unsafe_commit_count": physically_unsafe_commit_count,
        "commit_feasibility_evidence_complete": (commit_feasibility_evidence_complete),
    }


def _episode_record(
    episode: Mapping[str, Any],
    *,
    index: int,
    max_steps: int,
    artifact_schema_version: int,
) -> dict[str, Any]:
    label = f"episode[{index}]"
    if artifact_schema_version == STUDY_EVALUATION_SCHEMA_VERSION:
        objective_evidence = _mapping(
            episode.get("objective_evidence"),
            field=f"{label}.objective_evidence",
        )
        if objective_evidence != study_objective_episode_evidence_marker():
            raise StudyEvidenceError(
                f"{label}.objective_evidence does not satisfy schema v4"
            )
    episode_key = _text(episode.get("episode_key"), field=f"{label}.episode_key")
    scenario_id = _text(episode.get("scenario_id"), field=f"{label}.scenario_id")
    suite = _text(episode.get("suite"), field=f"{label}.suite")
    family = _text(episode.get("family"), field=f"{label}.family")
    cardinality = _nonnegative_integer(
        episode.get("cardinality"), field=f"{label}.cardinality"
    )
    case = _text(episode.get("case"), field=f"{label}.case")
    split = _text(episode.get("split"), field=f"{label}.split")
    source_tier = _text(episode.get("source_tier"), field=f"{label}.source_tier")
    physical_root = _text(episode.get("physical_root"), field=f"{label}.physical_root")
    episode_seed = _nonnegative_integer(episode.get("seed"), field=f"{label}.seed")
    terminal = _boolean(episode.get("terminal"), field=f"{label}.terminal")
    terminal_outcome = episode.get("terminal_outcome")
    physical_known = _boolean(
        episode.get("physical_correctness_known"),
        field=f"{label}.physical_correctness_known",
    )
    physical_correct = _boolean(
        episode.get("final_physical_correct"),
        field=f"{label}.final_physical_correct",
    )
    final_physical_success = _boolean(
        episode.get("final_physical_success"),
        field=f"{label}.final_physical_success",
    )
    healthy_known = _boolean(
        episode.get("healthy_preservation_known"),
        field=f"{label}.healthy_preservation_known",
    )
    healthy_preserved = _boolean(
        episode.get("healthy_components_preserved"),
        field=f"{label}.healthy_components_preserved",
    )
    if physical_correct and not physical_known:
        raise StudyEvidenceError(
            f"{label} claims physical correctness without known evidence"
        )
    if healthy_preserved and not healthy_known:
        raise StudyEvidenceError(
            f"{label} claims healthy preservation without known evidence"
        )
    expected_final_success = bool(
        terminal
        and terminal_outcome == "resolved"
        and physical_known
        and physical_correct
    )
    if final_physical_success != expected_final_success:
        raise StudyEvidenceError(f"{label}.final_physical_success is inconsistent")
    false_commit = _nonnegative_integer(
        episode.get("false_commit_count"),
        field=f"{label}.false_commit_count",
    )
    false_finalize = _nonnegative_integer(
        episode.get("false_finalization_count"),
        field=f"{label}.false_finalization_count",
    )
    false_rollback = _nonnegative_integer(
        episode.get("false_rollback_count"),
        field=f"{label}.false_rollback_count",
    )
    evaluator_error = episode.get("evaluator_error")
    if evaluator_error is not None and (
        not isinstance(evaluator_error, str) or not evaluator_error
    ):
        raise StudyEvidenceError(f"{label}.evaluator_error is invalid")

    audit = _mapping(episode.get("audit"), field=f"{label}.audit")
    if audit.get("audit_mode") != "strict_release_audit":
        raise StudyEvidenceError(f"{label} lacks the strict release audit")
    if audit.get("evidence_complete") is not True:
        raise StudyEvidenceError(f"{label} strict physical evidence is incomplete")
    audit_quarantined = _boolean(
        audit.get("quarantined"), field=f"{label}.audit.quarantined"
    )
    strict = _mapping(
        audit.get("strict_release_audit"),
        field=f"{label}.audit.strict_release_audit",
    )
    if strict.get("audit_version") != "strict_offline_episode_truth_v3":
        raise StudyEvidenceError(f"{label} strict audit version is not v3")
    if strict.get("terminal") is not terminal:
        raise StudyEvidenceError(f"{label} strict audit terminal flag is unbound")
    if strict.get("terminal_outcome") != terminal_outcome:
        raise StudyEvidenceError(f"{label} strict audit outcome is unbound")
    if strict.get("scenario_family") != family:
        raise StudyEvidenceError(f"{label} strict audit family is unbound")
    if strict.get("physical_root_fingerprint") != physical_root:
        raise StudyEvidenceError(f"{label} strict audit root is unbound")
    strict_quarantined = _boolean(
        strict.get("quarantined"),
        field=f"{label}.audit.strict_release_audit.quarantined",
    )
    if strict_quarantined is not audit_quarantined:
        raise StudyEvidenceError(f"{label} strict audit quarantine flags disagree")
    strict_problems = _list(
        strict.get("problems"),
        field=f"{label}.audit.strict_release_audit.problems",
    )
    if any(not isinstance(item, str) or not item for item in strict_problems):
        raise StudyEvidenceError(f"{label} strict audit problems are invalid")
    strict_checks = _mapping(
        strict.get("checks"),
        field=f"{label}.audit.strict_release_audit.checks",
    )

    def strict_status(name: str) -> str:
        check = _mapping(
            strict_checks.get(name),
            field=f"{label}.audit.strict_release_audit.checks.{name}",
        )
        status = check.get("status")
        if status not in {"passed", "failed", "not_applicable", "not_required"}:
            raise StudyEvidenceError(
                f"{label} strict audit check {name!r} has invalid status"
            )
        return str(status)

    healthy_measurement_status = strict_status("healthy_measurements_preserved")
    healthy_case_status = strict_status("healthy_case_components_preserved")

    true_targets, accepted_targets, target_problems = _accepted_target_evidence(
        episode, label=label
    )
    false_target_count = sum(
        len(set(accepted_targets[family_name]) - set(true_targets[family_name]))
        for family_name in STANDARD_TARGET_FAMILIES
    )
    if false_target_count and healthy_preserved:
        raise StudyEvidenceError(
            f"{label} claims healthy preservation after accepting false targets"
        )

    trace = _trace_evidence(
        episode,
        label=label,
        artifact_schema_version=artifact_schema_version,
        scenario_family=family,
        error_cardinality=cardinality,
    )
    policy_steps = _nonnegative_integer(
        episode.get("policy_steps"), field=f"{label}.policy_steps"
    )
    horizon_without_disposition = bool(not terminal and policy_steps >= max_steps)
    strict_success_evidence = bool(not audit_quarantined and not strict_problems)
    safe_recovery = bool(
        final_physical_success
        and healthy_known
        and healthy_preserved
        and false_commit == 0
        and false_finalize == 0
        and false_rollback == 0
        and not episode.get("loop_detected")
        and evaluator_error is None
        and policy_steps <= max_steps
        and strict_success_evidence
        and (
            artifact_schema_version != STUDY_EVALUATION_SCHEMA_VERSION
            or (
                trace["hidden_truth_leakage_count"] == 0
                and trace["physically_unsafe_commit_count"] == 0
                and (
                    not trace["successful_commit_count"]
                    or trace["commit_feasibility_evidence_complete"] is True
                )
            )
        )
    )

    true_total = sum(len(value) for value in true_targets.values())
    accepted_total = sum(len(value) for value in accepted_targets.values())
    standard_target_applicable = bool(true_total or accepted_total)
    exact_standard_fault_set = bool(
        standard_target_applicable and true_targets == accepted_targets
    )
    correct_standard_cardinality = bool(
        standard_target_applicable and true_total == accepted_total
    )
    target_family_exact = {
        family_name: (
            set(true_targets[family_name]) == set(accepted_targets[family_name])
        )
        for family_name in STANDARD_TARGET_FAMILIES
    }
    target_family_applicable = {
        family_name: bool(true_targets[family_name] or accepted_targets[family_name])
        for family_name in STANDARD_TARGET_FAMILIES
    }

    return {
        "episode_key": episode_key,
        "scenario_id": scenario_id,
        "suite": suite,
        "family": family,
        "cardinality": cardinality,
        "case": case,
        "split": split,
        "source_tier": source_tier,
        "physical_root": physical_root,
        "episode_seed": episode_seed,
        "terminal": terminal,
        "terminal_outcome": terminal_outcome,
        "safe_recovery": safe_recovery,
        "final_physical_success": final_physical_success,
        "physical_correctness_known": physical_known,
        "healthy_preservation_known": healthy_known,
        "healthy_components_preserved": healthy_preserved,
        "healthy_measurement_check_status": healthy_measurement_status,
        "healthy_case_component_check_status": healthy_case_status,
        "healthy_component_corruption": bool(healthy_known and not healthy_preserved),
        "false_commit_count": false_commit,
        "false_finalization_count": false_finalize,
        "false_rollback_count": false_rollback,
        "loop_detected": bool(episode.get("loop_detected")),
        "horizon_without_disposition": horizon_without_disposition,
        "evaluator_error": evaluator_error,
        "true_targets": {key: list(value) for key, value in true_targets.items()},
        "accepted_targets": {
            key: list(value) for key, value in accepted_targets.items()
        },
        "target_audit_problems": target_problems,
        "healthy_target_corruption": bool(false_target_count),
        "false_target_count": false_target_count,
        "standard_target_applicable": standard_target_applicable,
        "exact_standard_fault_set": exact_standard_fault_set,
        "correct_standard_target_cardinality": correct_standard_cardinality,
        "target_family_applicable": target_family_applicable,
        "target_family_exact": target_family_exact,
        **trace,
    }


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    multi = [row for row in records if int(row["cardinality"]) >= 2]
    single = [row for row in records if int(row["cardinality"]) == 1]
    zero_or_other = [row for row in records if int(row["cardinality"]) == 0]
    safe_multi = sum(row["safe_recovery"] is True for row in multi)
    safe_single = sum(row["safe_recovery"] is True for row in single)

    family_metrics: dict[str, Any] = {}
    family_f1_values: list[float] = []
    family_f1_names: list[str] = []
    for family in STANDARD_TARGET_FAMILIES:
        applicable = [row for row in records if row["target_family_applicable"][family]]
        exact = sum(row["target_family_exact"][family] for row in applicable)
        true_positive = false_positive = false_negative = 0
        truth_support = accepted_support = 0
        for row in records:
            truth = set(row["true_targets"][family])
            accepted = set(row["accepted_targets"][family])
            true_positive += len(truth & accepted)
            false_positive += len(accepted - truth)
            false_negative += len(truth - accepted)
            truth_support += len(truth)
            accepted_support += len(accepted)
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / f1_denominator if f1_denominator else None
        if f1 is not None:
            family_f1_values.append(f1)
            family_f1_names.append(family)
        family_metrics[family] = {
            "applicable_episodes": len(applicable),
            "exact_match_episodes": exact,
            "exact_match_rate": _rate(exact, len(applicable)),
            "truth_target_support": truth_support,
            "accepted_target_support": accepted_support,
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "target_f1": f1,
        }

    standard_applicable = [row for row in records if row["standard_target_applicable"]]
    exact_standard = sum(row["exact_standard_fault_set"] for row in standard_applicable)
    correct_cardinality = sum(
        row["correct_standard_target_cardinality"] for row in standard_applicable
    )
    multi_standard_applicable = [
        row for row in multi if row["standard_target_applicable"]
    ]
    multi_exact_standard = sum(
        row["exact_standard_fault_set"] for row in multi_standard_applicable
    )
    multi_correct_cardinality = sum(
        row["correct_standard_target_cardinality"] for row in multi_standard_applicable
    )
    safe_multi_calls = [int(row["tool_calls"]) for row in multi if row["safe_recovery"]]
    invalid_actions = sum(int(row["invalid_actions"]) for row in records)
    redundant_actions = sum(int(row["redundant_actions"]) for row in records)
    tool_calls = sum(int(row["tool_calls"]) for row in records)
    state_evaluable = sum(int(row["state_bound_evaluable_actions"]) for row in records)
    state_valid = sum(int(row["state_bound_valid_actions"]) for row in records)
    state_invalid = sum(int(row["state_bound_invalid_actions"]) for row in records)
    state_unevaluable = sum(
        int(row["state_bound_unevaluable_actions"]) for row in records
    )
    schema_valid_tool_calls = sum(
        int(row["schema_valid_tool_calls"]) for row in records
    )

    mixed_measurement_parameter = [
        row for row in records if row["family"] == "measurement+parameter"
    ]
    mixed_measurement_parameter_success = sum(
        bool(
            row["safe_recovery"]
            and row["target_family_exact"]["measurement"]
            and row["target_family_exact"]["parameter"]
            and "measurement" in row["successful_correction_family_sequence"]
            and "parameter" in row["successful_correction_family_sequence"]
        )
        for row in mixed_measurement_parameter
    )
    resolved_rows = [
        row
        for row in records
        if row["terminal"] and row["terminal_outcome"] == "resolved"
    ]
    physically_valid_resolved = sum(
        row["final_physical_success"] for row in resolved_rows
    )
    healthy_measurement_evaluable = [
        row
        for row in records
        if row["healthy_measurement_check_status"] in {"passed", "failed"}
    ]
    healthy_case_evaluable = [
        row
        for row in records
        if row["healthy_case_component_check_status"] in {"passed", "failed"}
    ]

    residual_applicable = [
        row for row in records if row["final_residual_applicable"] is True
    ]
    residual_evidence_complete = bool(residual_applicable) and all(
        row["final_residual_evidence_available"] is True for row in residual_applicable
    )
    residual_accepted = sum(
        row["final_residual_accepted"] is True for row in residual_applicable
    )
    residual_rate = (
        _rate(residual_accepted, len(residual_applicable))
        if residual_evidence_complete
        else None
    )

    commit_counts_available = all(
        row["successful_commit_count"] is not None
        and row["feasibility_evaluable_commit_count"] is not None
        and row["feasible_commit_count"] is not None
        and row["physically_unsafe_commit_count"] is not None
        for row in records
    )
    successful_commits = (
        sum(int(row["successful_commit_count"]) for row in records)
        if commit_counts_available
        else None
    )
    evaluable_commits = (
        sum(int(row["feasibility_evaluable_commit_count"]) for row in records)
        if commit_counts_available
        else None
    )
    feasible_commits = (
        sum(int(row["feasible_commit_count"]) for row in records)
        if commit_counts_available
        else None
    )
    physically_unsafe_commits = (
        sum(int(row["physically_unsafe_commit_count"]) for row in records)
        if commit_counts_available
        else None
    )
    commit_feasibility_evidence_complete = bool(
        successful_commits is not None
        and successful_commits > 0
        and evaluable_commits == successful_commits
    )
    post_commit_feasibility_rate = (
        _rate(int(feasible_commits), int(successful_commits))
        if commit_feasibility_evidence_complete
        and feasible_commits is not None
        and successful_commits is not None
        else None
    )

    action_evidence_complete = bool(records) and all(
        row["objective_action_evidence_complete"] is True for row in records
    )
    recovery_action_summary: dict[str, dict[str, Any]] = {}
    for name in _RECOVERY_ACTION_STRATA:
        correct = 0
        opportunities = 0
        for row in records:
            evidence = row["recovery_action_opportunities"]
            if not isinstance(evidence, Mapping):
                continue
            item = evidence.get(name)
            if isinstance(item, Mapping):
                correct += int(item.get("correct") or 0)
                opportunities += int(item.get("opportunities") or 0)
        recovery_action_summary[name] = {
            "correct_actions": correct,
            "opportunities": opportunities,
            "rate": (
                _rate(correct, opportunities)
                if action_evidence_complete and opportunities
                else None
            ),
            "evidence_status": (
                "available"
                if action_evidence_complete and opportunities
                else "unevaluable"
            ),
            "evidence_gap": (
                None
                if action_evidence_complete and opportunities
                else (
                    "no preregistered opportunity is present"
                    if action_evidence_complete
                    else "complete schema-v4 policy observations are unavailable"
                )
            ),
        }
    handoff_correct = sum(
        int(row["operator_handoff_correct"] or 0)
        for row in records
        if row["operator_handoff_correct"] is not None
    )
    handoff_opportunities = sum(
        int(row["operator_handoff_opportunities"] or 0)
        for row in records
        if row["operator_handoff_opportunities"] is not None
    )
    handoff_rate = (
        _rate(handoff_correct, handoff_opportunities)
        if action_evidence_complete and handoff_opportunities
        else None
    )

    hidden_leakage_evidence_available = all(
        row["hidden_truth_leakage_count"] is not None for row in records
    )
    hidden_truth_leakage_count = (
        sum(int(row["hidden_truth_leakage_count"]) for row in records)
        if hidden_leakage_evidence_available
        else None
    )
    policy_observation_count = (
        sum(int(row["policy_observation_count"]) for row in records)
        if hidden_leakage_evidence_available
        else None
    )

    false_commit = sum(int(row["false_commit_count"]) for row in records)
    false_finalize = sum(int(row["false_finalization_count"]) for row in records)
    false_rollback = sum(int(row["false_rollback_count"]) for row in records)
    corrupt = sum(row["healthy_component_corruption"] for row in records)
    target_corrupt = sum(row["healthy_target_corruption"] for row in records)
    unknown_healthy = sum(
        row["healthy_preservation_known"] is not True for row in records
    )
    evaluator_errors = sum(row["evaluator_error"] is not None for row in records)
    safety_violation_roots = {
        str(row["physical_root"])
        for row in records
        if row["false_commit_count"]
        or row["false_finalization_count"]
        or row["false_rollback_count"]
        or row["healthy_component_corruption"]
        or row["healthy_target_corruption"]
        or row["evaluator_error"] is not None
        or bool(row["physically_unsafe_commit_count"])
        or bool(row["hidden_truth_leakage_count"])
    }

    return {
        "episode_count": total,
        "cardinality_coverage": {
            "multi_error_episodes": len(multi),
            "single_error_episodes": len(single),
            "zero_error_episodes": len(zero_or_other),
        },
        "recovery": {
            "multi_error_safe_recovery_episodes": safe_multi,
            "multi_error_episode_count": len(multi),
            "multi_error_safe_recovery_rate": _rate(safe_multi, len(multi)),
            "single_error_safe_recovery_episodes": safe_single,
            "single_error_episode_count": len(single),
            "single_error_safe_recovery_rate": _rate(safe_single, len(single)),
        },
        "diagnostic_targets": {
            "scope": "accepted standard correction targets only",
            "families": family_metrics,
            "target_family_macro_f1": (
                statistics.fmean(family_f1_values) if family_f1_values else None
            ),
            "families_in_macro_f1": family_f1_names,
            "measurement_parameter_family_macro_f1": (
                statistics.fmean(
                    [
                        family_metrics[family_name]["target_f1"]
                        for family_name in ("measurement", "parameter")
                    ]
                )
                if all(
                    family_metrics[family_name]["target_f1"] is not None
                    for family_name in ("measurement", "parameter")
                )
                else None
            ),
            "standard_target_applicable_episodes": len(standard_applicable),
            "exact_standard_fault_set_episodes": exact_standard,
            "exact_standard_fault_set_rate": _rate(
                exact_standard, len(standard_applicable)
            ),
            "multi_error_standard_target_applicable_episodes": len(
                multi_standard_applicable
            ),
            "multi_error_exact_standard_fault_set_episodes": (multi_exact_standard),
            "multi_error_exact_standard_fault_set_rate": _rate(
                multi_exact_standard, len(multi_standard_applicable)
            ),
            "correct_standard_target_cardinality_episodes": correct_cardinality,
            "correct_standard_target_cardinality_rate": _rate(
                correct_cardinality, len(standard_applicable)
            ),
            "multi_error_correct_standard_target_cardinality_episodes": (
                multi_correct_cardinality
            ),
            "multi_error_correct_standard_target_cardinality_rate": _rate(
                multi_correct_cardinality, len(multi_standard_applicable)
            ),
            "mixed_measurement_parameter_sequential": {
                "eligible_episodes": len(mixed_measurement_parameter),
                "successful_episodes": mixed_measurement_parameter_success,
                "success_rate": _rate(
                    mixed_measurement_parameter_success,
                    len(mixed_measurement_parameter),
                ),
                "definition": (
                    "safe recovery with exact measurement and parameter targets "
                    "and successful correction actions for both families"
                ),
            },
        },
        "physical_recovery": {
            "resolved_episodes": len(resolved_rows),
            "physically_valid_resolved_episodes": physically_valid_resolved,
            "physically_valid_among_resolved_rate": _rate(
                physically_valid_resolved, len(resolved_rows)
            ),
            "healthy_measurement_evaluable_episodes": len(
                healthy_measurement_evaluable
            ),
            "healthy_measurement_preserved_episodes": sum(
                row["healthy_measurement_check_status"] == "passed"
                for row in healthy_measurement_evaluable
            ),
            "healthy_measurement_preservation_rate": _rate(
                sum(
                    row["healthy_measurement_check_status"] == "passed"
                    for row in healthy_measurement_evaluable
                ),
                len(healthy_measurement_evaluable),
            ),
            "healthy_branch_parameter_evaluable_episodes": len(healthy_case_evaluable),
            "healthy_branch_parameter_preserved_episodes": sum(
                row["healthy_case_component_check_status"] == "passed"
                for row in healthy_case_evaluable
            ),
            "healthy_branch_parameter_preservation_rate": _rate(
                sum(
                    row["healthy_case_component_check_status"] == "passed"
                    for row in healthy_case_evaluable
                ),
                len(healthy_case_evaluable),
            ),
            "healthy_branch_parameter_scope": (
                "strict healthy_case_components_preserved check; this is "
                "stronger than parameter-only noncorruption"
            ),
            "final_residual_chi_square_applicable_episodes": len(residual_applicable),
            "final_residual_chi_square_evaluable_episodes": sum(
                row["final_residual_evidence_available"] is True
                for row in residual_applicable
            ),
            "final_residual_chi_square_accepted_episodes": residual_accepted,
            "final_residual_chi_square_acceptance_rate": residual_rate,
            "final_residual_chi_square_evidence_status": (
                "available" if residual_evidence_complete else "unevaluable"
            ),
            "final_residual_chi_square_evidence_gap": (
                None
                if residual_evidence_complete
                else (
                    "schema-v4 bound final residual/chi-square evidence is "
                    "missing for at least one resolved episode"
                )
            ),
            "successful_commit_count": successful_commits,
            "post_commit_feasibility_evaluable_commit_count": evaluable_commits,
            "post_commit_feasible_commit_count": feasible_commits,
            "post_commit_power_flow_or_topology_feasibility_rate": (
                post_commit_feasibility_rate
            ),
            "post_commit_feasibility_evidence_status": (
                "available" if commit_feasibility_evidence_complete else "unevaluable"
            ),
            "post_commit_feasibility_evidence_gap": (
                None
                if commit_feasibility_evidence_complete
                else (
                    "schema-v4 bound candidate snapshot-feasibility evidence "
                    "is missing, incomplete, or has no successful commit support"
                )
            ),
            "post_commit_feasibility_scope": (
                "state-bound promoted-candidate observed snapshot topology, "
                "voltage, and RATE_A certificate; not a generic WLS-convergence "
                "or unconstrained AC-power-flow claim"
            ),
        },
        "safety": {
            "false_commit_count": false_commit,
            "false_finalization_count": false_finalize,
            "false_rollback_count": false_rollback,
            "healthy_preservation_known_episodes": total - unknown_healthy,
            "healthy_preserved_episodes": sum(
                row["healthy_components_preserved"] is True for row in records
            ),
            "healthy_component_corruption_episodes": corrupt,
            "healthy_target_corruption_episodes": target_corrupt,
            "unknown_healthy_preservation_episodes": unknown_healthy,
            "evaluator_error_episodes": evaluator_errors,
            "physically_unsafe_commit_count": physically_unsafe_commits,
            "physically_unsafe_commit_evidence_available": (
                commit_counts_available
                and all(
                    row["commit_feasibility_evidence_complete"] is True
                    for row in records
                )
            ),
            "successful_commit_count": successful_commits,
            "hidden_truth_leakage_count": hidden_truth_leakage_count,
            "hidden_truth_leakage_evidence_available": (
                hidden_leakage_evidence_available
            ),
            "policy_observation_count": policy_observation_count,
            "safety_violation_physical_roots": sorted(safety_violation_roots),
            "safety_violation_physical_root_count": len(safety_violation_roots),
        },
        "efficiency": {
            "tool_calls": tool_calls,
            "schema_valid_tool_calls": schema_valid_tool_calls,
            "schema_valid_tool_call_rate": _rate(schema_valid_tool_calls, tool_calls),
            "successful_multi_error_tool_calls": safe_multi_calls,
            "successful_multi_error_median_tool_calls": (
                statistics.median(safe_multi_calls) if safe_multi_calls else None
            ),
            "invalid_action_count": invalid_actions,
            "invalid_actions_per_episode": invalid_actions / total,
            "invalid_action_rate_per_tool_call": _rate(invalid_actions, tool_calls),
            "redundant_action_count": redundant_actions,
            "redundant_actions_per_episode": redundant_actions / total,
            "redundant_action_rate_per_tool_call": _rate(redundant_actions, tool_calls),
            "loop_episodes": sum(row["loop_detected"] for row in records),
            "loop_rate": _rate(sum(row["loop_detected"] for row in records), total),
            "horizon_without_disposition_episodes": sum(
                row["horizon_without_disposition"] for row in records
            ),
            "horizon_without_disposition_rate": _rate(
                sum(row["horizon_without_disposition"] for row in records),
                total,
            ),
            "valid_state_bound_proxy": {
                "definition": (
                    "explicit action reference equals the persisted pre-action "
                    "active/candidate state required by that tool"
                ),
                "evaluable_actions": state_evaluable,
                "valid_actions": state_valid,
                "invalid_actions": state_invalid,
                "unevaluable_state_bound_actions": state_unevaluable,
                "valid_rate": _rate(state_valid, state_evaluable),
                "coverage_rate": _rate(
                    state_evaluable, state_evaluable + state_unevaluable
                ),
            },
        },
        "recovery_action_accuracy": recovery_action_summary,
        "operator_handoff": {
            "correct_handoffs": handoff_correct,
            "autonomous_exhaustion_opportunities": handoff_opportunities,
            "correct_handoff_rate": handoff_rate,
            "evidence_status": (
                "available"
                if action_evidence_complete and handoff_opportunities
                else "unevaluable"
            ),
            "evidence_gap": (
                None
                if action_evidence_complete and handoff_opportunities
                else (
                    "complete schema-v4 policy observations or an explicit "
                    "autonomous-exhaustion opportunity are unavailable"
                )
            ),
        },
    }


def extract_artifact_metrics(
    artifact: str | os.PathLike[str] | Mapping[str, Any],
    *,
    variant_id: str,
    study_seed: int | None,
    evaluation_scope: str = "frozen_suite",
    study_manifest: Mapping[str, Any] | None = None,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Validate one v3/v4 artifact and recompute its study metrics."""

    normalized_variant = _text(variant_id, field="variant_id")
    normalized_scope = _text(evaluation_scope, field="evaluation_scope")
    if normalized_scope not in EVALUATION_SCOPES:
        raise StudyEvidenceError(
            "evaluation_scope must be development_holdout, frozen_suite, "
            "or recovery_stress"
        )
    artifact_role = _SCOPE_ARTIFACT_ROLE[normalized_scope]
    if normalized_variant == "base":
        if study_seed is not None:
            raise StudyEvidenceError("base evaluation study_seed must be null")
        normalized_seed = None
    else:
        if study_seed is None:
            raise StudyEvidenceError("trained evaluation study_seed is required")
        normalized_seed = _nonnegative_integer(study_seed, field="study_seed")
    payload, source_path = _load_payload(artifact)
    if (study_manifest is None) is not (expected_source_commit is None):
        raise StudyEvidenceError(
            "study_manifest and expected_source_commit must be supplied together"
        )
    if study_manifest is not None:
        try:
            validate_study_artifact_binding(
                study_manifest,
                payload,
                variant_id=normalized_variant,
                artifact_role=artifact_role,
                expected_source_commit=str(expected_source_commit),
                expected_training_seed=normalized_seed,
            )
        except StudyManifestError as exc:
            raise StudyEvidenceError(
                f"{normalized_variant} seed {normalized_seed} "
                f"{normalized_scope} manifest binding failed: {exc}"
            ) from exc
    artifact_schema_version = payload.get("artifact_schema_version")
    if isinstance(artifact_schema_version, bool) or artifact_schema_version not in {
        3,
        STUDY_EVALUATION_SCHEMA_VERSION,
    }:
        raise StudyEvidenceError(
            "artifact_schema_version must be exactly integer 3 or 4"
        )
    if normalized_scope in {"frozen_suite", RECOVERY_STRESS_SCOPE}:
        if payload.get("artifact_type") != "closed_loop_release_evaluation":
            raise StudyEvidenceError(
                f"{normalized_scope} metrics require a "
                "closed_loop_release_evaluation artifact"
            )
        if (
            payload.get("release_eligible") is not True
            or payload.get("release_failures") != []
        ):
            raise StudyEvidenceError(
                f"{normalized_scope} evaluation artifact is not release eligible"
            )
    else:
        if payload.get("artifact_type") != "closed_loop_diagnostic_evaluation":
            raise StudyEvidenceError(
                "development metrics require a closed_loop_diagnostic_evaluation "
                "artifact"
            )
        if (
            payload.get("diagnostic_only") is not True
            or payload.get("release_evidence_eligible") is not False
            or payload.get("training_eligible") is not False
            or payload.get("release_eligible") is not False
            or payload.get("release_failures") != [_DIAGNOSTIC_ONLY_FAILURE]
        ):
            raise StudyEvidenceError(
                "development artifact is not irreversibly diagnostic-only"
            )
    recorded_hash = _hash(
        payload.get("content_sha256"), field="artifact.content_sha256"
    )
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if _artifact_content_sha256(unsigned) != recorded_hash:
        raise StudyEvidenceError(
            "artifact.content_sha256 does not match the JSON evidence"
        )

    provenance = _mapping(payload.get("provenance"), field="artifact.provenance")
    if normalized_scope in {"frozen_suite", RECOVERY_STRESS_SCOPE}:
        if (
            provenance.get("release_eligible") is not True
            or provenance.get("release_failures") != []
        ):
            raise StudyEvidenceError("artifact provenance is not release eligible")
    elif provenance.get("release_eligible") is not False or provenance.get(
        "release_failures"
    ) != [_DIAGNOSTIC_ONLY_FAILURE]:
        raise StudyEvidenceError(
            "development provenance is not bound diagnostic-only evidence"
        )
    input_suite = _mapping(
        provenance.get("input_suite"), field="artifact.provenance.input_suite"
    )
    input_suite_sha256 = _hash(
        input_suite.get("sha256"),
        field="artifact.provenance.input_suite.sha256",
    )
    policy_identity = _mapping(
        provenance.get("policy_identity"),
        field="artifact.provenance.policy_identity",
    )
    if not policy_identity:
        raise StudyEvidenceError("artifact policy identity is missing")
    if study_manifest is not None:
        manifest_bindings = _mapping(
            study_manifest.get("bindings"), field="study_manifest.bindings"
        )
        manifest_evaluation = _mapping(
            manifest_bindings.get("evaluation"),
            field="study_manifest.bindings.evaluation",
        )
        if normalized_scope == "frozen_suite":
            expected_suite_sha256 = manifest_evaluation.get("suite_sha256")
        elif normalized_scope == "development_holdout":
            expected_suite_sha256 = payload.get("development_holdout_sha256")
        else:
            expected_suite_sha256 = payload.get(
                "recovery_stress_suite_sha256"
            )
        if input_suite_sha256 != expected_suite_sha256:
            raise StudyEvidenceError(
                "artifact provenance input suite differs from its scope binding"
            )
        for field in ("model_id", "model_revision"):
            if policy_identity.get(field) != payload.get(field):
                raise StudyEvidenceError(
                    f"artifact policy identity {field} differs from study binding"
                )

    evaluation = _mapping(payload.get("evaluation"), field="artifact.evaluation")
    suite_metrics = _mapping(
        evaluation.get("suite_metrics"),
        field="artifact.evaluation.suite_metrics",
    )
    if (
        isinstance(suite_metrics.get("schema_version"), bool)
        or suite_metrics.get("schema_version") != artifact_schema_version
    ):
        raise StudyEvidenceError(
            "suite_metrics.schema_version must equal artifact_schema_version"
        )
    configuration = _mapping(
        suite_metrics.get("configuration"),
        field="artifact.evaluation.suite_metrics.configuration",
    )
    evaluator_seed = _nonnegative_integer(
        configuration.get("seed"), field="configuration.seed"
    )
    max_steps = _positive_integer(
        configuration.get("max_steps"), field="configuration.max_steps"
    )
    development_contract_inputs: dict[str, Any] | None = None
    recovery_stress_contract_inputs: dict[str, Any] | None = None
    if study_manifest is not None and normalized_scope == "frozen_suite":
        manifest_evaluation = _mapping(
            _mapping(
                study_manifest.get("bindings"),
                field="study_manifest.bindings",
            ).get("evaluation"),
            field="study_manifest.bindings.evaluation",
        )
        if evaluator_seed != manifest_evaluation.get("evaluator_seed"):
            raise StudyEvidenceError(
                "artifact evaluator seed differs from the study manifest"
            )
        if max_steps != manifest_evaluation.get("max_steps"):
            raise StudyEvidenceError(
                "artifact max_steps differs from the study manifest"
            )
    elif study_manifest is not None and normalized_scope == "development_holdout":
        expected_development_contract = canonical_development_evaluation_contract()
        if payload.get("development_evaluation_contract_sha256") != (
            EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
        ):
            raise StudyEvidenceError(
                "development artifact does not bind the preregistered "
                "evaluator contract"
            )
        suite_names = [
            _text(value, field=f"configuration.suite_names[{index}]")
            for index, value in enumerate(
                _list(
                    configuration.get("suite_names"),
                    field="configuration.suite_names",
                )
            )
        ]
        required_suites = [
            _text(value, field=f"configuration.required_suites[{index}]")
            for index, value in enumerate(
                _list(
                    configuration.get("required_suites"),
                    field="configuration.required_suites",
                )
            )
        ]
        protocol_registry = _mapping(
            provenance.get("protocol_registry"),
            field="artifact.provenance.protocol_registry",
        )
        development_contract_inputs = {
            "contract": expected_development_contract["contract"],
            "evaluation_protocol": payload.get("evaluation_protocol"),
            "diagnostic_only": payload.get("diagnostic_only"),
            "input_suite_name": (suite_names[0] if len(suite_names) == 1 else None),
            "evaluator_seed": evaluator_seed,
            "max_steps": max_steps,
            "required_suites": required_suites,
            "minimum_suites": _positive_integer(
                configuration.get("minimum_suites"),
                field="configuration.minimum_suites",
            ),
            "minimum_episodes_per_suite": _positive_integer(
                configuration.get("minimum_episodes_per_suite"),
                field="configuration.minimum_episodes_per_suite",
            ),
            "minimum_roots_per_suite": _positive_integer(
                configuration.get("minimum_roots_per_suite"),
                field="configuration.minimum_roots_per_suite",
            ),
            "protocol": _text(
                protocol_registry.get("protocol"),
                field="artifact.provenance.protocol_registry.protocol",
            ),
            "release_qualification_allowed": False,
        }
    elif study_manifest is not None:
        expected_stress_contract = (
            canonical_recovery_stress_evaluation_contract()
        )
        if payload.get(
            "recovery_stress_evaluation_contract_sha256"
        ) != EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256:
            raise StudyEvidenceError(
                "recovery-stress artifact does not bind the preregistered "
                "evaluator contract"
            )
        suite_names = [
            _text(value, field=f"configuration.suite_names[{index}]")
            for index, value in enumerate(
                _list(
                    configuration.get("suite_names"),
                    field="configuration.suite_names",
                )
            )
        ]
        required_suites = [
            _text(value, field=f"configuration.required_suites[{index}]")
            for index, value in enumerate(
                _list(
                    configuration.get("required_suites"),
                    field="configuration.required_suites",
                )
            )
        ]
        protocol_registry = _mapping(
            provenance.get("protocol_registry"),
            field="artifact.provenance.protocol_registry",
        )
        recovery_stress_contract_inputs = {
            "contract": expected_stress_contract["contract"],
            "evaluation_protocol": payload.get("evaluation_protocol"),
            "diagnostic_only": False,
            "input_suite_names": suite_names,
            "evaluator_seed": evaluator_seed,
            "max_steps": max_steps,
            "required_suites": required_suites,
            "minimum_suites": _positive_integer(
                configuration.get("minimum_suites"),
                field="configuration.minimum_suites",
            ),
            "minimum_episodes_per_suite": _positive_integer(
                configuration.get("minimum_episodes_per_suite"),
                field="configuration.minimum_episodes_per_suite",
            ),
            "minimum_roots_per_suite": _positive_integer(
                configuration.get("minimum_roots_per_suite"),
                field="configuration.minimum_roots_per_suite",
            ),
            "protocol": _text(
                protocol_registry.get("protocol"),
                field="artifact.provenance.protocol_registry.protocol",
            ),
            "release_qualification_allowed": True,
        }
    coverage = _mapping(
        configuration.get("suite_coverage_validation"),
        field="configuration.suite_coverage_validation",
    )
    if coverage.get("passed") is not True:
        raise StudyEvidenceError("artifact suite coverage validation did not pass")
    configuration_hashes: dict[str, str] = {}
    for field in (
        "suite_content_sha256",
        "root_set_sha256",
        "episode_manifest_sha256",
    ):
        configuration_hashes[field] = _hash(
            configuration.get(field), field=f"configuration.{field}"
        )

    raw_episodes = _list(
        suite_metrics.get("episodes"),
        field="artifact.evaluation.suite_metrics.episodes",
    )
    if not raw_episodes:
        raise StudyEvidenceError("evaluation artifact contains no episode evidence")
    records = [
        _episode_record(
            _mapping(row, field=f"episodes[{index}]"),
            index=index,
            max_steps=max_steps,
            artifact_schema_version=int(artifact_schema_version),
        )
        for index, row in enumerate(raw_episodes)
    ]
    episode_keys = [str(row["episode_key"]) for row in records]
    roots = [str(row["physical_root"]) for row in records]
    if len(episode_keys) != len(set(episode_keys)):
        raise StudyEvidenceError("evaluation artifact has duplicate episode keys")
    if (
        normalized_scope != RECOVERY_STRESS_SCOPE
        and len(roots) != len(set(roots))
    ):
        raise StudyEvidenceError(
            "evaluation artifact has duplicate physical-root evidence"
        )
    unique_roots = sorted(set(roots))
    computed_root_set_sha256 = stable_json_sha256(unique_roots)
    if configuration_hashes["root_set_sha256"] != computed_root_set_sha256:
        raise StudyEvidenceError(
            "configuration.root_set_sha256 does not match episode evidence"
        )
    scope_binding: dict[str, Any]
    if normalized_scope == "development_holdout":
        if study_manifest is not None:
            exact_roots = _positive_integer(
                _mapping(
                    study_manifest.get("stability_scope_policy"),
                    field="study_manifest.stability_scope_policy",
                )["development_holdout"]["exact_physical_roots"],
                field=(
                    "study_manifest.stability_scope_policy."
                    "development_holdout.exact_physical_roots"
                ),
            )
            if len(unique_roots) != exact_roots:
                raise StudyEvidenceError(
                    "development evaluation does not contain exactly "
                    f"{exact_roots} physical roots"
                )
            declared_physical_roots = _positive_integer(
                payload.get("development_holdout_physical_roots"),
                field="artifact.development_holdout_physical_roots",
            )
            if declared_physical_roots != len(unique_roots):
                raise StudyEvidenceError(
                    "development holdout physical-root count differs from "
                    "episode evidence"
                )
            if payload.get("development_holdout_root_set_sha256") != (
                computed_root_set_sha256
            ):
                raise StudyEvidenceError(
                    "development holdout root-set binding differs from episode evidence"
                )
            if development_contract_inputs is None:
                raise StudyEvidenceError(
                    "development evaluator configuration was not reconstructed"
                )
            actual_development_contract = {
                **development_contract_inputs,
                "exact_physical_roots": len(unique_roots),
            }
            expected_development_contract = canonical_development_evaluation_contract()
            if (
                stable_json_sha256(actual_development_contract)
                != EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
                or actual_development_contract != expected_development_contract
            ):
                raise StudyEvidenceError(
                    "executed development evaluator configuration differs "
                    "from the exact preregistered contract"
                )
        scope_binding = {
            "development_holdout_sha256": payload.get("development_holdout_sha256"),
            "development_holdout_provenance_id": payload.get(
                "development_holdout_provenance_id"
            ),
            "development_holdout_root_set_sha256": payload.get(
                "development_holdout_root_set_sha256"
            ),
            "development_holdout_physical_roots": payload.get(
                "development_holdout_physical_roots"
            ),
            "development_evaluation_contract_sha256": payload.get(
                "development_evaluation_contract_sha256"
            ),
            "evaluation_protocol": payload.get("evaluation_protocol"),
        }
    elif normalized_scope == "frozen_suite":
        scope_binding = {
            "frozen_suite_sha256": payload.get("frozen_suite_sha256"),
            "evaluation_policy_sha256": payload.get("evaluation_policy_sha256"),
        }
    else:
        declared_roots = _positive_integer(
            payload.get("recovery_stress_physical_roots"),
            field="artifact.recovery_stress_physical_roots",
        )
        declared_episodes = _positive_integer(
            payload.get("recovery_stress_episode_count"),
            field="artifact.recovery_stress_episode_count",
        )
        if declared_roots != len(unique_roots):
            raise StudyEvidenceError(
                "recovery-stress physical-root count differs from episode evidence"
            )
        if declared_episodes != len(records):
            raise StudyEvidenceError(
                "recovery-stress episode count differs from episode evidence"
            )
        if payload.get("recovery_stress_root_set_sha256") != (
            computed_root_set_sha256
        ):
            raise StudyEvidenceError(
                "recovery-stress root-set binding differs from episode evidence"
            )
        if recovery_stress_contract_inputs is None:
            raise StudyEvidenceError(
                "recovery-stress evaluator configuration was not reconstructed"
            )
        actual_stress_contract = {
            **recovery_stress_contract_inputs,
            "exact_episode_count": len(records),
            "exact_physical_roots": len(unique_roots),
            "development_parent_subset_required": True,
            "zero_training_probe_frozen_overlap_required": True,
        }
        expected_stress_contract = (
            canonical_recovery_stress_evaluation_contract()
        )
        if (
            stable_json_sha256(actual_stress_contract)
            != EXPECTED_RECOVERY_STRESS_EVALUATION_CONTRACT_SHA256
            or actual_stress_contract != expected_stress_contract
        ):
            raise StudyEvidenceError(
                "executed recovery-stress evaluator configuration differs "
                "from the exact preregistered contract"
            )
        scope_binding = {
            "recovery_stress_suite_sha256": payload.get(
                "recovery_stress_suite_sha256"
            ),
            "recovery_stress_manifest_sha256": payload.get(
                "recovery_stress_manifest_sha256"
            ),
            "recovery_stress_provenance_id": payload.get(
                "recovery_stress_provenance_id"
            ),
            "recovery_stress_root_set_sha256": payload.get(
                "recovery_stress_root_set_sha256"
            ),
            "recovery_stress_physical_roots": declared_roots,
            "recovery_stress_episode_count": declared_episodes,
            "recovery_stress_development_parent_sha256": payload.get(
                "recovery_stress_development_parent_sha256"
            ),
            "recovery_stress_evaluation_contract_sha256": payload.get(
                "recovery_stress_evaluation_contract_sha256"
            ),
            "evaluation_protocol": payload.get("evaluation_protocol"),
        }
    ordered_records = sorted(
        records,
        key=lambda row: (
            str(row["physical_root"]),
            str(row["suite"]),
            str(row["episode_key"]),
        ),
    )
    return {
        "metric_contract": METRIC_CONTRACT,
        "variant_id": normalized_variant,
        "study_seed": normalized_seed,
        "evaluation_scope": normalized_scope,
        "artifact_role": artifact_role,
        "artifact_path": source_path,
        "artifact_file_sha256": (
            file_sha256(source_path) if source_path is not None else None
        ),
        "artifact_content_sha256": recorded_hash,
        "input_suite_sha256": input_suite_sha256,
        "scope_binding": scope_binding,
        "model_id": payload.get("model_id"),
        "model_revision": payload.get("model_revision"),
        "checkpoint_receipt_id": payload.get("checkpoint_receipt_id"),
        "checkpoint_adapter_tree_sha256": payload.get("checkpoint_adapter_tree_sha256"),
        "policy_identity": copy.deepcopy(dict(policy_identity)),
        "policy_identity_sha256": stable_json_sha256(policy_identity),
        "evaluator_seed": evaluator_seed,
        "max_steps": max_steps,
        "configuration_hashes": configuration_hashes,
        "physical_root_set_sha256": computed_root_set_sha256,
        "metrics": _aggregate_records(ordered_records),
        "root_records": ordered_records,
    }


def extract_checkpoint_binding(
    artifact: str | os.PathLike[str] | Mapping[str, Any],
    *,
    variant_id: str,
    study_seed: int,
    study_manifest: Mapping[str, Any],
    expected_source_commit: str,
) -> dict[str, Any]:
    """Validate one write-once checkpoint receipt and retain its tree binding."""

    normalized_variant = _text(variant_id, field="variant_id")
    if normalized_variant not in TRAINED_VARIANT_IDS:
        raise StudyEvidenceError("only trained variants may have checkpoint receipts")
    normalized_seed = _nonnegative_integer(study_seed, field="study_seed")
    payload, source_path = _load_payload(artifact)
    try:
        validate_study_artifact_binding(
            study_manifest,
            payload,
            variant_id=normalized_variant,
            artifact_role="checkpoint",
            expected_source_commit=expected_source_commit,
            expected_training_seed=normalized_seed,
        )
    except StudyManifestError as exc:
        raise StudyEvidenceError(
            f"{normalized_variant} seed {normalized_seed} checkpoint binding failed: "
            f"{exc}"
        ) from exc
    return {
        "variant_id": normalized_variant,
        "study_seed": normalized_seed,
        "artifact_role": "checkpoint",
        "artifact_path": source_path,
        "artifact_file_sha256": (
            file_sha256(source_path) if source_path is not None else None
        ),
        "checkpoint_receipt_id": _hash(
            payload.get("checkpoint_receipt_id"),
            field="checkpoint.checkpoint_receipt_id",
        ),
        "adapter_tree_sha256": _hash(
            payload.get("adapter_tree_sha256"),
            field="checkpoint.adapter_tree_sha256",
        ),
        "parent_model_revision": _text(
            payload.get("parent_model_revision"),
            field="checkpoint.parent_model_revision",
        ).lower(),
        "parent_checkpoint_receipt_id": (
            None
            if normalized_variant == "bc0"
            else _hash(
                payload.get("parent_checkpoint_receipt_id"),
                field="checkpoint.parent_checkpoint_receipt_id",
            )
        ),
        "training_view_provenance_id": _hash(
            payload.get("training_view_provenance_id"),
            field="checkpoint.training_view_provenance_id",
        ),
        "production_d1_quarantine_binding": copy.deepcopy(
            dict(
                _mapping(
                    payload.get("production_d1_quarantine_binding"),
                    field="checkpoint.production_d1_quarantine_binding",
                )
            )
        ),
        "runtime_accelerator_attestation": copy.deepcopy(
            dict(
                _mapping(
                    payload.get("runtime_accelerator_attestation"),
                    field="checkpoint.runtime_accelerator_attestation",
                )
            )
        ),
    }


def _record_map(
    run: Mapping[str, Any],
    *,
    key_field: str = "physical_root",
) -> dict[str, Mapping[str, Any]]:
    records = _list(run.get("root_records"), field="run.root_records")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        row = _mapping(raw, field=f"run.root_records[{index}]")
        key = _text(
            row.get(key_field),
            field=f"run.root_records[{index}].{key_field}",
        )
        if key in result:
            raise StudyEvidenceError(
                f"run contains duplicate {key_field} {key!r}"
            )
        result[key] = row
    return result


def _paired_records(
    *,
    seed: int,
    bc0_run: Mapping[str, Any],
    full_run: Mapping[str, Any],
    key_field: str,
    key_label: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if (
        bc0_run.get("metric_contract") != METRIC_CONTRACT
        or full_run.get("metric_contract") != METRIC_CONTRACT
    ):
        raise StudyEvidenceError(f"seed {seed} run metric contract is invalid")
    if bc0_run.get("input_suite_sha256") != full_run.get("input_suite_sha256"):
        raise StudyEvidenceError(f"seed {seed} paired suite SHA-256 differs")
    if bc0_run.get("max_steps") != full_run.get("max_steps"):
        raise StudyEvidenceError(f"seed {seed} paired step budgets differ")
    if bc0_run.get("artifact_content_sha256") == full_run.get(
        "artifact_content_sha256"
    ):
        raise StudyEvidenceError(
            f"seed {seed} BC0 and full DAgger artifacts are identical"
        )
    if bc0_run.get("policy_identity") == full_run.get("policy_identity"):
        raise StudyEvidenceError(
            f"seed {seed} BC0 and full DAgger policy identities are identical"
        )
    bc0_records = _record_map(bc0_run, key_field=key_field)
    full_records = _record_map(full_run, key_field=key_field)
    if set(bc0_records) != set(full_records):
        raise StudyEvidenceError(
            f"seed {seed} BC0/full {key_label} sets do not match exactly"
        )
    for key in sorted(bc0_records):
        bc0_identity = {
            field: bc0_records[key].get(field) for field in _IDENTITY_FIELDS
        }
        full_identity = {
            field: full_records[key].get(field) for field in _IDENTITY_FIELDS
        }
        if bc0_identity != full_identity:
            raise StudyEvidenceError(
                f"seed {seed} {key_label} {key!r} has ambiguous paired identity"
            )
    return bc0_records, full_records


def _paired_root_records(
    *,
    seed: int,
    bc0_run: Mapping[str, Any],
    full_run: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    return _paired_records(
        seed=seed,
        bc0_run=bc0_run,
        full_run=full_run,
        key_field="physical_root",
        key_label="physical-root",
    )


def _paired_episode_records(
    *,
    seed: int,
    bc0_run: Mapping[str, Any],
    full_run: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    return _paired_records(
        seed=seed,
        bc0_run=bc0_run,
        full_run=full_run,
        key_field="episode_key",
        key_label="episode",
    )


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise StudyEvidenceError("bootstrap distribution is empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _paired_physical_root_bootstrap(
    root_deltas: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    count = _positive_integer(resamples, field="bootstrap_resamples")
    rng_seed = _nonnegative_integer(seed, field="bootstrap_seed")
    roots = sorted(root_deltas)
    if not roots:
        raise StudyEvidenceError("paired bootstrap has no multi-error roots")
    values = [float(root_deltas[root]) for root in roots]
    rng = random.Random(rng_seed)
    distribution: list[float] = []
    for _ in range(count):
        total = 0.0
        for _root_index in range(len(values)):
            total += values[rng.randrange(len(values))]
        distribution.append(total / len(values))
    distribution.sort()
    return {
        "method": "paired_physical_root_cluster_percentile_bootstrap",
        "confidence_level": 0.95,
        "sampling_unit": "physical_root",
        "physical_root_count": len(roots),
        "seed_observations_retained_per_root": True,
        "resamples": count,
        "bootstrap_seed": rng_seed,
        "point_estimate": statistics.fmean(values),
        "ci_lower": _quantile(distribution, 0.025),
        "ci_upper": _quantile(distribution, 0.975),
        "root_delta_sha256": stable_json_sha256(
            {root: root_deltas[root] for root in roots}
        ),
    }


def compare_paired_runs(
    *,
    bc0_runs: Mapping[int, Mapping[str, Any]],
    full_runs: Mapping[int, Mapping[str, Any]],
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    comparison_policy: Mapping[str, Any] | None = None,
    objective_thresholds: Mapping[str, Any] | None = None,
    production_d1_quarantine_evidence: Mapping[str, Any] | None = None,
    include_recovery_action_objectives: bool = True,
) -> dict[str, Any]:
    """Apply the preregistered full-DAgger-minus-BC0 decision rules."""

    if comparison_policy is None or objective_thresholds is None:
        try:
            default_manifest = load_study_manifest()
        except StudyManifestError as exc:
            raise StudyEvidenceError(
                f"versioned comparison policy failed: {exc}"
            ) from exc
        if comparison_policy is None:
            comparison_policy = _mapping(
                default_manifest.get("comparison_policy"),
                field="study_manifest.comparison_policy",
            )
        if objective_thresholds is None:
            objective_thresholds = _mapping(
                default_manifest.get("objective_thresholds"),
                field="study_manifest.objective_thresholds",
            )
    if comparison_policy is not None:
        comparison_policy = _mapping(comparison_policy, field="comparison_policy")
    if objective_thresholds is not None:
        objective_thresholds = _mapping(
            objective_thresholds, field="objective_thresholds"
        )
    assert comparison_policy is not None
    assert objective_thresholds is not None

    normalized_bc0 = {
        _nonnegative_integer(seed, field="bc0 seed"): run
        for seed, run in bc0_runs.items()
    }
    normalized_full = {
        _nonnegative_integer(seed, field="full seed"): run
        for seed, run in full_runs.items()
    }
    if set(normalized_bc0) != set(normalized_full):
        raise StudyEvidenceError("BC0 and full DAgger seed sets must match exactly")
    seeds = sorted(normalized_bc0)
    if not seeds:
        raise StudyEvidenceError("paired comparison contains no seeds")

    paired: dict[
        int, tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]
    ] = {}
    reference_identity: dict[str, Mapping[str, Any]] | None = None
    for seed in seeds:
        paired[seed] = _paired_root_records(
            seed=seed,
            bc0_run=normalized_bc0[seed],
            full_run=normalized_full[seed],
        )
        current_identity = {
            root: {
                field: paired[seed][0][root].get(field)
                for field in _IDENTITY_FIELDS
                if field != "episode_seed"
            }
            for root in sorted(paired[seed][0])
        }
        if reference_identity is None:
            reference_identity = current_identity
        elif current_identity != reference_identity:
            raise StudyEvidenceError(
                "physical-root identities differ across study seeds"
            )

    seed_metrics: dict[str, Any] = {}
    multi_roots: set[str] | None = None
    single_roots: set[str] | None = None
    root_deltas_by_seed: dict[int, dict[str, float]] = {}
    full_multi_rates: list[float] = []
    bc0_multi_rates: list[float] = []
    full_single_rates: list[float] = []
    bc0_single_rates: list[float] = []
    expected_multi_families: set[str] | None = None
    family_seed_rates: dict[str, dict[str, list[float]]] = {}
    for seed in seeds:
        bc0_records, full_records = paired[seed]
        current_multi = {
            root for root, row in bc0_records.items() if int(row["cardinality"]) >= 2
        }
        current_single = {
            root for root, row in bc0_records.items() if int(row["cardinality"]) == 1
        }
        if not current_multi or not current_single:
            raise StudyEvidenceError(
                f"seed {seed} lacks multi-error or single-error paired roots"
            )
        if multi_roots is None:
            multi_roots = current_multi
            single_roots = current_single
        elif current_multi != multi_roots or current_single != single_roots:
            raise StudyEvidenceError(
                "multi/single physical-root strata differ across seeds"
            )
        bc0_multi = statistics.fmean(
            float(bc0_records[root]["safe_recovery"]) for root in sorted(current_multi)
        )
        full_multi = statistics.fmean(
            float(full_records[root]["safe_recovery"]) for root in sorted(current_multi)
        )
        bc0_single = statistics.fmean(
            float(bc0_records[root]["safe_recovery"]) for root in sorted(current_single)
        )
        full_single = statistics.fmean(
            float(full_records[root]["safe_recovery"])
            for root in sorted(current_single)
        )
        root_deltas_by_seed[seed] = {
            root: float(full_records[root]["safe_recovery"])
            - float(bc0_records[root]["safe_recovery"])
            for root in sorted(current_multi)
        }
        current_families = {str(bc0_records[root]["family"]) for root in current_multi}
        if expected_multi_families is None:
            expected_multi_families = current_families
        elif current_families != expected_multi_families:
            raise StudyEvidenceError(
                "multi-error family coverage differs across study seeds"
            )
        per_family: dict[str, Any] = {}
        for family_name in sorted(current_families):
            family_roots = sorted(
                root
                for root in current_multi
                if str(bc0_records[root]["family"]) == family_name
            )
            bc0_family_rate = statistics.fmean(
                float(bc0_records[root]["safe_recovery"]) for root in family_roots
            )
            full_family_rate = statistics.fmean(
                float(full_records[root]["safe_recovery"]) for root in family_roots
            )
            family_seed_rates.setdefault(family_name, {"bc0": [], "full": []})[
                "bc0"
            ].append(bc0_family_rate)
            family_seed_rates[family_name]["full"].append(full_family_rate)
            per_family[family_name] = {
                "physical_roots": len(family_roots),
                "bc0_rate": bc0_family_rate,
                "full_rate": full_family_rate,
                "delta": full_family_rate - bc0_family_rate,
            }
        full_multi_rates.append(full_multi)
        bc0_multi_rates.append(bc0_multi)
        full_single_rates.append(full_single)
        bc0_single_rates.append(bc0_single)
        seed_metrics[str(seed)] = {
            "multi_error_physical_roots": len(current_multi),
            "bc0_multi_error_safe_recovery_rate": bc0_multi,
            "full_multi_error_safe_recovery_rate": full_multi,
            "multi_error_delta": full_multi - bc0_multi,
            "single_error_physical_roots": len(current_single),
            "bc0_single_error_safe_recovery_rate": bc0_single,
            "full_single_error_safe_recovery_rate": full_single,
            "single_error_delta": full_single - bc0_single,
            "multi_error_families": per_family,
        }

    assert multi_roots is not None
    assert single_roots is not None
    root_mean_deltas = {
        root: statistics.fmean(root_deltas_by_seed[seed][root] for seed in seeds)
        for root in sorted(multi_roots)
    }
    bootstrap = _paired_physical_root_bootstrap(
        root_mean_deltas,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    mean_full_multi = statistics.fmean(full_multi_rates)
    mean_bc0_multi = statistics.fmean(bc0_multi_rates)
    mean_delta = mean_full_multi - mean_bc0_multi
    mean_full_single = statistics.fmean(full_single_rates)
    mean_bc0_single = statistics.fmean(bc0_single_rates)
    single_degradation = mean_bc0_single - mean_full_single
    seed_spread = max(full_multi_rates) - min(full_multi_rates)
    family_comparison = {
        family_name: {
            "mean_bc0_rate": statistics.fmean(values["bc0"]),
            "mean_full_rate": statistics.fmean(values["full"]),
            "mean_delta": statistics.fmean(values["full"])
            - statistics.fmean(values["bc0"]),
            "regression": max(
                0.0,
                statistics.fmean(values["bc0"]) - statistics.fmean(values["full"]),
            ),
        }
        for family_name, values in sorted(family_seed_rates.items())
    }
    maximum_family_regression = max(
        (row["regression"] for row in family_comparison.values()),
        default=math.inf,
    )

    quarantine_count: int | None = None
    quarantine_candidate_rows: int | None = None
    if production_d1_quarantine_evidence is not None:
        quarantine_evidence = _mapping(
            production_d1_quarantine_evidence,
            field="production_d1_quarantine_evidence",
        )
        if (
            quarantine_evidence.get("contract")
            != "production_d1_quarantine_checkpoint_matrix_v1"
            or quarantine_evidence.get("evidence_available") is not True
            or quarantine_evidence.get("passed") is not True
            or quarantine_evidence.get("audit_report_name")
            != PRODUCTION_D1_QUARANTINE_AUDIT_REPORT_NAME
            or _SHA256.fullmatch(
                str(quarantine_evidence.get("audit_report_sha256") or "")
            )
            is None
        ):
            raise StudyEvidenceError(
                "production-D1 quarantine checkpoint evidence is not authoritative"
            )
        quarantine_count = _nonnegative_integer(
            quarantine_evidence.get("quarantined_rows"),
            field="production_d1_quarantine_evidence.quarantined_rows",
        )
        quarantine_candidate_rows = _nonnegative_integer(
            quarantine_evidence.get("candidate_rows"),
            field="production_d1_quarantine_evidence.candidate_rows",
        )

    def pooled_optional_safety(count_name: str, availability_name: str) -> int | None:
        safety_rows = [normalized_full[seed]["metrics"]["safety"] for seed in seeds]
        if not all(
            row.get(availability_name) is True and row.get(count_name) is not None
            for row in safety_rows
        ):
            return None
        return sum(int(row[count_name]) for row in safety_rows)

    full_safety = {
        "false_commit_count": sum(
            int(normalized_full[seed]["metrics"]["safety"]["false_commit_count"])
            for seed in seeds
        ),
        "false_finalization_count": sum(
            int(normalized_full[seed]["metrics"]["safety"]["false_finalization_count"])
            for seed in seeds
        ),
        "false_rollback_count": sum(
            int(normalized_full[seed]["metrics"]["safety"]["false_rollback_count"])
            for seed in seeds
        ),
        "healthy_component_corruption_episodes": sum(
            int(
                normalized_full[seed]["metrics"]["safety"][
                    "healthy_component_corruption_episodes"
                ]
            )
            for seed in seeds
        ),
        "healthy_target_corruption_episodes": sum(
            int(
                normalized_full[seed]["metrics"]["safety"][
                    "healthy_target_corruption_episodes"
                ]
            )
            for seed in seeds
        ),
        "unknown_healthy_preservation_episodes": sum(
            int(
                normalized_full[seed]["metrics"]["safety"][
                    "unknown_healthy_preservation_episodes"
                ]
            )
            for seed in seeds
        ),
        "evaluator_error_episodes": sum(
            int(normalized_full[seed]["metrics"]["safety"]["evaluator_error_episodes"])
            for seed in seeds
        ),
        # This study-wide source fact is never inferred from evaluation
        # episodes.  It is populated only by the receipt matrix assembled from
        # source-gate-authenticated Round-1 evidence.
        "teacher_targets_quarantined_in_production_d1": quarantine_count,
        "finalize_with_unresolved_private_fault_count": sum(
            int(normalized_full[seed]["metrics"]["safety"]["false_finalization_count"])
            for seed in seeds
        ),
        "physically_unsafe_commit_count": pooled_optional_safety(
            "physically_unsafe_commit_count",
            "physically_unsafe_commit_evidence_available",
        ),
        "truth_safe_accepted_candidate_rollback_count": sum(
            int(normalized_full[seed]["metrics"]["safety"]["false_rollback_count"])
            for seed in seeds
        ),
        "hidden_truth_leakage_count": pooled_optional_safety(
            "hidden_truth_leakage_count",
            "hidden_truth_leakage_evidence_available",
        ),
    }
    full_safety_denominators = {
        "physically_unsafe_commit_count": (
            sum(
                int(
                    normalized_full[seed]["metrics"]["safety"][
                        "successful_commit_count"
                    ]
                )
                for seed in seeds
            )
            if full_safety["physically_unsafe_commit_count"] is not None
            else None
        ),
        "hidden_truth_leakage_count": (
            sum(
                int(
                    normalized_full[seed]["metrics"]["safety"][
                        "policy_observation_count"
                    ]
                )
                for seed in seeds
            )
            if full_safety["hidden_truth_leakage_count"] is not None
            else None
        ),
    }
    pooled_bc0_records = [
        record for seed in seeds for record in normalized_bc0[seed]["root_records"]
    ]
    pooled_full_records = [
        record for seed in seeds for record in normalized_full[seed]["root_records"]
    ]
    pooled_bc0_metrics = _aggregate_records(pooled_bc0_records)
    pooled_full_metrics = _aggregate_records(pooled_full_records)
    bc0_efficiency = pooled_bc0_metrics["efficiency"]
    full_efficiency = pooled_full_metrics["efficiency"]
    bc0_invalid_redundant_mean = (
        float(bc0_efficiency["invalid_action_count"])
        + float(bc0_efficiency["redundant_action_count"])
    ) / len(pooled_bc0_records)
    full_invalid_redundant_mean = (
        float(full_efficiency["invalid_action_count"])
        + float(full_efficiency["redundant_action_count"])
    ) / len(pooled_full_records)
    bc0_successful_multi_median = bc0_efficiency[
        "successful_multi_error_median_tool_calls"
    ]
    full_successful_multi_median = full_efficiency[
        "successful_multi_error_median_tool_calls"
    ]

    def policy_number(name: str) -> float:
        value = comparison_policy.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StudyEvidenceError(
                f"comparison_policy.{name} must be a finite number"
            )
        parsed = float(value)
        if not math.isfinite(parsed):
            raise StudyEvidenceError(
                f"comparison_policy.{name} must be a finite number"
            )
        return parsed

    def policy_integer(name: str) -> int:
        return _nonnegative_integer(
            comparison_policy.get(name), field=f"comparison_policy.{name}"
        )

    def policy_true(name: str) -> bool:
        if comparison_policy.get(name) is not True:
            raise StudyEvidenceError(f"comparison_policy.{name} must be true")
        return True

    minimum_seed_count = policy_integer("minimum_training_seed_count")
    minimum_full_recovery = policy_number("minimum_complete_variant_recovery_rate")
    minimum_delta = policy_number("minimum_mean_primary_improvement")
    maximum_seed_spread = policy_number("maximum_complete_variant_seed_spread")
    maximum_single_degradation = policy_number("maximum_single_error_degradation")
    maximum_family_regression_allowed = policy_number(
        "maximum_material_family_regression"
    )
    policy_true("require_every_seed_non_degrading_vs_bc0")
    policy_true("require_bootstrap_95ci_excludes_zero")
    policy_true("require_no_material_family_regression")
    if comparison_policy.get("unsupported_or_empty_family_policy") != "fail_closed":
        raise StudyEvidenceError(
            "comparison_policy.unsupported_or_empty_family_policy must be fail_closed"
        )
    safety_policy_names = {
        "false_commit_count": "maximum_false_commit_count",
        "false_finalization_count": "maximum_false_finalization_count",
        "false_rollback_count": "maximum_false_rollback_count",
        "healthy_component_corruption_episodes": (
            "maximum_healthy_component_corruption_episodes"
        ),
        "healthy_target_corruption_episodes": (
            "maximum_healthy_target_corruption_episodes"
        ),
        "unknown_healthy_preservation_episodes": (
            "maximum_unknown_healthy_preservation_episodes"
        ),
        "evaluator_error_episodes": "maximum_evaluator_error_episodes",
        "teacher_targets_quarantined_in_production_d1": (
            "maximum_teacher_targets_quarantined_in_production_d1"
        ),
        "finalize_with_unresolved_private_fault_count": (
            "maximum_finalize_with_unresolved_private_fault_count"
        ),
        "physically_unsafe_commit_count": ("maximum_physically_unsafe_commit_count"),
        "truth_safe_accepted_candidate_rollback_count": (
            "maximum_truth_safe_accepted_candidate_rollback_count"
        ),
        "hidden_truth_leakage_count": "maximum_hidden_truth_leakage_count",
    }
    safety_rule_results = {
        metric: bool(
            full_safety[metric] is not None
            and int(full_safety[metric]) <= policy_integer(policy_name)
        )
        for metric, policy_name in safety_policy_names.items()
    }
    zero_safety_violations = all(safety_rule_results.values())
    per_seed_nondegrading = all(
        seed_metrics[str(seed)]["multi_error_delta"] >= 0.0 for seed in seeds
    )
    rules = {
        "minimum_training_seed_count": len(seeds) >= minimum_seed_count,
        "minimum_complete_variant_recovery_rate": (
            mean_full_multi >= minimum_full_recovery
        ),
        "minimum_mean_primary_improvement": mean_delta >= minimum_delta,
        "paired_bootstrap_ci_lower_above_zero": bootstrap["ci_lower"] > 0.0,
        "zero_safety_violations": zero_safety_violations,
        "maximum_single_error_degradation": (
            single_degradation <= maximum_single_degradation
        ),
        "every_seed_nondegrading": per_seed_nondegrading,
        "maximum_complete_variant_seed_spread": (seed_spread <= maximum_seed_spread),
        "maximum_material_family_regression": bool(
            family_comparison
            and maximum_family_regression <= maximum_family_regression_allowed
        ),
        **{f"safety_{name}": passed for name, passed in safety_rule_results.items()},
    }
    full_recovery_summary = pooled_full_metrics["recovery"]
    comparison_rule_evidence: dict[str, dict[str, Any]] = {
        "minimum_training_seed_count": {
            "observed": len(seeds),
            "numerator": len(seeds),
            "denominator": minimum_seed_count,
            "threshold": {"operator": ">=", "value": minimum_seed_count},
            "evidence_available": True,
            "passed": rules["minimum_training_seed_count"],
        },
        "minimum_complete_variant_recovery_rate": {
            "observed": mean_full_multi,
            "numerator": full_recovery_summary["multi_error_safe_recovery_episodes"],
            "denominator": full_recovery_summary["multi_error_episode_count"],
            "threshold": {"operator": ">=", "value": minimum_full_recovery},
            "evidence_available": True,
            "passed": rules["minimum_complete_variant_recovery_rate"],
        },
        "minimum_mean_primary_improvement": {
            "observed": mean_delta,
            "numerator": None,
            "denominator": None,
            "threshold": {"operator": ">=", "value": minimum_delta},
            "evidence_available": True,
            "passed": rules["minimum_mean_primary_improvement"],
        },
        "paired_bootstrap_ci_lower_above_zero": {
            "observed": bootstrap["ci_lower"],
            "numerator": None,
            "denominator": bootstrap["physical_root_count"],
            "threshold": {"operator": ">", "value": 0.0},
            "evidence_available": True,
            "passed": rules["paired_bootstrap_ci_lower_above_zero"],
        },
        "zero_safety_violations": {
            "observed": (
                0
                if zero_safety_violations
                else sum(
                    int(value) for value in full_safety.values() if value is not None
                )
            ),
            "numerator": None,
            "denominator": len(pooled_full_records),
            "threshold": {"operator": "==", "value": 0},
            "evidence_available": all(
                value is not None for value in full_safety.values()
            ),
            "passed": rules["zero_safety_violations"],
        },
        "maximum_single_error_degradation": {
            "observed": single_degradation,
            "numerator": None,
            "denominator": len(single_roots) * len(seeds),
            "threshold": {
                "operator": "<=",
                "value": maximum_single_degradation,
            },
            "evidence_available": True,
            "passed": rules["maximum_single_error_degradation"],
        },
        "every_seed_nondegrading": {
            "observed": {
                str(seed): seed_metrics[str(seed)]["multi_error_delta"]
                for seed in seeds
            },
            "numerator": sum(
                seed_metrics[str(seed)]["multi_error_delta"] >= 0.0 for seed in seeds
            ),
            "denominator": len(seeds),
            "threshold": {"operator": "all", "value": ">=0"},
            "evidence_available": True,
            "passed": rules["every_seed_nondegrading"],
        },
        "maximum_complete_variant_seed_spread": {
            "observed": seed_spread,
            "numerator": None,
            "denominator": len(seeds),
            "threshold": {"operator": "<=", "value": maximum_seed_spread},
            "evidence_available": True,
            "passed": rules["maximum_complete_variant_seed_spread"],
        },
        "maximum_material_family_regression": {
            "observed": maximum_family_regression,
            "numerator": None,
            "denominator": len(family_comparison),
            "threshold": {
                "operator": "<=",
                "value": maximum_family_regression_allowed,
            },
            "evidence_available": bool(family_comparison),
            "passed": rules["maximum_material_family_regression"],
        },
    }
    for metric, policy_name in safety_policy_names.items():
        observed = full_safety[metric]
        if metric == "teacher_targets_quarantined_in_production_d1":
            denominator = quarantine_candidate_rows
        elif metric in full_safety_denominators:
            denominator = full_safety_denominators[metric]
        else:
            denominator = len(pooled_full_records)
        comparison_rule_evidence[f"safety_{metric}"] = {
            "observed": observed,
            "numerator": observed,
            "denominator": denominator,
            "threshold": {
                "operator": "<=",
                "value": policy_integer(policy_name),
            },
            "evidence_available": observed is not None,
            "evidence_failure": (
                None
                if observed is not None
                else "authoritative_study_safety_evidence_unavailable"
            ),
            "passed": rules[f"safety_{metric}"],
        }

    diagnostic = pooled_full_metrics["diagnostic_targets"]
    physical = pooled_full_metrics["physical_recovery"]
    recovery_accuracy = pooled_full_metrics["recovery_action_accuracy"]
    state_bound = full_efficiency["valid_state_bound_proxy"]
    objective_observed: dict[str, dict[str, Any]] = {
        "diagnostic": {
            "multi_error_exact_fault_set_identification_rate": diagnostic[
                "multi_error_exact_standard_fault_set_rate"
            ],
            "measurement_parameter_family_macro_f1": diagnostic[
                "measurement_parameter_family_macro_f1"
            ],
            "multi_error_correct_error_cardinality_rate": diagnostic[
                "multi_error_correct_standard_target_cardinality_rate"
            ],
            "mixed_measurement_parameter_sequential_resolution_rate": diagnostic[
                "mixed_measurement_parameter_sequential"
            ]["success_rate"],
        },
        "physical": {
            "physically_valid_recovery_among_resolved_rate": physical[
                "physically_valid_among_resolved_rate"
            ],
            "healthy_measurement_preservation_rate": physical[
                "healthy_measurement_preservation_rate"
            ],
            "healthy_branch_parameter_preservation_rate": physical[
                "healthy_branch_parameter_preservation_rate"
            ],
            "final_residual_chi_square_acceptance_rate": physical[
                "final_residual_chi_square_acceptance_rate"
            ],
            "post_commit_powerflow_topology_feasibility_rate": physical[
                "post_commit_power_flow_or_topology_feasibility_rate"
            ],
        },
        "recovery_action_accuracy": {
            name: recovery_accuracy[name]["rate"] for name in recovery_accuracy
        },
        "action_quality_efficiency": {
            "schema_valid_tool_call_rate": full_efficiency[
                "schema_valid_tool_call_rate"
            ],
            "state_bound_action_rate": state_bound["valid_rate"],
            "horizon_without_disposition_rate": full_efficiency[
                "horizon_without_disposition_rate"
            ],
            "repeated_action_loop_rate": full_efficiency["loop_rate"],
            "median_tool_calls_successful_multi_error": (full_successful_multi_median),
            "mean_invalid_redundant_actions": (full_invalid_redundant_mean),
            "correct_operator_handoff_rate": pooled_full_metrics["operator_handoff"][
                "correct_handoff_rate"
            ],
        },
    }
    objective_counts: dict[str, dict[str, tuple[int | None, int | None]]] = {
        "diagnostic": {
            "multi_error_exact_fault_set_identification_rate": (
                diagnostic["multi_error_exact_standard_fault_set_episodes"],
                diagnostic["multi_error_standard_target_applicable_episodes"],
            ),
            "measurement_parameter_family_macro_f1": (None, None),
            "multi_error_correct_error_cardinality_rate": (
                diagnostic["multi_error_correct_standard_target_cardinality_episodes"],
                diagnostic["multi_error_standard_target_applicable_episodes"],
            ),
            "mixed_measurement_parameter_sequential_resolution_rate": (
                diagnostic["mixed_measurement_parameter_sequential"][
                    "successful_episodes"
                ],
                diagnostic["mixed_measurement_parameter_sequential"][
                    "eligible_episodes"
                ],
            ),
        },
        "physical": {
            "physically_valid_recovery_among_resolved_rate": (
                physical["physically_valid_resolved_episodes"],
                physical["resolved_episodes"],
            ),
            "healthy_measurement_preservation_rate": (
                physical["healthy_measurement_preserved_episodes"],
                physical["healthy_measurement_evaluable_episodes"],
            ),
            "healthy_branch_parameter_preservation_rate": (
                physical["healthy_branch_parameter_preserved_episodes"],
                physical["healthy_branch_parameter_evaluable_episodes"],
            ),
            "final_residual_chi_square_acceptance_rate": (
                (
                    physical["final_residual_chi_square_accepted_episodes"],
                    physical["final_residual_chi_square_applicable_episodes"],
                )
                if physical["final_residual_chi_square_acceptance_rate"] is not None
                else (None, None)
            ),
            "post_commit_powerflow_topology_feasibility_rate": (
                (
                    physical["post_commit_feasible_commit_count"],
                    physical["successful_commit_count"],
                )
                if physical["post_commit_power_flow_or_topology_feasibility_rate"]
                is not None
                else (None, None)
            ),
        },
        "recovery_action_accuracy": {
            name: (
                (
                    recovery_accuracy[name]["correct_actions"],
                    recovery_accuracy[name]["opportunities"],
                )
                if recovery_accuracy[name]["rate"] is not None
                else (None, None)
            )
            for name in recovery_accuracy
        },
        "action_quality_efficiency": {
            "schema_valid_tool_call_rate": (
                full_efficiency["schema_valid_tool_calls"],
                full_efficiency["tool_calls"],
            ),
            "state_bound_action_rate": (
                state_bound["valid_actions"],
                state_bound["evaluable_actions"],
            ),
            "horizon_without_disposition_rate": (
                full_efficiency["horizon_without_disposition_episodes"],
                len(pooled_full_records),
            ),
            "repeated_action_loop_rate": (
                full_efficiency["loop_episodes"],
                len(pooled_full_records),
            ),
            "median_tool_calls_successful_multi_error": (None, None),
            "mean_invalid_redundant_actions": (None, None),
            "correct_operator_handoff_rate": (
                (
                    pooled_full_metrics["operator_handoff"]["correct_handoffs"],
                    pooled_full_metrics["operator_handoff"][
                        "autonomous_exhaustion_opportunities"
                    ],
                )
                if pooled_full_metrics["operator_handoff"]["correct_handoff_rate"]
                is not None
                else (None, None)
            ),
        },
    }

    def threshold_spec(section: str, name: str) -> Mapping[str, Any]:
        section_policy = _mapping(
            objective_thresholds.get(section),
            field=f"objective_thresholds.{section}",
        )
        return _mapping(
            section_policy.get(name),
            field=f"objective_thresholds.{section}.{name}",
        )

    def compare_threshold(
        observed: Any,
        spec: Mapping[str, Any],
        *,
        paired_reference: float | int | None = None,
    ) -> tuple[bool, str | None]:
        if observed is None:
            return False, "authoritative_episode_evidence_unavailable"
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise StudyEvidenceError(
                "objective observed metric must be numeric or null"
            )
        value = float(observed)
        if not math.isfinite(value):
            raise StudyEvidenceError("objective observed metric must be finite")
        operator = spec.get("operator")
        if operator == "<=paired_bc0":
            if paired_reference is None:
                return False, "paired_bc0_reference_unavailable"
            return value <= float(paired_reference), None
        if operator == "<=paired_bc0_ratio":
            ratio = spec.get("value")
            if (
                paired_reference is None
                or isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
            ):
                return False, "paired_bc0_ratio_reference_unavailable"
            return value <= float(paired_reference) * float(ratio), None
        threshold = spec.get("value")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise StudyEvidenceError("objective threshold value is invalid")
        target = float(threshold)
        if operator == ">=":
            return value >= target, None
        if operator == "==":
            return math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12), None
        if operator == "<":
            return value < target, None
        raise StudyEvidenceError(f"unsupported objective operator {operator!r}")

    if objective_thresholds.get("evidence_policy") != (
        "required_fail_closed_if_unavailable"
    ):
        raise StudyEvidenceError(
            "objective_thresholds.evidence_policy must fail closed"
        )
    objective_rules: dict[str, dict[str, Any]] = {}
    for section in ("diagnostic", "physical"):
        for name, observed in objective_observed[section].items():
            spec = threshold_spec(section, name)
            passed, evidence_failure = compare_threshold(observed, spec)
            objective_rules[f"{section}.{name}"] = {
                "observed": observed,
                "numerator": objective_counts[section][name][0],
                "denominator": objective_counts[section][name][1],
                "threshold": copy.deepcopy(dict(spec)),
                "evidence_available": observed is not None,
                "evidence_failure": evidence_failure,
                "passed": passed,
            }
    if not isinstance(include_recovery_action_objectives, bool):
        raise TypeError("include_recovery_action_objectives must be bool")
    if include_recovery_action_objectives:
        recovery_policy = _mapping(
            objective_thresholds.get("recovery_action_accuracy"),
            field="objective_thresholds.recovery_action_accuracy",
        )
        for name, observed in objective_observed[
            "recovery_action_accuracy"
        ].items():
            threshold = recovery_policy.get(name)
            if isinstance(threshold, bool) or not isinstance(
                threshold, (int, float)
            ):
                raise StudyEvidenceError(
                    f"objective recovery threshold {name!r} is invalid"
                )
            spec = {"operator": ">=", "value": float(threshold)}
            passed, evidence_failure = compare_threshold(observed, spec)
            objective_rules[f"recovery_action_accuracy.{name}"] = {
                "observed": observed,
                "numerator": objective_counts["recovery_action_accuracy"][name][0],
                "denominator": objective_counts["recovery_action_accuracy"][name][1],
                "threshold": spec,
                "evidence_available": observed is not None,
                "evidence_failure": evidence_failure,
                "passed": passed,
            }
    action_references = {
        "median_tool_calls_successful_multi_error": (bc0_successful_multi_median),
        "mean_invalid_redundant_actions": bc0_invalid_redundant_mean,
    }
    for name, observed in objective_observed["action_quality_efficiency"].items():
        spec = threshold_spec("action_quality_efficiency", name)
        passed, evidence_failure = compare_threshold(
            observed,
            spec,
            paired_reference=action_references.get(name),
        )
        objective_rules[f"action_quality_efficiency.{name}"] = {
            "observed": observed,
            "numerator": objective_counts["action_quality_efficiency"][name][0],
            "denominator": objective_counts["action_quality_efficiency"][name][1],
            "paired_bc0_reference": action_references.get(name),
            "threshold": copy.deepcopy(dict(spec)),
            "evidence_available": observed is not None,
            "evidence_failure": evidence_failure,
            "passed": passed,
        }

    comparison_failures = [name for name, passed in rules.items() if not passed]
    objective_failures = [
        name for name, result in objective_rules.items() if not result["passed"]
    ]
    failures = [
        *(f"comparison_policy.{name}" for name in comparison_failures),
        *(f"objective_thresholds.{name}" for name in objective_failures),
    ]
    primary_passed = not comparison_failures
    objective_passed = not objective_failures
    return {
        "comparison": "natural_dagger_probes_minus_bc0",
        "paired_by": "physical_root",
        "study_seeds": seeds,
        "seed_count": len(seeds),
        "multi_error_physical_root_count": len(multi_roots),
        "single_error_physical_root_count": len(single_roots),
        "seed_metrics": seed_metrics,
        "mean_bc0_multi_error_safe_recovery_rate": mean_bc0_multi,
        "mean_full_multi_error_safe_recovery_rate": mean_full_multi,
        "mean_multi_error_delta": mean_delta,
        "mean_bc0_single_error_safe_recovery_rate": mean_bc0_single,
        "mean_full_single_error_safe_recovery_rate": mean_full_single,
        "single_error_degradation": single_degradation,
        "full_multi_error_seed_spread": seed_spread,
        "multi_error_family_comparison": family_comparison,
        "maximum_multi_error_family_regression": maximum_family_regression,
        "full_safety": full_safety,
        "production_d1_quarantine_evidence": (
            copy.deepcopy(dict(production_d1_quarantine_evidence))
            if production_d1_quarantine_evidence is not None
            else None
        ),
        "pooled_bc0_metrics": pooled_bc0_metrics,
        "pooled_full_metrics": pooled_full_metrics,
        "paired_efficiency": {
            "bc0_successful_multi_error_median_tool_calls": (
                bc0_successful_multi_median
            ),
            "full_successful_multi_error_median_tool_calls": (
                full_successful_multi_median
            ),
            "bc0_mean_invalid_redundant_actions": (bc0_invalid_redundant_mean),
            "full_mean_invalid_redundant_actions": (full_invalid_redundant_mean),
        },
        "bootstrap_95_percent_ci": bootstrap,
        "decision_rule_pass_flags": rules,
        "decision_rules": comparison_rule_evidence,
        "primary_decision_passed": primary_passed,
        "objective_observed": objective_observed,
        "objective_decision_rules": objective_rules,
        "objective_decision_passed": objective_passed,
        "failures": failures,
        "passed": bool(primary_passed and objective_passed),
    }


def _load_study_manifest(
    path: str | os.PathLike[str] | None,
    *,
    seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = Path(path).expanduser() if path is not None else DEFAULT_STUDY_MANIFEST
    resolved = selected.resolve(strict=True)
    try:
        loaded = load_study_manifest(resolved)
    except StudyManifestError as exc:
        raise StudyEvidenceError(f"versioned study manifest failed: {exc}") from exc
    manifest_seeds = _list(
        loaded.get("training_seeds"), field="study_manifest.training_seeds"
    )
    normalized_manifest_seeds = [
        _nonnegative_integer(item, field=f"study_manifest.training_seeds[{index}]")
        for index, item in enumerate(manifest_seeds)
    ]
    if len(normalized_manifest_seeds) != len(set(normalized_manifest_seeds)):
        raise StudyEvidenceError("study manifest training seeds contain duplicates")
    if sorted(normalized_manifest_seeds) != sorted(seeds):
        raise StudyEvidenceError(
            "CLI study seeds do not match the versioned manifest exactly"
        )
    variants = loaded.get("variants")
    if isinstance(variants, Mapping):
        variant_ids = set(str(key) for key in variants)
    elif isinstance(variants, list):
        variant_ids = {
            str(_mapping(row, field="study_manifest.variants[]").get("variant_id"))
            for row in variants
        }
    else:
        raise StudyEvidenceError("study_manifest.variants has an invalid schema")
    required_variants = set(REQUIRED_VARIANT_IDS)
    if variant_ids != required_variants:
        raise StudyEvidenceError(
            "study manifest variant set differs from the four-variant contract"
        )
    descriptor = {
        "path": str(resolved),
        "sha256": loaded["manifest_sha256"],
        "schema_version": 1,
        "study_id": loaded["study_id"],
        "training_seeds": sorted(normalized_manifest_seeds),
    }
    return loaded, descriptor


def _require_exact_keys(
    value: Mapping[Any, Any],
    expected: Sequence[Any],
    *,
    field: str,
) -> None:
    observed = set(value)
    required = set(expected)
    if observed != required:
        missing = sorted(str(item) for item in required - observed)
        extra = sorted(str(item) for item in observed - required)
        raise StudyEvidenceError(
            f"{field} keys differ from the preregistration; "
            f"missing={missing}, extra={extra}"
        )


def _scope_root_identity(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = _list(run.get("root_records"), field="run.root_records")
    stress = run.get("evaluation_scope") == RECOVERY_STRESS_SCOPE
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(records):
        row = _mapping(raw, field=f"run.root_records[{index}]")
        key_field = "episode_key" if stress else "physical_root"
        key = _text(
            row.get(key_field),
            field=f"run.root_records[{index}].{key_field}",
        )
        if key in result:
            raise StudyEvidenceError(
                f"run repeats scope identity {key_field}={key!r}"
            )
        result[key] = {
            field: row.get(field) for field in _IDENTITY_FIELDS
        }
    return result


def _validate_scope_matrix(
    *,
    scope: str,
    base_run: Mapping[str, Any],
    trained_runs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    seeds: Sequence[int],
    exact_development_roots: int,
) -> dict[str, Any]:
    """Require one protocol/root binding for every model in one scope."""

    if scope not in EVALUATION_SCOPES:
        raise StudyEvidenceError(f"unknown stability scope {scope!r}")
    expected_role = _SCOPE_ARTIFACT_ROLE[scope]
    reference_suite = base_run.get("input_suite_sha256")
    reference_binding = base_run.get("scope_binding")
    reference_evaluator_seed = base_run.get("evaluator_seed")
    reference_max_steps = base_run.get("max_steps")
    reference_identity = _scope_root_identity(base_run)
    reference_records = _list(
        base_run.get("root_records"), field="base_run.root_records"
    )
    reference_roots = sorted(
        {
            _text(
                _mapping(row, field="base_run.root_records[]").get(
                    "physical_root"
                ),
                field="base_run.root_records[].physical_root",
            )
            for row in reference_records
        }
    )
    if (
        base_run.get("evaluation_scope") != scope
        or base_run.get("artifact_role") != expected_role
    ):
        raise StudyEvidenceError(f"base run is not bound to {scope}")
    expected_count = len(reference_roots)
    expected_episode_count = len(reference_identity)
    recomputed_root_set_sha256 = stable_json_sha256(sorted(reference_roots))
    if scope == "development_holdout" and expected_count != exact_development_roots:
        raise StudyEvidenceError(
            "base development evaluation does not contain the exact "
            f"{exact_development_roots}-root holdout"
        )
    if scope == "development_holdout":
        binding = _mapping(reference_binding, field="base_run.scope_binding")
        if binding.get("development_holdout_physical_roots") != expected_count:
            raise StudyEvidenceError(
                "development holdout root count binding differs from episode evidence"
            )
        if (
            binding.get("development_holdout_root_set_sha256")
            != recomputed_root_set_sha256
        ):
            raise StudyEvidenceError(
                "development holdout root-set binding differs from episode evidence"
            )
    elif scope == RECOVERY_STRESS_SCOPE:
        binding = _mapping(reference_binding, field="base_run.scope_binding")
        if binding.get("recovery_stress_physical_roots") != expected_count:
            raise StudyEvidenceError(
                "recovery-stress root count binding differs from episode evidence"
            )
        if binding.get("recovery_stress_episode_count") != expected_episode_count:
            raise StudyEvidenceError(
                "recovery-stress episode count binding differs from episode evidence"
            )
        if binding.get("recovery_stress_root_set_sha256") != (
            recomputed_root_set_sha256
        ):
            raise StudyEvidenceError(
                "recovery-stress root-set binding differs from episode evidence"
            )
    artifact_hashes = {str(base_run.get("artifact_content_sha256"))}
    run_count = 1
    for variant_id in TRAINED_VARIANT_IDS:
        for seed in seeds:
            run = trained_runs[variant_id][seed]
            run_count += 1
            if run.get("variant_id") != variant_id or run.get("study_seed") != seed:
                raise StudyEvidenceError(
                    f"{scope} {variant_id} seed {seed} run identity differs"
                )
            if (
                run.get("evaluation_scope") != scope
                or run.get("artifact_role") != expected_role
            ):
                raise StudyEvidenceError(
                    f"{variant_id} seed {seed} is not bound to {scope}"
                )
            if run.get("input_suite_sha256") != reference_suite:
                raise StudyEvidenceError(
                    f"{scope} suite differs for {variant_id} seed {seed}"
                )
            if run.get("scope_binding") != reference_binding:
                raise StudyEvidenceError(
                    f"{scope} provenance/root binding differs for "
                    f"{variant_id} seed {seed}"
                )
            if (
                run.get("evaluator_seed") != reference_evaluator_seed
                or run.get("max_steps") != reference_max_steps
            ):
                raise StudyEvidenceError(
                    f"{scope} evaluator protocol differs for {variant_id} seed {seed}"
                )
            if _scope_root_identity(run) != reference_identity:
                raise StudyEvidenceError(
                    f"{scope} episode/root identities differ for "
                    f"{variant_id} seed {seed}"
                )
            artifact_hash = str(run.get("artifact_content_sha256"))
            if artifact_hash in artifact_hashes:
                raise StudyEvidenceError(
                    f"{scope} reuses an evaluation artifact across model variants"
                )
            artifact_hashes.add(artifact_hash)
    return {
        "scope": scope,
        "artifact_role": expected_role,
        "evidence_available": True,
        "evaluation_artifact_count": run_count,
        "physical_root_count": expected_count,
        "episode_count": expected_episode_count,
        "physical_root_set_sha256": recomputed_root_set_sha256,
        "input_suite_sha256": reference_suite,
        "scope_binding": copy.deepcopy(reference_binding),
        "same_bound_roots_and_protocol": True,
        "passed": True,
    }


def _validate_checkpoint_matrix(
    *,
    checkpoints: Mapping[str, Mapping[int, Mapping[str, Any]]],
    trained_runs_by_scope: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Bind both evaluations to the exact same-seed checkpoint receipt/tree."""

    rows: dict[str, Any] = {}
    for variant_id in TRAINED_VARIANT_IDS:
        for seed in seeds:
            checkpoint = checkpoints[variant_id][seed]
            tree = checkpoint["adapter_tree_sha256"]
            receipt_id = checkpoint["checkpoint_receipt_id"]
            scope_rows: dict[str, Any] = {}
            for scope in EVALUATION_SCOPES:
                run = trained_runs_by_scope[scope][variant_id][seed]
                model_revision = run.get("model_revision")
                observed_tree = run.get("checkpoint_adapter_tree_sha256")
                observed_receipt = run.get("checkpoint_receipt_id")
                if (
                    model_revision != tree
                    or observed_tree != tree
                    or observed_receipt != receipt_id
                ):
                    raise StudyEvidenceError(
                        f"{variant_id} seed {seed} {scope} does not bind its "
                        "checkpoint receipt and adapter tree"
                    )
                scope_rows[scope] = {
                    "model_revision": model_revision,
                    "checkpoint_adapter_tree_sha256": observed_tree,
                    "checkpoint_receipt_id": observed_receipt,
                    "passed": True,
                }
            if variant_id in {"natural_dagger", "natural_dagger_probes"}:
                bc0_checkpoint = checkpoints["bc0"][seed]
                bc0_tree = bc0_checkpoint["adapter_tree_sha256"]
                bc0_receipt_id = bc0_checkpoint["checkpoint_receipt_id"]
                if checkpoint["parent_model_revision"] != bc0_tree:
                    raise StudyEvidenceError(
                        f"{variant_id} seed {seed} did not warm-start from the "
                        "same-seed BC0 checkpoint tree"
                    )
                if checkpoint["parent_checkpoint_receipt_id"] != bc0_receipt_id:
                    raise StudyEvidenceError(
                        f"{variant_id} seed {seed} did not bind the same-seed "
                        "BC0 parent checkpoint receipt ID"
                    )
            rows[f"{variant_id}:{seed}"] = {
                "checkpoint_receipt_id": receipt_id,
                "adapter_tree_sha256": tree,
                "parent_model_revision": checkpoint["parent_model_revision"],
                "parent_checkpoint_receipt_id": checkpoint[
                    "parent_checkpoint_receipt_id"
                ],
                "evaluation_bindings": scope_rows,
                "evidence_available": True,
                "passed": True,
            }
    return {
        "required_checkpoint_count": len(TRAINED_VARIANT_IDS) * len(seeds),
        "validated_checkpoint_count": len(rows),
        "same_seed_parentage_required": True,
        "same_seed_parent_receipt_id_required": True,
        "bindings": rows,
        "evidence_available": True,
        "passed": True,
    }


def _validate_production_d1_quarantine_checkpoint_matrix(
    *,
    checkpoints: Mapping[str, Mapping[int, Mapping[str, Any]]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Require one source-gate-authenticated D1 corpus across all receipts."""

    canonical_na = canonical_production_d1_quarantine_binding("base")
    variant_bindings: dict[str, Any] = {
        "base": {
            "binding": canonical_na,
            "checkpoint_receipt_ids": [],
            "evidence_available": True,
            "passed": True,
        }
    }
    bc0_na = canonical_production_d1_quarantine_binding("bc0")
    bc0_receipts: list[str] = []
    for seed in seeds:
        checkpoint = checkpoints["bc0"][seed]
        if checkpoint.get("production_d1_quarantine_binding") != bc0_na:
            raise StudyEvidenceError(
                f"BC0 seed {seed} production-D1 quarantine binding is not the "
                "canonical not-applicable/null value"
            )
        bc0_receipts.append(str(checkpoint["checkpoint_receipt_id"]))
    variant_bindings["bc0"] = {
        "binding": bc0_na,
        "checkpoint_receipt_ids": bc0_receipts,
        "evidence_available": True,
        "passed": True,
    }

    reference_binding: dict[str, Any] | None = None
    for variant_id in ("natural_dagger", "natural_dagger_probes"):
        receipt_ids: list[str] = []
        for seed in seeds:
            checkpoint = checkpoints[variant_id][seed]
            binding = dict(
                _mapping(
                    checkpoint.get("production_d1_quarantine_binding"),
                    field=(
                        f"checkpoints.{variant_id}[{seed}]."
                        "production_d1_quarantine_binding"
                    ),
                )
            )
            if reference_binding is None:
                reference_binding = copy.deepcopy(binding)
            elif binding != reference_binding:
                raise StudyEvidenceError(
                    "natural/full checkpoint receipts disagree on the exact "
                    "production-D1 quarantine evidence"
                )
            receipt_ids.append(str(checkpoint["checkpoint_receipt_id"]))
        assert reference_binding is not None
        variant_bindings[variant_id] = {
            "binding": copy.deepcopy(reference_binding),
            "checkpoint_receipt_ids": receipt_ids,
            "evidence_available": True,
            "passed": True,
        }

    if reference_binding is None:
        raise StudyEvidenceError(
            "production-D1 quarantine checkpoint evidence is unavailable"
        )
    if (
        reference_binding.get("contract") != PRODUCTION_D1_QUARANTINE_BINDING_CONTRACT
        or reference_binding.get("status") != "applicable"
        or reference_binding.get("quarantined_rows") != 0
    ):
        raise StudyEvidenceError(
            "production-D1 quarantine checkpoint evidence is not a validated zero"
        )
    candidate_rows = _nonnegative_integer(
        reference_binding.get("candidate_rows"),
        field="production_d1_quarantine_binding.candidate_rows",
    )
    quarantined_rows = _nonnegative_integer(
        reference_binding.get("quarantined_rows"),
        field="production_d1_quarantine_binding.quarantined_rows",
    )
    return {
        "contract": "production_d1_quarantine_checkpoint_matrix_v1",
        "counting_unit": "unique_production_d1_corpus",
        "generation_provenance_id": reference_binding["generation_provenance_id"],
        "audit_report_name": reference_binding["audit_report_name"],
        "audit_report_sha256": reference_binding["audit_report_sha256"],
        "candidate_rows": candidate_rows,
        "quarantined_rows": quarantined_rows,
        "numerator": quarantined_rows,
        "denominator": candidate_rows,
        "variant_bindings": variant_bindings,
        "evidence_available": True,
        "passed": True,
    }


def _pooled_run_metrics(runs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    records = [
        record
        for seed in sorted(runs)
        for record in _list(
            runs[seed].get("root_records"),
            field=f"runs[{seed}].root_records",
        )
    ]
    if not records:
        raise StudyEvidenceError("cannot pool an empty run set")
    return _aggregate_records(records)


def _development_stability_decision(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Use preregistered non-regression rules without treating dev as release QA."""

    source_rules = _mapping(
        comparison.get("decision_rules"), field="comparison.decision_rules"
    )
    names = (
        "minimum_training_seed_count",
        "minimum_complete_variant_recovery_rate",
        "maximum_single_error_degradation",
        "every_seed_nondegrading",
        "maximum_complete_variant_seed_spread",
        "maximum_material_family_regression",
    )
    rules = {
        name: copy.deepcopy(
            dict(_mapping(source_rules.get(name), field=f"decision_rules.{name}"))
        )
        for name in names
    }
    failures = [name for name, row in rules.items() if row.get("passed") is not True]
    return {
        "scope": "development_holdout",
        "release_qualification_allowed": False,
        "decision_basis": "preregistered_stability_nonregression_rules",
        "rules": rules,
        "failures": failures,
        "evidence_available": all(
            row.get("evidence_available") is True for row in rules.values()
        ),
        "passed": not failures,
    }


def _recovery_stress_decision(
    *,
    full_runs: Mapping[int, Mapping[str, Any]],
    objective_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the seven recovery-action targets only to guaranteed stress cells."""

    if not full_runs:
        raise StudyEvidenceError("recovery-stress decision has no full-model runs")
    seeds = sorted(full_runs)
    recovery_policy = _mapping(
        objective_thresholds.get("recovery_action_accuracy"),
        field="objective_thresholds.recovery_action_accuracy",
    )
    _require_exact_keys(
        recovery_policy,
        _RECOVERY_ACTION_STRATA,
        field="objective_thresholds.recovery_action_accuracy",
    )
    pooled = _pooled_run_metrics(full_runs)
    pooled_actions = _mapping(
        pooled.get("recovery_action_accuracy"),
        field="pooled.recovery_action_accuracy",
    )
    expected_opportunities = 10 * len(seeds)
    rules: dict[str, dict[str, Any]] = {}
    for name in _RECOVERY_ACTION_STRATA:
        threshold_value = recovery_policy.get(name)
        if isinstance(threshold_value, bool) or not isinstance(
            threshold_value, (int, float)
        ):
            raise StudyEvidenceError(
                f"recovery-stress threshold {name!r} is invalid"
            )
        threshold = float(threshold_value)
        pooled_item = _mapping(
            pooled_actions.get(name),
            field=f"pooled.recovery_action_accuracy.{name}",
        )
        correct = _nonnegative_integer(
            pooled_item.get("correct_actions"),
            field=f"pooled.recovery_action_accuracy.{name}.correct_actions",
        )
        opportunities = _nonnegative_integer(
            pooled_item.get("opportunities"),
            field=f"pooled.recovery_action_accuracy.{name}.opportunities",
        )
        rate = pooled_item.get("rate")
        rate = (
            float(rate)
            if not isinstance(rate, bool) and isinstance(rate, (int, float))
            else None
        )
        per_seed: dict[str, Any] = {}
        per_seed_evidence = True
        for seed in seeds:
            metrics = _mapping(
                full_runs[seed].get("metrics"),
                field=f"full_runs[{seed}].metrics",
            )
            action = _mapping(
                _mapping(
                    metrics.get("recovery_action_accuracy"),
                    field=f"full_runs[{seed}].recovery_action_accuracy",
                ).get(name),
                field=f"full_runs[{seed}].recovery_action_accuracy.{name}",
            )
            seed_correct = _nonnegative_integer(
                action.get("correct_actions"),
                field=f"full_runs[{seed}].{name}.correct_actions",
            )
            seed_opportunities = _nonnegative_integer(
                action.get("opportunities"),
                field=f"full_runs[{seed}].{name}.opportunities",
            )
            seed_rate_value = action.get("rate")
            seed_rate = (
                float(seed_rate_value)
                if not isinstance(seed_rate_value, bool)
                and isinstance(seed_rate_value, (int, float))
                else None
            )
            available = bool(
                seed_opportunities == 10
                and seed_rate is not None
                and action.get("evidence_status") == "available"
            )
            per_seed_evidence = per_seed_evidence and available
            per_seed[str(seed)] = {
                "correct_actions": seed_correct,
                "opportunities": seed_opportunities,
                "rate": seed_rate,
                "evidence_available": available,
                "meets_pooled_target_individually": bool(
                    available and seed_rate is not None and seed_rate >= threshold
                ),
            }
        evidence_available = bool(
            opportunities == expected_opportunities
            and rate is not None
            and pooled_item.get("evidence_status") == "available"
            and per_seed_evidence
        )
        passed = bool(
            evidence_available and rate is not None and rate >= threshold
        )
        rules[f"recovery_action_accuracy.{name}"] = {
            "observed": rate,
            "numerator": correct,
            "denominator": opportunities,
            "expected_denominator": expected_opportunities,
            "per_seed": per_seed,
            "threshold": {"operator": ">=", "value": threshold},
            "evidence_available": evidence_available,
            "passed": passed,
        }

    safety = _mapping(pooled.get("safety"), field="pooled.safety")
    safety_fields = (
        "false_commit_count",
        "false_finalization_count",
        "false_rollback_count",
        "healthy_component_corruption_episodes",
        "healthy_target_corruption_episodes",
        "evaluator_error_episodes",
        "physically_unsafe_commit_count",
        "hidden_truth_leakage_count",
    )
    for name in safety_fields:
        value = safety.get(name)
        available = (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value >= 0
        )
        rules[f"safety.{name}"] = {
            "observed": value,
            "numerator": value if available else None,
            "denominator": len(seeds) * 70,
            "threshold": {"operator": "==", "value": 0},
            "evidence_available": available,
            "passed": bool(available and value == 0),
        }
    failures = [name for name, row in rules.items() if row["passed"] is not True]
    return {
        "scope": RECOVERY_STRESS_SCOPE,
        "evaluation_protocol": "preregistered_recovery_stress_test",
        "study_seeds": seeds,
        "episode_count": len(seeds) * 70,
        "rules": rules,
        "failures": failures,
        "evidence_available": all(
            row["evidence_available"] is True for row in rules.values()
        ),
        "passed": not failures,
    }


def _reported_outcome_rate(metrics: Mapping[str, Any], name: str) -> float | None:
    """Resolve only manifest-preregistered normalized outcome rates."""

    diagnostic = _mapping(
        metrics.get("diagnostic_targets"), field="metrics.diagnostic_targets"
    )
    physical = _mapping(
        metrics.get("physical_recovery"), field="metrics.physical_recovery"
    )
    recovery = _mapping(metrics.get("recovery"), field="metrics.recovery")
    recovery_actions = _mapping(
        metrics.get("recovery_action_accuracy"),
        field="metrics.recovery_action_accuracy",
    )
    aliases: dict[str, Any] = {
        "multi_error_episode_recovery_rate": recovery.get(
            "multi_error_safe_recovery_rate"
        ),
        "diagnostic.multi_error_exact_fault_set_identification_rate": (
            diagnostic.get("multi_error_exact_standard_fault_set_rate")
        ),
        "diagnostic.measurement_parameter_family_macro_f1": diagnostic.get(
            "measurement_parameter_family_macro_f1"
        ),
        "diagnostic.multi_error_correct_error_cardinality_rate": diagnostic.get(
            "multi_error_correct_standard_target_cardinality_rate"
        ),
        "diagnostic.mixed_measurement_parameter_sequential_resolution_rate": (
            _mapping(
                diagnostic.get("mixed_measurement_parameter_sequential"),
                field=(
                    "metrics.diagnostic_targets.mixed_measurement_parameter_sequential"
                ),
            ).get("success_rate")
        ),
        "physical.physically_valid_recovery_among_resolved_rate": physical.get(
            "physically_valid_among_resolved_rate"
        ),
        "physical.healthy_measurement_preservation_rate": physical.get(
            "healthy_measurement_preservation_rate"
        ),
        "physical.healthy_branch_parameter_preservation_rate": physical.get(
            "healthy_branch_parameter_preservation_rate"
        ),
        "physical.final_residual_chi_square_acceptance_rate": physical.get(
            "final_residual_chi_square_acceptance_rate"
        ),
        "physical.post_commit_powerflow_topology_feasibility_rate": physical.get(
            "post_commit_power_flow_or_topology_feasibility_rate"
        ),
    }
    for action_name in (
        "post_failure_no_candidate",
        "unsupported_correction_recovery",
        "premature_commit_recovery",
        "premature_escalation_recovery",
        "rejected_candidate_rollback",
        "safe_continuation_after_partial_success",
        "measurement_parameter_sequential_handoff",
    ):
        action = _mapping(
            recovery_actions.get(action_name),
            field=f"metrics.recovery_action_accuracy.{action_name}",
        )
        aliases[f"recovery_action_accuracy.{action_name}"] = action.get("rate")
    aliases["post_failure_no_candidate_action_accuracy"] = aliases[
        "recovery_action_accuracy.post_failure_no_candidate"
    ]
    aliases["unsupported_correction_recovery_action_accuracy"] = aliases[
        "recovery_action_accuracy.unsupported_correction_recovery"
    ]
    if name not in aliases:
        raise StudyEvidenceError(
            f"probe ablation metric {name!r} has no authoritative registry mapping"
        )
    observed = aliases[name]
    if observed is None:
        return None
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise StudyEvidenceError(f"probe ablation metric {name!r} is not numeric")
    parsed = float(observed)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise StudyEvidenceError(
            f"probe ablation metric {name!r} is outside the unit interval"
        )
    return parsed


def _probe_ablation_decision(
    *,
    scope: str,
    natural_runs: Mapping[int, Mapping[str, Any]],
    full_runs: Mapping[int, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate full-minus-natural probe effects with same-seed pairing."""

    if set(natural_runs) != set(full_runs) or not natural_runs:
        raise StudyEvidenceError(
            f"{scope} probe-ablation seed sets must match and be non-empty"
        )
    seeds = sorted(natural_runs)
    pair_records = (
        _paired_episode_records
        if scope == RECOVERY_STRESS_SCOPE
        else _paired_root_records
    )
    for seed in seeds:
        pair_records(
            seed=seed,
            bc0_run=natural_runs[seed],
            full_run=full_runs[seed],
        )
    if policy.get("pairing") != "same_training_seed":
        raise StudyEvidenceError("probe ablation pairing must be same_training_seed")
    targeted = _list(
        policy.get("targeted_metrics"),
        field="probe_ablation_policy.targeted_metrics",
    )
    unrelated = _list(
        policy.get("unrelated_metric_registry"),
        field="probe_ablation_policy.unrelated_metric_registry",
    )
    if not targeted or not unrelated:
        raise StudyEvidenceError("probe ablation metric registries must not be empty")
    if set(targeted) & set(unrelated):
        raise StudyEvidenceError("probe ablation targeted/unrelated metrics overlap")
    if (
        policy.get("targeted_improvement_operator") != ">"
        or policy.get("unrelated_metric_scale") != "unit_interval"
        or policy.get("unrelated_metric_direction") != "higher_is_better"
        or policy.get("unsupported_or_empty_metric_policy") != "fail_closed"
    ):
        raise StudyEvidenceError("probe ablation policy semantics are not supported")
    minimum_improvement = policy.get("minimum_targeted_absolute_improvement_each")
    maximum_degradation = policy.get("maximum_unrelated_absolute_degradation")
    if (
        isinstance(minimum_improvement, bool)
        or not isinstance(minimum_improvement, (int, float))
        or isinstance(maximum_degradation, bool)
        or not isinstance(maximum_degradation, (int, float))
    ):
        raise StudyEvidenceError("probe ablation numerical thresholds are invalid")
    minimum_improvement = float(minimum_improvement)
    maximum_degradation = float(maximum_degradation)

    natural_pooled = _pooled_run_metrics(natural_runs)
    full_pooled = _pooled_run_metrics(full_runs)
    rules: dict[str, dict[str, Any]] = {}
    for raw_name in targeted:
        name = _text(raw_name, field="probe_ablation_policy.targeted_metrics[]")
        reference = _reported_outcome_rate(natural_pooled, name)
        candidate = _reported_outcome_rate(full_pooled, name)
        per_seed: dict[str, Any] = {}
        per_seed_available = True
        per_seed_passed = True
        for seed in seeds:
            seed_reference = _reported_outcome_rate(
                _mapping(
                    natural_runs[seed].get("metrics"),
                    field=f"natural_runs[{seed}].metrics",
                ),
                name,
            )
            seed_candidate = _reported_outcome_rate(
                _mapping(
                    full_runs[seed].get("metrics"),
                    field=f"full_runs[{seed}].metrics",
                ),
                name,
            )
            available = seed_reference is not None and seed_candidate is not None
            delta = seed_candidate - seed_reference if available else None
            passed = bool(
                available and delta is not None and delta > minimum_improvement
            )
            per_seed_available = per_seed_available and available
            per_seed_passed = per_seed_passed and passed
            per_seed[str(seed)] = {
                "natural_dagger": seed_reference,
                "natural_dagger_probes": seed_candidate,
                "delta": delta,
                "evidence_available": available,
                "passed": passed,
            }
        evidence_available = bool(
            reference is not None and candidate is not None and per_seed_available
        )
        delta = candidate - reference if evidence_available else None
        passed = bool(
            evidence_available
            and delta is not None
            and delta > minimum_improvement
            and per_seed_passed
        )
        rules[f"targeted.{name}"] = {
            "observed": delta,
            "natural_dagger_rate": reference,
            "natural_dagger_probes_rate": candidate,
            "numerator": None,
            "denominator": None,
            "per_seed": per_seed,
            "threshold": {"operator": ">", "value": minimum_improvement},
            "evidence_available": evidence_available,
            "evidence_failure": (
                None
                if evidence_available
                else "authoritative_action_accuracy_evidence_unavailable"
            ),
            "passed": passed,
        }

    for raw_name in unrelated:
        name = _text(
            raw_name, field="probe_ablation_policy.unrelated_metric_registry[]"
        )
        reference = _reported_outcome_rate(natural_pooled, name)
        candidate = _reported_outcome_rate(full_pooled, name)
        per_seed: dict[str, Any] = {}
        per_seed_available = True
        per_seed_passed = True
        for seed in seeds:
            seed_reference = _reported_outcome_rate(
                _mapping(
                    natural_runs[seed].get("metrics"),
                    field=f"natural_runs[{seed}].metrics",
                ),
                name,
            )
            seed_candidate = _reported_outcome_rate(
                _mapping(
                    full_runs[seed].get("metrics"),
                    field=f"full_runs[{seed}].metrics",
                ),
                name,
            )
            available = seed_reference is not None and seed_candidate is not None
            degradation = seed_reference - seed_candidate if available else None
            passed = bool(
                available
                and degradation is not None
                and degradation <= maximum_degradation
            )
            per_seed_available = per_seed_available and available
            per_seed_passed = per_seed_passed and passed
            per_seed[str(seed)] = {
                "natural_dagger": seed_reference,
                "natural_dagger_probes": seed_candidate,
                "degradation": degradation,
                "evidence_available": available,
                "passed": passed,
            }
        evidence_available = bool(
            reference is not None and candidate is not None and per_seed_available
        )
        degradation = reference - candidate if evidence_available else None
        passed = bool(
            evidence_available
            and degradation is not None
            and degradation <= maximum_degradation
            and per_seed_passed
        )
        rules[f"unrelated.{name}"] = {
            "observed": degradation,
            "natural_dagger_rate": reference,
            "natural_dagger_probes_rate": candidate,
            "numerator": None,
            "denominator": None,
            "per_seed": per_seed,
            "threshold": {"operator": "<=", "value": maximum_degradation},
            "evidence_available": evidence_available,
            "evidence_failure": (
                None
                if evidence_available
                else "registry_required_rate_evidence_unavailable"
            ),
            "passed": passed,
        }
    failures = [name for name, row in rules.items() if row["passed"] is not True]
    return {
        "comparison": "natural_dagger_probes_minus_natural_dagger",
        "scope": scope,
        "paired_by": "training_seed_and_physical_root",
        "study_seeds": seeds,
        "rules": rules,
        "failures": failures,
        "evidence_available": all(
            row["evidence_available"] is True for row in rules.values()
        ),
        "passed": not failures,
    }


def build_study_report(
    *,
    evaluation_artifacts: Mapping[str, Mapping[str, Any]],
    checkpoint_artifacts: Mapping[str, Mapping[int, Any]],
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    study_manifest: str | os.PathLike[str] | None = None,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Build the four-variant primary, stability, and recovery-stress decision."""

    evaluations = _mapping(evaluation_artifacts, field="evaluation_artifacts")
    raw_checkpoints = _mapping(checkpoint_artifacts, field="checkpoint_artifacts")
    _require_exact_keys(evaluations, REQUIRED_VARIANT_IDS, field="evaluation_artifacts")
    _require_exact_keys(
        raw_checkpoints, TRAINED_VARIANT_IDS, field="checkpoint_artifacts"
    )
    bc0_checkpoint_map = _mapping(
        raw_checkpoints["bc0"], field="checkpoint_artifacts.bc0"
    )
    seeds = sorted(
        _nonnegative_integer(seed, field="checkpoint_artifacts.bc0 seed")
        for seed in bc0_checkpoint_map
    )
    manifest, manifest_descriptor = _load_study_manifest(study_manifest, seeds=seeds)
    checkpoint_bindings: dict[str, dict[int, dict[str, Any]]] = {}
    for variant_id in TRAINED_VARIANT_IDS:
        seed_artifacts = _mapping(
            raw_checkpoints[variant_id],
            field=f"checkpoint_artifacts.{variant_id}",
        )
        _require_exact_keys(
            seed_artifacts,
            seeds,
            field=f"checkpoint_artifacts.{variant_id}",
        )
        checkpoint_bindings[variant_id] = {
            seed: extract_checkpoint_binding(
                seed_artifacts[seed],
                variant_id=variant_id,
                study_seed=seed,
                study_manifest=manifest,
                expected_source_commit=expected_source_commit,
            )
            for seed in seeds
        }

    base_artifacts = _mapping(evaluations["base"], field="evaluation_artifacts.base")
    _require_exact_keys(
        base_artifacts, EVALUATION_SCOPES, field="evaluation_artifacts.base"
    )
    base_runs = {
        scope: extract_artifact_metrics(
            base_artifacts[scope],
            variant_id="base",
            study_seed=None,
            evaluation_scope=scope,
            study_manifest=manifest,
            expected_source_commit=expected_source_commit,
        )
        for scope in EVALUATION_SCOPES
    }
    trained_runs_by_scope: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        scope: {} for scope in EVALUATION_SCOPES
    }
    for variant_id in TRAINED_VARIANT_IDS:
        variant_scopes = _mapping(
            evaluations[variant_id],
            field=f"evaluation_artifacts.{variant_id}",
        )
        _require_exact_keys(
            variant_scopes,
            EVALUATION_SCOPES,
            field=f"evaluation_artifacts.{variant_id}",
        )
        for scope in EVALUATION_SCOPES:
            seed_artifacts = _mapping(
                variant_scopes[scope],
                field=f"evaluation_artifacts.{variant_id}.{scope}",
            )
            _require_exact_keys(
                seed_artifacts,
                seeds,
                field=f"evaluation_artifacts.{variant_id}.{scope}",
            )
            trained_runs_by_scope[scope][variant_id] = {
                seed: extract_artifact_metrics(
                    seed_artifacts[seed],
                    variant_id=variant_id,
                    study_seed=seed,
                    evaluation_scope=scope,
                    study_manifest=manifest,
                    expected_source_commit=expected_source_commit,
                )
                for seed in seeds
            }

    stability_policy = _mapping(
        manifest.get("stability_scope_policy"),
        field="study_manifest.stability_scope_policy",
    )
    development_policy = _mapping(
        stability_policy.get("development_holdout"),
        field="study_manifest.stability_scope_policy.development_holdout",
    )
    exact_development_roots = _positive_integer(
        development_policy.get("exact_physical_roots"),
        field=(
            "study_manifest.stability_scope_policy."
            "development_holdout.exact_physical_roots"
        ),
    )
    scope_bindings = {
        scope: _validate_scope_matrix(
            scope=scope,
            base_run=base_runs[scope],
            trained_runs=trained_runs_by_scope[scope],
            seeds=seeds,
            exact_development_roots=exact_development_roots,
        )
        for scope in EVALUATION_SCOPES
    }
    if (
        scope_bindings["development_holdout"]["input_suite_sha256"]
        == scope_bindings["frozen_suite"]["input_suite_sha256"]
    ):
        raise StudyEvidenceError(
            "frozen-suite evidence cannot substitute for development holdout evidence"
        )
    if scope_bindings[RECOVERY_STRESS_SCOPE]["input_suite_sha256"] in {
        scope_bindings["development_holdout"]["input_suite_sha256"],
        scope_bindings["frozen_suite"]["input_suite_sha256"],
    }:
        raise StudyEvidenceError(
            "recovery-stress evidence must use its distinct preregistered suite"
        )
    checkpoint_decision = _validate_checkpoint_matrix(
        checkpoints=checkpoint_bindings,
        trained_runs_by_scope=trained_runs_by_scope,
        seeds=seeds,
    )
    production_d1_quarantine_decision = (
        _validate_production_d1_quarantine_checkpoint_matrix(
            checkpoints=checkpoint_bindings,
            seeds=seeds,
        )
    )

    comparison_policy = _mapping(
        manifest.get("comparison_policy"),
        field="study_manifest.comparison_policy",
    )
    objective_thresholds = _mapping(
        manifest.get("objective_thresholds"),
        field="study_manifest.objective_thresholds",
    )
    primary_by_scope = {
        scope: compare_paired_runs(
            bc0_runs=trained_runs_by_scope[scope]["bc0"],
            full_runs=trained_runs_by_scope[scope]["natural_dagger_probes"],
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            comparison_policy=comparison_policy,
            objective_thresholds=objective_thresholds,
            production_d1_quarantine_evidence=(production_d1_quarantine_decision),
            include_recovery_action_objectives=False,
        )
        for scope in PRIMARY_EVALUATION_SCOPES
    }
    development_stability = _development_stability_decision(
        primary_by_scope["development_holdout"]
    )
    ablation_policy = _mapping(
        comparison_policy.get("probe_ablation_policy"),
        field="study_manifest.comparison_policy.probe_ablation_policy",
    )
    recovery_stress_decision = _recovery_stress_decision(
        full_runs=trained_runs_by_scope[RECOVERY_STRESS_SCOPE][
            "natural_dagger_probes"
        ],
        objective_thresholds=objective_thresholds,
    )
    probe_ablation_by_scope = {
        RECOVERY_STRESS_SCOPE: _probe_ablation_decision(
            scope=RECOVERY_STRESS_SCOPE,
            natural_runs=trained_runs_by_scope[RECOVERY_STRESS_SCOPE][
                "natural_dagger"
            ],
            full_runs=trained_runs_by_scope[RECOVERY_STRESS_SCOPE][
                "natural_dagger_probes"
            ],
            policy=ablation_policy,
        )
    }
    all_scope_bindings_passed = all(row["passed"] for row in scope_bindings.values())
    all_ablation_scopes_passed = all(
        row["passed"] for row in probe_ablation_by_scope.values()
    )
    frozen_primary_evidence_available = all(
        row.get("evidence_available") is True
        for row in (
            *primary_by_scope["frozen_suite"]["decision_rules"].values(),
            *primary_by_scope["frozen_suite"]["objective_decision_rules"].values(),
        )
    )
    overall_rules = {
        "all_four_variants_and_all_scopes_bound": {
            "observed": {
                "variant_count": len(REQUIRED_VARIANT_IDS),
                "scope_count": len(EVALUATION_SCOPES),
            },
            "numerator": len(REQUIRED_VARIANT_IDS) * len(EVALUATION_SCOPES),
            "denominator": len(REQUIRED_VARIANT_IDS) * len(EVALUATION_SCOPES),
            "threshold": {"operator": "==", "value": "complete_matrix"},
            "evidence_available": True,
            "passed": all_scope_bindings_passed,
        },
        "checkpoint_receipt_and_tree_bindings": {
            "observed": checkpoint_decision["validated_checkpoint_count"],
            "numerator": checkpoint_decision["validated_checkpoint_count"],
            "denominator": checkpoint_decision["required_checkpoint_count"],
            "threshold": {
                "operator": "==",
                "value": checkpoint_decision["required_checkpoint_count"],
            },
            "evidence_available": checkpoint_decision["evidence_available"],
            "passed": checkpoint_decision["passed"],
        },
        "production_d1_zero_quarantine_receipt_binding": {
            "observed": production_d1_quarantine_decision["quarantined_rows"],
            "numerator": production_d1_quarantine_decision["numerator"],
            "denominator": production_d1_quarantine_decision["denominator"],
            "threshold": {"operator": "==", "value": 0},
            "evidence_available": production_d1_quarantine_decision[
                "evidence_available"
            ],
            "passed": production_d1_quarantine_decision["passed"],
        },
        "frozen_suite_primary_decision": {
            "observed": primary_by_scope["frozen_suite"]["passed"],
            "numerator": None,
            "denominator": None,
            "threshold": {"operator": "==", "value": True},
            "evidence_available": frozen_primary_evidence_available,
            "passed": primary_by_scope["frozen_suite"]["passed"],
        },
        "development_holdout_stability": {
            "observed": development_stability["passed"],
            "numerator": None,
            "denominator": len(seeds),
            "threshold": {"operator": "==", "value": True},
            "evidence_available": development_stability["evidence_available"],
            "passed": development_stability["passed"],
        },
        "recovery_stress_action_and_safety_targets": {
            "observed": recovery_stress_decision["passed"],
            "numerator": sum(
                row["passed"]
                for row in recovery_stress_decision["rules"].values()
            ),
            "denominator": len(recovery_stress_decision["rules"]),
            "threshold": {"operator": "all", "value": True},
            "evidence_available": recovery_stress_decision[
                "evidence_available"
            ],
            "passed": recovery_stress_decision["passed"],
        },
        "probe_ablation_recovery_stress": {
            "observed": {
                scope: row["passed"] for scope, row in probe_ablation_by_scope.items()
            },
            "numerator": sum(row["passed"] for row in probe_ablation_by_scope.values()),
            "denominator": len(probe_ablation_by_scope),
            "threshold": {"operator": "all", "value": True},
            "evidence_available": all(
                row["evidence_available"] for row in probe_ablation_by_scope.values()
            ),
            "passed": all_ablation_scopes_passed,
        },
    }
    overall_failures = [
        name for name, row in overall_rules.items() if row["passed"] is not True
    ]
    variant_runs: dict[str, Any] = {
        "base": {scope: base_runs[scope] for scope in EVALUATION_SCOPES}
    }
    for variant_id in TRAINED_VARIANT_IDS:
        variant_runs[variant_id] = {
            "checkpoint_receipts": {
                str(seed): checkpoint_bindings[variant_id][seed] for seed in seeds
            },
            **{
                scope: {
                    str(seed): trained_runs_by_scope[scope][variant_id][seed]
                    for seed in seeds
                }
                for scope in EVALUATION_SCOPES
            },
        }
    variant_metric_summary_by_scope = {
        scope: {
            "base": copy.deepcopy(base_runs[scope]["metrics"]),
            **{
                variant_id: _pooled_run_metrics(
                    trained_runs_by_scope[scope][variant_id]
                )
                for variant_id in TRAINED_VARIANT_IDS
            },
        }
        for scope in EVALUATION_SCOPES
    }
    payload: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "dagger_four_variant_three_scope_study_decision",
        "metric_contract": METRIC_CONTRACT,
        "study_manifest": manifest_descriptor,
        "reviewed_source_commit": expected_source_commit,
        "study_seeds": seeds,
        "variant_runs": variant_runs,
        "variant_metric_summary_by_scope": variant_metric_summary_by_scope,
        "scope_bindings": scope_bindings,
        "checkpoint_binding_decision": checkpoint_decision,
        "production_d1_quarantine_decision": (production_d1_quarantine_decision),
        "primary_comparison_by_scope": primary_by_scope,
        "development_stability_decision": development_stability,
        "recovery_stress_decision": recovery_stress_decision,
        "probe_ablation_by_scope": probe_ablation_by_scope,
        "decision_rules": overall_rules,
        "failures": overall_failures,
        "passed": not overall_failures,
    }
    payload["content_sha256"] = stable_json_sha256(payload)
    return payload


def _parse_seed_paths(values: Sequence[str], *, label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in values:
        seed_text, separator, path_text = raw.partition("=")
        if not separator or not seed_text or not path_text:
            raise StudyEvidenceError(
                f"{label} entries must use SEED=/path/to/artifact.json"
            )
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise StudyEvidenceError(f"{label} seed is not an integer: {raw}") from exc
        seed = _nonnegative_integer(seed, field=f"{label} seed")
        if seed in result:
            raise StudyEvidenceError(f"{label} contains duplicate seed {seed}")
        result[seed] = Path(path_text).expanduser()
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the fail-closed four-variant, three-scope DAgger study "
            "decision with checkpoint receipt bindings."
        )
    )
    parser.add_argument(
        "--base-frozen-run",
        required=True,
        type=Path,
        help="One frozen-suite evaluation of the untrained pinned base model.",
    )
    parser.add_argument(
        "--base-development-run",
        required=True,
        type=Path,
        help="One 30-root development evaluation of the pinned base model.",
    )
    parser.add_argument(
        "--base-recovery-stress-run",
        required=True,
        type=Path,
        help="One 70-episode recovery-stress evaluation of the pinned base model.",
    )
    parser.add_argument(
        "--bc0-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help=(
            "BC0 current schema-v4 frozen evaluation artifact; archival "
            "schema-v3 inputs remain ingestible but cannot pass v4 objective "
            "gates. Repeat per seed."
        ),
    )
    parser.add_argument(
        "--natural-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help="Natural-DAgger frozen-suite artifact; repeat per seed.",
    )
    parser.add_argument(
        "--full-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help="Full natural-DAgger-plus-probes artifact; repeat per seed.",
    )
    parser.add_argument(
        "--bc0-development-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help="BC0 30-root development artifact; repeat per seed.",
    )
    parser.add_argument(
        "--natural-development-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help="Natural-DAgger 30-root development artifact; repeat per seed.",
    )
    parser.add_argument(
        "--full-development-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help="Full-DAgger 30-root development artifact; repeat per seed.",
    )
    parser.add_argument(
        "--bc0-recovery-stress-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help="BC0 70-episode recovery-stress artifact; repeat per seed.",
    )
    parser.add_argument(
        "--natural-recovery-stress-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help="Natural-DAgger recovery-stress artifact; repeat per seed.",
    )
    parser.add_argument(
        "--full-recovery-stress-run",
        action="append",
        default=[],
        metavar="SEED=ARTIFACT",
        help="Full-DAgger recovery-stress artifact; repeat per seed.",
    )
    for option, description in (
        ("bc0", "BC0"),
        ("natural", "natural-DAgger"),
        ("full", "natural-DAgger-plus-probes"),
    ):
        parser.add_argument(
            f"--{option}-checkpoint",
            action="append",
            default=[],
            metavar="SEED=RECEIPT",
            help=(f"{description} checkpoint_receipt.json; repeat per seed."),
        )
    parser.add_argument("--study-manifest", type=Path)
    parser.add_argument(
        "--expected-source-commit",
        required=True,
        help="Externally reviewed clean 40-hex commit bound to every run.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        bc0 = _parse_seed_paths(args.bc0_run, label="--bc0-run")
        natural = _parse_seed_paths(args.natural_run, label="--natural-run")
        full = _parse_seed_paths(args.full_run, label="--full-run")
        bc0_development = _parse_seed_paths(
            args.bc0_development_run, label="--bc0-development-run"
        )
        natural_development = _parse_seed_paths(
            args.natural_development_run, label="--natural-development-run"
        )
        full_development = _parse_seed_paths(
            args.full_development_run, label="--full-development-run"
        )
        bc0_recovery_stress = _parse_seed_paths(
            args.bc0_recovery_stress_run,
            label="--bc0-recovery-stress-run",
        )
        natural_recovery_stress = _parse_seed_paths(
            args.natural_recovery_stress_run,
            label="--natural-recovery-stress-run",
        )
        full_recovery_stress = _parse_seed_paths(
            args.full_recovery_stress_run,
            label="--full-recovery-stress-run",
        )
        bc0_checkpoints = _parse_seed_paths(
            args.bc0_checkpoint, label="--bc0-checkpoint"
        )
        natural_checkpoints = _parse_seed_paths(
            args.natural_checkpoint, label="--natural-checkpoint"
        )
        full_checkpoints = _parse_seed_paths(
            args.full_checkpoint, label="--full-checkpoint"
        )
        report = build_study_report(
            evaluation_artifacts={
                "base": {
                    "frozen_suite": args.base_frozen_run,
                    "development_holdout": args.base_development_run,
                    RECOVERY_STRESS_SCOPE: args.base_recovery_stress_run,
                },
                "bc0": {
                    "frozen_suite": bc0,
                    "development_holdout": bc0_development,
                    RECOVERY_STRESS_SCOPE: bc0_recovery_stress,
                },
                "natural_dagger": {
                    "frozen_suite": natural,
                    "development_holdout": natural_development,
                    RECOVERY_STRESS_SCOPE: natural_recovery_stress,
                },
                "natural_dagger_probes": {
                    "frozen_suite": full,
                    "development_holdout": full_development,
                    RECOVERY_STRESS_SCOPE: full_recovery_stress,
                },
            },
            checkpoint_artifacts={
                "bc0": bc0_checkpoints,
                "natural_dagger": natural_checkpoints,
                "natural_dagger_probes": full_checkpoints,
            },
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            study_manifest=args.study_manifest,
            expected_source_commit=args.expected_source_commit,
        )
        _write_json_atomic(args.output, report)
        rendered = json.dumps(
            {
                "passed": report["passed"],
                "output": str(args.output.expanduser().resolve()),
                "content_sha256": report["content_sha256"],
                "failures": report["failures"],
            },
            indent=2,
            sort_keys=True,
        )
        stream = sys.stdout if report["passed"] else sys.stderr
        print(rendered, file=stream)
        return 0 if report["passed"] else 2
    except (OSError, StudyEvidenceError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"passed": False, "error": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EVALUATION_SCOPES",
    "METRIC_CONTRACT",
    "PRIMARY_EVALUATION_SCOPES",
    "RECOVERY_STRESS_SCOPE",
    "StudyEvidenceError",
    "build_study_report",
    "compare_paired_runs",
    "extract_artifact_metrics",
    "extract_checkpoint_binding",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
