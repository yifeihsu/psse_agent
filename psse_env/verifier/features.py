"""Leakage-safe numerical features for observable PSSE transitions.

The verifier operates on complete transitions rather than on a state in
isolation.  This module deliberately uses an allow-list of deployment-visible
signals.  In particular, synthetic-oracle fields such as ``hidden_truth``,
``target_fixed`` and ``healthy_component_modified`` are never copied into the
feature vector.
"""

from __future__ import annotations

import math
import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ACTION_TOOLS = (
    "run_wls",
    "verify_candidate",
    "get_measurement_context",
    "get_parameter_context",
    "get_topology_context",
    "correct_measurements",
    "correct_parameters",
    "correct_topology",
    "commit_state",
    "rollback_state",
    "finalize_diagnosis",
    "ask_for_more_evidence",
    "run_alternative_test",
)

CORRECTION_TOOLS = frozenset(
    {"correct_measurements", "correct_parameters", "correct_topology"}
)
CONTEXT_TOOLS = frozenset(
    {"get_measurement_context", "get_parameter_context", "get_topology_context"}
)
VERIFICATION_TOOLS = frozenset({"run_wls", "verify_candidate"})

# These fields are useful labels for a synthetic oracle, but are forbidden as
# deployment-verifier inputs.  Keeping the list public makes leakage audits
# straightforward.
PRIVILEGED_FIELDS = frozenset(
    {
        "hidden_truth",
        "truth",
        "suggested_actions",
        "oracle_action_hints",
        "oracle_cost_to_go",
        "candidate_hidden_truth_labels",
        "candidate_assessment",
        "clean_case",
        "clean_measurements",
        "clean_parameter_values",
        "true_injected_errors",
        "true_measurement_errors",
        "true_parameter_errors",
        "true_topology_errors",
        "true_error_locations",
        "remaining_true_faults",
        "remaining_true_fault_count",
        "target_fixed",
        "healthy_component_modified",
        "remaining_fault_count",
        "remaining_faults",
        "candidate_disposition",
        "progress_class",
    }
)


def observable_copy(value: Any, *, allow_fields: frozenset[str] = frozenset()) -> Any:
    """Recursively copy a payload while removing verifier-forbidden keys."""

    if isinstance(value, Mapping):
        return {
            str(key): observable_copy(item, allow_fields=allow_fields)
            for key, item in value.items()
            if str(key) not in PRIVILEGED_FIELDS or str(key) in allow_fields
        }
    if isinstance(value, (list, tuple)):
        return [observable_copy(item, allow_fields=allow_fields) for item in value]
    return copy.deepcopy(value)

OBSERVABLE_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "wls_objective": ("wls_objective", "objective", "chi_square_statistic", "j_value"),
    "previous_wls_objective": (
        "previous_wls_objective",
        "parent_wls_objective",
        "pre_action_wls_objective",
    ),
    "residual_norm": ("residual_norm", "max_normalized_residual", "max_residual"),
    "previous_residual_norm": (
        "previous_residual_norm",
        "parent_residual_norm",
        "pre_action_residual_norm",
    ),
    "target_progress": ("target_progress", "targeted_residual_improvement"),
    "global_progress": ("global_progress", "objective_improvement"),
    "remaining_anomaly_score": ("remaining_anomaly_score", "post_action_anomaly_score"),
    "anomaly_threshold": ("anomaly_threshold", "chi_square_threshold"),
    "new_violations": (
        "new_violations",
        "new_large_residuals",
        "physical_bound_violations",
    ),
    "power_flow_converged": ("power_flow_converged",),
    "topology_feasible": ("topology_feasible",),
    "post_action_resolved": ("post_action_resolved", "globally_resolved"),
    "modification_magnitude": (
        "modification_magnitude",
        "candidate_modification_magnitude",
        "normalized_modification_magnitude",
    ),
}


FEATURE_NAMES = (
    "execution_success",
    "execution_failure",
    "action_known",
    "action_is_verification",
    "action_is_context",
    "action_is_correction",
    "action_is_commit",
    "action_is_rollback",
    "action_is_finalize",
    "had_open_candidate",
    "had_unverified_candidate",
    "had_verified_candidate",
    "has_open_candidate_after",
    "has_unverified_candidate_after",
    "has_verified_candidate_after",
    "candidate_created",
    "active_state_changed",
    "state_hash_changed",
    "fresh_context_count",
    "remaining_budget",
    "remaining_budget_delta",
    "history_length",
    "history_failure_count",
    "accepted_correction_count",
    "rejected_hypothesis_count",
    "wls_objective_before_known",
    "wls_objective_before",
    "wls_objective_after_known",
    "wls_objective_after",
    "wls_objective_improvement",
    "residual_before_known",
    "residual_before",
    "residual_after_known",
    "residual_after",
    "residual_improvement",
    "target_progress_known",
    "target_progress",
    "global_progress_known",
    "global_progress",
    "anomaly_score_known",
    "anomaly_score",
    "anomaly_threshold_known",
    "anomaly_margin",
    "anomaly_resolved_known",
    "anomaly_resolved",
    "new_violations_known",
    "new_violations_count",
    "power_flow_converged_known",
    "power_flow_converged",
    "topology_feasible_known",
    "topology_feasible",
    "modification_magnitude_known",
    "modification_magnitude",
)


@dataclass(frozen=True)
class TransitionInput:
    """Normalized complete-transition input consumed by verifier components."""

    parent_state_summary: Mapping[str, Any]
    action: Mapping[str, Any]
    tool_output: Mapping[str, Any]
    candidate_state_summary: Mapping[str, Any]
    verification_metrics: Mapping[str, Any]
    history_summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "parent_state_summary": dict(self.parent_state_summary),
            "action": dict(self.action),
            "tool_output": dict(self.tool_output),
            "candidate_state_summary": dict(self.candidate_state_summary),
            "verification_metrics": dict(self.verification_metrics),
            "history_summary": dict(self.history_summary),
        }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key)
    return _mapping(nested)


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _bool_number(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "success", "converged", "feasible"}:
            return 1.0, 1.0
        if lowered in {"false", "no", "0", "failure", "failed", "infeasible"}:
            return 1.0, 0.0
        return 0.0, 0.0
    return 1.0, float(bool(value))


def _count(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return float(len(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return float(len(value))
    converted = _safe_float(value)
    return max(converted, 0.0) if converted is not None else None


def normalize_action(action: Any) -> dict[str, Any]:
    """Best-effort action normalization that never raises."""

    if isinstance(action, str):
        return {"tool": action, "arguments": {}}
    if not isinstance(action, Mapping):
        return {"tool": "__invalid_action__", "arguments": {"normalization_error": "action_not_mapping"}}
    function = action.get("function")
    if function is not None and not isinstance(function, Mapping):
        return {"tool": "__invalid_action__", "arguments": {"normalization_error": "function_not_mapping"}}
    function = function if isinstance(function, Mapping) else {}
    tool = action.get("tool") or action.get("name") or action.get("tool_name") or function.get("name")
    arguments = action.get("arguments", function.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"tool": "__invalid_action__", "arguments": {"normalization_error": "arguments_not_json"}}
    if not isinstance(arguments, Mapping):
        return {"tool": "__invalid_action__", "arguments": {"normalization_error": "arguments_not_mapping"}}
    if not tool:
        return {"tool": "__invalid_action__", "arguments": {"normalization_error": "missing_tool"}}
    return {
        "tool": str(tool) if tool else "__invalid_action__",
        "arguments": dict(arguments),
    }


def summarize_history(history: Any) -> dict[str, Any]:
    """Reduce raw history to observable counts without retaining free-form text."""

    if isinstance(history, Mapping):
        return dict(history)
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
        return {}

    failures = 0
    accepted = 0
    rejected = 0
    for item in history:
        if not isinstance(item, Mapping):
            continue
        output = _mapping(item.get("tool_output"))
        status = output.get("execution_status")
        if status == "failure":
            failures += 1
        # Accepted/rejected candidate counts are deployment-observable only
        # after a successful disposition action.  Never derive them from the
        # privileged transition-label candidate disposition.
        action = normalize_action(item.get("action") or item.get("executed_action"))
        tool = action.get("tool")
        if status == "success" and tool == "commit_state":
            accepted += 1
        elif status == "success" and tool == "rollback_state":
            rejected += 1
    return {
        "history_length": len(history),
        "failure_count": failures,
        "accepted_count": accepted,
        "rejected_count": rejected,
    }


def normalize_transition(
    transition: Mapping[str, Any] | None = None,
    *,
    parent_state_summary: Mapping[str, Any] | None = None,
    action: Mapping[str, Any] | str | None = None,
    tool_output: Mapping[str, Any] | None = None,
    candidate_state_summary: Mapping[str, Any] | None = None,
    verification_metrics: Mapping[str, Any] | None = None,
    history_summary: Mapping[str, Any] | Sequence[Any] | None = None,
) -> TransitionInput:
    """Normalize both roadmap-style and DAgger-style transition mappings."""

    row = _mapping(transition)
    nested = _mapping(row.get("transition"))
    if nested:
        merged = dict(row)
        merged.update(nested)
        row = merged

    parent = _mapping(
        parent_state_summary
        or row.get("parent_state_summary")
        or row.get("state_summary")
        or row.get("state")
    )
    normalized_action = normalize_action(
        action
        or row.get("action")
        or row.get("executed_action")
        or row.get("model_action")
    )
    output = _mapping(tool_output or row.get("tool_output") or row.get("observation"))
    candidate = _mapping(
        candidate_state_summary
        or row.get("candidate_state_summary")
        or row.get("next_state_summary")
        or row.get("next_state")
    )
    metrics = _mapping(verification_metrics or row.get("verification_metrics"))
    raw_history = history_summary
    if raw_history is None:
        raw_history = row.get("history_summary", row.get("history_window", row.get("history")))

    return TransitionInput(
        parent_state_summary=parent,
        action=normalized_action,
        tool_output=output,
        candidate_state_summary=candidate,
        verification_metrics=metrics,
        history_summary=summarize_history(raw_history),
    )


def _first_metric(sources: Sequence[Mapping[str, Any]], canonical_name: str) -> Any:
    aliases = OBSERVABLE_METRIC_ALIASES[canonical_name]
    for source in sources:
        for alias in aliases:
            if alias in source and alias not in PRIVILEGED_FIELDS:
                return source[alias]
    return None


def observable_verification_metrics(transition: Mapping[str, Any] | TransitionInput) -> dict[str, Any]:
    """Return only allow-listed metrics, with explicit metrics taking precedence."""

    item = transition if isinstance(transition, TransitionInput) else normalize_transition(transition)
    candidate_last = _nested_mapping(item.candidate_state_summary, "last_verification")
    candidate_output = _nested_mapping(item.candidate_state_summary, "last_tool_output")
    candidate_tool_metrics = _nested_mapping(candidate_output, "tool_metrics")
    output_tool_metrics = _nested_mapping(item.tool_output, "tool_metrics")
    sources = (
        item.verification_metrics,
        candidate_last,
        candidate_tool_metrics,
        output_tool_metrics,
        item.tool_output,
    )
    observed: dict[str, Any] = {}
    for canonical_name in OBSERVABLE_METRIC_ALIASES:
        value = _first_metric(sources, canonical_name)
        if value is not None:
            observed[canonical_name] = value
    return observed


def _metric_float(metrics: Mapping[str, Any], name: str) -> tuple[float, float]:
    value = _safe_float(metrics.get(name))
    return (0.0, 0.0) if value is None else (1.0, value)


def extract_transition_features(
    transition: Mapping[str, Any] | None = None,
    **transition_parts: Any,
) -> dict[str, float]:
    """Extract a fixed-order, finite numerical feature mapping.

    Missing values are represented by a zero value plus a corresponding
    ``*_known`` feature.  This avoids conflating a real zero with missing data.
    """

    item = normalize_transition(transition, **transition_parts)
    parent = item.parent_state_summary
    candidate = item.candidate_state_summary
    output = item.tool_output
    history = item.history_summary
    metrics = observable_verification_metrics(item)
    tool = str(item.action.get("tool", "__invalid_action__"))

    status = output.get("execution_status", candidate.get("last_tool_status"))
    success = float(status == "success")
    failure = float(status == "failure")

    before_id = parent.get("active_state_id")
    after_id = candidate.get("active_state_id")
    before_hash = parent.get("state_hash") or parent.get("active_state_hash")
    after_hash = candidate.get("state_hash") or candidate.get("active_state_hash")

    before_budget = _safe_float(parent.get("remaining_budget"))
    after_budget = _safe_float(candidate.get("remaining_budget"))

    parent_last = _nested_mapping(parent, "last_verification")
    before_objective = _safe_float(
        metrics.get("previous_wls_objective", parent_last.get("wls_objective", parent.get("wls_objective")))
    )
    after_objective = _safe_float(metrics.get("wls_objective"))
    before_residual = _safe_float(
        metrics.get("previous_residual_norm", parent_last.get("residual_norm", parent.get("residual_norm")))
    )
    after_residual = _safe_float(metrics.get("residual_norm"))

    target_known, target_progress = _metric_float(metrics, "target_progress")
    global_known, global_progress = _metric_float(metrics, "global_progress")
    anomaly_known, anomaly_score = _metric_float(metrics, "remaining_anomaly_score")
    threshold_known, anomaly_threshold = _metric_float(metrics, "anomaly_threshold")
    violation_count = _count(metrics.get("new_violations"))
    modification_known, modification_magnitude = _metric_float(metrics, "modification_magnitude")
    power_known, power_ok = _bool_number(metrics.get("power_flow_converged"))
    topology_known, topology_ok = _bool_number(metrics.get("topology_feasible"))

    resolved_known, resolved = _bool_number(metrics.get("post_action_resolved"))
    if anomaly_known and threshold_known:
        declared_known, declared_resolved = resolved_known, resolved
        resolved_known = 1.0
        score_resolved = float(anomaly_score < anomaly_threshold)
        resolved = min(score_resolved, declared_resolved) if declared_known else score_resolved
    elif not resolved_known and candidate.get("no_material_anomaly_remaining") is True:
        resolved_known, resolved = 1.0, 1.0
    if (
        (power_known and not power_ok)
        or (topology_known and not topology_ok)
        or (violation_count is not None and violation_count > 0.0)
    ):
        resolved_known, resolved = 1.0, 0.0

    fresh_context_count = sum(
        bool(parent.get(f"has_fresh_{family}_context"))
        for family in ("measurement", "parameter", "topology")
    )
    accepted_count = _count(parent.get("accepted_corrections")) or 0.0
    rejected_count = _count(parent.get("rejected_hypotheses")) or 0.0

    features = {name: 0.0 for name in FEATURE_NAMES}
    features.update(
        {
            "execution_success": success,
            "execution_failure": failure,
            "action_known": float(tool in ACTION_TOOLS),
            "action_is_verification": float(tool in VERIFICATION_TOOLS),
            "action_is_context": float(tool in CONTEXT_TOOLS),
            "action_is_correction": float(tool in CORRECTION_TOOLS),
            "action_is_commit": float(tool == "commit_state"),
            "action_is_rollback": float(tool == "rollback_state"),
            "action_is_finalize": float(tool == "finalize_diagnosis"),
            "had_open_candidate": float(bool(parent.get("has_open_candidate"))),
            "had_unverified_candidate": float(bool(parent.get("has_unverified_candidate"))),
            "had_verified_candidate": float(bool(parent.get("has_verified_candidate"))),
            "has_open_candidate_after": float(bool(candidate.get("has_open_candidate"))),
            "has_unverified_candidate_after": float(bool(candidate.get("has_unverified_candidate"))),
            "has_verified_candidate_after": float(bool(candidate.get("has_verified_candidate"))),
            "candidate_created": float(
                not parent.get("candidate_state_id") and bool(candidate.get("candidate_state_id"))
            ),
            "active_state_changed": float(
                before_id is not None and after_id is not None and str(before_id) != str(after_id)
            ),
            "state_hash_changed": float(
                before_hash is not None and after_hash is not None and str(before_hash) != str(after_hash)
            ),
            "fresh_context_count": float(fresh_context_count),
            "remaining_budget": after_budget if after_budget is not None else (before_budget or 0.0),
            "remaining_budget_delta": (
                after_budget - before_budget
                if before_budget is not None and after_budget is not None
                else 0.0
            ),
            "history_length": _safe_float(history.get("history_length")) or 0.0,
            "history_failure_count": _safe_float(history.get("failure_count")) or 0.0,
            "accepted_correction_count": accepted_count,
            "rejected_hypothesis_count": rejected_count,
            "wls_objective_before_known": float(before_objective is not None),
            "wls_objective_before": before_objective or 0.0,
            "wls_objective_after_known": float(after_objective is not None),
            "wls_objective_after": after_objective or 0.0,
            "wls_objective_improvement": (
                before_objective - after_objective
                if before_objective is not None and after_objective is not None
                else 0.0
            ),
            "residual_before_known": float(before_residual is not None),
            "residual_before": before_residual or 0.0,
            "residual_after_known": float(after_residual is not None),
            "residual_after": after_residual or 0.0,
            "residual_improvement": (
                before_residual - after_residual
                if before_residual is not None and after_residual is not None
                else 0.0
            ),
            "target_progress_known": target_known,
            "target_progress": target_progress,
            "global_progress_known": global_known,
            "global_progress": global_progress,
            "anomaly_score_known": anomaly_known,
            "anomaly_score": anomaly_score,
            "anomaly_threshold_known": threshold_known,
            "anomaly_margin": (
                anomaly_threshold - anomaly_score if anomaly_known and threshold_known else 0.0
            ),
            "anomaly_resolved_known": resolved_known,
            "anomaly_resolved": resolved,
            "new_violations_known": float(violation_count is not None),
            "new_violations_count": violation_count or 0.0,
            "power_flow_converged_known": power_known,
            "power_flow_converged": power_ok,
            "topology_feasible_known": topology_known,
            "topology_feasible": topology_ok,
            "modification_magnitude_known": modification_known,
            "modification_magnitude": modification_magnitude,
        }
    )
    # Defensive final pass: model code never receives NaN or infinity.
    return {
        name: float(value) if math.isfinite(float(value)) else 0.0
        for name, value in features.items()
    }


def feature_vector(
    features_or_transition: Mapping[str, Any],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> list[float]:
    """Return a deterministic vector from either features or a transition."""

    if all(name in features_or_transition for name in feature_names):
        features = features_or_transition
    else:
        features = extract_transition_features(features_or_transition)
    vector: list[float] = []
    for name in feature_names:
        value = _safe_float(features.get(name))
        vector.append(value if value is not None else 0.0)
    return vector


class TransitionFeatureExtractor:
    """Small fit-free transformer with a scikit-learn-like interface."""

    feature_names = FEATURE_NAMES

    def transform_one(self, transition: Mapping[str, Any]) -> dict[str, float]:
        return extract_transition_features(transition)

    def transform(self, transitions: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        return [feature_vector(self.transform_one(item)) for item in transitions]


# Concise compatibility aliases for callers that prefer generic names.
extract_features = extract_transition_features
build_feature_vector = feature_vector


__all__ = [
    "ACTION_TOOLS",
    "FEATURE_NAMES",
    "OBSERVABLE_METRIC_ALIASES",
    "PRIVILEGED_FIELDS",
    "TransitionFeatureExtractor",
    "TransitionInput",
    "build_feature_vector",
    "extract_features",
    "extract_transition_features",
    "feature_vector",
    "normalize_action",
    "normalize_transition",
    "observable_copy",
    "observable_verification_metrics",
    "summarize_history",
]
