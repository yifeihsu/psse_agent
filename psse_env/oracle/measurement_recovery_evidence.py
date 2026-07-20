from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    GET_MEASUREMENT_CONTEXT,
    RUN_WLS,
)
from psse_env.oracle.expert_types import history_action_tool, state_value


def measurement_target_indices(action: Mapping[str, Any]) -> set[int]:
    """Return validated zero-based meter targets from one correction action."""
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    group = arguments.get("suspect_group")
    if isinstance(group, (list, tuple)):
        values = group
    else:
        updates = arguments.get("measurement_updates")
        values = updates.keys() if isinstance(updates, Mapping) else ()
    indices: set[int] = set()
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            indices.add(value)
    return indices


def accepted_measurement_indices(state: Any) -> set[int]:
    """Return meter targets already committed on the current recovery path."""
    indices: set[int] = set()
    for item in state_value(state, "accepted_corrections", []) or []:
        if history_action_tool(item) != CORRECT_MEASUREMENTS:
            if not isinstance(item, Mapping) or str(
                item.get("family") or item.get("action_family") or ""
            ).lower() != "measurement":
                continue
        action = item.get("source_action") if isinstance(item, Mapping) else None
        action = action if isinstance(action, Mapping) else item
        if isinstance(action, Mapping):
            indices.update(measurement_target_indices(action))
    return indices


def eligible_joint_measurement_targets(
    state: Any,
    history: Sequence[Mapping[str, Any]],
    *,
    active_id: Any,
    supported_actions: Sequence[Mapping[str, Any]],
    accepted_indices: set[int],
    min_remaining_budget: int = 4,
) -> list[int]:
    """Return the evidence-closed new targets for one bounded joint retry.

    Provider-declared grouped corrections are conditional contracts.  A group
    becomes autonomously available only after its same-state singleton
    candidates establish either physical-safe coupled progress or an exact Vm
    dependency closure.  Keeping this predicate shared makes expert routing and
    terminal recovery-exhaustion auditing apply the same availability rule.
    """
    if active_id is None:
        return []
    try:
        remaining_budget = int(state_value(state, "remaining_budget", 0))
    except (TypeError, ValueError, OverflowError):
        return []
    if remaining_budget < int(min_remaining_budget):
        return []

    supported_target_sets: set[frozenset[int]] = set()
    for action in supported_actions:
        targets = measurement_target_indices(action)
        if targets:
            supported_target_sets.add(frozenset(targets))

    rejected_by_candidate: dict[str, frozenset[int]] = {}
    rejected_order: list[str] = []
    for item in state_value(state, "rejected_hypotheses", []) or []:
        if not isinstance(item, Mapping):
            continue
        action = item.get("source_action") or item.get("action") or {}
        if history_action_tool(action) != CORRECT_MEASUREMENTS:
            continue
        targets = measurement_target_indices(action)
        frozen_targets = frozenset(targets)
        if (
            not frozen_targets
            or frozen_targets & accepted_indices
            or frozen_targets not in supported_target_sets
        ):
            continue
        parent_id = item.get("candidate_parent_id")
        requested = (
            action.get("arguments", {}).get("state_id")
            if isinstance(action, Mapping)
            and isinstance(action.get("arguments"), Mapping)
            else None
        )
        if parent_id is not None and str(parent_id) != str(active_id):
            continue
        if requested is not None and str(requested) != str(active_id):
            continue
        candidate_id = item.get("candidate_state_id")
        if candidate_id is None:
            continue
        candidate_key = str(candidate_id)
        rejected_by_candidate[candidate_key] = frozen_targets
        rejected_order.append(candidate_key)

    verified: dict[str, dict[str, Any]] = {}
    for event in history:
        if not isinstance(event, Mapping):
            continue
        action = event.get("action") or event.get("executed_action") or {}
        if history_action_tool(action) != RUN_WLS:
            continue
        arguments = action.get("arguments") if isinstance(action, Mapping) else {}
        arguments = arguments if isinstance(arguments, Mapping) else {}
        candidate_key = str(arguments.get("state_id") or "")
        targets = rejected_by_candidate.get(candidate_key)
        if targets is None:
            continue
        output = event.get("tool_output")
        metrics = output.get("tool_metrics") if isinstance(output, Mapping) else None
        if not isinstance(metrics, Mapping):
            continue
        try:
            target_value = float(metrics["target_metric_value"])
            target_threshold = float(metrics["target_metric_threshold"])
            target_progress = float(metrics["target_progress"])
            global_progress = float(metrics["global_progress"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        target_fixed = bool(
            target_threshold > 0.0
            and target_value < target_threshold
            and target_progress >= 0.80
        )
        if not (
            output.get("execution_status") == "success"
            and target_fixed
            and global_progress > 0.0
            and metrics.get("globally_resolved") is False
        ):
            continue
        physical_ok = metrics.get("physical_constraints_ok")
        violation_indices: set[int] | None = set()
        if physical_ok is False:
            raw_violations = metrics.get("physical_bound_violations")
            if not isinstance(raw_violations, (list, tuple)) or not raw_violations:
                continue
            violation_indices = set()
            for violation in raw_violations:
                if not isinstance(violation, Mapping) or str(
                    violation.get("type") or ""
                ) != "bus_voltage_out_of_bounds":
                    violation_indices = None
                    break
                raw_index = violation.get("measurement_index0")
                if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                    violation_indices = None
                    break
                violation_indices.add(raw_index)
            if not violation_indices:
                continue
        elif physical_ok is not True:
            continue
        verified[candidate_key] = {
            "targets": targets,
            "global_progress": global_progress,
            "physical_ok": physical_ok is True,
            "violation_indices": violation_indices,
        }

    ordered = [verified[key] for key in rejected_order if key in verified]
    safe = [
        item
        for item in ordered
        if item["physical_ok"] and item["global_progress"] < 0.20
    ]
    violation_bound = [item for item in ordered if not item["physical_ok"]]

    context_physical_vm_targets: set[int] = set()
    for event in reversed(history):
        if not isinstance(event, Mapping):
            continue
        action = event.get("action") or event.get("executed_action") or {}
        if history_action_tool(action) != GET_MEASUREMENT_CONTEXT:
            continue
        arguments = action.get("arguments") if isinstance(action, Mapping) else {}
        arguments = arguments if isinstance(arguments, Mapping) else {}
        requested = arguments.get("state_id")
        if requested is not None and str(requested) != str(active_id):
            continue
        output = event.get("tool_output")
        metrics = output.get("tool_metrics") if isinstance(output, Mapping) else None
        if not isinstance(metrics, Mapping):
            break
        raw_targets = metrics.get("physical_vm_joint_targets")
        if isinstance(raw_targets, (list, tuple)) and all(
            isinstance(index, int) and not isinstance(index, bool) and index >= 0
            for index in raw_targets
        ):
            context_physical_vm_targets = set(raw_targets)
        break

    # Provider-declared out-of-bound Vm targets can close the exact physical
    # violations of a locally fixed blocked residual.  The executable union
    # must itself be present in the same context.
    if context_physical_vm_targets:
        for blocked in violation_bound:
            blocked_targets = set(blocked["targets"])
            combined = context_physical_vm_targets | blocked_targets
            if (
                blocked["violation_indices"] == context_physical_vm_targets
                and not (context_physical_vm_targets & blocked_targets)
                and frozenset(accepted_indices | combined) in supported_target_sets
            ):
                return sorted(combined)

    # A physical-safe rejected target can close the exact pre-existing Vm
    # violations of another locally fixed target.
    for blocked in violation_bound:
        violations = blocked["violation_indices"]
        for closure in safe:
            closure_targets = set(closure["targets"])
            blocked_targets = set(blocked["targets"])
            if (
                violations == closure_targets
                and not (closure_targets & blocked_targets)
                and frozenset(
                    accepted_indices | closure_targets | blocked_targets
                )
                in supported_target_sets
            ):
                return sorted(closure_targets | blocked_targets)

    # Otherwise allow exactly two physically safe, locally fixed singleton
    # targets whose separate progress fell below the partial-acceptance floor.
    safe_singletons = [
        next(iter(item["targets"]))
        for item in safe
        if len(item["targets"]) == 1
    ]
    unique_singletons = list(dict.fromkeys(safe_singletons))
    pair = set(unique_singletons[:2])
    return (
        sorted(pair)
        if len(pair) == 2
        and frozenset(accepted_indices | pair) in supported_target_sets
        else []
    )
