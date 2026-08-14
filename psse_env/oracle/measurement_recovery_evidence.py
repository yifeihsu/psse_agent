from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    GET_MEASUREMENT_CONTEXT,
    RUN_WLS,
)
from psse_env.oracle.expert_types import (
    history_action_tool,
    recovery_record_applies_to_state,
    state_value,
)


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
        elif isinstance(value, str):
            # measurement_updates keys arrive as strings after any JSON round
            # trip (JSON object keys are always strings); accept only the
            # canonical non-negative decimal form.
            text = value.strip()
            if text.isdigit():
                indices.add(int(text))
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


def verified_terminal_measurement_closure_action(
    context_evidence: Mapping[str, Any],
    *,
    active_id: Any,
    active_state_hash: Any | None = None,
    accepted_targets: set[int] | None = None,
) -> dict[str, Any] | None:
    """Validate one provider-attested terminal measurement closure action.

    The closure is an exceptional authorization for an accepted-target-plus-
    singleton grouped repair.  A target list alone is therefore insufficient:
    the exact same-state action and ordered candidate-quality screening records
    must be present in the context contract.
    """

    if active_id is None or not isinstance(context_evidence, Mapping):
        return None
    if str(context_evidence.get("state_id") or "") != str(active_id):
        return None
    context_hash = str(context_evidence.get("state_hash") or "")
    if not context_hash or (
        active_state_hash is not None
        and context_hash != str(active_state_hash)
    ):
        return None

    raw_targets = context_evidence.get(
        "verified_terminal_measurement_closure_targets"
    )
    if not isinstance(raw_targets, (list, tuple)) or len(raw_targets) < 2:
        return None
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in raw_targets
    ):
        return None
    targets = list(raw_targets)
    if targets != sorted(set(targets)):
        return None

    attestation = context_evidence.get(
        "verified_terminal_measurement_closure_evidence"
    )
    if not isinstance(attestation, Mapping):
        return None
    if (
        attestation.get("eligible") is not True
        or str(attestation.get("state_id") or "") != str(active_id)
        or str(attestation.get("state_hash") or "") != context_hash
        or attestation.get("screening_method")
        != "singleton_then_grouped_deployment_candidate_quality"
        or attestation.get("closure_targets") != targets
    ):
        return None
    new_target = attestation.get("new_target")
    if (
        not isinstance(new_target, int)
        or isinstance(new_target, bool)
        or new_target < 0
        or new_target not in targets
    ):
        return None
    if accepted_targets is not None and (
        not accepted_targets
        or new_target in accepted_targets
        or set(targets) != accepted_targets | {new_target}
        or len(set(targets) - accepted_targets) != 1
    ):
        return None

    supported = context_evidence.get("supported_corrections")
    if not isinstance(supported, (list, tuple)):
        return None
    exact_action: dict[str, Any] | None = None
    for action in supported:
        if (
            not isinstance(action, Mapping)
            or set(action) != {"tool", "arguments"}
            or action.get("tool") != CORRECT_MEASUREMENTS
        ):
            continue
        arguments = action.get("arguments")
        if (
            not isinstance(arguments, Mapping)
            or set(arguments) != {"state_id", "suspect_group"}
        ):
            continue
        if (
            str(arguments.get("state_id") or "") == str(active_id)
            and arguments.get("suspect_group") == targets
        ):
            exact_action = {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": str(active_id),
                    "suspect_group": targets,
                },
            }
            break
    if exact_action is None:
        return None

    attempts = attestation.get("attempts")
    if not isinstance(attempts, (list, tuple)):
        return None
    singleton_index: int | None = None
    closure_index: int | None = None
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            continue
        if (
            attempt.get("stage") == "new_target_singleton"
            and attempt.get("targets") == [new_target]
            and attempt.get("disposition") in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}
            and attempt.get("target_test_passed") is True
            and attempt.get("physical_constraints_ok") is True
        ):
            singleton_index = index
        if (
            attempt.get("stage") == "accepted_targets_plus_singleton"
            and attempt.get("targets") == targets
            and attempt.get("disposition") == "ACCEPT_FINAL"
            and attempt.get("target_test_passed") is True
            and attempt.get("globally_resolved") is True
            and attempt.get("physical_constraints_ok") is True
        ):
            closure_index = index
    if (
        singleton_index is None
        or closure_index is None
        or singleton_index >= closure_index
    ):
        return None
    return exact_action


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
    durable_verifications: dict[str, dict[str, Any]] = {}
    for item in state_value(state, "rejected_hypotheses", []) or []:
        if not recovery_record_applies_to_state(item, active_id):
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
        candidate_id = item.get("candidate_state_id")
        if candidate_id is None:
            continue
        candidate_key = str(candidate_id)
        rejected_by_candidate[candidate_key] = frozen_targets
        rejected_order.append(candidate_key)
        verification_summary = item.get("verification_summary")
        if isinstance(verification_summary, Mapping):
            durable_verifications[candidate_key] = dict(verification_summary)

    verified: dict[str, dict[str, Any]] = {}
    verification_events = list(history)
    # A rejected candidate's WLS transition can age out of the bounded policy
    # window while its same-state context remains fresh.  The rejection record
    # carries a compact copy of that already-observed verification evidence, so
    # evaluate it through the exact same predicate as a live history event.
    for candidate_key in rejected_order:
        metrics = durable_verifications.get(candidate_key)
        if metrics is None:
            continue
        verification_events.append(
            {
                "action": {
                    "tool": RUN_WLS,
                    "arguments": {"state_id": candidate_key},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": metrics,
                },
            }
        )
    for event in verification_events:
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
    fresh_context_evidence = state_value(state, "fresh_context_evidence", {})
    if isinstance(fresh_context_evidence, Mapping):
        measurement_evidence = fresh_context_evidence.get("measurement")
        if (
            isinstance(measurement_evidence, Mapping)
            and str(measurement_evidence.get("state_id") or "") == str(active_id)
        ):
            raw_targets = measurement_evidence.get("physical_vm_joint_targets")
            if isinstance(raw_targets, (list, tuple)) and all(
                isinstance(index, int) and not isinstance(index, bool) and index >= 0
                for index in raw_targets
            ):
                context_physical_vm_targets = set(raw_targets)
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
