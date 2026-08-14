from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from psse_env.actions import (
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    FINALIZE_DIAGNOSIS,
    ROLLBACK_STATE,
    safe_normalize_action,
)


@dataclass(frozen=True)
class InjectedAction:
    family: str
    action: dict[str, Any]
    setup_actions: tuple[dict[str, Any], ...] = ()


def plausible_wrong_actions(
    state: Mapping[str, Any],
    expert_actions: Iterable[Mapping[str, Any]] | None = None,
    *,
    physical_state: Mapping[str, Any] | None = None,
) -> list[InjectedAction]:
    """Create a bounded, deterministic family of plausible learner mistakes."""
    active_id = state.get("active_state_id")
    candidate_id = state.get("candidate_state_id") or "missing_candidate"
    injected: list[InjectedAction] = [
        InjectedAction(
            "stale_context",
            {"tool": "get_parameter_context", "arguments": {"state_id": "previous_episode:s0"}},
        ),
        InjectedAction(
            "premature_commit",
            {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}},
        ),
        InjectedAction("premature_finalization", {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}),
        InjectedAction(
            "rollback_without_disposition",
            {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}},
        ),
    ]
    for raw in expert_actions or []:
        action = safe_normalize_action(raw)
        tool = action["tool"]
        if tool not in {CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
            continue
        args = copy.deepcopy(action["arguments"])
        args["state_id"] = active_id
        wrong_family = {
            CORRECT_MEASUREMENTS: CORRECT_PARAMETERS,
            CORRECT_PARAMETERS: CORRECT_TOPOLOGY,
            CORRECT_TOPOLOGY: CORRECT_MEASUREMENTS,
        }[tool]
        wrong_family_args = {
            "state_id": active_id,
            "case_updates": {"counterfactual_wrong_family": wrong_family},
        }
        injected.append(
            InjectedAction(
                "wrong_fault_family",
                {"tool": wrong_family, "arguments": wrong_family_args},
            )
        )

        wrong_target_args = copy.deepcopy(args)
        physical = dict(physical_state or {})
        case = physical.get("case") if isinstance(physical.get("case"), Mapping) else {}
        branch_rows = list(case.get("branch") or []) if isinstance(case, Mapping) else []
        if tool == CORRECT_MEASUREMENTS and "measurement_updates" in wrong_target_args and isinstance(wrong_target_args["measurement_updates"], Mapping):
            updates = dict(wrong_target_args["measurement_updates"])
            if updates:
                first_key, first_value = next(iter(updates.items()))
                try:
                    original_index = int(first_key)
                    measurements = list(physical.get("measurements") or [])
                    alternatives = [index for index in range(len(measurements)) if index != original_index]
                    wrong_key: Any = alternatives[0] if alternatives else original_index + 1
                except (TypeError, ValueError):
                    wrong_key = f"{first_key}_healthy"
                wrong_target_args["measurement_updates"] = {wrong_key: first_value}
        elif tool == CORRECT_PARAMETERS:
            for key in ("branch_id", "cb_name", "line_index", "line_index1", "branch_row0", "target"):
                wrong_target_args.pop(key, None)
            if len(branch_rows) > 1:
                wrong_target_args["branch_row0"] = 1
                field = str(args.get("parameter") or args.get("field") or "x")
                row = branch_rows[1] if isinstance(branch_rows[1], Mapping) else {}
                current = row.get(field)
                wrong_target_args["parameter"] = field
                wrong_target_args["value"] = (
                    float(current) + 1.0 if isinstance(current, (int, float)) else "counterfactual_wrong"
                )
            else:
                current_field = str(args.get("parameter") or args.get("field") or "x")
                wrong_target_args["branch_row0"] = 0
                wrong_target_args["parameter"] = "r" if current_field != "r" else "x"
                value = wrong_target_args.get("value", wrong_target_args.get("corrected_value", 1.0))
                wrong_target_args["value"] = float(value) + 1.0 if isinstance(value, (int, float)) else 1.0
        elif tool == CORRECT_TOPOLOGY:
            for key in ("branch_id", "cb_name", "line_index", "line_index1", "branch_row0", "target"):
                wrong_target_args.pop(key, None)
            if len(branch_rows) > 1:
                wrong_target_args["branch_row0"] = 1
                row = branch_rows[1] if isinstance(branch_rows[1], Mapping) else {}
                status_field = str(args.get("status_field") or ("br_status" if "br_status" in row else "status"))
                current = row.get(status_field)
                if isinstance(current, (int, float)):
                    wrong_target_args["status"] = 1 - current if current in {0, 1} else float(current) + 1.0
                else:
                    wrong_target_args["status"] = "counterfactual_wrong"
            elif branch_rows:
                augmented_case = copy.deepcopy(dict(case))
                healthy = copy.deepcopy(branch_rows[0]) if isinstance(branch_rows[0], Mapping) else {}
                healthy = dict(healthy)
                healthy["branch_id"] = "__counterfactual_healthy_branch__"
                augmented_case["branch"] = [*copy.deepcopy(branch_rows), healthy]
                wrong_target_args["case"] = augmented_case
                wrong_target_args["branch_id"] = "__counterfactual_healthy_branch__"
            else:
                wrong_target_args["case_updates"] = {"counterfactual_topology_target": "healthy"}
        else:
            wrong_target_args["target"] = "healthy_component"
        injected.append(InjectedAction("wrong_target_component", {"tool": tool, "arguments": wrong_target_args}))

        for family, scale in (("wrong_correction_magnitude", 10.0), ("wrong_correction_sign", -1.0)):
            variant = copy.deepcopy(args)
            for key in ("value", "correction", "delta", "multiplier"):
                if isinstance(variant.get(key), (int, float)):
                    variant[key] = float(variant[key]) * scale
            updates = variant.get("measurement_updates")
            if isinstance(updates, Mapping):
                variant["measurement_updates"] = {
                    key: float(value) * scale if isinstance(value, (int, float)) else value
                    for key, value in updates.items()
                }
            if tool == CORRECT_TOPOLOGY and variant.get("status") is not None:
                status = variant["status"]
                if isinstance(status, (int, float)):
                    variant["status"] = 2.0 if family == "wrong_correction_magnitude" else -1.0
                else:
                    variant["status"] = f"wrong_{status}"
            injected.append(InjectedAction(family, {"tool": tool, "arguments": variant}))
        injected.append(
            InjectedAction(
                "skipped_verification",
                {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": "__candidate__"}},
                (copy.deepcopy(action),),
            )
        )
        injected.append(
            InjectedAction(
                "rollback_of_valid_partial_correction",
                {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": "__candidate__"}},
                (
                    copy.deepcopy(action),
                    {"tool": "run_wls", "arguments": {"state_id": "__candidate__"}},
                ),
            )
        )
    stale = next((item.action for item in injected if item.family == "stale_context"), None)
    if stale is not None:
        injected.append(
            InjectedAction(
                "repeated_failed_action",
                copy.deepcopy(stale),
                (copy.deepcopy(stale),),
            )
        )
    return injected
