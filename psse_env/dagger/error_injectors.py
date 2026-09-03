from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from hif_search_limits import HIF_ALPHA_GRID_SIZE_MAX
from three_phase_nlm.ieee14_adapter import ELIGIBLE_HIF_BRANCHES

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    DIAGNOSTIC_TOOLS,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
    FINALIZE_DIAGNOSIS,
    GET_HARMONIC_CONTEXT,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    safe_normalize_action,
)


@dataclass(frozen=True)
class InjectedAction:
    family: str
    action: dict[str, Any]
    setup_actions: tuple[dict[str, Any], ...] = ()


#: Argument placeholders that the counterfactual generator binds after the
#: injected branch has executed its setup actions.  They resolve against the
#: branch's own observable NLM output, never against hidden truth.
NLM_TOP_BRANCH_PLACEHOLDER = "__nlm_top_branch_row0__"
NLM_WRONG_BRANCH_PLACEHOLDER = "__nlm_wrong_branch_row0__"
NLM_BRANCH_PLACEHOLDERS = frozenset(
    {NLM_TOP_BRANCH_PLACEHOLDER, NLM_WRONG_BRANCH_PLACEHOLDER}
)


def resolve_nlm_branch_placeholder(
    placeholder: str, nlm_top_branch_row0: int | None
) -> int:
    """Bind an NLM branch placeholder to an observable localized branch.

    The top-branch placeholder is the localized line itself.  The wrong-branch
    placeholder is the next eligible line after it, so the injected estimator
    call targets a healthy line without consulting hidden truth.  Without a
    localized branch both fall back to the first eligible line: the call stays
    schema-valid and the environment, not the generator, rejects it.
    """
    eligible = [int(row) for row in ELIGIBLE_HIF_BRANCHES]
    if placeholder not in NLM_BRANCH_PLACEHOLDERS:
        raise ValueError(f"unknown NLM branch placeholder {placeholder!r}")
    if nlm_top_branch_row0 is None:
        return eligible[0]
    top = int(nlm_top_branch_row0)
    if placeholder == NLM_TOP_BRANCH_PLACEHOLDER:
        return top
    if top in eligible:
        return eligible[(eligible.index(top) + 1) % len(eligible)]
    return next(row for row in eligible if row != top)


def diagnostic_wrong_actions(
    state: Mapping[str, Any],
    expert_actions: Iterable[Mapping[str, Any]] | None = None,
) -> list[InjectedAction]:
    """Plausible learner mistakes on the specialized-diagnostic ladder.

    Emitted only when the expert's current proposals include a diagnostic
    tool, so classical correction roots are unaffected.  Every action is
    built from the policy observation and the expert's observable proposals;
    branch placeholders are bound later from the injected branch's own NLM
    output.  The mistakes mirror the ladder's failure modes: estimating
    before localizing, running the wrong family's diagnostic, estimating on a
    healthy line, overrunning the bounded search budget, escalating before
    the configured estimators ran, and applying a fundamental-frequency
    correction that would mask an explanation-only anomaly.
    """
    active_id = state.get("active_state_id")
    available = {str(item) for item in (state.get("available_evidence") or [])}
    proposed: set[str] = set()
    for raw in expert_actions or []:
        try:
            proposed.add(safe_normalize_action(raw)["tool"])
        except Exception:
            continue
    diagnostic = proposed & DIAGNOSTIC_TOOLS
    if not diagnostic:
        return []
    estimator = (
        ESTIMATE_HIF_MULTISCAN_FROM_PATH
        if "hif_scan_window" in available
        else ESTIMATE_HIF_FROM_PATH
    )
    first_line = int(ELIGIBLE_HIF_BRANCHES[0])
    premature_escalation = InjectedAction(
        "premature_diagnostic_escalation",
        {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": active_id,
                "request": HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
            },
        },
    )
    injected: list[InjectedAction] = []
    if RUN_THREE_PHASE_NLM_FROM_PATH in diagnostic:
        localize = {
            "tool": RUN_THREE_PHASE_NLM_FROM_PATH,
            "arguments": {"state_id": "__active__"},
        }
        injected.extend(
            [
                InjectedAction(
                    "premature_hif_estimation",
                    {
                        "tool": estimator,
                        "arguments": {
                            "state_id": active_id,
                            "candidate_branch_row0": first_line,
                        },
                    },
                ),
                InjectedAction(
                    "wrong_diagnostic_family",
                    {"tool": RUN_HSE_FROM_PATH, "arguments": {"state_id": active_id}},
                ),
                InjectedAction(
                    "wrong_hif_candidate_branch",
                    {
                        "tool": estimator,
                        "arguments": {
                            "state_id": "__active__",
                            "candidate_branch_row0": NLM_WRONG_BRANCH_PLACEHOLDER,
                        },
                    },
                    (copy.deepcopy(localize),),
                ),
                InjectedAction(
                    "hif_search_budget_overrun",
                    {
                        "tool": estimator,
                        "arguments": {
                            "state_id": "__active__",
                            "candidate_branch_row0": NLM_TOP_BRANCH_PLACEHOLDER,
                            "alpha_grid_size": HIF_ALPHA_GRID_SIZE_MAX + 1,
                        },
                    },
                    (copy.deepcopy(localize),),
                ),
                premature_escalation,
                InjectedAction(
                    "masking_correction_on_diagnostic_anomaly",
                    {
                        "tool": CORRECT_PARAMETERS,
                        "arguments": {
                            "state_id": active_id,
                            "branch_row0": first_line,
                        },
                    },
                ),
            ]
        )
    elif diagnostic & {GET_HARMONIC_CONTEXT, RUN_HSE_FROM_PATH}:
        injected.extend(
            [
                InjectedAction(
                    "wrong_diagnostic_family",
                    {
                        "tool": RUN_THREE_PHASE_NLM_FROM_PATH,
                        "arguments": {"state_id": active_id},
                    },
                ),
                premature_escalation,
            ]
        )
    return injected


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
    injected.extend(diagnostic_wrong_actions(state, expert_actions))
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
