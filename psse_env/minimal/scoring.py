"""The training and evaluation layer: what the truth says about a finished episode.

Nothing in this module runs while the model acts, and nothing it computes
reaches an observation.  It is called on a finished episode with the scenario's
audit part, and answers three separate questions that the environment never
mixes up:

* what the model **declared** (``resolved``, ``unresolved``, or nothing: the
  budget ran out);
* whether the **estimator passes** on the final state;
* whether the errors were **actually repaired** and whether a healthy
  component was **actually damaged**, read from the final state itself.

The last question is answered twice.

``physical``
    The final state compared with the clean one, object by object, with the
    tolerances the scenario declares (``audit.release_audit.tolerances``): a
    true error is repaired when its meter reading, line R and X, or status
    ends within tolerance of the clean value; a healthy meter or line is
    damaged when it ends outside the same tolerance.  This is what the reward
    reads: damage is what changed, not which target was named.

``strict``
    The audit every earlier result of the project was scored by
    (``release_audit.audit_truth_audited_task_success``), fed the corrections
    in effect at the end as its ledger.  It additionally fails an episode in
    which a correction in effect names a healthy target, whatever the change
    did.  Kept so that a number from this environment can stand beside the
    old ones.

``episode_reward`` is the first version's reward: the repair outcome, less the
actual damage, a false claim of resolution, and a small cost per action.  It
has no term for following a workflow, because there is none.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np

from psse_env.minimal.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
)

SCORING_CONTRACT = "minimal_scoring_v1"
BR_R, BR_X, BR_STATUS = 2, 3, 10
STRUCTURAL_EFFECTS = frozenset({"bus_split", "merge"})


def _close(observed: Any, expected: Any, absolute: float, relative: float) -> bool:
    from psse_env.dagger.release_audit import _values_close

    return bool(_values_close(observed, expected, abs_tolerance=absolute, rel_tolerance=relative))


def audit_scenario(root: Mapping[str, Any]) -> dict[str, Any]:
    """The scenario with its truth, as the audit reads it.  Never handed to an environment."""
    from psse_env.dagger.evaluator import _strict_audit_scenario

    return _strict_audit_scenario(root)


def truth_targets(root: Mapping[str, Any]) -> dict[str, Any]:
    """The injected errors: meters by index, lines by number, breakers by name."""
    truth = ((root.get("audit") or {}).get("truth") or {}) if "audit" in root else root
    return {
        "meters": sorted(int(item["index"]) for item in truth.get("true_measurement_errors") or []
                         if item.get("index") is not None),
        "lines": sorted(int(item["line_index1"]) for item in truth.get("true_parameter_errors") or []),
        "breakers": sorted(str(item["cb_name"]) for item in truth.get("true_topology_errors") or [] if item.get("cb_name")),
        "status_lines": sorted(int(item["line_index1"]) for item in truth.get("true_topology_errors") or []
                               if item.get("line_index1") is not None),
    }


def _reported_closed(metadata: Any, breaker: str) -> bool | None:
    from psse_env.dagger.release_audit import _reported_breaker_status_value

    value = _reported_breaker_status_value(metadata, breaker)
    return None if value is None else bool(value)


def _branch(case: Any) -> np.ndarray:
    from psse_env.dagger.release_factories import deterministic_case_loader

    loaded = case if isinstance(case, Mapping) and "branch" in case else deterministic_case_loader(case)
    return np.asarray(loaded["branch"], dtype=float)


def _bus_count(case: Any) -> int:
    from psse_env.dagger.release_factories import deterministic_case_loader

    loaded = case if isinstance(case, Mapping) and "bus" in case else deterministic_case_loader(case)
    return int(np.asarray(loaded["bus"]).shape[0])


def physical_comparison(scenario: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, Any]:
    """The final state against the clean one, object by object.

    ``scenario`` is ``audit_scenario(root)``.  Returns each injected error
    with whether it is repaired, and every healthy object that ended outside
    tolerance.
    """
    from psse_env.dagger.release_audit import _coerce_tolerances, _not_applicable_parameter_targets

    declared = (scenario.get("release_audit") or {}).get("tolerances") if isinstance(scenario.get("release_audit"), Mapping) else None
    tolerance = _coerce_tolerances(declared)
    clean_z = [float(value) for value in scenario.get("clean_measurements") or []]
    final_z = [float(value) for value in final.get("measurements") or []]
    clean_branch = _branch(scenario["clean_case"])
    final_branch = _branch(final["case"])
    errors: list[dict[str, Any]] = []
    damage: list[dict[str, Any]] = []

    comparable_meters = len(clean_z) == len(final_z)
    true_meters = {int(item["index"]) for item in scenario.get("true_measurement_errors") or [] if item.get("index") is not None}
    for index in sorted(true_meters):
        repaired = bool(comparable_meters and index < len(final_z)
                        and _close(final_z[index], clean_z[index], tolerance.measurement_abs, tolerance.measurement_rel))
        errors.append({"kind": "meter", "meter": index, "repaired": repaired})
    if comparable_meters:
        for index, (observed, expected) in enumerate(zip(final_z, clean_z)):
            if index not in true_meters and not _close(observed, expected, tolerance.measurement_abs, tolerance.measurement_rel):
                damage.append({"kind": "meter", "meter": index, "final": observed, "clean": expected})

    comparable_lines = clean_branch.shape[0] == final_branch.shape[0]
    skipped = {int(target[1]) for target in _not_applicable_parameter_targets(scenario) if target[0] == "branch_row0"}
    true_lines: set[int] = set()
    for item in scenario.get("true_parameter_errors") or []:
        row = int(item["branch_row0"]) if item.get("branch_row0") is not None else int(item["line_index1"]) - 1
        true_lines.add(row)
        if row in skipped:
            errors.append({"kind": "line", "line": row + 1, "repaired": True, "not_applicable": "line_isolated_by_breaker"})
            continue
        clean_r = item.get("clean_r", clean_branch[row, BR_R] if comparable_lines else None)
        clean_x = item.get("clean_x", clean_branch[row, BR_X] if comparable_lines else None)
        repaired = bool(comparable_lines and all(
            _close(final_branch[row, column], clean, tolerance.final_case_abs, tolerance.final_case_rel)
            for column, clean in ((BR_R, clean_r), (BR_X, clean_x))))
        errors.append({"kind": "line", "line": row + 1, "repaired": repaired,
                       "final": [float(final_branch[row, BR_R]), float(final_branch[row, BR_X])] if comparable_lines else None,
                       "clean": [float(clean_r), float(clean_x)]})
    status_rows: set[int] = set()
    structural_pending = False
    final_metadata = final.get("metadata") or {}
    for item in scenario.get("true_topology_errors") or []:
        name = str(item["cb_name"]) if item.get("cb_name") else None
        effect = str(item.get("physical_effect") or "")
        record: dict[str, Any] = {"kind": "breaker" if name else "line_status"}
        if name:
            record["breaker"] = name
            reported = _reported_closed(final_metadata, name)
            record["breaker_status_right"] = bool(reported is not None and reported == bool(item.get("true_cb_closed")))
        row = int(item["branch_row0"]) if item.get("branch_row0") is not None else (
            int(item["line_index1"]) - 1 if item.get("line_index1") is not None else None)
        if effect in STRUCTURAL_EFFECTS or row is None:
            # A bus split or merge has no line of its own: the breaker's reported status is the object.
            record["repaired"] = bool(record.get("breaker_status_right"))
            structural_pending = structural_pending or not record["repaired"]
        else:
            status_rows.add(row)
            record["line"] = row + 1
            expected = item.get("expected_status", clean_branch[row, BR_STATUS] if comparable_lines else None)
            record["repaired"] = bool(comparable_lines and expected is not None
                                      and float(final_branch[row, BR_STATUS]) == float(expected))
        errors.append(record)
    if comparable_lines:
        for row in range(final_branch.shape[0]):
            if row not in true_lines and not all(
                    _close(final_branch[row, column], clean_branch[row, column], tolerance.final_case_abs, tolerance.final_case_rel)
                    for column in (BR_R, BR_X)):
                damage.append({"kind": "line", "line": row + 1,
                               "final": [float(final_branch[row, BR_R]), float(final_branch[row, BR_X])],
                               "clean": [float(clean_branch[row, BR_R]), float(clean_branch[row, BR_X])]})
            if (row not in status_rows and final_branch.shape[1] > BR_STATUS and clean_branch.shape[1] > BR_STATUS
                    and float(final_branch[row, BR_STATUS]) != float(clean_branch[row, BR_STATUS])):
                damage.append({"kind": "line_status", "line": row + 1})
            in_service = final_branch.shape[1] <= BR_STATUS or float(final_branch[row, BR_STATUS]) > 0
            if in_service and not structural_pending and (
                    int(final_branch[row, 0]) != int(clean_branch[row, 0]) or int(final_branch[row, 1]) != int(clean_branch[row, 1])):
                # A line in service between other buses than in the clean model: the bus sections are wrong.
                damage.append({"kind": "line_ends", "line": row + 1})
    buses_final, buses_clean = _bus_count(final["case"]), _bus_count(scenario["clean_case"])
    if buses_final != buses_clean and not structural_pending:
        damage.append({"kind": "operator_model", "buses_final": buses_final, "buses_clean": buses_clean})
    repaired = sum(1 for item in errors if item["repaired"])
    return {
        "errors": errors,
        "true_errors": len(errors),
        "repaired": repaired,
        "repaired_all": repaired == len(errors),
        "damage": damage,
        "damaged": len(damage),
        "solved": bool(repaired == len(errors) and not damage),
    }


def correction_ledger(modifications: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The corrections in effect in the form the strict audit's ledger takes."""
    ledger: list[dict[str, Any]] = []
    for record in modifications:
        arguments = record.get("arguments") or {}
        tool = record.get("tool")
        if tool == CORRECT_MEASUREMENTS:
            action = {"tool": tool, "arguments": {"suspect_group": [int(index) for index in arguments["meters"]]}}
        elif tool == CORRECT_PARAMETERS:
            action = {"tool": tool, "arguments": {"line_index": int(arguments["line"])}}
        elif tool == CORRECT_TOPOLOGY and arguments.get("breaker") is not None:
            old: dict[str, Any] = {"cb_name": str(arguments["breaker"]), "status": int(arguments["status"] == "closed")}
            lines = [change["line"] for change in record.get("changes") or []
                     if change.get("object") == "line" and change.get("field") == "status"]
            if len(lines) == 1:
                old["line_index"] = int(lines[0])
            action = {"tool": tool, "arguments": old}
        elif tool == CORRECT_TOPOLOGY:
            action = {"tool": tool, "arguments": {"line_index": int(arguments["line"]), "status": int(arguments["status"] == "in")}}
        else:
            continue
        ledger.append({"source_action": action})
    return ledger


def strict_audit(scenario: Mapping[str, Any], final: Mapping[str, Any], modifications: Sequence[Mapping[str, Any]],
                 *, declared: str | None) -> dict[str, Any]:
    """The project's truth audit of the final state, with the corrections in effect as its ledger."""
    from psse_env.dagger.release_audit import audit_truth_audited_task_success
    from psse_env.dagger.release_factories import deterministic_case_loader

    assessment = audit_truth_audited_task_success(
        scenario,
        {"accepted_corrections": correction_ledger(modifications), "explained_anomalies": []},
        actual_terminal=declared in (OUTCOME_RESOLVED, OUTCOME_UNRESOLVED),
        actual_terminal_outcome="resolved" if declared == OUTCOME_RESOLVED else None,
        active_physical_state={"case": final["case"], "measurements": list(final["measurements"]),
                               "metadata": copy.deepcopy(final.get("metadata") or {})},
        case_loader=deterministic_case_loader,
    )
    return {"solved": bool(assessment.get("eligible")), "status": assessment.get("status"),
            "reasons": [str(reason) for reason in assessment.get("reasons") or []]}


def estimator_passes(estimate: Mapping[str, Any]) -> bool:
    """Both tests of the final state's estimate are below their thresholds."""
    if not estimate.get("converged"):
        return False
    passes = float(estimate["chi_square"]) < float(estimate["chi_square_threshold"])
    if estimate.get("residual_threshold") is not None:
        passes = passes and float(estimate["max_residual"]) < float(estimate["residual_threshold"])
    return bool(passes)


def score_episode(root: Mapping[str, Any], *, final: Mapping[str, Any], modifications: Sequence[Mapping[str, Any]],
                  declared: str | None, estimate: Mapping[str, Any], history: Sequence[Mapping[str, Any]] = (),
                  cost: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score a finished episode against the scenario's truth."""
    scenario = audit_scenario(root)
    physical = physical_comparison(scenario, final)
    strict = strict_audit(scenario, final, modifications, declared=declared)
    errors = sum(1 for event in history if (event.get("result") or {}).get("status") != "ok")
    return {
        "contract": SCORING_CONTRACT,
        "scenario_id": str((root.get("execution") or {}).get("scenario_id") or scenario.get("scenario_id") or ""),
        "class": ((root.get("audit") or {}).get("compound") or {}).get("class"),
        "declared": declared,
        "estimator_passes": estimator_passes(estimate),
        "solved": physical["solved"],
        "success": bool(physical["solved"] and declared == OUTCOME_RESOLVED),
        "physical": physical,
        "strict": strict,
        "actions": len(history),
        "tool_errors": errors,
        "corrections_in_effect": len(modifications),
        "cost": dict(cost or {}),
    }


def score_env(root: Mapping[str, Any], env: Any) -> dict[str, Any]:
    """Score the episode an environment has just finished."""
    declared = env.outcome if env.outcome in (OUTCOME_RESOLVED, OUTCOME_UNRESOLVED) else None
    return score_episode(root, final=env.final_state(), modifications=env.modifications_in_effect(), declared=declared,
                         estimate=env.estimate(), history=env.history(), cost=env.cost())


@dataclass(frozen=True)
class RewardWeights:
    """The first version's reward.  Outcome first; the costs are small on purpose.

    With these defaults a solved and declared episode earns at least
    ``success - action_cost * budget`` (0.84 at 16 actions), more than
    stopping at once (``-action_cost``), so trying is never priced out.
    """

    success: float = 1.0
    #: The state is repaired but the model did not say so (it declared ``unresolved`` or ran out of actions).
    unclaimed_success: float = 0.5
    #: ``resolved`` declared on a state that is not.
    false_resolved: float = 0.5
    #: Per healthy object left outside tolerance, up to ``damage_cap`` in all.
    damage: float = 0.25
    damage_cap: float = 0.5
    action_cost: float = 0.01
    #: Per call the execution contract refused, on top of its action cost.
    tool_error: float = 0.02
    #: Per second of numerical tool time; off until efficiency is the question.
    tool_second: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RewardWeights":
        value = dict(value or {})
        unknown = sorted(set(value) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown reward weights: {unknown}")
        return cls(**{key: float(item) for key, item in value.items()})


def episode_reward(score: Mapping[str, Any], weights: RewardWeights | None = None) -> dict[str, Any]:
    """The scalar reward of a scored episode and its terms."""
    weights = weights or RewardWeights()
    solved = bool(score["solved"])
    declared = score.get("declared")
    outcome = weights.success if solved and declared == OUTCOME_RESOLVED else weights.unclaimed_success if solved else 0.0
    false_claim = weights.false_resolved if declared == OUTCOME_RESOLVED and not solved else 0.0
    damage = min(weights.damage * int(score["physical"]["damaged"]), weights.damage_cap)
    actions = weights.action_cost * int(score.get("actions") or 0)
    errors = weights.tool_error * int(score.get("tool_errors") or 0)
    seconds = weights.tool_second * float((score.get("cost") or {}).get("tool_seconds") or 0.0)
    total = outcome - false_claim - damage - actions - errors - seconds
    return {"total": float(total), "outcome": float(outcome), "false_resolved": float(false_claim), "damage": float(damage),
            "action_cost": float(actions), "tool_error_cost": float(errors), "tool_time_cost": float(seconds),
            "weights": asdict(weights)}


__all__ = [
    "RewardWeights", "SCORING_CONTRACT", "audit_scenario", "correction_ledger", "episode_reward", "estimator_passes",
    "physical_comparison", "score_env", "score_episode", "strict_audit", "truth_targets",
]
