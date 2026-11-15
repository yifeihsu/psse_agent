"""The rule expert's decisions, executed through the minimal interface.

A baseline, and a check.  The rule expert decides inside the old environment:
it reads the old contexts, the old verifier's verdicts and the controller's
recovery hints, so its decision logic cannot run on the new observation.
What can be carried over without adding anything is what it *did*: every
state-changing decision of one of its episodes, replayed here step for step.

The translation is mechanical and adds no truth and no heuristic:

=============================  =============================================
old step (carried out)          minimal interface
=============================  =============================================
``correct_*``                   the same correction (the tool computes the values)
``rollback_state``              ``undo``
``get_topology_context``        ``inspect`` the breakers (where there is a breaker model)
``compare_parameter_candidates``  each compared line fitted and undone, in the request's order
``commit_state``                nothing: a correction stays in effect until undone
``run_wls``, ``verify_candidate``  nothing: every state is estimated when it is made
``get_measurement_context``,
``get_parameter_context``       nothing: those tables are part of every estimate
``finalize_diagnosis``          ``finish resolved``
hand-off                        ``finish``
=============================  =============================================

The old hand-off closes every episode with the same request whatever the
result, so it carries no claim to translate.  The one thing the translation
adds is that claim: ``resolved`` when the final state's estimate passes both
tests, ``unresolved`` otherwise.  It is read from the observation, never from
the truth, and the expert's reference number is ``solved`` (the final state),
which does not depend on it.

A step the old environment refused or whose tool failed changed nothing there
and is left out.  A correction with explicit measurement values has no
equivalent (the minimal interface lets the tool compute them) and makes the
episode untranslatable, which is reported, not patched.

Replayed, the translation shows two things: that the numerical tools give the
same final state through either interface (the expert's outcome must not
change), and what the expert's diagnosis costs in the new interface's actions.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    COMPARE_PARAMETER_CANDIDATES,
    FINALIZE_DIAGNOSIS,
    GET_TOPOLOGY_CONTEXT,
    POST_CORRECTION_CONFIRMATION_REQUEST,
    ROLLBACK_STATE,
)
from psse_env.actions import CORRECT_MEASUREMENTS as OLD_CORRECT_MEASUREMENTS
from psse_env.actions import CORRECT_PARAMETERS as OLD_CORRECT_PARAMETERS
from psse_env.actions import CORRECT_TOPOLOGY as OLD_CORRECT_TOPOLOGY
from psse_env.minimal.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    FINISH,
    INSPECT,
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    UNDO,
    VIEW_BREAKERS,
)

TRANSLATION_CONTRACT = "rule_expert_to_minimal_v1"


class Untranslatable(ValueError):
    """An old step with no mechanical equivalent in the minimal interface."""


def _step(step: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], bool]:
    """Tool, arguments and whether the step was carried out, from an evaluator trace step or a compact one."""
    action = step.get("action") if isinstance(step.get("action"), Mapping) else step
    tool = str(action.get("tool"))
    arguments = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
    status = step.get("execution_status", step.get("status"))
    return tool, arguments, status == "success"


def _line(arguments: Mapping[str, Any]) -> int:
    if arguments.get("line_index") is not None:
        return int(arguments["line_index"])
    if arguments.get("line_index1") is not None:
        return int(arguments["line_index1"])
    if arguments.get("branch_row0") is not None:
        return int(arguments["branch_row0"]) + 1
    raise Untranslatable("a line correction that names no line")


def _status(arguments: Mapping[str, Any]) -> bool:
    for key in ("status", "expected_status", "desired_status"):
        value = arguments.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value.strip().lower() in {"1", "closed", "close", "true", "in"}
        return bool(float(value))
    raise Untranslatable("a topology correction without a status")


def translate_correction(tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """One old correction as a correction of the minimal interface."""
    if tool == OLD_CORRECT_MEASUREMENTS:
        group = arguments.get("suspect_group")
        if not group or arguments.get("measurement_updates"):
            raise Untranslatable("a measurement correction with explicit values")
        return {"tool": CORRECT_MEASUREMENTS, "arguments": {"meters": sorted(int(index) for index in group)}}
    if tool == OLD_CORRECT_PARAMETERS:
        return {"tool": CORRECT_PARAMETERS, "arguments": {"line": _line(arguments)}}
    if tool == OLD_CORRECT_TOPOLOGY:
        if arguments.get("cb_name") is not None:
            return {"tool": CORRECT_TOPOLOGY, "arguments": {
                "breaker": str(arguments["cb_name"]), "status": "closed" if _status(arguments) else "open"}}
        return {"tool": CORRECT_TOPOLOGY, "arguments": {"line": _line(arguments), "status": "in" if _status(arguments) else "out"}}
    raise Untranslatable(f"not a correction: {tool}")


def translate_trace(trace: Sequence[Mapping[str, Any]], *, breakers: bool) -> list[dict[str, Any]]:
    """The minimal-interface actions of one old episode's policy steps, in order.

    ``trace`` holds the policy's steps (interventions removed), either as the
    evaluator wrote them or in the rollouts' compact form.  Raises
    ``Untranslatable`` for an episode that has no mechanical equivalent.
    """
    actions: list[dict[str, Any]] = []
    finished = False
    candidate_open = False
    for step in trace:
        tool, arguments, done = _step(step)
        if not done:
            continue
        if tool in (OLD_CORRECT_MEASUREMENTS, OLD_CORRECT_PARAMETERS, OLD_CORRECT_TOPOLOGY):
            actions.append(translate_correction(tool, arguments))
            candidate_open = True
        elif tool == ROLLBACK_STATE:
            actions.append({"tool": UNDO, "arguments": {}})
            candidate_open = False
        elif tool == COMMIT_STATE:
            candidate_open = False
        elif tool == GET_TOPOLOGY_CONTEXT and breakers:
            actions.append({"tool": INSPECT, "arguments": {"view": VIEW_BREAKERS}})
        elif tool == COMPARE_PARAMETER_CANDIDATES:
            for line in arguments.get("line_indices") or []:
                actions.append({"tool": CORRECT_PARAMETERS, "arguments": {"line": int(line)}, "trial": True})
                actions.append({"tool": UNDO, "arguments": {}, "trial": True})
        elif tool == FINALIZE_DIAGNOSIS:
            actions.append({"tool": FINISH, "arguments": {"outcome": OUTCOME_RESOLVED}})
            finished = True
        elif tool == ASK_FOR_MORE_EVIDENCE and str(arguments.get("request") or "").startswith("operator_escalation"):
            if arguments.get("request") == POST_CORRECTION_CONFIRMATION_REQUEST:
                actions.append({"tool": FINISH, "arguments": {"outcome": OUTCOME_RESOLVED}})
            else:
                actions.append({"tool": FINISH, "arguments": {}, "by_estimate": True})
            finished = True
        if finished:
            break
    if candidate_open and not finished:
        # The old episode ended on a candidate it never kept: the old active state did not hold it.
        actions.append({"tool": UNDO, "arguments": {}})
    return actions


def _declared_by_estimate(observation: Mapping[str, Any]) -> str:
    """``resolved`` when the current version's estimate passes both tests, read from the observation alone."""
    from psse_env.minimal.scoring import estimator_passes

    current = int(observation["current_version"])
    estimate = next(entry["estimate"] for entry in observation["versions"] if int(entry["version"]) == current)
    return OUTCOME_RESOLVED if estimator_passes(estimate) else OUTCOME_UNRESOLVED


class TranslatedExpertPolicy:
    """Replays a translated episode.

    A trial fit that fails leaves nothing to undo, so its ``undo`` is
    dropped.  The old hand-off made no claim about the result, so the
    declaration of a hand-off is read from the final estimate.
    """

    def __init__(self, actions: Sequence[Mapping[str, Any]]) -> None:
        self._actions = [dict(action) for action in actions]

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        history = observation.get("history") or []
        while self._actions:
            action = self._actions.pop(0)
            trial = bool(action.pop("trial", False))
            if trial and action["tool"] == UNDO:
                last = history[-1] if history else {}
                if (last.get("result") or {}).get("status") != "ok":
                    continue
            if action.pop("by_estimate", False):
                action["arguments"] = {"outcome": _declared_by_estimate(observation)}
            return action
        # The old episode ended without a terminal action (its budget ran out there).
        return {"tool": FINISH, "arguments": {"outcome": _declared_by_estimate(observation)}}


__all__ = ["TRANSLATION_CONTRACT", "TranslatedExpertPolicy", "Untranslatable", "translate_correction", "translate_trace"]
