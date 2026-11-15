"""The six actions and the execution contract that checks a call.

The check is of form only: the tool exists, the arguments have the declared
types, and every object named (a version, a line, a meter, a breaker) exists
in the state the call acts on.  It never asks whether the call is a sensible
diagnosis: no context has to precede a correction, no list of supported
targets is consulted, no state of the estimator closes a tool.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

INSPECT = "inspect"
CORRECT_TOPOLOGY = "correct_topology"
CORRECT_PARAMETERS = "correct_parameters"
CORRECT_MEASUREMENTS = "correct_measurements"
UNDO = "undo"
FINISH = "finish"
TOOLS = (INSPECT, CORRECT_TOPOLOGY, CORRECT_PARAMETERS, CORRECT_MEASUREMENTS, UNDO, FINISH)
CORRECTION_TOOLS = (CORRECT_TOPOLOGY, CORRECT_PARAMETERS, CORRECT_MEASUREMENTS)

VIEW_BREAKERS = "breakers"
VIEW_ESTIMATE = "estimate"
VIEWS = (VIEW_BREAKERS, VIEW_ESTIMATE)
BREAKER_STATUSES = ("open", "closed")
LINE_STATUSES = ("in", "out")
OUTCOME_RESOLVED = "resolved"
OUTCOME_UNRESOLVED = "unresolved"
OUTCOMES = (OUTCOME_RESOLVED, OUTCOME_UNRESOLVED)

#: Error codes of the execution contract.  None of them is a judgement of the diagnosis.
ERROR_UNPARSEABLE = "unparseable_action"
ERROR_UNKNOWN_TOOL = "unknown_tool"
ERROR_ARGUMENTS = "invalid_arguments"
ERROR_UNKNOWN_OBJECT = "unknown_object"
ERROR_NO_CHANGE = "no_change"
ERROR_TOOL = "tool_failure"


class ActionError(ValueError):
    """A call the execution contract refuses; ``code`` is one of the ``ERROR_*`` values."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "description": f"Arguments of {name}.", "properties": properties,
                       "required": required, "additionalProperties": False},
    }}


def tool_schemas(*, breakers: bool, max_flips: int = 8) -> list[dict[str, Any]]:
    """The tools as the model sees them.  ``breakers`` says whether the system has a breaker model."""
    version = {"type": "integer", "description": "A saved version of the state; the current one when omitted."}
    inspect_properties: dict[str, Any] = {
        "view": {"type": "string", "enum": [VIEW_BREAKERS, VIEW_ESTIMATE] if breakers else [VIEW_ESTIMATE],
                 "description": (
                     "'estimate': the state estimate of a version, with its tables of buses, lines and meters. "
                     + ("'breakers': the substation estimate of a version, with its table of every breaker."
                        if breakers else ""))},
        "version": version,
    }
    if breakers:
        inspect_properties["flip"] = {
            "type": "array", "items": {"type": "string", "description": "Breaker name."}, "maxItems": int(max_flips),
            "description": "With view 'breakers': breakers to re-estimate with their status flipped, by name.",
        }
    if breakers:
        topology = _function(
            CORRECT_TOPOLOGY,
            "Set the reported status of one breaker. The operator model is rebuilt from the breaker statuses, the "
            "state is re-estimated, and the changes and the new estimate are returned.",
            {"breaker": {"type": "string", "description": "Breaker name."},
             "status": {"type": "string", "enum": list(BREAKER_STATUSES), "description": "The status to report."}},
            ["breaker", "status"],
        )
    else:
        topology = _function(
            CORRECT_TOPOLOGY,
            "Set the service status of one line. The state is re-estimated, and the changes and the new estimate "
            "are returned.",
            {"line": {"type": "integer", "description": "Line number."},
             "status": {"type": "string", "enum": list(LINE_STATUSES), "description": "In service or out of service."}},
            ["line", "status"],
        )
    return [
        _function(
            INSPECT,
            "Compute and return numerical evidence about a saved version of the state. It changes nothing.",
            inspect_properties, ["view"],
        ),
        topology,
        _function(
            CORRECT_PARAMETERS,
            "Re-fit the series R and X of one line from the repeated measurement scans. The fitted values replace "
            "the model's, the state is re-estimated, and the changes and the new estimate are returned.",
            {"line": {"type": "integer", "description": "Line number."}}, ["line"],
        ),
        _function(
            CORRECT_MEASUREMENTS,
            "Repair the named measurements as one group: the tool computes their corrected values from the "
            "current estimate. The state is re-estimated, and the changes and the new estimate are returned.",
            {"meters": {"type": "array", "items": {"type": "integer", "description": "Measurement index."}, "minItems": 1,
                        "description": "Measurement indices."}}, ["meters"],
        ),
        _function(
            UNDO,
            "Return to a saved version of the state. Corrections that are not on that version's path are no longer "
            "in effect.",
            {"version": {"type": "integer",
                         "description": "The version to return to; the one before the latest correction in effect "
                                        "when omitted."}}, [],
        ),
        _function(
            FINISH,
            "End the episode with the current version as the final state.",
            {"outcome": {"type": "string", "enum": list(OUTCOMES),
                         "description": "'resolved': every error is corrected. 'unresolved': errors remain."}},
            ["outcome"],
        ),
    ]


def parse_action(raw: Any) -> dict[str, Any]:
    """A model output or a mapping as ``{"tool": name, "arguments": {...}}``; ``ActionError`` when it is neither."""
    value = raw
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ActionError(ERROR_UNPARSEABLE, "not text") from exc
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ActionError(ERROR_UNPARSEABLE, "not a JSON object") from exc
    if not isinstance(value, Mapping):
        raise ActionError(ERROR_UNPARSEABLE, "not a JSON object")
    function = value.get("function") if isinstance(value.get("function"), Mapping) else {}
    tool = value.get("tool") or value.get("name") or function.get("name")
    arguments = value.get("arguments", function.get("arguments"))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ActionError(ERROR_UNPARSEABLE, "arguments are not a JSON object") from exc
    if arguments is None:
        arguments = {}
    if not isinstance(tool, str) or not tool:
        raise ActionError(ERROR_UNPARSEABLE, "no tool name")
    if not isinstance(arguments, Mapping):
        raise ActionError(ERROR_UNPARSEABLE, "arguments are not a JSON object")
    return {"tool": tool, "arguments": dict(arguments)}


def _integer(arguments: Mapping[str, Any], key: str, *, required: bool = True) -> int | None:
    value = arguments.get(key)
    if value is None:
        if required:
            raise ActionError(ERROR_ARGUMENTS, f"'{key}' is required")
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionError(ERROR_ARGUMENTS, f"'{key}' must be an integer")
    return int(value)


def _choice(arguments: Mapping[str, Any], key: str, choices: tuple[str, ...]) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or value not in choices:
        raise ActionError(ERROR_ARGUMENTS, f"'{key}' must be one of {', '.join(choices)}")
    return value


def _only(arguments: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    extra = sorted(str(key) for key in arguments if key not in allowed)
    if extra:
        raise ActionError(ERROR_ARGUMENTS, f"unknown argument {', '.join(extra)}")


def validate(action: Mapping[str, Any], *, breakers: bool, max_flips: int = 8) -> dict[str, Any]:
    """The call with its arguments typed and in canonical form; ``ActionError`` when the form is wrong.

    Whether the named objects exist is the environment's check, because it
    depends on the state the call acts on.
    """
    tool = action.get("tool")
    arguments = action.get("arguments") or {}
    if tool not in TOOLS:
        raise ActionError(ERROR_UNKNOWN_TOOL, str(tool))
    if tool == INSPECT:
        _only(arguments, ("view", "version", "flip"))
        view = _choice(arguments, "view", VIEWS if breakers else (VIEW_ESTIMATE,))
        clean: dict[str, Any] = {"view": view}
        version = _integer(arguments, "version", required=False)
        if version is not None:
            clean["version"] = version
        flips = arguments.get("flip")
        if flips is not None:
            if view != VIEW_BREAKERS:
                raise ActionError(ERROR_ARGUMENTS, "'flip' belongs to view 'breakers'")
            if not isinstance(flips, (list, tuple)) or not all(isinstance(name, str) and name for name in flips):
                raise ActionError(ERROR_ARGUMENTS, "'flip' must be a list of breaker names")
            if len(flips) > int(max_flips):
                raise ActionError(ERROR_ARGUMENTS, f"'flip' names at most {int(max_flips)} breakers")
            if flips:
                clean["flip"] = list(dict.fromkeys(str(name) for name in flips))
        return {"tool": tool, "arguments": clean}
    if tool == CORRECT_TOPOLOGY:
        if breakers:
            _only(arguments, ("breaker", "status"))
            name = arguments.get("breaker")
            if not isinstance(name, str) or not name:
                raise ActionError(ERROR_ARGUMENTS, "'breaker' must be a breaker name")
            return {"tool": tool, "arguments": {"breaker": name, "status": _choice(arguments, "status", BREAKER_STATUSES)}}
        _only(arguments, ("line", "status"))
        return {"tool": tool, "arguments": {"line": _integer(arguments, "line"),
                                            "status": _choice(arguments, "status", LINE_STATUSES)}}
    if tool == CORRECT_PARAMETERS:
        _only(arguments, ("line",))
        return {"tool": tool, "arguments": {"line": _integer(arguments, "line")}}
    if tool == CORRECT_MEASUREMENTS:
        _only(arguments, ("meters",))
        meters = arguments.get("meters")
        if (not isinstance(meters, (list, tuple)) or not meters
                or any(isinstance(index, bool) or not isinstance(index, int) for index in meters)):
            raise ActionError(ERROR_ARGUMENTS, "'meters' must be a non-empty list of measurement indices")
        if len(set(meters)) != len(meters):
            raise ActionError(ERROR_ARGUMENTS, "'meters' names a measurement twice")
        return {"tool": tool, "arguments": {"meters": sorted(int(index) for index in meters)}}
    if tool == UNDO:
        _only(arguments, ("version",))
        version = _integer(arguments, "version", required=False)
        return {"tool": tool, "arguments": {} if version is None else {"version": version}}
    _only(arguments, ("outcome",))
    return {"tool": tool, "arguments": {"outcome": _choice(arguments, "outcome", OUTCOMES)}}


__all__ = [
    "ActionError", "BREAKER_STATUSES", "CORRECTION_TOOLS", "CORRECT_MEASUREMENTS", "CORRECT_PARAMETERS",
    "CORRECT_TOPOLOGY", "ERROR_ARGUMENTS", "ERROR_NO_CHANGE", "ERROR_TOOL", "ERROR_UNKNOWN_OBJECT",
    "ERROR_UNKNOWN_TOOL", "ERROR_UNPARSEABLE", "FINISH", "INSPECT", "LINE_STATUSES", "OUTCOMES",
    "OUTCOME_RESOLVED", "OUTCOME_UNRESOLVED", "TOOLS", "UNDO", "VIEWS", "VIEW_BREAKERS", "VIEW_ESTIMATE",
    "parse_action", "tool_schemas", "validate",
]
