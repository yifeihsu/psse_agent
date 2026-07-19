"""Bidirectional mapping between controller actions and the canonical tool surface.

The transactional DAgger environment speaks an internal controller protocol
(``run_wls``/``state_id``/``commit_state``).  The production SFT corpus and the
deployment agent speak the canonical power-tool protocol defined by
``trace_protocol.CANONICAL_POWER_TOOLS`` (``wls_from_path``/``case_path``).
This module lets DAgger export and inference use one model-visible protocol —
the canonical one — without changing transaction semantics:

- ``internal_to_canonical_action`` converts an expert/controller action to the
  canonical tool name and argument keys for SFT export.  Correction *values*
  are intentionally dropped: canonical correction tools compute the corrected
  values themselves, so the model is only supervised on target selection.
- ``canonical_to_internal_action`` converts a model-generated canonical call
  back to the controller protocol for env execution and alias binding.
  Canonical-only diagnostic tools (harmonics, HSE, three-phase NLM, HIF
  estimators) pass through unchanged; the environment's process-validity gate
  turns them into standardized no-op transitions until executors exist.

Note: reverse-mapped ``correct_*_from_path`` calls carry a target but no
replacement value.  Executing them requires deployment correction providers
that hydrate values (the same contract as the production runtime wrapper);
the deterministic pilot adapters do not support that yet.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    FINALIZE_DIAGNOSIS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    INVALID_ACTION,
    ROLLBACK_STATE,
    RUN_ALTERNATIVE_TEST,
    RUN_WLS,
    VERIFY_CANDIDATE,
    safe_normalize_action,
)

try:  # The canonical registry lives at the repository root.
    from trace_protocol import CANONICAL_POWER_TOOLS
except ImportError:  # pragma: no cover - archive layouts without the root module.
    CANONICAL_POWER_TOOLS = None


WLS_FROM_PATH = "wls_from_path"
GET_VERIFICATION_SNAPSHOT = "get_verification_snapshot"
CORRECT_MEASUREMENTS_FROM_PATH = "correct_measurements_from_path"
CORRECT_PARAMETERS_FROM_PATH = "correct_parameters_from_path"
CORRECT_TOPOLOGY_FROM_PATH = "correct_topology_from_path"

INTERNAL_TO_CANONICAL_TOOL: dict[str, str] = {
    RUN_WLS: WLS_FROM_PATH,
    VERIFY_CANDIDATE: GET_VERIFICATION_SNAPSHOT,
    GET_MEASUREMENT_CONTEXT: GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT: GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT: GET_TOPOLOGY_CONTEXT,
    CORRECT_MEASUREMENTS: CORRECT_MEASUREMENTS_FROM_PATH,
    CORRECT_PARAMETERS: CORRECT_PARAMETERS_FROM_PATH,
    CORRECT_TOPOLOGY: CORRECT_TOPOLOGY_FROM_PATH,
    COMMIT_STATE: COMMIT_STATE,
    ROLLBACK_STATE: ROLLBACK_STATE,
    FINALIZE_DIAGNOSIS: FINALIZE_DIAGNOSIS,
    ASK_FOR_MORE_EVIDENCE: ASK_FOR_MORE_EVIDENCE,
    RUN_ALTERNATIVE_TEST: RUN_ALTERNATIVE_TEST,
}
CANONICAL_TO_INTERNAL_TOOL: dict[str, str] = {
    canonical: internal for internal, canonical in INTERNAL_TO_CANONICAL_TOOL.items()
}

# Tools that reference the open candidate rather than the active state.
_CANDIDATE_REFERENCE_TOOLS = frozenset({COMMIT_STATE, ROLLBACK_STATE})
# Model-visible state-reference argument keys on the canonical surface.
CANONICAL_STATE_REFERENCE_KEYS = frozenset({"case_path", "scan_window_path"})

# Expert corrections carry replacement values; canonical correction tools
# compute values from the target, so these keys never reach the model.
_DROPPED_CORRECTION_VALUE_KEYS = frozenset(
    {
        "field",
        "parameter",
        "value",
        "corrected_value",
        "new_value",
        "multiplier",
        "status_field",
    }
)
_MEASUREMENT_TARGET_ALIAS_KEYS = frozenset(
    {"measurement_index", "index", "index0", "target", "meter", "measurement_id"}
)

_TRANSACTIONAL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": GET_MEASUREMENT_CONTEXT,
            "description": (
                "Retrieve observable measurement-residual context for bad-data "
                "correction on the referenced case."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                },
                "required": ["case_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": COMMIT_STATE,
            "description": "Commit a verified acceptable candidate case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {
                        "type": "string",
                        "description": "Candidate case identifier or path to commit.",
                    },
                },
                "required": ["case_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": ROLLBACK_STATE,
            "description": "Roll back a verified rejected or inconclusive candidate case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {
                        "type": "string",
                        "description": "Candidate case identifier or path to roll back.",
                    },
                },
                "required": ["case_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": FINALIZE_DIAGNOSIS,
            "description": (
                "Finish only after observable evidence shows no material anomaly remains."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": ASK_FOR_MORE_EVIDENCE,
            "description": "Request additional observable evidence for the referenced case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "request": {"type": "string", "description": "Evidence being requested."},
                },
                "required": ["case_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": RUN_ALTERNATIVE_TEST,
            "description": "Run an alternative observable diagnostic test on the referenced case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "test_name": {"type": "string", "description": "Diagnostic test to run."},
                },
                "required": ["case_path"],
            },
        },
    },
]

# Branch-target properties accepted by the transactional environment.  They are
# additive on top of the canonical schema so controller scenarios that identify
# a branch by id/breaker/zero-based row can still be exported and replayed; the
# environment enforces the exactly-one-target convention.
_ADDITIVE_BRANCH_TARGET_PROPERTIES: dict[str, Any] = {
    "branch_id": {"type": "string", "description": "Branch identifier."},
    "cb_name": {"type": "string", "description": "Breaker name."},
    "line_index1": {"type": "integer", "description": "1-based branch row index."},
    "branch_row0": {"type": "integer", "description": "0-based branch row index."},
}


def _require_registry() -> list[dict[str, Any]]:
    if CANONICAL_POWER_TOOLS is None:
        raise RuntimeError(
            "trace_protocol.CANONICAL_POWER_TOOLS is unavailable; the canonical "
            "protocol requires the repository-root trace_protocol module."
        )
    return copy.deepcopy(CANONICAL_POWER_TOOLS)


def unified_tool_schemas() -> list[dict[str, Any]]:
    """Return the canonical registry extended with transactional DAgger tools.

    Canonical schemas are preserved verbatim except for three deliberate,
    additive relaxations required by the transactional environment:

    - ``get_verification_snapshot`` gains an optional ``case_path`` so a
      verification can reference the open candidate explicitly.
    - ``correct_parameters_from_path`` and ``correct_topology_from_path`` gain
      the additive branch-target conventions and require only ``case_path``;
      the environment enforces exactly one branch target at execution time.
    """
    schemas = _require_registry()
    by_name = {schema["function"]["name"]: schema for schema in schemas}

    verification = by_name[GET_VERIFICATION_SNAPSHOT]["function"]["parameters"]
    verification["properties"].setdefault(
        "case_path",
        {"type": "string", "description": "Optional candidate case identifier to verify."},
    )
    for name in (CORRECT_PARAMETERS_FROM_PATH, CORRECT_TOPOLOGY_FROM_PATH):
        parameters = by_name[name]["function"]["parameters"]
        for key, prop in _ADDITIVE_BRANCH_TARGET_PROPERTIES.items():
            parameters["properties"].setdefault(key, copy.deepcopy(prop))
        parameters["required"] = ["case_path"]

    schemas.extend(copy.deepcopy(_TRANSACTIONAL_TOOL_SCHEMAS))
    return schemas


def canonical_tool_names() -> set[str]:
    return {schema["function"]["name"] for schema in unified_tool_schemas()}


def _normalize_or_raise(action: Mapping[str, Any] | str) -> dict[str, Any]:
    normalized = safe_normalize_action(action)
    if normalized["tool"] == INVALID_ACTION:
        raise ValueError(f"Cannot bridge malformed action: {normalized['arguments']}")
    return {"tool": normalized["tool"], "arguments": copy.deepcopy(normalized["arguments"])}


def _move_key(arguments: dict[str, Any], source: str, destination: str) -> None:
    if source in arguments and arguments[source] is not None:
        arguments[destination] = arguments.pop(source)
    else:
        arguments.pop(source, None)


def internal_to_canonical_action(action: Mapping[str, Any] | str) -> dict[str, Any]:
    """Convert a controller action to the canonical model-visible protocol."""
    normalized = _normalize_or_raise(action)
    tool = normalized["tool"]
    arguments = normalized["arguments"]

    if tool not in INTERNAL_TO_CANONICAL_TOOL:
        if tool in CANONICAL_TO_INTERNAL_TOOL or tool in canonical_tool_names():
            # Already canonical (for example a canonical-only diagnostic tool).
            return normalized
        raise ValueError(f"No canonical mapping for controller tool: {tool}")

    canonical = INTERNAL_TO_CANONICAL_TOOL[tool]
    if tool in _CANDIDATE_REFERENCE_TOOLS:
        _move_key(arguments, "candidate_state_id", "case_path")
    else:
        _move_key(arguments, "state_id", "case_path")

    if tool == CORRECT_MEASUREMENTS:
        updates = arguments.pop("measurement_updates", None)
        for alias_key in _MEASUREMENT_TARGET_ALIAS_KEYS:
            arguments.pop(alias_key, None)
        if isinstance(updates, Mapping):
            arguments["suspect_group"] = sorted(int(index) for index in updates)
        elif "suspect_group" not in arguments:
            raise ValueError(
                "correct_measurements requires measurement_updates or suspect_group."
            )
    elif tool == CORRECT_PARAMETERS:
        if arguments.get("branch_row0") is not None:
            arguments["line_index"] = int(arguments.pop("branch_row0")) + 1
        elif arguments.get("line_index1") is not None:
            arguments["line_index"] = int(arguments.pop("line_index1"))
        for key in _DROPPED_CORRECTION_VALUE_KEYS:
            arguments.pop(key, None)
    elif tool == CORRECT_TOPOLOGY:
        status = arguments.pop("status", arguments.pop("expected_status", None))
        if status is not None:
            arguments["desired_status"] = bool(int(status))
        arguments.pop("expected_status", None)
        for key in _DROPPED_CORRECTION_VALUE_KEYS:
            arguments.pop(key, None)

    return {"tool": canonical, "arguments": arguments}


def canonical_to_internal_action(action: Mapping[str, Any] | str) -> dict[str, Any]:
    """Convert a canonical model tool call back to the controller protocol.

    Canonical-only diagnostic tools are returned unchanged so the environment's
    process-validity gate can record them as standardized no-op transitions
    until real executors are integrated.
    """
    normalized = _normalize_or_raise(action)
    tool = normalized["tool"]
    arguments = normalized["arguments"]

    if tool not in CANONICAL_TO_INTERNAL_TOOL:
        return normalized

    internal = CANONICAL_TO_INTERNAL_TOOL[tool]
    if internal in _CANDIDATE_REFERENCE_TOOLS:
        _move_key(arguments, "case_path", "candidate_state_id")
    elif internal == VERIFY_CANDIDATE:
        _move_key(arguments, "case_path", "state_id")
        # A stage-only verification references the open candidate by alias.
        arguments.setdefault("state_id", "candidate")
    else:
        _move_key(arguments, "case_path", "state_id")

    if internal == CORRECT_TOPOLOGY:
        desired = arguments.pop("desired_status", None)
        if desired is not None:
            arguments["status"] = int(bool(desired))

    return safe_normalize_action({"tool": internal, "arguments": arguments})


__all__ = [
    "CANONICAL_STATE_REFERENCE_KEYS",
    "CANONICAL_TO_INTERNAL_TOOL",
    "INTERNAL_TO_CANONICAL_TOOL",
    "canonical_to_internal_action",
    "canonical_tool_names",
    "internal_to_canonical_action",
    "unified_tool_schemas",
]
