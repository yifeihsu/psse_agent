from __future__ import annotations

import json
from typing import Any, Mapping


RUN_WLS = "run_wls"
VERIFY_CANDIDATE = "verify_candidate"
GET_MEASUREMENT_CONTEXT = "get_measurement_context"
GET_PARAMETER_CONTEXT = "get_parameter_context"
GET_TOPOLOGY_CONTEXT = "get_topology_context"
CORRECT_MEASUREMENTS = "correct_measurements"
CORRECT_PARAMETERS = "correct_parameters"
CORRECT_TOPOLOGY = "correct_topology"
COMMIT_STATE = "commit_state"
ROLLBACK_STATE = "rollback_state"
FINALIZE_DIAGNOSIS = "finalize_diagnosis"
ASK_FOR_MORE_EVIDENCE = "ask_for_more_evidence"
RUN_ALTERNATIVE_TEST = "run_alternative_test"
HIF_DIAGNOSTICS_EXHAUSTED_REQUEST = (
    "operator_escalation:hif_diagnostics_exhausted"
)
RECOVERY_OPTIONS_EXHAUSTED_REQUEST = (
    "operator_escalation:recovery_options_exhausted"
)
RECOVERY_BUDGET_EXHAUSTED_REQUEST = (
    "operator_escalation:recovery_budget_exhausted"
)
# An accepted correction can make the candidate WLS statistic quiescent
# without proving that every physical error has been removed.  Production
# mode persists this policy-visible protocol obligation until a same-state
# investigation either supplies another supported correction or justifies an
# operator handoff.  It is deliberately an observable controller marker, not
# a hidden-truth fault label.
POST_CORRECTION_CONFIRMATION_SIGNATURE = (
    "post_correction_resolution_confirmation_required:measurement_context"
)
GET_HARMONIC_CONTEXT = "get_harmonic_context"
RUN_HSE_FROM_PATH = "run_hse_from_path"
RUN_THREE_PHASE_NLM_FROM_PATH = "run_three_phase_nlm_from_path"
ESTIMATE_HIF_FROM_PATH = "estimate_hif_location_magnitude_from_path"
ESTIMATE_HIF_MULTISCAN_FROM_PATH = "estimate_hif_location_magnitude_multiscan_from_path"
INVALID_ACTION = "__invalid_action__"

CORRECTION_TOOLS = {
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
}

CONTEXT_TOOLS = {
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
}

STATE_MANAGEMENT_TOOLS = {
    COMMIT_STATE,
    ROLLBACK_STATE,
    FINALIZE_DIAGNOSIS,
}

# Read-only specialized diagnostics executed through configured evidence
# providers.  They share the canonical deployment tool names so DAgger data
# and the production corpus keep one model-visible surface.
DIAGNOSTIC_TOOLS = {
    GET_HARMONIC_CONTEXT,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
}

MACRO_ACTIONS = {
    RUN_WLS,
    VERIFY_CANDIDATE,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    COMMIT_STATE,
    ROLLBACK_STATE,
    FINALIZE_DIAGNOSIS,
    ASK_FOR_MORE_EVIDENCE,
    RUN_ALTERNATIVE_TEST,
    *DIAGNOSTIC_TOOLS,
}

# Observable anomaly-signature vocabulary shared by expert routing and the
# explained-anomaly termination semantics.  A diagnostic explanation for a
# family accounts for the unresolved signatures matching that family's
# markers; families not listed here can only be resolved by corrections.
ANOMALY_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "harmonic": ("harmonic", "harmonics", "thd", "distortion", "waveform"),
    "three_phase_unbalance": (
        "three_phase_unbalance",
        "voltage_unbalance",
        "unbalance",
        "imbalance",
        "negative_sequence",
        "vuf",
    ),
    "hif": (
        "hif",
        "high_impedance",
        "high_impedance_fault",
        "arc",
        "arcing",
        "downed_conductor",
        "zero_sequence_hif",
    ),
}


def unexplained_signatures(
    unresolved: Any,
    explained_records: Any,
) -> list[str]:
    """Unresolved signatures not covered by any recorded diagnostic explanation."""
    explained: set[str] = set()
    for record in explained_records or []:
        if not isinstance(record, Mapping):
            continue
        for signature in record.get("explained_signatures") or []:
            explained.add(str(signature))
    return [str(item) for item in (unresolved or []) if str(item) not in explained]


def terminal_explanation_signatures(unresolved: Any) -> list[str]:
    """Return physical/diagnostic signatures relevant to explanation closure.

    The post-correction confirmation marker is a process obligation rather
    than an anomaly that a diagnostic estimator can explain.  Filtering it
    here preserves valid explanation-only closure while ensuring that the
    marker by itself never authorizes finalization.
    """

    return [
        str(item)
        for item in (unresolved or [])
        if str(item) != POST_CORRECTION_CONFIRMATION_SIGNATURE
    ]


def invalid_action(error_code: str, error_detail: str | None = None) -> dict[str, Any]:
    """Return the canonical learner-action representation for malformed output."""
    arguments: dict[str, Any] = {"error_code": str(error_code)}
    if error_detail:
        arguments["error_detail"] = str(error_detail)
    return {"tool": INVALID_ACTION, "arguments": arguments}


def normalize_action(action: Mapping[str, Any] | str) -> dict[str, Any]:
    """Normalize a tool-call-like object to {"tool": name, "arguments": {...}}."""
    if isinstance(action, str):
        stripped = action.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            decoded = json.loads(stripped)
            if not isinstance(decoded, Mapping):
                raise ValueError("JSON action must decode to an object.")
            return normalize_action(decoded)
        return {"tool": stripped, "arguments": {}}
    if not isinstance(action, Mapping):
        raise TypeError(f"action must be a mapping or string, got {type(action).__name__}")

    function = action.get("function")
    if function is not None and not isinstance(function, Mapping):
        raise ValueError("Action function must be a mapping.")
    tool = (
        action.get("tool")
        or action.get("name")
        or action.get("tool_name")
        or (function or {}).get("name")
    )
    if not isinstance(tool, str) or not tool:
        raise ValueError(f"Action has no tool name: {action!r}")

    arguments = action.get("arguments")
    if arguments is None and isinstance(action.get("function"), Mapping):
        arguments = action["function"].get("arguments")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError(f"Action arguments for {tool} must be a mapping.")

    normalized_arguments = _canonicalize_correction_arguments(tool, dict(arguments))
    normalized = {"tool": tool, "arguments": normalized_arguments}
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action must be JSON-serializable: {exc}") from exc
    return normalized


def _canonicalize_correction_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Flatten the one supported correction payload and reject ambiguity.

    Older traces may wrap physical edits in ``arguments.modification`` while
    newer tool calls put them directly in ``arguments``.  Execution, process
    checks, provenance, and candidate assessment must all see the same fields;
    accepting conflicting copies would let a call claim one target and mutate
    another.
    """
    if tool not in CORRECTION_TOOLS:
        return arguments

    nested = arguments.pop("modification", None)
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("Correction modification must be a mapping.")
        for key, value in nested.items():
            if key in arguments and arguments[key] != value:
                raise ValueError(f"Conflicting correction field in modification: {key}")
            arguments.setdefault(str(key), value)

    updates = arguments.get("measurement_updates")
    if isinstance(updates, (list, tuple)):
        normalized_updates: dict[int, Any] = {}
        for item in updates:
            if not isinstance(item, Mapping):
                raise ValueError("measurement_updates list entries must be mappings.")
            raw_index = item.get("index", item.get("index0"))
            if raw_index is None or "value" not in item:
                raise ValueError("measurement_updates entries require index and value.")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("measurement update indices must be integers.") from exc
            if index in normalized_updates:
                raise ValueError(f"Duplicate measurement update index: {index}")
            normalized_updates[index] = item["value"]
        arguments["measurement_updates"] = normalized_updates
    elif isinstance(updates, Mapping):
        normalized_updates = {}
        for raw_index, value in updates.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("measurement update indices must be integers.") from exc
            if index in normalized_updates:
                raise ValueError(f"Duplicate measurement update index: {index}")
            normalized_updates[index] = value
        arguments["measurement_updates"] = normalized_updates

    if tool == CORRECT_MEASUREMENTS and isinstance(arguments.get("measurement_updates"), Mapping):
        declared = set(arguments["measurement_updates"])
        aliases: set[int] = set()
        for key in ("measurement_index", "index", "index0", "target", "meter", "measurement_id"):
            if arguments.get(key) is None:
                continue
            try:
                aliases.add(int(arguments[key]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Measurement target {key!r} must be the numeric updated index."
                ) from exc
        if aliases and aliases != declared:
            raise ValueError("Measurement target aliases conflict with measurement_updates.")

    if tool in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
        branch_targets = [
            key
            for key in ("branch_id", "cb_name", "line_index", "line_index1", "branch_row0")
            if arguments.get(key) is not None
        ]
        if len(branch_targets) > 1:
            raise ValueError("Correction must use exactly one branch target convention.")
    return arguments


def safe_normalize_action(action: Any) -> dict[str, Any]:
    """Normalize arbitrary policy output without raising.

    DAgger must retain malformed learner outputs as recovery examples.  This
    helper therefore converts parsing and schema failures to the same sentinel
    action consumed by the process-validity gate.
    """
    if isinstance(action, bytes):
        try:
            action = action.decode("utf-8")
        except UnicodeDecodeError as exc:
            return invalid_action("argument_decode_error", str(exc))

    if isinstance(action, str):
        stripped = action.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                action = json.loads(stripped)
            except json.JSONDecodeError as exc:
                return invalid_action("json_parse_error", exc.msg)

    if isinstance(action, Mapping):
        arguments = action.get("arguments")
        function = action.get("function")
        if arguments is None and isinstance(function, Mapping):
            arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                decoded_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return invalid_action("argument_decode_error", exc.msg)
            action = dict(action)
            if "arguments" in action:
                action["arguments"] = decoded_arguments
            elif isinstance(function, Mapping):
                action["function"] = dict(function)
                action["function"]["arguments"] = decoded_arguments

    try:
        return normalize_action(action)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return invalid_action("schema_error", str(exc))


def action_signature(action: Mapping[str, Any] | str) -> str:
    normalized = normalize_action(action)
    args_text = json.dumps(normalized["arguments"], sort_keys=True, separators=(",", ":"))
    return f"{normalized['tool']}:{args_text}"


def action_target(action: Mapping[str, Any] | str) -> str | None:
    args = normalize_action(action)["arguments"]
    for key in (
        "target",
        "meter",
        "measurement_id",
        "measurement_index",
        "branch_id",
        "line_index",
        "line_index1",
        "branch_row0",
        "cb_name",
    ):
        if key in args and args[key] is not None:
            return f"{key}={args[key]}"
    return None
