from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MEASUREMENT_ORDER = ["Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt"]
ERROR_FAMILIES = [
    "measurement_error",
    "parameter_error",
    "topology_error",
    "harmonic_anomaly",
    "no_error",
]
CONTEXT_TOOL_NAMES = {
    "get_parameter_context",
    "get_topology_context",
    "get_harmonic_context",
    "get_verification_snapshot",
}
USER_FLOAT_DECIMALS = 6
DIAGNOSTIC_FLOAT_DECIMALS = 4
CONFIDENCE_DECIMALS = 2
TOPK_EVIDENCE = 5
EVIDENCE_ABS_THRESHOLD = 3.0
NO_ERROR_RATIO_MAX = 0.9
BALANCED_SPLIT_COUNTS = {"train": 400, "valid": 50, "test": 50}
BALANCED_TOTAL_PER_CLASS = sum(BALANCED_SPLIT_COUNTS.values())
PROJECT_ROOT = Path(__file__).resolve().parent


DECISION_SCHEMA_TEXT = {
    "verdict": {
        "has_error": "boolean",
        "error_family": "scalar enum: measurement_error|parameter_error|topology_error|harmonic_anomaly|no_error",
        "confidence": "number in [0,1]",
    },
    "evidence": {
        "global_metrics": {
            "global_residual_sum": "number",
            "global_residual_threshold": "number",
            "global_residual_ratio": "number",
        },
        "top_residuals": [
            {
                "index0": "0-based measurement index",
                "channel": "string",
                "channel_offset": "0-based offset within channel",
                "value": "number",
            }
        ],
        "top_lagrange": [
            {
                "lambda_index0": "0-based normalized Lagrange index",
                "line_row0": "0-based branch row or null",
                "from_bus": "int or null",
                "to_bus": "int or null",
                "terminal": "'from'|'to'|'unknown'",
                "value": "number",
            }
        ],
    },
    "suspect_location": {
        "domain": "measurement|parameter|topology|harmonic|none",
        "details": "object",
    },
    "action": {
        "applied_tool": "tool name already used in this trace, or null",
        "arguments_hint": "object or null; use tool-schema field names, where line_index is 1-based",
        "request_more_data": "boolean",
        "requested_data": "array[string] or null",
        "verification_summary": "object or null; when present it includes post_action_global_residual_sum, post_action_global_residual_threshold, post_action_global_residual_ratio, post_action_executed, post_action_improved, post_action_resolved",
    },
    "summary": "short factual summary string",
}


SYSTEM_PROMPT = (
    "You are a power-system state-estimation diagnostic agent.\n"
    "You must begin with `wls_from_path` for every snapshot.\n"
    "Use Harmony/native tool calling only.\n"
    "Large numeric payloads are provided once in user messages and should not be repeated in tool arguments.\n"
    "If you need repeated scans, breaker context, harmonic measurements, or a post-action verification snapshot, "
    "retrieve them through the helper tools instead of asking the user for follow-up payloads.\n"
    "Available tools:\n"
    "- `wls_from_path(case_path)`: run weighted least-squares state estimation on the current user snapshot.\n"
    "- `get_parameter_context(case_path, line_index?)`: retrieve repeated scans and initial states for parameter correction.\n"
    "- `get_topology_context(case_path)`: retrieve compact breaker context for topology correction.\n"
    "- `get_harmonic_context(case_path)`: retrieve harmonic measurements for HSE.\n"
    "- `get_verification_snapshot(case_path, stage)`: retrieve a compact post-action verification snapshot.\n"
    "- `correct_measurements_from_path(case_path, suspect_group, ...)`: correct suspected bad measurements using the current snapshot.\n"
    "- `correct_parameters_from_path(case_path, line_index)`: correct line parameters after retrieving parameter context.\n"
    "- `correct_topology_from_path(case_path, cb_name, desired_status)`: correct a topology mismatch after retrieving breaker context.\n"
    "- `run_hse_from_path(case_path)`: run harmonic state estimation after retrieving harmonic measurements.\n\n"
    "Decision policy:\n"
    "1. Use concentrated large normalized residuals to localize likely measurement errors.\n"
    "2. Use large normalized Lagrange multipliers concentrated on one branch to suspect parameter errors.\n"
    "3. Use widespread residual patterns to suspect topology mismatch.\n"
    "4. If parameter context, breaker context, harmonic measurements, or verification snapshots are needed, call the matching helper tool.\n"
    "5. If the global residual is elevated without a dominant bad measurement and harmonic measurements are available, call `run_hse_from_path`.\n"
    "6. Prefer compact tool use over asking the user to restate numeric payloads.\n\n"
    "Indexing convention: fields ending in `0` are 0-based; `line_index` follows the tool schema and is 1-based.\n\n"
    "Return only strict JSON with this structure:\n"
    f"{json.dumps(DECISION_SCHEMA_TEXT, ensure_ascii=False)}\n"
    "Do not reveal chain-of-thought. Report only observable evidence and the final decision."
)


CANONICAL_POWER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "wls_from_path",
            "description": (
                "Run weighted least-squares state estimation on the current user snapshot. "
                "Do not resend z_obs in the tool arguments."
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
            "name": "get_parameter_context",
            "description": (
                "Retrieve repeated measurement scans and initial states needed for parameter correction. "
                "Returns a compact summary while the runtime binds the full arrays for the next tool call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "line_index": {"type": "integer", "description": "1-based suspected line index."},
                },
                "required": ["case_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_topology_context",
            "description": "Retrieve compact breaker context for topology correction.",
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
            "name": "get_harmonic_context",
            "description": (
                "Retrieve harmonic measurements and orders for harmonic follow-up. "
                "Returns a compact summary while the runtime binds the full measurements for HSE."
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
            "name": "get_verification_snapshot",
            "description": (
                "Retrieve a post-action verification snapshot. "
                "Returns a compact summary while the runtime binds the full z_obs for the next WLS call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "stage": {"type": "string", "description": "Verification stage identifier."},
                },
                "required": ["case_path", "stage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_measurements_from_path",
            "description": (
                "Correct suspected bad measurements using the current user snapshot. "
                "Provide the suspect group only; the wrapper will hydrate z_obs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "suspect_group": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "0-based suspected measurement indices.",
                    },
                    "enable_correction": {"type": "boolean", "description": "Whether to apply the correction."},
                    "max_correction_iterations": {
                        "type": "integer",
                        "description": "Maximum correction iterations.",
                    },
                    "error_tolerance": {"type": "number", "description": "Correction stopping tolerance."},
                },
                "required": ["case_path", "suspect_group"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_parameters_from_path",
            "description": (
                "Correct line-parameter errors after the user provides repeated measurement scans "
                "and initial states. Do not repeat those arrays in the tool call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "line_index": {"type": "integer", "description": "1-based MATPOWER branch row index."},
                },
                "required": ["case_path", "line_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_topology_from_path",
            "description": "Correct a suspected topology mismatch using a breaker name and desired status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "cb_name": {"type": "string", "description": "Breaker name."},
                    "desired_status": {"type": "boolean", "description": "Desired breaker status after correction."},
                },
                "required": ["case_path", "cb_name", "desired_status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_hse_from_path",
            "description": (
                "Run harmonic state estimation after the user provides harmonic measurements. "
                "Do not repeat the full harmonic payload in the tool call."
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
]


def json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in "{["


def parse_json_text(text: Any) -> Any | None:
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def prune_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: cleaned for k, item in value.items() if (cleaned := prune_none(item)) is not None}
    if isinstance(value, list):
        return [prune_none(item) for item in value if item is not None]
    return value


def maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str) or not looks_like_json(value):
        return value
    parsed = parse_json_text(value)
    return parsed if parsed is not None else value


def _round_float(value: float, decimals: int) -> float:
    rounded = round(value, decimals)
    return 0.0 if rounded == -0.0 else rounded


def round_user_payload(value: Any) -> Any:
    if isinstance(value, float):
        return _round_float(value, USER_FLOAT_DECIMALS)
    if isinstance(value, list):
        return [round_user_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: round_user_payload(item) for key, item in value.items()}
    return value


def round_assistant_payload(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, float):
        decimals = CONFIDENCE_DECIMALS if path and path[-1] == "confidence" else DIAGNOSTIC_FLOAT_DECIMALS
        return _round_float(value, decimals)
    if isinstance(value, list):
        return [round_assistant_payload(item, path) for item in value]
    if isinstance(value, dict):
        return {key: round_assistant_payload(item, path + (key,)) for key, item in value.items()}
    return value


def round_tool_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return round_user_payload(dict(arguments))


def default_schema_description(name: str | None, schema: dict[str, Any]) -> str:
    label = (name or "value").replace("_", " ")
    schema_type = schema.get("type")
    if schema_type == "boolean":
        return f"Whether to set {label}."
    if schema_type == "array":
        return f"List of {label}."
    if schema_type == "object":
        return f"{label.capitalize()} object."
    return f"{label.capitalize()} value."


def fill_schema_descriptions(schema: Any, name: str | None = None) -> Any:
    if isinstance(schema, list):
        return [fill_schema_descriptions(item, name=name) for item in schema]
    if not isinstance(schema, dict):
        return schema

    filled = {key: fill_schema_descriptions(value, name=key) for key, value in schema.items()}
    if any(key in filled for key in ("type", "properties", "items", "anyOf", "oneOf", "allOf")):
        filled.setdefault("description", default_schema_description(name, filled))
    return filled


def sanitize_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized.append(tool)
            continue
        fixed_tool = copy.deepcopy(tool)
        function_info = fixed_tool.get("function")
        if isinstance(function_info, dict):
            function_info.setdefault("description", f"Call the {function_info.get('name', 'tool')} tool.")
            parameters = function_info.get("parameters")
            if isinstance(parameters, dict):
                function_info["parameters"] = fill_schema_descriptions(parameters, name=function_info.get("name"))
            fixed_tool["function"] = function_info
        sanitized.append(fixed_tool)
    return sanitized


def canonical_tool_schemas() -> list[dict[str, Any]]:
    return sanitize_tool_schemas(CANONICAL_POWER_TOOLS)


def _normal_quantile(probability: float) -> float:
    try:
        from statistics import NormalDist

        return float(NormalDist().inv_cdf(probability))
    except Exception:
        # Common fallback values used in the repo.
        if abs(probability - 0.95) < 1e-12:
            return 1.6448536269514722
        if abs(probability - 0.99) < 1e-12:
            return 2.3263478740408408
        raise


def chi2_threshold(dof: int, alpha: float = 0.05) -> float:
    if dof <= 0:
        raise ValueError(f"dof must be positive, got {dof}")
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(1.0 - alpha, dof))
    except Exception:
        z = _normal_quantile(1.0 - alpha)
        d = float(dof)
        return d * (1.0 - 2.0 / (9.0 * d) + z * math.sqrt(2.0 / (9.0 * d))) ** 3


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def topk_abs(values: Sequence[float], k: int, *, min_abs: float | None = None) -> list[tuple[int, float]]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    order = np.argsort(-np.abs(arr))
    out: list[tuple[int, float]] = []
    for idx in order:
        value = float(arr[idx])
        if min_abs is not None and abs(value) < float(min_abs):
            continue
        out.append((int(idx), value))
        if len(out) >= k:
            break
    return out


def channel_from_index(index0: int, index_map: Mapping[str, slice]) -> tuple[str, int]:
    for channel in MEASUREMENT_ORDER:
        sl = index_map.get(channel)
        if not isinstance(sl, slice):
            continue
        if sl.start <= index0 < sl.stop:
            return channel, index0 - sl.start
    return "unknown", index0


def estimate_global_threshold(residuals: Sequence[float], nb: int | None) -> float | None:
    if nb is None:
        return None
    r = np.asarray(residuals, dtype=float)
    if r.size == 0:
        return None
    dof = max(int(r.size - (2 * int(nb) - 1)), 1)
    return float(chi2_threshold(dof))


def build_residual_evidence(
    residuals: Sequence[float],
    index_map: Mapping[str, slice],
    *,
    k: int = TOPK_EVIDENCE,
    min_abs: float = EVIDENCE_ABS_THRESHOLD,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for idx0, value in topk_abs(residuals, k, min_abs=min_abs):
        channel, offset = channel_from_index(idx0, index_map)
        evidence.append(
            {
                "index0": int(idx0),
                "channel": channel,
                "channel_offset": int(offset),
                "value": float(value),
            }
        )
    return round_assistant_payload(evidence)


def build_lambda_evidence(
    lambda_values: Sequence[float],
    branch_info: Sequence[Mapping[str, Any]],
    *,
    k: int = TOPK_EVIDENCE,
    min_abs: float = EVIDENCE_ABS_THRESHOLD,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for idx0, value in topk_abs(lambda_values, k, min_abs=min_abs):
        line_row0 = idx0 // 2 if idx0 >= 0 else None
        terminal = "from" if idx0 % 2 == 0 else "to"
        branch = branch_info[line_row0] if line_row0 is not None and 0 <= line_row0 < len(branch_info) else {}
        evidence.append(
            {
                "lambda_index0": int(idx0),
                "line_row0": int(line_row0) if line_row0 is not None else None,
                "from_bus": _maybe_int(branch.get("from_bus")),
                "to_bus": _maybe_int(branch.get("to_bus")),
                "terminal": terminal if branch else "unknown",
                "value": float(value),
            }
        )
    return round_assistant_payload(evidence)


def build_global_metrics(
    residuals: Sequence[float],
    meta: Mapping[str, Any],
    *,
    global_residual_sum: float | None = None,
    global_residual_threshold: float | None = None,
) -> dict[str, float]:
    residual_array = np.asarray(residuals, dtype=float)
    J = _maybe_float(global_residual_sum)
    if J is None:
        J = float(np.sum(residual_array**2))

    threshold = _maybe_float(global_residual_threshold)
    if threshold is None:
        threshold = estimate_global_threshold(residual_array, _maybe_int(meta.get("nb")))
    if threshold is None:
        raise ValueError("global_residual_threshold could not be determined")

    ratio = J / threshold if abs(threshold) > 1e-12 else math.inf
    return round_assistant_payload(
        {
            "global_residual_sum": float(J),
            "global_residual_threshold": float(threshold),
            "global_residual_ratio": float(ratio),
        }
    )


def summarize_wls_payload(
    tool_payload: Mapping[str, Any],
    meta: Mapping[str, Any],
    index_map: Mapping[str, slice],
    *,
    force_empty_evidence: bool = False,
) -> dict[str, Any]:
    residuals = tool_payload.get("r", []) or []
    lambda_values = tool_payload.get("lambdaN", []) or []
    summary: dict[str, Any] = {
        "success": bool(tool_payload.get("success", True)),
        "top_residuals": [] if force_empty_evidence else build_residual_evidence(residuals, index_map),
        "top_lagrange": [] if force_empty_evidence else build_lambda_evidence(lambda_values, meta.get("branch_info", [])),
    }
    have_global_inputs = bool(residuals) or tool_payload.get("global_residual_sum") is not None or tool_payload.get("global_residual_threshold") is not None
    if have_global_inputs:
        try:
            summary["global_metrics"] = build_global_metrics(
                residuals,
                meta,
                global_residual_sum=_maybe_float(tool_payload.get("global_residual_sum")),
                global_residual_threshold=_maybe_float(tool_payload.get("global_residual_threshold")),
            )
        except ValueError:
            pass
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(summary)


def summarize_measurement_correction_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    corrected = list(tool_payload.get("corrected_measurements") or [])
    corrected_sorted = sorted(
        corrected,
        key=lambda item: abs(float(item.get("estimated_error", 0.0))),
        reverse=True,
    )[:3]
    compact_corrected = []
    for item in corrected_sorted:
        compact_corrected.append(
            {
                "index0": _maybe_int(item.get("index0")),
                "corrected": _maybe_float(item.get("corrected")),
                "estimated_error": _maybe_float(item.get("estimated_error")),
            }
        )
    summary = {
        "success": bool(tool_payload.get("success", True)),
        "applied_any_correction": bool(tool_payload.get("applied_any_correction", False)),
        "iterations_performed": _maybe_int(tool_payload.get("iterations_performed")),
        "corrected_measurements": compact_corrected,
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_parameter_correction_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    corrected_params = tool_payload.get("corrected_params") or []
    summary = {
        "success": bool(tool_payload.get("success", True)),
        "corrected_parameters": {
            "r": _maybe_float(corrected_params[0]) if len(corrected_params) >= 1 else None,
            "x": _maybe_float(corrected_params[1]) if len(corrected_params) >= 2 else None,
        },
        "meta": {
            "line_index": _maybe_int((tool_payload.get("meta") or {}).get("line_index")),
            "from_bus": _maybe_int((tool_payload.get("meta") or {}).get("from_bus")),
            "to_bus": _maybe_int((tool_payload.get("meta") or {}).get("to_bus")),
            "scans": _maybe_int((tool_payload.get("meta") or {}).get("scans")),
        },
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_topology_correction_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "success": bool(tool_payload.get("success", True)),
        "cb_name": tool_payload.get("cb_name"),
        "applied_status": tool_payload.get("new_status"),
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_hse_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    ranking = list(tool_payload.get("ranking_top10") or [])[:TOPK_EVIDENCE]
    thd_map = tool_payload.get("estimated_thd_percent") or {}
    best_bus = _maybe_int(tool_payload.get("best_candidate_bus_1based"))
    best_thd = None
    if best_bus is not None:
        best_thd = _maybe_float(thd_map.get(str(best_bus)))
    summary = {
        "success": bool(tool_payload.get("success", True)),
        "best_candidate_bus_1based": best_bus,
        "best_candidate_thd_percent": best_thd,
        "ranking_top5": ranking,
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_parameter_context_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    z_scans = tool_payload.get("z_scans") or []
    initial_states = tool_payload.get("initial_states") or []
    summary = {
        "case_path": tool_payload.get("case_path"),
        "scans": len(z_scans) if isinstance(z_scans, list) else None,
        "measurement_vector_length": len(z_scans[0]) if isinstance(z_scans, list) and z_scans else None,
        "state_vector_length": len(initial_states[0]) if isinstance(initial_states, list) and initial_states else None,
        "suspect_line": tool_payload.get("suspect_line"),
        "note": tool_payload.get("note"),
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_topology_context_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "breaker_context": tool_payload.get("breaker_context"),
        "note": tool_payload.get("note"),
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_harmonic_context_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    measurements = tool_payload.get("harmonic_measurements") or []
    summary = {
        "case_path": tool_payload.get("case_path"),
        "measurement_count": len(measurements) if isinstance(measurements, list) else None,
        "harmonic_orders": tool_payload.get("harmonic_orders"),
        "note": tool_payload.get("note"),
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_verification_snapshot_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    z_obs = tool_payload.get("z_obs") or []
    summary = {
        "case_path": tool_payload.get("case_path"),
        "measurement_count": len(z_obs) if isinstance(z_obs, list) else None,
        "note": tool_payload.get("note"),
        "stage": tool_payload.get("stage"),
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_tool_result_for_conversation(
    tool_name: str,
    tool_result: Mapping[str, Any],
    meta: Mapping[str, Any],
    index_map: Mapping[str, slice],
    *,
    force_empty_evidence: bool = False,
) -> dict[str, Any]:
    if tool_name == "wls_from_path":
        return summarize_wls_payload(tool_result, meta, index_map, force_empty_evidence=force_empty_evidence)
    if tool_name == "correct_measurements_from_path":
        return summarize_measurement_correction_payload(tool_result)
    if tool_name == "correct_parameters_from_path":
        return summarize_parameter_correction_payload(tool_result)
    if tool_name == "correct_topology_from_path":
        return summarize_topology_correction_payload(tool_result)
    if tool_name == "run_hse_from_path":
        return summarize_hse_payload(tool_result)
    if tool_name == "get_parameter_context":
        return summarize_parameter_context_payload(tool_result)
    if tool_name == "get_topology_context":
        return summarize_topology_context_payload(tool_result)
    if tool_name == "get_harmonic_context":
        return summarize_harmonic_context_payload(tool_result)
    if tool_name == "get_verification_snapshot":
        return summarize_verification_snapshot_payload(tool_result)
    return round_assistant_payload(dict(tool_result))


def extract_user_payloads(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        obj = content if isinstance(content, dict) else parse_json_text(content)
        if isinstance(obj, dict):
            payloads.append(obj)
    return payloads


def extract_tool_payloads(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        obj = content if isinstance(content, dict) else parse_json_text(content)
        if isinstance(obj, dict):
            payloads.append({"name": message.get("name"), "payload": obj})
    return payloads


def _coerce_index_slice(value: Any) -> slice | None:
    if isinstance(value, slice):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start = _maybe_int(value[0])
        stop = _maybe_int(value[1])
        if start is not None and stop is not None:
            return slice(start, stop)
    return None


def extract_conversation_context(
    messages: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, slice]]:
    meta: dict[str, Any] = {}
    index_map: dict[str, slice] = {}
    for payload in extract_user_payloads(messages):
        meta_hint = payload.get("meta_hint")
        if isinstance(meta_hint, Mapping):
            meta.update(dict(meta_hint))
        for key in ("nb", "nl", "baseMVA", "case"):
            if key in payload and payload.get(key) is not None:
                meta[key] = payload.get(key)
        branch_info = payload.get("branch_info")
        if isinstance(branch_info, list):
            meta["branch_info"] = branch_info
        raw_index_map = payload.get("index_map")
        if isinstance(raw_index_map, Mapping):
            converted = {
                str(channel): sl
                for channel, bounds in raw_index_map.items()
                if (sl := _coerce_index_slice(bounds)) is not None
            }
            if converted:
                index_map = converted
    return meta, index_map


def latest_user_payload(messages: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    payloads = extract_user_payloads(messages)
    return payloads[-1] if payloads else None


def latest_user_payload_with_keys(
    messages: Iterable[Mapping[str, Any]],
    required_keys: Sequence[str],
) -> dict[str, Any] | None:
    required = set(required_keys)
    for payload in reversed(extract_user_payloads(messages)):
        if required.issubset(payload.keys()):
            return payload
    return None


def latest_tool_payload_with_keys(
    messages: Iterable[Mapping[str, Any]],
    required_keys: Sequence[str],
    *,
    tool_name: str | None = None,
) -> dict[str, Any] | None:
    required = set(required_keys)
    for item in reversed(extract_tool_payloads(messages)):
        if tool_name is not None and item.get("name") != tool_name:
            continue
        payload = item.get("payload")
        if isinstance(payload, dict) and required.issubset(payload.keys()):
            return payload
    return None


def hydrate_tool_arguments(
    tool_name: str,
    arguments: Any,
    messages: Sequence[Mapping[str, Any]],
    hidden_context: Mapping[str, Any] | None = None,
) -> tuple[Any, list[str]]:
    notes: list[str] = []
    if not isinstance(arguments, dict):
        return arguments, notes

    hydrated = copy.deepcopy(arguments)
    hidden = dict(hidden_context or {})
    latest_payload = latest_user_payload(messages)

    if isinstance(latest_payload, dict):
        user_case = latest_payload.get("case_path") or latest_payload.get("case")
        if user_case and not hydrated.get("case_path"):
            hydrated["case_path"] = user_case
            notes.append("filled_case_path_from_user")

    if tool_name in {"wls_from_path", "correct_measurements_from_path"}:
        source = hidden.get("snapshot_context")
        source_from_hidden = isinstance(source, dict)
        if not source_from_hidden:
            source = latest_user_payload_with_keys(messages, ("z_obs",))
        if isinstance(source, dict) and isinstance(source.get("z_obs"), list) and "z" not in hydrated:
            hydrated["z"] = source["z_obs"]
            notes.append(f"hydrated_{tool_name}_z_from_user")
        if isinstance(source, dict) and source.get("case_path") and (source_from_hidden or not hydrated.get("case_path")):
            hydrated["case_path"] = source["case_path"]
            notes.append(f"hydrated_{tool_name}_case_path_from_snapshot")

    if tool_name == "correct_parameters_from_path":
        source = hidden.get("parameter_context")
        source_from_hidden = isinstance(source, dict)
        if not source_from_hidden:
            source = latest_user_payload_with_keys(messages, ("z_scans", "initial_states"))
        if isinstance(source, dict):
            if source.get("case_path") and (source_from_hidden or not hydrated.get("case_path")):
                hydrated["case_path"] = source["case_path"]
                notes.append("hydrated_correct_parameters_case_path")
            if "z_scans" not in hydrated and isinstance(source.get("z_scans"), list):
                hydrated["z_scans"] = source["z_scans"]
                notes.append("hydrated_correct_parameters_z_scans")
            if "initial_states" not in hydrated and isinstance(source.get("initial_states"), list):
                hydrated["initial_states"] = source["initial_states"]
                notes.append("hydrated_correct_parameters_initial_states")

    if tool_name == "run_hse_from_path":
        source = hidden.get("harmonic_context")
        source_from_hidden = isinstance(source, dict)
        if not source_from_hidden:
            source = latest_user_payload_with_keys(messages, ("harmonic_measurements",))
        if isinstance(source, dict):
            if source.get("case_path") and (source_from_hidden or not hydrated.get("case_path")):
                hydrated["case_path"] = source["case_path"]
                notes.append("hydrated_hse_case_path")
            if "harmonic_measurements" not in hydrated and isinstance(source.get("harmonic_measurements"), list):
                hydrated["harmonic_measurements"] = source["harmonic_measurements"]
                notes.append("hydrated_hse_measurements")
            if "harmonic_orders" not in hydrated and isinstance(source.get("harmonic_orders"), list):
                hydrated["harmonic_orders"] = source["harmonic_orders"]
                notes.append("hydrated_hse_orders")

    if tool_name == "correct_topology_from_path":
        source = hidden.get("topology_context")
        source_from_hidden = isinstance(source, dict)
        if not source_from_hidden:
            source = latest_user_payload_with_keys(messages, ("breaker_context",))
        if not isinstance(source, dict):
            source = latest_tool_payload_with_keys(messages, ("breaker_context",), tool_name="get_topology_context")
        if isinstance(source, dict):
            if source.get("case_path") and (source_from_hidden or not hydrated.get("case_path")):
                hydrated["case_path"] = source["case_path"]
                notes.append("hydrated_topology_case_path")
            context = source.get("breaker_context")
            if isinstance(context, dict):
                if "cb_name" not in hydrated and context.get("cb_name"):
                    hydrated["cb_name"] = context["cb_name"]
                    notes.append("hydrated_topology_cb_name")
                if "desired_status" not in hydrated and "desired_status" in context:
                    hydrated["desired_status"] = context["desired_status"]
                    notes.append("hydrated_topology_desired_status")

    return round_tool_arguments(hydrated), notes


def resolve_case_path_alias(case_path: Any, hidden_context: Mapping[str, Any] | None = None) -> Any:
    if not isinstance(case_path, str):
        return case_path
    aliases = (hidden_context or {}).get("case_aliases")
    if isinstance(aliases, Mapping) and isinstance(aliases.get(case_path), str):
        resolved = aliases[case_path]
        candidate = Path(resolved)
        if not candidate.is_absolute() and (candidate.suffix or "/" in resolved or "\\" in resolved):
            repo_candidate = PROJECT_ROOT / candidate
            if repo_candidate.exists():
                return str(repo_candidate)
        return resolved
    return case_path


def normalize_error_family(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value in ERROR_FAMILIES else None
    if isinstance(value, list):
        for item in value:
            family = normalize_error_family(item)
            if family is not None:
                return family
    return None


def normalize_verdict(verdict_obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(verdict_obj, dict):
        return None
    verdict = verdict_obj.get("verdict")
    if not isinstance(verdict, dict):
        return verdict_obj

    normalized = json.loads(json.dumps(verdict_obj))
    verdict_block = normalized.setdefault("verdict", {})
    if "has_error" in verdict_block:
        verdict_block["has_error"] = bool(verdict_block["has_error"])
    verdict_block["error_family"] = normalize_error_family(verdict_block.get("error_family"))
    if "confidence" in verdict_block and verdict_block["confidence"] is not None:
        verdict_block["confidence"] = _round_float(float(verdict_block["confidence"]), CONFIDENCE_DECIMALS)
    return normalized
