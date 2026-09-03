from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hif_search_limits import (
    HIF_ALPHA_GRID_SIZE_MAX,
    HIF_ALPHA_GRID_SIZE_MIN,
    HIF_MAX_SCANS_MAX,
    HIF_MAX_SCANS_MIN,
    HIF_R_GRID_SIZE_MAX,
    HIF_R_GRID_SIZE_MIN,
)


MEASUREMENT_ORDER = ["Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt"]
ERROR_FAMILIES = [
    "measurement_error",
    "parameter_error",
    "topology_error",
    "three_phase_imbalance",
    "harmonic_anomaly",
    "high_impedance_fault",
    "no_error",
]
CONTEXT_TOOL_NAMES = {
    "get_parameter_context",
    "get_topology_context",
    "get_harmonic_context",
    "get_verification_snapshot",
}
USER_FLOAT_DECIMALS = 6
TOOL_RESULT_FLOAT_DECIMALS = 6
DIAGNOSTIC_FLOAT_DECIMALS = 4
SCORE_MARGIN_FLOAT_DECIMALS = 8
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
        "error_family": "scalar enum: measurement_error|parameter_error|topology_error|three_phase_imbalance|harmonic_anomaly|high_impedance_fault|no_error",
        "error_families": "optional array of error families for multi-error snapshots",
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
                "parameter": "'R'|'X'|'unknown' (which series parameter the multiplier belongs to)",
                "value": "number",
            }
        ],
        "top_hif_groups": [
            {
                "rank": "1-based ranking from three-phase NLM line-group evidence",
                "branch_row0": "0-based IEEE-14 branch row",
                "line_index1": "1-based IEEE-14 branch row",
                "dss_element": "OpenDSS element name",
                "from_bus": "int or null",
                "to_bus": "int or null",
                "score": "number",
            }
        ],
        "hif_parameter_estimate": {
            "alpha_from_from_bus": "estimated line fraction from the from bus",
            "distance_percent_from_from_bus": "estimated distance percent from the from bus",
            "phase": "optional estimated phase if the estimator searched or scored phases",
            "r_hif_pu": "estimated HIF resistance in per unit",
            "r_hif_ohm": "estimated HIF resistance in ohms",
            "p_hif_kw": "estimated HIF active power",
            "i_hif_amp": "estimated HIF current",
            "localization_certainty": "well_separated|moderately_separated|ambiguous_top2",
            "parameter_identifiable": "boolean from multi-scan observability diagnostics when available",
            "observability_status": "optional full_rank_well_conditioned|full_rank_weakly_conditioned|rank_deficient|noise_averaging_only|model_mismatch_suspected|diagnostic_partial",
        },
    },
    "evidence_by_stage": {
        "optional": "multi-error only; maps stage name to compact WLS evidence observed at that stage",
        "initial": "same compact evidence shape as evidence",
        "post_topology_correction": "optional compact evidence after topology correction",
        "post_parameter_correction": "optional compact evidence after parameter correction",
        "post_measurement_correction": "optional compact evidence after measurement correction",
    },
    "suspect_location": {
        "domain": "measurement|parameter|topology|imbalance|harmonic|fault|none",
        "details": "object",
    },
    "suspect_locations": "optional array of suspect_location objects for multi-error snapshots",
    "action": {
        "applied_tool": "tool name already used in this trace, or null",
        "first_applied_tool": "optional first tool name already used in a multi-error trace",
        "last_applied_tool": "optional last tool name already used in a multi-error trace",
        "applied_tools": "optional array of tool names already used in this trace for multi-error snapshots",
        "arguments_hint": "object or null; use tool-schema field names, where line_index is 1-based",
        "request_more_data": "boolean",
        "requested_data": "array[string] or null",
        "verification_summary": "object or null; when present it includes post_action_global_residual_sum, post_action_global_residual_threshold, post_action_global_residual_ratio, post_action_executed, post_action_improved, post_action_resolved",
        "verification_summaries": "optional object keyed by error family for multi-error snapshots",
        "last_verified_summary": "optional object with family plus the most recent verified WLS summary when later steps are diagnostic or sequence-only",
        "tool_steps": "optional ordered array of tool-use steps with family, tool, tool_role, verification_policy, and residual summaries when applicable",
        "correction_steps": "optional ordered array of multi-error correction steps with family, tool, pre/post residual ratios, improvement flags, and remaining_candidate_families",
        "diagnosis_status": "complete|near_threshold|elevated|partial|unknown|not_applicable|curriculum_only|sequence_only",
        "remaining_candidate_families": "array[string], e.g. ['unexplained_residual'] when diagnosis_status is partial",
        "measurement_correction_policy": "object explaining whether measurement correction was allowed, why, residual_pattern localized|distributed|unknown|not_applicable, suspect_group_size, structural_checks_completed, structural_tools_before_measurement, and request_more_data",
    },
    "summary": "short factual summary string",
}


def decision_schema_text_for_prompt(*, include_extended_diagnostics: bool = True) -> dict[str, Any]:
    schema = copy.deepcopy(DECISION_SCHEMA_TEXT)
    if include_extended_diagnostics:
        return schema
    schema["verdict"][
        "error_family"
    ] = "scalar enum: measurement_error|parameter_error|topology_error|harmonic_anomaly|no_error"
    schema["suspect_location"]["domain"] = "measurement|parameter|topology|harmonic|none"
    evidence = schema.get("evidence")
    if isinstance(evidence, dict):
        evidence.pop("top_hif_groups", None)
        evidence.pop("hif_parameter_estimate", None)
    return schema


SYSTEM_PROMPT = (
    "You are a power-system state-estimation diagnostic agent.\n"
    "You must begin with `wls_from_path` for every snapshot.\n"
    "Use structured tool calls that match the provided tool schema.\n"
    "Use the tool name and argument keys exactly as provided.\n"
    "Large numeric payloads are provided once in user messages and should not be repeated in tool arguments.\n"
    "If you need repeated scans, breaker context, harmonic measurements, or a post-action verification snapshot, "
    "retrieve them through the helper tools instead of asking the user for follow-up payloads.\n"
    "Available tools:\n"
    "- `wls_from_path(case_path)`: run weighted least-squares state estimation on the current user snapshot.\n"
    "- `get_parameter_context(case_path, line_index?)`: retrieve repeated scans and initial states for parameter correction.\n"
    "- `get_topology_context(case_path)`: retrieve compact breaker context for topology correction.\n"
    "- `get_harmonic_context(case_path)`: retrieve harmonic measurements for HSE.\n"
    "- `get_verification_snapshot(stage?)`: retrieve the current post-action verification snapshot by stage; do not invent snapshot aliases.\n"
    "- `correct_measurements_from_path(case_path, suspect_group, ...)`: correct suspected bad measurements using the current snapshot.\n"
    "- `correct_parameters_from_path(case_path, line_index)`: correct line parameters after retrieving parameter context.\n"
    "- `correct_topology_from_path(case_path, cb_name, desired_status)`: correct a topology mismatch after retrieving breaker context.\n"
    "- `run_hse_from_path(case_path)`: run harmonic state estimation after retrieving harmonic measurements.\n"
    "- `run_three_phase_nlm_from_path(case_path)`: run compact three-phase NLM HIF localization evidence.\n"
    "- `estimate_hif_location_magnitude_from_path(case_path, candidate_branch_row0, candidate_phase?)`: estimate HIF line fraction and resistance after NLM selects a suspected line.\n"
    "- `estimate_hif_location_magnitude_multiscan_from_path(scan_window_path, candidate_branch_row0, candidate_phase?)`: estimate shared HIF parameters and observability from a persistent scan window.\n\n"
    "Decision policy:\n"
    "1. Use widespread residual patterns to suspect topology mismatch or three-phase imbalance.\n"
    "2. Use large normalized Lagrange multipliers concentrated on one branch to suspect parameter errors.\n"
    "3. Use concentrated large normalized residuals to localize likely measurement errors.\n"
    "4. If parameter context, breaker context, harmonic measurements, or verification snapshots are needed, call the matching helper tool.\n"
    "5. If three-phase imbalance is suspected, request three-phase substation VLN voltages before finalizing.\n"
    "6. Measurement correction is only valid for localized bad-data patterns; do not use `correct_measurements_from_path` as a generic residual-reduction tool.\n"
    "7. If residuals are distributed across many channels or branches, check topology and parameter evidence before measurement cleanup.\n"
    "8. After topology or parameter correction, use measurement correction only if the remaining residuals are localized; otherwise request more data or report an unresolved model/data inconsistency.\n"
    "9. If a localized gross residual remains, correct the measurement error before running or finalizing harmonic HSE; HSE does not replace SCADA measurement cleanup.\n"
    "10. If the global residual is elevated without a dominant bad measurement and harmonic measurements are available, call `run_hse_from_path`.\n"
    "11. If a hidden high-impedance fault is suspected, call `run_three_phase_nlm_from_path` and use top_hif_groups evidence.\n"
    "12. If NLM returns a suspected HIF line and a persistent scan window is available, call `estimate_hif_location_magnitude_multiscan_from_path`; otherwise call the single-scan estimator.\n"
    "13. Report HIF line fraction and magnitude only from the parameter-estimation tool; claim a point location only when ambiguity and observability diagnostics permit it.\n"
    "14. In multi-error traces, prefer structural correction before measurement cleanup: topology, then parameter, then measurement, then harmonic follow-up.\n"
    "15. After a correction tool succeeds, request the verification snapshot by `stage` only, then run WLS on the returned `case_path` exactly once.\n"
    "16. Prefer compact tool use over asking the user to restate numeric payloads.\n\n"
    "Indexing convention: fields ending in `0` are 0-based; `line_index` follows the tool schema and is 1-based.\n\n"
    "Return only strict JSON with this structure:\n"
    f"{json.dumps(DECISION_SCHEMA_TEXT, ensure_ascii=False)}\n"
    "For multi-error snapshots, keep `error_family` as the primary family and also report all families in "
    "`error_families`, `suspect_locations`, and `applied_tools`.\n"
    "Do not reveal chain-of-thought. Report only observable evidence and the final decision."
)

SCADA_HARMONIC_SYSTEM_PROMPT = (
    "You are a power-system state-estimation diagnostic agent.\n"
    "You must begin with `wls_from_path` for every snapshot.\n"
    "Use structured tool calls that match the provided tool schema.\n"
    "Use the tool name and argument keys exactly as provided.\n"
    "Large numeric payloads are provided once in user messages and should not be repeated in tool arguments.\n"
    "If you need repeated scans, breaker context, harmonic measurements, or a post-action verification snapshot, "
    "retrieve them through the helper tools instead of asking the user for follow-up payloads.\n"
    "Available tools:\n"
    "- `wls_from_path(case_path)`: run weighted least-squares state estimation on the current user snapshot.\n"
    "- `get_parameter_context(case_path, line_index?)`: retrieve repeated scans and initial states for parameter correction.\n"
    "- `get_topology_context(case_path)`: retrieve compact breaker context for topology correction.\n"
    "- `get_harmonic_context(case_path)`: retrieve harmonic measurements for HSE.\n"
    "- `get_verification_snapshot(stage?)`: retrieve the current post-action verification snapshot by stage; do not invent snapshot aliases.\n"
    "- `correct_measurements_from_path(case_path, suspect_group, ...)`: correct suspected bad measurements using the current snapshot.\n"
    "- `correct_parameters_from_path(case_path, line_index)`: correct line parameters after retrieving parameter context.\n"
    "- `correct_topology_from_path(case_path, cb_name, desired_status)`: correct a topology mismatch after retrieving breaker context.\n"
    "- `run_hse_from_path(case_path)`: run harmonic state estimation after retrieving harmonic measurements.\n\n"
    "Decision policy:\n"
    "1. Use widespread residual patterns to suspect topology mismatch.\n"
    "2. Use large normalized Lagrange multipliers concentrated on one branch to suspect parameter errors.\n"
    "3. Use concentrated large normalized residuals to localize likely measurement errors.\n"
    "4. If parameter context, breaker context, harmonic measurements, or verification snapshots are needed, call the matching helper tool.\n"
    "5. Measurement correction is only valid for localized bad-data patterns; do not use `correct_measurements_from_path` as a generic residual-reduction tool.\n"
    "6. If residuals are distributed across many channels or branches, check topology and parameter evidence before measurement cleanup.\n"
    "7. After topology or parameter correction, use measurement correction only if the remaining residuals are localized; otherwise request more data or report an unresolved model/data inconsistency.\n"
    "8. If a localized gross residual remains, correct the measurement error before running or finalizing harmonic HSE; HSE does not replace SCADA measurement cleanup.\n"
    "9. If the global residual is elevated without a dominant bad measurement and harmonic measurements are available, call `run_hse_from_path`.\n"
    "10. In multi-error traces, prefer structural correction before measurement cleanup: topology, then parameter, then measurement, then harmonic follow-up.\n"
    "11. After a correction tool succeeds, request the verification snapshot by `stage` only, then run WLS on the returned `case_path` exactly once.\n"
    "12. Prefer compact tool use over asking the user to restate numeric payloads.\n\n"
    "Indexing convention: fields ending in `0` are 0-based; `line_index` follows the tool schema and is 1-based.\n\n"
    "Return only strict JSON with this structure:\n"
    f"{json.dumps(decision_schema_text_for_prompt(include_extended_diagnostics=False), ensure_ascii=False)}\n"
    "For multi-error snapshots, keep `error_family` as the primary family and also report all families in "
    "`error_families`, `suspect_locations`, and `applied_tools`.\n"
    "Do not reveal chain-of-thought. Report only observable evidence and the final decision."
)

SYSTEM_PROMPT_PREFIX = (
    "You are a power-system state-estimation diagnostic agent.\n"
    "You must begin with `wls_from_path` for every snapshot.\n"
)
LEGACY_SYSTEM_PROMPT_MARKERS = (
    "Use Harmony/native tool calling only.",
    "Return only strict JSON with this structure:",
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
                "Retrieve the active post-action verification snapshot by stage. "
                "Do not synthesize a snapshot case alias; use the returned case_path for the next WLS call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "description": "Optional verification stage identifier, e.g. post_measurement_correction.",
                    },
                },
                "required": [],
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
    {
        "type": "function",
        "function": {
            "name": "run_three_phase_nlm_from_path",
            "description": (
                "Run compact three-phase NLM high-impedance-fault localization evidence. "
                "Do not repeat OpenDSS models or measurement arrays in the tool arguments."
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
            "name": "estimate_hif_location_magnitude_from_path",
            "description": (
                "Estimate HIF position along a suspected IEEE-14 line and estimate HIF magnitude "
                "using model-based OpenDSS residual fitting. Do not repeat measurement arrays in "
                "the tool arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_path": {"type": "string", "description": "Case identifier or path."},
                    "candidate_branch_row0": {
                        "type": "integer",
                        "description": "Zero-based IEEE-14 branch row selected by the line-level HIF locator.",
                    },
                    "candidate_phase": {
                        "type": ["string", "null"],
                        "enum": ["A", "B", "C", None],
                        "description": "Optional phase hint. If omitted or null, all phases are searched.",
                    },
                    "top_k": {"type": "integer", "default": 5},
                    "alpha_grid_size": {
                        "type": "integer",
                        "minimum": HIF_ALPHA_GRID_SIZE_MIN,
                        "maximum": HIF_ALPHA_GRID_SIZE_MAX,
                        "default": HIF_ALPHA_GRID_SIZE_MAX,
                    },
                    "r_grid_size": {
                        "type": "integer",
                        "minimum": HIF_R_GRID_SIZE_MIN,
                        "maximum": HIF_R_GRID_SIZE_MAX,
                        "default": HIF_R_GRID_SIZE_MAX,
                    },
                    "r_hif_pu_min": {"type": "number", "default": 5.0},
                    "r_hif_pu_max": {"type": "number", "default": 1000.0},
                },
                "required": ["case_path", "candidate_branch_row0"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_hif_location_magnitude_multiscan_from_path",
            "description": (
                "Estimate shared HIF position, phase, and resistance from a persistent IEEE-14 "
                "scan window, with scan-selection and observability diagnostics. Do not repeat "
                "measurement arrays in the tool arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_window_path": {
                        "type": "string",
                        "description": "Path or runtime-bound identifier for the persistent HIF scan window.",
                    },
                    "candidate_branch_row0": {
                        "type": "integer",
                        "description": "Zero-based IEEE-14 branch row selected by the line-level HIF locator.",
                    },
                    "candidate_phase": {
                        "type": ["string", "null"],
                        "enum": ["A", "B", "C", None],
                        "description": "Optional phase hint. If omitted or null, all phases are searched.",
                    },
                    "resistance_mode": {
                        "type": "string",
                        "enum": ["shared", "scan_specific_smooth"],
                        "default": "shared",
                    },
                    "max_scans": {
                        "type": "integer",
                        "minimum": HIF_MAX_SCANS_MIN,
                        "maximum": HIF_MAX_SCANS_MAX,
                        "default": HIF_MAX_SCANS_MAX,
                    },
                    "scan_selection": {
                        "type": "string",
                        "enum": ["all", "diversity_greedy", "information_greedy"],
                        "default": "information_greedy",
                    },
                    "top_k": {"type": "integer", "default": 5},
                    "alpha_grid_size": {
                        "type": "integer",
                        "minimum": HIF_ALPHA_GRID_SIZE_MIN,
                        "maximum": HIF_ALPHA_GRID_SIZE_MAX,
                        "default": HIF_ALPHA_GRID_SIZE_MAX,
                    },
                    "r_grid_size": {
                        "type": "integer",
                        "minimum": HIF_R_GRID_SIZE_MIN,
                        "maximum": HIF_R_GRID_SIZE_MAX,
                        "default": HIF_R_GRID_SIZE_MAX,
                    },
                    "r_hif_pu_min": {"type": "number", "default": 5.0},
                    "r_hif_pu_max": {"type": "number", "default": 1000.0},
                    "robust_loss": {
                        "type": "string",
                        "enum": ["linear", "soft_l1", "huber"],
                        "default": "soft_l1",
                    },
                    "smoothness_lambda": {"type": "number", "default": 0.10},
                },
                "required": ["scan_window_path", "candidate_branch_row0"],
            },
        },
    },
]


def json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_instruction_content(role: Any, content: Any) -> Any:
    """Rewrite repo-owned legacy system prompts to the current model-neutral form."""
    if role not in {"system", "developer"} or not isinstance(content, str):
        return content

    text = content.strip()
    if not text:
        return content

    if text == SYSTEM_PROMPT.strip():
        return SYSTEM_PROMPT
    if text == SCADA_HARMONIC_SYSTEM_PROMPT.strip():
        return SCADA_HARMONIC_SYSTEM_PROMPT
    if text.startswith(SYSTEM_PROMPT_PREFIX.strip()):
        return SYSTEM_PROMPT
    if any(marker in text for marker in LEGACY_SYSTEM_PROMPT_MARKERS):
        return SYSTEM_PROMPT
    return content


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
        if path and path[-1] == "confidence":
            decimals = CONFIDENCE_DECIMALS
        elif path and path[-1] in {"top_score_margin", "top_score_relative_margin"}:
            decimals = SCORE_MARGIN_FLOAT_DECIMALS
        else:
            decimals = DIAGNOSTIC_FLOAT_DECIMALS
        return _round_float(value, decimals)
    if isinstance(value, list):
        return [round_assistant_payload(item, path) for item in value]
    if isinstance(value, dict):
        return {key: round_assistant_payload(item, path + (key,)) for key, item in value.items()}
    return value


def round_tool_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return round_user_payload(dict(arguments))


def round_tool_result_payload(value: Any) -> Any:
    if isinstance(value, float):
        return _round_float(value, TOOL_RESULT_FLOAT_DECIMALS)
    if isinstance(value, list):
        return [round_tool_result_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: round_tool_result_payload(item) for key, item in value.items()}
    return value


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


SCADA_HARMONIC_TOOL_NAMES = {
    "wls_from_path",
    "get_parameter_context",
    "get_topology_context",
    "get_harmonic_context",
    "get_verification_snapshot",
    "correct_measurements_from_path",
    "correct_parameters_from_path",
    "correct_topology_from_path",
    "run_hse_from_path",
}


def scoped_tool_schemas(tool_names: Iterable[str]) -> list[dict[str, Any]]:
    allowed = set(tool_names)
    return sanitize_tool_schemas(
        [
            tool
            for tool in CANONICAL_POWER_TOOLS
            if str(tool.get("function", {}).get("name")) in allowed
        ]
    )


def scada_harmonic_tool_schemas() -> list[dict[str, Any]]:
    return scoped_tool_schemas(SCADA_HARMONIC_TOOL_NAMES)


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
        # ``lambdaN`` is laid out per branch as [R_k, X_k]; the two entries are
        # the series resistance and reactance multipliers of the same line,
        # not its two terminals.
        line_row0 = idx0 // 2 if idx0 >= 0 else None
        parameter = "R" if idx0 % 2 == 0 else "X"
        branch = branch_info[line_row0] if line_row0 is not None and 0 <= line_row0 < len(branch_info) else {}
        evidence.append(
            {
                "lambda_index0": int(idx0),
                "line_row0": int(line_row0) if line_row0 is not None else None,
                "from_bus": _maybe_int(branch.get("from_bus")),
                "to_bus": _maybe_int(branch.get("to_bus")),
                "parameter": parameter if branch else "unknown",
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
        # The residuals handed to this helper are *normalized* residuals.
        # Their squared sum is not the WLS objective and is not chi-square
        # distributed with m - n degrees of freedom, so it must not be used as
        # a stand-in for the global statistic.
        raise ValueError(
            "global_residual_sum (raw WLS objective e'R^-1 e) is required; "
            "the squared sum of normalized residuals is not a chi-square statistic"
        )

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
    active_branch_info = tool_payload.get("branch_info")
    branch_info = active_branch_info if isinstance(active_branch_info, list) else meta.get("branch_info", [])
    summary: dict[str, Any] = {
        "success": bool(tool_payload.get("success", True)),
        "top_residuals": [] if force_empty_evidence else build_residual_evidence(residuals, index_map),
        "top_lagrange": [] if force_empty_evidence else build_lambda_evidence(lambda_values, branch_info),
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
        "sse_reduction_vs_null": _maybe_float(tool_payload.get("sse_reduction_vs_null")),
        "fundamental_voltage_source": tool_payload.get("fundamental_voltage_source"),
    }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_three_phase_nlm_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    groups = list(tool_payload.get("top_hif_groups") or [])[:TOPK_EVIDENCE]
    compact_groups = []
    for item in groups:
        if not isinstance(item, Mapping):
            continue
        compact_groups.append(
            {
                "rank": _maybe_int(item.get("rank")),
                "branch_row0": _maybe_int(item.get("branch_row0")),
                "line_index1": _maybe_int(item.get("line_index1")),
                "dss_element": item.get("dss_element"),
                "from_bus": _maybe_int(item.get("from_bus")),
                "to_bus": _maybe_int(item.get("to_bus")),
                "score": _maybe_float(item.get("score")),
            }
        )
    summary = {
        "success": bool(tool_payload.get("success", True)),
        "converged": bool(tool_payload.get("converged", tool_payload.get("success", True))),
        "top_hif_groups": compact_groups,
        "detected": bool(tool_payload.get("detected", False)),
    }
    if tool_payload.get("suspected_phase") is not None:
        summary["suspected_phase"] = str(tool_payload.get("suspected_phase"))
    if isinstance(tool_payload.get("phase_scores"), Mapping):
        summary["phase_scores"] = tool_payload.get("phase_scores")
    # Terminal-current evidence (per-phase branch currents): line differential
    # separation, closed-form position/resistance, and unbalance-source ranking.
    for key in (
        "separation_ratio",
        "max_line_differential_pu",
        "differential_detected",
        "differential_detection_floor_pu",
        "scan_count",
        "aggregation",
        "per_scan_line_votes",
        "per_scan_phase_votes",
    ):
        if tool_payload.get(key) is not None:
            summary[key] = tool_payload.get(key)
    summary["terminal_current_estimate"] = compact_terminal_current_estimate(
        tool_payload.get("terminal_current_estimate")
    )
    localization = tool_payload.get("localization")
    if isinstance(localization, Mapping):
        summary["localization"] = {
            "method": localization.get("method"),
            "bus_1based": _maybe_int(localization.get("bus_1based")),
            "phase_power_spread_rel": _maybe_float(localization.get("phase_power_spread_rel")),
            "separation_ratio": _maybe_float(localization.get("separation_ratio")),
            "significant": (
                bool(localization.get("significant"))
                if localization.get("significant") is not None
                else None
            ),
            "significant_bus_count": _maybe_int(localization.get("significant_bus_count")),
        }
    source_buses = tool_payload.get("top_unbalance_source_buses")
    if isinstance(source_buses, list):
        summary["top_unbalance_source_buses"] = [
            {
                "rank": _maybe_int(item.get("rank")),
                "bus": _maybe_int(item.get("bus")),
                "phase_power_spread_rel": _maybe_float(item.get("phase_power_spread_rel")),
                "negative_sequence_current_pu": _maybe_float(
                    item.get("negative_sequence_current_pu")
                ),
            }
            for item in source_buses[:TOPK_EVIDENCE]
            if isinstance(item, Mapping)
        ]
    null_test = tool_payload.get("line_differential_null")
    if isinstance(null_test, Mapping):
        summary["line_differential_null"] = {
            "max_line_differential_pu": _maybe_float(null_test.get("max_line_differential_pu")),
            "differential_detection_floor_pu": _maybe_float(
                null_test.get("differential_detection_floor_pu")
            ),
            "hif_like_differential_present": bool(
                null_test.get("hif_like_differential_present", False)
            ),
        }
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    if tool_payload.get("method"):
        summary["method"] = str(tool_payload["method"])
    return round_assistant_payload(prune_none(summary))


def compact_terminal_current_estimate(payload: Any) -> dict[str, Any] | None:
    """Model-visible summary of a two-terminal closed-form HIF estimate."""
    if not isinstance(payload, Mapping):
        return None
    compact = {
        "method": payload.get("method"),
        "branch_row0": _maybe_int(payload.get("branch_row0")),
        "dss_element": payload.get("dss_element"),
        "phase": payload.get("phase"),
        "phase_confident": (
            bool(payload.get("phase_confident"))
            if payload.get("phase_confident") is not None
            else None
        ),
        "alpha_from_from_bus": _maybe_float(payload.get("alpha_from_from_bus")),
        "alpha_interval": payload.get("alpha_interval"),
        "r_hif_pu": _maybe_float(payload.get("r_hif_pu")),
        "r_hif_pu_interval": payload.get("r_hif_pu_interval"),
        "i_hif_pu": _maybe_float(payload.get("i_hif_pu")),
        "fit_mismatch_pu": _maybe_float(payload.get("fit_mismatch_pu")),
        "scan_count": _maybe_int(payload.get("scan_count")),
        "agreeing_scan_count": _maybe_int(payload.get("agreeing_scan_count")),
        "phase_votes": payload.get("phase_votes"),
        "differential_detected": (
            bool(payload.get("differential_detected"))
            if payload.get("differential_detected") is not None
            else None
        ),
        "line_rank": _maybe_int(payload.get("line_rank")),
    }
    return prune_none(compact)


def summarize_hif_parameter_estimate_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    estimated = tool_payload.get("estimated") if isinstance(tool_payload.get("estimated"), Mapping) else {}
    fit = tool_payload.get("fit") if isinstance(tool_payload.get("fit"), Mapping) else {}
    uncertainty = tool_payload.get("uncertainty") if isinstance(tool_payload.get("uncertainty"), Mapping) else {}
    observability = tool_payload.get("observability") if isinstance(tool_payload.get("observability"), Mapping) else {}
    top_candidates = tool_payload.get("top_parameter_candidates")
    compact_candidates = []
    if isinstance(top_candidates, list):
        for item in top_candidates[:TOPK_EVIDENCE]:
            if not isinstance(item, Mapping):
                continue
            compact_candidates.append(
                {
                    "rank": _maybe_int(item.get("rank")),
                    "alpha_from_from_bus": _maybe_float(item.get("alpha_from_from_bus")),
                    "distance_percent_from_from_bus": _maybe_float(item.get("distance_percent_from_from_bus")),
                    "phase": item.get("phase"),
                    "r_hif_pu": _maybe_float(item.get("r_hif_pu")),
                    "r_hif_pu_range": item.get("r_hif_pu_range"),
                    "score": _maybe_float(item.get("score")),
                }
            )
    summary = {
        "success": bool(tool_payload.get("success", True)),
        "synthetic_oracle": bool(tool_payload.get("synthetic_oracle", False)),
        "method": tool_payload.get("method"),
        "candidate_branch_row0": _maybe_int(tool_payload.get("candidate_branch_row0")),
        "dss_element": tool_payload.get("dss_element"),
        "from_bus": _maybe_int(tool_payload.get("from_bus")),
        "to_bus": _maybe_int(tool_payload.get("to_bus")),
        "scan_count": _maybe_int(tool_payload.get("scan_count")),
        "selected_scan_count": _maybe_int(tool_payload.get("selected_scan_count")),
        "selected_scan_indices": tool_payload.get(
            "selected_scan_indices",
            tool_payload.get("scan_selection", {}).get("selected_scan_indices")
            if isinstance(tool_payload.get("scan_selection"), Mapping)
            else None,
        ),
        "parameter_identifiable": bool(tool_payload.get("parameter_identifiable", False)),
        "terminal_current_estimate": compact_terminal_current_estimate(
            tool_payload.get("terminal_current_estimate")
        ),
        "estimated": {
            "alpha_from_from_bus": _maybe_float(estimated.get("alpha_from_from_bus")),
            "distance_percent_from_from_bus": _maybe_float(estimated.get("distance_percent_from_from_bus")),
            "phase": estimated.get("phase"),
            "r_hif_pu": _maybe_float(estimated.get("r_hif_pu")),
            "r_hif_pu_median": _maybe_float(estimated.get("r_hif_pu_median")),
            "r_hif_pu_range": estimated.get("r_hif_pu_range"),
            "r_hif_ohm": _maybe_float(estimated.get("r_hif_ohm")),
            "resistance_model": estimated.get("resistance_model"),
            "g_hif_siemens": _maybe_float(estimated.get("g_hif_siemens")),
            "i_hif_amp": _maybe_float(estimated.get("i_hif_amp")),
            "p_hif_kw": _maybe_float(estimated.get("p_hif_kw")),
            "q_hif_kvar": _maybe_float(estimated.get("q_hif_kvar")),
        },
        "fit": {
            "weighted_residual_norm": _maybe_float(fit.get("weighted_residual_norm")),
            "multiscan_weighted_residual_norm": _maybe_float(
                fit.get("multiscan_weighted_residual_norm")
            ),
            "residual_reduction_vs_no_refinement": _maybe_float(
                fit.get("residual_reduction_vs_no_refinement")
            ),
            "relative_residual_improvement": _maybe_float(
                fit.get("relative_residual_improvement")
            ),
            "localization_certainty": fit.get("localization_certainty"),
            "ambiguity": bool(fit.get("ambiguity", False)),
            "model_mismatch_suspected": bool(fit.get("model_mismatch_suspected", False)),
        },
        "observability": {
            "parameter_dimension": _maybe_int(observability.get("parameter_dimension")),
            "effective_rank": _maybe_int(observability.get("effective_rank")),
            "smallest_singular_value": _maybe_float(observability.get("smallest_singular_value")),
            "condition_number": _maybe_float(observability.get("condition_number")),
            "alpha_log_r_correlation": _maybe_float(observability.get("alpha_log_r_correlation")),
            "information_gain_vs_best_single_scan": _maybe_float(
                observability.get("information_gain_vs_best_single_scan")
            ),
            "scan_diversity_score": _maybe_float(observability.get("scan_diversity_score")),
            "status": observability.get("status"),
            "parameter_identifiable": bool(observability.get("parameter_identifiable", False)),
            "diagnostic_method": observability.get("diagnostic_method"),
            "diagnostic_complete": bool(observability.get("diagnostic_complete", True)),
            "diagnostic_scan_count": _maybe_int(observability.get("diagnostic_scan_count")),
            "diagnostic_failed_scan_count": _maybe_int(
                observability.get("diagnostic_failed_scan_count")
            ),
        },
        "uncertainty": {
            "near_best_alpha_interval": uncertainty.get("near_best_alpha_interval", uncertainty.get("alpha_ci90")),
            "near_best_r_hif_pu_interval": uncertainty.get(
                "near_best_r_hif_pu_interval",
                uncertainty.get("r_hif_pu_ci90"),
            ),
            "interval_method": uncertainty.get("interval_method"),
        },
        "top_parameter_candidates": compact_candidates,
    }
    if isinstance(tool_payload.get("phase_scores"), Mapping):
        summary["phase_scores"] = tool_payload.get("phase_scores")
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
        "available_context_tools": tool_payload.get("available_context_tools"),
        "allowed_next_tools": tool_payload.get("allowed_next_tools"),
        "note": tool_payload.get("note"),
    }
    if tool_payload.get("success") is False:
        summary["success"] = False
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_topology_context_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "breaker_context": tool_payload.get("breaker_context"),
        "note": tool_payload.get("note"),
    }
    if tool_payload.get("success") is False:
        summary["success"] = False
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_harmonic_context_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    measurements = tool_payload.get("harmonic_measurements") or []
    summary = {
        "case_path": tool_payload.get("case_path"),
        "measurement_count": len(measurements) if isinstance(measurements, list) else None,
        "harmonic_orders": tool_payload.get("harmonic_orders"),
        "available_context_tools": tool_payload.get("available_context_tools"),
        "allowed_next_tools": tool_payload.get("allowed_next_tools"),
        "note": tool_payload.get("note"),
    }
    if tool_payload.get("success") is False:
        summary["success"] = False
    if tool_payload.get("error"):
        summary["error"] = str(tool_payload["error"])
    return round_assistant_payload(prune_none(summary))


def summarize_verification_snapshot_payload(tool_payload: Mapping[str, Any]) -> dict[str, Any]:
    z_obs = tool_payload.get("z_obs") or []
    summary = {
        "case_path": tool_payload.get("case_path"),
        "measurement_count": len(z_obs) if isinstance(z_obs, list) else None,
        "available_context_tools": tool_payload.get("available_context_tools"),
        "allowed_next_tools": tool_payload.get("allowed_next_tools"),
        "note": tool_payload.get("note"),
        "stage": tool_payload.get("stage"),
    }
    if tool_payload.get("success") is False:
        summary["success"] = False
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
    if tool_name == "run_three_phase_nlm_from_path":
        return summarize_three_phase_nlm_payload(tool_result)
    if tool_name in {
        "estimate_hif_location_magnitude_from_path",
        "estimate_hif_location_magnitude_multiscan_from_path",
    }:
        return summarize_hif_parameter_estimate_payload(tool_result)
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
    nested_tool_context = hidden.get("tool_context")

    def hidden_tool_context(name: str) -> Any:
        direct = hidden.get(name)
        if direct is not None:
            return direct
        if isinstance(nested_tool_context, Mapping):
            return nested_tool_context.get(name)
        return None

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

    if tool_name == "run_three_phase_nlm_from_path":
        source = hidden_tool_context("hif_context")
        source_from_hidden = isinstance(source, dict)
        if not source_from_hidden:
            source = latest_user_payload_with_keys(messages, ("nlm_diagnostic",))
        if isinstance(source, dict):
            if source.get("case_path") and (source_from_hidden or not hydrated.get("case_path")):
                hydrated["case_path"] = source["case_path"]
                notes.append("hydrated_hif_case_path")
            if "nlm_diagnostic" not in hydrated and isinstance(source.get("nlm_diagnostic"), dict):
                hydrated["nlm_diagnostic"] = source["nlm_diagnostic"]
                notes.append("hydrated_hif_nlm_diagnostic")
            label = source.get("label")
            if "target_branch_row0" not in hydrated and isinstance(label, Mapping):
                target = _maybe_int(label.get("branch_row0"))
                if target is not None:
                    hydrated["target_branch_row0"] = target
                    notes.append("hydrated_hif_target_branch")
            if "target_dss_element" not in hydrated and isinstance(label, Mapping) and label.get("dss_element"):
                hydrated["target_dss_element"] = label.get("dss_element")
                notes.append("hydrated_hif_target_element")
            for key in (
                "pristine_model_dir",
                "faulted_model_dir",
                "phase",
                "r_hif_ohm",
                "load_scale",
            ):
                if key not in hydrated and source.get(key) is not None:
                    hydrated[key] = source.get(key)
                    notes.append(f"hydrated_hif_{key}")
            # Observable three-phase telemetry lets the NLM tool localize the
            # faulted line and phase from the snapshot instead of a stored
            # diagnostic; hydrate it from the bound context or the latest
            # user payload that carries it.
            for key in ("three_phase_voltages", "three_phase_branch_currents"):
                if key in hydrated:
                    continue
                channel_source = (
                    source
                    if isinstance(source.get(key), list) and source.get(key)
                    else latest_user_payload_with_keys(messages, (key,))
                )
                if isinstance(channel_source, dict) and isinstance(channel_source.get(key), list):
                    hydrated[key] = channel_source[key]
                    notes.append(f"hydrated_hif_nlm_{key}")
                    if (
                        key == "three_phase_branch_currents"
                        and "branch_current_sigma_pu" not in hydrated
                        and channel_source.get("branch_current_sigma_pu") is not None
                    ):
                        hydrated["branch_current_sigma_pu"] = channel_source[
                            "branch_current_sigma_pu"
                        ]

    if tool_name == "estimate_hif_location_magnitude_from_path":
        source = hidden_tool_context("hif_context")
        source_from_hidden = isinstance(source, dict)
        if not source_from_hidden:
            source = latest_user_payload_with_keys(messages, ("z_obs",))
        if not isinstance(source, dict):
            source = {}
        if source.get("case_path") and (source_from_hidden or not hydrated.get("case_path")):
            hydrated["case_path"] = source["case_path"]
            notes.append("hydrated_hif_estimator_case_path")
        if "z_obs" not in hydrated:
            z_source = source if isinstance(source.get("z_obs"), list) else latest_user_payload_with_keys(messages, ("z_obs",))
            if isinstance(z_source, dict) and isinstance(z_source.get("z_obs"), list):
                hydrated["z_obs"] = z_source["z_obs"]
                notes.append("hydrated_hif_estimator_z_obs")
        if "three_phase_voltages" not in hydrated:
            v_source = source if isinstance(source.get("three_phase_voltages"), list) else latest_user_payload_with_keys(
                messages,
                ("three_phase_voltages",),
            )
            if isinstance(v_source, dict) and isinstance(v_source.get("three_phase_voltages"), list):
                hydrated["three_phase_voltages"] = v_source["three_phase_voltages"]
                notes.append("hydrated_hif_estimator_three_phase_voltages")
        if "three_phase_branch_currents" not in hydrated:
            i_source = (
                source
                if isinstance(source.get("three_phase_branch_currents"), list)
                else latest_user_payload_with_keys(messages, ("three_phase_branch_currents",))
            )
            if isinstance(i_source, dict) and isinstance(
                i_source.get("three_phase_branch_currents"), list
            ):
                hydrated["three_phase_branch_currents"] = i_source["three_phase_branch_currents"]
                notes.append("hydrated_hif_estimator_three_phase_branch_currents")
                if (
                    "branch_current_sigma_pu" not in hydrated
                    and i_source.get("branch_current_sigma_pu") is not None
                ):
                    hydrated["branch_current_sigma_pu"] = i_source["branch_current_sigma_pu"]
        if "candidate_branch_row0" not in hydrated:
            nlm_source = latest_tool_payload_with_keys(
                messages,
                ("top_hif_groups",),
                tool_name="run_three_phase_nlm_from_path",
            )
            groups = nlm_source.get("top_hif_groups") if isinstance(nlm_source, dict) else None
            if isinstance(groups, list) and groups and isinstance(groups[0], Mapping):
                branch = _maybe_int(groups[0].get("branch_row0"))
                if branch is not None:
                    hydrated["candidate_branch_row0"] = branch
                    notes.append("hydrated_hif_estimator_candidate_branch_from_nlm")
        for key in ("pristine_model_dir", "load_scale"):
            if key not in hydrated and source.get(key) is not None:
                hydrated[key] = source.get(key)
                notes.append(f"hydrated_hif_estimator_{key}")

    if tool_name == "estimate_hif_location_magnitude_multiscan_from_path":
        source = hidden_tool_context("hif_context")
        source_from_hidden = isinstance(source, dict)
        if not source_from_hidden:
            source = latest_user_payload_with_keys(messages, ("scan_window_path",))
        if not isinstance(source, dict):
            source = {}
        if "scan_window_path" not in hydrated and source.get("scan_window_path"):
            hydrated["scan_window_path"] = source["scan_window_path"]
            notes.append("hydrated_hif_multiscan_window_path")
        if "scans" not in hydrated and isinstance(source.get("scans"), list):
            hydrated["scans"] = source["scans"]
            notes.append("hydrated_hif_multiscan_scans")
        if "sigma_z" not in hydrated and isinstance(source.get("sigma_z"), list):
            hydrated["sigma_z"] = source["sigma_z"]
            notes.append("hydrated_hif_multiscan_sigma_z")
        if source.get("case_path") and (source_from_hidden or not hydrated.get("case_path")):
            hydrated["case_path"] = source["case_path"]
            notes.append("hydrated_hif_multiscan_case_path")
        if "candidate_branch_row0" not in hydrated:
            nlm_source = latest_tool_payload_with_keys(
                messages,
                ("top_hif_groups",),
                tool_name="run_three_phase_nlm_from_path",
            )
            groups = nlm_source.get("top_hif_groups") if isinstance(nlm_source, dict) else None
            if isinstance(groups, list) and groups and isinstance(groups[0], Mapping):
                branch = _maybe_int(groups[0].get("branch_row0"))
                if branch is not None:
                    hydrated["candidate_branch_row0"] = branch
                    notes.append("hydrated_hif_multiscan_candidate_branch_from_nlm")
        if "pristine_model_dir" not in hydrated and source.get("pristine_model_dir") is not None:
            hydrated["pristine_model_dir"] = source.get("pristine_model_dir")
            notes.append("hydrated_hif_multiscan_pristine_model_dir")

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
