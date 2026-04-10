#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised SFT trace builder for the power-system diagnostic agent.

Key upgrades over the original:
- Aligns the prompt, available tools, and final target schema.
- Centralizes MCP JSON parsing and mock payload generation.
- Normalizes scenario naming (`negative` -> `no_error`).
- Produces a more informative final JSON target with explicit evidence.
- Uses deterministic hash-based train/valid/test splitting without rereading the output.
- Avoids heuristic "undo bias/scale" no-error synthesis by default, because the paired
  generator already emits explicit negative samples.

The script still emits OpenAI-style chat JSONL suitable for SFT with tool use.
It does not emit hidden chain-of-thought; the final assistant response is structured JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
from tqdm import tqdm

from trace_protocol import (
    EVIDENCE_ABS_THRESHOLD,
    ERROR_FAMILIES,
    MEASUREMENT_ORDER,
    NO_ERROR_RATIO_MAX,
    SYSTEM_PROMPT,
    build_global_metrics,
    build_lambda_evidence,
    build_residual_evidence,
    hydrate_tool_arguments,
    json_compact,
    latest_user_payload,
    normalize_error_family,
    resolve_case_path_alias,
    round_assistant_payload,
    round_tool_arguments,
    round_user_payload,
    summarize_hse_payload,
    summarize_tool_result_for_conversation,
    summarize_wls_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BuilderConfig:
    samples_path: Path
    meta_path: Path
    imbalance_samples_path: Optional[Path]
    imbalance_meta_path: Optional[Path]
    case_name: Optional[str]
    endpoint: str
    out_path: Path
    analysis_out_path: Optional[Path]
    mock: bool
    seed: int
    add_no_error: int
    with_correction: bool
    corr_max_iter: int
    corr_tol: float
    timeout_s: int = 60


# ----------------------------- low-level helpers -----------------------------


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha_short(x: Any) -> str:
    return hashlib.sha1(json.dumps(x, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]


def as_tool_return_text(obj: Mapping[str, Any]) -> str:
    return json_compact(obj)


def normalize_scenario(scenario: str) -> str:
    if scenario in ("negative", "no_error"):
        return "no_error"
    return scenario


def visible_case_id(case_ref: Any) -> str:
    text = str(case_ref or "case")
    name = text.replace("\\", "/").split("/")[-1]
    if name.endswith(".m"):
        name = name[:-2]
    return name or "case"


def make_case_alias(base_case_id: str, tag: str, sid: str) -> str:
    return f"{base_case_id}::{tag}::{sha_short(sid)}"


def runtime_case_reference(case_ref: Any) -> str:
    text = str(case_ref or "")
    if not text:
        return text
    path = Path(text)
    if path.is_absolute():
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            try:
                return Path(os.path.relpath(path, REPO_ROOT)).as_posix()
            except ValueError:
                return text
    return path.as_posix() if ("/" in text or "\\" in text) else text


def status_to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"closed", "true", "1", "on", "in_service"}:
            return True
        if s in {"open", "false", "0", "off", "out_of_service"}:
            return False
    return None


def _maybe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        parsed = float(x)
        if not np.isfinite(parsed):
            return None
        return parsed
    except Exception:
        return None


def _meta_core(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case": meta.get("case"),
        "baseMVA": meta.get("baseMVA"),
        "nb": meta.get("nb"),
        "nl": meta.get("nl"),
        "index_map": meta.get("index_map"),
        "measurement_order": meta.get("measurement_order"),
        "branch_info": meta.get("branch_info"),
    }


def load_sample_sources(config: BuilderConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_meta = json.loads(config.meta_path.read_text(encoding="utf-8"))
    samples = list(iter_jsonl(config.samples_path))

    if config.imbalance_samples_path and config.imbalance_meta_path:
        imbalance_meta = json.loads(config.imbalance_meta_path.read_text(encoding="utf-8"))
        if _meta_core(base_meta) != _meta_core(imbalance_meta):
            raise ValueError("Primary and imbalance metadata do not match on core measurement fields.")
        imbalance_samples = [
            rec
            for rec in iter_jsonl(config.imbalance_samples_path)
            if normalize_scenario(str(rec.get("scenario", ""))) == "three_phase_imbalance"
        ]
        samples.extend(imbalance_samples)

    return base_meta, samples


def call_tool_json(endpoint: str, name: str, arguments: Mapping[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """
    Directly calls the Python tool functions from mcp_server.matpower_server 
    to bypass FastMCP v2 SSE networking issues on Windows.
    Map the LLM-chosen tool names (from prompt) to the underlying Python functions.
    """
    import sys
    import os
    
    # Ensure mcp_server is in path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    import mcp_server.matpower_server as mp_tools
    
    try:
        # FastMCP decorates these functions, wrapping them in a Tool object. 
        # We must call .fn() to execute the actual original python routine locally.
        
        import numpy as np
        def _make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_make_serializable(v) for v in obj]
            elif hasattr(np, "generic") and isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj

        # LLM calls wls_from_path with {case_path, z}
        if name in ("wls_from_path", "wls_from_text"):
            if "case_text" in arguments:
                result = mp_tools.wls_from_text.fn(
                    case_name=arguments.get("case_path", arguments.get("case_name", "temp")),
                    case_text=arguments["case_text"],
                    z=arguments.get("z", arguments.get("z_obs", []))
                )
            else:
                result = mp_tools.wls_from_path.fn(
                    case_path=arguments["case_path"],
                    z=arguments.get("z", arguments.get("z_obs", []))
                )
            return _make_serializable(result)
            
        elif name in ("correct_parameters_from_path", "correct_parameter_error"):
            result = mp_tools.correct_parameters_from_path.fn(
                case_path=arguments["case_path"],
                line_index=arguments["line_index"],
                z_scans=arguments.get("z_scans", []),
                initial_states=arguments.get("initial_states")
            )
            return _make_serializable(result)
            
        elif name == "correct_measurements_from_path":
            result = mp_tools.correct_measurements_from_path.fn(
                case_path=arguments["case_path"],
                z=arguments.get("z", arguments.get("z_obs", [])),
                suspect_group=arguments["suspect_group"]
            )
            return _make_serializable(result)
            
        elif name == "correct_topology_from_path":
            result = mp_tools.correct_topology_from_path.fn(
                case_path=arguments["case_path"],
                cb_name=arguments["cb_name"],
                desired_status=arguments["desired_status"]
            )
            return _make_serializable(result)
            
        elif name == "run_hse_from_path":
            result = mp_tools.run_hse_from_path.fn(
                case_path=arguments["case_path"],
                harmonic_measurements=arguments["harmonic_measurements"],
                harmonic_orders=arguments.get("harmonic_orders")
            )
            return _make_serializable(result)
            
        else:
            return {"success": False, "error": f"Unknown tool name locally: {name}"}
            
    except Exception as e:
        return {"success": False, "error": f"Local tool execution failed: {e}"}


def call_backend_tool(
    endpoint: str,
    name: str,
    arguments: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    hidden_context: Mapping[str, Any],
    *,
    timeout: int,
) -> Dict[str, Any]:
    exec_args, _ = hydrate_tool_arguments(name, arguments, messages, hidden_context=hidden_context)
    exec_args = dict(exec_args)
    if "case_path" in exec_args:
        exec_args["case_path"] = resolve_case_path_alias(exec_args["case_path"], hidden_context)
    return call_tool_json(endpoint, name, exec_args, timeout=timeout)


# ----------------------------- mock payloads -----------------------------


def make_mock_wls_payload(
    rec: Mapping[str, Any],
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    scenario = normalize_scenario(str(rec.get("scenario", "")))
    m = int(meta["nb"]) * 3 + int(meta["nl"]) * 4
    r = np.zeros(m, dtype=float)
    lam = np.full(int(meta["nl"]) * 2, 0.12, dtype=float)

    if scenario == "measurement_error":
        lab = rec.get("label", {})
        ch = lab.get("channel")
        subtype = lab.get("subtype")
        if subtype == "single_gross_outlier" and isinstance(lab.get("index"), int):
            r[int(lab["index"])] = 6.5
        elif isinstance(lab.get("indices"), list):
            for i in lab["indices"]:
                r[int(i)] = 4.5
        elif ch in idx_map:
            sl = idx_map[ch]
            r[sl.start:sl.stop] = 3.2
        else:
            r[rng.integers(0, m)] = 5.5

    elif scenario == "parameter_error":
        line_row0 = int(rec.get("label", {}).get("line_row", 0))
        if 0 <= 2 * line_row0 + 1 < lam.size:
            lam[2 * line_row0] = 5.0
            lam[2 * line_row0 + 1] = 6.0
        r += rng.normal(0.0, 0.15, size=m)

    elif scenario == "topology_error":
        for ch, level in (("Pf", 4.2), ("Qf", 4.0), ("Pt", 3.8), ("Qt", 3.6)):
            sl = idx_map[ch]
            r[sl.start:sl.stop] = level
        lam += 0.1

    elif scenario == "three_phase_imbalance":
        for ch, level in (("Vm", 2.4), ("Pinj", 2.0), ("Qinj", 2.0), ("Pf", 4.1), ("Qf", 3.9), ("Pt", 3.8), ("Qt", 3.6)):
            sl = idx_map[ch]
            r[sl.start:sl.stop] = level

    elif scenario == "harmonic_anomaly":
        r = rng.normal(0.0, 0.4, size=m)
        vm = idx_map["Vm"]
        r[vm.start:vm.stop] = rng.normal(1.4, 0.4, size=vm.stop - vm.start)
        norm = np.linalg.norm(r)
        if norm > 1e-9:
            r *= (200.0 / float(np.sum(r**2))) ** 0.5
        lam[:] = 0.08

    else:  # no_error
        r = rng.normal(0.0, 0.08, size=m)
        lam[:] = 0.08

    payload = {
        "success": True,
        "r": r.tolist(),
        "lambdaN": lam.tolist(),
        "global_residual_sum": float(np.sum(r**2)),
    }
    payload["global_residual_threshold"] = build_global_metrics(r, meta)["global_residual_threshold"]
    return payload


def make_mock_hse_payload(rec: Mapping[str, Any]) -> Dict[str, Any]:
    lab = rec.get("label", {})
    src = int(lab.get("source_bus", 3))
    thd = float(lab.get("thd_target", 10.0))
    return {
        "success": True,
        "best_candidate_bus_1based": src,
        "estimated_thd_percent": {str(src): thd},
        "notes": "Harmonic source identified.",
    }


# ----------------------------- conversation builders -----------------------------


def make_tool_call(tool_name: str, call_id: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "id": call_id,
        "function": {
            "name": tool_name,
            "arguments": json_compact(round_tool_arguments(arguments)),
        },
    }


def append_helper_tool_result(
    messages: List[Dict[str, Any]],
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    *,
    tool_name: str,
    call_id: str,
    arguments: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    helper_call = make_tool_call(tool_name, call_id, arguments)
    messages.append({"role": "assistant", "tool_calls": [helper_call]})
    messages.append(
        {
            "role": "tool",
            "tool_call_id": helper_call["id"],
            "name": tool_name,
            "content": as_tool_return_text(
                summarize_tool_result_for_conversation(
                    tool_name,
                    payload,
                    meta,
                    idx_map,
                )
            ),
        }
    )


def make_user_payload(rec: Mapping[str, Any], meta: Mapping[str, Any], case_path: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "case_path": case_path,
        "z_obs": rec["z_obs"],
        "measurement_order": MEASUREMENT_ORDER,
        "index_map": meta["index_map"],
        "branch_info": meta.get("branch_info"),
        "meta_hint": {
            "nb": meta["nb"],
            "nl": meta["nl"],
            "baseMVA": meta.get("baseMVA"),
            "case": meta.get("case"),
        },
        "task": "Call wls_from_path first, then decide whether any correction or follow-up tool is required.",
    }
    if normalize_scenario(str(rec.get("scenario", ""))) == "three_phase_imbalance":
        payload["note"] = (
            "This snapshot is a 1φ-equivalent operator vector (phase-A voltage magnitudes plus 3φ totals). "
            "If imbalance is suspected, request three-phase substation voltages."
        )
    return round_user_payload(payload)


def make_parameter_followup_payload(rec: Mapping[str, Any], case_path: str) -> Dict[str, Any]:
    label = rec.get("label", {})
    return round_user_payload(
        {
            "case_path": case_path,
            "z_scans": rec.get("z_scans", []),
            "initial_states": rec.get("initial_states", []),
            "note": "Repeated measurement scans and initial states for parameter correction.",
            "suspect_line": {
                "line_row0": _maybe_int(label.get("line_row")),
                "from_bus": _maybe_int(label.get("from_bus")),
                "to_bus": _maybe_int(label.get("to_bus")),
            },
        }
    )


def make_topology_followup_payload(rec: Mapping[str, Any], case_path: str) -> Dict[str, Any]:
    label = rec.get("label", {})
    return round_user_payload(
        {
            "case_path": case_path,
            "breaker_context": {
                "substation": _maybe_int(label.get("substation")),
                "cb_name": label.get("cb_name"),
                "observed_status": label.get("new_status"),
                "desired_status": status_to_bool(label.get("old_status")),
                "desired_status_text": label.get("old_status"),
            },
            "note": "Compact breaker context for the suspect substation.",
        }
    )


def make_harmonic_followup_payload(rec: Mapping[str, Any], case_path: str) -> Dict[str, Any]:
    return round_user_payload(
        {
            "case_path": case_path,
            "harmonic_measurements": rec.get("harmonic_measurements", []),
            "harmonic_orders": rec.get("harmonic_orders", []),
            "note": "Harmonic measurements for HSE follow-up.",
        }
    )


def make_imbalance_followup_payload(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return round_user_payload(
        {
            "three_phase_voltages": rec.get("three_phase_voltages", []),
            "note": "Per-bus three-phase VLN voltage measurements from substations.",
        }
    )


def make_verification_snapshot_payload(case_path: str, z_obs: Sequence[float], note: str, stage: str) -> Dict[str, Any]:
    return round_user_payload(
        {
            "case_path": case_path,
            "z_obs": list(z_obs),
            "note": note,
            "stage": stage,
        }
    )


def choose_measurement_suspect_group(
    rec: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    tool_payload: Mapping[str, Any],
) -> List[int]:
    r_vec = np.asarray(tool_payload.get("r", []), dtype=float)
    if r_vec.size:
        try:
            return [int(np.nanargmax(np.abs(r_vec)))]
        except Exception:
            pass

    lab = rec.get("label", {})
    if isinstance(lab.get("index"), int):
        return [int(lab["index"])]
    if isinstance(lab.get("indices"), list):
        return [int(i) for i in lab["indices"]]
    ch = lab.get("channel")
    if ch in idx_map:
        sl = idx_map[ch]
        return list(range(sl.start, sl.stop))
    return []


def build_verification_summary(
    verify_payload: Optional[Mapping[str, Any]],
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    *,
    pre_action_payload: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(verify_payload, Mapping):
        return None
    compact = summarize_wls_payload(verify_payload, meta, idx_map)
    gm = compact["global_metrics"]
    post_ratio = _maybe_float(gm.get("global_residual_ratio"))
    pre_ratio = None
    if isinstance(pre_action_payload, Mapping):
        pre_compact = summarize_wls_payload(pre_action_payload, meta, idx_map)
        pre_ratio = _maybe_float((pre_compact.get("global_metrics") or {}).get("global_residual_ratio"))
    executed = bool(compact.get("success", True))
    improved = bool(executed and pre_ratio is not None and post_ratio is not None and post_ratio < pre_ratio)
    resolved = bool(executed and post_ratio is not None and post_ratio < 1.0)
    return round_assistant_payload(
        {
            "post_action_global_residual_sum": gm["global_residual_sum"],
            "post_action_global_residual_threshold": gm["global_residual_threshold"],
            "post_action_global_residual_ratio": gm["global_residual_ratio"],
            "post_action_executed": executed,
            "post_action_improved": improved,
            "post_action_resolved": resolved,
        }
    )


def build_final_target(
    rec: Mapping[str, Any],
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    primary_wls: Mapping[str, Any],
    *,
    measurement_suspect_group: Optional[List[int]] = None,
    verification_payload: Optional[Mapping[str, Any]] = None,
    hse_payload: Optional[Mapping[str, Any]] = None,
    correction_tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    scenario = normalize_scenario(str(rec.get("scenario", "")))
    label = rec.get("label", {})
    primary_summary = summarize_wls_payload(
        primary_wls,
        meta,
        idx_map,
        force_empty_evidence=scenario == "no_error",
    )

    evidence = {
        "global_metrics": primary_summary["global_metrics"],
        "top_residuals": primary_summary["top_residuals"],
        "top_lagrange": primary_summary["top_lagrange"],
    }

    verdict = {
        "has_error": scenario != "no_error",
        "error_family": scenario if scenario in ERROR_FAMILIES else "no_error",
        "confidence": 0.98 if scenario == "no_error" else 0.95,
    }
    if scenario in {"measurement_error", "parameter_error", "topology_error"}:
        verdict["confidence"] = 0.99

    suspect_location: Dict[str, Any]
    action: Dict[str, Any]
    summary: str

    if scenario == "measurement_error":
        details: Dict[str, Any] = {
            "channel": label.get("channel"),
            "index0": _maybe_int(label.get("index")),
            "indices0": [int(i) for i in label.get("indices", [])] if isinstance(label.get("indices"), list) else None,
            "subtype": label.get("subtype"),
        }
        details = {k: v for k, v in details.items() if v not in (None, [], {})}
        suspect_location = {"domain": "measurement", "details": details}
        action = {
            "applied_tool": correction_tool_name,
            "arguments_hint": {"suspect_group": measurement_suspect_group} if correction_tool_name else None,
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": build_verification_summary(
                verification_payload,
                meta,
                idx_map,
                pre_action_payload=primary_wls,
            ),
        }
        summary = "Residual evidence is concentrated in one measurement location/channel."

    elif scenario == "parameter_error":
        suspect_location = {
            "domain": "parameter",
            "details": {
                "line_row0": _maybe_int(label.get("line_row")),
                "line_index1": _maybe_int(label.get("line_row")) + 1 if _maybe_int(label.get("line_row")) is not None else None,
                "from_bus": _maybe_int(label.get("from_bus")),
                "to_bus": _maybe_int(label.get("to_bus")),
                "subtype": label.get("subtype"),
            },
        }
        action = {
            "applied_tool": correction_tool_name,
            "arguments_hint": (
                {"line_index": _maybe_int(label.get("line_row")) + 1}
                if correction_tool_name and _maybe_int(label.get("line_row")) is not None
                else None
            ),
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": build_verification_summary(
                verification_payload,
                meta,
                idx_map,
                pre_action_payload=primary_wls,
            ),
        }
        summary = "Top normalized Lagrange multipliers concentrate on one branch, consistent with a parameter issue."

    elif scenario == "topology_error":
        suspect_location = {
            "domain": "topology",
            "details": {
                "substation": _maybe_int(label.get("substation")),
                "cb_name": label.get("cb_name"),
                "observed_status": label.get("new_status"),
                "expected_status": label.get("old_status"),
            },
        }
        action = {
            "applied_tool": correction_tool_name,
            "arguments_hint": (
                {
                    "cb_name": label.get("cb_name"),
                    "desired_status": status_to_bool(label.get("old_status")),
                    "desired_status_text": label.get("old_status"),
                }
                if correction_tool_name and label.get("cb_name")
                else None
            ),
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": build_verification_summary(
                verification_payload,
                meta,
                idx_map,
                pre_action_payload=primary_wls,
            ),
        }
        summary = "Residuals are widespread and consistent with a model/topology mismatch."

    elif scenario == "three_phase_imbalance":
        have_three_phase = bool(rec.get("three_phase_voltages"))
        suspect_location = {
            "domain": "imbalance",
            "details": {"unbalance_bus": _maybe_int(label.get("unbalance_bus"))},
        }
        action = {
            "applied_tool": None,
            "arguments_hint": None,
            "request_more_data": not have_three_phase,
            "requested_data": None if have_three_phase else ["three_phase_substation_voltages"],
            "verification_summary": None,
        }
        summary = "Residual pattern suggests possible three-phase imbalance rather than a single bad scalar measurement."

    elif scenario == "harmonic_anomaly":
        details = {"source_bus": _maybe_int(label.get("source_bus"))}
        if isinstance(hse_payload, Mapping):
            compact_hse = summarize_hse_payload(hse_payload)
            details["hse_best_candidate_bus_1based"] = compact_hse.get("best_candidate_bus_1based")
            details["best_candidate_thd_percent"] = compact_hse.get("best_candidate_thd_percent")
            details["ranking_top5"] = compact_hse.get("ranking_top5")
        suspect_location = {"domain": "harmonic", "details": details}
        action = {
            "applied_tool": "run_hse_from_path",
            "arguments_hint": {"harmonic_measurements": "bound_via_get_harmonic_context"},
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": None,
        }
        summary = "The global residual is elevated without a single dominant bad measurement; harmonic follow-up is warranted."

    else:
        suspect_location = {"domain": "none", "details": {}}
        action = {
            "applied_tool": None,
            "arguments_hint": None,
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": None,
        }
        summary = "No error pattern is strong enough to justify a corrective action."

    return round_assistant_payload(
        {
        "verdict": verdict,
        "evidence": evidence,
        "suspect_location": suspect_location,
        "action": action,
        "summary": summary,
        }
    )


def rejection_reason(final_target: Mapping[str, Any]) -> Optional[str]:
    verdict = final_target.get("verdict", {}) if isinstance(final_target, Mapping) else {}
    evidence = final_target.get("evidence", {}) if isinstance(final_target, Mapping) else {}
    action = final_target.get("action", {}) if isinstance(final_target, Mapping) else {}
    family = normalize_error_family(verdict.get("error_family"))
    gm = evidence.get("global_metrics", {}) if isinstance(evidence, Mapping) else {}
    ratio = _maybe_float(gm.get("global_residual_ratio"))
    if family is None:
        return "invalid_error_family"
    if _maybe_float(gm.get("global_residual_sum")) is None:
        return "missing_global_residual_sum"
    if _maybe_float(gm.get("global_residual_threshold")) is None:
        return "missing_global_residual_threshold"
    if ratio is None:
        return "missing_global_residual_ratio"

    top_residuals = evidence.get("top_residuals", [])
    top_lagrange = evidence.get("top_lagrange", [])
    verification = action.get("verification_summary") if isinstance(action, Mapping) else None

    if family == "no_error":
        if ratio >= NO_ERROR_RATIO_MAX:
            return "borderline_no_error"
        if top_residuals or top_lagrange:
            return "no_error_has_significant_evidence"
    elif family == "measurement_error":
        if not top_residuals:
            return "measurement_error_missing_residual_evidence"
        if verification is None:
            return "measurement_error_missing_verification"
    elif family == "parameter_error":
        if not top_lagrange:
            return "parameter_error_missing_lagrange_evidence"
        if verification is None:
            return "parameter_error_missing_verification"
    elif family == "topology_error":
        if verification is None:
            return "topology_error_missing_verification"
    elif family == "harmonic_anomaly":
        details = final_target.get("suspect_location", {}).get("details", {})
        if not details or details.get("hse_best_candidate_bus_1based") is None:
            return "harmonic_anomaly_missing_hse_result"
    elif family == "three_phase_imbalance":
        if bool(action.get("request_more_data")):
            return "three_phase_imbalance_missing_followup"

    return None


# ----------------------------- main builder -----------------------------


def build_sft(config: BuilderConfig) -> None:
    rng_std = random.Random(config.seed)
    rng_np = np.random.default_rng(config.seed)

    meta, samples = load_sample_sources(config)
    idx_map = {k: slice(v[0], v[1]) for k, v in meta["index_map"].items()}
    base_case_backend = meta.get("case") if config.case_name in (None, "auto") else config.case_name
    base_case_visible = visible_case_id(base_case_backend)

    if config.add_no_error > 0:
        negatives = [s for s in samples if normalize_scenario(str(s.get("scenario", ""))) == "no_error"]
        extra: List[Dict[str, Any]] = []
        for _ in range(config.add_no_error):
            if not negatives:
                break
            base = dict(rng_std.choice(negatives))
            base["id"] = f"ne_{rng_std.randrange(10**12)}"
            base["scenario"] = "no_error"
            base["label"] = {"error_type": "no_error"}
            extra.append(base)
        samples.extend(extra)

    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    rejected_rows: list[dict[str, Any]] = []

    with config.out_path.open("w", encoding="utf-8") as fout_all:
        for rec in tqdm(samples, desc="Building SFT traces"):
            sid = str(rec["id"])
            scenario = normalize_scenario(str(rec["scenario"]))
            runtime_context: Dict[str, Any] = {
                "case_aliases": {base_case_visible: runtime_case_reference(base_case_backend)},
                "tool_context": {},
            }
            hidden_context: Dict[str, Any] = {
                "case_aliases": runtime_context["case_aliases"],
            }

            user_payload = make_user_payload(rec, meta, base_case_visible)
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json_compact(user_payload)},
            ]

            wls_call_args = {"case_path": base_case_visible}
            wls_call = make_tool_call("wls_from_path", f"call_wls_{sha_short(sid)}", wls_call_args)
            messages.append({"role": "assistant", "tool_calls": [wls_call]})

            try:
                if config.mock:
                    wls_payload = make_mock_wls_payload(rec, meta, idx_map, rng_np)
                else:
                    wls_payload = call_backend_tool(
                        config.endpoint,
                        "wls_from_path",
                        wls_call_args,
                        messages,
                        hidden_context,
                        timeout=config.timeout_s,
                    )
            except Exception:
                n_skipped += 1
                continue

            wls_summary = summarize_tool_result_for_conversation(
                "wls_from_path",
                wls_payload,
                meta,
                idx_map,
                force_empty_evidence=scenario == "no_error",
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": wls_call["id"],
                    "name": "wls_from_path",
                    "content": as_tool_return_text(wls_summary),
                }
            )

            correction_tool_name: Optional[str] = None
            measurement_suspect_group: Optional[List[int]] = None
            verification_payload: Optional[Dict[str, Any]] = None
            hse_payload: Optional[Dict[str, Any]] = None

            if not wls_payload.get("success", True):
                rejected_rows.append({"id": sid, "scenario": scenario, "reason": "initial_wls_failed"})
                continue

            if scenario == "measurement_error" and config.with_correction:
                measurement_suspect_group = choose_measurement_suspect_group(rec, idx_map, wls_payload)
                correction_tool_name = "correct_measurements_from_path"

                corr_call_args = {
                    "case_path": base_case_visible,
                    "suspect_group": measurement_suspect_group,
                }
                corr_call = make_tool_call(
                    "correct_measurements_from_path",
                    f"call_corr_meas_{sha_short(sid)}",
                    corr_call_args,
                )
                messages.append({"role": "assistant", "tool_calls": [corr_call]})

                try:
                    corr_payload = (
                        {"success": False, "error": "mock correction not implemented"}
                        if config.mock
                        else call_backend_tool(
                            config.endpoint,
                            "correct_measurements_from_path",
                            corr_call_args,
                            messages,
                            hidden_context,
                            timeout=config.timeout_s,
                        )
                    )
                except Exception as exc:
                    corr_payload = {"success": False, "error": str(exc)}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": corr_call["id"],
                        "name": "correct_measurements_from_path",
                        "content": as_tool_return_text(
                            summarize_tool_result_for_conversation(
                                "correct_measurements_from_path",
                                corr_payload,
                                meta,
                                idx_map,
                            )
                        ),
                    }
                )

                cms = corr_payload.get("corrected_measurements") or []
                chosen = None
                if cms:
                    if measurement_suspect_group:
                        preferred = set(int(i) for i in measurement_suspect_group)
                        for item in cms:
                            if int(item.get("index0", -1)) in preferred:
                                chosen = item
                                break
                    if chosen is None:
                        chosen = max(
                            cms,
                            key=lambda e: abs(float(e.get("estimated_error", 0.0))),
                            default=None,
                        )
                if chosen is not None:
                    z2 = list(rec["z_obs"])
                    idx0 = int(chosen.get("index0"))
                    corrected = float(chosen.get("corrected"))
                    if 0 <= idx0 < len(z2):
                        z2[idx0] = corrected
                        verify_stage = "post_measurement_correction"
                        verification_case_visible = make_case_alias(base_case_visible, "measurement_verify", sid)
                        runtime_context["case_aliases"][verification_case_visible] = runtime_case_reference(
                            base_case_backend
                        )
                        verification_snapshot = make_verification_snapshot_payload(
                            verification_case_visible,
                            z2,
                            "Post-correction verification snapshot.",
                            verify_stage,
                        )
                        runtime_context["tool_context"].setdefault("verification_snapshots", {})[verify_stage] = verification_snapshot
                        hidden_context["snapshot_context"] = verification_snapshot
                        append_helper_tool_result(
                            messages,
                            meta,
                            idx_map,
                            tool_name="get_verification_snapshot",
                            call_id=f"call_ctx_verify_meas_{sha_short(sid)}",
                            arguments={"case_path": verification_case_visible, "stage": verify_stage},
                            payload=verification_snapshot,
                        )
                        verify_call = make_tool_call(
                            "wls_from_path",
                            f"call_wls_verify_{sha_short(sid)}",
                            {"case_path": verification_case_visible},
                        )
                        messages.append({"role": "assistant", "tool_calls": [verify_call]})
                        try:
                            verification_payload = (
                                make_mock_wls_payload({"scenario": "no_error"}, meta, idx_map, rng_np)
                                if config.mock
                                else call_backend_tool(
                                    config.endpoint,
                                    "wls_from_path",
                                    {"case_path": verification_case_visible},
                                    messages,
                                    hidden_context,
                                    timeout=config.timeout_s,
                                )
                            )
                        except Exception as exc:
                            verification_payload = {"success": False, "error": str(exc)}
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": verify_call["id"],
                                "name": "wls_from_path",
                                "content": as_tool_return_text(
                                    summarize_tool_result_for_conversation(
                                        "wls_from_path",
                                        verification_payload,
                                        meta,
                                        idx_map,
                                    )
                                ),
                            }
                        )

            elif scenario == "parameter_error" and config.with_correction:
                line_row0 = _maybe_int(rec.get("label", {}).get("line_row"))
                if line_row0 is not None and isinstance(rec.get("z_scans"), list) and isinstance(rec.get("initial_states"), list):
                    correction_tool_name = "correct_parameters_from_path"
                    correction_case_backend = rec.get("parameter_error_case_path") or rec.get("correction_case_path") or base_case_backend
                    correction_case_visible = base_case_visible
                    if correction_case_backend != base_case_backend:
                        correction_case_visible = make_case_alias(base_case_visible, "parameter_case", sid)
                        runtime_context["case_aliases"][correction_case_visible] = runtime_case_reference(
                            correction_case_backend
                        )
                    parameter_context = make_parameter_followup_payload(rec, correction_case_visible)
                    runtime_context["tool_context"]["parameter_context"] = parameter_context
                    hidden_context["parameter_context"] = parameter_context
                    append_helper_tool_result(
                        messages,
                        meta,
                        idx_map,
                        tool_name="get_parameter_context",
                        call_id=f"call_ctx_param_{sha_short(sid)}",
                        arguments={"case_path": base_case_visible, "line_index": int(line_row0) + 1},
                        payload=parameter_context,
                    )
                    param_args = {"case_path": correction_case_visible, "line_index": int(line_row0) + 1}
                    param_call = make_tool_call(
                        "correct_parameters_from_path",
                        f"call_corr_param_{sha_short(sid)}",
                        param_args,
                    )
                    messages.append({"role": "assistant", "tool_calls": [param_call]})
                    try:
                        param_payload = (
                            {"success": False, "error": "mock parameter correction not implemented"}
                            if config.mock
                            else call_backend_tool(
                                config.endpoint,
                                "correct_parameters_from_path",
                                param_args,
                                messages,
                                hidden_context,
                                timeout=config.timeout_s,
                            )
                        )
                    except Exception as exc:
                        param_payload = {"success": False, "error": str(exc)}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": param_call["id"],
                            "name": "correct_parameters_from_path",
                            "content": as_tool_return_text(
                                summarize_tool_result_for_conversation(
                                    "correct_parameters_from_path",
                                    param_payload,
                                    meta,
                                    idx_map,
                                )
                            ),
                        }
                    )
                    if param_payload.get("success", True) and isinstance(rec.get("z_true"), list):
                        verify_stage = "post_parameter_correction"
                        verification_case_visible = make_case_alias(base_case_visible, "parameter_verify", sid)
                        runtime_context["case_aliases"][verification_case_visible] = runtime_case_reference(
                            base_case_backend
                        )
                        verification_snapshot = make_verification_snapshot_payload(
                            verification_case_visible,
                            rec["z_true"],
                            "Post-parameter-correction verification snapshot.",
                            verify_stage,
                        )
                        runtime_context["tool_context"].setdefault("verification_snapshots", {})[verify_stage] = verification_snapshot
                        hidden_context["snapshot_context"] = verification_snapshot
                        append_helper_tool_result(
                            messages,
                            meta,
                            idx_map,
                            tool_name="get_verification_snapshot",
                            call_id=f"call_ctx_verify_param_{sha_short(sid)}",
                            arguments={"case_path": verification_case_visible, "stage": verify_stage},
                            payload=verification_snapshot,
                        )
                        verify_call = make_tool_call(
                            "wls_from_path",
                            f"call_wls_verify_param_{sha_short(sid)}",
                            {"case_path": verification_case_visible},
                        )
                        messages.append({"role": "assistant", "tool_calls": [verify_call]})
                        try:
                            verification_payload = (
                                make_mock_wls_payload({"scenario": "no_error"}, meta, idx_map, rng_np)
                                if config.mock
                                else call_backend_tool(
                                    config.endpoint,
                                    "wls_from_path",
                                    {"case_path": verification_case_visible},
                                    messages,
                                    hidden_context,
                                    timeout=config.timeout_s,
                                )
                            )
                        except Exception as exc:
                            verification_payload = {"success": False, "error": str(exc)}
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": verify_call["id"],
                                "name": "wls_from_path",
                                "content": as_tool_return_text(
                                    summarize_tool_result_for_conversation(
                                        "wls_from_path",
                                        verification_payload,
                                        meta,
                                        idx_map,
                                    )
                                ),
                            }
                        )

            elif scenario == "topology_error" and config.with_correction:
                lab = rec.get("label", {})
                cb_name = lab.get("cb_name")
                desired_status = status_to_bool(lab.get("old_status"))
                if cb_name and desired_status is not None:
                    correction_tool_name = "correct_topology_from_path"
                    topology_context = make_topology_followup_payload(rec, base_case_visible)
                    runtime_context["tool_context"]["topology_context"] = topology_context
                    hidden_context["topology_context"] = topology_context
                    append_helper_tool_result(
                        messages,
                        meta,
                        idx_map,
                        tool_name="get_topology_context",
                        call_id=f"call_ctx_topo_{sha_short(sid)}",
                        arguments={"case_path": base_case_visible},
                        payload=topology_context,
                    )
                    topo_args = {"case_path": base_case_visible, "cb_name": cb_name, "desired_status": desired_status}
                    topo_call = make_tool_call(
                        "correct_topology_from_path",
                        f"call_corr_topo_{sha_short(sid)}",
                        topo_args,
                    )
                    messages.append({"role": "assistant", "tool_calls": [topo_call]})
                    try:
                        topo_payload = (
                            {"success": False, "error": "mock topology correction not implemented"}
                            if config.mock
                            else call_backend_tool(
                                config.endpoint,
                                "correct_topology_from_path",
                                topo_args,
                                messages,
                                hidden_context,
                                timeout=config.timeout_s,
                            )
                        )
                    except Exception as exc:
                        topo_payload = {"success": False, "error": str(exc)}

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": topo_call["id"],
                            "name": "correct_topology_from_path",
                            "content": as_tool_return_text(
                                summarize_tool_result_for_conversation(
                                    "correct_topology_from_path",
                                    topo_payload,
                                    meta,
                                    idx_map,
                                )
                            ),
                        }
                    )

                    case_path_verify = make_case_alias(base_case_visible, "topology_verify", sid)
                    z_verify = None
                    if "corrected_model_path" in rec and "z_true_full_model" in rec:
                        runtime_context["case_aliases"][case_path_verify] = runtime_case_reference(
                            rec["corrected_model_path"]
                        )
                        z_verify = rec["z_true_full_model"]
                    elif isinstance(topo_payload.get("z_corrected"), list):
                        runtime_context["case_aliases"][case_path_verify] = runtime_case_reference(base_case_backend)
                        z_verify = topo_payload["z_corrected"]

                    if isinstance(z_verify, list):
                        verify_stage = "post_topology_correction"
                        verification_snapshot = make_verification_snapshot_payload(
                            case_path_verify,
                            z_verify,
                            "Post-topology-correction verification snapshot.",
                            verify_stage,
                        )
                        runtime_context["tool_context"].setdefault("verification_snapshots", {})[verify_stage] = verification_snapshot
                        hidden_context["snapshot_context"] = verification_snapshot
                        append_helper_tool_result(
                            messages,
                            meta,
                            idx_map,
                            tool_name="get_verification_snapshot",
                            call_id=f"call_ctx_verify_topo_{sha_short(sid)}",
                            arguments={"case_path": case_path_verify, "stage": verify_stage},
                            payload=verification_snapshot,
                        )
                        verify_call = make_tool_call(
                            "wls_from_path",
                            f"call_wls_verify_topo_{sha_short(sid)}",
                            {"case_path": case_path_verify},
                        )
                        messages.append({"role": "assistant", "tool_calls": [verify_call]})
                        try:
                            verification_payload = (
                                make_mock_wls_payload({"scenario": "no_error"}, meta, idx_map, rng_np)
                                if config.mock
                                else call_backend_tool(
                                    config.endpoint,
                                    "wls_from_path",
                                    {"case_path": case_path_verify},
                                    messages,
                                    hidden_context,
                                    timeout=config.timeout_s,
                                )
                            )
                        except Exception as exc:
                            verification_payload = {"success": False, "error": str(exc)}
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": verify_call["id"],
                                "name": "wls_from_path",
                                "content": as_tool_return_text(
                                    summarize_tool_result_for_conversation(
                                        "wls_from_path",
                                        verification_payload,
                                        meta,
                                        idx_map,
                                    )
                                ),
                            }
                        )

            elif scenario == "harmonic_anomaly":
                if rec.get("harmonic_measurements"):
                    harmonic_context = make_harmonic_followup_payload(rec, base_case_visible)
                    runtime_context["tool_context"]["harmonic_context"] = harmonic_context
                    hidden_context["harmonic_context"] = harmonic_context
                    append_helper_tool_result(
                        messages,
                        meta,
                        idx_map,
                        tool_name="get_harmonic_context",
                        call_id=f"call_ctx_harm_{sha_short(sid)}",
                        arguments={"case_path": base_case_visible},
                        payload=harmonic_context,
                    )
                    hse_args = {"case_path": base_case_visible}
                    hse_call = make_tool_call(
                        "run_hse_from_path",
                        f"call_hse_{sha_short(sid)}",
                        hse_args,
                    )
                    messages.append({"role": "assistant", "tool_calls": [hse_call]})
                    try:
                        hse_payload = (
                            make_mock_hse_payload(rec)
                            if config.mock
                            else call_backend_tool(
                                config.endpoint,
                                "run_hse_from_path",
                                hse_args,
                                messages,
                                hidden_context,
                                timeout=config.timeout_s,
                            )
                        )
                    except Exception as exc:
                        hse_payload = {"success": False, "error": str(exc)}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": hse_call["id"],
                            "name": "run_hse_from_path",
                            "content": as_tool_return_text(
                                summarize_tool_result_for_conversation(
                                    "run_hse_from_path",
                                    hse_payload,
                                    meta,
                                    idx_map,
                                )
                            ),
                        }
                    )

            elif scenario == "three_phase_imbalance":
                three_phase = rec.get("three_phase_voltages")
                if isinstance(three_phase, list) and three_phase:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": (
                                    "Please provide three-phase substation voltages "
                                    "to continue imbalance assessment."
                                ),
                            },
                            {
                                "role": "user",
                                "content": json_compact(make_imbalance_followup_payload(rec)),
                            },
                        ]
                    )

            final = build_final_target(
                rec,
                meta,
                idx_map,
                wls_payload,
                measurement_suspect_group=measurement_suspect_group,
                verification_payload=verification_payload,
                hse_payload=hse_payload,
                correction_tool_name=correction_tool_name,
            )
            reject_reason = rejection_reason(final)
            if reject_reason is not None:
                rejected_rows.append(
                    {
                        "id": sid,
                        "scenario": scenario,
                        "reason": reject_reason,
                        "global_metrics": final["evidence"]["global_metrics"],
                    }
                )
                continue

            messages.append({"role": "assistant", "content": json_compact(final)})

            row = {"messages": messages, "runtime_context": runtime_context}
            line = json.dumps(row, ensure_ascii=False)
            fout_all.write(line + "\n")
            n_written += 1

    print(f"Wrote combined SFT file: {config.out_path}")
    if config.analysis_out_path is not None:
        config.analysis_out_path.parent.mkdir(parents=True, exist_ok=True)
        with config.analysis_out_path.open("w", encoding="utf-8") as handle:
            for row in rejected_rows:
                handle.write(json.dumps(round_assistant_payload(row), ensure_ascii=False) + "\n")
        print(f"Wrote rejected-trace analysis file: {config.analysis_out_path}")
    print(f"Rejected traces: {len(rejected_rows)}")
    if n_skipped:
        print(f"Skipped {n_skipped} examples due to MCP errors/timeouts.")
    print(f"Total written: {n_written}")


def parse_args() -> BuilderConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="out_measurements_balanced/samples.jsonl")
    p.add_argument("--meta", default="out_measurements_balanced/meta.json")
    p.add_argument("--imbalance-samples", default="")
    p.add_argument("--imbalance-meta", default="")
    p.add_argument("--case", default="auto", choices=["auto", "case14", "case118"])
    p.add_argument("--endpoint", default="http://localhost:3929/tools")
    p.add_argument("--out", default="data/sft_with_tools.jsonl")
    p.add_argument("--analysis-out", default="data/sft_with_tools.rejected.jsonl")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-error", type=int, default=0, help="Extra replicated clean controls from existing negative samples.")
    p.add_argument("--no-correction", action="store_true")
    p.add_argument("--corr-iters", type=int, default=2)
    p.add_argument("--corr-tol", type=float, default=1e-3)
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    return BuilderConfig(
        samples_path=Path(args.samples),
        meta_path=Path(args.meta),
        imbalance_samples_path=Path(args.imbalance_samples) if args.imbalance_samples else None,
        imbalance_meta_path=Path(args.imbalance_meta) if args.imbalance_meta else None,
        case_name=None if args.case == "auto" else args.case,
        endpoint=args.endpoint,
        out_path=Path(args.out),
        analysis_out_path=Path(args.analysis_out) if args.analysis_out else None,
        mock=bool(args.mock),
        seed=int(args.seed),
        add_no_error=int(args.no_error),
        with_correction=not bool(args.no_correction),
        corr_max_iter=int(args.corr_iters),
        corr_tol=float(args.corr_tol),
        timeout_s=int(args.timeout),
    )


if __name__ == "__main__":
    build_sft(parse_args())
