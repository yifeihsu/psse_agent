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
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests
from tqdm import tqdm

try:
    from scripts.trigger_hse import chi2_threshold as _chi2_threshold
except Exception:
    _chi2_threshold = None


MEASUREMENT_ORDER = ["Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt"]
ERROR_FAMILIES = [
    "measurement_error",
    "parameter_error",
    "topology_error",
    "three_phase_imbalance",
    "harmonic_anomaly",
    "no_error",
]


DECISION_SCHEMA_TEXT = {
    "verdict": {
        "has_error": "boolean",
        "error_family": ERROR_FAMILIES,
        "confidence": "number in [0,1]",
    },
    "evidence": {
        "global_metrics": {
            "global_residual_sum": "number or null",
            "global_residual_threshold": "number or null",
            "global_residual_ratio": "number or null",
        },
        "top_residuals": [
            {
                "index0": "int",
                "channel": "string",
                "channel_offset": "int",
                "value": "number",
            }
        ],
        "top_lagrange": [
            {
                "lambda_index0": "int",
                "line_row0": "int or null",
                "from_bus": "int or null",
                "to_bus": "int or null",
                "terminal": "'from'|'to'|'unknown'",
                "value": "number",
            }
        ],
    },
    "suspect_location": {
        "domain": "measurement|parameter|topology|harmonic|imbalance|none",
        "details": "object",
    },
    "action": {
        "recommended_tool": "tool name or null",
        "arguments_hint": "object or null",
        "request_more_data": "boolean",
        "requested_data": "array[string] or null",
        "verification_summary": "object or null",
    },
    "summary": "short factual summary string",
}

SYSTEM_PROMPT = (
    "You are a power-system state-estimation diagnostic agent.\n"
    "You must begin with `wls_from_path` for every snapshot.\n"
    "Available tools:\n"
    "- `wls_from_path(case_path, z)`: weighted least-squares state estimation with bad-data indicators.\n"
    "- `correct_measurements_from_path(case_path, z, suspect_group, ...)`: measurement correction.\n"
    "- `correct_parameters_from_path(case_path, line_index, z_scans, initial_states, ...)`: line-parameter correction.\n"
    "- `correct_topology_from_path(case_path, cb_name, desired_status)`: topology correction.\n"
    "- `run_hse_from_path(case_path, harmonic_measurements, harmonic_orders?)`: harmonic state estimation.\n\n"
    "Decision policy:\n"
    "1. Use concentrated large normalized residuals to localize likely measurement errors.\n"
    "2. Use large normalized Lagrange multipliers concentrated on one branch to suspect parameter errors.\n"
    "3. Use widespread residual patterns to suspect topology mismatch or three-phase imbalance.\n"
    "4. If the global residual is elevated without a dominant bad measurement and harmonic measurements are available, call `run_hse_from_path`.\n"
    "5. If three-phase data are required, explicitly request three-phase substation voltages.\n\n"
    "Return only strict JSON with this structure:\n"
    f"{json.dumps(DECISION_SCHEMA_TEXT, ensure_ascii=False)}\n"
    "Do not reveal chain-of-thought. Report only observable evidence and the final decision."
)


@dataclass(frozen=True)
class BuilderConfig:
    samples_path: Path
    meta_path: Path
    case_name: Optional[str]
    endpoint: str
    out_path: Path
    mock: bool
    seed: int
    add_no_error: int
    with_correction: bool
    corr_max_iter: int
    corr_tol: float
    topk_r: int = 5
    topk_lambda: int = 5
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


def json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def as_tool_return_text(obj: Mapping[str, Any]) -> str:
    return json_compact(obj)


def normalize_scenario(scenario: str) -> str:
    if scenario in ("negative", "no_error"):
        return "no_error"
    return scenario


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


def stable_split(sample_id: str) -> str:
    """80/10/10 deterministic hash split."""
    h = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "valid"
    return "test"


def topk_abs(values: Sequence[float], k: int) -> List[Tuple[int, float]]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    order = np.argsort(-np.abs(arr))[: min(k, arr.size)]
    return [(int(i), float(arr[i])) for i in order]


def channel_from_index(index0: int, index_map: Mapping[str, slice]) -> Tuple[str, int]:
    for ch in MEASUREMENT_ORDER:
        sl = index_map[ch]
        if sl.start <= index0 < sl.stop:
            return ch, index0 - sl.start
    return "unknown", index0


def estimate_global_threshold(residuals: Sequence[float], nb: Optional[int]) -> Optional[float]:
    if _chi2_threshold is None:
        return None
    r = np.asarray(residuals, dtype=float)
    if r.size == 0 or nb is None:
        return None
    # Approximate dof for AC SE: m - (2*nb - 1). This is only a proxy.
    dof = max(int(r.size - (2 * int(nb) - 1)), 1)
    try:
        return float(_chi2_threshold(dof))
    except Exception:
        return None


def build_residual_evidence(
    residuals: Sequence[float],
    index_map: Mapping[str, slice],
    k: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx0, value in topk_abs(residuals, k):
        ch, off = channel_from_index(idx0, index_map)
        out.append(
            {
                "index0": int(idx0),
                "channel": ch,
                "channel_offset": int(off),
                "value": float(value),
            }
        )
    return out


def build_lambda_evidence(
    lambdaN: Sequence[float],
    branch_info: Sequence[Mapping[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx0, value in topk_abs(lambdaN, k):
        line_row0 = idx0 // 2 if idx0 >= 0 else None
        terminal = "from" if idx0 % 2 == 0 else "to"
        br = branch_info[line_row0] if line_row0 is not None and 0 <= line_row0 < len(branch_info) else {}
        out.append(
            {
                "lambda_index0": int(idx0),
                "line_row0": int(line_row0) if line_row0 is not None else None,
                "from_bus": _maybe_int(br.get("from_bus")),
                "to_bus": _maybe_int(br.get("to_bus")),
                "terminal": terminal if br else "unknown",
                "value": float(value),
            }
        )
    return out


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
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    except Exception:
        return None


import httpx

def _round_or_none(x: Any, ndigits: int = 6) -> Optional[float]:
    v = _maybe_float(x)
    if v is None:
        return None
    return round(v, ndigits)


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
    thr = estimate_global_threshold(r, _maybe_int(meta.get("nb")))
    if thr is not None:
        payload["global_residual_threshold"] = float(thr)
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
            "arguments": json_compact(arguments),
        },
    }


def make_user_payload(rec: Mapping[str, Any], meta: Mapping[str, Any], case_path: str) -> Dict[str, Any]:
    scenario = normalize_scenario(str(rec.get("scenario", "")))
    payload: Dict[str, Any] = {
        "case_path": case_path,
        "z_obs": rec["z_obs"],
        "measurement_order": MEASUREMENT_ORDER,
        "index_map": meta["index_map"],
        "meta_hint": {
            "nb": meta["nb"],
            "nl": meta["nl"],
            "baseMVA": meta.get("baseMVA"),
            "case": meta.get("case"),
        },
        "task": "Call wls_from_path first, then decide whether any correction or follow-up tool is required.",
    }
    if scenario == "harmonic_anomaly" and rec.get("harmonic_measurements"):
        payload["harmonic_measurements_available"] = True
        payload["harmonic_orders_available"] = bool(rec.get("harmonic_orders"))
    if scenario == "three_phase_imbalance":
        payload["note"] = (
            "This snapshot is a 1φ-equivalent operator vector (phase-A voltage magnitudes plus 3φ totals). "
            "If imbalance is suspected, request three-phase substation voltages."
        )
        payload["three_phase_voltages_available"] = bool(rec.get("three_phase_voltages"))
    return payload


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


def build_global_metrics(
    tool_payload: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    residuals = np.asarray(tool_payload.get("r", []), dtype=float)
    J = _maybe_float(tool_payload.get("global_residual_sum"))
    if J is None and residuals.size:
        J = float(np.sum(residuals**2))

    threshold = _maybe_float(tool_payload.get("global_residual_threshold"))
    if threshold is None:
        threshold = estimate_global_threshold(residuals, _maybe_int(meta.get("nb")))

    ratio = None
    if J is not None and threshold is not None and abs(threshold) > 1e-12:
        ratio = J / threshold

    return {
        "global_residual_sum": _round_or_none(J),
        "global_residual_threshold": _round_or_none(threshold),
        "global_residual_ratio": _round_or_none(ratio),
    }


def build_verification_summary(
    verify_payload: Optional[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(verify_payload, Mapping):
        return None
    gm = build_global_metrics(verify_payload, meta)
    return {
        "post_action_global_residual_sum": gm["global_residual_sum"],
        "post_action_global_residual_threshold": gm["global_residual_threshold"],
        "post_action_global_residual_ratio": gm["global_residual_ratio"],
        "post_action_success": bool(verify_payload.get("success", True)),
    }


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
    residuals = primary_wls.get("r", []) or []
    lambdaN = primary_wls.get("lambdaN", []) or []

    evidence = {
        "global_metrics": build_global_metrics(primary_wls, meta),
        "top_residuals": build_residual_evidence(residuals, idx_map, k=5),
        "top_lagrange": build_lambda_evidence(lambdaN, meta.get("branch_info", []), k=5),
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
            "recommended_tool": correction_tool_name,
            "arguments_hint": {"suspect_group": measurement_suspect_group} if correction_tool_name else None,
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": build_verification_summary(verification_payload, meta),
        }
        summary = "Residual evidence is concentrated in one measurement location/channel."

    elif scenario == "parameter_error":
        suspect_location = {
            "domain": "parameter",
            "details": {
                "line_row0": _maybe_int(label.get("line_row")),
                "from_bus": _maybe_int(label.get("from_bus")),
                "to_bus": _maybe_int(label.get("to_bus")),
                "subtype": label.get("subtype"),
            },
        }
        action = {
            "recommended_tool": correction_tool_name,
            "arguments_hint": (
                {"line_index": _maybe_int(label.get("line_row")) + 1}
                if correction_tool_name and _maybe_int(label.get("line_row")) is not None
                else None
            ),
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": build_verification_summary(verification_payload, meta),
        }
        summary = "Top normalized Lagrange multipliers concentrate on one branch, consistent with a parameter issue."

    elif scenario == "topology_error":
        suspect_location = {
            "domain": "topology",
            "details": {
                "substation": _maybe_int(label.get("substation")),
                "cb_name": label.get("cb_name"),
                "old_status": label.get("old_status"),
                "new_status": label.get("new_status"),
            },
        }
        action = {
            "recommended_tool": correction_tool_name,
            "arguments_hint": (
                {
                    "cb_name": label.get("cb_name"),
                    "desired_status": status_to_bool(label.get("old_status")),
                }
                if correction_tool_name and label.get("cb_name")
                else None
            ),
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": build_verification_summary(verification_payload, meta),
        }
        summary = "Residuals are widespread and consistent with a model/topology mismatch."

    elif scenario == "three_phase_imbalance":
        have_three_phase = bool(rec.get("three_phase_voltages"))
        suspect_location = {
            "domain": "imbalance",
            "details": {"unbalance_bus": _maybe_int(label.get("unbalance_bus"))},
        }
        action = {
            "recommended_tool": None,
            "arguments_hint": None,
            "request_more_data": not have_three_phase,
            "requested_data": None if have_three_phase else ["three_phase_substation_voltages"],
            "verification_summary": None,
        }
        summary = "Residual pattern suggests possible three-phase imbalance rather than a single bad scalar measurement."

    elif scenario == "harmonic_anomaly":
        details = {"source_bus": _maybe_int(label.get("source_bus"))}
        if isinstance(hse_payload, Mapping):
            details["hse_best_candidate_bus_1based"] = _maybe_int(hse_payload.get("best_candidate_bus_1based"))
            details["estimated_thd_percent"] = hse_payload.get("estimated_thd_percent")
        suspect_location = {"domain": "harmonic", "details": details}
        hse_hint = None
        if rec.get("harmonic_measurements"):
            hse_hint = {"harmonic_measurements": "provided_in_dialog"}
            if rec.get("harmonic_orders"):
                hse_hint["harmonic_orders"] = rec.get("harmonic_orders")
        action = {
            "recommended_tool": "run_hse_from_path",
            "arguments_hint": hse_hint,
            "request_more_data": not bool(rec.get("harmonic_measurements")),
            "requested_data": None if rec.get("harmonic_measurements") else ["harmonic_measurements"],
            "verification_summary": None,
        }
        summary = "The global residual is elevated without a single dominant bad measurement; harmonic follow-up is warranted."

    else:
        suspect_location = {"domain": "none", "details": {}}
        action = {
            "recommended_tool": None,
            "arguments_hint": None,
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": None,
        }
        summary = "No error pattern is strong enough to justify a corrective action."

    return {
        "verdict": verdict,
        "evidence": evidence,
        "suspect_location": suspect_location,
        "action": action,
        "summary": summary,
    }


# ----------------------------- main builder -----------------------------


def build_sft(config: BuilderConfig) -> None:
    rng_std = random.Random(config.seed)
    rng_np = np.random.default_rng(config.seed)

    meta = json.loads(config.meta_path.read_text(encoding="utf-8"))
    idx_map = {k: slice(v[0], v[1]) for k, v in meta["index_map"].items()}
    case_path = meta.get("case") if config.case_name in (None, "auto") else config.case_name

    samples = list(iter_jsonl(config.samples_path))

    # Optional class-balance helper: duplicate already clean negatives instead of trying to
    # "undo" measurement corruption heuristically.
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
    split_paths = {
        "train": config.out_path.with_name(config.out_path.stem + ".train.jsonl"),
        "valid": config.out_path.with_name(config.out_path.stem + ".valid.jsonl"),
        "test": config.out_path.with_name(config.out_path.stem + ".test.jsonl"),
    }

    n_written = 0
    n_skipped = 0

    with (
        config.out_path.open("w", encoding="utf-8") as fout_all,
        split_paths["train"].open("w", encoding="utf-8") as fout_train,
        split_paths["valid"].open("w", encoding="utf-8") as fout_valid,
        split_paths["test"].open("w", encoding="utf-8") as fout_test,
    ):
        split_writers = {"train": fout_train, "valid": fout_valid, "test": fout_test}

        for rec in tqdm(samples, desc="Building SFT traces"):
            sid = str(rec["id"])
            scenario = normalize_scenario(str(rec["scenario"]))
            z_obs = rec["z_obs"]

            user_payload = make_user_payload(rec, meta, case_path)

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json_compact(user_payload)},
            ]

            # ---- Step 1: mandatory WLS ----
            wls_call = make_tool_call(
                "wls_from_path",
                f"call_wls_{sha_short(sid)}",
                {"case_path": case_path, "z": z_obs},
            )
            messages.append({"role": "assistant", "tool_calls": [wls_call]})

            try:
                if config.mock:
                    wls_payload = make_mock_wls_payload(rec, meta, idx_map, rng_np)
                else:
                    wls_payload = call_tool_json(
                        config.endpoint,
                        "wls_from_path",
                        {"case_path": case_path, "z": z_obs},
                        timeout=config.timeout_s,
                    )
            except Exception as exc:
                n_skipped += 1
                continue

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": wls_call["id"],
                    "name": "wls_from_path",
                    "content": as_tool_return_text(wls_payload),
                }
            )

            correction_tool_name: Optional[str] = None
            measurement_suspect_group: Optional[List[int]] = None
            verification_payload: Optional[Dict[str, Any]] = None
            hse_payload: Optional[Dict[str, Any]] = None

            # ---- Scenario-specific follow-up ----
            if scenario == "measurement_error" and config.with_correction:
                measurement_suspect_group = choose_measurement_suspect_group(rec, idx_map, wls_payload)
                correction_tool_name = "correct_measurements_from_path"

                corr_args = {
                    "case_path": case_path,
                    "z": z_obs,
                    "suspect_group": measurement_suspect_group,
                    "enable_correction": True,
                    "max_correction_iterations": int(config.corr_max_iter),
                    "error_tolerance": float(config.corr_tol),
                }
                corr_call = make_tool_call(
                    "correct_measurements_from_path",
                    f"call_corr_meas_{sha_short(sid)}",
                    corr_args,
                )
                messages.append({"role": "assistant", "tool_calls": [corr_call]})

                try:
                    corr_payload = (
                        {"success": False, "error": "mock correction not implemented"}
                        if config.mock
                        else call_tool_json(
                            config.endpoint,
                            "correct_measurements_from_path",
                            {**corr_args, "R_variances_full": None},
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
                        "content": as_tool_return_text(corr_payload),
                    }
                )

                # optional verification pass
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
                    z2 = list(z_obs)
                    idx0 = int(chosen.get("index0"))
                    corrected = float(chosen.get("corrected"))
                    if 0 <= idx0 < len(z2):
                        z2[idx0] = corrected
                        verify_call = make_tool_call(
                            "wls_from_path",
                            f"call_wls_verify_{sha_short(sid)}",
                            {"case_path": case_path, "z": z2},
                        )
                        messages.append({"role": "assistant", "tool_calls": [verify_call]})
                        try:
                            verification_payload = (
                                make_mock_wls_payload({"scenario": "no_error"}, meta, idx_map, rng_np)
                                if config.mock
                                else call_tool_json(
                                    config.endpoint,
                                    "wls_from_path",
                                    {"case_path": case_path, "z": z2},
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
                                "content": as_tool_return_text(verification_payload),
                            }
                        )

            elif scenario == "parameter_error" and config.with_correction:
                line_row0 = _maybe_int(rec.get("label", {}).get("line_row"))
                z_scans = rec.get("z_scans")
                initial_states = rec.get("initial_states")
                if line_row0 is not None and isinstance(z_scans, list) and isinstance(initial_states, list):
                    correction_tool_name = "correct_parameters_from_path"
                    correction_case_path = rec.get("parameter_error_case_path") or rec.get("correction_case_path") or case_path
                    param_args = {
                        "case_path": correction_case_path,
                        "line_index": int(line_row0) + 1,  # correction tool expects 1-based branch row
                        "z_scans": z_scans,
                        "initial_states": initial_states,
                        "R_variances_full": None,
                    }
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
                            else call_tool_json(
                                config.endpoint,
                                "correct_parameters_from_path",
                                param_args,
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
                            "content": as_tool_return_text(param_payload),
                        }
                    )

            elif scenario == "topology_error" and config.with_correction:
                lab = rec.get("label", {})
                cb_name = lab.get("cb_name")
                desired_status = status_to_bool(lab.get("old_status"))
                if cb_name and desired_status is not None:
                    correction_tool_name = "correct_topology_from_path"
                    topo_args = {
                        "case_path": case_path,
                        "cb_name": cb_name,
                        "desired_status": desired_status,
                    }
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
                            else call_tool_json(
                                config.endpoint,
                                "correct_topology_from_path",
                                topo_args,
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
                            "content": as_tool_return_text(topo_payload),
                        }
                    )

                    case_path_verify = case_path
                    z_verify = None
                    if "corrected_model_path" in rec and "z_true_full_model" in rec:
                        case_path_verify = rec["corrected_model_path"]
                        z_verify = rec["z_true_full_model"]
                    elif isinstance(topo_payload.get("z_corrected"), list):
                        z_verify = topo_payload["z_corrected"]

                    if isinstance(z_verify, list):
                        verify_call = make_tool_call(
                            "wls_from_path",
                            f"call_wls_verify_topo_{sha_short(sid)}",
                            {"case_path": case_path_verify, "z": z_verify},
                        )
                        messages.append({"role": "assistant", "tool_calls": [verify_call]})
                        try:
                            verification_payload = (
                                make_mock_wls_payload({"scenario": "no_error"}, meta, idx_map, rng_np)
                                if config.mock
                                else call_tool_json(
                                    config.endpoint,
                                    "wls_from_path",
                                    {"case_path": case_path_verify, "z": z_verify},
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
                                "content": as_tool_return_text(verification_payload),
                            }
                        )

            elif scenario == "harmonic_anomaly":
                if rec.get("harmonic_measurements"):
                    hse_args = {"case_path": case_path, "harmonic_measurements": rec.get("harmonic_measurements", [])}
                    if isinstance(rec.get("harmonic_orders"), list) and rec.get("harmonic_orders"):
                        hse_args["harmonic_orders"] = rec.get("harmonic_orders")
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
                            else call_tool_json(
                                config.endpoint,
                                "run_hse_from_path",
                                hse_args,
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
                            "content": as_tool_return_text(hse_payload),
                        }
                    )

            elif scenario == "three_phase_imbalance":
                # Keep a multi-turn request/response trace if synthetic 3φ voltages are available.
                three_phase = rec.get("three_phase_voltages")
                if isinstance(three_phase, list) and three_phase:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": (
                                    "Please provide three-phase substation voltages "
                                    "(phase A/B/C magnitude-angle or equivalent phasor format) "
                                    "to continue imbalance assessment."
                                ),
                            },
                            {
                                "role": "user",
                                "content": json_compact(
                                    {
                                        "three_phase_voltages": three_phase,
                                        "note": "Per-bus 3φ substation voltages.",
                                    }
                                ),
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
            messages.append({"role": "assistant", "content": json_compact(final)})

            row = {"messages": messages}
            line = json.dumps(row, ensure_ascii=False)
            fout_all.write(line + "\n")
            split_writers[stable_split(sid)].write(line + "\n")
            n_written += 1

    print(f"Wrote combined SFT file: {config.out_path}")
    print(f"Wrote split files: {split_paths['train']}, {split_paths['valid']}, {split_paths['test']}")
    if n_skipped:
        print(f"Skipped {n_skipped} examples due to MCP errors/timeouts.")
    print(f"Total written: {n_written}")


def parse_args() -> BuilderConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="out_sft_measurements/samples.jsonl")
    p.add_argument("--meta", default="out_sft_measurements/meta.json")
    p.add_argument("--case", default="auto", choices=["auto", "case14", "case118"])
    p.add_argument("--endpoint", default="http://localhost:3929/tools")
    p.add_argument("--out", default="sft_with_tools_revised.jsonl")
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
        case_name=None if args.case == "auto" else args.case,
        endpoint=args.endpoint,
        out_path=Path(args.out),
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
