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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    summarize_three_phase_nlm_payload,
    summarize_tool_result_for_conversation,
    summarize_wls_payload,
)


KNOWN_CASE_DIRS = [
    REPO_ROOT / "out_measurements_balanced" / "cases_parameter_error",
    REPO_ROOT / "out_measurements_balanced" / "models_topology",
    REPO_ROOT / "out_measurements_balanced_topup" / "cases_parameter_error",
    REPO_ROOT / "out_measurements_balanced_topup" / "models_topology",
    REPO_ROOT / "out_sft_measurements" / "cases_parameter_error",
    REPO_ROOT / "out_sft_measurements" / "models_topology",
]
REPORT_FAMILY_ORDER = [
    "measurement_error",
    "parameter_error",
    "topology_error",
    "harmonic_anomaly",
    "high_impedance_fault",
]
CORRECTION_TOOL_ORDER = [
    "topology_error",
    "parameter_error",
    "measurement_error",
    "harmonic_anomaly",
]
MULTI_ERROR_TOOL_ORDER = REPORT_FAMILY_ORDER
TOOL_BY_FAMILY = {
    "measurement_error": "correct_measurements_from_path",
    "parameter_error": "correct_parameters_from_path",
    "topology_error": "correct_topology_from_path",
    "harmonic_anomaly": "run_hse_from_path",
    "high_impedance_fault": "run_three_phase_nlm_from_path",
}
HARDENING_MAX_EXAMPLES = 150
HARDENING_TEMPLATES = (
    "parameter_helper_unavailable",
    "harmonic_helper_unavailable",
    "verification_snapshot_unavailable",
)


@dataclass(frozen=True)
class BuilderConfig:
    samples_path: Path
    meta_path: Path
    imbalance_samples_path: Optional[Path]
    imbalance_meta_path: Optional[Path]
    hif_samples_path: Optional[Path]
    hif_meta_path: Optional[Path]
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
    allow_hif_metadata_fallback: bool
    hardening_examples: int
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
    normalized = text.replace("\\", "/")
    basename = Path(normalized).name
    if basename:
        for case_dir in KNOWN_CASE_DIRS:
            candidate = case_dir / basename
            if candidate.exists():
                try:
                    return candidate.relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    return str(candidate)
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


def multi_error_families(rec: Mapping[str, Any]) -> list[str]:
    label = rec.get("label", {})
    raw_values: list[Any] = []
    if isinstance(label, Mapping):
        raw = label.get("error_families")
        if isinstance(raw, list):
            raw_values.extend(raw)
        errors = label.get("errors")
        if isinstance(errors, list):
            raw_values.extend(item.get("error_type") for item in errors if isinstance(item, Mapping))

    out: list[str] = []
    for value in raw_values:
        family = normalize_error_family(value)
        if family is not None and family != "no_error" and family not in out:
            out.append(family)
    return [family for family in REPORT_FAMILY_ORDER if family in out]


def correction_family_order(
    rec: Mapping[str, Any],
    wls_payload: Mapping[str, Any],
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    *,
    policy: str = "structural_first",
) -> list[str]:
    del wls_payload, meta, idx_map
    families = multi_error_families(rec)
    if policy == "structural_first":
        return [family for family in CORRECTION_TOOL_ORDER if family in families]
    if policy == "measurement_first":
        return [family for family in REPORT_FAMILY_ORDER if family in families]
    if policy == "evidence_driven":
        # Placeholder for future evidence ranking; structural-first is the safe physical default.
        return [family for family in CORRECTION_TOOL_ORDER if family in families]
    raise ValueError(f"Unknown correction policy: {policy}")


def primary_error_family(rec: Mapping[str, Any]) -> Optional[str]:
    label = rec.get("label", {})
    if isinstance(label, Mapping):
        family = normalize_error_family(label.get("primary_error_family"))
        if family is not None:
            return family
    families = multi_error_families(rec)
    return families[0] if families else None


def component_label(rec: Mapping[str, Any], family: str) -> dict[str, Any]:
    label = rec.get("label", {})
    if isinstance(label, Mapping):
        errors = label.get("errors")
        if isinstance(errors, list):
            for item in errors:
                if isinstance(item, Mapping) and normalize_error_family(item.get("error_type")) == family:
                    out = dict(item)
                    out["error_type"] = family
                    return out
    return {"error_type": family}


def component_record(rec: Mapping[str, Any], family: str) -> dict[str, Any]:
    out = dict(rec)
    out["scenario"] = family
    out["label"] = component_label(rec, family)
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

    if config.hif_samples_path and config.hif_meta_path:
        hif_meta = json.loads(config.hif_meta_path.read_text(encoding="utf-8"))
        if _meta_core(base_meta) != _meta_core(hif_meta):
            raise ValueError("Primary and HIF metadata do not match on core measurement fields.")
        hif_samples = [
            rec
            for rec in iter_jsonl(config.hif_samples_path)
            if normalize_scenario(str(rec.get("scenario", ""))) == "high_impedance_fault"
        ]
        samples.extend(hif_samples)

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

        elif name == "run_three_phase_nlm_from_path":
            result = mp_tools.run_three_phase_nlm_from_path.fn(
                case_path=arguments["case_path"],
                nlm_diagnostic=arguments.get("nlm_diagnostic"),
                target_branch_row0=arguments.get("target_branch_row0"),
                target_dss_element=arguments.get("target_dss_element"),
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

    if scenario == "multi_error":
        families = multi_error_families(rec)
        if "measurement_error" in families:
            lab = component_label(rec, "measurement_error")
            ch = lab.get("channel")
            subtype = lab.get("subtype")
            if subtype == "single_gross_outlier" and isinstance(lab.get("index"), int):
                r[int(lab["index"])] = 6.5
            elif isinstance(lab.get("indices"), list):
                for i in lab["indices"]:
                    r[int(i)] = 4.5
            elif ch in idx_map:
                sl = idx_map[ch]
                r[sl.start:sl.stop] = np.maximum(np.abs(r[sl.start:sl.stop]), 3.2)
        if "parameter_error" in families:
            line_row0 = int(component_label(rec, "parameter_error").get("line_row", 0))
            if 0 <= 2 * line_row0 + 1 < lam.size:
                lam[2 * line_row0] = 5.0
                lam[2 * line_row0 + 1] = 6.0
            r += rng.normal(0.0, 0.15, size=m)
        if "topology_error" in families:
            for ch, level in (("Pf", 4.2), ("Qf", 4.0), ("Pt", 3.8), ("Qt", 3.6)):
                sl = idx_map[ch]
                signs = np.sign(r[sl.start:sl.stop])
                signs[signs == 0.0] = 1.0
                r[sl.start:sl.stop] = signs * np.maximum(np.abs(r[sl.start:sl.stop]), level)
            lam += 0.1
        if "harmonic_anomaly" in families:
            vm = idx_map["Vm"]
            r[vm.start:vm.stop] = np.maximum(r[vm.start:vm.stop], rng.normal(1.4, 0.4, size=vm.stop - vm.start))
            if not any(family in families for family in ("measurement_error", "topology_error")):
                norm = np.linalg.norm(r)
                if norm > 1e-9:
                    r *= (200.0 / float(np.sum(r**2))) ** 0.5
            lam[:] = np.maximum(lam, 0.08)

    elif scenario == "measurement_error":
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

    elif scenario == "high_impedance_fault":
        lab = rec.get("label", {})
        row0 = _maybe_int(lab.get("branch_row0")) if isinstance(lab, Mapping) else None
        r = rng.normal(0.0, 0.35, size=m)
        if row0 is not None:
            for ch, level in (("Pf", 4.8), ("Qf", 4.3), ("Pt", 4.5), ("Qt", 4.0)):
                sl = idx_map[ch]
                idx = sl.start + int(row0)
                if 0 <= idx < len(r):
                    r[idx] = level
        lam[:] = 0.15

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


def make_mock_measurement_correction_payload(
    rec: Mapping[str, Any],
    suspect_group: Sequence[int],
) -> Dict[str, Any]:
    z_obs = list(rec.get("z_obs", []))
    z_true = list(rec.get("z_true", z_obs))
    corrected = []
    for raw_idx in suspect_group:
        idx = _maybe_int(raw_idx)
        if idx is None or idx < 0 or idx >= len(z_obs) or idx >= len(z_true):
            continue
        corrected.append(
            {
                "index0": idx,
                "corrected": float(z_true[idx]),
                "estimated_error": float(z_obs[idx]) - float(z_true[idx]),
            }
        )
    return {
        "success": bool(corrected),
        "applied_any_correction": bool(corrected),
        "iterations_performed": 1 if corrected else 0,
        "corrected_measurements": corrected,
    }


def make_mock_three_phase_nlm_payload(rec: Mapping[str, Any]) -> Dict[str, Any]:
    payload = rec.get("nlm_diagnostic")
    if isinstance(payload, Mapping):
        return dict(payload)
    lab = rec.get("label", {})
    try:
        from three_phase_nlm.nlm_runner import metadata_hif_diagnostic

        return metadata_hif_diagnostic(
            target_dss_element=lab.get("dss_element") if isinstance(lab, Mapping) else None,
            target_branch_row0=_maybe_int(lab.get("branch_row0")) if isinstance(lab, Mapping) else None,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc), "top_hif_groups": []}


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


def append_missing_context_tool_result(
    messages: List[Dict[str, Any]],
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    *,
    tool_name: str,
    call_id: str,
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "success": False,
        "error": f"Missing runtime context for {tool_name}",
        "available_context_tools": [],
        "note": "Tool-precondition hardening example; recover without calling dependent tools.",
    }
    for key in ("case_path", "stage", "line_index"):
        if key in arguments:
            payload[key] = arguments[key]
    append_helper_tool_result(
        messages,
        meta,
        idx_map,
        tool_name=tool_name,
        call_id=call_id,
        arguments=arguments,
        payload=payload,
    )
    return payload


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
    if normalize_scenario(str(rec.get("scenario", ""))) == "high_impedance_fault":
        payload["note"] = (
            "This snapshot is a 1φ-equivalent operator vector from a copied IEEE-14 OpenDSS scenario. "
            "If a hidden high-impedance fault is suspected, use the three-phase NLM localization tool."
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


def make_hif_context_payload(rec: Mapping[str, Any], case_path: str) -> Dict[str, Any]:
    return round_user_payload(
        {
            "case_path": case_path,
            "nlm_diagnostic": rec.get("nlm_diagnostic", {}),
            "label": rec.get("label", {}),
            "note": "Compact three-phase NLM HIF localization context bound from the generated sample.",
        }
    )


def make_imbalance_followup_payload(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return round_user_payload(
        {
            "three_phase_voltages": rec.get("three_phase_voltages", []),
            "note": "Per-bus three-phase VLN voltage measurements from substations.",
        }
    )


def make_verification_snapshot_payload(
    case_path: str,
    z_obs: Sequence[float],
    note: str,
    stage: str,
    *,
    remaining_families: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    payload: dict[str, Any] = {
        "case_path": case_path,
        "z_obs": list(z_obs),
        "note": note,
        "stage": stage,
    }
    if remaining_families is not None:
        payload["remaining_families"] = list(remaining_families)
    return round_user_payload(payload)


def measurement_indices_from_label(rec: Mapping[str, Any]) -> List[int]:
    label = rec.get("label", {})
    if not isinstance(label, Mapping):
        return []
    if isinstance(label.get("index"), int):
        return [int(label["index"])]
    if isinstance(label.get("indices"), list):
        out: list[int] = []
        for item in label["indices"]:
            try:
                out.append(int(item))
            except Exception:
                pass
        return out
    return []


def choose_measurement_suspect_group(
    rec: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    tool_payload: Mapping[str, Any],
    *,
    prefer_label: bool = True,
) -> List[int]:
    if prefer_label:
        gold = measurement_indices_from_label(rec)
        if gold:
            return gold

    r_vec = np.asarray(tool_payload.get("r", []), dtype=float)
    if r_vec.size:
        try:
            return [int(np.nanargmax(np.abs(r_vec)))]
        except Exception:
            pass

    lab = rec.get("label", {})
    ch = lab.get("channel") if isinstance(lab, Mapping) else None
    if ch in idx_map:
        sl = idx_map[ch]
        return list(range(sl.start, sl.stop))
    return []


def apply_measurement_corrections_to_snapshot(
    z_obs: Sequence[float],
    corr_payload: Mapping[str, Any],
    suspect_group: Optional[Sequence[int]],
) -> List[float]:
    z2 = list(z_obs)
    suspect_set = {int(i) for i in (suspect_group or [])}
    corrected_items = corr_payload.get("corrected_measurements") or []
    if not isinstance(corrected_items, list):
        return z2
    for item in corrected_items:
        if not isinstance(item, Mapping):
            continue
        try:
            idx0 = int(item.get("index0"))
        except Exception:
            continue
        if suspect_set and idx0 not in suspect_set:
            continue
        if 0 <= idx0 < len(z2) and item.get("corrected") is not None:
            try:
                z2[idx0] = float(item["corrected"])
            except Exception:
                pass
    return z2


def apply_measurement_label_to_snapshot(
    z_obs: Sequence[float],
    label: Mapping[str, Any],
) -> List[float]:
    z2 = list(z_obs)
    if isinstance(label.get("index"), int) and label.get("amplitude") is not None:
        idx0 = int(label["index"])
        if 0 <= idx0 < len(z2):
            try:
                z2[idx0] = float(z2[idx0]) + float(label["amplitude"])
            except Exception:
                pass
    indices = label.get("indices")
    amplitudes = label.get("amplitudes")
    if isinstance(indices, list) and isinstance(amplitudes, list):
        for idx, amp in zip(indices, amplitudes):
            try:
                idx0 = int(idx)
                if 0 <= idx0 < len(z2):
                    z2[idx0] = float(z2[idx0]) + float(amp)
            except Exception:
                pass
    return z2


def get_explicit_verification_snapshot(rec: Mapping[str, Any], stage: str) -> Optional[dict[str, Any]]:
    snapshots = rec.get("verification_snapshots")
    if not isinstance(snapshots, Mapping):
        return None
    item = snapshots.get(stage)
    if not isinstance(item, Mapping):
        return None
    return dict(item)


def bind_verification_snapshot_case(
    *,
    snapshot: Mapping[str, Any],
    stage: str,
    sid: str,
    base_case_visible: str,
    base_case_backend: Any,
    runtime_context: dict[str, Any],
    current_case_visible: Optional[str] = None,
    current_case_backend: Any = None,
) -> str:
    if snapshot.get("case_path_policy") == "preserve_current_case":
        visible = current_case_visible or base_case_visible
        if visible not in runtime_context.get("case_aliases", {}):
            runtime_context.setdefault("case_aliases", {})[visible] = runtime_case_reference(
                current_case_backend or base_case_backend
            )
        return visible

    raw_case = snapshot.get("case_path")
    visible = make_case_alias(base_case_visible, stage, sid)
    runtime_context["case_aliases"][visible] = runtime_case_reference(
        raw_case or current_case_backend or base_case_backend
    )
    return visible


def verification_z_obs_from_snapshot(snapshot: Mapping[str, Any], current_z_obs: Sequence[float]) -> List[float]:
    if snapshot.get("z_obs_policy") == "preserve_current_z_obs":
        return list(current_z_obs)
    if isinstance(snapshot.get("z_obs"), list):
        return list(snapshot["z_obs"])
    return list(current_z_obs)


def explicit_snapshot_compatible_with_current_z(
    snapshot: Mapping[str, Any],
    current_z_obs: Sequence[float],
) -> bool:
    z_obs = snapshot.get("z_obs")
    if snapshot.get("z_obs_policy") == "preserve_current_z_obs" or not isinstance(z_obs, list):
        return True
    return len(z_obs) == len(current_z_obs)


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
    gm = compact.get("global_metrics")
    if not isinstance(gm, Mapping):
        return round_assistant_payload(
            {
                "post_action_global_residual_sum": None,
                "post_action_global_residual_threshold": None,
                "post_action_global_residual_ratio": None,
                "post_action_executed": False,
                "post_action_improved": False,
                "post_action_resolved": False,
            }
        )
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


def wls_global_residual_ratio(
    payload: Optional[Mapping[str, Any]],
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
) -> Optional[float]:
    if not isinstance(payload, Mapping):
        return None
    compact = summarize_wls_payload(payload, meta, idx_map)
    return _maybe_float((compact.get("global_metrics") or {}).get("global_residual_ratio"))


def build_final_target(
    rec: Mapping[str, Any],
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    primary_wls: Mapping[str, Any],
    *,
    measurement_suspect_group: Optional[List[int]] = None,
    verification_payload: Optional[Mapping[str, Any]] = None,
    verification_payloads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    verification_pre_payloads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    hse_payload: Optional[Mapping[str, Any]] = None,
    nlm_payload: Optional[Mapping[str, Any]] = None,
    correction_tool_name: Optional[str] = None,
    applied_tools: Optional[List[str]] = None,
    correction_steps: Optional[List[Mapping[str, Any]]] = None,
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
    if isinstance(nlm_payload, Mapping):
        evidence["top_hif_groups"] = summarize_three_phase_nlm_payload(nlm_payload).get("top_hif_groups", [])

    families = multi_error_families(rec)
    if scenario == "multi_error" and families:
        primary_family = primary_error_family(rec) or families[0]
        tool_list = list(applied_tools or [])
        if not tool_list:
            tool_list = [TOOL_BY_FAMILY[family] for family in families if family in TOOL_BY_FAMILY]

        def _location_for_family(family: str) -> dict[str, Any]:
            family_label = component_label(rec, family)
            if family == "measurement_error":
                details = {
                    "channel": family_label.get("channel"),
                    "index0": _maybe_int(family_label.get("index")),
                    "indices0": (
                        [int(i) for i in family_label.get("indices", [])]
                        if isinstance(family_label.get("indices"), list)
                        else None
                    ),
                    "subtype": family_label.get("subtype"),
                }
                return {"domain": "measurement", "details": {k: v for k, v in details.items() if v not in (None, [], {})}}
            if family == "parameter_error":
                line_row0 = _maybe_int(family_label.get("line_row"))
                return {
                    "domain": "parameter",
                    "details": {
                        "line_row0": line_row0,
                        "line_index1": line_row0 + 1 if line_row0 is not None else None,
                        "from_bus": _maybe_int(family_label.get("from_bus")),
                        "to_bus": _maybe_int(family_label.get("to_bus")),
                        "subtype": family_label.get("subtype"),
                    },
                }
            if family == "topology_error":
                return {
                    "domain": "topology",
                    "details": {
                        "substation": _maybe_int(family_label.get("substation")),
                        "cb_name": family_label.get("cb_name"),
                        "observed_status": family_label.get("new_status"),
                        "expected_status": family_label.get("old_status"),
                    },
                }
            if family == "harmonic_anomaly":
                details: dict[str, Any] = {"source_bus": _maybe_int(family_label.get("source_bus"))}
                if isinstance(hse_payload, Mapping):
                    compact_hse = summarize_hse_payload(hse_payload)
                    details["hse_best_candidate_bus_1based"] = compact_hse.get("best_candidate_bus_1based")
                    details["best_candidate_thd_percent"] = compact_hse.get("best_candidate_thd_percent")
                    details["ranking_top5"] = compact_hse.get("ranking_top5")
                return {"domain": "harmonic", "details": details}
            return {"domain": "none", "details": {}}

        suspect_locations = [_location_for_family(family) for family in families]
        primary_location = suspect_locations[0]
        for family, location in zip(families, suspect_locations):
            if family == primary_family:
                primary_location = location
                break

        verification_by_family: dict[str, Any] = {}
        for family, payload in dict(verification_payloads or {}).items():
            summary = build_verification_summary(
                payload,
                meta,
                idx_map,
                pre_action_payload=dict(verification_pre_payloads or {}).get(family, primary_wls),
            )
            if summary is not None:
                verification_by_family[family] = summary
        primary_verification = verification_by_family.get(primary_family)
        if primary_verification is None and isinstance(verification_payload, Mapping):
            primary_verification = build_verification_summary(
                verification_payload,
                meta,
                idx_map,
                pre_action_payload=primary_wls,
            )

        hint_by_family: dict[str, Any] = {}
        if "measurement_error" in families and measurement_suspect_group is not None:
            hint_by_family["measurement_error"] = {"suspect_group": measurement_suspect_group}
        if "parameter_error" in families:
            line_row0 = _maybe_int(component_label(rec, "parameter_error").get("line_row"))
            if line_row0 is not None:
                hint_by_family["parameter_error"] = {"line_index": line_row0 + 1}
        if "topology_error" in families:
            topo_label = component_label(rec, "topology_error")
            if topo_label.get("cb_name"):
                hint_by_family["topology_error"] = {
                    "cb_name": topo_label.get("cb_name"),
                    "desired_status": status_to_bool(topo_label.get("old_status")),
                    "desired_status_text": topo_label.get("old_status"),
                }
        if "harmonic_anomaly" in families:
            hint_by_family["harmonic_anomaly"] = {"harmonic_measurements": "bound_via_get_harmonic_context"}

        return round_assistant_payload(
            {
                "verdict": {
                    "has_error": True,
                    "error_family": primary_family,
                    "error_families": families,
                    "confidence": 0.96,
                },
                "evidence": evidence,
                "suspect_location": primary_location,
                "suspect_locations": suspect_locations,
                "action": {
                    "applied_tool": tool_list[-1] if tool_list else correction_tool_name,
                    "first_applied_tool": tool_list[0] if tool_list else None,
                    "last_applied_tool": tool_list[-1] if tool_list else correction_tool_name,
                    "applied_tools": tool_list,
                    "arguments_hint": hint_by_family or None,
                    "request_more_data": False,
                    "requested_data": None,
                    "verification_summary": primary_verification,
                    "verification_summaries": verification_by_family or None,
                    "correction_steps": list(correction_steps or []),
                },
                "summary": "Multiple error mechanisms are present in this snapshot: " + ", ".join(families) + ".",
            }
        )

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

    elif scenario == "high_impedance_fault":
        compact_nlm = summarize_three_phase_nlm_payload(nlm_payload or rec.get("nlm_diagnostic", {}))
        top_groups = compact_nlm.get("top_hif_groups", [])
        if not evidence.get("top_hif_groups"):
            evidence["top_hif_groups"] = top_groups
        top1 = top_groups[0] if top_groups else {}
        branch_row0 = _maybe_int(top1.get("branch_row0"))
        line_index1 = _maybe_int(top1.get("line_index1"))
        from_bus = _maybe_int(top1.get("from_bus"))
        to_bus = _maybe_int(top1.get("to_bus"))
        details = {
            "fault_type": "high_impedance_fault",
            "branch_row0": branch_row0,
            "line_index1": line_index1,
            "from_bus": from_bus,
            "to_bus": to_bus,
            "dss_element": top1.get("dss_element"),
        }
        suspected_phase = compact_nlm.get("suspected_phase")
        if suspected_phase is not None:
            details["phase"] = suspected_phase
        if compact_nlm.get("phase_scores") is not None:
            details["phase_scores"] = compact_nlm.get("phase_scores")
        suspect_location = {
            "domain": "fault",
            "details": {k: v for k, v in details.items() if v not in (None, [], {})},
        }
        action = {
            "applied_tool": "run_three_phase_nlm_from_path",
            "arguments_hint": {"case_path": "case14"},
            "request_more_data": False,
            "requested_data": None,
            "verification_summary": None,
        }
        summary = "Three-phase NLM line-group evidence is most consistent with a hidden high-impedance fault."

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
    multi_families = []
    raw_families = verdict.get("error_families") if isinstance(verdict, Mapping) else None
    if isinstance(raw_families, list):
        for item in raw_families:
            parsed = normalize_error_family(item)
            if parsed is not None and parsed != "no_error" and parsed not in multi_families:
                multi_families.append(parsed)
    if multi_families:
        if len(multi_families) < 2:
            return "multi_error_requires_at_least_two_families"
        applied = action.get("applied_tools") if isinstance(action, Mapping) else None
        if not isinstance(applied, list):
            applied = [action.get("applied_tool")] if isinstance(action, Mapping) and action.get("applied_tool") else []
        missing_tools = [
            TOOL_BY_FAMILY[family]
            for family in multi_families
            if TOOL_BY_FAMILY.get(family) and TOOL_BY_FAMILY[family] not in applied
        ]
        if missing_tools:
            return "multi_error_missing_applied_tools"
        suspect_locations = final_target.get("suspect_locations")
        if not isinstance(suspect_locations, list) or len(suspect_locations) < len(multi_families):
            return "multi_error_missing_suspect_locations"
        if "harmonic_anomaly" in multi_families:
            harmonic_locations = [
                loc
                for loc in suspect_locations
                if isinstance(loc, Mapping) and loc.get("domain") == "harmonic"
            ]
            if not harmonic_locations or harmonic_locations[0].get("details", {}).get("hse_best_candidate_bus_1based") is None:
                return "multi_error_missing_hse_result"
        return None

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
    elif family == "high_impedance_fault":
        if action.get("applied_tool") != "run_three_phase_nlm_from_path":
            return "high_impedance_fault_missing_nlm_tool"
        hif_groups = evidence.get("top_hif_groups", [])
        if not isinstance(hif_groups, list) or not hif_groups:
            return "high_impedance_fault_missing_nlm_evidence"
    elif family == "three_phase_imbalance":
        if bool(action.get("request_more_data")):
            return "three_phase_imbalance_missing_followup"

    return None


def append_multi_error_actions(
    *,
    config: BuilderConfig,
    rec: Mapping[str, Any],
    sid: str,
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    base_case_backend: Any,
    base_case_visible: str,
    runtime_context: Dict[str, Any],
    hidden_context: Dict[str, Any],
    messages: List[Dict[str, Any]],
    wls_payload: Mapping[str, Any],
    rng_np: np.random.Generator,
) -> dict[str, Any]:
    families = multi_error_families(rec)
    correction_order = correction_family_order(rec, wls_payload, meta, idx_map)
    applied_tools: list[str] = []
    verification_payloads: dict[str, Mapping[str, Any]] = {}
    verification_pre_payloads: dict[str, Mapping[str, Any]] = {}
    correction_steps: list[dict[str, Any]] = []
    measurement_suspect_group: Optional[List[int]] = None
    hse_payload: Optional[Mapping[str, Any]] = None
    current_case_visible = base_case_visible
    current_case_backend: Any = base_case_backend
    current_z_obs: list[float] = list(rec.get("z_obs", []))
    previous_wls_payload: Mapping[str, Any] = wls_payload
    corrected_families: list[str] = []

    def _remaining_after(family: str, explicit_remaining: Any = None) -> list[str]:
        if isinstance(explicit_remaining, list):
            return [str(item) for item in explicit_remaining]
        return [
            candidate
            for candidate in correction_order
            if candidate != family and candidate not in corrected_families
        ]

    def _append_verification_wls(
        *,
        family: str,
        tool_name: str,
        verify_stage: str,
        verification_case_visible: str,
        z_verify: Sequence[float],
        note: str,
        remaining_families: Optional[Sequence[str]] = None,
    ) -> Mapping[str, Any]:
        nonlocal current_z_obs, previous_wls_payload
        pre_payload = previous_wls_payload
        current_z_obs = list(z_verify)
        verification_snapshot = make_verification_snapshot_payload(
            verification_case_visible,
            current_z_obs,
            note,
            verify_stage,
            remaining_families=remaining_families,
        )
        runtime_context["tool_context"].setdefault("verification_snapshots", {})[verify_stage] = verification_snapshot
        hidden_context["snapshot_context"] = verification_snapshot
        append_helper_tool_result(
            messages,
            meta,
            idx_map,
            tool_name="get_verification_snapshot",
            call_id=f"call_ctx_verify_{family}_{sha_short(sid)}",
            arguments={"stage": verify_stage},
            payload=verification_snapshot,
        )
        verify_call = make_tool_call(
            "wls_from_path",
            f"call_wls_verify_{family}_{sha_short(sid)}",
            {"case_path": verification_case_visible},
        )
        messages.append({"role": "assistant", "tool_calls": [verify_call]})
        try:
            verify_payload = (
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
            verify_payload = {"success": False, "error": str(exc)}
        verification_pre_payloads[family] = pre_payload
        verification_payloads[family] = verify_payload
        previous_wls_payload = verify_payload
        summary = build_verification_summary(
            verify_payload,
            meta,
            idx_map,
            pre_action_payload=pre_payload,
        ) or {}
        correction_steps.append(
            round_assistant_payload(
                {
                    "step": len(correction_steps) + 1,
                    "family": family,
                    "tool": tool_name,
                    "pre_global_residual_ratio": wls_global_residual_ratio(pre_payload, meta, idx_map),
                    "post_global_residual_ratio": wls_global_residual_ratio(verify_payload, meta, idx_map),
                    "post_action_improved": bool(summary.get("post_action_improved", False)),
                    "post_action_resolved": bool(summary.get("post_action_resolved", False)),
                    "remaining_candidate_families": list(remaining_families or []),
                }
            )
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": verify_call["id"],
                "name": "wls_from_path",
                "content": as_tool_return_text(
                    summarize_tool_result_for_conversation("wls_from_path", verify_payload, meta, idx_map)
                ),
            }
        )
        return verify_payload

    for family in correction_order:
        if family == "topology_error" and config.with_correction:
            topo_rec = component_record(rec, "topology_error")
            topo_label = topo_rec.get("label", {})
            cb_name = topo_label.get("cb_name")
            desired_status = status_to_bool(topo_label.get("old_status"))
            if not cb_name or desired_status is None:
                continue

            topology_context = make_topology_followup_payload(topo_rec, current_case_visible)
            runtime_context["tool_context"]["topology_context"] = topology_context
            hidden_context["topology_context"] = topology_context
            append_helper_tool_result(
                messages,
                meta,
                idx_map,
                tool_name="get_topology_context",
                call_id=f"call_ctx_topo_{sha_short(sid)}",
                arguments={"case_path": current_case_visible},
                payload=topology_context,
            )
            topo_args = {"case_path": current_case_visible, "cb_name": cb_name, "desired_status": desired_status}
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
            applied_tools.append("correct_topology_from_path")

            if topo_payload.get("success", True):
                verify_stage = "post_topology_correction"
                explicit = get_explicit_verification_snapshot(rec, verify_stage)
                note = "Post-topology-correction verification snapshot."
                remaining = None
                if explicit is not None:
                    verification_case_visible = bind_verification_snapshot_case(
                        snapshot=explicit,
                        stage=verify_stage,
                        sid=sid,
                        base_case_visible=base_case_visible,
                        base_case_backend=base_case_backend,
                        runtime_context=runtime_context,
                        current_case_visible=current_case_visible,
                        current_case_backend=current_case_backend,
                    )
                    z_verify = verification_z_obs_from_snapshot(explicit, current_z_obs)
                    note = str(explicit.get("note") or note)
                    remaining = explicit.get("remaining_families")
                    if explicit.get("case_path_policy") != "preserve_current_case":
                        current_case_visible = verification_case_visible
                        current_case_backend = explicit.get("case_path") or current_case_backend
                elif rec.get("corrected_model_path"):
                    verification_case_visible = make_case_alias(base_case_visible, "topology_verify", sid)
                    current_case_visible = verification_case_visible
                    current_case_backend = rec["corrected_model_path"]
                    runtime_context["case_aliases"][verification_case_visible] = runtime_case_reference(
                        current_case_backend
                    )
                    z_verify = current_z_obs
                elif isinstance(topo_payload.get("z_corrected"), list):
                    verification_case_visible = current_case_visible
                    z_verify = list(topo_payload["z_corrected"])
                    if "measurement_error" in families and "measurement_error" not in corrected_families:
                        z_verify = apply_measurement_label_to_snapshot(
                            z_verify,
                            component_label(rec, "measurement_error"),
                        )
                else:
                    z_verify = None
                    verification_case_visible = current_case_visible

                if isinstance(z_verify, list):
                    remaining_list = _remaining_after("topology_error", remaining)
                    _append_verification_wls(
                        family="topology_error",
                        tool_name="correct_topology_from_path",
                        verify_stage=verify_stage,
                        verification_case_visible=verification_case_visible,
                        z_verify=z_verify,
                        note=note,
                        remaining_families=remaining_list,
                    )
            corrected_families.append("topology_error")

        elif family == "parameter_error" and config.with_correction:
            param_rec = component_record(rec, "parameter_error")
            line_row0 = _maybe_int(param_rec.get("label", {}).get("line_row"))
            if line_row0 is None or not isinstance(rec.get("z_scans"), list) or not isinstance(rec.get("initial_states"), list):
                continue

            correction_case_backend = (
                rec.get("parameter_error_case_path")
                or rec.get("correction_case_path")
                or current_case_backend
            )
            correction_case_visible = current_case_visible
            if correction_case_backend != current_case_backend:
                correction_case_visible = make_case_alias(base_case_visible, "parameter_case", sid)
                runtime_context["case_aliases"][correction_case_visible] = runtime_case_reference(
                    correction_case_backend
                )

            parameter_context = make_parameter_followup_payload(param_rec, correction_case_visible)
            runtime_context["tool_context"]["parameter_context"] = parameter_context
            hidden_context["parameter_context"] = parameter_context
            append_helper_tool_result(
                messages,
                meta,
                idx_map,
                tool_name="get_parameter_context",
                call_id=f"call_ctx_param_{sha_short(sid)}",
                arguments={"case_path": correction_case_visible, "line_index": line_row0 + 1},
                payload=parameter_context,
            )
            param_args = {"case_path": correction_case_visible, "line_index": line_row0 + 1}
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
            applied_tools.append("correct_parameters_from_path")

            if param_payload.get("success", True):
                verify_stage = "post_parameter_correction"
                explicit = get_explicit_verification_snapshot(rec, verify_stage)
                note = "Post-parameter-correction verification snapshot."
                remaining = None
                if explicit is not None:
                    verification_case_visible = bind_verification_snapshot_case(
                        snapshot=explicit,
                        stage=verify_stage,
                        sid=sid,
                        base_case_visible=base_case_visible,
                        base_case_backend=base_case_backend,
                        runtime_context=runtime_context,
                        current_case_visible=current_case_visible,
                        current_case_backend=current_case_backend,
                    )
                    if (
                        explicit.get("z_obs_policy") == "preserve_current_z_obs"
                        and correction_case_backend != current_case_backend
                        and len(current_z_obs) != len(rec.get("z_obs", []))
                        and isinstance(rec.get("z_obs"), list)
                    ):
                        z_verify = list(rec["z_obs"])
                    else:
                        z_verify = verification_z_obs_from_snapshot(explicit, current_z_obs)
                    note = str(explicit.get("note") or note)
                    remaining = explicit.get("remaining_families")
                    if explicit.get("case_path_policy") != "preserve_current_case":
                        current_case_visible = verification_case_visible
                        current_case_backend = explicit.get("case_path") or current_case_backend
                else:
                    correction_case_backend = rec.get("parameter_error_case_path") or rec.get("correction_case_path")
                    if correction_case_backend:
                        verification_case_visible = make_case_alias(base_case_visible, "parameter_verify", sid)
                        current_case_visible = verification_case_visible
                        current_case_backend = correction_case_backend
                        runtime_context["case_aliases"][verification_case_visible] = runtime_case_reference(
                            current_case_backend
                        )
                    else:
                        verification_case_visible = current_case_visible
                    z_verify = current_z_obs

                remaining_list = _remaining_after("parameter_error", remaining)
                _append_verification_wls(
                    family="parameter_error",
                    tool_name="correct_parameters_from_path",
                    verify_stage=verify_stage,
                    verification_case_visible=verification_case_visible,
                    z_verify=z_verify,
                    note=note,
                    remaining_families=remaining_list,
                )
            corrected_families.append("parameter_error")

        elif family == "measurement_error" and config.with_correction:
            meas_rec = component_record(rec, "measurement_error")
            measurement_suspect_group = choose_measurement_suspect_group(
                meas_rec,
                idx_map,
                previous_wls_payload,
                prefer_label="topology_error" not in corrected_families,
            )
            corr_call_args = {"case_path": current_case_visible, "suspect_group": measurement_suspect_group}
            corr_call = make_tool_call(
                "correct_measurements_from_path",
                f"call_corr_meas_{sha_short(sid)}",
                corr_call_args,
            )
            messages.append({"role": "assistant", "tool_calls": [corr_call]})
            try:
                corr_payload = (
                    make_mock_measurement_correction_payload(meas_rec, measurement_suspect_group)
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
            applied_tools.append("correct_measurements_from_path")

            verify_stage = "post_measurement_correction"
            explicit = get_explicit_verification_snapshot(rec, verify_stage)
            note = "Post-measurement-correction verification snapshot."
            remaining = None
            if explicit is not None and explicit_snapshot_compatible_with_current_z(explicit, current_z_obs):
                verification_case_visible = bind_verification_snapshot_case(
                    snapshot=explicit,
                    stage=verify_stage,
                    sid=sid,
                    base_case_visible=base_case_visible,
                    base_case_backend=base_case_backend,
                    runtime_context=runtime_context,
                    current_case_visible=current_case_visible,
                    current_case_backend=current_case_backend,
                )
                z_verify = verification_z_obs_from_snapshot(explicit, current_z_obs)
                note = str(explicit.get("note") or note)
                remaining = explicit.get("remaining_families")
            else:
                verification_case_visible = current_case_visible
                z_verify = apply_measurement_corrections_to_snapshot(
                    current_z_obs,
                    corr_payload,
                    measurement_suspect_group,
                )

            remaining_list = _remaining_after("measurement_error", remaining)
            _append_verification_wls(
                family="measurement_error",
                tool_name="correct_measurements_from_path",
                verify_stage=verify_stage,
                verification_case_visible=verification_case_visible,
                z_verify=z_verify,
                note=note,
                remaining_families=remaining_list,
            )
            corrected_families.append("measurement_error")

        elif family == "harmonic_anomaly":
            if not rec.get("harmonic_measurements"):
                continue
            harmonic_case_visible = current_case_visible
            if "topology_error" in corrected_families:
                harmonic_case_visible = make_case_alias(base_case_visible, "harmonic_channel", sid)
                runtime_context["case_aliases"][harmonic_case_visible] = runtime_case_reference(base_case_backend)
            harmonic_context = make_harmonic_followup_payload(rec, harmonic_case_visible)
            runtime_context["tool_context"]["harmonic_context"] = harmonic_context
            hidden_context["harmonic_context"] = harmonic_context
            append_helper_tool_result(
                messages,
                meta,
                idx_map,
                tool_name="get_harmonic_context",
                call_id=f"call_ctx_harm_{sha_short(sid)}",
                arguments={"case_path": harmonic_case_visible},
                payload=harmonic_context,
            )
            hse_args = {"case_path": harmonic_case_visible}
            hse_call = make_tool_call("run_hse_from_path", f"call_hse_{sha_short(sid)}", hse_args)
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
                        summarize_tool_result_for_conversation("run_hse_from_path", hse_payload, meta, idx_map)
                    ),
                }
            )
            applied_tools.append("run_hse_from_path")
            correction_steps.append(
                round_assistant_payload(
                    {
                        "step": len(correction_steps) + 1,
                        "family": "harmonic_anomaly",
                        "tool": "run_hse_from_path",
                        "pre_global_residual_ratio": wls_global_residual_ratio(
                            previous_wls_payload,
                            meta,
                            idx_map,
                        ),
                        "post_global_residual_ratio": None,
                        "post_action_improved": None,
                        "post_action_resolved": None,
                        "remaining_candidate_families": [],
                    }
                )
            )
            corrected_families.append("harmonic_anomaly")

    return {
        "measurement_suspect_group": measurement_suspect_group,
        "verification_payloads": verification_payloads,
        "verification_pre_payloads": verification_pre_payloads,
        "hse_payload": hse_payload,
        "applied_tools": applied_tools,
        "correction_steps": correction_steps,
        "correction_tool_name": applied_tools[-1] if applied_tools else None,
    }


def build_tool_precondition_hardening_trace(
    *,
    config: BuilderConfig,
    rec: Mapping[str, Any],
    template: str,
    sid: str,
    meta: Mapping[str, Any],
    idx_map: Mapping[str, slice],
    base_case_backend: Any,
    base_case_visible: str,
    rng_np: np.random.Generator,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    scenario = normalize_scenario(str(rec.get("scenario", "")))
    if template not in HARDENING_TEMPLATES:
        return None, "unknown_hardening_template"
    if scenario != "measurement_error":
        return None, "hardening_requires_measurement_error_source"
    if not config.with_correction:
        return None, "hardening_requires_correction_enabled"

    runtime_context: Dict[str, Any] = {
        "case_aliases": {base_case_visible: runtime_case_reference(base_case_backend)},
        "tool_context": {},
    }
    hidden_context: Dict[str, Any] = {
        "case_aliases": runtime_context["case_aliases"],
    }
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json_compact(make_user_payload(rec, meta, base_case_visible))},
    ]

    wls_call_args = {"case_path": base_case_visible}
    wls_call = make_tool_call("wls_from_path", f"call_wls_harden_{sha_short(sid)}", wls_call_args)
    messages.append({"role": "assistant", "tool_calls": [wls_call]})
    try:
        wls_payload = (
            make_mock_wls_payload(rec, meta, idx_map, rng_np)
            if config.mock
            else call_backend_tool(
                config.endpoint,
                "wls_from_path",
                wls_call_args,
                messages,
                hidden_context,
                timeout=config.timeout_s,
            )
        )
    except Exception as exc:
        return None, f"hardening_initial_wls_failed:{exc}"
    if not wls_payload.get("success", True):
        return None, "hardening_initial_wls_failed"
    messages.append(
        {
            "role": "tool",
            "tool_call_id": wls_call["id"],
            "name": "wls_from_path",
            "content": as_tool_return_text(
                summarize_tool_result_for_conversation("wls_from_path", wls_payload, meta, idx_map)
            ),
        }
    )

    if template == "parameter_helper_unavailable":
        append_missing_context_tool_result(
            messages,
            meta,
            idx_map,
            tool_name="get_parameter_context",
            call_id=f"call_ctx_param_missing_{sha_short(sid)}",
            arguments={"case_path": base_case_visible},
        )
    elif template == "harmonic_helper_unavailable":
        append_missing_context_tool_result(
            messages,
            meta,
            idx_map,
            tool_name="get_harmonic_context",
            call_id=f"call_ctx_harm_missing_{sha_short(sid)}",
            arguments={"case_path": base_case_visible},
        )
    elif template == "verification_snapshot_unavailable":
        append_missing_context_tool_result(
            messages,
            meta,
            idx_map,
            tool_name="get_verification_snapshot",
            call_id=f"call_ctx_verify_missing_{sha_short(sid)}",
            arguments={"stage": "post_measurement_correction"},
        )

    measurement_suspect_group = choose_measurement_suspect_group(rec, idx_map, wls_payload)
    if not measurement_suspect_group:
        return None, "hardening_missing_measurement_suspect_group"

    corr_call_args = {
        "case_path": base_case_visible,
        "suspect_group": measurement_suspect_group,
    }
    corr_call = make_tool_call(
        "correct_measurements_from_path",
        f"call_corr_meas_harden_{sha_short(sid)}",
        corr_call_args,
    )
    messages.append({"role": "assistant", "tool_calls": [corr_call]})
    try:
        corr_payload = (
            make_mock_measurement_correction_payload(rec, measurement_suspect_group)
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

    z_verify = apply_measurement_corrections_to_snapshot(
        rec["z_obs"],
        corr_payload,
        measurement_suspect_group,
    )
    if z_verify == list(rec.get("z_obs", [])):
        return None, "hardening_measurement_correction_noop"

    verify_stage = "post_measurement_correction"
    verification_case_visible = make_case_alias(base_case_visible, "measurement_verify_hardening", sid)
    runtime_context["case_aliases"][verification_case_visible] = runtime_case_reference(base_case_backend)
    verification_snapshot = make_verification_snapshot_payload(
        verification_case_visible,
        z_verify,
        "Post-measurement-correction verification snapshot.",
        verify_stage,
    )
    runtime_context["tool_context"].setdefault("verification_snapshots", {})[verify_stage] = verification_snapshot
    hidden_context["snapshot_context"] = verification_snapshot
    append_helper_tool_result(
        messages,
        meta,
        idx_map,
        tool_name="get_verification_snapshot",
        call_id=f"call_ctx_verify_meas_harden_{sha_short(sid)}",
        arguments={"stage": verify_stage},
        payload=verification_snapshot,
    )

    verify_call = make_tool_call(
        "wls_from_path",
        f"call_wls_verify_harden_{sha_short(sid)}",
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
                summarize_tool_result_for_conversation("wls_from_path", verification_payload, meta, idx_map)
            ),
        }
    )

    final = build_final_target(
        rec,
        meta,
        idx_map,
        wls_payload,
        measurement_suspect_group=measurement_suspect_group,
        verification_payload=verification_payload,
        correction_tool_name="correct_measurements_from_path",
        applied_tools=["correct_measurements_from_path"],
    )
    reject_reason = rejection_reason(final)
    if reject_reason is not None:
        return None, f"hardening_final_rejected:{reject_reason}"

    messages.append({"role": "assistant", "content": json_compact(final)})
    return (
        {
            "messages": messages,
            "runtime_context": runtime_context,
            "trace_metadata": {
                "trace_kind": "tool_precondition_hardening",
                "template": template,
                "source_id": str(rec.get("id")),
            },
        },
        None,
    )


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
    n_hardening_written = 0
    hardening_rejections = 0
    rejected_rows: list[dict[str, Any]] = []
    hardening_target = max(0, int(config.hardening_examples))
    if hardening_target > HARDENING_MAX_EXAMPLES:
        print(
            f"Requested {hardening_target} hardening examples; "
            f"capping at {HARDENING_MAX_EXAMPLES}."
        )
        hardening_target = HARDENING_MAX_EXAMPLES

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
            verification_payloads: dict[str, Mapping[str, Any]] = {}
            verification_pre_payloads: dict[str, Mapping[str, Any]] = {}
            hse_payload: Optional[Dict[str, Any]] = None
            nlm_payload: Optional[Dict[str, Any]] = None
            applied_tools: list[str] = []
            correction_steps: list[Mapping[str, Any]] = []

            if not wls_payload.get("success", True):
                rejected_rows.append({"id": sid, "scenario": scenario, "reason": "initial_wls_failed"})
                continue

            if scenario == "multi_error":
                multi_result = append_multi_error_actions(
                    config=config,
                    rec=rec,
                    sid=sid,
                    meta=meta,
                    idx_map=idx_map,
                    base_case_backend=base_case_backend,
                    base_case_visible=base_case_visible,
                    runtime_context=runtime_context,
                    hidden_context=hidden_context,
                    messages=messages,
                    wls_payload=wls_payload,
                    rng_np=rng_np,
                )
                measurement_suspect_group = multi_result["measurement_suspect_group"]
                verification_payloads = dict(multi_result["verification_payloads"])
                verification_pre_payloads = dict(multi_result["verification_pre_payloads"])
                hse_payload = multi_result["hse_payload"]
                applied_tools = list(multi_result["applied_tools"])
                correction_steps = list(multi_result["correction_steps"])
                correction_tool_name = multi_result["correction_tool_name"]

            elif scenario == "measurement_error" and config.with_correction:
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
                        make_mock_measurement_correction_payload(rec, measurement_suspect_group)
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

                z2 = apply_measurement_corrections_to_snapshot(
                    rec["z_obs"],
                    corr_payload,
                    measurement_suspect_group,
                )
                if z2 != list(rec["z_obs"]):
                    verify_stage = "post_measurement_correction"
                    verification_case_visible = make_case_alias(base_case_visible, "measurement_verify", sid)
                    runtime_context["case_aliases"][verification_case_visible] = runtime_case_reference(
                        base_case_backend
                    )
                    verification_snapshot = make_verification_snapshot_payload(
                        verification_case_visible,
                        z2,
                        "Post-measurement-correction verification snapshot.",
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
                        arguments={"stage": verify_stage},
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
                            arguments={"stage": verify_stage},
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
                            arguments={"stage": verify_stage},
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

            elif scenario == "high_impedance_fault":
                hif_context = make_hif_context_payload(rec, base_case_visible)
                runtime_context["tool_context"]["hif_context"] = hif_context
                hidden_context["hif_context"] = hif_context
                nlm_args = {"case_path": base_case_visible}
                nlm_call = make_tool_call(
                    "run_three_phase_nlm_from_path",
                    f"call_nlm_hif_{sha_short(sid)}",
                    nlm_args,
                )
                messages.append({"role": "assistant", "tool_calls": [nlm_call]})
                try:
                    nlm_payload = (
                        make_mock_three_phase_nlm_payload(rec)
                        if config.mock
                        else call_backend_tool(
                            config.endpoint,
                            "run_three_phase_nlm_from_path",
                            nlm_args,
                            messages,
                            hidden_context,
                            timeout=config.timeout_s,
                        )
                    )
                except Exception as exc:
                    nlm_payload = {"success": False, "error": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": nlm_call["id"],
                        "name": "run_three_phase_nlm_from_path",
                        "content": as_tool_return_text(
                            summarize_tool_result_for_conversation(
                                "run_three_phase_nlm_from_path",
                                nlm_payload,
                                meta,
                                idx_map,
                            )
                        ),
                    }
                )
                if (
                    isinstance(nlm_payload, Mapping)
                    and nlm_payload.get("method") == "metadata_fallback"
                    and not config.allow_hif_metadata_fallback
                ):
                    rejected_rows.append(
                        {
                            "id": sid,
                            "scenario": scenario,
                            "reason": "high_impedance_fault_metadata_fallback",
                        }
                    )
                    continue

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
                verification_payloads=verification_payloads,
                verification_pre_payloads=verification_pre_payloads,
                hse_payload=hse_payload,
                nlm_payload=nlm_payload,
                correction_tool_name=correction_tool_name,
                applied_tools=applied_tools,
                correction_steps=correction_steps,
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

        if hardening_target:
            if not config.with_correction:
                rejected_rows.append(
                    {
                        "id": "tool_precondition_hardening",
                        "scenario": "tool_precondition_hardening",
                        "reason": "hardening_requires_correction_enabled",
                    }
                )
            else:
                measurement_sources = [
                    s
                    for s in samples
                    if normalize_scenario(str(s.get("scenario", ""))) == "measurement_error"
                ]
                rng_std.shuffle(measurement_sources)
                if not measurement_sources:
                    rejected_rows.append(
                        {
                            "id": "tool_precondition_hardening",
                            "scenario": "tool_precondition_hardening",
                            "reason": "no_measurement_error_sources_for_hardening",
                        }
                    )
                else:
                    max_attempts = max(hardening_target * 10, len(HARDENING_TEMPLATES))
                    with tqdm(total=hardening_target, desc="Building hardening traces") as pbar:
                        for attempt in range(max_attempts):
                            if n_hardening_written >= hardening_target:
                                break
                            source_idx = (attempt // len(HARDENING_TEMPLATES)) % len(measurement_sources)
                            source = measurement_sources[source_idx]
                            template = HARDENING_TEMPLATES[attempt % len(HARDENING_TEMPLATES)]
                            hardening_sid = (
                                f"{source.get('id', 'sample')}::hardening::{template}::{attempt}"
                            )
                            row, reason = build_tool_precondition_hardening_trace(
                                config=config,
                                rec=source,
                                template=template,
                                sid=hardening_sid,
                                meta=meta,
                                idx_map=idx_map,
                                base_case_backend=base_case_backend,
                                base_case_visible=base_case_visible,
                                rng_np=rng_np,
                            )
                            if row is None:
                                hardening_rejections += 1
                                if hardening_rejections <= 50:
                                    rejected_rows.append(
                                        {
                                            "id": hardening_sid,
                                            "scenario": "tool_precondition_hardening",
                                            "template": template,
                                            "source_id": str(source.get("id")),
                                            "reason": reason or "unknown_hardening_rejection",
                                        }
                                    )
                                continue
                            fout_all.write(json.dumps(row, ensure_ascii=False) + "\n")
                            n_written += 1
                            n_hardening_written += 1
                            pbar.update(1)

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
    if hardening_target:
        print(f"Hardening traces written: {n_hardening_written}/{hardening_target}")
        if hardening_rejections:
            print(f"Hardening trace build rejections: {hardening_rejections}")
    print(f"Total written: {n_written}")


def parse_args() -> BuilderConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="out_measurements_balanced/samples.jsonl")
    p.add_argument("--meta", default="out_measurements_balanced/meta.json")
    p.add_argument("--imbalance-samples", default="")
    p.add_argument("--imbalance-meta", default="")
    p.add_argument("--hif-samples", default="")
    p.add_argument("--hif-meta", default="")
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
    p.add_argument(
        "--allow-hif-metadata-fallback",
        action="store_true",
        help="Allow oracle-backed HIF metadata fallback traces for smoke tests only.",
    )
    p.add_argument(
        "--hardening-examples",
        type=int,
        default=0,
        help=(
            "Optional small recovery set for failed helper preconditions. "
            f"Default 0; capped at {HARDENING_MAX_EXAMPLES}."
        ),
    )
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    return BuilderConfig(
        samples_path=Path(args.samples),
        meta_path=Path(args.meta),
        imbalance_samples_path=Path(args.imbalance_samples) if args.imbalance_samples else None,
        imbalance_meta_path=Path(args.imbalance_meta) if args.imbalance_meta else None,
        hif_samples_path=Path(args.hif_samples) if args.hif_samples else None,
        hif_meta_path=Path(args.hif_meta) if args.hif_meta else None,
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
        allow_hif_metadata_fallback=bool(args.allow_hif_metadata_fallback),
        hardening_examples=max(0, int(args.hardening_examples)),
        timeout_s=int(args.timeout),
    )


if __name__ == "__main__":
    build_sft(parse_args())
