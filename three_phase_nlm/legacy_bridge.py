from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dss_hif_injector import copy_ieee14_model, write_balanced_ieee14_load_override
from .ieee14_adapter import branch_info_for_row0, branch_row0_for_dss_element


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_SRC = _REPO_ROOT / "Three_phase_NLM-cable-burnout" / "src" / "three_phase_nlm_legacy"
_PHASE_TO_NUMBER = {"A": 1, "B": 2, "C": 3, "1": 1, "2": 2, "3": 3}
_NUMBER_TO_PHASE = {1: "A", 2: "B", 3: "C"}
_HIF_LOAD_RE = re.compile(r"\b(?:new|edit)\s+load\.[^\s]*hif[^\s]*\b(?P<body>.*)$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"(\w+)\s*=\s*([^ \t]+)")


class LegacyNLMBridgeError(RuntimeError):
    """Raised when the legacy NLM backend cannot produce IEEE-14 evidence."""


def _ensure_legacy_imports() -> None:
    if not _LEGACY_SRC.is_dir():
        raise LegacyNLMBridgeError(f"Legacy three-phase NLM source not found: {_LEGACY_SRC}")
    legacy_path = str(_LEGACY_SRC)
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)


def _run_quietly(fn, *args, capture_stdout: bool = True, **kwargs):
    if not capture_stdout:
        return fn(*args, **kwargs), ""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = fn(*args, **kwargs)
    return result, stream.getvalue()


def _parse_hif_load_spec(model_dir: str | Path) -> dict[str, Any]:
    for path in sorted(Path(model_dir).glob("*.DSS")) + sorted(Path(model_dir).glob("*.dss")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            match = _HIF_LOAD_RE.search(line)
            if not match:
                continue
            tokens = {k.lower(): v for k, v in _TOKEN_RE.findall(match.group("body"))}
            bus_raw = str(tokens.get("bus1", "")).strip()
            bus_parts = [part for part in bus_raw.split(".") if part]
            fault_bus = bus_parts[0] if bus_parts else None
            phase_number = int(bus_parts[1]) if len(bus_parts) > 1 and bus_parts[1].isdigit() else None
            kv_ln = _maybe_float(tokens.get("kv"))
            p_kw = _maybe_float(tokens.get("kw"))
            r_ohm = None
            if kv_ln is not None and p_kw is not None and p_kw > 0:
                r_ohm = (kv_ln * 1e3) ** 2 / (p_kw * 1e3)
            return {
                "fault_bus": fault_bus,
                "phase_number": phase_number,
                "phase": _NUMBER_TO_PHASE.get(phase_number),
                "kv_ln": kv_ln,
                "p_kw": p_kw,
                "r_hif_ohm": r_ohm,
                "source_file": str(path),
            }
    return {}


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except Exception:
        return None


def _scale_base_loads(mpc: Mapping[str, Any], load_scale: float) -> None:
    loads = mpc.get("load3p")
    if loads is None:
        return
    names = list(mpc.get("load_busnames") or [])
    for i, row in enumerate(loads):
        bus_name = str(names[i]).lower() if i < len(names) else ""
        if bus_name.startswith("fault_") or bus_name == "faultbus":
            continue
        row[3:9] = row[3:9] * float(load_scale)


def _bus_id_to_name(mpc: Mapping[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for name, bus_id in dict(mpc.get("busname_to_id") or {}).items():
        bus_name = str(name).lower()
        if not re.fullmatch(r"b\d+", bus_name):
            continue
        out[int(bus_id)] = bus_name
    if out:
        return out
    for name, bus_id in dict(mpc.get("busname_to_id") or {}).items():
        out.setdefault(int(bus_id), str(name).lower())
    return out


def _target_node_order(
    mpc_orig: Mapping[str, Any],
    mpc_fault: Mapping[str, Any],
    fault_busphase_map: Mapping[tuple[int, int], int],
) -> tuple[list[str], list[int]]:
    orig_id_to_name = _bus_id_to_name(mpc_orig)
    fault_name_to_id = {str(k).lower(): int(v) for k, v in dict(mpc_fault.get("busname_to_id") or {}).items()}
    node_names: list[str] = []
    voltage_indices: list[int] = []
    for row in mpc_orig["bus3p"]:
        orig_id = int(row[0])
        bus_name = orig_id_to_name.get(orig_id)
        fault_id = fault_name_to_id.get(bus_name, orig_id) if bus_name else orig_id
        for phase_idx in range(3):
            node_names.append(f"bus{fault_id}.{phase_idx + 1}")
            voltage_indices.append(fault_busphase_map[(fault_id, phase_idx)])
    return node_names, voltage_indices


def _rows_to_hif_groups(
    sorted_group_distances: list[tuple[int, float]],
    mpc_orig: Mapping[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_idx, score in sorted_group_distances:
        if int(line_idx) < 0 or int(line_idx) >= len(mpc_orig["line3p"]):
            continue
        line_name = str(mpc_orig["line3p"][int(line_idx)][6])
        dss_element = line_name if "." in line_name else f"Line.{line_name}"
        row0 = branch_row0_for_dss_element(dss_element)
        if row0 is None or row0 in seen:
            continue
        seen.add(row0)
        info = branch_info_for_row0(row0)
        groups.append(
            {
                "rank": len(groups) + 1,
                "branch_row0": info["branch_row0"],
                "line_index1": info["line_index1"],
                "dss_element": info["dss_element"],
                "from_bus": info["from_bus"],
                "to_bus": info["to_bus"],
                "score": float(score),
            }
        )
        if len(groups) >= int(top_k):
            break
    return groups


def run_legacy_ieee14_hif_nlm(
    *,
    pristine_model_dir: str | Path,
    faulted_model_dir: str | Path,
    target_branch_row0: int | None = None,
    target_dss_element: str | None = None,
    phase: str | None = None,
    r_hif_ohm: float | None = None,
    load_scale: float = 1.0,
    base_mva: float = 100.0,
    slack_bus: str = "b1",
    top_k: int = 5,
    max_iter: int = 20,
    capture_stdout: bool = True,
) -> dict[str, Any]:
    """
    Run the imported legacy three-phase NLM backend on a generated IEEE-14 HIF scenario.

    The backend uses a balanced pristine IEEE-14 model for the estimator, the generated
    faulted OpenDSS model for synthetic three-phase measurements, and returns compact
    line-group evidence mapped back to the IEEE-14 branch order used by the traces.
    """
    _ensure_legacy_imports()
    from cal_pf import run_newton_powerflow_3p  # type: ignore
    from lagrangian_m import run_lagrangian_polar  # type: ignore
    from midspan_hif_utils import kron_reduce  # type: ignore
    from parse_opendss_file import (  # type: ignore
        build_global_y_per_unit,
        merge_closed_switches_in_mpc_and_dss,
        parse_opendss_to_mpc,
    )
    from utilities.mea_fun import measurement_function  # type: ignore
    import opendssdirect as dss  # type: ignore

    faulted_dir = Path(faulted_model_dir).resolve()
    if not faulted_dir.is_dir():
        raise LegacyNLMBridgeError(f"Faulted IEEE-14 model directory not found: {faulted_dir}")

    hif_spec = _parse_hif_load_spec(faulted_dir)
    fault_bus = str(hif_spec.get("fault_bus") or "").lower()
    if not fault_bus:
        raise LegacyNLMBridgeError(f"No HIF load/fault bus found in {faulted_dir}")
    phase_number = _PHASE_TO_NUMBER.get(str(phase or hif_spec.get("phase") or "A").strip().upper(), 1)
    kv_ln = _maybe_float(hif_spec.get("kv_ln")) or (1.0 / np.sqrt(3.0))
    r_hif = _maybe_float(r_hif_ohm) or _maybe_float(hif_spec.get("r_hif_ohm"))

    with tempfile.TemporaryDirectory(prefix="ieee14_legacy_nlm_pristine_") as tmp_dir:
        pristine_dir = copy_ieee14_model(pristine_model_dir, tmp_dir, overwrite=True)
        write_balanced_ieee14_load_override(pristine_dir)
        pristine_master = Path(pristine_dir) / "Run_IEEE14Bus.dss"
        pristine_lc = Path(pristine_dir) / "IEEE14Lines.DSS"
        fault_master = faulted_dir / "Run_IEEE14Bus.dss"
        fault_lc = faulted_dir / "IEEE14Lines.DSS"

        mpc_orig = parse_opendss_to_mpc(
            str(pristine_master),
            baseMVA=float(base_mva),
            lc_filename=str(pristine_lc),
            slack_bus=slack_bus,
        )
        _, log = _run_quietly(
            merge_closed_switches_in_mpc_and_dss,
            mpc_orig,
            1e-9,
            capture_stdout=capture_stdout,
        )
        _scale_base_loads(mpc_orig, float(load_scale))

        mpc_fault = parse_opendss_to_mpc(
            str(fault_master),
            baseMVA=float(base_mva),
            lc_filename=str(fault_lc),
            slack_bus=slack_bus,
        )
        _, log = _run_quietly(
            merge_closed_switches_in_mpc_and_dss,
            mpc_fault,
            1e-9,
            capture_stdout=capture_stdout,
        )
        _scale_base_loads(mpc_fault, float(load_scale))

        (pf_result, log) = _run_quietly(
            run_newton_powerflow_3p,
            mpc_fault,
            1e-6,
            20,
            capture_stdout=capture_stdout,
        )
        vr_full, vi_full, fault_busphase_map = pf_result
        vc_full = vr_full + 1j * vi_full

        dss.Command("Clear")
        dss.Command(f'Redirect "{fault_master}"')
        y_full, node_order = build_global_y_per_unit(mpc_fault)
        node_phase_map = {node_name: i for i, node_name in enumerate(node_order)}
        fault_id = int(dict(mpc_fault.get("busname_to_id") or {}).get(fault_bus, 0))
        if fault_id <= 0:
            raise LegacyNLMBridgeError(f"Fault bus {fault_bus!r} not found in parsed faulted model.")

        if r_hif is not None and r_hif > 0:
            hif_node_name = f"bus{fault_id}.{phase_number}"
            if hif_node_name in node_phase_map:
                hif_admittance = 1.0 / float(r_hif)
                s_base_3ph = float(mpc_fault["baseMVA"]) * 1e6
                z_base = (float(kv_ln) * 1e3) ** 2 / s_base_3ph
                y_hif_pu = hif_admittance / (1.0 / z_base)
                y_full = y_full.tolil()
                hif_idx = node_phase_map[hif_node_name]
                y_full[hif_idx, hif_idx] += y_hif_pu
                y_full = y_full.tocsc()

        elim_idx = [
            node_phase_map[node_name]
            for node_name in (f"bus{fault_id}.{ph}" for ph in (1, 2, 3))
            if node_name in node_phase_map
        ]
        if not elim_idx:
            raise LegacyNLMBridgeError(f"Cannot Kron-reduce HIF bus {fault_bus!r}; no matching phases found.")
        y_reduced = kron_reduce(y_full, elim_idx)
        remaining_nodes = [node for i, node in enumerate(node_order) if i not in set(elim_idx)]

        target_nodes, voltage_indices = _target_node_order(mpc_orig, mpc_fault, fault_busphase_map)
        remaining_index = {node: i for i, node in enumerate(remaining_nodes)}
        try:
            perm = [remaining_index[node] for node in target_nodes]
        except KeyError as exc:
            raise LegacyNLMBridgeError(f"Reduced Y-bus missing original IEEE-14 node {exc.args[0]!r}") from exc
        y_reduced = y_reduced[perm, :][:, perm]

        v_red = np.asarray([vc_full[idx] for idx in voltage_indices], dtype=complex)
        x_red = np.hstack([np.abs(v_red), np.angle(v_red)])
        busphase_map = {
            (int(row[0]), phase_idx): 3 * i + phase_idx
            for i, row in enumerate(mpc_orig["bus3p"])
            for phase_idx in range(3)
        }
        z = measurement_function(x_red, y_reduced, mpc_orig, busphase_map)
        n_phase_nodes = len(busphase_map)
        r_diag = np.array(
            [(1e-4) ** 2] * n_phase_nodes
            + [(1e-4) ** 2] * n_phase_nodes
            + [(2e-4) ** 2] * n_phase_nodes
        )
        r_matrix = np.diag(r_diag)

        dss.Command("Clear")
        dss.Command(f'Redirect "{pristine_master}"')
        y_orig, _ = build_global_y_per_unit(mpc_orig)
        (lagrangian_result, log) = _run_quietly(
            run_lagrangian_polar,
            z,
            x_red,
            busphase_map,
            y_orig,
            r_matrix,
            mpc_orig,
            max_iter=max_iter,
            capture_stdout=capture_stdout,
        )

    _x_est, converged, _lambda_n, sorted_group_distances = lagrangian_result
    top_groups = _rows_to_hif_groups(
        list(sorted_group_distances or []),
        mpc_orig,
        top_k=int(top_k),
    )
    if not top_groups:
        raise LegacyNLMBridgeError("Legacy NLM returned no mappable IEEE-14 line-group evidence.")

    target_row = target_branch_row0
    if target_row is None and target_dss_element:
        target_row = branch_row0_for_dss_element(target_dss_element)
    top_rows = [item["branch_row0"] for item in top_groups]
    payload: dict[str, Any] = {
        "success": bool(converged),
        "converged": bool(converged),
        "method": "legacy_three_phase_nlm",
        "backend": "Three_phase_NLM-cable-burnout",
        "top_hif_groups": top_groups,
        "detected": target_row in top_rows[:3] if target_row is not None else bool(top_groups),
        "detected_top1": bool(target_row is not None and top_rows and top_rows[0] == target_row),
        "detected_top3": bool(target_row is not None and target_row in top_rows[:3]),
    }
    return payload
