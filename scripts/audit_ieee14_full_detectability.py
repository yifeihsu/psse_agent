#!/usr/bin/env python3
"""Detectability audit of single-switch status errors in the full IEEE-14 node/breaker model.

Question answered: for each of the 73 switches in ``ieee14_full_schematic_v1``, if its TRUE
state differs from the reported (schematic-normal) state, does the error

  (a) change the connectivity partition, and
  (b) change the external SCADA measurements enough that the pipeline's ordinary 14-bus
      WLS (``mcp_server.matpower_server._wls_json`` on ``case14``) flags the reported
      model as anomalous under the round-0 chi-square gate?

Only flips that satisfy both are admissible topology-error scenarios. Flips that leave
the partition unchanged reproduce the normal-state measurements exactly under the
ideal-switch model; the script verifies that numerically instead of assuming it.

Measurement identity is fixed and physical (never averaged across sections):

  Vm[b]        voltage at a declared meter node of planning bus b (0 when de-energized);
               three placements are audited: the model's voltage anchor (injection
               bay), the main busbar ``{b}B1`` where one exists, and the energized
               main section (most line terminals), which admission is based on
  Pinj/Qinj[b] sum of the unit, load and shunt meters attached to planning bus b's
               equipment (a shed load or dropped unit reads zero)
  Pf,Qf,Pt,Qt  terminal flows of the 20 original branches in reference orientation

laid out in the operator order [Vm(14) Pinj(14) Qinj(14) Pf(20) Qf(20) Pt(20) Qt(20)].

Truth is solved with PYPOWER ``runopf`` on the ideal-switch contracted case (the
operating-point solver of the tabular corpus and of the round-0 synthesized families;
``--solver runpf`` reproduces the pre-2026-09-10 power-flow point). Sections that end up without a slack are
de-energized: their loads are shed and their units dropped, and that is recorded.

For splits that isolate exactly one line terminal, the script also checks whether the
existing branch-status correction (``correct_topology`` with ``status=0`` on that branch)
already makes the WLS clean, i.e. whether the current pipeline can represent the fix.

Usage:
    python scripts/audit_ieee14_full_detectability.py [--load-scales 0.80,1.0,1.25]
        [--seeds 10] [--out-dir models/ieee14_full_detectability] [--solver runopf|runpf]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypower.api import case14, ppoption, runopf, runpf  # noqa: E402
from pypower.idx_brch import BR_STATUS  # noqa: E402
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, REF, VMAX, VMIN  # noqa: E402
from pypower.idx_gen import GEN_BUS, PG, VG  # noqa: E402

from Transmission.ieee14_full_measurements import (  # noqa: E402
    MODEL_ID,
    classify_flip,
    flipped_case,
    main_section_nodes,
    operator_measurements,
)
from Transmission.ieee14_full_topology import build_full_topology  # noqa: E402
from mcp_server.matpower_server import _load_python_case, _wls_json  # noqa: E402
from psse_env.providers.matpower import _render_matpower_case  # noqa: E402
from tools.lagrangian_correct_port import make_ybus  # noqa: E402
from trace_protocol import chi2_threshold  # noqa: E402

NB, NL = 14, 20
NZ = 3 * NB + 4 * NL
STATE_COUNT = 2 * NB - 1
# Round-0 scenario generator defaults (psse_env/providers/scenario_generator.py).
CHI2_ALPHA = 0.01
ANOMALY_MARGIN = 1.25
# Weights of the deployment WLS port (tools/lagrangian_port.py): 1e-3 pu on Vm,
# 1e-2 pu on injections and flows.
SIGMA = np.r_[np.full(NB, 1e-3), np.full(2 * NB, 1e-2), np.full(4 * NL, 1e-2)]


# ----------------------------------------------------------------------------- reference


def reference_case(load_scale: float, solver: str = "runopf") -> dict:
    """PYPOWER case14 at a load scale.

    ``runopf`` (default) leaves dispatch to the OPF, which is how the tabular corpus
    and the round-0 synthesized families set their operating point.  ``runpf`` keeps
    the case14 dispatch scaled by load with voltage setpoints clamped to the bus
    limits, which is what the pre-2026-09-10 topology builder did.
    """
    ppc = case14()
    ppc["bus"][:, PD] *= load_scale
    ppc["bus"][:, QD] *= load_scale
    if solver == "runopf":
        return ppc
    bounds = {int(r[BUS_I]): (float(r[VMIN]), float(r[VMAX])) for r in ppc["bus"]}
    for g in ppc["gen"]:
        lo, hi = bounds[int(g[GEN_BUS])]
        g[VG] = min(max(float(g[VG]), lo), hi)
    slack = set(ppc["bus"][ppc["bus"][:, BUS_TYPE] == REF, BUS_I].astype(int))
    for g in ppc["gen"]:
        if int(g[GEN_BUS]) not in slack:
            g[PG] *= load_scale
    return ppc


def build_measurement_vector(ppc: dict) -> np.ndarray:
    """h(x) at the stored state; mirrors scenario_generator.build_measurement_vector."""
    bus = np.asarray(ppc["bus"], dtype=float).copy()
    branch = np.asarray(ppc["branch"], dtype=float).copy()
    branch[:, 0] -= 1.0
    branch[:, 1] -= 1.0
    ybus, yf, yt = make_ybus(float(ppc["baseMVA"]), bus, branch)
    v = bus[:, 7] * np.exp(1j * np.pi / 180.0 * bus[:, 8])
    inj = v * np.conj(ybus @ v)
    f = branch[:, 0].astype(int)
    t = branch[:, 1].astype(int)
    sf = v[f] * np.conj(yf @ v)
    st = v[t] * np.conj(yt @ v)
    return np.r_[np.abs(v), inj.real, inj.imag, sf.real, sf.imag, st.real, st.imag]


# ------------------------------------------------------------------------------- physics


def solve(case: dict, solver: str) -> tuple[dict, bool]:
    if solver == "runopf":
        sol = runopf(deepcopy(case), ppoption(VERBOSE=0, OUT_ALL=0))
        return sol, bool(sol.get("success"))
    sol, ok = runpf(deepcopy(case), ppoption(VERBOSE=0, OUT_ALL=0))
    return sol, bool(ok)


# ----------------------------------------------------------------------------------- WLS


def wls_objective(case_path: str, z: np.ndarray) -> float:
    payload = _wls_json(case_path, [float(x) for x in z])
    if not payload.get("success"):
        return float("nan")
    return float(payload.get("global_residual_sum") or 0.0)


def derived_case_path(ppc: dict, tag: str, tmpdir: str) -> str:
    text = _render_matpower_case(ppc, f"derived_{tag}")
    path = os.path.join(tmpdir, f"{tag}.m")
    Path(path).write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------- main


def audit(load_scales: list[float], seeds: int, out_dir: Path, solver: str = "runopf") -> dict:
    model = build_full_topology()
    normal_groups = model.components()
    busbar_nodes = {b: (f"{b}B1" if f"{b}B1" in model.nodes else model.anchors[b]) for b in range(1, NB + 1)}
    anchor_nodes = dict(model.anchors)
    dof = NZ - STATE_COUNT
    limit = float(chi2_threshold(dof, CHI2_ALPHA))
    detect_at = ANOMALY_MARGIN * limit
    rng = np.random.default_rng(20260910)
    tmpdir = tempfile.mkdtemp(prefix="ieee14_full_audit_")

    rows: dict[str, dict] = {}
    normal_checks = []
    for scale in load_scales:
        ref = reference_case(scale, solver)
        case_n, info_n, dz_n = flipped_case(ref, {}, model=model)
        sol_n, ok_n = solve(case_n, solver)
        if not ok_n or dz_n["dead_buses"]:
            raise RuntimeError(f"normal state failed at load_scale={scale}: {dz_n}")
        lookup_n = info_n["node_to_bus"]
        z_normal = operator_measurements(sol_n, ref, model, lookup_n, [], anchor_nodes)
        z_busbar_normal = operator_measurements(sol_n, ref, model, lookup_n, [], busbar_nodes)
        h_normal = build_measurement_vector(sol_n)
        consistency = float(np.max(np.abs(z_normal - h_normal)))
        j_normal = wls_objective("case14", z_normal)
        normal_checks.append({
            "load_scale": scale,
            "fixed_identity_vs_h_x_max_abs_diff": consistency,
            "J_case14_noiseless": j_normal,
            "normal_state_clean": bool(j_normal < limit),
            "vm_anchor_equals_vm_busbar": bool(np.allclose(z_normal, z_busbar_normal)),
        })
        if consistency > 1e-6:
            raise RuntimeError(f"fixed-identity vector disagrees with h(x) in the normal state: {consistency}")

        for cb in model.breakers:
            true_closed = not cb.closed
            flipped_groups = model.components({cb.name: true_closed})
            cls = classify_flip(model, ref, flipped_groups, normal_groups)
            case_f, info_f, dz = flipped_case(ref, {cb.name: true_closed}, model=model)
            sol_f, ok_f = solve(case_f, solver)
            entry = rows.setdefault(cb.name, {
                "cb_name": cb.name, "yard": cb.yard, "reported_closed": cb.closed,
                "true_closed": true_closed, "partition_changed": cls["effect"] != "equivalent",
                "topological_buses_true": len(flipped_groups), **cls, "per_scale": {},
            })
            per = {"load_scale": scale, "pf_converged": ok_f, **dz}
            if ok_f:
                lookup_f = info_f["node_to_bus"]
                placements = {
                    "anchor": anchor_nodes,
                    "busbar": busbar_nodes,
                    "main": main_section_nodes(model, lookup_f, dz["dead_buses"]),
                }
                z_by = {name: operator_measurements(sol_f, ref, model, lookup_f, dz["dead_buses"], nodes)
                        for name, nodes in placements.items()}
                per["measurements_changed"] = False
                for name, z_flip in z_by.items():
                    dz_sigma = float(np.max(np.abs(z_flip - z_normal) / SIGMA))
                    per[f"max_abs_dz_over_sigma_{name}_vm"] = dz_sigma
                    per["measurements_changed"] = bool(per["measurements_changed"] or dz_sigma > 1e-6)
                    j = wls_objective("case14", z_flip)
                    per[f"J_case14_{name}_vm"] = j
                    per[f"wls_failed_{name}_vm"] = bool(not np.isfinite(j))
                    per[f"detect_noiseless_{name}_vm"] = bool(np.isfinite(j) and j > detect_at)
                    hits = 0
                    for _ in range(seeds):
                        jn = wls_objective("case14", z_flip + rng.normal(0.0, SIGMA))
                        hits += bool(np.isfinite(jn) and jn > detect_at)
                    per[f"detection_rate_noisy_{name}_vm"] = hits / seeds if seeds else None
                if cls["category"] == "dangling_line_terminal":
                    k = cls["minor_section"]["terminal_rows"][0]
                    corr = deepcopy(_load_python_case("case14"))
                    corr["branch"][k][BR_STATUS] = 0.0
                    path = derived_case_path(corr, f"case14_br{k}_out", tmpdir)
                    per["equivalent_branch_row0"] = k
                    per["J_branch_out_correction_main_vm"] = wls_objective(path, z_by["main"])
                    per["branch_status_correction_clean"] = bool(
                        np.isfinite(per["J_branch_out_correction_main_vm"])
                        and per["J_branch_out_correction_main_vm"] < limit)
            entry["per_scale"][str(scale)] = per

    # Aggregate across load scales.
    for entry in rows.values():
        per = list(entry["per_scale"].values())
        entry["pf_converged_all_scales"] = all(p["pf_converged"] for p in per)
        conv = [p for p in per if p["pf_converged"]]
        entry["measurements_changed"] = any(p["measurements_changed"] for p in conv) if conv else None
        entry["shed_load_mw_max"] = max((p["shed_p_mw"] for p in per), default=0.0)
        entry["dead_sections_any_scale"] = any(p["dead_buses"] for p in per)
        entry["dropped_gen_any_scale"] = any(p["dropped_gen_rows"] for p in per)
        for key in ("detect_noiseless_anchor_vm", "detect_noiseless_busbar_vm", "detect_noiseless_main_vm",
                    "branch_status_correction_clean"):
            vals = [p[key] for p in conv if key in p]
            entry[key + "_all_scales"] = (all(vals) if vals else None)
        for key in ("wls_failed_anchor_vm", "wls_failed_busbar_vm", "wls_failed_main_vm"):
            vals = [p[key] for p in conv if key in p]
            entry[key + "_any_scale"] = (any(vals) if vals else None)
        for key in ("detection_rate_noisy_anchor_vm", "detection_rate_noisy_busbar_vm",
                    "detection_rate_noisy_main_vm"):
            vals = [p[key] for p in conv if p.get(key) is not None]
            entry[key + "_min"] = (min(vals) if vals else None)
        for key in ("J_case14_anchor_vm", "J_case14_busbar_vm", "J_case14_main_vm",
                    "J_branch_out_correction_main_vm"):
            vals = [p[key] for p in conv if key in p and np.isfinite(p[key])]
            entry[key + "_min"] = (min(vals) if vals else None)
            entry[key + "_max"] = (max(vals) if vals else None)
        for key in ("max_abs_dz_over_sigma_main_vm", "max_abs_dz_over_sigma_anchor_vm"):
            vals = [p[key] for p in conv if key in p]
            entry[key + "_min"] = (min(vals) if vals else None)
        entry["equivalent_branch_row0"] = next((p.get("equivalent_branch_row0") for p in per
                                                if p.get("equivalent_branch_row0") is not None), None)
        # Admission rule: partition changed AND the ordinary WLS on the reported model
        # detects it at every load scale. Conservative = voltage meter on the energized
        # main section (no dead-bay reading is needed for the detection).
        entry["admissible_conservative"] = bool(
            entry["partition_changed"] and entry["pf_converged_all_scales"]
            and entry["detect_noiseless_main_vm_all_scales"]
        )
        entry["admissible_with_dead_bay_voltage"] = bool(
            entry["partition_changed"] and entry["pf_converged_all_scales"]
            and entry["detect_noiseless_anchor_vm_all_scales"]
        )

    summary = {
        "model_id": MODEL_ID,
        "model_fingerprint": model.fingerprint(),
        "operating_point_solver": solver,
        "load_scales": load_scales,
        "noisy_seeds_per_scale": seeds,
        "chi2_dof": dof,
        "chi2_alpha": CHI2_ALPHA,
        "chi2_limit": limit,
        "anomaly_detect_threshold": detect_at,
        "normal_state_checks": normal_checks,
        "counts_by_category": {},
        "admissible_conservative": sorted(k for k, e in rows.items() if e["admissible_conservative"]),
        "admissible_only_with_dead_bay_voltage": sorted(
            k for k, e in rows.items()
            if e["admissible_with_dead_bay_voltage"] and not e["admissible_conservative"]),
        "excluded_partition_unchanged": sorted(k for k, e in rows.items() if not e["partition_changed"]),
        "excluded_partition_changed_but_wls_blind": sorted(
            k for k, e in rows.items()
            if e["partition_changed"] and not e["admissible_with_dead_bay_voltage"]
            and not e["admissible_conservative"]),
        "wls_solver_failed_dead_bay_voltage": sorted(
            k for k, e in rows.items() if e.get("wls_failed_anchor_vm_any_scale")),
        "dangling_terminal_fixable_by_branch_status": sorted(
            k for k, e in rows.items() if e.get("branch_status_correction_clean_all_scales")),
        "dangling_terminal_not_fixable_by_branch_status": sorted(
            k for k, e in rows.items()
            if e["category"] == "dangling_line_terminal" and not e.get("branch_status_correction_clean_all_scales")),
    }
    for e in rows.values():
        c = summary["counts_by_category"].setdefault(e["category"], {"total": 0, "admissible_conservative": 0,
                                                                     "admissible_with_dead_bay_voltage": 0})
        c["total"] += 1
        c["admissible_conservative"] += int(e["admissible_conservative"])
        c["admissible_with_dead_bay_voltage"] += int(e["admissible_with_dead_bay_voltage"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "single_flip_detectability.json").write_text(
        json.dumps({"summary": summary, "flips": rows}, indent=2, default=_json_default), encoding="utf-8")
    columns = ["cb_name", "yard", "reported_closed", "true_closed", "partition_changed", "effect", "category",
               "affected_planning_buses", "topological_buses_true", "measurements_changed",
               "max_abs_dz_over_sigma_main_vm_min", "dead_sections_any_scale", "shed_load_mw_max",
               "dropped_gen_any_scale", "J_case14_main_vm_min", "J_case14_main_vm_max",
               "J_case14_busbar_vm_min", "J_case14_anchor_vm_min",
               "detect_noiseless_main_vm_all_scales", "detect_noiseless_busbar_vm_all_scales",
               "detect_noiseless_anchor_vm_all_scales", "wls_failed_anchor_vm_any_scale",
               "detection_rate_noisy_main_vm_min", "detection_rate_noisy_anchor_vm_min",
               "equivalent_branch_row0", "J_branch_out_correction_main_vm_min",
               "branch_status_correction_clean_all_scales", "admissible_conservative",
               "admissible_with_dead_bay_voltage"]
    with open(out_dir / "single_flip_detectability.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for e in rows.values():
            writer.writerow({k: _csv_value(e.get(k)) for k in columns})
    return {"summary": summary, "flips": rows}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unserializable {type(value)}")


def _csv_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=_json_default)
    return value


def _print_report(result: dict) -> None:
    s = result["summary"]
    print(f"model {s['model_id']} fingerprint {s['model_fingerprint'][:12]} solver {s['operating_point_solver']}")
    print(f"chi2 limit (dof={s['chi2_dof']}, alpha={s['chi2_alpha']}) = {s['chi2_limit']:.1f}; "
          f"detect above {s['anomaly_detect_threshold']:.1f}")
    for chk in s["normal_state_checks"]:
        print(f"  normal state @load {chk['load_scale']}: J={chk['J_case14_noiseless']:.3g} "
              f"clean={chk['normal_state_clean']} h(x) agreement={chk['fixed_identity_vs_h_x_max_abs_diff']:.2e}")
    print("\ncategory                 total  admissible(main-section Vm)  admissible(dead-bay Vm)")
    for cat, c in sorted(s["counts_by_category"].items()):
        print(f"{cat:<24s} {c['total']:>5d}  {c['admissible_conservative']:>26d}  {c['admissible_with_dead_bay_voltage']:>22d}")
    print(f"\nadmissible (conservative, main-section Vm): {len(s['admissible_conservative'])}")
    print(f"admissible only with dead-bay voltage: {len(s['admissible_only_with_dead_bay_voltage'])}"
          f" {s['admissible_only_with_dead_bay_voltage']}")
    print(f"excluded, partition unchanged: {len(s['excluded_partition_unchanged'])}")
    print(f"excluded, partition changed but WLS blind: {len(s['excluded_partition_changed_but_wls_blind'])}"
          f" {s['excluded_partition_changed_but_wls_blind']}")
    print(f"WLS solver failed on a dead-bay voltage reading: {s['wls_solver_failed_dead_bay_voltage']}")
    print(f"dangling terminal fixable by branch-status correction: "
          f"{len(s['dangling_terminal_fixable_by_branch_status'])} / "
          f"{len(s['dangling_terminal_fixable_by_branch_status']) + len(s['dangling_terminal_not_fixable_by_branch_status'])}"
          f"  not fixable: {s['dangling_terminal_not_fixable_by_branch_status']}")
    print("\n%-22s %-6s %-22s %-5s %-8s %-9s %-9s %-6s %-6s %-6s %-6s %-6s" % (
        "cb", "yard", "category", "nbus", "dz/sig", "J_main", "J_anchor", "detM", "detA", "rateM", "shed", "brfix"))
    for e in sorted(result["flips"].values(), key=lambda e: (e["category"], e["yard"], e["cb_name"])):
        jm = e.get("J_case14_main_vm_min"); ja = e.get("J_case14_anchor_vm_min")
        dz = e.get("max_abs_dz_over_sigma_main_vm_min"); rm = e.get("detection_rate_noisy_main_vm_min")
        print("%-22s %-6s %-22s %-5d %-8s %-9s %-9s %-6s %-6s %-6s %-6.1f %-6s" % (
            e["cb_name"], e["yard"], e["category"], e["topological_buses_true"],
            f"{dz:.3g}" if dz is not None else "n/a",
            f"{jm:.3g}" if jm is not None else "n/a", f"{ja:.3g}" if ja is not None else "n/a",
            e["detect_noiseless_main_vm_all_scales"], e["detect_noiseless_anchor_vm_all_scales"],
            f"{rm:.2f}" if rm is not None else "n/a",
            e["shed_load_mw_max"], e.get("branch_status_correction_clean_all_scales")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-scales", default="0.80,1.0,1.25",
                        help="corpus load-scale range endpoints and midpoint by default")
    parser.add_argument("--seeds", type=int, default=10, help="noisy replicates per flip and load scale")
    parser.add_argument("--out-dir", default=str(ROOT / "models" / "ieee14_full_detectability"))
    parser.add_argument("--solver", choices=("runopf", "runpf"), default="runopf",
                        help="operating-point solver: AC OPF (corpus and round-0 default) or plain power flow")
    args = parser.parse_args()
    scales = [float(x) for x in args.load_scales.split(",") if x.strip()]
    result = audit(scales, args.seeds, Path(args.out_dir), solver=args.solver)
    _print_report(result)
    print(f"\nwrote {Path(args.out_dir) / 'single_flip_detectability.json'} and .csv")


if __name__ == "__main__":
    main()
