#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate IEEE-14 high-impedance-fault measurement records.

The operator-facing vector remains the standard 122-entry IEEE-14 layout:
    z = [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]

The hidden HIF bus is only present inside the copied OpenDSS scenario model.
It is not exposed as a bus in z_obs or z_true.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

try:
    import opendssdirect as dss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("opendssdirect is required for IEEE-14 HIF generation.") from e

from pypower.api import case14, ppoption, runopf  # type: ignore
from pypower.idx_brch import BR_STATUS, F_BUS, TAP, T_BUS  # type: ignore
from pypower.idx_bus import PD, QD  # type: ignore

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from IEEE_14_OpenDSS.export_measurement_series import (  # type: ignore
    BRANCH_ORDER,
    BUS_ORDER,
    extract_measurement_series,
    extract_three_phase_voltage_measurements,
)
from three_phase_nlm import (  # type: ignore
    copy_ieee14_model,
    hif_ohms_from_pu,
    inject_midspan_hif_ieee14,
    run_ieee14_hif_nlm,
    write_balanced_ieee14_load_override,
)
from three_phase_nlm.ieee14_adapter import ELIGIBLE_HIF_BRANCHES, branch_info_for_row0
from Transmission.generate_measurements import (  # type: ignore
    MEASUREMENT_ORDER,
    compute_measurements_pu,
    make_index_map,
)


def _compile_ieee14_opendss(model_dir: str | Path) -> None:
    dss.Basic.DataPath(str(Path(model_dir).resolve()))
    dss.Text.Command("Clear")
    dss.Text.Command("Redirect Run_IEEE14Bus.dss")


def _solve_or_raise() -> None:
    dss.Text.Command("Solve")
    if hasattr(dss, "Solution") and not bool(dss.Solution.Converged()):
        raise RuntimeError("OpenDSS solve did not converge")


def _scale_pypower_loads(ppc: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    ppc2 = deepcopy(ppc)
    ppc2["bus"][:, PD] *= float(alpha)
    ppc2["bus"][:, QD] *= float(alpha)
    return ppc2


def _solve_pypower(ppc: Dict[str, Any]) -> Dict[str, Any] | None:
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    result = runopf(ppc, ppopt)
    return result if result.get("success") else None


def _enabled_load_snapshot() -> dict[str, dict[str, Any]]:
    base: dict[str, dict[str, Any]] = {}
    for name in dss.Loads.AllNames() or []:
        dss.Loads.Name(name)
        if str(name).lower().startswith("hif_"):
            continue
        if hasattr(dss.CktElement, "Enabled") and not bool(dss.CktElement.Enabled()):
            continue
        base[str(name).lower()] = {
            "name": str(name),
            "kW": float(dss.Loads.kW()),
            "kvar": float(dss.Loads.kvar()),
        }
    return base


def _scale_named_loads(loads: Mapping[str, Mapping[str, Any]], load_scale: float) -> None:
    for item in loads.values():
        dss.Loads.Name(str(item["name"]))
        dss.Loads.kW(float(item["kW"]) * float(load_scale))
        dss.Loads.kvar(float(item["kvar"]) * float(load_scale))


def _branch_info_case14() -> list[dict[str, Any]]:
    ppc = case14()
    br = ppc["branch"]
    out = []
    for i in range(br.shape[0]):
        out.append(
            {
                "i": int(i),
                "from_bus": int(br[i, F_BUS]),
                "to_bus": int(br[i, T_BUS]),
                "is_line": bool(float(br[i, TAP]) == 0.0 and float(br[i, BR_STATUS]) > 0.0),
            }
        )
    return out


def _maybe_add_noise(z_obs: list[float], rng: np.random.Generator, noise_scale: float) -> list[float]:
    if float(noise_scale) <= 0:
        return [float(x) for x in z_obs]
    arr = np.asarray(z_obs, dtype=float)
    nb = 14
    nl = 20
    sigma = np.zeros_like(arr)
    sigma[0:nb] = 1e-3
    sigma[nb : 3 * nb] = 1e-2
    sigma[3 * nb : 3 * nb + 4 * nl] = 1e-2
    return (arr + rng.normal(0.0, sigma * float(noise_scale))).astype(float).tolist()


def _build_meta(
    *,
    load_scale_min: float,
    load_scale_max: float,
    r_hif_pu_min: float,
    r_hif_pu_max: float,
    split_min: float,
    split_max: float,
    branch_sampling: str,
) -> dict[str, Any]:
    nb = 14
    nl = 20
    idx_map = make_index_map(nb, nl)
    return {
        "case": "case14",
        "baseMVA": 100.0,
        "nb": nb,
        "nl": nl,
        "index_map": {k: [int(v.start), int(v.stop)] for k, v in idx_map.items()},
        "measurement_order": MEASUREMENT_ORDER,
        "branch_info": _branch_info_case14(),
        "hif": {
            "scenario": "high_impedance_fault",
            "eligible_branch_row0": [int(i) for i in ELIGIBLE_HIF_BRANCHES],
            "branch_order": BRANCH_ORDER,
            "bus_order": BUS_ORDER,
            "load_scale_range": [float(load_scale_min), float(load_scale_max)],
            "r_hif_pu_range": [float(r_hif_pu_min), float(r_hif_pu_max)],
            "split_ratio_range": [float(split_min), float(split_max)],
            "branch_sampling": str(branch_sampling),
            "phases": ["A", "B", "C"],
            "measurement_vector": "operator IEEE-14 122-entry z; hidden fault bus excluded",
            "nlm_diagnostic": {
                "fields": ["success", "converged", "top_hif_groups", "detected_top1", "detected_top3"],
                "note": "Generated HIF samples use the legacy three-phase NLM bridge when scenario models are available; metadata fallback is only for adapter smoke tests without model-backed evidence.",
            },
        },
    }


def generate_dataset(
    *,
    out_dir: str,
    n_hif: int,
    n_no_error: int,
    seed: int,
    load_scale_min: float,
    load_scale_max: float,
    r_hif_pu_min: float,
    r_hif_pu_max: float,
    split_min: float,
    split_max: float,
    noise_scale: float,
    keep_scenarios: bool,
    branch_sampling: str,
) -> None:
    rng = np.random.default_rng(seed)
    out = Path(os.path.abspath(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    meta = _build_meta(
        load_scale_min=load_scale_min,
        load_scale_max=load_scale_max,
        r_hif_pu_min=r_hif_pu_min,
        r_hif_pu_max=r_hif_pu_max,
        split_min=split_min,
        split_max=split_max,
        branch_sampling=branch_sampling,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ppc_base = case14()
    base_dss_dir = Path(_REPO_ROOT) / "IEEE_14_OpenDSS"
    scenarios_root = out / "scenarios"
    if keep_scenarios:
        scenarios_root.mkdir(exist_ok=True)

    eligible_hif_branches = [int(i) for i in ELIGIBLE_HIF_BRANCHES]
    if branch_sampling not in {"balanced", "random"}:
        raise ValueError(f"Unknown branch_sampling={branch_sampling!r}")
    branch_schedule: list[int] = []
    if branch_sampling == "balanced":
        while len(branch_schedule) < int(n_hif):
            branch_schedule.extend(int(i) for i in rng.permutation(eligible_hif_branches))
        branch_schedule = branch_schedule[: int(n_hif)]

    with (out / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for _ in range(int(n_no_error)):
            alpha = float(rng.uniform(load_scale_min, load_scale_max))
            solved = _solve_pypower(_scale_pypower_loads(ppc_base, alpha))
            if solved is None:
                continue
            z = compute_measurements_pu(solved).astype(float).tolist()
            rec = {
                "id": f"nehif_{rng.integers(1e12)}",
                "scenario": "no_error",
                "z_true": z,
                "z_obs": _maybe_add_noise(z, rng, noise_scale),
                "label": {"error_type": "no_error"},
                "op_point": {"load_scale": alpha, "seed": int(seed)},
            }
            handle.write(json.dumps(rec) + "\n")

        for sample_idx in range(int(n_hif)):
            sample_id = f"ieee14_hif_{sample_idx:06d}"
            alpha = float(rng.uniform(load_scale_min, load_scale_max))
            branch_row0 = (
                int(branch_schedule[sample_idx])
                if branch_sampling == "balanced"
                else int(rng.choice(eligible_hif_branches))
            )
            dss_element = BRANCH_ORDER[branch_row0]
            split_ratio = float(rng.uniform(split_min, split_max))
            phase = str(rng.choice(["A", "B", "C"]))
            r_hif_pu = float(rng.uniform(r_hif_pu_min, r_hif_pu_max))
            r_hif_ohm = hif_ohms_from_pu(r_hif_pu, base_mva=100.0, kv_ll=1.0)

            if keep_scenarios:
                scenario_dir = scenarios_root / sample_id
                if scenario_dir.exists():
                    shutil.rmtree(scenario_dir)
                tmp_context = None
            else:
                tmp_context = tempfile.TemporaryDirectory(prefix=f"{sample_id}_")
                scenario_dir = Path(tmp_context.name)

            try:
                copy_ieee14_model(base_dss_dir, scenario_dir, overwrite=True)
                write_balanced_ieee14_load_override(scenario_dir)
                injection = inject_midspan_hif_ieee14(
                    scenario_dir,
                    dss_element,
                    split_ratio=split_ratio,
                    phase=phase,
                    r_hif_ohm=r_hif_ohm,
                    base_mva=100.0,
                    kv_ll=1.0,
                    fault_bus=f"Fault_{branch_row0 + 1}_{sample_idx:06d}",
                    hif_load_name=f"Load.HIF_{branch_row0 + 1}_{sample_idx:06d}",
                )

                _compile_ieee14_opendss(scenario_dir)
                base_loads = _enabled_load_snapshot()
                _scale_named_loads(base_loads, alpha)
                _solve_or_raise()

                z_obs, buses, branches = extract_measurement_series(
                    branch_element_overrides=injection.branch_element_overrides
                )
                if len(z_obs) != 122:
                    raise RuntimeError(f"Unexpected z_obs length={len(z_obs)}, expected 122")
                if list(buses) != list(BUS_ORDER):
                    raise RuntimeError("Unexpected bus order from OpenDSS extractor.")
                if list(branches) != list(BRANCH_ORDER):
                    raise RuntimeError("Unexpected branch order from OpenDSS extractor.")
                three_phase_voltages = extract_three_phase_voltage_measurements()

                solved = _solve_pypower(_scale_pypower_loads(ppc_base, alpha))
                if solved is None:
                    continue
                z_true = compute_measurements_pu(solved).astype(float).tolist()

                branch_info = branch_info_for_row0(branch_row0)
                nlm_diagnostic = run_ieee14_hif_nlm(
                    pristine_model_dir=str(base_dss_dir),
                    faulted_model_dir=str(scenario_dir),
                    target_dss_element=dss_element,
                    phase=phase,
                    r_hif_ohm=r_hif_ohm,
                    target_branch_row0=branch_row0,
                    load_scale=alpha,
                )
                top_rows = [
                    group.get("branch_row0")
                    for group in nlm_diagnostic.get("top_hif_groups", [])
                    if isinstance(group, Mapping)
                ]
                nlm_diagnostic["detected_top1"] = bool(top_rows and top_rows[0] == branch_row0)
                nlm_diagnostic["detected_top3"] = branch_row0 in top_rows[:3]
                nlm_diagnostic["detected"] = bool(nlm_diagnostic["detected_top3"])

                rec = {
                    "id": sample_id,
                    "scenario": "high_impedance_fault",
                    "case": "IEEE14",
                    "z_true": z_true,
                    "z_obs": _maybe_add_noise([float(x) for x in z_obs], rng, noise_scale),
                    "three_phase_voltages": three_phase_voltages,
                    "label": {
                        "error_type": "high_impedance_fault",
                        "branch_row0": branch_row0,
                        "line_index1": branch_row0 + 1,
                        "dss_element": dss_element,
                        "from_bus": branch_info["from_bus"],
                        "to_bus": branch_info["to_bus"],
                        "phase": phase,
                        "split_ratio": split_ratio,
                        "fault_bus": injection.fault_bus,
                        "r_hif_pu": r_hif_pu,
                        "r_hif_ohm": r_hif_ohm,
                        "kv_ln": injection.kv_ln,
                    },
                    "nlm_diagnostic": nlm_diagnostic,
                    "op_point": {
                        "load_scale": alpha,
                        "seed": int(seed),
                        "sample_index": int(sample_idx),
                    },
                }
                if keep_scenarios:
                    rec["scenario_model_dir"] = str(scenario_dir.relative_to(out))
                handle.write(json.dumps(rec) + "\n")
            finally:
                if tmp_context is not None:
                    tmp_context.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out_measurements_hif", help="Output directory")
    parser.add_argument("--n-hif", type=int, default=200)
    parser.add_argument("--n-no-error", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--load-scale-min", type=float, default=0.80)
    parser.add_argument("--load-scale-max", type=float, default=1.25)
    parser.add_argument("--r-hif-pu-min", type=float, default=20.0)
    parser.add_argument("--r-hif-pu-max", type=float, default=200.0)
    parser.add_argument("--split-min", type=float, default=0.25)
    parser.add_argument("--split-max", type=float, default=0.75)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--keep-scenarios", action="store_true")
    parser.add_argument(
        "--branch-sampling",
        choices=["balanced", "random"],
        default="balanced",
        help="Balanced cycles through eligible Line.* branches; random samples with replacement.",
    )
    args = parser.parse_args()

    generate_dataset(
        out_dir=args.out,
        n_hif=args.n_hif,
        n_no_error=args.n_no_error,
        seed=args.seed,
        load_scale_min=args.load_scale_min,
        load_scale_max=args.load_scale_max,
        r_hif_pu_min=args.r_hif_pu_min,
        r_hif_pu_max=args.r_hif_pu_max,
        split_min=args.split_min,
        split_max=args.split_max,
        noise_scale=args.noise_scale,
        keep_scenarios=bool(args.keep_scenarios),
        branch_sampling=args.branch_sampling,
    )
    print(f"Wrote IEEE-14 HIF dataset to: {args.out}")


if __name__ == "__main__":
    main()
