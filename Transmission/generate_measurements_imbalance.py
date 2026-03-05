#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a dataset for the IEEE-14 three-phase imbalance workflow.

Key idea
- Operator-facing measurements stay in the standard 1ϕ-equivalent z layout (122 entries).
- When imbalance is detected, the operator requests 3ϕ substation voltage measurements.
  We attach those 3ϕ voltages to the sample record as additional context.

Outputs (out_dir):
- samples.jsonl: one JSON object per scenario
- meta.json: index map + branch order info (aligned with MATPOWER case14)

Scenarios produced here:
- three_phase_imbalance: z_obs comes from OpenDSS unbalanced PF (phase-A + 3ϕ totals),
  and we attach per-bus 3ϕ VLN voltage measurements.
- no_error: balanced positive-sequence z_obs generated from PYPOWER (optional).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import opendssdirect as dss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "opendssdirect is required for three-phase imbalance dataset generation."
    ) from e

from pypower.api import case14, runopf, ppoption  # type: ignore
from pypower.idx_bus import PD, QD  # type: ignore
from pypower.idx_brch import BR_STATUS, F_BUS, TAP, T_BUS  # type: ignore

# Ensure repo root is importable when running as a script (python Transmission/....py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from IEEE_14_OpenDSS.export_measurement_series import (  # type: ignore
    BRANCH_ORDER,
    BUS_ORDER,
    extract_measurement_series,
    extract_three_phase_voltage_measurements,
)

from Transmission.generate_measurements import compute_measurements_pu, make_index_map  # type: ignore


def _scale_pypower_loads(ppc: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    ppc2 = deepcopy(ppc)
    ppc2["bus"][:, PD] *= float(alpha)
    ppc2["bus"][:, QD] *= float(alpha)
    return ppc2


def _solve_pypower(ppc: Dict[str, Any]) -> Dict[str, Any] | None:
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    res = runopf(ppc, ppopt)
    return res if res.get("success") else None


def _compile_ieee14_opendss(repo_dir: str) -> None:
    dss.Basic.DataPath(repo_dir)
    dss.Text.Command("Clear")
    dss.Text.Command("Redirect Run_IEEE14Bus.dss")


def _read_base_loads() -> Dict[str, Tuple[float, float]]:
    base: Dict[str, Tuple[float, float]] = {}
    for name in dss.Loads.AllNames() or []:
        dss.Loads.Name(name)
        base[str(name).lower()] = (float(dss.Loads.kW()), float(dss.Loads.kvar()))
    return base


def _set_loads_scaled_with_bus3_unbalance(
    base_loads: Dict[str, Tuple[float, float]],
    *,
    load_scale: float,
    bus3_fracs: Tuple[float, float, float],
) -> Dict[str, Any]:
    """
    Apply load scaling and overwrite Bus 3 phase loads (B3A/B3B/B3C) to the requested split.

    Returns a dict of the actually applied per-phase P/Q (kW/kvar) for labeling.
    """
    a, b, c = [float(x) for x in bus3_fracs]
    s = a + b + c
    if s <= 0:
        raise ValueError("bus3_fracs must sum to a positive value")
    a, b, c = a / s, b / s, c / s

    # scale all loads first (including B3A/B3B/B3C to set a baseline), then override bus3 split
    for name, (kw0, kvar0) in base_loads.items():
        dss.Loads.Name(name)
        dss.Loads.kW(float(kw0) * float(load_scale))
        dss.Loads.kvar(float(kvar0) * float(load_scale))

    # overwrite the bus3 split while preserving total P/Q (sum of base B3A/B3B/B3C)
    bus3_names = ["b3a", "b3b", "b3c"]
    totals = [base_loads.get(n) for n in bus3_names]
    if any(v is None for v in totals):
        missing = [n for n, v in zip(bus3_names, totals) if v is None]
        raise RuntimeError(f"Expected Bus 3 split loads not found in OpenDSS: {missing}")

    p_tot = sum(v[0] for v in totals) * float(load_scale)
    q_tot = sum(v[1] for v in totals) * float(load_scale)
    splits = {"a": a, "b": b, "c": c}
    applied = {}
    for name, frac in zip(bus3_names, (a, b, c)):
        dss.Loads.Name(name)
        kw = float(p_tot) * float(frac)
        kvar = float(q_tot) * float(frac)
        dss.Loads.kW(kw)
        dss.Loads.kvar(kvar)
        applied[name] = {"kW": kw, "kvar": kvar, "frac": frac}
    applied["total"] = {"kW": float(p_tot), "kvar": float(q_tot)}
    applied["fractions"] = splits
    return applied


def _branch_info_case14() -> List[Dict[str, Any]]:
    """Branch info in MATPOWER case14 order."""
    ppc = case14()
    br = ppc["branch"]
    out = []
    for i in range(br.shape[0]):
        out.append(
            dict(
                i=int(i),
                from_bus=int(br[i, F_BUS]),
                to_bus=int(br[i, T_BUS]),
                is_line=bool(float(br[i, TAP]) == 0.0 and float(br[i, BR_STATUS]) > 0.0),
            )
        )
    return out


def generate_dataset(
    *,
    out_dir: str,
    n_imbalance: int,
    n_no_error: int,
    seed: int,
    load_scale_min: float,
    load_scale_max: float,
    dirichlet_alpha: float,
) -> None:
    rng = np.random.default_rng(seed)
    # IMPORTANT: OpenDSS `Basic.DataPath()` changes the process CWD, so always use absolute output paths.
    out = Path(os.path.abspath(out_dir))
    out.mkdir(parents=True, exist_ok=True)

    # --- meta.json (operator measurement map) ---
    nb = 14
    nl = 20
    idx_map = make_index_map(nb, nl)
    meta = dict(
        case="case14",
        baseMVA=100.0,
        nb=nb,
        nl=nl,
        index_map={k: [int(v.start), int(v.stop)] for k, v in idx_map.items()},
        branch_info=_branch_info_case14(),
        imbalance=dict(
            bus_order=BUS_ORDER,
            branch_order=BRANCH_ORDER,
            three_phase_voltage_measurements=dict(
                type="VLN",
                phases=["A", "B", "C"],
                fields=["vln_pu", "ang_deg", "kvbase_ln"],
            ),
        ),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- OpenDSS init (compile once) ---
    dss_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "IEEE_14_OpenDSS"))
    _compile_ieee14_opendss(dss_repo)
    base_loads = _read_base_loads()

    # --- write samples.jsonl ---
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        # 1) balanced controls (positive-sequence)
        ppc_base = case14()
        for _ in range(int(n_no_error)):
            alpha = float(rng.uniform(load_scale_min, load_scale_max))
            ppc_scaled = _scale_pypower_loads(ppc_base, alpha)
            solved = _solve_pypower(ppc_scaled)
            if solved is None:
                continue
            z = compute_measurements_pu(solved).astype(float).tolist()
            rec = dict(
                id=f"ne3p_{rng.integers(1e12)}",
                scenario="no_error",
                z_true=z,
                z_obs=z,
                label=dict(error_type="no_error"),
                op_point=dict(load_scale=alpha),
            )
            f.write(json.dumps(rec) + "\n")

        # 2) three-phase imbalance (OpenDSS → 1ϕ-equivalent z + attach 3ϕ voltages)
        for _ in range(int(n_imbalance)):
            alpha = float(rng.uniform(load_scale_min, load_scale_max))
            fracs = tuple(
                float(x)
                for x in rng.dirichlet([float(dirichlet_alpha)] * 3).tolist()
            )
            applied = _set_loads_scaled_with_bus3_unbalance(
                base_loads, load_scale=alpha, bus3_fracs=fracs
            )
            dss.Text.Command("Solve")

            z_obs, buses, branches = extract_measurement_series()
            if len(z_obs) != 3 * nb + 4 * nl:
                raise RuntimeError(f"Unexpected z length={len(z_obs)} (expected 122)")
            if list(buses) != list(BUS_ORDER):
                raise RuntimeError("Unexpected bus order from OpenDSS extractor.")
            if list(branches) != list(BRANCH_ORDER):
                raise RuntimeError("Unexpected branch order from OpenDSS extractor.")

            three_phase_voltages = extract_three_phase_voltage_measurements()

            # Positive-sequence reference with same total load scaling (for analysis/labeling)
            ppc_scaled = _scale_pypower_loads(ppc_base, alpha)
            solved = _solve_pypower(ppc_scaled)
            if solved is None:
                continue
            z_true = compute_measurements_pu(solved).astype(float).tolist()

            rec = dict(
                id=f"imb3p_{rng.integers(1e12)}",
                scenario="three_phase_imbalance",
                z_true=z_true,
                z_obs=[float(x) for x in z_obs],
                three_phase_voltages=three_phase_voltages,
                label=dict(
                    error_type="three_phase_imbalance",
                    unbalance_bus=3,
                    bus3_load_split=applied,
                ),
                op_point=dict(load_scale=alpha),
            )
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out_sft_imbalance", help="Output directory")
    p.add_argument("--n-imbalance", type=int, default=200)
    p.add_argument("--n-no-error", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--load-scale-min", type=float, default=0.80)
    p.add_argument("--load-scale-max", type=float, default=1.25)
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=3.0,
        help="Larger -> more balanced phase split; smaller -> more extreme imbalance.",
    )
    args = p.parse_args()

    generate_dataset(
        out_dir=args.out,
        n_imbalance=args.n_imbalance,
        n_no_error=args.n_no_error,
        seed=args.seed,
        load_scale_min=args.load_scale_min,
        load_scale_max=args.load_scale_max,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    print(f"Wrote imbalance dataset to: {args.out}")


if __name__ == "__main__":
    main()
