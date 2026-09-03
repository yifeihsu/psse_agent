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
  with the imbalanced load sampled from any eligible load bus. We also attach per-bus
  3ϕ VLN voltage measurements.
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
    extract_three_phase_branch_current_measurements,
    extract_three_phase_voltage_measurements,
)
from three_phase_nlm.branch_current_analysis import (  # type: ignore
    BRANCH_CURRENT_CHANNEL,
    BRANCH_CURRENT_SIGMA_KEY,
    DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    add_branch_current_noise,
)

from Transmission.generate_measurements import (  # type: ignore
    MEASUREMENT_ORDER,
    compute_measurements_pu,
    make_index_map,
)

#: The checked-in IEEE14Loads.DSS splits Bus 3 unevenly (B3A/B3B/B3C) for the
#: original single-bus unbalance study.  Every generated sample must start from
#: a balanced base so the labeled target bus is the *only* unbalance source.
BALANCED_BUS3_LOAD_NAME = "__BAL_B3"
BALANCED_BUS3_SPLIT_LOADS = ("B3A", "B3B", "B3C")


def _scale_pypower_loads(ppc: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    ppc2 = deepcopy(ppc)
    ppc2["bus"][:, PD] *= float(alpha)
    ppc2["bus"][:, QD] *= float(alpha)
    return ppc2


def _solve_pypower(ppc: Dict[str, Any]) -> Dict[str, Any] | None:
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    res = runopf(ppc, ppopt)
    return res if res.get("success") else None


def _balance_bus3_loads() -> None:
    """Replace the checked-in Bus 3 split loads with one balanced load."""
    existing = {str(name).lower() for name in (dss.Loads.AllNames() or [])}
    for name in BALANCED_BUS3_SPLIT_LOADS:
        if name.lower() in existing:
            dss.Text.Command(f"Edit Load.{name} enabled=no")
    if BALANCED_BUS3_LOAD_NAME.lower() not in existing:
        dss.Text.Command(
            f"New Load.{BALANCED_BUS3_LOAD_NAME} Bus1=B3 kV=1 kW=94200 kvar=19000 "
            "vmaxpu=1.06 vminpu=0.94"
        )


def _compile_ieee14_opendss(repo_dir: str) -> None:
    caller_cwd = os.getcwd()
    try:
        dss.Basic.DataPath(repo_dir)
        dss.Text.Command("Clear")
        dss.Text.Command("Redirect Run_IEEE14Bus.dss")
    finally:
        os.chdir(caller_cwd)
    _balance_bus3_loads()


def _normalize_bus_name(bus_ref: str) -> str:
    return str(bus_ref).split(".")[0].lower()


def _read_base_loads() -> Dict[str, Dict[str, Any]]:
    base: Dict[str, Dict[str, Any]] = {}
    for name in dss.Loads.AllNames() or []:
        dss.Loads.Name(name)
        if hasattr(dss.CktElement, "Enabled") and not bool(dss.CktElement.Enabled()):
            continue
        bus_ref = str((dss.CktElement.BusNames() or [""])[0])
        base[str(name).lower()] = {
            "name": str(name),
            "bus_ref": bus_ref,
            "bus_name": _normalize_bus_name(bus_ref),
            "kW": float(dss.Loads.kW()),
            "kvar": float(dss.Loads.kvar()),
            "phases": int(dss.Loads.Phases()),
        }
    return base


def _group_loads_by_bus(base_loads: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for load_key, info in base_loads.items():
        grouped.setdefault(str(info["bus_name"]), []).append(load_key)
    return {bus: sorted(loads) for bus, loads in grouped.items()}


def _eligible_imbalance_buses(base_loads: Dict[str, Dict[str, Any]]) -> List[str]:
    eligible: List[str] = []
    for bus_name, load_keys in _group_loads_by_bus(base_loads).items():
        infos = [base_loads[key] for key in load_keys]
        if len(infos) == 1:
            eligible.append(bus_name)
            continue
        phase_suffixes = {
            str(info["bus_ref"]).split(".", 1)[1]
            for info in infos
            if "." in str(info["bus_ref"])
        }
        if all(int(info["phases"]) == 1 for info in infos) and phase_suffixes == {"1", "2", "3"}:
            eligible.append(bus_name)
    return sorted(
        eligible,
        key=lambda name: int(name[1:]) if name.startswith("b") and name[1:].isdigit() else name,
    )


def _bus_kvbase_ln(bus_name: str) -> float:
    dss.Circuit.SetActiveBus(bus_name.upper())
    kvbase_ln = float(dss.Bus.kVBase() or 0.0)
    if kvbase_ln <= 0:
        raise RuntimeError(f"OpenDSS did not report a valid LN base kV for bus {bus_name}.")
    return kvbase_ln


def _phase_split(a: float, b: float, c: float) -> Dict[str, float]:
    return {"a": float(a), "b": float(b), "c": float(c)}


def _scale_all_loads(base_loads: Dict[str, Dict[str, Any]], load_scale: float) -> None:
    for info in base_loads.values():
        dss.Loads.Name(str(info["name"]))
        dss.Loads.kW(float(info["kW"]) * float(load_scale))
        dss.Loads.kvar(float(info["kvar"]) * float(load_scale))


def _set_loads_scaled_with_bus_unbalance(
    base_loads: Dict[str, Dict[str, Any]],
    *,
    target_bus: str,
    load_scale: float,
    bus_fracs: Tuple[float, float, float],
) -> Dict[str, Any]:
    """
    Apply load scaling and inject an unbalanced three-phase load split at the target load bus.

    Returns a dict of the actually applied per-phase P/Q (kW/kvar) for labeling.
    """
    a, b, c = [float(x) for x in bus_fracs]
    s = a + b + c
    if s <= 0:
        raise ValueError("bus_fracs must sum to a positive value")
    a, b, c = a / s, b / s, c / s
    target_bus = str(target_bus).lower()
    grouped = _group_loads_by_bus(base_loads)
    if target_bus not in grouped:
        raise RuntimeError(f"Target bus {target_bus} is not an eligible load bus.")

    _scale_all_loads(base_loads, load_scale)

    load_keys = grouped[target_bus]
    load_infos = [base_loads[key] for key in load_keys]
    p_tot = sum(float(info["kW"]) for info in load_infos) * float(load_scale)
    q_tot = sum(float(info["kvar"]) for info in load_infos) * float(load_scale)
    fractions = _phase_split(a, b, c)
    phase_order = (("1", "a"), ("2", "b"), ("3", "c"))
    phase_payload: Dict[str, Dict[str, Any]] = {}

    if len(load_infos) == 1:
        original = load_infos[0]
        dss.Text.Command(f"Edit Load.{original['name']} enabled=no")
        kvbase_ln = _bus_kvbase_ln(target_bus)
        for phase_suffix, phase_name in phase_order:
            frac = fractions[phase_name]
            load_name = f"__imb_{target_bus.upper()}_{phase_name.upper()}"
            kw = float(p_tot) * float(frac)
            kvar = float(q_tot) * float(frac)
            dss.Text.Command(
                f"New Load.{load_name} Phases=1 Bus1={target_bus.upper()}.{phase_suffix} "
                f"kV={kvbase_ln:.12g} kW={kw:.12g} kvar={kvar:.12g} vmaxpu=1.06 vminpu=0.94"
            )
            phase_payload[phase_name] = {
                "load_name": load_name,
                "kW": kw,
                "kvar": kvar,
                "frac": float(frac),
            }
        return {
            "bus": target_bus,
            "source_mode": "split_balanced_load",
            "original_load_names": [str(original["name"])],
            "fractions": fractions,
            "total": {"kW": float(p_tot), "kvar": float(q_tot)},
            "phases": phase_payload,
        }

    if len(load_infos) == 3 and all(int(info["phases"]) == 1 for info in load_infos):
        phase_lookup = {
            str(info["bus_ref"]).split(".", 1)[1]: info
            for info in load_infos
            if "." in str(info["bus_ref"])
        }
        if set(phase_lookup) != {"1", "2", "3"}:
            raise RuntimeError(f"Unexpected single-phase load layout at bus {target_bus}: {sorted(phase_lookup)}")
        for phase_suffix, phase_name in phase_order:
            info = phase_lookup[phase_suffix]
            frac = fractions[phase_name]
            kw = float(p_tot) * float(frac)
            kvar = float(q_tot) * float(frac)
            dss.Loads.Name(str(info["name"]))
            dss.Loads.kW(kw)
            dss.Loads.kvar(kvar)
            phase_payload[phase_name] = {
                "load_name": str(info["name"]),
                "kW": kw,
                "kvar": kvar,
                "frac": float(frac),
            }
        return {
            "bus": target_bus,
            "source_mode": "rescale_existing_single_phase_loads",
            "original_load_names": [str(info["name"]) for info in load_infos],
            "fractions": fractions,
            "total": {"kW": float(p_tot), "kvar": float(q_tot)},
            "phases": phase_payload,
        }

    raise RuntimeError(
        f"Cannot construct a three-phase imbalance at bus {target_bus}: unsupported load layout with {len(load_infos)} loads."
    )


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
    branch_current_noise_pu: float = 0.0,
    branch_current_sigma_pu: float = DEFAULT_BRANCH_CURRENT_SIGMA_PU,
) -> None:
    if float(branch_current_noise_pu) < 0.0:
        raise ValueError("branch_current_noise_pu must be non-negative")
    if float(branch_current_sigma_pu) <= 0.0:
        raise ValueError("branch_current_sigma_pu must be positive")
    # The declared sigma is the nominal sensor accuracy used to weight the
    # channel; applied noise may be zero (clean telemetry) or larger.
    declared_current_sigma = max(float(branch_current_sigma_pu), float(branch_current_noise_pu))
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
        measurement_order=MEASUREMENT_ORDER,
        branch_info=_branch_info_case14(),
        imbalance=dict(
            eligible_load_buses=[],
            bus_order=BUS_ORDER,
            branch_order=BRANCH_ORDER,
            three_phase_voltage_measurements=dict(
                type="VLN",
                phases=["A", "B", "C"],
                fields=["vln_pu", "ang_deg", "kvbase_ln"],
            ),
            three_phase_branch_current_measurements={
                "channel": BRANCH_CURRENT_CHANNEL,
                "type": "per_phase_terminal_current_phasors",
                "phases": ["A", "B", "C"],
                "fields": [
                    "branch",
                    "branch_row0",
                    "from_bus",
                    "to_bus",
                    "i_from_pu",
                    "ang_from_deg",
                    "i_to_pu",
                    "ang_to_deg",
                    "ibase_from_a",
                    "ibase_to_a",
                ],
                "sign_convention": "current flowing into the branch from each terminal",
                "per_unit_base": "(S_base/3) / V_LN,base at the terminal bus, S_base=100 MVA",
                "applied_noise_sigma_pu": float(branch_current_noise_pu),
                BRANCH_CURRENT_SIGMA_KEY: float(declared_current_sigma),
            },
            base_model_override={
                "balanced_bus3": True,
                "disabled_loads": list(BALANCED_BUS3_SPLIT_LOADS),
                "balanced_load": BALANCED_BUS3_LOAD_NAME,
                "note": (
                    "The checked-in OpenDSS load file splits Bus 3 unevenly; every "
                    "sample rebalances it so the labeled bus is the only unbalance source."
                ),
            },
        ),
    )
    # --- OpenDSS init (compile once) ---
    dss_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "IEEE_14_OpenDSS"))
    _compile_ieee14_opendss(dss_repo)
    base_loads = _read_base_loads()
    eligible_buses = _eligible_imbalance_buses(base_loads)
    if not eligible_buses:
        raise RuntimeError("No eligible load buses were found for imbalance generation.")
    meta["imbalance"]["eligible_load_buses"] = eligible_buses
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
            target_bus = str(rng.choice(eligible_buses))
            _compile_ieee14_opendss(dss_repo)
            applied = _set_loads_scaled_with_bus_unbalance(
                base_loads,
                target_bus=target_bus,
                load_scale=alpha,
                bus_fracs=fracs,
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
            branch_currents = add_branch_current_noise(
                extract_three_phase_branch_current_measurements(),
                rng,
                float(branch_current_noise_pu),
            )

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
                **{
                    BRANCH_CURRENT_CHANNEL: branch_currents,
                    BRANCH_CURRENT_SIGMA_KEY: float(declared_current_sigma),
                },
                label=dict(
                    error_type="three_phase_imbalance",
                    unbalance_bus=int(target_bus[1:]),
                    unbalance_bus_name=target_bus,
                    load_split=applied,
                ),
                op_point=dict(load_scale=alpha, target_bus=target_bus),
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
    p.add_argument(
        "--branch-current-noise-pu",
        type=float,
        default=0.0,
        help="Applied per-component Gaussian noise on branch-current phasors (pu); 0 keeps them clean.",
    )
    p.add_argument(
        "--branch-current-sigma-pu",
        type=float,
        default=DEFAULT_BRANCH_CURRENT_SIGMA_PU,
        help="Declared nominal branch-current sensor sigma (pu) used to weight the channel.",
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
        branch_current_noise_pu=float(args.branch_current_noise_pu),
        branch_current_sigma_pu=float(args.branch_current_sigma_pu),
    )
    print(f"Wrote imbalance dataset to: {args.out}")


if __name__ == "__main__":
    main()
