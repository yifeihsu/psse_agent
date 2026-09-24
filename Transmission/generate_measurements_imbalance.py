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
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

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
from three_phase_nlm.measurement_noise import (
    DEFAULT_THREE_PHASE_SIGMA_PU,
    add_scada_noise,
    add_voltage_phasor_noise,
    generated_noise_contract,
    positive_sigma,
    scada_noise_sigma,
    scaled_sensor_sigma,
)
from IEEE_14_OpenDSS.measurement_convention import (  # type: ignore
    SHUNT_CONVENTION_LEGACY,
    SHUNT_CONVENTION_YBUS,
    measurement_convention_payload,
    validate_shunt_convention,
)
from three_phase_model.voltage_bases import (  # type: ignore
    IEEE14_NOMINAL_KV,
    IEEE14_VOLTAGE_BASE_PROFILE_ID,
    ieee14_voltage_base_profile,
)
from three_phase_nlm.hif_operating_point import (  # type: ignore
    DISPATCH_MODE_CASE14,
    DISPATCH_MODE_OPF,
    DISPATCH_MODES,
    OPF_SOLVER,
    OPFDispatchError,
    apply_ieee14_dispatch_and_setpoints,
    capture_operating_point_baseline,
    case14_dispatch_receipt,
    ieee14_opf_operating_point,
    opf_dispatch_receipt,
)

TELEMETRY_BASE_SEMANTICS = ("physical_local_bases", "normalized_model_bases")
#: Keys of a row op_point that carry a dispatch to replay (absent on load-only op points).
DISPATCH_OP_POINT_KEYS = ("generator_dispatch_kw", "voltage_setpoints_pu", "source_voltage_pu")
BALANCED_REFERENCE_OPENDSS = "opendss_same_operating_point"
BALANCED_REFERENCE_PYPOWER_OPF = "pypower_opf"
BALANCED_REFERENCE_MODES = (BALANCED_REFERENCE_OPENDSS, BALANCED_REFERENCE_PYPOWER_OPF)
#: The operator's single voltage-magnitude channel per bus is the phase-A
#: line-to-neutral magnitude exported by OpenDSS (decision 2026-09-21). Under
#: unbalance it departs from the positive-sequence magnitude, which is real
#: single-phase-meter physics, not an export artifact.
OPERATOR_VM_CHANNEL = {
    "channel": "Vm",
    "semantics": "phase_a_line_to_neutral_voltage_magnitude_pu",
    "not": "positive_sequence_or_three_phase_average_magnitude",
    "note": "a single-phase SCADA voltage meter on phase A; unbalance therefore reaches the balanced WLS through Vm",
}


def _bus_number(bus) -> int:
    text = str(bus).strip().lower()
    text = text[1:] if text.startswith("b") else text
    return int(text.split(".")[0])


def _physical_kvbase_ln(bus) -> float:
    """Physical line-to-neutral base kV of an IEEE-14 bus (kV_LL / sqrt(3))."""
    return float(IEEE14_NOMINAL_KV[_bus_number(bus)]) / math.sqrt(3.0)


def _physical_ibase_a(bus) -> float:
    """Per-phase current base (S_base/3) / V_LN,base on the physical bus base."""
    return (100.0 * 1e6 / 3.0) / (_physical_kvbase_ln(bus) * 1e3)


def _rewrite_voltage_bases(rows):
    """Replace ``kvbase_ln`` by the physical local base; pu values untouched."""
    return [{**dict(row), "kvbase_ln": _physical_kvbase_ln(row["bus"])} for row in rows]


def _rewrite_current_bases(rows):
    """Replace ``ibase_from_a``/``ibase_to_a`` by the physical terminal bases; pu untouched."""
    return [{**dict(row), "ibase_from_a": _physical_ibase_a(row["from_bus"]),
             "ibase_to_a": _physical_ibase_a(row["to_bus"])} for row in rows]

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


def _solve_or_raise() -> None:
    """Solve the active circuit; a non-converged power flow is never exported.

    One retry continues the same fixed-point iteration with a larger cap (the
    model converges at tolerance 1e-8 within about 80 iterations), so a
    converged result is unchanged and only a genuinely stuck solve raises.
    """
    dss.Text.Command("Solve")
    if bool(dss.Solution.Converged()):
        return
    first = int(dss.Solution.Iterations())
    dss.Text.Command("Set maxiterations=1000")
    dss.Text.Command("Solve")
    if not bool(dss.Solution.Converged()):
        raise RuntimeError(f"OpenDSS solve did not converge (tolerance {dss.Solution.Convergence():g}; "
                           f"{first} then {int(dss.Solution.Iterations())} iterations)")


FRESH_SOLVE_ATTEMPTS = 3


def _solve_from_fresh_compile(build) -> None:
    """Run ``build`` (compile, edit, solve) again from scratch if its solve diverges.

    The OpenDSS engine occasionally diverges to NaN on a solve whose inputs
    converge on every other run; a converged solve is deterministic, so a fresh
    compile either reproduces it or exhausts the attempts.
    """
    last = None
    for attempt in range(1, FRESH_SOLVE_ATTEMPTS + 1):
        try:
            build()
            return
        except RuntimeError as exc:
            if "did not converge" not in str(exc):
                raise
            last = exc
            print(f"warning: OpenDSS solve diverged on attempt {attempt}/{FRESH_SOLVE_ATTEMPTS}; "
                  f"retrying from a fresh compile ({exc})", file=sys.stderr, flush=True)
    raise RuntimeError(f"OpenDSS solve did not converge in {FRESH_SOLVE_ATTEMPTS} fresh attempts: {last}")


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


def _apply_operating_point_dispatch(op_point: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    """Write the dispatch an op_point carries (unit kW, PV setpoints, source pu) to the active circuit.

    Loads are left as the caller scaled or split them. A load-only op_point
    (the case14 dispatch mode and the pre-2026-09-23opf corpora) is a no-op,
    so replaying such a row keeps the checked-in model dispatch exactly.
    """
    if not isinstance(op_point, Mapping) or not any(key in op_point for key in DISPATCH_OP_POINT_KEYS):
        return None
    return apply_ieee14_dispatch_and_setpoints(capture_operating_point_baseline(), op_point)


def _validated_dispatch_mode(dispatch_mode: str) -> str:
    normalized = str(dispatch_mode).strip().lower()
    if normalized not in DISPATCH_MODES:
        raise ValueError(f"dispatch_mode must be one of {DISPATCH_MODES}, got {dispatch_mode!r}")
    return normalized


def _dispatch_meta(dispatch_mode: str) -> Dict[str, Any]:
    if dispatch_mode == DISPATCH_MODE_OPF:
        return {
            "mode": DISPATCH_MODE_OPF,
            "solver": OPF_SOLVER,
            "load_scaling": "case14 PD/QD at every bus times load_scale (uniform; the same factor every OpenDSS load carries)",
            "generator_dispatch_kw": "OPF active output of the units at buses 2, 3, 6 and 8, applied to the unbalanced "
                                     "solve and to the balanced z_true solve",
            "voltage_setpoints_pu": "OPF voltage magnitude at buses 2, 3, 6 and 8 (PV setpoints)",
            "source_voltage_pu": "OPF voltage magnitude at bus 1; the OpenDSS Vsource reproduces the slack and supplies "
                                 "its active and reactive power",
            "failure_policy": "a window whose AC-OPF does not converge is skipped and listed in generation.skipped_windows; "
                              "the case14 dispatch is never substituted",
            "annotation": "each row carries the dispatch inside op_point (canonical keys) and a `dispatch` block "
                          "(mode, objective, slack output, unit reactive outputs)",
            "healthy_controls": "pypower AC-OPF rows in every dispatch mode",
        }
    return {
        "mode": DISPATCH_MODE_CASE14,
        "solver": None,
        "note": "checked-in IEEE14Gen.DSS dispatch and setpoints (bus 2 at 40 MW, 1 kW condensers at 3/6/8, source 1.06)",
        "healthy_controls": "pypower AC-OPF rows in every dispatch mode",
    }


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
    branch_current_noise_pu: float = DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    branch_current_sigma_pu: float | None = None,
    three_phase_noise_pu: float = DEFAULT_THREE_PHASE_SIGMA_PU,
    noise_scale: float = 1.0,
    shunt_convention: str = SHUNT_CONVENTION_YBUS,
    telemetry_bases: str = "physical_local_bases",
    balanced_reference: str = BALANCED_REFERENCE_OPENDSS,
    dispatch_mode: str = DISPATCH_MODE_OPF,
) -> None:
    # Bus injections follow the operator WLS convention by default (fixed
    # shunts stay in Ybus); the historical corpus used legacy_injection.
    shunt_convention = validate_shunt_convention(shunt_convention)
    dispatch_mode = _validated_dispatch_mode(dispatch_mode)
    if telemetry_bases not in TELEMETRY_BASE_SEMANTICS:
        raise ValueError(f"telemetry_bases must be one of {TELEMETRY_BASE_SEMANTICS}, got {telemetry_bases!r}")
    physical_bases = telemetry_bases == "physical_local_bases"
    if balanced_reference not in BALANCED_REFERENCE_MODES:
        raise ValueError(f"balanced_reference must be one of {BALANCED_REFERENCE_MODES}, got {balanced_reference!r}")
    convention_payload = measurement_convention_payload(shunt_convention)
    noise_scale = positive_sigma(noise_scale, field="noise_scale")
    branch_current_noise_pu = positive_sigma(branch_current_noise_pu, field="branch_current_noise_pu")
    if branch_current_sigma_pu is not None:
        declared = positive_sigma(branch_current_sigma_pu, field="branch_current_sigma_pu")
        if not np.isclose(declared, branch_current_noise_pu, rtol=1e-12, atol=0.0):
            raise ValueError("branch_current_sigma_pu must equal applied branch_current_noise_pu before noise_scale")
    declared_current_sigma = scaled_sensor_sigma(branch_current_noise_pu, noise_scale, field="branch_current_noise_pu")
    voltage_sigma = scaled_sensor_sigma(three_phase_noise_pu, noise_scale, field="three_phase_noise_pu")
    sigma_z = scada_noise_sigma(noise_scale).tolist()
    noise_contract = generated_noise_contract(
        sigma_z, noise_scale=noise_scale, three_phase_sigma=voltage_sigma,
        branch_current_sigma_pu=declared_current_sigma,
    )
    rng = np.random.default_rng(seed)
    measurement_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
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
        sigma_z=sigma_z,
        three_phase_sigma=voltage_sigma,
        branch_current_sigma_pu=declared_current_sigma,
        noise_contract=noise_contract,
        measurement_convention=convention_payload,
        operator_vm_channel=dict(OPERATOR_VM_CHANNEL),
        telemetry_base_semantics=telemetry_bases,
        voltage_base_profile=(ieee14_voltage_base_profile() if physical_bases else None),
        imbalance=dict(
            eligible_load_buses=[],
            shunt_convention=shunt_convention,
            balanced_reference=balanced_reference,
            dispatch_mode=dispatch_mode,
            dispatch=_dispatch_meta(dispatch_mode),
            generation=dict(
                seed=int(seed),
                dispatch_mode=dispatch_mode,
                noise_scale=float(noise_scale),
                three_phase_noise_pu=float(three_phase_noise_pu),
                branch_current_noise_pu=float(branch_current_noise_pu),
                applied_three_phase_sigma=voltage_sigma,
                applied_branch_current_sigma_pu=declared_current_sigma,
                skipped_windows=[],
                skipped_window_count=0,
                skipped_controls=[],
            ),
            z_true_semantics=(
                "balanced OpenDSS solve at the same load scale with every load balanced (bus 3 rebalanced), "
                "same dispatch and shunt convention as the unbalanced solve; z_reference_opf is the pypower OPF vector"
                + (" at the same load scale (the applied dispatch is that OPF's dispatch)" if dispatch_mode == DISPATCH_MODE_OPF
                   else " (different dispatch than the OpenDSS solve)")
                if balanced_reference == BALANCED_REFERENCE_OPENDSS else
                "pypower OPF balanced case"
                + (" (the applied dispatch is that OPF's dispatch)" if dispatch_mode == DISPATCH_MODE_OPF
                   else " (different dispatch than the OpenDSS solve)")),
            z_obs_semantics="phase-A Vm plus three-phase total P/Q injections and flows; unbalanced OpenDSS solve",
            bus_order=BUS_ORDER,
            branch_order=BRANCH_ORDER,
            three_phase_voltage_measurements=dict(
                type="VLN",
                phases=["A", "B", "C"],
                fields=["vln_pu", "ang_deg", "kvbase_ln"],
                three_phase_sigma=voltage_sigma,
                noise_model="independent Gaussian per real/imaginary component",
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
                "applied_noise_sigma_pu": declared_current_sigma,
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
    skipped_windows: List[Dict[str, Any]] = meta["imbalance"]["generation"]["skipped_windows"]
    skipped_controls: List[Dict[str, Any]] = meta["imbalance"]["generation"]["skipped_controls"]

    # --- write samples.jsonl ---
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        # 1) balanced controls (positive-sequence)
        ppc_base = case14()
        for control_idx in range(int(n_no_error)):
            alpha = float(rng.uniform(load_scale_min, load_scale_max))
            ppc_scaled = _scale_pypower_loads(ppc_base, alpha)
            solved = _solve_pypower(ppc_scaled)
            if solved is None:
                skipped_controls.append({"control_index": int(control_idx), "load_scale": alpha,
                                         "reason": "pypower AC-OPF did not converge"})
                print(f"WARNING: skipping healthy control {control_idx} (load_scale={alpha:.6f}): "
                      "pypower AC-OPF did not converge", file=sys.stderr)
                continue
            z = compute_measurements_pu(solved).astype(float).tolist()
            rec = dict(
                id=f"ne3p_{rng.integers(1e12)}",
                scenario="no_error",
                z_true=z,
                z_clean=z,
                z_obs=add_scada_noise(z, measurement_rng, sigma_z),
                sigma_z=sigma_z,
                noise_contract=generated_noise_contract(sigma_z, noise_scale=noise_scale),
                measurement_convention=dict(measurement_convention_payload(SHUNT_CONVENTION_YBUS),
                                            source="pypower balanced reference; makeSbus excludes shunts"),
                label=dict(error_type="no_error"),
                op_point=dict(load_scale=alpha),
                dispatch={**opf_dispatch_receipt(solved),
                          "note": "healthy controls are pypower AC-OPF rows in every dispatch mode"},
            )
            f.write(json.dumps(rec) + "\n")

        # 2) three-phase imbalance (OpenDSS → 1ϕ-equivalent z + attach 3ϕ voltages)
        for window_idx in range(int(n_imbalance)):
            alpha = float(rng.uniform(load_scale_min, load_scale_max))
            fracs = tuple(
                float(x)
                for x in rng.dirichlet([float(dirichlet_alpha)] * 3).tolist()
            )
            target_bus = str(rng.choice(eligible_buses))
            applied: Dict[str, Any] = {}

            # Dispatch law of this window. The label draws above are complete and the
            # row id is drawn after the solves, so a skipped window leaves every other
            # window's labels and ids unchanged.
            dispatch_op_point: Dict[str, Any] | None = None
            if dispatch_mode == DISPATCH_MODE_OPF:
                try:
                    opf = ieee14_opf_operating_point(alpha)
                except OPFDispatchError as exc:
                    skipped_windows.append({"window_index": int(window_idx), "load_scale": alpha,
                                            "target_bus": target_bus, "reason": str(exc)})
                    print(f"WARNING: skipping unbalance window {window_idx}: {exc}", file=sys.stderr)
                    continue
                dispatch_op_point, dispatch_receipt, solved = opf.op_point, opf.receipt, opf.solution
            else:
                dispatch_receipt = case14_dispatch_receipt()
                # pypower OPF balanced case at the same total load (different dispatch; kept for continuity)
                solved = _solve_pypower(_scale_pypower_loads(ppc_base, alpha))
                if solved is None:
                    skipped_windows.append({"window_index": int(window_idx), "load_scale": alpha, "target_bus": target_bus,
                                            "reason": "pypower AC-OPF reference did not converge"})
                    print(f"WARNING: skipping unbalance window {window_idx}: pypower AC-OPF reference did not converge",
                          file=sys.stderr)
                    continue
            z_reference_opf = compute_measurements_pu(solved).astype(float).tolist()

            def build_unbalanced() -> None:
                _compile_ieee14_opendss(dss_repo)
                applied.clear()
                applied.update(_set_loads_scaled_with_bus_unbalance(
                    base_loads,
                    target_bus=target_bus,
                    load_scale=alpha,
                    bus_fracs=fracs,
                ))
                _apply_operating_point_dispatch(dispatch_op_point)
                _solve_or_raise()

            _solve_from_fresh_compile(build_unbalanced)

            z_clean, buses, branches = extract_measurement_series(shunt_convention=shunt_convention)
            if len(z_clean) != 3 * nb + 4 * nl:
                raise RuntimeError(f"Unexpected z length={len(z_clean)} (expected 122)")
            if list(buses) != list(BUS_ORDER):
                raise RuntimeError("Unexpected bus order from OpenDSS extractor.")
            if list(branches) != list(BRANCH_ORDER):
                raise RuntimeError("Unexpected branch order from OpenDSS extractor.")

            three_phase_voltages_clean = extract_three_phase_voltage_measurements()
            three_phase_voltages = add_voltage_phasor_noise(
                three_phase_voltages_clean, measurement_rng, voltage_sigma
            )
            branch_currents_clean = extract_three_phase_branch_current_measurements()
            branch_currents = add_branch_current_noise(
                branch_currents_clean,
                measurement_rng,
                declared_current_sigma,
            )

            if physical_bases:
                three_phase_voltages_clean = _rewrite_voltage_bases(three_phase_voltages_clean)
                three_phase_voltages = _rewrite_voltage_bases(three_phase_voltages)
                branch_currents_clean = _rewrite_current_bases(branch_currents_clean)
                branch_currents = _rewrite_current_bases(branch_currents)

            if balanced_reference == BALANCED_REFERENCE_OPENDSS:
                # Paired balanced reference: the same OpenDSS model, dispatch, load scale and
                # shunt convention with every load balanced (the unbalanced exports above are
                # complete, so recompiling here is safe).
                def build_balanced() -> None:
                    _compile_ieee14_opendss(dss_repo)
                    _scale_all_loads(base_loads, alpha)
                    _apply_operating_point_dispatch(dispatch_op_point)
                    _solve_or_raise()

                _solve_from_fresh_compile(build_balanced)
                z_balanced, _, _ = extract_measurement_series(shunt_convention=shunt_convention)
                z_true = [float(x) for x in z_balanced]
                z_true_semantics = ("balanced_same_operating_point_opendss_reference; every load balanced, same "
                                    "dispatch, load scale and shunt convention; z_clean is the unbalanced sensor mean")
            else:
                z_true = z_reference_opf
                z_true_semantics = "balanced_reference_pypower_opf; different dispatch than the OpenDSS solve"

            rec = dict(
                id=f"imb3p_{rng.integers(1e12)}",
                scenario="three_phase_imbalance",
                z_true=z_true,
                z_true_semantics=z_true_semantics,
                z_reference_opf=z_reference_opf,
                balanced_reference=balanced_reference,
                z_clean=[float(x) for x in z_clean],
                z_obs=add_scada_noise(z_clean, measurement_rng, sigma_z),
                sigma_z=sigma_z,
                three_phase_voltages=three_phase_voltages,
                three_phase_voltages_clean=three_phase_voltages_clean,
                three_phase_sigma=voltage_sigma,
                noise_contract=noise_contract,
                measurement_convention=convention_payload,
                **{
                    BRANCH_CURRENT_CHANNEL: branch_currents,
                    f"{BRANCH_CURRENT_CHANNEL}_clean": branch_currents_clean,
                    BRANCH_CURRENT_SIGMA_KEY: float(declared_current_sigma),
                },
                label=dict(
                    error_type="three_phase_imbalance",
                    unbalance_bus=int(target_bus[1:]),
                    unbalance_bus_name=target_bus,
                    load_split=applied,
                ),
                # load_scale and target_bus first (legacy readers); in opf mode the canonical
                # dispatch keys follow so every replay reproduces the applied dispatch.
                op_point=dict(load_scale=alpha, target_bus=target_bus,
                              **{k: v for k, v in (dispatch_op_point or {}).items() if k != "load_scale"}),
                dispatch=deepcopy(dispatch_receipt),
            )
            f.write(json.dumps(rec) + "\n")

    meta["imbalance"]["generation"]["skipped_window_count"] = len(skipped_windows)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if skipped_windows or skipped_controls:
        print(f"WARNING: {len(skipped_windows)} unbalance windows and {len(skipped_controls)} healthy controls were "
              f"skipped; see meta.json imbalance.generation.skipped_windows / skipped_controls", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out_sft_imbalance", help="Output directory")
    p.add_argument("--n-imbalance", type=int, default=200)
    p.add_argument("--n-no-error", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--load-scale-min", type=float, default=0.80)
    p.add_argument("--load-scale-max", type=float, default=1.25)
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="Positive common multiplier for applied noise and exported sensor sigmas.")
    p.add_argument("--three-phase-noise-pu", type=float, default=DEFAULT_THREE_PHASE_SIGMA_PU,
                   help="Phase-voltage real/imaginary component sigma before --noise-scale.")
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=3.0,
        help="Larger -> more balanced phase split; smaller -> more extreme imbalance.",
    )
    p.add_argument(
        "--branch-current-noise-pu",
        type=float,
        default=DEFAULT_BRANCH_CURRENT_SIGMA_PU,
        help="Positive branch-current component sigma before --noise-scale; exported weights use the same value.",
    )
    p.add_argument(
        "--branch-current-sigma-pu",
        type=float,
        default=None,
        help="Optional consistency assertion: must equal --branch-current-noise-pu before --noise-scale.",
    )
    p.add_argument(
        "--shunt-convention",
        choices=[SHUNT_CONVENTION_YBUS, SHUNT_CONVENTION_LEGACY],
        default=SHUNT_CONVENTION_YBUS,
        help="Bus-injection convention of the exported z: ybus keeps fixed shunts in Ybus (operator WLS); "
             "legacy_injection reproduces the historical corpora that counted the bus-9 capacitor in Qinj.",
    )
    p.add_argument(
        "--balanced-reference",
        choices=list(BALANCED_REFERENCE_MODES),
        default=BALANCED_REFERENCE_OPENDSS,
        help="Row-level z_true: balanced OpenDSS solve at the same operating point (default) or the pypower OPF case.",
    )
    p.add_argument(
        "--telemetry-bases",
        choices=list(TELEMETRY_BASE_SEMANTICS),
        default="physical_local_bases",
        help="Report kvbase_ln/ibase_*_a on the declared 69/13.8/18 kV bases or on the normalized 1 kV model.",
    )
    p.add_argument(
        "--dispatch-mode",
        choices=list(DISPATCH_MODES),
        default=DISPATCH_MODE_OPF,
        help="opf (default): generator dispatch, PV setpoints and source voltage of every window come from the pypower "
             "AC-OPF on case14 at the window's load scale (the dispatch law of the pypower scenario families); "
             "case14: the checked-in model dispatch. A non-converged OPF skips the window and is recorded in meta.json.",
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
        branch_current_sigma_pu=args.branch_current_sigma_pu,
        three_phase_noise_pu=float(args.three_phase_noise_pu),
        noise_scale=float(args.noise_scale),
        shunt_convention=args.shunt_convention,
        telemetry_bases=args.telemetry_bases,
        balanced_reference=args.balanced_reference,
        dispatch_mode=args.dispatch_mode,
    )
    print(f"Wrote imbalance dataset to: {args.out} [shunt_convention={args.shunt_convention} "
          f"telemetry_bases={args.telemetry_bases} operator_vm=phase_a_magnitude dispatch_mode={args.dispatch_mode} "
          f"three_phase_noise_pu={args.three_phase_noise_pu:g} branch_current_noise_pu={args.branch_current_noise_pu:g}]")


if __name__ == "__main__":
    main()
