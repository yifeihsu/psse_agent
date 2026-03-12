#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate measurement data for SFT/RFT.

This revision aligns the dataset semantics with the revised SFT trace builder:
  - Clean samples are emitted as scenario='no_error' (not 'negative').
  - Measurement-error samples only use the subtypes actually modeled by the builder:
      {single_gross_outlier, multi_gross_outliers}.
  - Parameter-error samples now persist the erroneous MATPOWER case as a text `.m` file so
    the parameter-correction tool can operate on the same erroneous model that generated the data.
  - Topology-error verification models are persisted as text `.m` files that are compatible with
    the MCP server's regex MATPOWER parser.
  - Harmonic-anomaly samples now include `harmonic_orders`, matching the HSE tool contract.

Scenarios emitted by this script:
  - no_error
  - measurement_error
  - parameter_error
  - topology_error   (IEEE-14 only)
  - harmonic_anomaly (IEEE-14 only)

This script does NOT synthesize three_phase_imbalance yet because the current stack is based on
single-phase PYPOWER/MATPOWER style cases. It is better to add that family from a dedicated 3φ
simulation source than to fabricate weak labels.

Outputs:
  - samples.jsonl
  - meta.json
  - cases_parameter_error/*.m
  - models_topology/*.m
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from tqdm import tqdm

# Add project root to path for Harmonics and scripts imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# PYPOWER API: cases, solver, options, admittance builder, indices
from pypower.api import case14, case118, runopf, ppoption, makeYbus
from pypower.idx_bus import VM, VA, PD, QD, BUS_I
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, BR_STATUS

# Harmonics utilities (for harmonic_anomaly)
try:
    from Transmission.generate_hse_traces import build_trace
    HARMONICS_AVAILABLE = True
except ImportError as e:
    HARMONICS_AVAILABLE = False
    print(f"[warn] Harmonics modules not found ({e}). Harmonic anomaly subset will be skipped.")


MEASUREMENT_ORDER = ["Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt"]
DEFAULT_SIGMAS = {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2}
HARMONIC_DEFAULT_CANDIDATES = [2, 3, 4, 5, 9, 10, 11, 12, 13, 14]


# ------------------------ configuration ------------------------

DEFAULTS = dict(
    case_name="14",            # "14" or "118"
    n_total=2000,              # total scenarios counted over measurement_error + parameter_error + topology_error
    frac_param_err=0.50,       # fraction parameter-error within n_total (topology excluded)
    n_negative=500,            # clean no-error scenarios
    seed=42,                   # reproducibility
    sigma_vm=DEFAULT_SIGMAS["vm"],
    sigma_inj=DEFAULT_SIGMAS["inj"],
    sigma_flow=DEFAULT_SIGMAS["flow"],
    load_scale_min=0.80,
    load_scale_max=1.25,
    r_err_range=[(0.1, 0.5), (2.0, 5.0)],
    x_err_range=[(0.1, 0.5), (2.0, 5.0)],
    max_attempt_multiplier=10,
)


# ------------------------ helpers: case, power flow, measurements ------------------------


def load_case(case_name: str):
    if case_name == "14":
        return case14()
    if case_name == "118":
        return case118()
    raise ValueError("case_name must be '14' or '118'")


def scale_loads(ppc, alpha_p: float, alpha_q: float | None = None):
    """Scale Pd/Qd to diversify operating points."""
    if alpha_q is None:
        alpha_q = alpha_p
    ppc = deepcopy(ppc)
    ppc["bus"][:, PD] *= alpha_p
    ppc["bus"][:, QD] *= alpha_q
    return ppc


def solve_ac_opf(ppc):
    """Run AC optimal power flow with quiet options; return solved ppc or None if it fails."""
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    results = runopf(deepcopy(ppc), ppopt)
    return results if results.get("success") else None


def compute_measurements_pu(ppc_solved):
    """
    Build z in per-unit using solved voltages and admittance matrices:
      z = [ Vm; Pinj; Qinj; Pf; Qf; Pt; Qt ]

    Robust to PYPOWER/MATPOWER-style 1-based bus indexing and pandapower-to-ppc output that is already 0-based.
    """
    baseMVA = ppc_solved["baseMVA"]
    bus = ppc_solved["bus"]
    branch = ppc_solved["branch"]
    nb = bus.shape[0]

    Vm = bus[:, VM]
    Va = np.deg2rad(bus[:, VA])
    V = Vm * np.exp(1j * Va)

    bus_int = bus.copy()
    bus_int[:, BUS_I] = np.arange(nb)
    branch_int = branch.copy()
    min_f = branch_int[:, F_BUS].min() if branch_int.shape[0] else 0
    min_t = branch_int[:, T_BUS].min() if branch_int.shape[0] else 0
    if min_f >= 1 and min_t >= 1:
        branch_int[:, F_BUS] = branch_int[:, F_BUS].astype(int) - 1
        branch_int[:, T_BUS] = branch_int[:, T_BUS].astype(int) - 1
    else:
        branch_int[:, F_BUS] = branch_int[:, F_BUS].astype(int)
        branch_int[:, T_BUS] = branch_int[:, T_BUS].astype(int)

    Ybus, Yf, Yt = makeYbus(baseMVA, bus_int, branch_int)

    Ibus = Ybus.dot(V)
    Sinj = V * np.conj(Ibus)
    Pinj = Sinj.real
    Qinj = Sinj.imag

    If = Yf.dot(V)
    It = Yt.dot(V)
    if min_f >= 1 and min_t >= 1:
        fbus = branch[:, F_BUS].astype(int) - 1
        tbus = branch[:, T_BUS].astype(int) - 1
    else:
        fbus = branch[:, F_BUS].astype(int)
        tbus = branch[:, T_BUS].astype(int)

    Sf = V[fbus] * np.conj(If)
    St = V[tbus] * np.conj(It)
    Pf, Qf = Sf.real, Sf.imag
    Pt, Qt = St.real, St.imag

    return np.concatenate([Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]).astype(float)


def make_index_map(nb: int, nl: int):
    return dict(
        Vm=slice(0, nb),
        Pinj=slice(nb, 2 * nb),
        Qinj=slice(2 * nb, 3 * nb),
        Pf=slice(3 * nb, 3 * nb + nl),
        Qf=slice(3 * nb + nl, 3 * nb + 2 * nl),
        Pt=slice(3 * nb + 2 * nl, 3 * nb + 3 * nl),
        Qt=slice(3 * nb + 3 * nl, 3 * nb + 4 * nl),
    )


def branch_line_mask(ppc):
    """Return boolean mask for in-service transmission lines (transformers excluded)."""
    br = ppc["branch"]
    in_service = br[:, BR_STATUS] > 0
    is_line = br[:, TAP] == 0.0
    return in_service & is_line


# ------------------------ helpers: MATPOWER text export ------------------------


def _safe_case_name(name: str) -> str:
    safe = "".join(ch for ch in str(name) if ch.isalnum() or ch == "_")
    return safe or "case_tmp"


def _matrix_to_matpower_text(mat: np.ndarray) -> str:
    rows = []
    for row in np.asarray(mat, dtype=float):
        vals = " ".join(f"{float(v):.12g}" for v in row)
        rows.append(f"    {vals};")
    return "\n".join(rows)


def write_ppc_as_matpower_m(ppc: Dict[str, Any], path: Path, case_name: str) -> Path:
    """Write a minimal MATPOWER case.m text file compatible with the MCP server regex parser."""
    safe_name = _safe_case_name(case_name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".m":
        path = path.with_suffix(".m")

    with path.open("w", encoding="utf-8") as f:
        f.write(f"function mpc = {safe_name}\n")
        f.write("mpc.version = '2';\n")
        f.write(f"mpc.baseMVA = {float(ppc['baseMVA']):.12g};\n\n")
        for key in ("bus", "gen", "branch"):
            arr = np.asarray(ppc[key], dtype=float)
            f.write(f"mpc.{key} = [\n")
            f.write(_matrix_to_matpower_text(arr))
            f.write("\n];\n\n")
    return path.resolve()


# ------------------------ helpers: node-breaker pocket + mapping ------------------------


def _extract_busnum_from_name(name: str) -> int | None:
    s = str(name)
    if s.startswith("B"):
        digits = "".join(ch for ch in s[1:] if ch.isdigit())
        return int(digits) if digits else None
    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def _choose_random_cb_open(rng, target_bus: int | None = None):
    """Choose exactly one CLOSED CB in pocket {1,2,3} and OPEN it."""
    try:
        from . import nodebreaker_pp14 as nb  # type: ignore
    except Exception:
        import sys as _sys
        import os as _os
        _sys.path.append(_os.path.dirname(__file__))
        import nodebreaker_pp14 as nb  # type: ignore

    pocket = nb.detailed_substations_1_2_3()
    buses = [1, 2, 3] if target_bus is None else [int(target_bus)]
    candidates = []
    for b in buses:
        for cb in pocket[b].cbs:
            if bool(cb.closed):
                candidates.append((b, cb))
    if not candidates and target_bus is not None:
        for b in [1, 2, 3]:
            for cb in pocket[b].cbs:
                if bool(cb.closed):
                    candidates.append((b, cb))
    if not candidates:
        raise RuntimeError("No CLOSED CB available to open in pocket substations 1–3.")

    b, cb = rng.choice(candidates)
    status_map = {cb.name: False}
    label = dict(
        error_type="topology_error",
        substation=int(b),
        cb_name=str(cb.name),
        old_status="closed" if cb.closed else "open",
        new_status="open",
    )
    return status_map, label


def _nb_to_operator_z(net, line_idx: dict, trafo_idx: dict, ppc_base) -> np.ndarray:
    """Aggregate node-breaker PF results to operator z order."""
    import numpy as _np

    baseMVA = float(ppc_base["baseMVA"])
    bus = ppc_base["bus"]
    branch = ppc_base["branch"]
    nb = bus.shape[0]
    nl = branch.shape[0]

    groups: dict[int, list[int]] = {i + 1: [] for i in range(nb)}
    for bidx, row in net.bus.iterrows():
        name = str(row.get("name", f"bus{bidx}"))
        bnum = _extract_busnum_from_name(name)
        if bnum in groups:
            groups[bnum].append(int(bidx))

    Vm = _np.zeros(nb, dtype=float)
    for i in range(1, nb + 1):
        idxs = groups.get(i, [])
        if not idxs:
            Vm[i - 1] = _np.nan
            continue
        vals = []
        for j in idxs:
            try:
                if bool(net.bus.at[j, "in_service"]):
                    v = float(net.res_bus.at[j, "vm_pu"])
                    if _np.isfinite(v):
                        vals.append(v)
            except Exception:
                pass
        Vm[i - 1] = float(_np.mean(vals)) if vals else _np.nan

    Pinj = _np.zeros(nb, dtype=float)
    Qinj = _np.zeros(nb, dtype=float)
    for i in range(1, nb + 1):
        idxs = groups.get(i, [])
        p = 0.0
        q = 0.0
        for j in idxs:
            try:
                if bool(net.bus.at[j, "in_service"]):
                    pv = float(net.res_bus.at[j, "p_mw"])
                    qv = float(net.res_bus.at[j, "q_mvar"])
                    if _np.isfinite(pv):
                        p += pv
                    if _np.isfinite(qv):
                        q += qv
            except Exception:
                pass

            if hasattr(net, "shunt") and len(net.shunt) > 0:
                try:
                    shunts_at_j = net.shunt[net.shunt.bus == j].index
                    for s_idx in shunts_at_j:
                        if bool(net.shunt.at[s_idx, "in_service"]):
                            pv = float(net.res_shunt.at[s_idx, "p_mw"])
                            qv = float(net.res_shunt.at[s_idx, "q_mvar"])
                            if _np.isfinite(pv):
                                p -= pv
                            if _np.isfinite(qv):
                                q -= qv
                except Exception:
                    pass
        Pinj[i - 1] = -p / baseMVA
        Qinj[i - 1] = -q / baseMVA

    Pf = _np.zeros(nl, dtype=float)
    Qf = _np.zeros(nl, dtype=float)
    Pt = _np.zeros(nl, dtype=float)
    Qt = _np.zeros(nl, dtype=float)

    for k in range(nl):
        fb = int(branch[k, F_BUS])
        tb = int(branch[k, T_BUS])
        name_l = f"line_{fb}-{tb}"
        name_t = f"trafo_{fb}-{tb}"
        name_l_rev = f"line_{tb}-{fb}"
        name_t_rev = f"trafo_{tb}-{fb}"

        if name_l in line_idx:
            idx = int(line_idx[name_l])
            Pf[k] = float(net.res_line.at[idx, "p_from_mw"]) / baseMVA
            Qf[k] = float(net.res_line.at[idx, "q_from_mvar"]) / baseMVA
            Pt[k] = float(net.res_line.at[idx, "p_to_mw"]) / baseMVA
            Qt[k] = float(net.res_line.at[idx, "q_to_mvar"]) / baseMVA
        elif name_t in trafo_idx:
            idx = int(trafo_idx[name_t])
            Pf[k] = float(net.res_trafo.at[idx, "p_hv_mw"]) / baseMVA
            Qf[k] = float(net.res_trafo.at[idx, "q_hv_mvar"]) / baseMVA
            Pt[k] = float(net.res_trafo.at[idx, "p_lv_mw"]) / baseMVA
            Qt[k] = float(net.res_trafo.at[idx, "q_lv_mvar"]) / baseMVA
        elif name_l_rev in line_idx:
            idx = int(line_idx[name_l_rev])
            Pf[k] = float(net.res_line.at[idx, "p_to_mw"]) / baseMVA
            Qf[k] = float(net.res_line.at[idx, "q_to_mvar"]) / baseMVA
            Pt[k] = float(net.res_line.at[idx, "p_from_mw"]) / baseMVA
            Qt[k] = float(net.res_line.at[idx, "q_from_mvar"]) / baseMVA
        elif name_t_rev in trafo_idx:
            idx = int(trafo_idx[name_t_rev])
            Pf[k] = float(net.res_trafo.at[idx, "p_lv_mw"]) / baseMVA
            Qf[k] = float(net.res_trafo.at[idx, "q_lv_mvar"]) / baseMVA
            Pt[k] = float(net.res_trafo.at[idx, "p_hv_mw"]) / baseMVA
            Qt[k] = float(net.res_trafo.at[idx, "q_hv_mvar"]) / baseMVA

    return _np.concatenate([Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]).astype(float)


# ------------------------ helpers: errors ------------------------


def base_gaussian_noise(z, idx_map, sigmas, rng=None):
    """Small zero-mean Gaussian noise consistent with the W matrix."""
    nb = idx_map["Vm"].stop - idx_map["Vm"].start
    nl = idx_map["Pf"].stop - idx_map["Pf"].start

    noise = np.zeros_like(z)
    rnorm = np.random.randn if rng is None else rng.standard_normal
    noise[idx_map["Vm"]] = sigmas["vm"] * rnorm(nb)
    noise[idx_map["Pinj"]] = sigmas["inj"] * rnorm(nb)
    noise[idx_map["Qinj"]] = sigmas["inj"] * rnorm(nb)
    noise[idx_map["Pf"]] = sigmas["flow"] * rnorm(nl)
    noise[idx_map["Qf"]] = sigmas["flow"] * rnorm(nl)
    noise[idx_map["Pt"]] = sigmas["flow"] * rnorm(nl)
    noise[idx_map["Qt"]] = sigmas["flow"] * rnorm(nl)
    return noise


def sigma_vector(idx_map, sigmas=DEFAULT_SIGMAS) -> np.ndarray:
    n = idx_map["Qt"].stop
    out = np.zeros(n, dtype=float)
    out[idx_map["Vm"]] = sigmas["vm"]
    out[idx_map["Pinj"]] = sigmas["inj"]
    out[idx_map["Qinj"]] = sigmas["inj"]
    out[idx_map["Pf"]] = sigmas["flow"]
    out[idx_map["Qf"]] = sigmas["flow"]
    out[idx_map["Pt"]] = sigmas["flow"]
    out[idx_map["Qt"]] = sigmas["flow"]
    return out


def apply_measurement_error(z_true, idx_map, rng):
    """
    Add base Gaussian noise plus one additional measurement-error subtype:
      - single_gross_outlier
      - multi_gross_outliers

    The unreachable channel-wide bias/scale branches from the old script are intentionally removed so
    the emitted scenario semantics match the trace builder and label space exactly.
    """
    sigmas = DEFAULT_SIGMAS
    z_base = z_true + base_gaussian_noise(z_true, idx_map, sigmas, rng)
    z_obs = z_base.copy()

    subtype = rng.choice(["single_gross_outlier", "multi_gross_outliers"])
    ch = rng.choice(MEASUREMENT_ORDER)
    sl = idx_map[ch]
    ch_sigma = {
        "Vm": sigmas["vm"],
        "Pinj": sigmas["inj"],
        "Qinj": sigmas["inj"],
        "Pf": sigmas["flow"],
        "Qf": sigmas["flow"],
        "Pt": sigmas["flow"],
        "Qt": sigmas["flow"],
    }[ch]

    if subtype == "single_gross_outlier":
        k = int(rng.integers(sl.start, sl.stop))
        amp = float(rng.uniform(5.0, 15.0) * ch_sigma)
        signed_amp = float(rng.choice([-1, 1]) * amp)
        z_obs[k] = z_base[k] + signed_amp
        info = dict(
            error_type="measurement_error",
            subtype=subtype,
            channel=ch,
            index=k,
            amplitude=signed_amp,
        )
    else:
        n_in_ch = sl.stop - sl.start
        m_max = min(5, n_in_ch)
        m_min = 2 if n_in_ch >= 2 else 1
        m = int(rng.integers(m_min, m_max + 1))
        idxs = rng.choice(np.arange(sl.start, sl.stop), size=m, replace=False)
        amps = rng.uniform(5.0, 15.0, size=m) * ch_sigma
        signs = rng.choice([-1, 1], size=m)
        deltas = signs * amps
        z_obs[idxs] = z_base[idxs] + deltas
        info = dict(
            error_type="measurement_error",
            subtype=subtype,
            channel=ch,
            indices=[int(i) for i in idxs],
            amplitudes=[float(a) for a in deltas],
        )

    return z_obs, info


def apply_parameter_error_oneline(ppc_nominal, rng, r_range, x_range):
    """Create a true ppc with ONE line-parameter error (transformers excluded)."""
    ppc_true = deepcopy(ppc_nominal)
    br = ppc_true["branch"]

    mask_line = branch_line_mask(ppc_true)
    eps = 1e-9
    mask_normal = mask_line & (np.abs(br[:, BR_R]) > eps) & (np.abs(br[:, BR_X]) > eps)
    idx_candidates = np.where(mask_normal)[0]
    if len(idx_candidates) == 0:
        raise RuntimeError("No in-service non-zero-R/X lines found to perturb.")

    i = int(rng.choice(idx_candidates))
    which = rng.choice(["R", "X", "RX"])

    def _pick_factor(local_rng, val_range):
        if isinstance(val_range, list):
            sub_range = local_rng.choice(val_range)
            return local_rng.uniform(*sub_range)
        return local_rng.uniform(*val_range)

    r_factor = 1.0
    x_factor = 1.0
    if which in ("R", "RX"):
        r_factor = _pick_factor(rng, r_range)
    if which in ("X", "RX"):
        x_factor = _pick_factor(rng, x_range)

    br[i, BR_R] = max(1e-6, br[i, BR_R] * r_factor)
    br[i, BR_X] = max(1e-6, br[i, BR_X] * x_factor)

    label = dict(
        error_type="parameter_error",
        subtype=which,
        line_row=int(i),
        from_bus=int(br[i, F_BUS]),
        to_bus=int(br[i, T_BUS]),
        r_factor=float(r_factor),
        x_factor=float(x_factor),
    )
    return ppc_true, label


def make_initial_state_guess(bus_solved: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Small perturbation around the solved state for multi-scan parameter correction."""
    vm = np.asarray(bus_solved[:, VM], dtype=float).copy()
    va = np.asarray(bus_solved[:, VA], dtype=float).copy()
    vm += rng.normal(0.0, 2e-4, size=vm.shape[0])
    va += rng.normal(0.0, 5e-2, size=va.shape[0])  # degrees
    return np.concatenate([vm, va]).astype(float).tolist()


# ------------------------ sample builders ------------------------


def make_no_error_record(rng, ppc_base, idx_map, load_scale_min, load_scale_max):
    alpha = float(rng.uniform(load_scale_min, load_scale_max))
    ppc_scaled = scale_loads(ppc_base, alpha, alpha)
    solved = solve_ac_opf(ppc_scaled)
    if solved is None:
        return None

    z_true = compute_measurements_pu(solved)
    z_obs = z_true + base_gaussian_noise(z_true, idx_map, DEFAULT_SIGMAS, rng)
    return dict(
        id=f"ne_{rng.integers(1e12)}",
        scenario="no_error",
        z_true=z_true.tolist(),
        z_obs=z_obs.tolist(),
        label=dict(error_type="no_error"),
        op_point=dict(load_scale=alpha),
    )


def make_measurement_error_record(rng, ppc_base, idx_map, load_scale_min, load_scale_max):
    alpha = float(rng.uniform(load_scale_min, load_scale_max))
    ppc_scaled = scale_loads(ppc_base, alpha, alpha)
    solved = solve_ac_opf(ppc_scaled)
    if solved is None:
        return None

    z_true = compute_measurements_pu(solved)
    z_obs, info = apply_measurement_error(z_true, idx_map, rng)
    return dict(
        id=f"me_{rng.integers(1e12)}",
        scenario="measurement_error",
        z_true=z_true.tolist(),
        z_obs=z_obs.tolist(),
        label=info,
        op_point=dict(load_scale=alpha),
    )


def make_parameter_error_record(
    rng,
    ppc_base,
    idx_map,
    load_scale_min,
    load_scale_max,
    r_err_range,
    x_err_range,
    num_scans_for_correction,
    out_dir: Path,
):
    alpha = float(rng.uniform(load_scale_min, load_scale_max))
    ppc_scaled = scale_loads(ppc_base, alpha, alpha)
    ppc_true, label = apply_parameter_error_oneline(ppc_scaled, rng, r_err_range, x_err_range)
    solved_true = solve_ac_opf(ppc_true)
    if solved_true is None:
        return None

    z_true = compute_measurements_pu(solved_true)
    z_obs = z_true + base_gaussian_noise(z_true, idx_map, DEFAULT_SIGMAS, rng)

    sigma_R = sigma_vector(idx_map, DEFAULT_SIGMAS)
    nz = z_true.shape[0]

    z_scans = []
    initial_states = []
    for _ in range(int(num_scans_for_correction)):
        z_scan = (z_true + rng.standard_normal(nz) * sigma_R).astype(float).tolist()
        z_scans.append(z_scan)
        initial_states.append(make_initial_state_guess(solved_true["bus"], rng))

    case_basename = f"case_param_err_{rng.integers(1e12)}"
    case_path = write_ppc_as_matpower_m(
        ppc_true,
        out_dir / "cases_parameter_error" / f"{case_basename}.m",
        case_basename,
    )

    rec = dict(
        id=f"pe_{rng.integers(1e12)}",
        scenario="parameter_error",
        z_true=z_true.tolist(),
        z_obs=z_obs.tolist(),
        z_scans=z_scans,
        initial_states=initial_states,
        label=label,
        op_point=dict(load_scale=alpha),
        parameter_error_case_path=str(case_path),
        correction_case_path=str(case_path),
    )
    return rec


def make_topology_error_record(rng, ppc_base, idx_map, out_dir: Path):
    try:
        try:
            from . import nodebreaker_pp14 as nb  # type: ignore
        except Exception:
            import sys as _sys
            import os as _os
            _sys.path.append(_os.path.dirname(__file__))
            import nodebreaker_pp14 as nb  # type: ignore
    except Exception:
        return None

    status_map, topo_label = _choose_random_cb_open(rng, None)
    try:
        net, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123(status_map=status_map)
    except Exception:
        return None

    alpha = 1.0
    try:
        import pandapower as pp  # type: ignore
        pp.runpp(net, init="dc")
    except Exception:
        return None

    z_true = _nb_to_operator_z(net, line_idx, trafo_idx, ppc_base)
    z_obs = z_true + base_gaussian_noise(z_true, idx_map, DEFAULT_SIGMAS, rng)

    z_true_full_model = None
    corrected_model_path = None
    try:
        try:
            from . import nb_to_matpower as nb2mp
        except ImportError:
            import sys as _sys
            import os as _os
            _sys.path.append(_os.path.dirname(__file__))
            import nb_to_matpower as nb2mp

        net_bb, _ = nb2mp.topology_processed_busbranch(net)
        if hasattr(nb2mp, "_prune_dangling_buses"):
            nb2mp._prune_dangling_buses(net_bb)

        import pandapower as pp  # type: ignore
        try:
            pp.runpp(net_bb, init="flat")
        except Exception:
            pass

        # Export through the existing converter to get a reordered bus-branch model,
        # then persist our own text `.m` so the MCP regex parser can load it reliably.
        tmp_export_path = out_dir / "_tmp_topology_exports" / f"tmp_topology_{rng.integers(1e12)}.m"
        tmp_export_path.parent.mkdir(parents=True, exist_ok=True)
        ppc_full = nb2mp.export_to_matpower(net_bb, filename_mat=str(tmp_export_path))

        res_vm = net_bb.res_bus.sort_index()["vm_pu"].values
        res_va = net_bb.res_bus.sort_index()["va_degree"].values
        if len(res_vm) == ppc_full["bus"].shape[0]:
            ppc_full["bus"][:, VM] = res_vm
            ppc_full["bus"][:, VA] = res_va

        z_true_full_model = compute_measurements_pu(ppc_full)
        nb_full = ppc_full["bus"].shape[0]
        nl_full = ppc_full["branch"].shape[0]
        idx_map_full = make_index_map(nb_full, nl_full)
        z_true_full_model = z_true_full_model + base_gaussian_noise(
            z_true_full_model, idx_map_full, DEFAULT_SIGMAS, rng
        )

        model_basename = f"case_topology_corrected_{rng.integers(1e12)}"
        corrected_model_path = write_ppc_as_matpower_m(
            ppc_full,
            out_dir / "models_topology" / f"{model_basename}.m",
            model_basename,
        )
    except Exception as e:
        print(f"[warn] Failed to build topology verification model: {e}")
        z_true_full_model = None
        corrected_model_path = None

    rec = dict(
        id=f"to_{rng.integers(1e12)}",
        scenario="topology_error",
        z_true=z_true.tolist(),
        z_obs=z_obs.tolist(),
        label=topo_label,
        op_point=dict(load_scale=float(alpha)),
    )
    if z_true_full_model is not None:
        rec["z_true_full_model"] = z_true_full_model.tolist()
    if corrected_model_path is not None:
        rec["corrected_model_path"] = str(corrected_model_path)
    return rec


def make_harmonic_anomaly_record(rng):
    if not HARMONICS_AVAILABLE:
        return None

    src = int(rng.choice(HARMONIC_DEFAULT_CANDIDATES))
    thd = float(rng.uniform(0.10, 0.20))
    trace = build_trace(
        source_bus_1based=src,
        target_thd=thd,
        seed=int(rng.integers(1e9)),
    )

    harmonic_measurements = []
    harmonic_orders = []
    for h_str, meas_list in trace["harmonic_phasors"].items():
        h = int(h_str)
        harmonic_orders.append(h)
        for m in meas_list:
            c = m["V_complex_noisy"]
            harmonic_measurements.append(
                {
                    "bus": int(m["bus_1based"]),
                    "h": h,
                    "V_real": float(c[0]),
                    "V_imag": float(c[1]),
                    "sigma": float(m["sigma"]),
                }
            )

    return dict(
        id=f"ha_{rng.integers(1e12)}",
        scenario="harmonic_anomaly",
        z_true=trace["z_scada_true"],
        z_obs=trace["z_scada_meas"],
        harmonic_measurements=harmonic_measurements,
        harmonic_orders=sorted(set(int(h) for h in harmonic_orders)),
        label=dict(error_type="harmonic_anomaly", source_bus=src, thd_target=thd),
        op_point=dict(load_scale=1.0),
    )


# ------------------------ generation loop helpers ------------------------


def _emit_records(
    target_count: int,
    desc: str,
    make_record: Callable[[], Optional[Dict[str, Any]]],
    fout,
    *,
    max_attempt_multiplier: int,
) -> int:
    target_count = int(target_count)
    if target_count <= 0:
        return 0

    written = 0
    attempts = 0
    max_attempts = max(target_count * max_attempt_multiplier, target_count + 25)
    with tqdm(total=target_count, desc=desc) as pbar:
        while written < target_count and attempts < max_attempts:
            attempts += 1
            rec = make_record()
            if rec is None:
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            pbar.update(1)

    if written < target_count:
        print(f"[warn] {desc}: wrote {written}/{target_count} after {attempts} attempts.")
    return written


# ------------------------ main generator ------------------------


def generate_dataset(
    case_name="14",
    n_total=1000,
    frac_param_err=0.50,
    n_negative=500,
    n_topology=0,
    n_harmonic=0,
    seed=42,
    load_scale_min=0.80,
    load_scale_max=1.25,
    r_err_range=[(0.1, 0.5), (2.0, 5.0)],
    x_err_range=[(0.1, 0.5), (2.0, 5.0)],
    out_dir="out_sft_measurements",
    num_scans_for_correction: int = 8,
    max_attempt_multiplier: int = DEFAULTS["max_attempt_multiplier"],
):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ppc_base = load_case(case_name)
    baseMVA = ppc_base["baseMVA"]
    nb = ppc_base["bus"].shape[0]
    nl = ppc_base["branch"].shape[0]
    idx_map = make_index_map(nb, nl)

    br = ppc_base["branch"]
    branch_info = [
        dict(
            i=int(i),
            from_bus=int(br[i, F_BUS]),
            to_bus=int(br[i, T_BUS]),
            is_line=bool(br[i, TAP] == 0.0 and br[i, BR_STATUS] > 0),
        )
        for i in range(nl)
    ]

    n_topo = int(n_topology)
    n_param = int(round(n_total * frac_param_err))
    n_meas = max(0, int(n_total) - n_param - n_topo)
    n_harm = int(n_harmonic)

    meta = dict(
        case=f"case{case_name}",
        baseMVA=float(baseMVA),
        nb=int(nb),
        nl=int(nl),
        index_map={k: [v.start, v.stop] for k, v in idx_map.items()},
        measurement_order=MEASUREMENT_ORDER,
        branch_info=branch_info,
        sigmas=DEFAULT_SIGMAS,
        scenarios_emitted=["no_error", "measurement_error", "parameter_error", "topology_error", "harmonic_anomaly"],
        omitted_scenarios=["three_phase_imbalance"],
        requested_counts={
            "no_error": int(n_negative),
            "measurement_error": int(n_meas),
            "parameter_error": int(n_param),
            "topology_error": int(n_topo),
            "harmonic_anomaly": int(n_harm),
        },
        note="Measurements follow order [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]. Topology verification models and parameter-error models are stored as MATPOWER text .m files.",
    )

    samples_path = out / "samples.jsonl"
    written_counts = {k: 0 for k in meta["requested_counts"].keys()}

    with samples_path.open("w", encoding="utf-8") as fout:
        written_counts["no_error"] = _emit_records(
            int(n_negative),
            "No-error scenarios",
            lambda: make_no_error_record(rng, ppc_base, idx_map, load_scale_min, load_scale_max),
            fout,
            max_attempt_multiplier=max_attempt_multiplier,
        )

        written_counts["measurement_error"] = _emit_records(
            n_meas,
            "Measurement-error scenarios",
            lambda: make_measurement_error_record(rng, ppc_base, idx_map, load_scale_min, load_scale_max),
            fout,
            max_attempt_multiplier=max_attempt_multiplier,
        )

        written_counts["parameter_error"] = _emit_records(
            n_param,
            "Parameter-error scenarios",
            lambda: make_parameter_error_record(
                rng,
                ppc_base,
                idx_map,
                load_scale_min,
                load_scale_max,
                r_err_range,
                x_err_range,
                num_scans_for_correction,
                out,
            ),
            fout,
            max_attempt_multiplier=max_attempt_multiplier,
        )

        if case_name != "14" and n_topo > 0:
            print("[warn] Topology-error synthesis is only supported for case14; skipping.")
        written_counts["topology_error"] = _emit_records(
            n_topo if case_name == "14" else 0,
            "Topology-error scenarios",
            lambda: make_topology_error_record(rng, ppc_base, idx_map, out),
            fout,
            max_attempt_multiplier=max_attempt_multiplier,
        )

        if n_harm > 0 and (case_name != "14" or not HARMONICS_AVAILABLE):
            print("[warn] Harmonic-anomaly synthesis is only supported for case14 with the Harmonics module; skipping.")
        written_counts["harmonic_anomaly"] = _emit_records(
            n_harm if case_name == "14" and HARMONICS_AVAILABLE else 0,
            "Harmonic-anomaly scenarios",
            lambda: make_harmonic_anomaly_record(rng),
            fout,
            max_attempt_multiplier=max_attempt_multiplier,
        )

    meta["written_counts"] = {k: int(v) for k, v in written_counts.items()}
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote dataset to: {out.resolve()}")
    print(f"Samples: {samples_path.resolve()}")
    print(f"Meta: {(out / 'meta.json').resolve()}")


# ------------------------ CLI ------------------------


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["14", "118"], default=DEFAULTS["case_name"])
    p.add_argument("--n", type=int, default=DEFAULTS["n_total"])
    p.add_argument("--frac-param", type=float, default=DEFAULTS["frac_param_err"])
    p.add_argument("--neg", type=int, default=DEFAULTS["n_negative"], help="number of clean no-error samples")
    p.add_argument("--topo", type=int, default=200, help="number of topology-error samples (IEEE-14 only)")
    p.add_argument("--harm", type=int, default=100, help="number of harmonic-anomaly samples (IEEE-14 only)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--out", type=str, default="out_sft_measurements")
    p.add_argument("--ls-min", type=float, default=DEFAULTS["load_scale_min"])
    p.add_argument("--ls-max", type=float, default=DEFAULTS["load_scale_max"])
    p.add_argument("--scans", type=int, default=8, help="number of multi-scan snapshots for parameter correction")
    p.add_argument("--attempt-mult", type=int, default=DEFAULTS["max_attempt_multiplier"], help="max attempts per requested sample = target * attempt_mult")
    args = p.parse_args()

    generate_dataset(
        case_name=args.case,
        n_total=args.n,
        frac_param_err=args.frac_param,
        n_negative=args.neg,
        n_topology=args.topo,
        n_harmonic=args.harm,
        seed=args.seed,
        load_scale_min=args.ls_min,
        load_scale_max=args.ls_max,
        out_dir=args.out,
        num_scans_for_correction=args.scans,
        max_attempt_multiplier=args.attempt_mult,
    )
