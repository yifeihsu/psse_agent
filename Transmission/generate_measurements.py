#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""\nGenerate measurement data for SFT/RFT.\n\nScenarios:\n  - Negative: no parameter error and no gross measurement error; only base sensor noise.\n  - Measurement errors: gaussian noise + one of {single_gross_outlier, multi_gross_outliers}.\n  - Parameter errors (lines only): perturb R, X, or both on ONE line; transformers are excluded.\n\nFor parameter-error scenarios we now also synthesize a small multi-snapshot series\nfor later parameter correction, exposing:\n  - z_scans: 5–10 noisy snapshots around the same operating point (each full z vector).\n  - initial_states: corresponding initial state guesses [Vm(1..nb), Va_deg(1..nb)] per scan.\n\nOutputs:\n  - samples.jsonl    (one JSON object per scenario with z_true, z_obs, labels, and for\n                      parameter_error: z_scans, initial_states)\n  - meta.json        (index slices, branch order, line mask, baseMVA, case)\n"""

import json
import math
import random
import sys
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path for Harmonics and scripts imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# PYPOWER API: cases, solver, options, admittance builder, indices
from pypower.api import case14, case118, runopf, ppoption, makeYbus
from pypower.idx_bus import VM, VA, PD, QD, BUS_I
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, BR_STATUS

# Harmonics utilities (for harmonic_anomaly)
try:
    from scripts.trigger_hse import build_full_harmonic_z, simulate_harmonic_voltage_meter_measurements
    from Harmonics.ieee14_verification import BUS as H_BUS, BRANCH as H_BRANCH, BASE_MVA as H_BASE_MVA
    HARMONICS_AVAILABLE = True
except ImportError as e:
    HARMONICS_AVAILABLE = False
    print(f"[warn] Harmonics modules not found ({e}). Harmonic anomaly subset will be skipped.")



# ------------------------ configuration ------------------------

DEFAULTS = dict(
    case_name="14",            # "14" or "118"
    n_total=2000,              # total scenarios
    frac_param_err=0.50,       # fraction parameter-error vs measurement-error
    n_negative=500,            # additional negative (clean) scenarios
    seed=42,                   # reproducibility
    # per-unit noise levels aligned with your earlier W matrix
    sigma_vm=1e-3,
    sigma_inj=1e-2,
    sigma_flow=1e-2,
    # load scaling to diversify operating points (kept conservative for feasibility)
    load_scale_min=0.80,
    load_scale_max=1.25,
    # parameter error magnitudes (multiplicative factors)
    # parameter error magnitudes (multiplicative factors)
    # List of tuples for disjoint ranges: e.g. [(0.1, 0.5), (2.0, 5.0)]
    r_err_range=[(0.1, 0.5), (2.0, 5.0)], 
    x_err_range=[(0.1, 0.5), (2.0, 5.0)],
)


# ------------------------ helpers: case, power flow, measurements ------------------------

def load_case(case_name: str):
    if case_name == "14":
        ppc = case14()
    elif case_name == "118":
        ppc = case118()
    else:
        raise ValueError("case_name must be '14' or '118'")
    return ppc

def scale_loads(ppc, alpha_p: float, alpha_q: float = None):
    """Scale Pd/Qd to diversify operating points."""
    if alpha_q is None:
        alpha_q = alpha_p
    ppc = deepcopy(ppc)
    ppc["bus"][:, PD] *= alpha_p
    ppc["bus"][:, QD] *= alpha_q
    return ppc

def solve_ac_opf(ppc):
    """Run AC optimal power flow with quiet options; return solved ppc or None if fails."""
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    results = runopf(deepcopy(ppc), ppopt)
    return results if results.get("success") else None

def compute_measurements_pu(ppc_solved):
    """
    Build z in per-unit using solved voltages and admittance matrices:
      z = [ Vm; Pinj; Qinj; Pf; Qf; Pt; Qt ]
    Lines and transformers are both included in flows here; a 'line mask' is provided in meta.

    This function is robust to PYPOWER/MATPOWER-style 1-based bus indexing as well
    as pandapower-to-ppc output which is already 0-based. It detects the minimum
    F_BUS/T_BUS and only subtracts 1 if buses are 1-based.
    """
    baseMVA = ppc_solved["baseMVA"]
    bus     = ppc_solved["bus"]
    branch  = ppc_solved["branch"]
    nb      = bus.shape[0]

    # complex voltages
    Vm = bus[:, VM]
    Va = np.deg2rad(bus[:, VA])
    V  = Vm * np.exp(1j * Va)

    # admittances (per-unit)
    bus_int = bus.copy()
    bus_int[:, BUS_I] = np.arange(nb)
    branch_int = branch.copy()
    # Detect indexing scheme: 1-based (MATPOWER) vs 0-based (pandapower to_ppc)
    min_f = branch_int[:, F_BUS].min() if branch_int.shape[0] else 0
    min_t = branch_int[:, T_BUS].min() if branch_int.shape[0] else 0
    if min_f >= 1 and min_t >= 1:
        branch_int[:, F_BUS] = branch_int[:, F_BUS].astype(int) - 1
        branch_int[:, T_BUS] = branch_int[:, T_BUS].astype(int) - 1
    else:
        branch_int[:, F_BUS] = branch_int[:, F_BUS].astype(int)
        branch_int[:, T_BUS] = branch_int[:, T_BUS].astype(int)
    Ybus, Yf, Yt = makeYbus(baseMVA, bus_int, branch_int)

    # injections S = V * conj(I)
    Ibus = Ybus.dot(V)
    Sinj = V * np.conj(Ibus)
    Pinj = Sinj.real
    Qinj = Sinj.imag

    # branch-end flows (from/to)
    If = Yf.dot(V)
    It = Yt.dot(V)
    # map “from” and “to” voltages to V indices
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

    z = np.concatenate([Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]).astype(float)
    return z

def make_index_map(nb: int, nl: int):
    """Index slices for the concatenated measurement vector."""
    i_vm  = slice(0, nb)
    i_p   = slice(nb, 2*nb)
    i_q   = slice(2*nb, 3*nb)
    i_pf  = slice(3*nb, 3*nb + nl)
    i_qf  = slice(3*nb + nl, 3*nb + 2*nl)
    i_pt  = slice(3*nb + 2*nl, 3*nb + 3*nl)
    i_qt  = slice(3*nb + 3*nl, 3*nb + 4*nl)
    return dict(Vm=i_vm, Pinj=i_p, Qinj=i_q, Pf=i_pf, Qf=i_qf, Pt=i_pt, Qt=i_qt)

def branch_line_mask(ppc):
    """
    Return boolean mask for branches that are **lines** (not transformers).
    According to MATPOWER case format, TAP=0 indicates a transmission line (transformers have non-zero ratio). 
    """
    br = ppc["branch"]
    in_service = (br[:, BR_STATUS] > 0)
    is_line    = (br[:, TAP] == 0.0)  # transformers excluded
    return (in_service & is_line)


# ------------------------ helpers: node-breaker pocket + mapping ------------------------

def _extract_busnum_from_name(name: str) -> int | None:
    """Best-effort extract planning bus number from a pandapower bus name.

    Handles pocket names like '1N3', '2R5', '3|L34' (leading digits), and non-pocket
    names like 'B4' (prefix 'B' + digits). Returns None if not parseable.
    """
    s = str(name)
    # case 'B4', 'B10', ...
    if s.startswith("B"):
        digits = ''.join(ch for ch in s[1:] if ch.isdigit())
        try:
            return int(digits) if digits else None
        except Exception:
            return None
    # leading digits (e.g., '1N3', '3|L34')
    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    try:
        return int(''.join(digits)) if digits else None
    except Exception:
        return None


def _choose_random_cb_open(rng, target_bus: int | None = None):
    """Choose exactly one CLOSED CB in pocket {1,2,3} and OPEN it.

    Per requirement: never close an open CB, as open ones in this model are
    redundant and do not change the equivalent bus-branch model.

    Returns (status_map, label_dict) compatible with nodebreaker_pp14.build_nb_ieee14_pocket123.
    """
    # Import lazily to avoid imposing dependency when not used
    try:
        from . import nodebreaker_pp14 as nb  # type: ignore
    except Exception:
        import sys as _sys, os as _os
        _sys.path.append(_os.path.dirname(__file__))
        import nodebreaker_pp14 as nb  # type: ignore

    pocket = nb.detailed_substations_1_2_3()
    # Gather all closed CB candidates (optionally restricted to one substation)
    buses = [1, 2, 3] if target_bus is None else [int(target_bus)]
    candidates = []
    for b in buses:
        for cb in pocket[b].cbs:
            if bool(cb.closed):  # only closed -> can open
                candidates.append((b, cb))
    # If none in requested bus, fallback to all pocket buses
    if not candidates and target_bus is not None:
        for b in [1, 2, 3]:
            for cb in pocket[b].cbs:
                if bool(cb.closed):
                    candidates.append((b, cb))
    if not candidates:
        raise RuntimeError("No CLOSED CB available to open in pocket substations 1–3.")
    b, cb = rng.choice(candidates)
    new_status = False  # open it
    status_map = {cb.name: new_status}
    label = dict(
        error_type="topology_error",
        substation=int(b),
        cb_name=str(cb.name),
        old_status="closed" if cb.closed else "open",
        new_status="open",
    )
    return status_map, label


def _scale_nb_net(net, alpha_p: float, alpha_q: float | None = None):
    """Scale loads / gens / sgens P/Q by alpha to diversify operating points."""
    import numpy as _np
    if alpha_q is None:
        alpha_q = alpha_p
    # loads
    if hasattr(net, 'load') and len(net.load.index):
        net.load['p_mw'] = _np.array(net.load['p_mw'], dtype=float) * alpha_p
        net.load['q_mvar'] = _np.array(net.load['q_mvar'], dtype=float) * alpha_q
    # sgens
    if hasattr(net, 'sgen') and len(net.sgen.index):
        if 'p_mw' in net.sgen:
            net.sgen['p_mw'] = _np.array(net.sgen['p_mw'], dtype=float) * alpha_p
        if 'q_mvar' in net.sgen:
            net.sgen['q_mvar'] = _np.array(net.sgen['q_mvar'], dtype=float) * alpha_q
    # gens (treat as scheduled injections)
    if hasattr(net, 'gen') and len(net.gen.index):
        if 'p_mw' in net.gen:
            net.gen['p_mw'] = _np.array(net.gen['p_mw'], dtype=float) * alpha_p
        if 'q_mvar' in net.gen:
            net.gen['q_mvar'] = _np.array(net.gen['q_mvar'], dtype=float) * alpha_q


def _nb_to_operator_z(net, line_idx: dict, trafo_idx: dict, ppc_base) -> np.ndarray:
    """Aggregate node-breaker PF results to operator's standard z order.

    z = [ Vm(1..nb); Pinj(1..nb); Qinj(1..nb); Pf(1..nl); Qf(1..nl); Pt(1..nl); Qt(1..nl) ]
    where branch order and bus count are taken from ppc_base.
    """
    import numpy as _np
    baseMVA = float(ppc_base["baseMVA"])
    bus = ppc_base["bus"]
    branch = ppc_base["branch"]
    nb = bus.shape[0]
    nl = branch.shape[0]

    # Build planning-bus -> list of NB bus indices mapping
    groups: dict[int, list[int]] = {i+1: [] for i in range(nb)}
    for bidx, row in net.bus.iterrows():
        name = str(row.get('name', f'bus{bidx}'))
        bnum = _extract_busnum_from_name(name)
        if bnum in groups:
            groups[bnum].append(int(bidx))

    # 1) Vm: average Vm of in-service and finite entries only
    Vm = _np.zeros(nb, dtype=float)
    for i in range(1, nb+1):
        idxs = groups.get(i, [])
        if not idxs:
            Vm[i-1] = _np.nan
        else:
            vms = []
            for j in idxs:
                try:
                    if bool(net.bus.at[j, 'in_service']):
                        val = float(net.res_bus.at[j, 'vm_pu'])
                        if _np.isfinite(val):
                            vms.append(val)
                except Exception:
                    # ignore missing/invalid
                    pass
            Vm[i-1] = float(_np.mean(vms)) if vms else _np.nan

    # 2) Pinj/Qinj: sum net injections over in-service and finite entries, convert to pu
    Pinj = _np.zeros(nb, dtype=float)
    Qinj = _np.zeros(nb, dtype=float)
    for i in range(1, nb+1):
        idxs = groups.get(i, [])
        p = 0.0
        q = 0.0
        if idxs:
            for j in idxs:
                try:
                    if bool(net.bus.at[j, 'in_service']):
                        pv = float(net.res_bus.at[j, 'p_mw'])
                        qv = float(net.res_bus.at[j, 'q_mvar'])
                        if _np.isfinite(pv):
                            p += pv
                        if _np.isfinite(qv):
                            q += qv
                except Exception:
                    # ignore missing/invalid
                    # ignore missing/invalid
                    pass
            
            # Add shunt injections if any
            if hasattr(net, "shunt") and len(net.shunt) > 0:
                # We need to find shunts connected to bus j
                # This is inefficient inside the loop, but robust. 
                # Optimization: pre-build map if needed, but n_shunt is small.
                try:
                    shunts_at_j = net.shunt[net.shunt.bus == j].index
                    for s_idx in shunts_at_j:
                        if bool(net.shunt.at[s_idx, 'in_service']):
                            pv = float(net.res_shunt.at[s_idx, 'p_mw'])
                            qv = float(net.res_shunt.at[s_idx, 'q_mvar'])
                            if _np.isfinite(pv):
                                p -= pv
                            if _np.isfinite(qv):
                                q -= qv
                except Exception:
                    pass
        Pinj[i-1] = -p / baseMVA
        Qinj[i-1] = -q / baseMVA

    # 3) Branch-end flows in the MATPOWER branch order
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
            pf = float(net.res_line.at[idx, 'p_from_mw'])
            qf = float(net.res_line.at[idx, 'q_from_mvar'])
            pt = float(net.res_line.at[idx, 'p_to_mw'])
            qt = float(net.res_line.at[idx, 'q_to_mvar'])
            Pf[k] = pf / baseMVA
            Qf[k] = qf / baseMVA
            Pt[k] = pt / baseMVA
            Qt[k] = qt / baseMVA
        elif name_t in trafo_idx:
            idx = int(trafo_idx[name_t])
            pf = float(net.res_trafo.at[idx, 'p_hv_mw'])
            qf = float(net.res_trafo.at[idx, 'q_hv_mvar'])
            pt = float(net.res_trafo.at[idx, 'p_lv_mw'])
            qt = float(net.res_trafo.at[idx, 'q_lv_mvar'])
            Pf[k] = pf / baseMVA
            Qf[k] = qf / baseMVA
            Pt[k] = pt / baseMVA
            Qt[k] = qt / baseMVA
        elif name_l_rev in line_idx:
            idx = int(line_idx[name_l_rev])
            pf = float(net.res_line.at[idx, 'p_to_mw'])
            qf = float(net.res_line.at[idx, 'q_to_mvar'])
            pt = float(net.res_line.at[idx, 'p_from_mw'])
            qt = float(net.res_line.at[idx, 'q_from_mvar'])
            Pf[k] = pf / baseMVA
            Qf[k] = qf / baseMVA
            Pt[k] = pt / baseMVA
            Qt[k] = qt / baseMVA
        elif name_t_rev in trafo_idx:
            idx = int(trafo_idx[name_t_rev])
            pf = float(net.res_trafo.at[idx, 'p_lv_mw'])
            qf = float(net.res_trafo.at[idx, 'q_lv_mvar'])
            pt = float(net.res_trafo.at[idx, 'p_hv_mw'])
            qt = float(net.res_trafo.at[idx, 'q_hv_mvar'])
            Pf[k] = pf / baseMVA
            Qf[k] = qf / baseMVA
            Pt[k] = pt / baseMVA
            Qt[k] = qt / baseMVA


    z = _np.concatenate([Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]).astype(float)
    return z


# ------------------------ helpers: errors ------------------------

def base_gaussian_noise(z, idx_map, sigmas, rng=None):
    """Small zero-mean Gaussian noise consistent with W used earlier.

    Uses the provided RNG if given; otherwise falls back to numpy's global RNG.
    """
    nb_vm = idx_map["Vm"].stop - idx_map["Vm"].start
    nb    = nb_vm
    # we need nl for flows
    nl    = idx_map["Pf"].stop - idx_map["Pf"].start

    noise = np.zeros_like(z)
    if rng is None:
        rnorm = np.random.randn
    else:
        rnorm = rng.standard_normal
    noise[idx_map["Vm"]]  = sigmas["vm"]   * rnorm(nb)
    noise[idx_map["Pinj"]] = sigmas["inj"] * rnorm(nb)
    noise[idx_map["Qinj"]] = sigmas["inj"] * rnorm(nb)
    noise[idx_map["Pf"]]   = sigmas["flow"]* rnorm(nl)
    noise[idx_map["Qf"]]   = sigmas["flow"]* rnorm(nl)
    noise[idx_map["Pt"]]   = sigmas["flow"]* rnorm(nl)
    noise[idx_map["Qt"]]   = sigmas["flow"]* rnorm(nl)
    return noise

def apply_measurement_error(z_true, idx_map, rng):
    """
    Add base Gaussian noise + one additional measurement error:
      - single_gross_outlier: one random measurement gets a large additive error
      - multi_gross_outliers: multiple random measurements (within one channel) get large additive errors
    """
    # build baseline as actual measurements + base sensor noise
    sigmas = {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2}
    z_base = z_true + base_gaussian_noise(z_true, idx_map, sigmas, rng)
    z_obs = z_base.copy()

    subtypes = ["single_gross_outlier", "multi_gross_outliers"]
    subtype  = rng.choice(subtypes)

    # pick a channel
    channels = ["Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt"]
    ch = rng.choice(channels)
    sl = idx_map[ch]

    if subtype == "single_gross_outlier":
        k = rng.integers(sl.start, sl.stop)
        # 5–15x the nominal sigma of the channel
        ch_sigma = {"Vm":sigmas["vm"], "Pinj":sigmas["inj"], "Qinj":sigmas["inj"],
                    "Pf":sigmas["flow"], "Qf":sigmas["flow"], "Pt":sigmas["flow"], "Qt":sigmas["flow"]}[ch]
        amp = rng.uniform(5.0, 15.0) * ch_sigma
        # outlier added relative to baseline (actual measurement + base noise)
        z_obs[k] = z_base[k] + rng.choice([-1, 1]) * amp
        info = dict(error_type="measurement_error", subtype=subtype, channel=ch, index=int(k), amplitude=float(amp))

    elif subtype == "multi_gross_outliers":
        # choose 2-5 distinct indices within the chosen channel
        ch_sigma = {"Vm":sigmas["vm"], "Pinj":sigmas["inj"], "Qinj":sigmas["inj"],
                    "Pf":sigmas["flow"], "Qf":sigmas["flow"], "Pt":sigmas["flow"], "Qt":sigmas["flow"]}[ch]
        n_in_ch = sl.stop - sl.start
        m_max = min(5, n_in_ch)
        m_min = 2 if n_in_ch >= 2 else 1
        m = int(rng.integers(m_min, m_max + 1))
        idxs = rng.choice(np.arange(sl.start, sl.stop), size=m, replace=False)
        amps = rng.uniform(5.0, 15.0, size=m) * ch_sigma
        signs = rng.choice([-1, 1], size=m)
        deltas = signs * amps
        # outliers added relative to baseline for each selected index
        z_obs[idxs] = z_base[idxs] + deltas
        info = dict(error_type="measurement_error", subtype=subtype, channel=ch,
                    indices=[int(i) for i in idxs], amplitudes=[float(a) for a in deltas])

    elif subtype == "channel_bias":
        # additive bias 2–8 sigmas for the whole channel
        ch_sigma = {"Vm":sigmas["vm"], "Pinj":sigmas["inj"], "Qinj":sigmas["inj"],
                    "Pf":sigmas["flow"], "Qf":sigmas["flow"], "Pt":sigmas["flow"], "Qt":sigmas["flow"]}[ch]
        bias = rng.uniform(2.0, 8.0) * ch_sigma * rng.choice([-1, 1])
        z_obs[sl] += bias
        info = dict(error_type="measurement_error", subtype=subtype, channel=ch, bias=float(bias))

    else:  # channel_scale
        # multiplicative 1±[1..5]%
        scale = 1.0 + rng.choice([-1, 1]) * rng.uniform(0.01, 0.05)
        z_obs[sl] *= scale
        info = dict(error_type="measurement_error", subtype=subtype, channel=ch, scale=float(scale))

    return z_obs, info

def apply_parameter_error_oneline(ppc_nominal, rng, r_range, x_range):
    """
    Create 'true' ppc with ONE **line** parameter error (transformers excluded):
    randomly choose R-only, X-only or both; return true_ppc and ground-truth dict.
    """
    ppc_true = deepcopy(ppc_nominal)
    br = ppc_true["branch"]

    # Only perturb "normal" branches: in-service lines (not transformers) with BOTH R and X non-zero.
    # This avoids cases like R=0 lines (purely reactive) which are difficult for the current
    # parameter-correction routine (it enforces a strictly-positive lower bound on R/X).
    mask_line = branch_line_mask(ppc_true)  # exclude transformers :contentReference[oaicite:10]{index=10}
    eps = 1e-9
    mask_normal = mask_line & (np.abs(br[:, BR_R]) > eps) & (np.abs(br[:, BR_X]) > eps)
    idx_candidates = np.where(mask_normal)[0]
    if len(idx_candidates) == 0:
        raise RuntimeError("No in-service non-zero-R/X lines found to perturb.")

    i = int(rng.choice(idx_candidates))
    which = rng.choice(["R", "X", "RX"])

    
    def _pick_factor(rng, val_range):
        """Helper to pick a value from a range which might be a tuple (min, max) or list of tuples."""
        if isinstance(val_range, list):
            # disjoint ranges
            sub_range = rng.choice(val_range)
            return rng.uniform(*sub_range)
        else:
            # single tuple
            return rng.uniform(*val_range)

    r_factor = 1.0
    x_factor = 1.0
    if which in ("R", "RX"):
        r_factor = _pick_factor(rng, r_range)
    if which in ("X", "RX"):
        x_factor = _pick_factor(rng, x_range)

    # ensure positivity
    br[i, BR_R] = max(1e-6, br[i, BR_R] * r_factor)
    br[i, BR_X] = max(1e-6, br[i, BR_X] * x_factor)

    label = dict(
        error_type="parameter_error",
        subtype=which,
        line_row=i,  # row in MATPOWER/PYPOWER branch matrix
        from_bus=int(br[i, F_BUS]),
        to_bus=int(br[i, T_BUS]),
        r_factor=float(r_factor),
        x_factor=float(x_factor)
    )
    return ppc_true, label


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
):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ppc_base = load_case(case_name)
    baseMVA  = ppc_base["baseMVA"]
    nb       = ppc_base["bus"].shape[0]
    nl       = ppc_base["branch"].shape[0]
    idx_map  = make_index_map(nb, nl)

    # metadata
    br = ppc_base["branch"]
    branch_info = [
        dict(i=int(i),
             from_bus=int(br[i, F_BUS]),
             to_bus=int(br[i, T_BUS]),
             is_line=bool(br[i, TAP] == 0.0 and br[i, BR_STATUS] > 0))
        for i in range(nl)
    ]

    meta = dict(
        case=f"case{case_name}",
        baseMVA=float(baseMVA),
        nb=int(nb), nl=int(nl),
        index_map={k:[v.start, v.stop] for k, v in idx_map.items()},
        branch_info=branch_info,
        note="Flows (Pf/Qf/Pt/Qt) follow PYPOWER branch order."
    )
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # counts
    n_param = int(round(n_total * frac_param_err))
    n_meas  = max(0, n_total - n_param - int(n_topology))
    n_topo  = int(n_topology)
    n_harm  = int(n_harmonic)

    # writers
    fout = open(out / "samples.jsonl", "w")

    # -------- generate negative scenarios (clean; only base noise) --------
    for _ in tqdm(range(n_negative), desc="Negative scenarios"):
        alpha = rng.uniform(load_scale_min, load_scale_max)
        ppc_scaled = scale_loads(ppc_base, alpha, alpha)
        solved = solve_ac_opf(ppc_scaled)
        if solved is None:
            continue

        z_true = compute_measurements_pu(solved)
        z_obs  = z_true + base_gaussian_noise(z_true, idx_map, {"vm":1e-3, "inj":1e-2, "flow":1e-2}, rng)

        rec = dict(
            id=f"neg_{rng.integers(1e12)}",
            scenario="negative",
            z_true=z_true.tolist(),
            z_obs=z_obs.tolist(),
            label=dict(error_type="none"),
            op_point=dict(load_scale=float(alpha))
        )
        fout.write(json.dumps(rec) + "\n")

    # -------- generate measurement-error scenarios --------
    for _ in tqdm(range(n_meas), desc="Measurement-error scenarios"):
        # diversify the operating point
        alpha = rng.uniform(load_scale_min, load_scale_max)
        ppc_scaled = scale_loads(ppc_base, alpha, alpha)
        solved = solve_ac_opf(ppc_scaled)
        if solved is None:
            continue  # skip and let the loop proceed; or retry if you prefer

        z_true = compute_measurements_pu(solved)
        z_obs, info = apply_measurement_error(z_true, idx_map, rng)

        rec = dict(
            id=f"me_{rng.integers(1e12)}",
            scenario="measurement_error",
            z_true=z_true.tolist(),
            z_obs=z_obs.tolist(),
            label=info,
            op_point=dict(load_scale=float(alpha))
        )
        fout.write(json.dumps(rec) + "\n")

    # -------- generate parameter-error scenarios --------
    for _ in tqdm(range(n_param), desc="Parameter-error scenarios"):
        alpha = rng.uniform(load_scale_min, load_scale_max)
        ppc_scaled = scale_loads(ppc_base, alpha, alpha)

        # create "true" network with one line parameter error
        ppc_true, label = apply_parameter_error_oneline(ppc_scaled, rng, r_err_range, x_err_range)

        solved_true = solve_ac_opf(ppc_true)
        if solved_true is None:
            continue

        # measure from true physics, add only small base sensor noise
        z_true = compute_measurements_pu(solved_true)
        z_obs  = z_true + base_gaussian_noise(
            z_true, idx_map, {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2}, rng
        )

        # --- multi-snapshot series for parameter correction ---
        # Follow the MATLAB logic:
        #   multi_scan_measurements_z(:, scan) = z_true_clean_base + randn .* sigma_R;
        #   initial_states_multi_scan(:, scan) = [Vm; angle_deg];
        nz = z_true.shape[0]
        # Build sigma_R consistent with base_gaussian_noise
        sigma_R = np.zeros_like(z_true)
        sigma_R[idx_map["Vm"]] = 1e-3
        sigma_R[idx_map["Pinj"]] = 1e-2
        sigma_R[idx_map["Qinj"]] = 1e-2
        sigma_R[idx_map["Pf"]] = 1e-2
        sigma_R[idx_map["Qf"]] = 1e-2
        sigma_R[idx_map["Pt"]] = 1e-2
        sigma_R[idx_map["Qt"]] = 1e-2

        # Base clean measurement and state
        z_base = z_true
        bus_true = solved_true["bus"]
        # [V(1..nb); angle_deg(1..nb)] using VM, VA columns
        state0 = np.concatenate([bus_true[:, VM], bus_true[:, VA]]).astype(float).tolist()

        z_scans = []
        initial_states = []
        for _scan in range(num_scans_for_correction):
            noise_vec = rng.standard_normal(nz) * sigma_R
            z_scan = (z_base + noise_vec).astype(float).tolist()
            z_scans.append(z_scan)
            initial_states.append(state0)

        rec = dict(
            id=f"pe_{rng.integers(1e12)}",
            scenario="parameter_error",
            z_true=z_true.tolist(),
            z_obs=z_obs.tolist(),
            z_scans=z_scans,
            initial_states=initial_states,
            label=label,
            op_point=dict(load_scale=float(alpha))
        )
        fout.write(json.dumps(rec) + "\n")

    # -------- generate topology-error scenarios (IEEE-14 only) --------
    if case_name != "14" and n_topo > 0:
        print("[warn] Topology-error synthesis only supported for case14; skipping.")
    for _ in tqdm(range(n_topo), desc="Topology-error scenarios"):
        if case_name != "14":
            break
        # pick a random CB in pocket {1,2,3} and flip it
        rng_local = rng  # alias
        status_map, topo_label = _choose_random_cb_open(rng_local, None)
        # build node-breaker net with this modification
        try:
            try:
                from . import nodebreaker_pp14 as nb  # type: ignore
            except Exception:
                import sys as _sys, os as _os
                _sys.path.append(_os.path.dirname(__file__))
                import nodebreaker_pp14 as nb  # type: ignore
            net, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123(status_map=status_map)
        except Exception as e:
            # cannot build NB net; skip
            continue
        # scale operating point
        # alpha = rng_local.uniform(load_scale_min, load_scale_max)
        # _scale_nb_net(net, alpha, alpha)
        # For topology verification, we want to match the base case model used in quick_check_topology.py
        # So we skip scaling (alpha=1.0)
        alpha = 1.0
        # run PF (robust init)
        try:
            import pandapower as pp  # type: ignore
            pp.runpp(net, init="dc")
        except Exception:
            continue

        # Map NB PF measurements to operator (MATPOWER) z order (fixed operator model)
        z_true = _nb_to_operator_z(net, line_idx, trafo_idx, ppc_base)
        z_obs  = z_true + base_gaussian_noise(z_true, idx_map, {"vm":1e-3,"inj":1e-2,"flow":1e-2}, rng_local)

        # Also attach a full measurement set for the actual NB system model by
        # converting the pandapower net directly to a PYPOWER-style ppc and
        # computing z = [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt] on that model.
        z_true_full_model = None
        try:
            # Use the same logic as the topology correction to ensure model alignment
            try:
                from . import nb_to_matpower as nb2mp
            except ImportError:
                import sys, os
                sys.path.append(os.path.dirname(__file__))
                import nb_to_matpower as nb2mp
            
            # We need to rebuild the net to get the "processed" bus-branch model
            # nb_to_matpower.topology_processed_busbranch expects a net with impedance/switch status
            # The 'net' here already has the status_map applied.
            
            # 1. Convert to bus-branch using the standard logic
            net_bb, _ = nb2mp.topology_processed_busbranch(net)
            
            # Prune dangling buses (sync with nb_to_matpower logic)
            if hasattr(nb2mp, "_prune_dangling_buses"):
                nb2mp._prune_dangling_buses(net_bb)
            
            # 2. Run PF on the bus-branch net (needed for results)
            import pandapower as pp
            try:
                pp.runpp(net_bb, init="flat")
            except:
                pass
                
            # 3. Export to MATPOWER ppc (with reordering and tap fixes)
            # Save the corrected model to a file
            model_filename = f"model_{rng_local.integers(1e12)}.mat"
            model_path = out / "models" / model_filename
            model_path.parent.mkdir(parents=True, exist_ok=True)
            ppc_full = nb2mp.export_to_matpower(net_bb, filename_mat=str(model_path))
            
            # CRITICAL FIX: Populate ppc_full with the SOLVED results from net_bb
            # to_mpc does not copy results to VM/VA columns, it uses init values.
            # We must map net_bb.res_bus to ppc_full['bus'].
            # Assuming ppc_full['bus'] follows net_bb.bus order (which it should for 0..N-1 indices)
            
            # Ensure we extract results in the correct order
            res_vm = net_bb.res_bus.sort_index()['vm_pu'].values
            res_va = net_bb.res_bus.sort_index()['va_degree'].values
            
            if len(res_vm) == ppc_full['bus'].shape[0]:
                # VM/VA are imported from pypower.idx_bus (0-based indices)
                ppc_full['bus'][:, VM] = res_vm
                ppc_full['bus'][:, VA] = res_va
            else:
                print(f"Warning: Result shape mismatch. net_bb: {len(res_vm)}, ppc: {ppc_full['bus'].shape[0]}")

            z_true_full_model = compute_measurements_pu(ppc_full)
            
            # Add base Gaussian noise to the full model measurements as well.
            # We must use an index map consistent with ppc_full (which may have different NB/NL than base).
            if z_true_full_model is not None:
                width_full = z_true_full_model.shape[0]
                nb_full = ppc_full["bus"].shape[0]
                nl_full = ppc_full["branch"].shape[0]
                idx_map_full = make_index_map(nb_full, nl_full)
                
                # Double check sizes
                expected = 3*nb_full + 4*nl_full
                if width_full == expected:
                    z_true_full_model += base_gaussian_noise(
                        z_true_full_model, idx_map_full, {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2}, rng_local
                    )
                else:
                    print(f"Warning: z_true_full_model size {width_full} does not match expected {expected} (nb={nb_full}, nl={nl_full}).")



        except Exception as e:
            print(f"Warning: Failed to generate z_true_full_model: {e}")
            z_true_full_model = None
            model_path = None

        rec = dict(
            id=f"to_{rng_local.integers(1e12)}",
            scenario="topology_error",
            z_true=z_true.tolist(),
            z_obs=z_obs.tolist(),
            label=topo_label,
            op_point=dict(load_scale=float(alpha))
        )
        if z_true_full_model is not None:
            rec["z_true_full_model"] = z_true_full_model.tolist()
        if model_path is not None:
            # Store absolute path for easy access
            rec["corrected_model_path"] = str(model_path.resolve())
        fout.write(json.dumps(rec) + "\n")

    # -------- generate harmonic-anomaly scenarios (IEEE-14 only) --------
    if n_harm > 0:
        if case_name != "14" or not HARMONICS_AVAILABLE:
            print("[warn] Harmonic-anomaly synthesis only supported for case14 with Harmonics module. Skipping.")
        else:
            for _ in tqdm(range(n_harm), desc="Harmonic-anomaly scenarios"):
                # Always generate with Harmonics on
                # Since harmonic modeling in `ieee14_verification` is hardcoded to a specific base model,
                # we don't scale operating loads here natively unless we also update the harmonics base load arrays.
                # For now, default op scale is used inside build_full_harmonic_z implicitly.
                
                # We can randomize the seed for different noise realizations and source levels
                h_seed = int(rng.integers(1e9))
                n_seed = int(rng.integers(1e9))
                
                # Run the trigger_hse builder (which gives SCADA + Harmonic states)
                z_full, V_by_h_true, Iinj_by_h_true = build_full_harmonic_z(
                    bus=H_BUS,
                    branch=H_BRANCH,
                    base_mva=H_BASE_MVA,
                    harmonic_on=True,
                    rng_seed_harmonic=h_seed,
                    rng_seed_noise=n_seed,
                    add_noise=True,
                    return_harmonic_states=True
                )
                
                # Simulate the sparse harmonic voltage measurements needed by HSE
                # E.g., assume 6 harmonic meters are available 
                meter_buses = [2, 3, 4, 5, 9, 14] # 1-based
                harmonic_orders = [5, 7, 11, 13, 17, 19]
                Vh_meas_by_h = simulate_harmonic_voltage_meter_measurements(
                    V_by_h_true,
                    meter_buses_1based=meter_buses,
                    harmonic_orders=harmonic_orders,
                    sigma_v=5e-4,
                    rng_seed=n_seed + 1
                )
                
                # Flatten Vh_meas_by_h into a list of dictionaries for JSON serialization and tool input
                harmonic_measurements_list = []
                for h in harmonic_orders:
                    buses0, Vmeas, sigma = Vh_meas_by_h.get(h, ([], [], []))
                    for i_idx, b0 in enumerate(buses0):
                        harmonic_measurements_list.append({
                            "bus": int(b0) + 1,  # 1-based
                            "h": int(h),
                            "V_real": float(Vmeas[i_idx].real),
                            "V_imag": float(Vmeas[i_idx].imag),
                            "sigma": float(sigma[i_idx])
                        })

                # Create the true SCADA vector (re-run without noise) to get z_true
                z_true, _, _ = build_full_harmonic_z(
                    bus=H_BUS,
                    branch=H_BRANCH,
                    base_mva=H_BASE_MVA,
                    harmonic_on=True,
                    rng_seed_harmonic=h_seed,
                    rng_seed_noise=n_seed,
                    add_noise=False,
                    return_harmonic_states=True
                )

                rec = dict(
                    id=f"ha_{rng.integers(1e12)}",
                    scenario="harmonic_anomaly",
                    z_true=z_true,
                    z_obs=z_full,
                    harmonic_measurements=harmonic_measurements_list,
                    label=dict(error_type="harmonic_anomaly", source_bus=3, thd_target=10.0), # Hardcoded to 3 in current `build_full_harmonic_z`
                    op_point=dict(load_scale=1.0)
                )
                fout.write(json.dumps(rec) + "\n")

    fout.close()
    print(f"\nWrote dataset to: {out.resolve()}")


# ------------------------ CLI ------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["14","118"], default=DEFAULTS["case_name"])
    p.add_argument("--n", type=int, default=DEFAULTS["n_total"])
    p.add_argument("--frac-param", type=float, default=DEFAULTS["frac_param_err"])
    p.add_argument("--neg", type=int, default=DEFAULTS["n_negative"], help="number of negative (clean) samples")
    p.add_argument("--topo", type=int, default=200, help="number of topology-error samples (IEEE-14 only)")
    p.add_argument("--harm", type=int, default=100, help="number of harmonic-anomaly samples (IEEE-14 only)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--out", type=str, default="out_sft_measurements")
    p.add_argument("--ls-min", type=float, default=DEFAULTS["load_scale_min"])
    p.add_argument("--ls-max", type=float, default=DEFAULTS["load_scale_max"])
    # Ranges are now complex lists of tuples, removing simple CLI args for them to use defaults
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
        # Use defaults for ranges
        out_dir=args.out
    )

