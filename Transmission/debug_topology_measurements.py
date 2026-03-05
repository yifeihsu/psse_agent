#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug script for topology measurement generation.
Isolates the logic from generate_measurements.py to allow inspecting a single scenario.
"""

import json
import numpy as np
import pandapower as pp
from copy import deepcopy
from pypower.api import case14, makeYbus
from pypower.idx_bus import VM, VA, PD, QD, BUS_I
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, BR_STATUS

# Import local modules
try:
    import nodebreaker_pp14 as nb
    import nb_to_matpower as nb2mp
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    import nodebreaker_pp14 as nb
    import nb_to_matpower as nb2mp

# ------------------------ Helpers ------------------------

def load_case(case_name: str):
    if case_name == "14":
        return case14()
    raise ValueError("Only case14 supported")

def compute_measurements_pu(ppc_solved):
    """
    Build z in per-unit using solved voltages and admittance matrices:
      z = [ Vm; Pinj; Qinj; Pf; Qf; Pt; Qt ]
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
    # Detect indexing scheme
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

    # branch-end flows
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

    z = np.concatenate([Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]).astype(float)
    return z

def _extract_busnum_from_name(name: str) -> int | None:
    s = str(name)
    if s.startswith("B"):
        digits = ''.join(ch for ch in s[1:] if ch.isdigit())
        try:
            return int(digits) if digits else None
        except:
            return None
    digits = []
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    try:
        return int(''.join(digits)) if digits else None
    except:
        return None

def _nb_to_operator_z(net, line_idx: dict, trafo_idx: dict, ppc_base) -> np.ndarray:
    """Aggregate node-breaker PF results to operator's standard z order."""
    baseMVA = float(ppc_base["baseMVA"])
    bus = ppc_base["bus"]
    branch = ppc_base["branch"]
    nb = bus.shape[0]
    nl = branch.shape[0]

    groups: dict[int, list[int]] = {i+1: [] for i in range(nb)}
    for bidx, row in net.bus.iterrows():
        name = str(row.get('name', f'bus{bidx}'))
        bnum = _extract_busnum_from_name(name)
        if bnum in groups:
            groups[bnum].append(int(bidx))

    # 1) Vm
    Vm = np.zeros(nb, dtype=float)
    for i in range(1, nb+1):
        idxs = groups.get(i, [])
        if not idxs:
            Vm[i-1] = np.nan
        else:
            vms = []
            for j in idxs:
                try:
                    if bool(net.bus.at[j, 'in_service']):
                        val = float(net.res_bus.at[j, 'vm_pu'])
                        if np.isfinite(val):
                            vms.append(val)
                except: pass
            Vm[i-1] = float(np.mean(vms)) if vms else np.nan

    # 2) Pinj/Qinj
    Pinj = np.zeros(nb, dtype=float)
    Qinj = np.zeros(nb, dtype=float)
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
                        if np.isfinite(pv): p += pv
                        if np.isfinite(qv): q += qv
                except: pass
            
            if hasattr(net, "shunt") and len(net.shunt) > 0:
                try:
                    shunts_at_j = net.shunt[net.shunt.bus.isin(idxs)].index
                    for s_idx in shunts_at_j:
                        if bool(net.shunt.at[s_idx, 'in_service']):
                            pv = float(net.res_shunt.at[s_idx, 'p_mw'])
                            qv = float(net.res_shunt.at[s_idx, 'q_mvar'])
                            if np.isfinite(pv): p -= pv
                            if np.isfinite(qv): q -= qv
                except: pass
        Pinj[i-1] = -p / baseMVA
        Qinj[i-1] = -q / baseMVA

    # 3) Branch flows
    Pf = np.zeros(nl, dtype=float)
    Qf = np.zeros(nl, dtype=float)
    Pt = np.zeros(nl, dtype=float)
    Qt = np.zeros(nl, dtype=float)

    for k in range(nl):
        fb = int(branch[k, F_BUS])
        tb = int(branch[k, T_BUS])
        
        name_l = f"line_{fb}-{tb}"
        name_t = f"trafo_{fb}-{tb}"
        name_l_rev = f"line_{tb}-{fb}"
        name_t_rev = f"trafo_{tb}-{fb}"
        
        if name_l in line_idx:
            idx = int(line_idx[name_l])
            Pf[k] = float(net.res_line.at[idx, 'p_from_mw']) / baseMVA
            Qf[k] = float(net.res_line.at[idx, 'q_from_mvar']) / baseMVA
            Pt[k] = float(net.res_line.at[idx, 'p_to_mw']) / baseMVA
            Qt[k] = float(net.res_line.at[idx, 'q_to_mvar']) / baseMVA
        elif name_t in trafo_idx:
            idx = int(trafo_idx[name_t])
            Pf[k] = float(net.res_trafo.at[idx, 'p_hv_mw']) / baseMVA
            Qf[k] = float(net.res_trafo.at[idx, 'q_hv_mvar']) / baseMVA
            Pt[k] = float(net.res_trafo.at[idx, 'p_lv_mw']) / baseMVA
            Qt[k] = float(net.res_trafo.at[idx, 'q_lv_mvar']) / baseMVA
        elif name_l_rev in line_idx:
            idx = int(line_idx[name_l_rev])
            Pf[k] = float(net.res_line.at[idx, 'p_to_mw']) / baseMVA
            Qf[k] = float(net.res_line.at[idx, 'q_to_mvar']) / baseMVA
            Pt[k] = float(net.res_line.at[idx, 'p_from_mw']) / baseMVA
            Qt[k] = float(net.res_line.at[idx, 'q_from_mvar']) / baseMVA
        elif name_t_rev in trafo_idx:
            idx = int(trafo_idx[name_t_rev])
            Pf[k] = float(net.res_trafo.at[idx, 'p_lv_mw']) / baseMVA
            Qf[k] = float(net.res_trafo.at[idx, 'q_lv_mvar']) / baseMVA
            Pt[k] = float(net.res_trafo.at[idx, 'p_hv_mw']) / baseMVA
            Qt[k] = float(net.res_trafo.at[idx, 'q_hv_mvar']) / baseMVA

    z = np.concatenate([Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]).astype(float)
    return z

# ------------------------ Main Logic ------------------------

def generate_single_sample(cb_name: str, desired_status: bool):
    print(f"Generating sample for {cb_name} = {desired_status}...")
    
    # 1. Build Node-Breaker Net
    net_truth, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123()
    status_map = {name: bool(net_truth.impedance.at[idx, "in_service"]) for name, idx in cb_idx.items()}
    
    if cb_name in status_map:
        status_map[cb_name] = desired_status
    else:
        print(f"Error: {cb_name} not found in status_map")
        return

    net, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123(status_map=status_map)
    
    # 2. Run PF (No scaling)
    pp.runpp(net, init="dc")
    
    # 3. Generate z_true (Operator View)
    ppc_base = load_case("14")
    z_true = _nb_to_operator_z(net, line_idx, trafo_idx, ppc_base)
    print(f"z_true shape: {z_true.shape}")
    
    # 4. Generate z_true_full_model (Corrected View)
    net_bb, _ = nb2mp.topology_processed_busbranch(net)
    
    print("\nDEBUG: net_bb branch names:")
    if hasattr(net_bb, "line"):
        print("Lines:", net_bb.line.name.tolist())
    if hasattr(net_bb, "trafo"):
        print("Trafos:", net_bb.trafo.name.tolist())
    
    # Prune dangling buses (sync with nb_to_matpower logic)
    # This handles the zero-injection tail nodes.
    if hasattr(nb2mp, "_prune_dangling_buses"):
        nb2mp._prune_dangling_buses(net_bb)
    
    pp.runpp(net_bb, init="flat")
    
    # Save to a debug file for WLS
    debug_mat_path = "debug_topology.mat"
    ppc_full = nb2mp.export_to_matpower(net_bb, filename_mat=debug_mat_path)
    print(f"Exported debug model to {debug_mat_path}")
    
    # Populate results
    VM_idx = 7
    VA_idx = 8
    res_vm = net_bb.res_bus.sort_index()['vm_pu'].values
    res_va = net_bb.res_bus.sort_index()['va_degree'].values
    
    if len(res_vm) == ppc_full['bus'].shape[0]:
        ppc_full['bus'][:, VM_idx] = res_vm
        ppc_full['bus'][:, VA_idx] = res_va
    else:
        print(f"Warning: Result shape mismatch. net_bb: {len(res_vm)}, ppc: {ppc_full['bus'].shape[0]}")

    z_true_full = compute_measurements_pu(ppc_full)
    print(f"z_true_full shape: {z_true_full.shape}")

    # Save z to .mat file
    try:
        from scipy.io import savemat
        savemat("debug_measurements.mat", {"z": z_true_full})
        print("Saved measurement vector to debug_measurements.mat")
    except ImportError:
        print("Could not save .mat file (scipy not installed)")
    
    # 5. Compare specific values (e.g., Pf on split branches)
    print("\nRunning WLS...")
    import requests
    import os
    
    endpoint = "http://127.0.0.1:3929/tools"
    def mcp_call(name: str, args: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        headers = {"Accept": "application/json, text/event-stream"}
        try:
            r = requests.post(endpoint, json=payload, headers=headers, timeout=90)
            r.raise_for_status()
            res = r.json().get("result", {})
            sc = res.get("structuredContent")
            if isinstance(sc, dict):
                return sc
            texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
            for t in texts:
                try:
                    return json.loads(t)
                except:
                    pass
            return {}
        except Exception as e:
            print(f"MCP Call Failed: {e}")
            return {}

    abs_path = os.path.abspath(debug_mat_path)
    wls = mcp_call("wls_from_path", {"case_path": abs_path, "z": z_true_full.tolist()})
    
    if "error" in wls:
        print("WLS Error:", wls["error"])
    
    r = np.array(wls.get("r", []), float)
    if r.size:
        print(f"WLS success: {wls.get('success')}")
        print(f"Max |r_norm|: {np.nanmax(np.abs(r)):.3f}")
        
        # Print top residuals
        print("Top 5 residuals:")
        top_r_idxs = np.argsort(np.abs(r))[-5:][::-1]
        for idx in top_r_idxs:
            print(f"  idx {idx}: r={r[idx]:.4f} (z={z_true_full[idx]:.4f})")
    else:
        print("WLS returned no residuals.")
        print(wls)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cb", type=str, default="CB_2R3_2R4", help="CB Name")
    p.add_argument("--status", type=str, default="open", help="open/closed")
    args = p.parse_args()
    
    status = False if args.status.lower() in ["open", "false", "0"] else True
    generate_single_sample(args.cb, status)
