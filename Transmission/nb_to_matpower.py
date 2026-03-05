"""
Helpers to convert the IEEE‑14 node‑breaker pocket model into a
topology‑processed bus‑branch model and export it as a MATPOWER case.

Short workflow:
- Build the node‑breaker net via nodebreaker_pp14.build_nb_ieee14_pocket123.
- Collapse buses connected by CLOSED CBs / bus‑bus switches into
  topological buses (NB → bus‑branch).
- Export the resulting bus‑branch pandapower net to an mpc struct
  using pandapower.converter.matpower.to_mpc, optionally writing a
  .mat file for MATLAB / MATPOWER.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandapower as pp  # type: ignore
import pandapower.converter.matpower as pcm  # type: ignore


def _import_nb_module():
    """Import node-breaker helper with a robust path for script or package use."""
    try:
        from . import nodebreaker_pp14 as nb  # type: ignore
        return nb
    except Exception:
        # fallback when running directly from repo root
        import sys, os

        sys.path.append(os.path.dirname(__file__))
        import nodebreaker_pp14 as nb  # type: ignore

        return nb


def _extract_busnum(name: str) -> int:
    """
    Extract integer bus number from names like 'bus1', 'bus1_a', '1', etc.
    Returns 999999 if no integer found.
    """
    import re
    m = re.search(r"(\d+)", str(name))
    if m:
        return int(m.group(1))
    return 999999


def topology_processed_busbranch(net_nb: pp.pandapowerNet) -> Tuple[pp.pandapowerNet, Dict[int, int]]:
    """
    From a node-breaker pandapower net (with CBs as impedance/switches),
    build a new bus-branch net with all CLOSED CBs collapsed into
    topological buses.

    Returns
    -------
    net_bb : pandapowerNet
        Topology-processed bus-branch network.
    old2topo_bus : dict
        Mapping from original bus index -> topological bus index in net_bb.
    """

    n_buses = len(net_nb.bus)

    # --- 1) Union-Find over buses using CLOSED CBs / bus-bus switches ---
    parent = np.arange(n_buses, dtype=int)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # (a) Closed CBs as impedance elements
    if hasattr(net_nb, "impedance") and len(net_nb.impedance):
        for idx in net_nb.impedance.index:
            if bool(net_nb.impedance.at[idx, "in_service"]):
                fb = int(net_nb.impedance.at[idx, "from_bus"])
                tb = int(net_nb.impedance.at[idx, "to_bus"])
                union(fb, tb)

    # (b) Closed bus-bus switches, if present
    if hasattr(net_nb, "switch") and len(net_nb.switch):
        for idx in net_nb.switch.index:
            if net_nb.switch.at[idx, "et"] == "b" and bool(net_nb.switch.at[idx, "closed"]):
                b1 = int(net_nb.switch.at[idx, "bus"])
                b2 = int(net_nb.switch.at[idx, "element"])  # bus-bus
                union(b1, b2)

    # Now each bus belongs to a topological component root
    root_of = {b: find(b) for b in net_nb.bus.index}

    # Group buses by root
    comps: Dict[int, list[int]] = {}
    for b, r in root_of.items():
        comps.setdefault(r, []).append(b)

    # --- 2) Build new bus-branch net ---
    net_bb = pp.create_empty_network(
        sn_mva=float(net_nb.sn_mva),
        f_hz=float(getattr(net_nb, "f_hz", 50.0)),
    )

    # Determine a "planning bus number" for each component to sort them
    # We look at all member buses and take the minimum extracted number.
    comp_sort_keys = []
    for r, members in comps.items():
        # Find min bus number among members
        min_bnum = 999999
        for m in members:
            name = net_nb.bus.at[m, "name"]
            bnum = _extract_busnum(name)
            if bnum < min_bnum:
                min_bnum = bnum
        comp_sort_keys.append((min_bnum, r))
    
    # Sort components by planning bus number
    comp_sort_keys.sort(key=lambda x: x[0])

    topo_root2bus: Dict[int, int] = {}
    
    for bnum, r in comp_sort_keys:
        members = comps[r]
        # pick a representative bus to inherit vn_kv, name, etc.
        # ideally pick the one that matched the bnum, but first is fine for properties
        rep = members[0]
        vn_kv = float(net_nb.bus.at[rep, "vn_kv"])
        # Use the planning bus number in the name if possible
        if bnum < 999999:
            name = f"Bus_{bnum}"
        else:
            name = f"TN_{r}"
        topo_bus = pp.create_bus(net_bb, vn_kv=vn_kv, name=name)
        topo_root2bus[r] = topo_bus

    old2topo_bus: Dict[int, int] = {b: topo_root2bus[root_of[b]] for b in net_nb.bus.index}

    # --- 3) Copy ext_grids ---
    if hasattr(net_nb, "ext_grid") and len(net_nb.ext_grid):
        for i in net_nb.ext_grid.index:
            b_old = int(net_nb.ext_grid.at[i, "bus"])
            b_new = old2topo_bus[b_old]
            pp.create_ext_grid(
                net_bb,
                bus=b_new,
                vm_pu=float(net_nb.ext_grid.at[i, "vm_pu"]),
                va_degree=float(net_nb.ext_grid.at[i, "va_degree"]),
                name=str(net_nb.ext_grid.at[i, "name"]) if "name" in net_nb.ext_grid.columns else f"EG_{b_new}",
            )

    # --- 4) Copy generators, loads, sgens, shunts ---
    if hasattr(net_nb, "gen") and len(net_nb.gen):
        for i in net_nb.gen.index:
            b_old = int(net_nb.gen.at[i, "bus"])
            b_new = old2topo_bus[b_old]
            pp.create_gen(
                net_bb,
                bus=b_new,
                p_mw=float(net_nb.gen.at[i, "p_mw"]),
                vm_pu=float(net_nb.gen.at[i, "vm_pu"]),
                name=str(net_nb.gen.at[i, "name"]) if "name" in net_nb.gen.columns else f"GEN_{b_new}",
            )

    if hasattr(net_nb, "load") and len(net_nb.load):
        for i in net_nb.load.index:
            b_old = int(net_nb.load.at[i, "bus"])
            b_new = old2topo_bus[b_old]
            pp.create_load(
                net_bb,
                bus=b_new,
                p_mw=float(net_nb.load.at[i, "p_mw"]),
                q_mvar=float(net_nb.load.at[i, "q_mvar"]),
                name=str(net_nb.load.at[i, "name"]) if "name" in net_nb.load.columns else f"LOAD_{b_new}",
            )

    if hasattr(net_nb, "sgen") and len(net_nb.sgen):
        for i in net_nb.sgen.index:
            b_old = int(net_nb.sgen.at[i, "bus"])
            b_new = old2topo_bus[b_old]
            pp.create_sgen(
                net_bb,
                bus=b_new,
                p_mw=float(net_nb.sgen.at[i, "p_mw"]),
                q_mvar=float(net_nb.sgen.at[i, "q_mvar"]),
                name=str(net_nb.sgen.at[i, "name"]) if "name" in net_nb.sgen.columns else f"SGEN_{b_new}",
            )

    if hasattr(net_nb, "shunt") and len(net_nb.shunt):
        for i in net_nb.shunt.index:
            b_old = int(net_nb.shunt.at[i, "bus"])
            b_new = old2topo_bus[b_old]
            pp.create_shunt(
                net_bb,
                bus=b_new,
                q_mvar=float(net_nb.shunt.at[i, "q_mvar"]),
                p_mw=float(net_nb.shunt.at[i, "p_mw"]),
                name=str(net_nb.shunt.at[i, "name"]) if "name" in net_nb.shunt.columns else f"SHUNT_{b_new}",
            )

    # --- 5) Copy conventional branches (lines & trafos) ---
    if hasattr(net_nb, "line") and len(net_nb.line):
        for i in net_nb.line.index:
            fb_old = int(net_nb.line.at[i, "from_bus"])
            tb_old = int(net_nb.line.at[i, "to_bus"])
            fb_new = old2topo_bus[fb_old]
            tb_new = old2topo_bus[tb_old]
            if fb_new == tb_new:
                # line is now internal to a topological bus, skip it
                continue
            pp.create_line_from_parameters(
                net_bb,
                from_bus=fb_new,
                to_bus=tb_new,
                length_km=float(net_nb.line.at[i, "length_km"]),
                r_ohm_per_km=float(net_nb.line.at[i, "r_ohm_per_km"]),
                x_ohm_per_km=float(net_nb.line.at[i, "x_ohm_per_km"]),
                c_nf_per_km=float(net_nb.line.at[i, "c_nf_per_km"]),
                max_i_ka=float(net_nb.line.at[i, "max_i_ka"]),
                name=str(net_nb.line.at[i, "name"]) if "name" in net_nb.line.columns else f"LINE_{i}",
            )

    if hasattr(net_nb, "trafo") and len(net_nb.trafo):
        for i in net_nb.trafo.index:
            hv_old = int(net_nb.trafo.at[i, "hv_bus"])
            lv_old = int(net_nb.trafo.at[i, "lv_bus"])
            hv_new = old2topo_bus[hv_old]
            lv_new = old2topo_bus[lv_old]
            if hv_new == lv_new:
                continue
            # Check for tap parameters
            tap_pos = float(net_nb.trafo.at[i, "tap_pos"]) if "tap_pos" in net_nb.trafo.columns and not np.isnan(net_nb.trafo.at[i, "tap_pos"]) else np.nan
            tap_step_percent = float(net_nb.trafo.at[i, "tap_step_percent"]) if "tap_step_percent" in net_nb.trafo.columns and not np.isnan(net_nb.trafo.at[i, "tap_step_percent"]) else np.nan
            tap_step_degree = float(net_nb.trafo.at[i, "tap_step_degree"]) if "tap_step_degree" in net_nb.trafo.columns and not np.isnan(net_nb.trafo.at[i, "tap_step_degree"]) else np.nan
            tap_side = str(net_nb.trafo.at[i, "tap_side"]) if "tap_side" in net_nb.trafo.columns else None

            pp.create_transformer_from_parameters(
                net_bb,
                hv_bus=hv_new,
                lv_bus=lv_new,
                sn_mva=float(net_nb.trafo.at[i, "sn_mva"]),
                vn_hv_kv=float(net_nb.trafo.at[i, "vn_hv_kv"]),
                vn_lv_kv=float(net_nb.trafo.at[i, "vn_lv_kv"]),
                vkr_percent=float(net_nb.trafo.at[i, "vkr_percent"]),
                vk_percent=float(net_nb.trafo.at[i, "vk_percent"]),
                pfe_kw=float(net_nb.trafo.at[i, "pfe_kw"]),
                i0_percent=float(net_nb.trafo.at[i, "i0_percent"]),
                shift_degree=float(net_nb.trafo.at[i, "shift_degree"]) if "shift_degree" in net_nb.trafo.columns else 0.0,
                tap_pos=tap_pos,
                tap_step_percent=tap_step_percent,
                tap_step_degree=tap_step_degree,
                tap_side=tap_side,
                name=str(net_nb.trafo.at[i, "name"]) if "name" in net_nb.trafo.columns else f"TR_{i}",
            )

    return net_bb, old2topo_bus


def _prune_dangling_buses(net_bb: pp.pandapowerNet):
    """
    Iteratively remove 'dangling' buses:
      - Degree = 1 (connected to exactly 1 branch: line or trafo)
      - No power injection (no gen, sgen, load, shunt, or ext_grid)
    
    This handles 'tail' nodes created by bus splits where the split section
    has no active elements and is just fed by a line that carries no current 
    (except for charging), which the user wants removed from the verified model.
    """
    import pandapower.topology as ppt
    
    dirty = True
    while dirty:
        dirty = False
        
        # 1. Calculate bus degrees (connected branches)
        # We can use net_bb.line and net_bb.trafo
        bus_degree = {b: 0 for b in net_bb.bus.index}
        
        for idx in net_bb.line.index:
            f = int(net_bb.line.at[idx, 'from_bus'])
            t = int(net_bb.line.at[idx, 'to_bus'])
            if bool(net_bb.line.at[idx, 'in_service']):
                if f in bus_degree: bus_degree[f] += 1
                if t in bus_degree: bus_degree[t] += 1
                
        for idx in net_bb.trafo.index:
            hv = int(net_bb.trafo.at[idx, 'hv_bus'])
            lv = int(net_bb.trafo.at[idx, 'lv_bus'])
            # Trafo active status check? usually assume active if in net
            # But standard pp fields: 'in_service' might not be on trafo in some versions?
            # Assuming active since we just built it.
            if hv in bus_degree: bus_degree[hv] += 1
            if lv in bus_degree: bus_degree[lv] += 1
            
        # 2. Identify candidates for removal (Deg=1 & No Injection)
        # Injections: Gen, Sgen, Load, Shunt, ExtGrid
        
        # Build sets of injection buses
        inj_buses = set()
        if hasattr(net_bb, "gen") and len(net_bb.gen):
            inj_buses.update(net_bb.gen.bus.values)
        if hasattr(net_bb, "sgen") and len(net_bb.sgen):
            inj_buses.update(net_bb.sgen.bus.values)
        if hasattr(net_bb, "load") and len(net_bb.load):
            inj_buses.update(net_bb.load.bus.values)
        if hasattr(net_bb, "shunt") and len(net_bb.shunt):
            inj_buses.update(net_bb.shunt.bus.values)
        if hasattr(net_bb, "ext_grid") and len(net_bb.ext_grid):
            inj_buses.update(net_bb.ext_grid.bus.values)
            
        buses_to_remove = []
        for b, deg in bus_degree.items():
            if deg <= 1 and b not in inj_buses:
                buses_to_remove.append(b)
                
        if buses_to_remove:
            # print(f"Pruning {len(buses_to_remove)} dangling buses: {buses_to_remove}")
            dirty = True
            
            # Remove buses and connected elements
            # pandapower.drop_buses handles connected lines/trafos automatically
            try:
                pp.drop_buses(net_bb, buses_to_remove)
            except Exception as e:
                print(f"Warning: Failed to drop buses {buses_to_remove}: {e}")
                dirty = False # prevent loop if fail



def export_to_matpower(net_bb: pp.pandapowerNet, filename_mat: str = "case14_topology.mat"):
    """
    Export a bus-branch pandapower net to MATPOWER case struct (.mat).

    Parameters
    ----------
    net_bb : pandapowerNet
        Topology-processed bus-branch network.
    filename_mat : str
        Output .mat filename; passed through to pandapower's to_mpc.

    Returns
    -------
    ppc : dict
        MATPOWER ppc/mpc struct with keys 'baseMVA', 'bus', 'branch', 'gen', etc.
    """

    # Pass None for filename so pandaspower doesn't write it yet.
    # We will write it manually after reordering.
    ppc_obj = pcm.to_mpc(net_bb, filename=None)
    # pcm.to_mpc may return a tuple (ppc, ppci) on some versions; normalize to ppc dict
    if isinstance(ppc_obj, tuple) and len(ppc_obj) >= 1:
        ppc = ppc_obj[0]
    else:
        ppc = ppc_obj

    # Fix: sometimes to_mpc returns a dict {'mpc': actual_ppc}
    if isinstance(ppc, dict) and "mpc" in ppc and len(ppc) == 1:
        ppc = ppc["mpc"]
    
    # --- Reorder branches to match standard IEEE-14 case using NAMES ---
    # This is critical because the measurement vector z is ordered by the standard case14 branch list.
    # Even if bus numbers change (due to splits), the physical lines (and their names) persist.
    
    CASE14_STANDARD_NAMES = [
        'line_1-2', 'line_1-5', 'line_2-3', 'line_2-4', 'line_2-5', 
        'line_3-4', 'line_4-5', 'trafo_4-7', 'trafo_4-9', 'trafo_5-6', 
        'line_6-11', 'line_6-12', 'line_6-13', 'trafo_7-8', 'trafo_7-9', 
        'line_9-10', 'line_9-14', 'line_10-11', 'line_12-13', 'line_13-14'
    ]
    
    try:
        # Build a map of Name -> Row Index in the current ppc['branch']
        # We assume net_bb.line and net_bb.trafo correspond to the ppc rows.
        # Pandapower to_mpc stacks lines then trafos (sorted by index).
        # We use enumerate to handle non-contiguous indices (e.g. after pruning).
        
        name_to_ppc_row = {}
        
        # Lines
        # to_mpc sorts by index
        pp_lines = net_bb.line.sort_index()
        n_lines = len(pp_lines)
        for row_idx, (pp_idx, row_data) in enumerate(pp_lines.iterrows()):
            name = row_data['name']
            name_to_ppc_row[name] = row_idx
            
        # Trafos
        pp_trafos = net_bb.trafo.sort_index()
        for row_idx, (pp_idx, row_data) in enumerate(pp_trafos.iterrows()):
            name = row_data['name']
            name_to_ppc_row[name] = n_lines + row_idx
            
        # Debug check counts
        total_ppc_rows = ppc['branch'].shape[0]
        mapped_rows = len(name_to_ppc_row)
        if mapped_rows != total_ppc_rows:
            print(f"DEBUG REORDER: Map Size {mapped_rows} != PPC Size {total_ppc_rows}")
        
        # Now build the new branch matrix
        new_branch_rows = []
        used_rows = set()
        
        # 1. Add Standard Branches in Order
        missing_std = []
        for name in CASE14_STANDARD_NAMES:
            if name in name_to_ppc_row:
                row_idx = name_to_ppc_row[name]
                new_branch_rows.append(ppc['branch'][row_idx])
                used_rows.add(row_idx)
            else:
                missing_std.append(name)
        
        if missing_std:
            print(f"WARNING: Missing Standard Branches in net_bb: {missing_std}")

        # 2. Add Remaining Branches (e.g. split lines, new switches)
        total_rows = ppc['branch'].shape[0]
        for i in range(total_rows):
            if i not in used_rows:
                new_branch_rows.append(ppc['branch'][i])
        
        if len(new_branch_rows) == total_rows:
            ppc['branch'] = np.array(new_branch_rows)
            print("DEBUG: Successfully reordered branches to Standard Case14 order.")
        else:
            print(f"WARNING: Reordering failed. Result {len(new_branch_rows)} != Expected {total_rows}.")

    except Exception as e:
        print(f"WARNING: Branch reordering failed: {e}")

    # --- Manually fix taps and merge generators ---
    _fix_taps_and_gens(ppc, net_bb)

    # --- Manually inject Solved Voltages (VM, VA) if available ---
    # to_mpc initializes VM=1.0, VA=0.0. We want the PF result if it exists.
    # net_bb.res_bus should align with net_bb.bus (sorted or not?)
    # ppc['bus'] is generated from net_bb.bus sorted by index (usually).
    # We assume 1-to-1 mapping by row position because we haven't reordered ppc['bus'].
    try:
        if hasattr(net_bb, "res_bus") and len(net_bb.res_bus) == len(net_bb.bus):
            # Check for NaN
             if not net_bb.res_bus["vm_pu"].isnull().values.any():
                 # Valid results exist
                 # Sort res_bus by index to match standard bus iteration order
                 res_sorted = net_bb.res_bus.sort_index()
                 # ppc['bus'] columns: BUS_I(0), TYPE(1), PD(2), QD(3), GS(4), BS(5), AREA(6), VM(7), VA(8)
                 # MATPOWER VM is col 7, VA is col 8
                 ppc["bus"][:, 7] = res_sorted["vm_pu"].values
                 ppc["bus"][:, 8] = res_sorted["va_degree"].values
                 # print("DEBUG: Injected solved voltages into PPC.")
    except Exception as e:
        print(f"Warning: Failed to inject solved voltages: {e}")


    # --- Save to file if requested ---
    # We must do this AFTER reordering and fixes so the file on disk matches the returned struct.
    if filename_mat:
        import scipy.io
        # MATPOWER expects struct typically named 'mpc' or the vars directly.
        # Check if user wants top-level vars or 'mpc' struct. 
        # Requirement: "save the output model as a mpc struct" -> wrap in {"mpc": ppc}
        scipy.io.savemat(filename_mat, {"mpc": ppc})
        # print(f"DEBUG: Saved reordered ppc to {filename_mat}")

    return ppc


def _fix_taps_and_gens(ppc: dict, net_bb: pp.pandapowerNet):
    """
    Manually inject transformer taps from net_bb into ppc['branch']
    and merge duplicate generators at the same bus in ppc['gen'].
    """
    # 1. Fix Taps
    if hasattr(net_bb, "trafo") and len(net_bb.trafo):
        # We assume net_bb buses are sorted by planning number 1..N
        # So net_bb bus index i maps to ppc bus i+1.
        branches = ppc["branch"]
        for i in net_bb.trafo.index:
            t = net_bb.trafo.loc[i]
            if "tap_pos" in t and not np.isnan(t.tap_pos):
                step = t.tap_step_percent if "tap_step_percent" in t and not np.isnan(t.tap_step_percent) else 0.0
                tau = 1 + (t.tap_pos * step / 100.0)
                
                # Map HV/LV to ppc bus numbers
                hv = int(t.hv_bus) + 1
                lv = int(t.lv_bus) + 1
                
                # Find matching branch
                # We look for (hv, lv) or (lv, hv)
                found = False
                for r in range(branches.shape[0]):
                    f = int(branches[r, 0])
                    to = int(branches[r, 1])
                    if (f == hv and to == lv):
                        # Tap is at 'from' (HV) -> set TAP = tau
                        branches[r, 8] = tau
                        found = True
                        break
                    elif (f == lv and to == hv):
                        # Tap is at 'to' (HV) -> set TAP = 1/tau?
                        # MATPOWER TAP is ratio at 'from' bus.
                        # If physical tap is at HV, and HV is 'to' bus, then effective ratio at 'from' (LV) is 1/tau.
                        # But usually we orient branches so tap is at 'from'.
                        # If we can't reorient, we set TAP = 1/tau.
                        if tau != 0:
                            branches[r, 8] = 1.0 / tau
                        found = True
                        break
                if not found:
                    print(f"Warning: Could not find branch for trafo {t['name']} ({hv}-{lv}) to set tap {tau:.4f}")

    # 2. Merge Generators
    if "gen" in ppc and len(ppc["gen"]) > 0:
        gens = ppc["gen"]
        # Group by bus (column 0)
        # Columns: BUS(0), PG(1), QG(2), QMAX(3), QMIN(4), VG(5), MBASE(6), STATUS(7), PMAX(8), PMIN(9)
        new_gens = []
        
        # Sort by bus first
        gens = gens[gens[:, 0].argsort()]
        
        # Iterate and merge
        curr_bus = -1
        merged = None
        
        for row in gens:
            bus = int(row[0])
            if bus != curr_bus:
                if merged is not None:
                    new_gens.append(merged)
                curr_bus = bus
                merged = row.copy()
            else:
                # Merge with current
                merged[1] += row[1] # PG
                merged[2] += row[2] # QG
                merged[3] += row[3] # QMAX
                merged[4] += row[4] # QMIN
                # VG (5) keep first
                merged[6] += row[6] # MBASE
                merged[7] = 1 if (merged[7] > 0 or row[7] > 0) else 0 # STATUS
                merged[8] += row[8] # PMAX
                merged[9] += row[9] # PMIN
                
        if merged is not None:
            new_gens.append(merged)
            
        ppc["gen"] = np.array(new_gens)


def build_corrected_busbranch_and_export(
    bad_cb_name: str,
    desired_status: bool = False,
    filename_mat: str = "case14_topology.mat",
):
    """
    Convenience helper:
      - Build the IEEE-14 node-breaker net with default statuses.
      - Flip one CB (bad_cb_name) to OPEN (False) in the status_map.
      - Rebuild the NB net with that corrected status_map.
      - Topology-process to bus-branch.
      - Export to MATPOWER .mat file via export_to_matpower.

    Returns
    -------
    ppc : dict
        MATPOWER ppc/mpc struct for the corrected topology.
    net_bb : pandapowerNet
        Topology-processed bus-branch network.
    """

    nb = _import_nb_module()

    # 1) Build "truth" NB net and derive status_map for all CBs
    net_truth, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123()
    status_map = {name: bool(net_truth.impedance.at[idx, "in_service"]) for name, idx in cb_idx.items()}

    # In the corrected model, set this CB to the desired_status
    if bad_cb_name in status_map:
        status_map[bad_cb_name] = bool(desired_status)

    # 2) Rebuild NB net with this status_map
    net_nb, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123(status_map=status_map)

    # 3) NB → bus-branch
    net_bb, old2topo_bus = topology_processed_busbranch(net_nb)
    
    # 3b) Prune dangling buses (tail nodes with no injection)
    # This removes split-bus artifacts requested by user.
    _prune_dangling_buses(net_bb)

    # 4) Optionally run a PF on the bus-branch net to populate res_bus,
    #    which some pandapower routines expect when using init=\"results\".
    try:
        pp.runpp(net_bb, init="flat")
    except Exception as pf_err:  # pragma: no cover - diagnostic only
        # For export purposes we can proceed without successful PF;
        # to_mpc works structurally on net data.
        print(f"Warning: PF on topology-processed net failed: {pf_err}")

    # 5) Export bus-branch net to MATPOWER
    if hasattr(net_bb, "trafo") and len(net_bb.trafo):
        print("DEBUG: net_bb.trafo taps:")
        print(net_bb.trafo[['name', 'tap_pos', 'tap_step_percent', 'tap_side']])

    ppc = export_to_matpower(net_bb, filename_mat=filename_mat)
    # Be defensive about the returned structure when printing a summary.
    try:
        if isinstance(ppc, dict):
            n_bus = len(ppc.get("bus", []))
            n_branch = len(ppc.get("branch", []))
        else:
            n_bus = n_branch = "n/a"
        print(
            f"Exported corrected topology for {bad_cb_name} (desired_status={bool(desired_status)}) "
            f"to {filename_mat} with {n_bus} buses and {n_branch} branches."
        )
    except Exception as info_err:  # pragma: no cover - informational only
        print(
            f"Exported corrected topology for {bad_cb_name} (desired_status={bool(desired_status)}) "
            f"to {filename_mat} (could not summarize buses/branches: {info_err})"
        )
    return ppc, net_bb


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Export corrected NB topology to MATPOWER case")
    ap.add_argument(
        "--cb-name",
        type=str,
        required=True,
        help="Breaker name as in nodebreaker_pp14 (e.g., 'CB_2R3_2R4')",
    )
    ap.add_argument(
        "--desired-status",
        type=str,
        default="open",
        help="Target CB status: 'open' or 'closed' (default: open)",
    )
    ap.add_argument(
        "--out-mat",
        type=str,
        default="case14_topology.mat",
        help="Output MATPOWER .mat filename (default: case14_topology.mat)",
    )
    args = ap.parse_args()

    ds = str(args.desired_status).strip().lower()
    desired = False if ds in ("open", "false", "0") else True

    build_corrected_busbranch_and_export(args.cb_name, desired_status=desired, filename_mat=args.out_mat)
