#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node-breaker IEEE-14 with *full* models of substations 1, 2, 3 (zoomed pocket).
- Every bus-section is a bus.
- Every CB is an 'impedance' element with tiny pu impedance when CLOSED
  (so PF reports Pcb/Qcb in net.res_impedance); OPEN CBs are set out_of_service.
- All other substations (4..14) stay bus-branch for simplicity.

References:
  • Detailed substation layout (Fig. 3.28) & GSE measurement equations (Eqs. 3.76–3.84).  # report Sec. 3.4
  • Closed = solid square; open = hollow; thick = lines; thin = breaker connections.  # see RG figure caption
  • pandapower: bus-bus switch fuses buses; use 'impedance' to read flows. 
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---- Substation description -----------------------------------------------------------

@dataclass
class CB:
    name: str
    a: str
    b: str
    closed: bool = True   # default truth (you can override via status_map)

@dataclass
class SubstationNB:
    busnum: int
    # section buses inside the yard (busbars / ring nodes)
    sections: List[str]
    # which section buses are used as "injection connection points"
    inj_sections: List[str]
    # mapping: planning neighbor bus -> line-terminal *bay* bus on this side
    # (each physical branch end lands on a dedicated bay bus)
    bays: Dict[int, str]
    # circuit-breakers (CBs) inside the yard
    cbs: List[CB]

def detailed_substations_1_2_3() -> Dict[int, SubstationNB]:
    """
    Pocket (1,2,3) mapped to Fig. 3.28:
      • Bus 1: breaker-and-a-half; left stack is an injection bay; right stack hosts lines 1–5 and 1–2;
        center breaker between 1N3–1N4 is OPEN.
      • Bus 2: five-breaker ring; 2R1→Bus1, 2R2→Bus5, 2R3→Bus4, 2R4→Bus3; 2R5 is local injection;
        ring breaker 2R1–2R2 is OPEN, others CLOSED.
      • Bus 3: double-bus single-breaker; 3–2 tied to 3B2 only; 3–4 tied to 3B1 only.
    """
    S: Dict[int, SubstationNB] = {}

    # -------- Bus 1 (Breaker-and-a-half, two bays) --------
    # Busbars + intermediate nodes for the two bays
    sec1 = ["1B1", "1B2", "1N1", "1N2", "1N3", "1N4"]

    # External lines land on the right stack nodes
    bays1 = {
        5: "1N3",  # 1–5 connects at 1N3
        2: "1N4",  # 1–2 connects at 1N4
    }

    # CBs: left bay (injection) and right bay (two lines share the middle breaker)
    cbs1 = [
        # Left stack = injection bay (all CLOSED)
        CB("CB_1_B1_N1", "1B1", "1N1", True),
        CB("CB_1_N1_N2", "1N1", "1N2", True),
        CB("CB_1_N2_B2", "1N2", "1B2", True),
        # Right stack = two line bays; center breaker OPEN (hollow square in the figure)
        CB("CB_1_B1_N3", "1B1", "1N3", True),
        CB("CB_1_N3_N4", "1N3", "1N4", False),  # OPEN
        CB("CB_1_N4_B2", "1N4", "1B2", True),
    ]

    # Injection is applied at the injection bay; split evenly across 1N1 & 1N2
    S[1] = SubstationNB(1, sec1, inj_sections=["1N1", "1N2"], bays=bays1, cbs=cbs1)

    # -------- Bus 2 (Five-breaker Ring Bus) --------
    sec2 = ["2R1", "2R2", "2R3", "2R4", "2R5"]  # ring nodes (2R5 hosts the local load)
    bays2 = {
        1: "2R1",
        5: "2R2",
        4: "2R3",
        3: "2R4",
    }
    cbs2 = [
        CB("CB_2R1_2R2", "2R1", "2R2", False),  # OPEN link
        CB("CB_2R2_2R3", "2R2", "2R3", True),
        CB("CB_2R3_2R4", "2R3", "2R4", True),
        CB("CB_2R4_2R5", "2R4", "2R5", True),
        CB("CB_2R5_2R1", "2R5", "2R1", True),
    ]
    S[2] = SubstationNB(2, sec2, inj_sections=["2R5"], bays=bays2, cbs=cbs2)

    # -------- Bus 3 (Double-bus, single-breaker per bay) --------
    sec3 = ["3B1", "3B2"]
    bays3 = {
        2: "3|L32",
        4: "3|L34",
    }
    cbs3 = [
        # Line 3–2 tied to 3B2 only
        CB("CB_3_L32_B1", "3|L32", "3B1", False),  # OPEN
        CB("CB_3_L32_B2", "3|L32", "3B2", True),   # CLOSED

        CB("CB_3_L34_B1", "3|L34", "3B1", True),   # CLOSED
        CB("CB_3_L34_B2", "3|L34", "3B2", True),  # CLOSED
    ]
    S[3] = SubstationNB(3, sec3, inj_sections=["3B1", "3B2"], bays=bays3, cbs=cbs3)

    return S


# ---- Pandapower node-breaker builder --------------------------------------------------

def build_nb_ieee14_pocket123(status_map: Optional[Dict[str,bool]] = None,
                              zcb_r_pu: float = 5e-6,
                              zcb_x_pu: float = 5e-5):
    """
    Create node-breaker pandapower net:
      - NB for substations 1,2,3 as above; the rest are single-bus.
      - CBs are 'impedance' elements so PF gives Pcb/Qcb (pu).
    Returns:
      net, sec_bus (section/bay -> bus idx), cb_idx (CB name -> impedance idx),
      line_idx map {(name)->idx}, trafo_idx map {(name)->idx}
    """
    import pandapower as pp
    import pandapower.networks as pn

    pocket = detailed_substations_1_2_3()
    pocket_busnums = set(pocket.keys())

    ref = pn.case14()
    sn = float(ref.sn_mva)

    # map original "planning" bus numbers to pp bus indices
    if "name" in ref.bus.columns:
        busnum_to_pp = {}
        for idx, nm in ref.bus["name"].items():
            try:
                num = int(str(nm).split()[-1])
            except Exception:
                num = idx+1
            busnum_to_pp[num] = idx
    else:
        busnum_to_pp = {i+1:i for i in range(len(ref.bus))}

    pp2num = {v:k for k,v in busnum_to_pp.items()}

    # new empty net
    net = pp.create_empty_network(sn_mva=sn, f_hz=getattr(ref,"f_hz", 60.0))

    # --- Create buses ---
    sec_bus: Dict[str,int] = {}        # all section & bay buses for pocket
    plan_bus: Dict[int,int] = {}       # single planning-bus for non-pocket buses

    # 4..14 except pocket: collapsed single bus per planning bus
    for b in range(1,15):
        if b in pocket_busnums:  # pocket handled below
            continue
        ppidx = busnum_to_pp[b]
        plan_bus[b] = pp.create_bus(net, vn_kv=float(ref.bus.at[ppidx,"vn_kv"]), name=f"B{b}")

    # Pocket stations (1,2,3): create section buses & bay buses
    for b, ss in pocket.items():
        ppidx = busnum_to_pp[b]
        vn = float(ref.bus.at[ppidx, "vn_kv"])
        # sections
        for s in ss.sections:
            sec_bus[s] = pp.create_bus(net, vn_kv=vn, name=s)
        # bay terminal buses (only if not already created as a section)
        for nb, bay in ss.bays.items():
            if bay not in sec_bus:
                sec_bus[bay] = pp.create_bus(net, vn_kv=vn, name=bay)


    # --- Copy injections (ext_grid, gens, loads, shunts) with splitting over inj sections ---
    # helper to split a scalar equally over k>0 recipients
    def split(val, n):
        if n<=0: return []
        base = val / n
        # keep totals exact
        return [base]*(n-1) + [val - base*(n-1)]

    # ext_grid (keep a single slack per original ext_grid)
    if hasattr(ref,"ext_grid"):
        for i in ref.ext_grid.index:
            bnum = pp2num[int(ref.ext_grid.at[i,"bus"])]
            target = pocket.get(bnum)
            if target is None:
                pp.create_ext_grid(net, bus=plan_bus[bnum],
                                   vm_pu=float(ref.ext_grid.at[i,"vm_pu"]),
                                   va_degree=float(ref.ext_grid.at[i,"va_degree"]))
            else:
                # put slack at the *first* injection section
                s0 = target.inj_sections[0]
                pp.create_ext_grid(net, bus=sec_bus[s0],
                                   vm_pu=float(ref.ext_grid.at[i,"vm_pu"]),
                                   va_degree=float(ref.ext_grid.at[i,"va_degree"]))

    # gens
    if hasattr(ref,"gen"):
        for i in ref.gen.index:
            bnum = pp2num[int(ref.gen.at[i,"bus"])]
            p = float(ref.gen.at[i,"p_mw"])
            vm = float(ref.gen.at[i,"vm_pu"])
            if bnum in pocket_busnums:
                inj = pocket[bnum].inj_sections
                parts = split(p, len(inj))
                for p_i, s in zip(parts, inj):
                    # place multiple gen units to split P setpoint
                    pp.create_gen(net, bus=sec_bus[s], p_mw=p_i, vm_pu=vm, name=f"gen{bnum}@{s}")
            else:
                pp.create_gen(net, bus=plan_bus[bnum], p_mw=p, vm_pu=vm, name=f"gen{bnum}")

    # sgens (if any)
    if hasattr(ref,"sgen"):
        for i in ref.sgen.index:
            bnum = pp2num[int(ref.sgen.at[i,"bus"])]
            p = float(ref.sgen.at[i,"p_mw"]); q = float(ref.sgen.at[i,"q_mvar"])
            if bnum in pocket_busnums:
                inj = pocket[bnum].inj_sections
                parts_p = split(p, len(inj)); parts_q = split(q, len(inj))
                for p_i,q_i,s in zip(parts_p, parts_q, inj):
                    pp.create_sgen(net, bus=sec_bus[s], p_mw=p_i, q_mvar=q_i, name=f"sgen{bnum}@{s}")
            else:
                pp.create_sgen(net, bus=plan_bus[bnum], p_mw=p, q_mvar=q, name=f"sgen{bnum}")

    # loads
    if hasattr(ref,"load"):
        for i in ref.load.index:
            bnum = pp2num[int(ref.load.at[i,"bus"])]
            p = float(ref.load.at[i,"p_mw"]); q = float(ref.load.at[i,"q_mvar"])
            if bnum in pocket_busnums:
                inj = pocket[bnum].inj_sections
                parts_p = split(p, len(inj)); parts_q = split(q, len(inj))
                for p_i,q_i,s in zip(parts_p, parts_q, inj):
                    pp.create_load(net, bus=sec_bus[s], p_mw=p_i, q_mvar=q_i, name=f"load{bnum}@{s}")
            else:
                pp.create_load(net, bus=plan_bus[bnum], p_mw=p, q_mvar=q, name=f"load{bnum}")

    # shunts
    if hasattr(ref,"shunt"):
        for i in ref.shunt.index:
            bnum = pp2num[int(ref.shunt.at[i,"bus"])]
            q = float(ref.shunt.at[i,"q_mvar"]); p = float(ref.shunt.at[i,"p_mw"])
            if bnum in pocket_busnums:
                inj = pocket[bnum].inj_sections
                parts_q = split(q, len(inj)); parts_p = split(p, len(inj))
                for p_i,q_i,s in zip(parts_p, parts_q, inj):
                    pp.create_shunt(net, bus=sec_bus[s], q_mvar=q_i, p_mw=p_i)
            else:
                pp.create_shunt(net, bus=plan_bus[bnum], q_mvar=q, p_mw=p)

    # --- Breakers as impedance elements (so PF yields Pcb/Qcb) ---
    cb_idx: Dict[str,int] = {}
    def add_cb(name, a, b, closed=True):
        val = bool(closed if status_map is None else status_map.get(name, closed))
        idx = pp.create_impedance(net, from_bus=sec_bus[a], to_bus=sec_bus[b],
                                  rft_pu=zcb_r_pu, xft_pu=zcb_x_pu, sn_mva=sn,
                                  rtf_pu=zcb_r_pu, xtf_pu=zcb_x_pu,
                                  name=name, in_service=val)
        cb_idx[name] = idx

    for b, ss in pocket.items():
        for cb in ss.cbs:
            add_cb(cb.name, cb.a, cb.b, cb.closed)

    # --- Lines / transformers: connect bay buses for pocket ends, otherwise planning buses ---
    line_idx: Dict[str,int] = {}
    trafo_idx: Dict[str,int] = {}

    # helper: return the connection bus index for an end (planning bus 'b', neighbor 'nb')
    def end_bus(b: int, nb: int) -> int:
        ss = pocket.get(b)
        if ss is None:
            return plan_bus[b]
        # pocket bus -> use the *bay* for that neighbor
        bay = ss.bays[nb]
        return sec_bus[bay]

    # Lines
    if hasattr(ref,"line"):
        for li in ref.line.index:
            fb_pp = int(ref.line.at[li,"from_bus"]); tb_pp = int(ref.line.at[li,"to_bus"])
            fb = pp2num[fb_pp]; tb = pp2num[tb_pp]
            fbus = end_bus(fb, tb); tbus = end_bus(tb, fb)
            idx = pp.create_line_from_parameters(
                net, from_bus=fbus, to_bus=tbus,
                length_km=float(ref.line.at[li,"length_km"]),
                r_ohm_per_km=float(ref.line.at[li,"r_ohm_per_km"]),
                x_ohm_per_km=float(ref.line.at[li,"x_ohm_per_km"]),
                c_nf_per_km=float(ref.line.at[li,"c_nf_per_km"]),
                max_i_ka=float(ref.line.at[li,"max_i_ka"]),
                name=f"line_{fb}-{tb}"
            )
            line_idx[f"line_{fb}-{tb}"] = idx

    # Transformers
    if hasattr(ref,"trafo"):
        for ti in ref.trafo.index:
            hv_pp = int(ref.trafo.at[ti,"hv_bus"]); lv_pp = int(ref.trafo.at[ti,"lv_bus"])
            hv = pp2num[hv_pp]; lv = pp2num[lv_pp]
            hbus = end_bus(hv, lv); lbus = end_bus(lv, hv)
            # Check for tap parameters
            tap_pos = float(ref.trafo.at[ti, "tap_pos"]) if "tap_pos" in ref.trafo.columns and not np.isnan(ref.trafo.at[ti, "tap_pos"]) else np.nan
            tap_step_percent = float(ref.trafo.at[ti, "tap_step_percent"]) if "tap_step_percent" in ref.trafo.columns and not np.isnan(ref.trafo.at[ti, "tap_step_percent"]) else np.nan
            tap_step_degree = float(ref.trafo.at[ti, "tap_step_degree"]) if "tap_step_degree" in ref.trafo.columns and not np.isnan(ref.trafo.at[ti, "tap_step_degree"]) else np.nan
            tap_side = str(ref.trafo.at[ti, "tap_side"]) if "tap_side" in ref.trafo.columns else None

            idx = pp.create_transformer_from_parameters(
                net, hv_bus=hbus, lv_bus=lbus,
                sn_mva=float(ref.trafo.at[ti,"sn_mva"]),
                vn_hv_kv=float(ref.trafo.at[ti,"vn_hv_kv"]),
                vn_lv_kv=float(ref.trafo.at[ti,"vn_lv_kv"]),
                vkr_percent=float(ref.trafo.at[ti,"vkr_percent"]),
                vk_percent=float(ref.trafo.at[ti,"vk_percent"]),
                pfe_kw=float(ref.trafo.at[ti,"pfe_kw"]),
                i0_percent=float(ref.trafo.at[ti,"i0_percent"]),
                shift_degree=float(ref.trafo.at[ti,"shift_degree"]) if "shift_degree" in ref.trafo.columns else 0.0,
                tap_pos=tap_pos,
                tap_step_percent=tap_step_percent,
                tap_step_degree=tap_step_degree,
                tap_side="lv",
                name=f"trafo_{hv}-{lv}"
            )
            trafo_idx[f"trafo_{hv}-{lv}"] = idx

    return net, sec_bus, cb_idx, line_idx, trafo_idx

# ---- Measurement extraction -----------------------------------------------------------

def run_pf_and_measure(net, sec_bus, cb_idx, line_idx, trafo_idx,
                       with_angles: bool = False,
                       seed: int = 1,
                       sigma: Optional[Dict[str,float]] = None):
    """Return SCADA-like measurements incl. CB flows (pu)."""
    import pandapower as pp
    rng = np.random.default_rng(seed)
    if sigma is None:
        sigma = {"Vmag":0.005, "Vang":0.002, "Pinj":0.01, "Qinj":0.01,
                 "Pij":0.01, "Qij":0.01, "Pcb":0.01, "Qcb":0.01}
    # Use a robust initialization to avoid divergence with near-zero CB impedances
    pp.runpp(net, init="dc")
    sn = float(net.sn_mva)
    meas = []

    # Vmag / Vang and injections at every (section or bay) bus (you can filter later)
    for name, bidx in sec_bus.items():
        vm = float(net.res_bus.at[bidx,"vm_pu"])
        va = float(net.res_bus.at[bidx,"va_degree"])
        p  = float(net.res_bus.at[bidx,"p_mw"])/sn
        q  = float(net.res_bus.at[bidx,"q_mvar"])/sn
        meas.append(("Vmag", name, vm + rng.normal(0,sigma["Vmag"]), sigma["Vmag"]))
        if with_angles:
            meas.append(("Vang", name, np.deg2rad(va) + rng.normal(0,sigma["Vang"]), sigma["Vang"]))
        meas.append(("Pinj", name, p + rng.normal(0,sigma["Pinj"]), sigma["Pinj"]))
        meas.append(("Qinj", name, q + rng.normal(0,sigma["Qinj"]), sigma["Qinj"]))

    # Line terminal flows
    for nm, idx in line_idx.items():
        pf = float(net.res_line.at[idx,"p_from_mw"])/sn
        qf = float(net.res_line.at[idx,"q_from_mvar"])/sn
        fb = int(net.line.at[idx,"from_bus"]); tb = int(net.line.at[idx,"to_bus"])
        name = f"{net.bus.at[fb,'name']}->{net.bus.at[tb,'name']}"
        meas.append(("Pij", name, pf + rng.normal(0,sigma["Pij"]), sigma["Pij"]))
        meas.append(("Qij", name, qf + rng.normal(0,sigma["Qij"]), sigma["Qij"]))

    # Trafo HV terminal flows (oriented)
    for nm, idx in trafo_idx.items():
        pf = float(net.res_trafo.at[idx,"p_hv_mw"])/sn
        qf = float(net.res_trafo.at[idx,"q_hv_mvar"])/sn
        hv = int(net.trafo.at[idx,"hv_bus"]); lv = int(net.trafo.at[idx,"lv_bus"])
        name = f"{net.bus.at[hv,'name']}->{net.bus.at[lv,'name']}"
        meas.append(("Pij", name, pf + rng.normal(0,sigma["Pij"]), sigma["Pij"]))
        meas.append(("Qij", name, qf + rng.normal(0,sigma["Qij"]), sigma["Qij"]))

    # CB flows from res_impedance (exactly your Pcb/Qcb)
    for cbname, idx in cb_idx.items():
        if bool(net.impedance.at[idx,"in_service"]):
            pf = float(net.res_impedance.at[idx,"p_from_mw"])/sn
            qf = float(net.res_impedance.at[idx,"q_from_mvar"])/sn
        else:
            pf = 0.0; qf = 0.0
        meas.append(("Pcb", cbname, pf + rng.normal(0,sigma["Pcb"]), sigma["Pcb"]))
        meas.append(("Qcb", cbname, qf + rng.normal(0,sigma["Qcb"]), sigma["Qcb"]))

    return meas

# ---- Quick demo ----------------------------------------------------------------------

def _demo():
    net, sec_bus, cb_idx, line_idx, trafo_idx = build_nb_ieee14_pocket123()
    meas = run_pf_and_measure(net, sec_bus, cb_idx, line_idx, trafo_idx, with_angles=False)
    # print a few CBs
    print("Sample CB P/Q (pu):")
    for t,name,val,_ in meas:
        if t=="Pcb":
            q = next(v for (tt,nn,v,_) in meas if tt=="Qcb" and nn==name)
            print(f"  {name:>14s}: Pcb={val:+.4f}  Qcb={q:+.4f}")

if __name__ == "__main__":
    _demo()
