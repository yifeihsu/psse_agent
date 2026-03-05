#!/usr/bin/env python3
"""
Run power flow for the IEEE 14-bus system and print results.

Defaults to the node-breaker pocket model (substations 1–3 detailed) so that
breaker flows are available in `res_impedance`. Use `--standard` to run the
classic bus-branch IEEE 14 case instead.

Usage examples:
  python Transmission/run_pf_ieee14.py            # node-breaker (with CB flows)
  python Transmission/run_pf_ieee14.py --standard # classic bus-branch
"""

from __future__ import annotations
import argparse
from typing import Optional


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


def run_nodebreaker_ieee14():
    import pandapower as pp
    nb = _import_nb_module()

    net, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123()
    # Robust init helps with tiny breaker impedances
    pp.runpp(net, init="dc")

    sn = float(net.sn_mva)
    print(f"=== IEEE-14 Power Flow (Node-Breaker pocket 1–3) | Sbase={sn:.3f} MVA ===")

    # Buses
    print("\n-- Bus Voltages / Injections --")
    for bidx, row in net.bus.iterrows():
        name = str(row.get("name", f"bus{bidx}"))
        vm = float(net.res_bus.at[bidx, "vm_pu"]) if bidx in net.res_bus.index else float('nan')
        va = float(net.res_bus.at[bidx, "va_degree"]) if bidx in net.res_bus.index else float('nan')
        p = float(net.res_bus.at[bidx, "p_mw"]) if bidx in net.res_bus.index else float('nan')
        q = float(net.res_bus.at[bidx, "q_mvar"]) if bidx in net.res_bus.index else float('nan')
        print(f"{name:>10s}  Vm={vm:6.4f} pu  Va={va:7.3f} deg  Pinj={p:9.4f} MW  Qinj={q:9.4f} MVAr")

    # Lines
    if len(net.line.index):
        print("\n-- Line Terminal Flows (from->to) --")
        for idx in net.line.index:
            fb = int(net.line.at[idx, "from_bus"]) ; tb = int(net.line.at[idx, "to_bus"]) 
            fn = str(net.bus.at[fb, "name"]) ; tn = str(net.bus.at[tb, "name"]) 
            p = float(net.res_line.at[idx, "p_from_mw"]) ; q = float(net.res_line.at[idx, "q_from_mvar"]) 
            i_ka = float(net.res_line.at[idx, "i_from_ka"]) 
            print(f"{fn:>10s} -> {tn:<10s}  P={p:9.4f} MW  Q={q:9.4f} MVAr  I={i_ka:7.4f} kA")

    # Transformers
    if len(net.trafo.index):
        print("\n-- Transformer HV-Side Flows (hv->lv) --")
        for idx in net.trafo.index:
            hv = int(net.trafo.at[idx, "hv_bus"]) ; lv = int(net.trafo.at[idx, "lv_bus"]) 
            hn = str(net.bus.at[hv, "name"]) ; ln = str(net.bus.at[lv, "name"]) 
            p = float(net.res_trafo.at[idx, "p_hv_mw"]) ; q = float(net.res_trafo.at[idx, "q_hv_mvar"]) 
            print(f"{hn:>10s} -> {ln:<10s}  P={p:9.4f} MW  Q={q:9.4f} MVAr")

    # Breakers (as impedances)
    if len(net.impedance.index):
        print("\n-- Breaker Flows (from->to) via res_impedance --")
        for idx in net.impedance.index:
            name = str(net.impedance.at[idx, "name"]) or f"imp_{idx}"
            fb = int(net.impedance.at[idx, "from_bus"]) ; tb = int(net.impedance.at[idx, "to_bus"]) 
            fn = str(net.bus.at[fb, "name"]) ; tn = str(net.bus.at[tb, "name"]) 
            in_service = bool(net.impedance.at[idx, "in_service"]) 
            if in_service:
                p = float(net.res_impedance.at[idx, "p_from_mw"]) ; q = float(net.res_impedance.at[idx, "q_from_mvar"]) 
            else:
                p = 0.0 ; q = 0.0
            state = "CLOSED" if in_service else "OPEN  "
            print(f"{name:>16s}  [{state}]  {fn:>10s} -> {tn:<10s}  Pcb={p:9.5f} MW  Qcb={q:9.5f} MVAr")

    return net


def run_standard_ieee14():
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case14()
    pp.runpp(net)
    sn = float(net.sn_mva)
    print(f"=== IEEE-14 Power Flow (Standard bus-branch) | Sbase={sn:.3f} MVA ===")

    # Buses
    print("\n-- Bus Voltages / Injections --")
    for bidx, row in net.bus.iterrows():
        name = str(row.get("name", f"bus{bidx}"))
        vm = float(net.res_bus.at[bidx, "vm_pu"]) if bidx in net.res_bus.index else float('nan')
        va = float(net.res_bus.at[bidx, "va_degree"]) if bidx in net.res_bus.index else float('nan')
        p = float(net.res_bus.at[bidx, "p_mw"]) if bidx in net.res_bus.index else float('nan')
        q = float(net.res_bus.at[bidx, "q_mvar"]) if bidx in net.res_bus.index else float('nan')
        print(f"{name:>10s}  Vm={vm:6.4f} pu  Va={va:7.3f} deg  Pinj={p:9.4f} MW  Qinj={q:9.4f} MVAr")

    # Lines
    if len(net.line.index):
        print("\n-- Line Terminal Flows (from->to) --")
        for idx in net.line.index:
            fb = int(net.line.at[idx, "from_bus"]) ; tb = int(net.line.at[idx, "to_bus"]) 
            fn = str(net.bus.at[fb, "name"]) ; tn = str(net.bus.at[tb, "name"]) 
            p = float(net.res_line.at[idx, "p_from_mw"]) ; q = float(net.res_line.at[idx, "q_from_mvar"]) 
            i_ka = float(net.res_line.at[idx, "i_from_ka"]) 
            print(f"{fn:>10s} -> {tn:<10s}  P={p:9.4f} MW  Q={q:9.4f} MVAr  I={i_ka:7.4f} kA")

    # Transformers
    if len(net.trafo.index):
        print("\n-- Transformer HV-Side Flows (hv->lv) --")
        for idx in net.trafo.index:
            hv = int(net.trafo.at[idx, "hv_bus"]) ; lv = int(net.trafo.at[idx, "lv_bus"]) 
            hn = str(net.bus.at[hv, "name"]) ; ln = str(net.bus.at[lv, "name"]) 
            p = float(net.res_trafo.at[idx, "p_hv_mw"]) ; q = float(net.res_trafo.at[idx, "q_hv_mvar"]) 
            print(f"{hn:>10s} -> {ln:<10s}  P={p:9.4f} MW  Q={q:9.4f} MVAr")

    # No breakers in standard case
    return net


def main(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser(description="Run IEEE-14 power flow and print results")
    ap.add_argument("--standard", action="store_true", help="Use classic bus-branch case14 instead of node-breaker")
    args = ap.parse_args(argv)

    if args.standard:
        run_standard_ieee14()
    else:
        run_nodebreaker_ieee14()


if __name__ == "__main__":
    main()

