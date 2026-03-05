import os
from typing import List, Tuple

import numpy as np
import pandapower as pp
import pandapower.networks as pn


def build_pandapower_measurement_series() -> Tuple[List[float], List[str], List[Tuple[str, Tuple[int, int]]]]:
    """
    Build flattened per-unit measurement vector for the balanced IEEE 14-bus case
    using pandapower, mimicking the MATPOWER script:

      - Vm   = results.bus(:, VM)
      - Pinj = real(makeSbus(baseMVA, bus, gen))
      - Qinj = imag(makeSbus(baseMVA, bus, gen))
      - Pf,Qf,Pt,Qt from branch PF/QF/PT/QT, all in per-unit on baseMVA

    The layout is:

      [Vm(1..nb),
       Pinj(1..nb),
       Qinj(1..nb),
       Pf(1..nl),
       Qf(1..nl),
       Pt(1..nl),
       Qt(1..nl)]
    """
    net = pn.case14()
    pp.runpp(net)

    base_mva = float(net.sn_mva)  # 100 MVA for IEEE 14
    nb = len(net.bus)

    # Bus order is 1..14, corresponding to pandapower bus indices 0..13
    buses = [f"bus_{i}" for i in range(1, nb + 1)]

    # --- Vm: directly from pandapower results (already per-unit) ---
    Vm = net.res_bus.vm_pu.to_numpy()

    # --- Pinj, Qinj: emulate MATPOWER makeSbus(baseMVA, bus, gen) ---
    # Sbus = (Pg - Pl) + j(Qg - Ql) in per-unit
    Pgen = np.zeros(nb)
    Qgen = np.zeros(nb)
    Pload = np.zeros(nb)
    Qload = np.zeros(nb)

    # Loads: use original P/Q from net.load (fixed demands)
    for _, row in net.load.iterrows():
        b = int(row.bus)
        Pload[b] += float(row.p_mw)
        Qload[b] += float(row.q_mvar)

    # Generators: use solved outputs from res_gen
    for i, row in net.gen.iterrows():
        b = int(row.bus)
        Pgen[b] += float(net.res_gen.p_mw.iloc[i])
        Qgen[b] += float(net.res_gen.q_mvar.iloc[i])

    # Slack/ext_grid: treat as a generator at its bus
    for i, row in net.ext_grid.iterrows():
        b = int(row.bus)
        Pgen[b] += float(net.res_ext_grid.p_mw.iloc[i])
        Qgen[b] += float(net.res_ext_grid.q_mvar.iloc[i])

    Pinj = (Pgen - Pload) / base_mva
    Qinj = (Qgen - Qload) / base_mva

    # --- Branch flows: enforce MATPOWER IEEE 14 branch order ---
    # MATPOWER case14 branch order (fbus, tbus), 1-based bus indices:
    branch_order: List[Tuple[str, Tuple[int, int]]] = [
        ("line", (1, 2)),
        ("line", (1, 5)),
        ("line", (2, 3)),
        ("line", (2, 4)),
        ("line", (2, 5)),
        ("line", (3, 4)),
        ("line", (4, 5)),
        ("trafo", (4, 7)),
        ("trafo", (4, 9)),
        ("trafo", (5, 6)),
        ("line", (6, 11)),
        ("line", (6, 12)),
        ("line", (6, 13)),
        ("trafo", (7, 8)),
        ("trafo", (7, 9)),
        ("line", (9, 10)),
        ("line", (9, 14)),
        ("line", (10, 11)),
        ("line", (12, 13)),
        ("line", (13, 14)),
    ]

    # Build lookups from (bus_from, bus_to) to pandapower line/trafo index
    line_pair_to_idx = {}
    for i, row in net.line.iterrows():
        fb = int(row.from_bus) + 1
        tb = int(row.to_bus) + 1
        line_pair_to_idx[(fb, tb)] = i

    trafo_pair_to_idx = {}
    for i, row in net.trafo.iterrows():
        hb = int(row.hv_bus) + 1
        lb = int(row.lv_bus) + 1
        trafo_pair_to_idx[(hb, lb)] = i

    Pf = []
    Qf = []
    Pt = []
    Qt = []

    for kind, (fb, tb) in branch_order:
        if kind == "line":
            idx = line_pair_to_idx[(fb, tb)]
            r = net.res_line.iloc[idx]
            Pf.append(float(r.p_from_mw) / base_mva)
            Qf.append(float(r.q_from_mvar) / base_mva)
            Pt.append(float(r.p_to_mw) / base_mva)
            Qt.append(float(r.q_to_mvar) / base_mva)
        else:
            idx = trafo_pair_to_idx[(fb, tb)]
            r = net.res_trafo.iloc[idx]
            # Treat "from" as HV side, "to" as LV side
            Pf.append(float(r.p_hv_mw) / base_mva)
            Qf.append(float(r.q_hv_mvar) / base_mva)
            Pt.append(float(r.p_lv_mw) / base_mva)
            Qt.append(float(r.q_lv_mvar) / base_mva)

    series = np.concatenate([Vm, Pinj, Qinj, Pf, Qf, Pt, Qt])
    return series.tolist(), buses, branch_order


def main() -> None:
    series, buses, branches = build_pandapower_measurement_series()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "measurement_series_balanced_pu.txt",
    )
    with open(out_path, "w") as f:
        f.write("# Flattened per-unit measurement series for IEEE-14 balanced case (pandapower)\n")
        f.write("# Order: Vm(1..nb), Pinj(1..nb), Qinj(1..nb), Pf(1..nl), Qf(1..nl), Pt(1..nl), Qt(1..nl)\n")
        f.write(f"# nb={len(buses)}, nl={len(branches)}\n")
        f.write("# Branch order for Pf/Qf/Pt/Qt (type, from_bus, to_bus):\n")
        for i, (kind, (fb, tb)) in enumerate(branches):
            f.write(f"#   {i}: {kind} {fb}->{tb}\n")
        f.write("# Data (one value per line, index 0-based):\n")
        for val in series:
            f.write(f"{val:.9f}\n")


if __name__ == "__main__":
    main()

