import os

import opendssdirect as dss
from opendssdirect import Capacitors, Generators, Loads, Vsources


BUS_ORDER = [f"b{i}" for i in range(1, 15)]

# MATPOWER IEEE-14 branch order (20 branches). Used to align Pf/Qf/Pt/Qt indices with case14.
BRANCH_ORDER = [
    "Line.1-2",
    "Line.1-5",
    "Line.2-3",
    "Line.2-4",
    "Line.2-5",
    "Line.3-4",
    "Line.4-5",
    "Transformer.4-7",
    "Transformer.4-9",
    "Transformer.5-6",
    "Line.6-11",
    "Line.6-12",
    "Line.6-13",
    "Line.7-8",
    "Line.7-9",
    "Line.9-10",
    "Line.9-14",
    "Line.10-11",
    "Line.12-13",
    "Line.13-14",
]


def element_pq_3ph_per_terminal():
    """
    Return list of (P_MW, Q_Mvar) per terminal for the active CktElement,
    summing over all phases.
    """
    nph = dss.CktElement.NumPhases()
    nterm = dss.CktElement.NumTerminals()
    vals = dss.CktElement.Powers()  # kW, kvar per phase per terminal
    out = []
    for t in range(nterm):
        s = 2 * nph * t
        e = s + 2 * nph
        seg = vals[s:e]
        p_kw = sum(seg[0::2])
        q_kvar = sum(seg[1::2])
        out.append((p_kw / 1000.0, q_kvar / 1000.0))  # MW, Mvar
    return out


def _compile_and_solve(repo_dir: str, *, load_mult: float = 1.0) -> None:
    dss.Basic.DataPath(repo_dir)
    dss.Text.Command("Clear")
    dss.Text.Command("Redirect Run_IEEE14Bus.dss")
    if load_mult != 1.0:
        dss.Text.Command(f"Set LoadMult={float(load_mult)}")
    dss.Text.Command("Solve")


def _bus_vmagangle_pu(bus: str):
    """Return (v_pu[3], ang_deg[3], kvbase_ln) for bus phases 1/2/3 (best-effort)."""
    dss.Circuit.SetActiveBus(bus)
    magang = dss.Bus.VMagAngle() or []
    kvbase = float(dss.Bus.kVBase() or 0.0)  # LN base in kV

    v_pu = []
    ang_deg = []
    for ph in range(3):
        mag_v = magang[2 * ph] if len(magang) > 2 * ph else None
        ang = magang[2 * ph + 1] if len(magang) > 2 * ph + 1 else None
        if mag_v is None or kvbase <= 0:
            v_pu.append(None)
        else:
            v_pu.append((float(mag_v) / 1000.0) / kvbase)  # V -> kV -> pu
        ang_deg.append(float(ang) if ang is not None else None)

    return v_pu, ang_deg, kvbase


def extract_three_phase_voltage_measurements(buses=None):
    """
    Extract per-bus 3ϕ VLN voltage magnitudes/angles in per-unit.

    Returns a list aligned to `buses` with entries:
      {bus: str, kvbase_ln: float, vln_pu: [Va,Vb,Vc], ang_deg: [Aa,Ab,Ac]}
    """
    buses = BUS_ORDER if buses is None else list(buses)
    out = []
    for b in buses:
        v_pu, ang_deg, kvbase = _bus_vmagangle_pu(b)
        out.append(
            dict(
                bus=str(b),
                kvbase_ln=float(kvbase),
                vln_pu=v_pu,
                ang_deg=ang_deg,
            )
        )
    return out


def extract_measurement_series(*, buses=None, branch_names=None):
    """
    Extract the 1ϕ-equivalent (phase-A) operator measurement vector from the *currently solved* circuit.

    Layout:
      [Vm(1..nb), Pinj(1..nb), Qinj(1..nb), Pf(1..nl), Qf(1..nl), Pt(1..nl), Qt(1..nl)]

    - Vm uses phase-1 (phase A) VLN magnitude per bus.
    - Pinj/Qinj follow MATPOWER makeSbus convention in per-unit on 100 MVA.
    - Branch flows use BRANCH_ORDER to match MATPOWER case14 branch rows.
    """
    buses = BUS_ORDER if buses is None else list(buses)
    branch_names = BRANCH_ORDER if branch_names is None else list(branch_names)

    MVA_BASE = 100.0

    # Bus injections (MW/Mvar), MATPOWER convention:
    #   Sbus = (Pg - Pl) + j(Qg - Ql) on baseMVA
    # OpenDSS CktElement.Powers() reports power INTO the element from each terminal.
    # For bus injections into the network, we flip sign: P_inj_bus += -P_into_element.
    P_inj = {b: 0.0 for b in buses}
    Q_inj = {b: 0.0 for b in buses}

    # Loads
    for name in Loads.AllNames() or []:
        Loads.Name(name)
        buses_el = dss.CktElement.BusNames()
        bus = buses_el[0].split(".")[0].lower()
        pqs = element_pq_3ph_per_terminal()
        p, q = pqs[0] if pqs else (0.0, 0.0)
        if bus in P_inj:
            P_inj[bus] += -p
            Q_inj[bus] += -q

    # Generators
    for name in Generators.AllNames() or []:
        Generators.Name(name)
        buses_el = dss.CktElement.BusNames()
        bus = buses_el[0].split(".")[0].lower()
        pqs = element_pq_3ph_per_terminal()
        p, q = pqs[0] if pqs else (0.0, 0.0)
        if bus in P_inj:
            P_inj[bus] += -p
            Q_inj[bus] += -q

    # Slack source(s)
    for name in Vsources.AllNames() or []:
        Vsources.Name(name)
        buses_el = dss.CktElement.BusNames()
        bus = buses_el[0].split(".")[0].lower()
        pqs = element_pq_3ph_per_terminal()
        p, q = pqs[0] if pqs else (0.0, 0.0)
        if bus in P_inj:
            P_inj[bus] += -p
            Q_inj[bus] += -q

    # Capacitors (reactive injections)
    for name in Capacitors.AllNames() or []:
        Capacitors.Name(name)
        buses_el = dss.CktElement.BusNames()
        bus = buses_el[0].split(".")[0].lower()
        pqs = element_pq_3ph_per_terminal()
        p, q = pqs[0] if pqs else (0.0, 0.0)
        if bus in P_inj:
            P_inj[bus] += -p
            Q_inj[bus] += -q

    # Bus voltages: phase-1 VLN, per-unit
    Vm = []
    for b in buses:
        v_pu, _, _kvbase = _bus_vmagangle_pu(b)
        Vm.append(v_pu[0] if v_pu and v_pu[0] is not None else 0.0)

    # Branch flows (lines + transformers), MW/Mvar
    Pf = []
    Qf = []
    Pt = []
    Qt = []

    for elem in branch_names:
        dss.Circuit.SetActiveElement(elem)
        pqs = element_pq_3ph_per_terminal()
        # from terminal = 0, to terminal = 1 (if present)
        p_from, q_from = (pqs[0] if len(pqs) > 0 else (0.0, 0.0))
        p_to, q_to = (pqs[1] if len(pqs) > 1 else (0.0, 0.0))
        Pf.append(p_from)
        Qf.append(q_from)
        Pt.append(p_to)
        Qt.append(q_to)

    # Convert to per-unit on MVA base
    Vm_pu = Vm
    Pinj_pu = [P_inj[b] / MVA_BASE for b in buses]
    Qinj_pu = [Q_inj[b] / MVA_BASE for b in buses]
    Pf_pu = [p / MVA_BASE for p in Pf]
    Qf_pu = [q / MVA_BASE for q in Qf]
    Pt_pu = [p / MVA_BASE for p in Pt]
    Qt_pu = [q / MVA_BASE for q in Qt]

    series = Vm_pu + Pinj_pu + Qinj_pu + Pf_pu + Qf_pu + Pt_pu + Qt_pu
    return series, buses, branch_names


def build_measurement_series():
    """
    Build flattened per-unit measurement vector in the order:

    [Vm(1..nb),
     Pinj(1..nb),
     Qinj(1..nb),
     Pf(1..nl),
     Qf(1..nl),
     Pt(1..nl),
     Qt(1..nl)]
    """
    repo = os.path.dirname(os.path.abspath(__file__))
    _compile_and_solve(repo)
    return extract_measurement_series()


def main():
    series, buses, branches = build_measurement_series()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurement_series_bus3_pu.txt")
    with open(out_path, "w") as f:
        f.write("# Flattened per-unit measurement series for IEEE-14 with unbalanced Bus 3\n")
        f.write("# Order: Vm(1..nb), Pinj(1..nb), Qinj(1..nb), Pf(1..nl), Qf(1..nl), Pt(1..nl), Qt(1..nl)\n")
        f.write(f"# nb={len(buses)}, nl={len(branches)}\n")
        f.write("# Branch order for Pf/Qf/Pt/Qt:\n")
        for i, name in enumerate(branches):
            f.write(f"#   {i}: {name}\n")
        f.write("# Data (one value per line, index 0-based):\n")
        for val in series:
            f.write(f"{val:.9f}\n")


if __name__ == "__main__":
    main()
