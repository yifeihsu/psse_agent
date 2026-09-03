import os

import numpy as np
import opendssdirect as dss
from opendssdirect import Capacitors, Generators, Loads, Vsources

from IEEE_14_OpenDSS.constants import BRANCH_ORDER, BUS_ORDER


def _phase_terminal_complex_powers(
    powers,
    node_order,
    *,
    n_conductors,
    n_terminals,
    phase_nodes=(1, 2, 3),
):
    """Split an OpenDSS power array by terminal and sum phase conductors."""
    n_conductors = int(n_conductors)
    n_terminals = int(n_terminals)
    if n_conductors < 1 or n_terminals < 1:
        raise ValueError("n_conductors and n_terminals must be positive")

    raw = np.asarray(powers, dtype=float).reshape(-1)
    if raw.size % 2:
        raise ValueError(f"Unexpected odd power-array size: {raw.size}")
    complex_powers = raw[0::2] + 1j * raw[1::2]
    expected = n_conductors * n_terminals
    if complex_powers.size != expected:
        raise ValueError(
            f"Unexpected power-array size: got {complex_powers.size}, expected {expected}"
        )

    nodes = [int(node) for node in node_order]
    if len(nodes) != expected:
        raise ValueError(f"Unexpected NodeOrder size: got {len(nodes)}, expected {expected}")

    selected_nodes = {int(node) for node in phase_nodes}
    terminal_powers = []
    for terminal in range(n_terminals):
        start = terminal * n_conductors
        stop = start + n_conductors
        terminal_values = complex_powers[start:stop]
        terminal_nodes = nodes[start:stop]
        phase_values = [
            value
            for value, node in zip(terminal_values, terminal_nodes)
            if node in selected_nodes
        ]
        terminal_powers.append(sum(phase_values, 0j))
    return terminal_powers


def _phase_terminal_complex_currents(
    currents,
    node_order,
    *,
    n_conductors,
    n_terminals,
    phase_nodes=(1, 2, 3),
):
    """Split an OpenDSS current array by terminal and order phase conductors.

    Returns one ``[Ia, Ib, Ic]`` list of complex amps per terminal.  Conductors
    are matched by node number so four-conductor transformer terminals drop
    the neutral, and a missing phase node yields ``0j`` rather than shifting
    the remaining phases.
    """
    n_conductors = int(n_conductors)
    n_terminals = int(n_terminals)
    if n_conductors < 1 or n_terminals < 1:
        raise ValueError("n_conductors and n_terminals must be positive")

    raw = np.asarray(currents, dtype=float).reshape(-1)
    if raw.size % 2:
        raise ValueError(f"Unexpected odd current-array size: {raw.size}")
    complex_currents = raw[0::2] + 1j * raw[1::2]
    expected = n_conductors * n_terminals
    if complex_currents.size != expected:
        raise ValueError(
            f"Unexpected current-array size: got {complex_currents.size}, expected {expected}"
        )
    nodes = [int(node) for node in node_order]
    if len(nodes) != expected:
        raise ValueError(f"Unexpected NodeOrder size: got {len(nodes)}, expected {expected}")

    terminal_currents = []
    for terminal in range(n_terminals):
        start = terminal * n_conductors
        stop = start + n_conductors
        by_node = {
            node: value
            for value, node in zip(complex_currents[start:stop], nodes[start:stop])
        }
        terminal_currents.append([complex(by_node.get(int(node), 0j)) for node in phase_nodes])
    return terminal_currents


def element_pq_3ph_per_terminal():
    """
    Return list of (P_MW, Q_Mvar) per terminal for the active CktElement,
    summing over all phases.
    """
    terminal_powers = _phase_terminal_complex_powers(
        dss.CktElement.Powers(),
        dss.CktElement.NodeOrder(),
        n_conductors=dss.CktElement.NumConductors(),
        n_terminals=dss.CktElement.NumTerminals(),
    )
    return [
        (float(value.real) / 1000.0, float(value.imag) / 1000.0)
        for value in terminal_powers
    ]


def _compile_and_solve(repo_dir: str, *, load_mult: float = 1.0) -> None:
    caller_cwd = os.getcwd()
    try:
        dss.Basic.DataPath(repo_dir)
        dss.Text.Command("Clear")
        dss.Text.Command("Redirect Run_IEEE14Bus.dss")
    finally:
        os.chdir(caller_cwd)
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


def _normalize_branch_overrides(branch_element_overrides):
    if not branch_element_overrides:
        return {}
    return {str(key).lower(): value for key, value in dict(branch_element_overrides).items()}


def _terminal_bus_name(terminal):
    names = dss.CktElement.BusNames() or []
    if len(names) <= int(terminal):
        raise ValueError(
            f"Element {dss.CktElement.Name()} has no terminal {terminal}"
        )
    return str(names[int(terminal)]).split(".")[0].lower()


def _bus_current_base_amps(bus, *, mva_base=100.0):
    """Per-phase current base: (S_base / 3) / V_LN,base."""
    dss.Circuit.SetActiveBus(bus)
    kvbase_ln = float(dss.Bus.kVBase() or 0.0)
    if kvbase_ln <= 0.0:
        raise ValueError(f"OpenDSS reports no LN base kV for bus {bus}")
    return (float(mva_base) * 1e6 / 3.0) / (kvbase_ln * 1000.0)


def _element_terminal_current_phasors(elem, terminal):
    if not dss.Circuit.SetActiveElement(elem):
        raise ValueError(f"OpenDSS element not found: {elem}")
    terminals = _phase_terminal_complex_currents(
        dss.CktElement.Currents(),
        dss.CktElement.NodeOrder(),
        n_conductors=dss.CktElement.NumConductors(),
        n_terminals=dss.CktElement.NumTerminals(),
    )
    if len(terminals) <= int(terminal):
        raise ValueError(f"Element {elem} has no terminal {terminal}")
    return terminals[int(terminal)], _terminal_bus_name(terminal)


def extract_three_phase_branch_current_measurements(
    *, branch_names=None, branch_element_overrides=None, mva_base=100.0
):
    """
    Extract per-branch, per-terminal, per-phase current phasors in per-unit.

    Every phasor is the current flowing *into* the branch from that terminal,
    the native OpenDSS ``CktElement.Currents()`` convention, so the two
    terminal currents of a healthy line sum to its charging current and a
    mid-span shunt fault appears as a per-phase differential.

    Returns a list aligned to ``branch_names`` (default ``BRANCH_ORDER``) with
    entries::

        {branch, branch_row0, from_bus, to_bus,
         ibase_from_a, ibase_to_a,
         i_from_pu: [Ia, Ib, Ic], ang_from_deg: [...],
         i_to_pu:   [Ia, Ib, Ic], ang_to_deg:   [...]}

    ``branch_element_overrides`` uses the same mapping as
    ``extract_measurement_series`` so hidden split-line HIF scenarios keep the
    external branch identity and terminal buses.
    """
    branch_names = BRANCH_ORDER if branch_names is None else list(branch_names)
    overrides = _normalize_branch_overrides(branch_element_overrides)
    rows = []
    for row0, elem in enumerate(branch_names):
        override = overrides.get(str(elem).lower())
        if isinstance(override, dict):
            terminal_specs = (
                (override.get("from", elem), int(override.get("from_terminal", 0))),
                (override.get("to", elem), int(override.get("to_terminal", 1))),
            )
        else:
            terminal_specs = ((elem, 0), (elem, 1))
        terminal_payloads = []
        for element_name, terminal in terminal_specs:
            phasors, bus = _element_terminal_current_phasors(element_name, terminal)
            ibase = _bus_current_base_amps(bus, mva_base=mva_base)
            per_unit = [value / ibase for value in phasors]
            terminal_payloads.append(
                {
                    "bus": bus,
                    "ibase": float(ibase),
                    "magnitude": [float(abs(value)) for value in per_unit],
                    "angle": [float(np.degrees(np.angle(value))) for value in per_unit],
                }
            )
        from_payload, to_payload = terminal_payloads
        rows.append(
            dict(
                branch=str(elem),
                branch_row0=int(row0),
                from_bus=from_payload["bus"],
                to_bus=to_payload["bus"],
                ibase_from_a=from_payload["ibase"],
                ibase_to_a=to_payload["ibase"],
                i_from_pu=from_payload["magnitude"],
                ang_from_deg=from_payload["angle"],
                i_to_pu=to_payload["magnitude"],
                ang_to_deg=to_payload["angle"],
            )
        )
    return rows


def _branch_terminal_pq(elem, terminal):
    dss.Circuit.SetActiveElement(elem)
    pqs = element_pq_3ph_per_terminal()
    if len(pqs) > int(terminal):
        return pqs[int(terminal)]
    return (0.0, 0.0)


def _active_element_enabled():
    return not hasattr(dss.CktElement, "Enabled") or bool(dss.CktElement.Enabled())


def extract_measurement_series(*, buses=None, branch_names=None, branch_element_overrides=None):
    """
    Extract the 1ϕ-equivalent (phase-A) operator measurement vector from the *currently solved* circuit.

    Layout:
      [Vm(1..nb), Pinj(1..nb), Qinj(1..nb), Pf(1..nl), Qf(1..nl), Pt(1..nl), Qt(1..nl)]

    - Vm uses phase-1 (phase A) VLN magnitude per bus.
    - Pinj/Qinj follow MATPOWER makeSbus convention in per-unit on 100 MVA.
    - Branch flows use BRANCH_ORDER to match MATPOWER case14 branch rows.
    - branch_element_overrides can map an external branch name to replacement
      OpenDSS elements, e.g. for hidden midspan HIF buses:
      {"Line.2-3": {"from": "Line.2-3_hif_a", "to": "Line.2-3_hif_b"}}.
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
        if not _active_element_enabled():
            continue
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
        if not _active_element_enabled():
            continue
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
        if not _active_element_enabled():
            continue
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
        if not _active_element_enabled():
            continue
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

    overrides = _normalize_branch_overrides(branch_element_overrides)
    for elem in branch_names:
        override = overrides.get(str(elem).lower())
        if isinstance(override, dict):
            from_elem = override.get("from", elem)
            to_elem = override.get("to", elem)
            from_terminal = int(override.get("from_terminal", 0))
            to_terminal = int(override.get("to_terminal", 1))
            p_from, q_from = _branch_terminal_pq(from_elem, from_terminal)
            p_to, q_to = _branch_terminal_pq(to_elem, to_terminal)
        else:
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
