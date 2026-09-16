"""Registry-driven measurements from an already solved OpenDSS context.

No IEEE-14 names, dimensions, global DSS engine, or inferred asset endpoints
are used here. Powers are total three-phase quantities on the system MVA
base; phase voltage/current phasors use LN and per-phase power bases.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Mapping, Sequence


PHASES = (1, 2, 3)
SEQUENCES = ("zero", "positive", "negative")


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise ValueError(f"Non-positive {label}")
    return result


def _rect(values: Sequence[complex]) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def _pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _sequence(values: Sequence[complex]) -> list[complex]:
    va, vb, vc = values
    rotation = cmath.exp(2j * math.pi / 3.0)
    return [
        (va + vb + vc) / 3.0,
        (va + rotation * vb + rotation**2 * vc) / 3.0,
        (va + rotation**2 * vb + rotation * vc) / 3.0,
    ]


def _phasor_fields(values: Sequence[complex], prefix: str) -> dict[str, Any]:
    sequence = _sequence(values)
    return {
        f"{prefix}_pu": [float(abs(value)) for value in values],
        f"{prefix}_ang_deg": [float(math.degrees(cmath.phase(value))) for value in values],
        f"{prefix}_pu_rect": _rect(values),
        f"{prefix}_sequence_pu_rect": _rect(sequence),
        f"{prefix}_sequence_pu": [float(abs(value)) for value in sequence],
    }


def _ordered_registry(registry: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    buses = sorted(list(registry["buses"]), key=lambda row: int(row["row0"]))
    branches = sorted(list(registry["branches"]), key=lambda row: int(row["branch_row0"]))
    if not buses:
        raise ValueError("The bus registry is empty")
    for rows, field in ((buses, "row0"), (branches, "branch_row0")):
        if [int(row[field]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"Registry {field} must be unique and contiguous from zero")
    if len({row["external_bus"] for row in buses}) != len(buses):
        raise ValueError("Duplicate external bus identifiers")
    if len({str(row["dss_bus"]).lower() for row in buses}) != len(buses):
        raise ValueError("Duplicate OpenDSS bus identifiers")
    if len({row["asset_id"] for row in branches}) != len(branches):
        raise ValueError("Duplicate branch asset identifiers")
    bus_ids = {row["external_bus"] for row in buses}
    if any(row[end] not in bus_ids for row in branches for end in ("from_bus", "to_bus")):
        raise ValueError("Branch endpoint is absent from the bus registry")
    return buses, branches


def write_layout(registry: Mapping[str, Any], assumptions: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-ready layout; despite its name this performs no file IO."""

    buses, branches = _ordered_registry(registry)
    base_mva = _positive(assumptions["base_mva"], "base_mva")
    bus_ids = [row["external_bus"] for row in buses]
    branch_ids = [row["asset_id"] for row in branches]
    channels = []
    cursor = 0
    for channel, identifiers in (
        ("Vm", bus_ids), ("Pinj", bus_ids), ("Qinj", bus_ids),
        ("Pf", branch_ids), ("Qf", branch_ids), ("Pt", branch_ids), ("Qt", branch_ids),
    ):
        channels.append({
            "channel": channel,
            "start_index0": cursor,
            "stop_index0_exclusive": cursor + len(identifiers),
            "count": len(identifiers),
            "asset_ids": identifiers,
        })
        cursor += len(identifiers)
    return {
        "schema_version": 1,
        "contract": "registry_three_phase_measurement_layout_v1",
        "measurement_count": cursor,
        "bus_count": len(buses),
        "branch_count": len(branches),
        "base_mva_three_phase": base_mva,
        "phase_order": list(PHASES),
        "sequence_order": list(SEQUENCES),
        "complex_encoding": "[real, imaginary]",
        "operator_voltage": "phase_A_line_to_neutral_magnitude_pu",
        "operator_power": "total_three_phase_power_on_system_base_mva",
        "operator_injection": "generation_plus_source_minus_load_excluding_bus_shunts",
        "branch_terminal_direction": "current_and_power_into_branch_at_each_terminal",
        "registry_terminal_indexing": "one_based",
        "phase_current_base": "(base_mva*1e6/3)/(bus_kv_ll*1000/sqrt(3))",
        "phase_voltage_base": "bus_kv_ll*1000/sqrt(3)",
        "phase_power_base": "base_mva/3",
        "positive_sequence_power": "3*V1_physical*conj(I1_physical)/system_va_base",
        "legacy_ieee14_compatible_vector": {
            "difference": "Pinj/Qinj also include negative power into bus shunt elements",
            "injection": "generation_plus_source_minus_load_minus_bus_shunt_consumption",
            "voltage_and_branch_power": "same_as_measurement_vector",
        },
        "bus_order": bus_ids,
        "branch_order": branch_ids,
        "channels": channels,
    }


def _complex_array(raw: Sequence[float], label: str) -> list[complex]:
    if len(raw) % 2:
        raise ValueError(f"Odd complex-array length for {label}")
    return [
        complex(_finite(raw[index], label), _finite(raw[index + 1], label))
        for index in range(0, len(raw), 2)
    ]


def _element_terminal(dss: Any, element: str, terminal: int) -> dict[str, Any]:
    """Read ABC conductors by node identity, excluding all ground/neutral nodes."""

    # OpenDSS returns a zero-based element index: source index 0 is valid.
    try:
        dss.Circuit.SetActiveElement(str(element))
    except Exception as exc:
        raise ValueError(f"Missing OpenDSS element {element}") from exc
    if str(dss.CktElement.Name()).lower() != str(element).lower():
        raise ValueError(f"Missing OpenDSS element {element}")
    ncond = int(dss.CktElement.NumConductors())
    nterm = int(dss.CktElement.NumTerminals())
    if terminal < 1 or terminal > nterm:
        raise ValueError(f"Invalid terminal {terminal} for {element}")
    if not bool(dss.CktElement.Enabled()):
        # OpenDSS never initializes NodeOrder for assets disabled at compile
        # time. Their physical terminal currents/powers are zero. Read only
        # the declared terminal node identity, preserving fixed sensor rows.
        bus_ref = str(dss.CktElement.BusNames()[terminal - 1]).lower().split(".")
        declared = ([int(node) for node in bus_ref[1:]] if len(bus_ref) > 1
                    else list(range(1, int(dss.CktElement.NumPhases()) + 1)))
        phase_nodes = [node for node in declared if node in PHASES]
        if len(phase_nodes) != len(set(phase_nodes)):
            raise ValueError(f"Duplicate phase at {element} terminal {terminal}")
        return {"current_a": [0j] * 3, "power_va": [0j] * 3,
                "phase_nodes": sorted(phase_nodes), "bus": bus_ref[0]}
    nodes = list(dss.CktElement.NodeOrder())
    currents = _complex_array(list(dss.CktElement.Currents()), f"{element} currents")
    powers = _complex_array(list(dss.CktElement.Powers()), f"{element} powers")
    if any(len(values) != ncond * nterm for values in (nodes, currents, powers)):
        raise ValueError(f"Inconsistent conductor arrays for {element}")
    start = (terminal - 1) * ncond
    out_current = [0j, 0j, 0j]
    out_power = [0j, 0j, 0j]
    seen = set()
    for index in range(start, start + ncond):
        node = int(nodes[index])
        if node not in PHASES:
            continue
        if node in seen:
            raise ValueError(f"Duplicate phase {node} at {element} terminal {terminal}")
        seen.add(node)
        out_current[node - 1] = currents[index]
        out_power[node - 1] = powers[index] * 1000.0  # DSS kW/kvar -> W/var
    return {
        "current_a": out_current,
        "power_va": out_power,
        "phase_nodes": sorted(seen),
        "bus": str(dss.CktElement.BusNames()[terminal - 1]).split(".")[0].lower(),
    }


def _sum_vectors(target: list[complex], values: Sequence[complex], scale: float = 1.0) -> None:
    for index in range(3):
        target[index] += scale * values[index]


def _branch_terminal_spec(
    branch: Mapping[str, Any], end: str, override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve a physical asset's external terminal without exposing split IDs."""

    spec = {
        "element": branch["dss_element"],
        "terminal": int(branch[f"{end}_terminal"]),
        "charging_elements": branch.get("charging_elements", {}).get(end, []),
    }
    if override is None:
        return spec
    terminal = override.get(end)
    if not isinstance(terminal, Mapping) or not terminal.get("element"):
        raise ValueError(f"Incomplete branch override for {branch['asset_id']} {end}")
    if "terminal" not in terminal:
        raise ValueError(f"Missing one-based terminal for {branch['asset_id']} {end}")
    spec.update(terminal)
    raw_terminal = spec["terminal"]
    if isinstance(raw_terminal, bool) or int(raw_terminal) != raw_terminal or int(raw_terminal) < 1:
        raise ValueError(f"Invalid one-based terminal for {branch['asset_id']} {end}")
    spec["terminal"] = int(raw_terminal)
    if not isinstance(spec["charging_elements"], (list, tuple)):
        raise ValueError(f"Invalid charging elements for {branch['asset_id']} {end}")
    return spec


def audit_full_circuit_kcl(dss: Any, assumptions: Mapping[str, Any]) -> dict[str, Any]:
    """Audit every enabled element and non-ground node for offline physics QA.

    This is deliberately separate from diagnostic telemetry: it inspects hidden
    split/fault nodes and must not be included in a policy observation. Ground
    node 0 is the reference, while explicit nonzero neutral nodes are audited.
    """

    if not dss.Solution.Converged():
        raise ValueError("OpenDSS solution did not converge")
    base_va = _positive(assumptions["base_mva"], "base_mva") * 1e6
    node_currents: dict[tuple[str, int], complex] = {}
    current_bases: dict[str, float] = {}
    for name in dss.Circuit.AllBusNames():
        bus = str(name).lower()
        dss.Circuit.SetActiveBus(bus)
        kv_ln = float(dss.Bus.kVBase())
        if kv_ln <= 0.0:
            # Newly introduced hidden nodes may not yet have a DSS voltage
            # base. This package's declared normalized uniform realization
            # supplies their base without inferring an equipment voltage.
            kv_ln = _positive(assumptions["base_kv_ll"], "base_kv_ll") / math.sqrt(3.0)
        current_bases[bus] = base_va / 3.0 / (1000.0 * _positive(kv_ln, "bus kV LN"))
        for node in dss.Bus.Nodes():
            if int(node) != 0:
                node_currents[(bus, int(node))] = 0j
    if not node_currents:
        raise ValueError("OpenDSS circuit has no non-ground nodes")
    enabled_elements = 0
    for name in dss.Circuit.AllElementNames():
        dss.Circuit.SetActiveElement(str(name))
        if not dss.CktElement.Enabled():
            continue
        enabled_elements += 1
        ncond, nterm = int(dss.CktElement.NumConductors()), int(dss.CktElement.NumTerminals())
        buses = list(dss.CktElement.BusNames())
        nodes = list(dss.CktElement.NodeOrder())
        currents = _complex_array(list(dss.CktElement.Currents()), f"{name} currents")
        if len(buses) != nterm or len(nodes) != ncond * nterm or len(currents) != len(nodes):
            raise ValueError(f"Inconsistent conductor arrays for {name}")
        for index, current in enumerate(currents):
            node = int(nodes[index])
            if node == 0:
                continue
            bus = str(buses[index // ncond]).split(".")[0].lower()
            if (bus, node) not in node_currents:
                raise ValueError(f"Element {name} references a missing non-ground node")
            node_currents[(bus, node)] += current
    return {
        "contract": "all_active_dss_nodes_kcl_offline_v1",
        "policy_observable": False,
        "scope": "all_enabled_elements_and_all_non_ground_nodes_including_hidden_nodes",
        "node_count": len(node_currents),
        "enabled_element_count": enabled_elements,
        "max_kcl_mismatch_pu": max(abs(value) / current_bases[bus] for (bus, _), value in node_currents.items()),
        "max_kcl_mismatch_a": max(abs(value) for value in node_currents.values()),
    }


def extract_measurements(
    dss: Any, registry: Mapping[str, Any], assumptions: Mapping[str, Any],
    *, branch_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Export external telemetry without compiling, solving, or changing devices.

    ``branch_overrides`` maps immutable asset IDs to replacement ``from`` and
    ``to`` terminal specifications. Only the original physical asset identity
    is exported; hidden split elements, fault devices, and buses stay offline.
    """

    if not dss.Solution.Converged():
        raise ValueError("OpenDSS solution did not converge")
    buses, branches = _ordered_registry(registry)
    branch_overrides = {} if branch_overrides is None else branch_overrides
    if not isinstance(branch_overrides, Mapping):
        raise ValueError("branch_overrides must be a mapping keyed by asset_id")
    unknown_assets = set(branch_overrides) - {row["asset_id"] for row in branches}
    if unknown_assets:
        raise ValueError("Branch overrides contain unregistered asset identifiers")
    if any(not isinstance(value, Mapping) for value in branch_overrides.values()):
        raise ValueError("Every branch override must specify external terminals")
    layout = write_layout(registry, assumptions)
    base_va = layout["base_mva_three_phase"] * 1e6
    by_external = {row["external_bus"]: row for row in buses}
    volts: dict[Any, list[complex]] = {}
    ibases: dict[Any, float] = {}
    voltage_rows = []
    for bus in buses:
        bus_id, name = bus["external_bus"], str(bus["dss_bus"])
        dss.Circuit.SetActiveBus(name)
        if str(dss.Bus.Name()).lower() != name.lower():
            raise ValueError(f"Missing OpenDSS bus {name}")
        nodes = list(dss.Bus.Nodes())
        values = _complex_array(list(dss.Bus.Voltages()), f"{name} voltages")
        if len(nodes) != len(values):
            raise ValueError(f"Inconsistent bus voltage arrays at {name}")
        by_node = dict(zip(nodes, values))
        if not all(phase in by_node for phase in PHASES):
            raise ValueError(f"Bus {name} does not have all three phase nodes")
        kv_ll = _positive(bus.get("kv_ll", assumptions["base_kv_ll"]), f"{name} kV LL")
        vbase = kv_ll * 1000.0 / math.sqrt(3.0)
        volts[bus_id] = [by_node[phase] for phase in PHASES]
        ibases[bus_id] = base_va / 3.0 / vbase
        pu = [value / vbase for value in volts[bus_id]]
        fields = _phasor_fields(pu, "vln")
        voltage_rows.append({
            "bus": name, "external_bus": bus_id, "row0": int(bus["row0"]),
            "kvbase_ln": kv_ll / math.sqrt(3.0),
            **fields,
            "ang_deg": fields["vln_ang_deg"],
            "vm_positive_sequence_pu": float(abs(_sequence(pu)[1])),
        })

    gross_power = {row["external_bus"]: [0j] * 3 for row in buses}
    shunt_consumption = {row["external_bus"]: [0j] * 3 for row in buses}
    net_device_current = {row["external_bus"]: [0j] * 3 for row in buses}
    device_exports: dict[str, list[dict]] = {"loads": [], "generators": [], "shunts": [], "source": []}
    source = registry.get("source")
    for family in device_exports:
        entries = [source] if family == "source" and source else registry.get(family, [])
        for spec in entries:
            bus_id = spec["bus"]
            if bus_id not in by_external:
                raise ValueError(f"Unregistered device bus {bus_id}")
            element = str(spec["element"])
            terminal = _element_terminal(dss, element, int(spec.get("terminal", 1)))
            if terminal["bus"] != str(by_external[bus_id]["dss_bus"]).lower():
                raise ValueError(f"Device bus mismatch for {element}")
            injected = [-value for value in terminal["power_va"]]
            _sum_vectors(net_device_current[bus_id], terminal["current_a"], -1.0)
            if family == "shunts":
                _sum_vectors(shunt_consumption[bus_id], terminal["power_va"])
            else:
                _sum_vectors(gross_power[bus_id], injected)
            device_exports[family].append({
                **dict(spec),
                "phase_nodes": terminal["phase_nodes"],
                "power_into_element_pu": _pair(sum(terminal["power_va"]) / base_va),
                "injection_total_pu": _pair(sum(injected) / base_va),
                "phase_injection_pu_rect": _rect([value / (base_va / 3.0) for value in injected]),
            })

    branch_sum_current = {row["external_bus"]: [0j] * 3 for row in buses}
    current_rows, power_rows = [], []
    for branch in branches:
        terminals = []
        for end in ("from", "to"):
            bus_id = branch[f"{end}_bus"]
            spec = _branch_terminal_spec(branch, end, branch_overrides.get(branch["asset_id"]))
            terminal = _element_terminal(dss, str(spec["element"]), spec["terminal"])
            if terminal["bus"] != str(by_external[bus_id]["dss_bus"]).lower():
                raise ValueError(f"Branch orientation mismatch for {branch['asset_id']} {end}")
            if terminal["phase_nodes"] != list(PHASES):
                raise ValueError(f"Branch {branch['asset_id']} {end} lacks three phases")
            charging = spec["charging_elements"]
            for item in charging:
                spec = item if isinstance(item, Mapping) else {"element": item}
                charge = _element_terminal(dss, str(spec["element"]), int(spec.get("terminal", 1)))
                if charge["bus"] != terminal["bus"]:
                    raise ValueError(f"Charging element bus mismatch for {branch['asset_id']} {end}")
                _sum_vectors(terminal["current_a"], charge["current_a"])
                _sum_vectors(terminal["power_va"], charge["power_va"])
            _sum_vectors(branch_sum_current[bus_id], terminal["current_a"])
            terminal["current_pu"] = [value / ibases[bus_id] for value in terminal["current_a"]]
            terminal["sequence_power_pu"] = [
                3.0 * voltage * current.conjugate() / base_va
                for voltage, current in zip(_sequence(volts[bus_id]), _sequence(terminal["current_a"]))
            ]
            terminals.append(terminal)
        from_term, to_term = terminals
        common = {
            "asset_id": branch["asset_id"], "branch": branch["dss_element"],
            "branch_row0": int(branch["branch_row0"]),
            "from_bus": branch["from_bus"], "to_bus": branch["to_bus"],
        }
        current_row = {**common}
        power_row = {**common}
        for end, terminal in zip(("from", "to"), terminals):
            fields = _phasor_fields(terminal["current_pu"], f"i_{end}")
            current_row.update(fields)
            current_row[f"ang_{end}_deg"] = fields[f"i_{end}_ang_deg"]
            current_row[f"ibase_{end}_a"] = ibases[branch[f"{end}_bus"]]
            power = sum(terminal["power_va"]) / base_va
            power_row[f"p_{end}_pu"] = float(power.real)
            power_row[f"q_{end}_pu"] = float(power.imag)
            power_row[f"s_{end}_total_pu"] = _pair(power)
            power_row[f"s_{end}_phase_pu_rect"] = _rect([value / (base_va / 3.0) for value in terminal["power_va"]])
            power_row[f"s_{end}_sequence_pu_rect"] = _rect(terminal["sequence_power_pu"])
            power_row[f"s_{end}_positive_sequence_pu"] = _pair(terminal["sequence_power_pu"][1])
        current_rows.append(current_row)
        power_rows.append(power_row)

    injection_rows = []
    for bus in buses:
        bus_id = bus["external_bus"]
        gross = sum(gross_power[bus_id]) / base_va
        shunt = sum(shunt_consumption[bus_id]) / base_va
        net = gross - shunt
        kcl = [
            (device - branch) / ibases[bus_id]
            for device, branch in zip(net_device_current[bus_id], branch_sum_current[bus_id])
        ]
        injection_rows.append({
            "bus": bus["dss_bus"], "external_bus": bus_id, "row0": int(bus["row0"]),
            "p_inj_pu": float(gross.real), "q_inj_pu": float(gross.imag),
            "p_net_into_branches_pu": float(net.real), "q_net_into_branches_pu": float(net.imag),
            "bus_shunt_consumption_pu": _pair(shunt),
            "net_device_current_pu_rect": _rect([value / ibases[bus_id] for value in net_device_current[bus_id]]),
            "branch_sum_current_pu_rect": _rect([value / ibases[bus_id] for value in branch_sum_current[bus_id]]),
            "kcl_mismatch_pu_rect": _rect(kcl),
            "max_kcl_mismatch_pu": max(float(abs(value)) for value in kcl),
        })
    voltage_block = [row["vln_pu"][0] for row in voltage_rows]
    branch_blocks = [row[key] for key in ("p_from_pu", "q_from_pu", "p_to_pu", "q_to_pu") for row in power_rows]
    vector = voltage_block + [row["p_inj_pu"] for row in injection_rows] + [row["q_inj_pu"] for row in injection_rows] + branch_blocks
    legacy = voltage_block + [row["p_net_into_branches_pu"] for row in injection_rows] + [row["q_net_into_branches_pu"] for row in injection_rows] + branch_blocks
    return {
        "schema_version": 1,
        "contract": "registry_three_phase_snapshot_measurements_v1",
        "measurement_layout": layout,
        "measurement_vector": vector,
        "legacy_ieee14_compatible_vector": legacy,
        "three_phase_voltages": voltage_rows,
        "three_phase_branch_currents": current_rows,
        "branch_powers": power_rows,
        "bus_injections": injection_rows,
        "load_powers": device_exports["loads"],
        "generator_injections": device_exports["generators"],
        "source_injection": device_exports["source"][0] if device_exports["source"] else None,
        "shunt_powers": device_exports["shunts"],
        "kcl_scope": "registered_external_bus_phase_nodes",
        "max_kcl_mismatch_pu": max(row["max_kcl_mismatch_pu"] for row in injection_rows),
    }
