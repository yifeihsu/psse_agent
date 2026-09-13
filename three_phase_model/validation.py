"""Independent numerical audit of a compiled OpenDSS three-phase realization.

Admittances come from OpenDSS YPrim, in column-major complex order. Reference
terminal admittances are formed directly from MATPOWER branch R/X/B/tap data;
exporter matrices and claimed equivalent parameters are never used as evidence.
Currents and powers use the OpenDSS convention of flow into an element.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ValidationTolerances:
    voltage_magnitude_pu: float = 1e-6
    voltage_angle_deg: float = 1e-4
    branch_power_pu: float = 1e-5
    device_power_pu: float = 1e-5
    admittance_pu: float = 1e-8
    phase_kcl_pu: float = 1e-7
    constitutive_current_pu: float = 1e-8
    sequence_voltage_pu: float = 1e-8
    sequence_current_pu: float = 1e-8
    source_impedance_pu: float = 1e-12


_A = np.exp(2j * np.pi / 3)
_PHASE = np.asarray([1, _A**2, _A], dtype=complex)
_SEQ = np.asarray([[1, 1, 1], [1, _A, _A**2], [1, _A**2, _A]], dtype=complex) / 3


def _complex_array(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        result = raw.astype(complex).reshape(-1)
    else:
        raw = raw.astype(float).reshape(-1)
        if len(raw) % 2:
            raise ValueError(f"{name} has an odd interleaved complex-array length")
        result = raw[::2] + 1j * raw[1::2]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _bus_name(name: str) -> str:
    return str(name).strip().lower().split(".")[0]


def _pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _matrix_pairs(value: np.ndarray) -> list[list[list[float]]]:
    return [[_pair(item) for item in row] for row in value]


def _max(values: Any) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _element(dss: Any, name: str) -> dict[str, Any]:
    if dss.Circuit.SetActiveElement(name) < 0:
        raise ValueError(f"OpenDSS element is missing: {name}")
    if not bool(dss.CktElement.Enabled()):
        # Disabled assets need not have initialized NodeOrder in OpenDSS. They
        # contribute no circuit currents or admittance; coverage still detects
        # an asset whose registry incorrectly claims it is enabled.
        return {"name": name.lower(), "enabled": False, "ncond": 0, "nterm": 0,
                "nodes": np.zeros(0, dtype=int), "buses": [],
                "currents": np.zeros(0, dtype=complex), "volts": np.zeros(0, dtype=complex),
                "powers_kva": np.zeros(0, dtype=complex)}
    ncond, nterm = int(dss.CktElement.NumConductors()), int(dss.CktElement.NumTerminals())
    size = ncond * nterm
    nodes = np.asarray(dss.CktElement.NodeOrder(), dtype=int)
    buses = list(dss.CktElement.BusNames())
    if len(nodes) != size or len(buses) != nterm:
        raise ValueError(f"invalid terminal/node dimensions for {name}")
    currents = _complex_array(dss.CktElement.Currents(), name=f"{name} currents")
    volts = _complex_array(dss.CktElement.Voltages(), name=f"{name} voltages")
    powers = _complex_array(dss.CktElement.Powers(), name=f"{name} powers")
    if any(len(item) != size for item in (currents, volts, powers)):
        raise ValueError(f"invalid voltage/current/power dimensions for {name}")
    return {
        "name": name.lower(), "enabled": bool(dss.CktElement.Enabled()),
        "ncond": ncond, "nterm": nterm, "nodes": nodes,
        "buses": buses, "currents": currents, "volts": volts, "powers_kva": powers,
    }


def _yprim(dss: Any, element: Mapping[str, Any]) -> np.ndarray:
    dss.Circuit.SetActiveElement(element["name"])
    size = len(element["nodes"])
    flat = _complex_array(dss.CktElement.YPrim(), name=f"{element['name']} YPrim")
    if flat.size != size * size:
        raise ValueError(f"invalid YPrim dimensions for {element['name']}")
    return flat.reshape((size, size), order="F")


def positive_sequence_terminal_admittance(
    dss: Any, element_name: str, from_bus: str, to_bus: str, *, zbase_ohm: float,
) -> np.ndarray:
    """Project the actual grounded phase-domain primitive onto two terminals."""
    element = _element(dss, element_name)
    if not element["enabled"]:
        return np.zeros((2, 2), dtype=complex)
    y = _yprim(dss, element)
    excitation = np.zeros((len(element["nodes"]), 2), dtype=complex)
    endpoints = {_bus_name(from_bus): 0, _bus_name(to_bus): 1}
    if len(endpoints) != 2:
        raise ValueError("branch endpoints must be distinct")
    for terminal, bus in enumerate(element["buses"]):
        for conductor in range(element["ncond"]):
            index = terminal * element["ncond"] + conductor
            node = element["nodes"][index]
            if node == 0:
                continue
            if node not in (1, 2, 3) or _bus_name(bus) not in endpoints:
                raise ValueError(f"unexpected non-ground conductor in {element_name}")
            excitation[index, endpoints[_bus_name(bus)]] = _PHASE[node - 1]
    # Conjugate transpose / 3 extracts I1 from [Ia, Ib, Ic]. Grounded neutral
    # conductors get zero excitation; no unsupported neutral elimination occurs.
    return excitation.conj().T @ y @ excitation * (float(zbase_ohm) / 3)


def reference_terminal_admittance(branch: Any) -> np.ndarray:
    row = np.asarray(branch, dtype=float)
    if int(row[10]) == 0:
        return np.zeros((2, 2), dtype=complex)
    series = 1 / complex(row[2], row[3])
    charging = 0.5j * row[4]
    ratio = row[8] if row[8] else 1.0
    tap = ratio * np.exp(1j * np.deg2rad(row[9]))
    return np.asarray([
        [(series + charging) / abs(tap)**2, -series / np.conj(tap)],
        [-series / tap, series + charging],
    ])


def validate_model(
    dss: Any, reference: Mapping[str, Any], registry: Mapping[str, Any],
    assumptions: Mapping[str, Any], *, balanced: bool = True,
    tolerances: ValidationTolerances | None = None,
) -> dict[str, Any]:
    """Audit a compiled, solved engine without changing its model or solution.

``balanced=False`` keeps component equations, coverage, phase KCL and power
accounting checks, while skipping balanced-reference and sequence-null claims.
Malformed or unavailable engine evidence raises rather than producing a pass.
"""
    tol = tolerances or ValidationTolerances()
    for key, value in asdict(tol).items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid validation tolerance: {key}")
    if int(dss.Circuit.NumBuses()) <= 0:
        raise ValueError("OpenDSS has no compiled circuit")
    base_mva = float(reference["baseMVA"])
    kv_ll = float(assumptions["base_kv_ll"])
    if not math.isfinite(base_mva) or not math.isfinite(kv_ll) or min(base_mva, kv_ll) <= 0:
        raise ValueError("positive finite power and voltage bases are required")
    zbase = kv_ll**2 / base_mva
    ibase = base_mva * 1000 / (np.sqrt(3) * kv_ll)
    sbase_kva = base_mva * 1000
    bus_table = np.asarray(reference["bus"], dtype=float)
    branch_table = np.asarray(reference["branch"], dtype=float)
    gen_table = np.asarray(reference["gen"], dtype=float)
    if not all(np.all(np.isfinite(table)) for table in (bus_table, branch_table, gen_table)):
        raise ValueError("reference tables contain non-finite values")
    buses = list(registry["buses"])
    by_external = {int(row["external_bus"]): row for row in buses}
    if len(by_external) != len(buses):
        raise ValueError("duplicate external bus identifiers")
    checks: dict[str, Any] = {}

    def check(name: str, error: float, limit: float, **details: Any) -> None:
        checks[name] = {"passed": bool(np.isfinite(error) and error <= limit),
                        "max_error": float(error), "tolerance": float(limit), **details}

    source = registry["source"]
    branch_rows = list(registry["branches"])
    expected_names = {str(source["element"]).lower()}
    assigned_names = [str(source["element"]).lower()]
    passive_names: set[str] = set()
    for row in branch_rows:
        if int(row["status"]) == 0:
            continue
        names = [row["dss_element"]]
        for end in ("from", "to"):
            names.extend(row.get("charging_elements", {}).get(end, []))
        passive_names.update(str(name).lower() for name in names)
        assigned_names.extend(str(name).lower() for name in names)
    for kind in ("loads", "generators", "shunts"):
        expected_names.update(str(item["element"]).lower() for item in registry.get(kind, []))
        assigned_names.extend(str(item["element"]).lower() for item in registry.get(kind, []))
    passive_names.update(str(item["element"]).lower() for item in registry.get("shunts", []))
    expected_names.update(passive_names)
    elements = {str(name).lower(): _element(dss, str(name)) for name in dss.Circuit.AllElementNames()}
    enabled = {name for name, element in elements.items() if element["enabled"]}
    coverage_errors = sorted(expected_names - enabled) + sorted(enabled - expected_names)
    reused_names = sorted(name for name, count in Counter(assigned_names).items() if count != 1)
    bus_rows = [int(row["row0"]) for row in buses]
    actual_branch_rows = [int(row["branch_row0"]) for row in branch_rows]
    coverage_ok = (
        not coverage_errors and not reused_names and sorted(bus_rows) == list(range(len(bus_table)))
        and sorted(actual_branch_rows) == list(range(len(branch_table)))
        and all(int(bus_table[int(row["row0"]), 0]) == int(row["external_bus"]) for row in buses)
        and all(int(branch_table[int(row["branch_row0"]), 10]) == int(row["status"]) for row in branch_rows)
    )
    checks["asset_coverage"] = {"passed": coverage_ok, "missing_or_unregistered_enabled_elements": coverage_errors,
                               "reused_registry_elements": reused_names}
    checks["solution_converged"] = {"passed": bool(dss.Solution.Converged())}
    checks["base_mva_matches"] = {"passed": float(assumptions["base_mva"]) == base_mva}
    identity_problems: list[str] = []
    for kind in ("loads", "generators"):
        for device in registry.get(kind, []):
            element = elements[device["element"].lower()]
            energized = [(_bus_name(bus), int(element["nodes"][terminal * element["ncond"] + conductor]))
                         for terminal, bus in enumerate(element["buses"])
                         for conductor in range(element["ncond"])
                         if element["nodes"][terminal * element["ncond"] + conductor] != 0]
            expected = [(_bus_name(by_external[int(device["bus"])]["dss_bus"]), int(device["phase"]))]
            if energized != expected:
                identity_problems.append(f"{device['element']}:phase_or_bus_mapping_mismatch")
            if kind == "generators" and int(gen_table[int(device["gen_row0"]), 0]) != int(device["bus"]):
                identity_problems.append(f"{device['element']}:reference_generator_bus_mismatch")

    bus_results: list[dict[str, Any]] = []
    bus_phase_pu: dict[int, np.ndarray] = {}
    vm_error: list[float] = []
    va_error: list[float] = []
    v_null: list[float] = []
    for row in buses:
        external, index = int(row["external_bus"]), int(row["row0"])
        if float(row["kv_ll"]) != kv_ll:
            raise ValueError("validator currently requires a uniform declared line-to-line voltage base")
        if dss.Circuit.SetActiveBus(row["dss_bus"]) < 0:
            raise ValueError(f"OpenDSS bus missing: {row['dss_bus']}")
        nodes = list(dss.Bus.Nodes())
        raw = _complex_array(dss.Bus.Voltages(), name=f"bus {external} voltages")
        if len(nodes) != len(raw) or any(nodes.count(phase) != 1 for phase in (1, 2, 3)):
            raise ValueError(f"bus {external} does not have exactly one voltage per phase")
        volts = np.asarray([raw[nodes.index(phase)] for phase in (1, 2, 3)]) / (kv_ll * 1000 / np.sqrt(3))
        bus_phase_pu[external] = volts
        sequence = _SEQ @ volts
        v_null.extend(np.abs(sequence[[0, 2]]).tolist())
        result = {"bus": external, "vm_pu": np.abs(volts).tolist(),
                  "va_deg": np.rad2deg(np.angle(volts)).tolist(),
                  "sequence_voltage_pu": [_pair(v) for v in sequence]}
        if balanced:
            target = bus_table[index, 7] * np.exp(1j * np.deg2rad(bus_table[index, 8])) * _PHASE
            vm_error.extend((np.abs(volts) - np.abs(target)).tolist())
            va_error.extend(np.rad2deg(np.angle(volts / target)).tolist())
            result["reference_vm_pu"] = float(bus_table[index, 7])
            result["reference_va_deg"] = float(bus_table[index, 8])
        bus_results.append(result)

    def side_values(names: list[str], endpoint: int) -> tuple[complex, np.ndarray]:
        power, currents = 0j, np.zeros(3, dtype=complex)
        endpoint_name = _bus_name(by_external[endpoint]["dss_bus"])
        for name in names:
            element = elements[name.lower()]
            for terminal, bus in enumerate(element["buses"]):
                if _bus_name(bus) != endpoint_name:
                    continue
                for conductor in range(element["ncond"]):
                    index = terminal * element["ncond"] + conductor
                    node = element["nodes"][index]
                    if node in (1, 2, 3):
                        power += element["powers_kva"][index] / sbase_kva
                        currents[node - 1] += element["currents"][index] / ibase
                    elif node != 0:
                        raise ValueError("only phase nodes 1/2/3 and grounded node 0 are supported")
        return power, currents

    branch_results: list[dict[str, Any]] = []
    y_errors, power_errors, i_null = [], [], []
    branch_loss_actual, branch_loss_reference = 0j, 0j
    for asset in branch_rows:
        index = int(asset["branch_row0"])
        row = branch_table[index]
        fb, tb = int(asset["from_bus"]), int(asset["to_bus"])
        if (fb, tb) != (int(row[0]), int(row[1])):
            raise ValueError(f"branch {index} terminal orientation disagrees with reference")
        target_y = reference_terminal_admittance(row)
        names = [] if int(asset["status"]) == 0 else [asset["dss_element"]]
        if names:
            main = elements[names[0].lower()]
            for endpoint, terminal_key in ((fb, "from_terminal"), (tb, "to_terminal")):
                terminal = int(asset[terminal_key]) - 1
                if not 0 <= terminal < main["nterm"] or _bus_name(main["buses"][terminal]) != _bus_name(by_external[endpoint]["dss_bus"]):
                    identity_problems.append(f"{asset['dss_element']}:{terminal_key}_mapping_mismatch")
        if int(asset["status"]) != 0:
            for end in ("from", "to"):
                names.extend(asset.get("charging_elements", {}).get(end, []))
        actual_y = np.zeros((2, 2), dtype=complex)
        for name in names:
            actual_y += positive_sequence_terminal_admittance(
                dss, name, by_external[fb]["dss_bus"], by_external[tb]["dss_bus"], zbase_ohm=zbase,
            )
        y_error = _max(actual_y - target_y)
        y_errors.append(y_error)
        sf, ifrom = side_values(names, fb)
        st, ito = side_values(names, tb)
        i_null.extend(np.abs((_SEQ @ ifrom)[[0, 2]]).tolist())
        i_null.extend(np.abs((_SEQ @ ito)[[0, 2]]).tolist())
        branch_loss_actual += sf + st
        result = {"asset_id": asset["asset_id"], "branch_row0": index,
                  "terminal_y_max_error_pu": y_error,
                  "actual_terminal_y_pu": _matrix_pairs(actual_y),
                  "reference_terminal_y_pu": _matrix_pairs(target_y),
                  "from_power_pu": _pair(sf), "to_power_pu": _pair(st)}
        if balanced:
            target_sf = complex(row[13], row[14]) / base_mva
            target_st = complex(row[15], row[16]) / base_mva
            errors = [sf.real-target_sf.real, sf.imag-target_sf.imag, st.real-target_st.real, st.imag-target_st.imag]
            power_errors.extend(errors)
            branch_loss_reference += target_sf + target_st
            result.update(reference_from_power_pu=_pair(target_sf), reference_to_power_pu=_pair(target_st),
                          power_max_error_pu=_max(errors))
        branch_results.append(result)

    check("branch_positive_sequence_terminal_admittance", _max(y_errors), tol.admittance_pu)
    checks["asset_terminal_identity"] = {"passed": not identity_problems, "problems": identity_problems}
    kcl: dict[tuple[str, int], complex] = defaultdict(complex)
    constitutive, reported_power_errors = [], []
    for name in enabled:
        element = elements[name]
        reported_power_errors.extend(
            ((element["volts"] * element["currents"].conj() / 1000 - element["powers_kva"]) / sbase_kva).tolist()
        )
        for terminal, bus in enumerate(element["buses"]):
            for conductor in range(element["ncond"]):
                index = terminal * element["ncond"] + conductor
                node = int(element["nodes"][index])
                if node:
                    kcl[(_bus_name(bus), node)] += element["currents"][index] / ibase
        if name in passive_names:
            constitutive.extend(((_yprim(dss, element) @ element["volts"] - element["currents"]) / ibase).tolist())
    check("phase_node_kcl", _max(list(kcl.values())), tol.phase_kcl_pu,
          node_count=len(kcl), worst_nodes=[{"bus": bus, "node": node, "residual_current_pu": _pair(value)}
          for (bus, node), value in sorted(kcl.items(), key=lambda item: abs(item[1]), reverse=True)[:8]])
    check("passive_element_constitutive_current", _max(constitutive), tol.constitutive_current_pu)
    check("reported_power_matches_voltage_current", _max(reported_power_errors), tol.device_power_pu)

    device_errors: dict[str, list[complex]] = defaultdict(list)
    load_sum, generator_sum, shunt_sum = 0j, 0j, 0j
    load_reference_by_bus: dict[int, complex] = defaultdict(complex)
    shunt_reference_by_bus: dict[int, complex] = defaultdict(complex)
    gen_reference_by_row: dict[int, complex] = defaultdict(complex)
    for load in registry.get("loads", []):
        actual, _ = side_values([load["element"]], int(load["bus"]))
        target = complex(load["kw"], load["kvar"]) / sbase_kva
        if not balanced:
            dss.Loads.Name(load["element"].split(".", 1)[1])
            target = complex(dss.Loads.kW(), dss.Loads.kvar()) / sbase_kva
        load_sum += actual
        load_reference_by_bus[int(load["bus"])] += target
        device_errors["loads"].append(actual - target)
    for gen in registry.get("generators", []):
        actual, _ = side_values([gen["element"]], int(gen["bus"]))
        target = complex(gen["kw"], gen["kvar"]) / sbase_kva
        if not balanced:
            dss.Generators.Name(gen["element"].split(".", 1)[1])
            target = complex(dss.Generators.kW(), dss.Generators.kvar()) / sbase_kva
        generator_sum -= actual
        gen_reference_by_row[int(gen["gen_row0"])] += target
        device_errors["generators"].append(-actual - target)
    for shunt in registry.get("shunts", []):
        bus = int(shunt["bus"])
        actual, _ = side_values([shunt["element"]], bus)
        nominal = complex(shunt["gs_mw"], -shunt["bs_mvar"]) / base_mva
        target = nominal * float(np.mean(np.abs(bus_phase_pu[bus])**2))
        shunt_sum += actual
        shunt_reference_by_bus[bus] += nominal
        device_errors["shunts"].append(actual - target)
    source_into, source_current = side_values([source["element"]], int(source["bus"]))
    source_injection = -source_into
    source_target = sum(complex(gen_table[int(i), 1], gen_table[int(i), 2]) / base_mva for i in source["gen_rows0"])
    i_null.extend(np.abs((_SEQ @ source_current)[[0, 2]]).tolist())
    registry_power_errors: list[complex] = []
    for row in bus_table:
        bus = int(row[0])
        registry_power_errors.append(load_reference_by_bus[bus] - complex(row[2], row[3])/base_mva)
        registry_power_errors.append(shunt_reference_by_bus[bus] - complex(row[4], -row[5])/base_mva)
    slack_rows = {int(i) for i in source["gen_rows0"]}
    expected_slack_rows = {index for index, row in enumerate(gen_table)
                           if row[7] > 0 and int(row[0]) == int(source["bus"])}
    checks["source_reference_mapping"] = {"passed": slack_rows == expected_slack_rows,
                                           "expected_gen_rows0": sorted(expected_slack_rows),
                                           "registered_gen_rows0": sorted(slack_rows)}
    for index, row in enumerate(gen_table):
        if index not in slack_rows and row[7] > 0:
            registry_power_errors.append(gen_reference_by_row[index] - complex(row[1], row[2])/base_mva)
    if balanced:
        check("registry_injections_match_reference", _max(registry_power_errors), tol.device_power_pu)
    for kind in ("loads", "generators", "shunts"):
        check(f"{kind}_device_power", _max(device_errors[kind]), tol.device_power_pu)
    balance_error = source_injection + generator_sum - load_sum - shunt_sum - branch_loss_actual
    check("network_complex_power_balance", abs(balance_error), tol.device_power_pu)
    source_element = elements[source["element"].lower()]
    source_excitation = np.zeros((len(source_element["nodes"]), 3), dtype=complex)
    sequence_phase = np.column_stack((np.ones(3), _PHASE, _PHASE.conj()))
    source_bus = _bus_name(by_external[int(source["bus"])]["dss_bus"])
    for terminal, bus in enumerate(source_element["buses"]):
        for conductor in range(source_element["ncond"]):
            index = terminal * source_element["ncond"] + conductor
            node = source_element["nodes"][index]
            if node == 0:
                continue
            if node not in (1, 2, 3) or _bus_name(bus) != source_bus:
                raise ValueError("source must terminate at its declared phase bus and ground")
            source_excitation[index] = sequence_phase[node - 1]
    source_y_sequence_pu = source_excitation.conj().T @ _yprim(dss, source_element) @ source_excitation * zbase / 3
    source_z_sequence_pu = np.linalg.inv(source_y_sequence_pu)
    source_z_target = np.diag([complex(*assumptions[f"source_z{sequence}_pu"]) for sequence in (0, 1, 2)])
    check("source_sequence_impedance", _max(source_z_sequence_pu - source_z_target), tol.source_impedance_pu)
    if balanced:
        check("balanced_bus_voltage_magnitude", _max(vm_error), tol.voltage_magnitude_pu)
        check("balanced_bus_voltage_angle", _max(va_error), tol.voltage_angle_deg)
        check("balanced_branch_both_end_power", _max(power_errors), tol.branch_power_pu)
        check("balanced_source_injection", abs(source_injection - source_target), tol.device_power_pu)
        check("balanced_branch_losses", abs(branch_loss_actual - branch_loss_reference), tol.branch_power_pu)
        check("balanced_voltage_sequence_null", _max(v_null), tol.sequence_voltage_pu)
        check("balanced_current_sequence_null", _max(i_null), tol.sequence_current_pu)
    failed = [name for name, result in checks.items() if not result["passed"]]
    return {
        "contract": "compiled_opendss_independent_equivalence_v1", "balanced_reference_check": balanced,
        "passed": not failed, "failed_checks": failed, "checks": checks,
        "tolerances": asdict(tol), "bases": {"base_mva": base_mva, "base_kv_ll": kv_ll,
        "zbase_ohm": zbase, "ibase_ampere": float(ibase)},
        "power_accounting_pu": {"source_injection": _pair(source_injection),
        "generator_injection": _pair(generator_sum), "load_consumption": _pair(load_sum),
        "shunt_consumption": _pair(shunt_sum), "branch_losses": _pair(branch_loss_actual),
        "balance_residual": _pair(balance_error)},
        "source_sequence_impedance_pu": _matrix_pairs(source_z_sequence_pu),
        "buses": bus_results, "branches": branch_results,
    }
