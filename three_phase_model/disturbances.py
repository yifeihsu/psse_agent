"""Physical steady-state high-resistance faults in isolated OpenDSS contexts.

The split retains the original lumped pi model: only series impedance is
divided, and half of the original full phase capacitance remains at each
external endpoint. This is a resistive surrogate, not an arcing/harmonic model.
No model files or caller registries are modified.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import math
from numbers import Integral
from typing import Any, Mapping

import numpy as np

from .runtime import solve
from .validation import _element, _yprim
from .voltage_bases import hif_resistance_class, hif_resistance_spec, impedance_base_ohm


def _matrix(values: np.ndarray) -> str:
    return "[" + " | ".join(" ".join(f"{value:.17g}" for value in row[:i + 1])
                              for i, row in enumerate(values)) + "]"


def _pairs(values: np.ndarray) -> list:
    return np.stack((values.real, values.imag), axis=-1).tolist()


def _unpair(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[..., 0] + 1j * array[..., 1]


def _bus(name: str) -> str:
    return str(name).split(".", 1)[0].lower()


def _registry_voltage_bases(registry: Mapping[str, Any]) -> dict[str, float]:
    """Read physical bus bases from the registry, never the slack reference base."""
    result = {}
    for row in registry["buses"]:
        name, kv = _bus(row["dss_bus"]), float(row["kv_ll"])
        if name in result or not math.isfinite(kv) or kv <= 0:
            raise ValueError("Registry buses require unique names and positive finite voltage bases")
        result[name] = kv
    return result


def _node_voltage_snapshot(dss: Any) -> dict[str, complex]:
    """Capture actual solved NodeV as a numerical continuation starting point."""
    names = list(dss.Circuit.YNodeOrder())
    values = np.asarray(dss.Circuit.YNodeVArray(), dtype=float)
    if values.shape != (2 * len(names),) or not np.isfinite(values).all():
        raise ValueError("Cannot capture finite solved node voltages for initialization")
    return {str(name).lower(): complex(*value) for name, value in zip(names, values.reshape((-1, 2)))}


def _solve_from_node_voltages(dss: Any, seeds: Mapping[str, complex]) -> None:
    """Initialize the solver, then solve the actual circuit equations normally.

    The documented DSS-Extensions YMatrix API exposes the internal voltage
    pointer (ground entry first), and BuildYMatrixD(2, True) allocates the
    whole-system matrix/VI arrays after topology changes. Names are read AFTER
    rebuilding because integer node ordering can change. No exported voltage
    or current measurements are written; all outputs come from the next solve.
    https://dss-extensions.org/OpenDSSDirect.py/opendssdirect.html
    """
    dss.YMatrix.BuildYMatrixD(2, True)
    names = list(dss.Circuit.YNodeOrder())
    missing = [name for name in names if str(name).lower() not in seeds]
    if missing:
        raise ValueError(f"Missing numerical voltage initialization for nodes: {missing}")
    pointer = dss.YMatrix.VVector()
    pointer[0], pointer[1] = 0.0, 0.0
    for index, name in enumerate(names, 1):
        value = complex(seeds[str(name).lower()])
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError("Nonfinite numerical voltage initialization")
        pointer[2 * index], pointer[2 * index + 1] = value.real, value.imag
    dss.YMatrix.SolutionInitialized(True)
    dss.YMatrix.LoadsNeedUpdating(True)
    solve(dss)


def _grounded_stamp(element: Mapping[str, Any], primitive: np.ndarray, buses: list[str]) -> np.ndarray:
    """Map actual terminal conductors to ABC buses; reference ground is fixed."""
    mapping = np.zeros((len(element["nodes"]), 3 * len(buses)))
    for terminal, name in enumerate(element["buses"]):
        for conductor in range(element["ncond"]):
            index = terminal * element["ncond"] + conductor
            node = int(element["nodes"][index])
            if node == 0:
                continue
            if node not in (1, 2, 3) or _bus(name) not in buses:
                raise ValueError(f"Unexpected conductor in {element['name']}")
            mapping[index, 3 * buses.index(_bus(name)) + node - 1] = 1
    return mapping.T @ primitive @ mapping


def eligible_hif_branch_rows(registry: Mapping[str, Any]) -> tuple[int, ...]:
    """Active same-voltage lines only, regardless of a source branch's TAP flag."""
    _registry_voltage_bases(registry)
    kv = {int(row["external_bus"]): float(row["kv_ll"]) for row in registry["buses"]}
    return tuple(int(row["branch_row0"]) for row in registry["branches"]
                 if str(row["dss_element"]).lower().startswith("line.") and row["status"] == 1
                 and math.isclose(kv[int(row["from_bus"])], kv[int(row["to_bus"])],
                                  rel_tol=1e-12, abs_tol=0))


def inject_midspan_hif(
    dss: Any, registry: Mapping[str, Any], assumptions: Mapping[str, Any], *,
    branch_row0: int, alpha: float = 0.5, phase: int = 1,
    resistance_pu: float | None = None, resistance_ohm: float | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Install and solve one midpoint phase-to-ground resistor; return a receipt.

    ``alpha`` is measured from the canonical from-terminal. Supply resistance
    in exactly one unit. Omitting both is accepted ONLY on a uniform registry
    (every bus shares one ``kv_ll``, i.e. the legacy normalized model), where
    the historical 10 pu default is retained; on a multi-voltage registry the
    default would silently mean a different physical resistance on every line
    (476 ohm at 69 kV, 19.04 ohm at 13.8 kV), so a ``ValueError`` is raised
    before any circuit mutation. Zbase = local endpoint kV_LL**2 /
    MVA_three_phase. The receipt records both units, the local bases and the
    physical-ohm ``resistance_class``. The receipt's ``branch_overrides`` is
    directly consumable by the registry-driven measurement extractor.
    ``enabled=False`` installs the exactly equivalent no-fault split control.
    """
    if isinstance(branch_row0, bool) or not isinstance(branch_row0, Integral):
        raise ValueError("branch_row0 must be an integer")
    if isinstance(phase, bool) or not isinstance(phase, Integral) or phase not in (1, 2, 3):
        raise ValueError("phase must be one of 1, 2, 3")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be boolean")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    if resistance_pu is not None and resistance_ohm is not None:
        raise ValueError("Supply exactly one resistance unit: resistance_pu or resistance_ohm")
    resistance_defaulted = resistance_pu is None and resistance_ohm is None
    if resistance_defaulted:
        registry_kv = list(_registry_voltage_bases(registry).values())
        if not all(math.isclose(kv, registry_kv[0], rel_tol=1e-12, abs_tol=0) for kv in registry_kv):
            raise ValueError("resistance_ohm or resistance_pu is required for a multi-voltage registry")
    resistance_input_unit = "ohm" if resistance_ohm is not None else "pu"
    resistance_value = resistance_ohm if resistance_ohm is not None else (
        10.0 if resistance_pu is None else resistance_pu)
    if isinstance(resistance_value, (bool, np.bool_)):
        raise ValueError("Resistance must be finite and positive, not boolean")
    resistance_value = float(resistance_value)
    if not math.isfinite(resistance_value) or resistance_value <= 0:
        raise ValueError("Resistance must be finite and positive")
    matches = [row for row in registry["branches"] if row["branch_row0"] == branch_row0]
    if len(matches) != 1 or branch_row0 not in eligible_hif_branch_rows(registry):
        raise ValueError("Midspan HIF requires one active same-voltage line asset; transformers are unsupported")
    branch = matches[0]
    by_bus = {row["external_bus"]: row for row in registry["buses"]}
    from_bus = str(by_bus[branch["from_bus"]]["dss_bus"]).lower()
    to_bus = str(by_bus[branch["to_bus"]]["dss_bus"]).lower()
    voltage_bases = _registry_voltage_bases(registry)
    kv, base = voltage_bases[from_bus], float(assumptions["base_mva"])
    frequency = float(assumptions["frequency_hz"])
    if not np.isfinite([kv, base, frequency]).all() or min(kv, base, frequency) <= 0:
        raise ValueError("Voltage, power and frequency bases must be finite and positive")
    if not math.isclose(float(dss.Solution.Frequency()), frequency, rel_tol=0, abs_tol=1e-9):
        raise ValueError("The compiled circuit frequency disagrees with assumptions")
    original = str(branch["dss_element"])
    element = _element(dss, original)
    if not element["enabled"]:
        raise ValueError("The original line must be enabled before injection")
    if element["ncond"] != 3 or element["nterm"] != 2 or element["nodes"].tolist() != [1, 2, 3, 1, 2, 3]:
        raise ValueError("The injector requires an explicit ABC line without neutral conductors")
    if list(map(_bus, element["buses"])) != [from_bus, to_bus]:
        raise ValueError("The active line endpoints disagree with the registry")
    healthy_voltages = _node_voltage_snapshot(dss)
    original_y = _grounded_stamp(element, _yprim(dss, element), [from_bus, to_bus])
    dss.Lines.Name(original.split(".", 1)[1])
    length = float(dss.Lines.Length())
    r = np.asarray(dss.Lines.RMatrix(), dtype=float).reshape((3, 3))
    x = np.asarray(dss.Lines.XMatrix(), dtype=float).reshape((3, 3))
    c = np.asarray(dss.Lines.CMatrix(), dtype=float).reshape((3, 3))
    if not math.isfinite(length) or length <= 0 or not all(np.isfinite(m).all() for m in (r, x, c)):
        raise ValueError("Invalid original line length or phase matrices")
    names = {str(name).lower() for name in dss.Circuit.AllElementNames()}
    bus_names = {str(name).lower() for name in dss.Circuit.AllBusNames()}
    ordinal = 1
    while True:
        prefix = f"hif_b{branch_row0 + 1:04d}_{ordinal:03d}"
        hidden = prefix + "_bus"
        segment_from, segment_to = f"Line.{prefix}_from", f"Line.{prefix}_to"
        fault = f"Fault.{prefix}_fault"
        if hidden not in bus_names and all(name.lower() not in names for name in (segment_from, segment_to, fault)):
            break
        ordinal += 1
    zbase = impedance_base_ohm(kv, base)
    resistance_ohm = resistance_value if resistance_input_unit == "ohm" else resistance_value * zbase
    resistance_pu = resistance_ohm / zbase
    resistance_spec = hif_resistance_spec(resistance_ohm, kv, base)
    voltage_bases[hidden] = kv
    zero = _matrix(np.zeros((3, 3)))
    commands = [f"Edit {original} Enabled=no"]
    # Like retains the declared length units and line-frequency model. Setting
    # the full matrices explicitly retains coupling while eliminating all C
    # on the internal sections. Only Length scales the series impedance.
    for segment, bus1, bus2, fraction in (
        (segment_from, element["buses"][0], f"{hidden}.1.2.3", alpha),
        (segment_to, f"{hidden}.1.2.3", element["buses"][1], 1 - alpha),
    ):
        commands.append(
            f"New {segment} Like={original.split('.', 1)[1]} Bus1={bus1} Bus2={bus2} "
            f"Length={length * fraction:.17g} Rmatrix={_matrix(r)} Xmatrix={_matrix(x)} "
            f"Cmatrix={zero} Enabled=yes"
        )
    created = [segment_from, segment_to]
    external = {
        "from": {"element": segment_from, "terminal": 1, "charging_elements": []},
        "to": {"element": segment_to, "terminal": 2, "charging_elements": []},
    }
    if np.any(c != 0):
        # Line CMatrix is nF per unit length; Capacitor Cmatrix is microfarads.
        cap_microfarad = c * length / 2000.0
        for end, bus in (("from", from_bus), ("to", to_bus)):
            capacitor = f"Capacitor.{prefix}_{end}_charging"
            commands.append(
                f"New {capacitor} Phases=3 Bus1={bus}.1.2.3 Bus2={bus}.0.0.0 "
                f"Conn=wye kV={kv:.17g} Cmatrix={_matrix(cap_microfarad)} "
                f"BaseFreq={frequency:.17g} Enabled=yes"
            )
            external[end]["charging_elements"].append({"element": capacitor, "terminal": 1})
            created.append(capacitor)
    commands.append(f"New {fault} Phases=1 Bus1={hidden}.{phase} Bus2={hidden}.0 "
                    f"R={resistance_ohm:.17g} %StdDev=0 Temporary=no MinAmps=0 "
                    "Enabled=no")
    created.append(fault)
    # CalcVoltageBases also provides a deterministic initializer for standalone
    # DSS replay. In the live context we additionally seed the exact preceding
    # healthy voltages, preventing a new node from choosing a low-voltage root.
    commands.append("CalcVoltageBases")
    numerical_seeds = dict(healthy_voltages)
    for node in (1, 2, 3):
        numerical_seeds[f"{hidden}.{node}"] = ((1-alpha) * healthy_voltages[f"{from_bus}.{node}"]
                                                + alpha * healthy_voltages[f"{to_bus}.{node}"])
    replay_commands = commands + ["Solve"]
    if enabled:
        replay_commands += [f"Edit {fault} Enabled=yes", "Solve"]
    restore_commands = ([f"Edit {name} Enabled=no" for name in created]
                        + [f"Edit {original} Enabled=yes", "CalcVoltageBases", "Solve"])
    receipt = {
        "contract": "midspan_resistive_hif_exact_pi_v1", "circuit_name": dss.Circuit.Name(),
        "physical_model": "steady_state_phase_to_ground_resistor_no_arcing_or_harmonic_claim",
        "asset_id": branch["asset_id"], "branch_row0": int(branch_row0),
        "line_index1": int(branch_row0) + 1, "from_bus": branch["from_bus"], "to_bus": branch["to_bus"],
        "original_element": original, "hidden_bus": hidden, "phase": int(phase), "alpha": alpha,
        "resistance_pu": resistance_pu, "resistance_ohm": resistance_ohm, "zbase_ohm": zbase,
        "resistance_input_unit": resistance_input_unit,
        "resistance_defaulted": resistance_defaulted,
        # Physical-ohm class of the engine resistor; on a normalized 1 kV base
        # the ohms are model ohms, so the class is only physical for declared bases.
        "resistance_class": hif_resistance_class(resistance_ohm),
        "resistance_class_scope": "classification of resistance_ohm on the registry's declared local base",
        "local_base_kv_ll": kv, "local_voltage_base_ln_v": kv * 1000 / math.sqrt(3),
        "local_current_base_a": base * 1000 / (math.sqrt(3) * kv), "base_mva": base,
        "resistance_physical_spec": resistance_spec,
        "fault_element": fault, "fault_enabled": enabled, "restored": False,
        "segments": {"from": segment_from, "to": segment_to}, "external_terminals": copy.deepcopy(external),
        "branch_overrides": {branch["asset_id"]: copy.deepcopy(external)},
        "created_elements": created, "commands": replay_commands, "restore_commands": restore_commands,
        "numerical_initialization": {
            "method": "actual_healthy_node_voltages_with_exact_no_fault_series_interpolation",
            "source": "previous physically solved circuit before topology change",
            "solver_api": "YMatrix.BuildYMatrixD(2,True),VVector,SolutionInitialized(True),Solve",
            "standalone_dss_initializer": "CalcVoltageBases then no-fault Solve before enabling resistor",
            "node_voltage_seeds_v": {name: [value.real, value.imag] for name, value in numerical_seeds.items()},
            "measurement_values_are_not_overwritten": True,
        },
        "original_terminal_admittance_siemens": _pairs(original_y),
        "original_line": {"length": length, "rmatrix": r.tolist(), "xmatrix": x.tolist(), "cmatrix_nf": c.tolist()},
        "original_endpoint_bus_names": [from_bus, to_bus],
    }
    try:
        for command in commands:
            dss.Text.Command(command)
        _solve_from_node_voltages(dss, numerical_seeds)
        after = _node_voltage_snapshot(dss)
        null_deviation = max(abs(after[name] - value) / (voltage_bases[_bus(name)] * 1000 / math.sqrt(3))
                             for name, value in healthy_voltages.items() if _bus(name) in voltage_bases)
        hidden_deviation = max(abs(after[f"{hidden}.{node}"] - numerical_seeds[f"{hidden}.{node}"])
                               / (kv * 1000 / math.sqrt(3)) for node in (1, 2, 3))
        receipt["numerical_initialization"]["no_fault_external_voltage_max_deviation_pu"] = null_deviation
        receipt["numerical_initialization"]["no_fault_hidden_voltage_max_deviation_pu"] = hidden_deviation
        receipt["numerical_initialization"]["voltage_normalization"] = "each_node_declared_local_phase_voltage_base"
        if max(null_deviation, hidden_deviation) > 1e-6:
            raise RuntimeError("No-fault split converged away from the preceding healthy operating solution")
        if enabled:
            before_fault = _node_voltage_snapshot(dss)
            dss.Text.Command(f"Edit {fault} Enabled=yes")
            _solve_from_node_voltages(dss, before_fault)
    except Exception as failure:
        present = {name.lower() for name in dss.Circuit.AllElementNames()}
        for name in created:
            if name.lower() in present:
                dss.Text.Command(f"Edit {name} Enabled=no")
        dss.Text.Command(f"Edit {original} Enabled=yes")
        # Disabled hidden buses can remain in OpenDSS's node allocation. Extra
        # seeds are harmless; retaining the hidden seed avoids masking the
        # original failure with an unrelated missing-initialization error.
        try:
            _solve_from_node_voltages(dss, numerical_seeds)
        except Exception as restore_failure:
            failure.add_note(f"restoring the original line also failed: {restore_failure}")
        raise
    return receipt


def set_hif_enabled(dss: Any, receipt: dict[str, Any], enabled: bool) -> None:
    """Toggle only the physical resistor; keep the same paired split model."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be boolean")
    if receipt.get("restored") or dss.Circuit.Name() != receipt["circuit_name"]:
        raise ValueError("The HIF split has been restored or belongs to another circuit")
    previous = _node_voltage_snapshot(dss)
    dss.Text.Command(f"Edit {receipt['fault_element']} Enabled={'yes' if enabled else 'no'}")
    _solve_from_node_voltages(dss, previous)
    receipt["fault_enabled"] = enabled


def restore_midspan_hif(dss: Any, receipt: dict[str, Any]) -> None:
    """Disable the added primitives and restore the untouched original line."""
    if dss.Circuit.Name() != receipt["circuit_name"]:
        raise ValueError("The HIF receipt belongs to another circuit")
    previous = _node_voltage_snapshot(dss)
    for command in receipt["restore_commands"][:-1]:
        dss.Text.Command(command)
    _solve_from_node_voltages(dss, previous)
    receipt["restored"], receipt["fault_enabled"] = True, False


def audit_disturbed_circuit(
    dss: Any, receipt: Mapping[str, Any], registry: Mapping[str, Any],
    assumptions: Mapping[str, Any], *, tolerance_pu: float = 1e-7,
) -> dict[str, Any]:
    """Check all active nodes/elements, including the hidden fault node.

    Checks use actual engine currents, voltages, power and YPrim. Disabled
    original/created elements are excluded from KCL and energy sums, but their
    expected status is checked separately. No privileged localization labels
    are required to compute the all-node equations.
    """
    if not math.isfinite(tolerance_pu) or tolerance_pu <= 0:
        raise ValueError("tolerance_pu must be finite and positive")
    base = float(assumptions["base_mva"])
    if not math.isfinite(base) or base <= 0:
        raise ValueError("Power base must be finite and positive")
    voltage_bases = _registry_voltage_bases(registry)
    endpoint_bases = [voltage_bases[_bus(name)] for name in receipt["original_endpoint_bus_names"]]
    if not math.isclose(*endpoint_bases, rel_tol=1e-12, abs_tol=0):
        raise ValueError("HIF split endpoints must have the same voltage base")
    voltage_bases[_bus(receipt["hidden_bus"])] = endpoint_bases[0]
    current_bases = {name: base * 1000 / (math.sqrt(3) * kv) for name, kv in voltage_bases.items()}
    sbase = base * 1e6
    elements = {name.lower(): _element(dss, name) for name in dss.Circuit.AllElementNames()}
    active = {name: element for name, element in elements.items() if element["enabled"]}
    expected = {str(registry["source"]["element"]).lower()}
    for row in registry["branches"]:
        if row["status"]:
            expected.add(str(row["dss_element"]).lower())
            for end in ("from", "to"):
                for item in row.get("charging_elements", {}).get(end, []):
                    expected.add(str(item["element"] if isinstance(item, Mapping) else item).lower())
    for family in ("loads", "generators", "shunts"):
        expected.update(str(row["element"]).lower() for row in registry.get(family, []))
    if not receipt["restored"]:
        expected.remove(receipt["original_element"].lower())
        expected.update(name.lower() for name in receipt["created_elements"])
        if not receipt["fault_enabled"]:
            expected.remove(receipt["fault_element"].lower())
    checks: dict[str, Any] = {}

    def check(name: str, error: float, limit: float = tolerance_pu) -> None:
        checks[name] = {"passed": bool(math.isfinite(error) and error <= limit), "max_error_pu": float(error), "tolerance_pu": limit}

    checks["solution_converged"] = {"passed": bool(dss.Solution.Converged())}
    checks["active_element_coverage"] = {"passed": set(active) == expected,
                                          "missing": sorted(expected - set(active)), "unexpected": sorted(set(active) - expected)}
    checks["original_line_status"] = {"passed": elements[receipt["original_element"].lower()]["enabled"] is bool(receipt["restored"])}
    checks["created_element_presence"] = {"passed": all(name.lower() in elements for name in receipt["created_elements"])}
    kcl: dict[tuple[str, int], complex] = defaultdict(complex)
    power_sum, passive_error = 0j, 0.0
    for name, element in active.items():
        power_sum += sum(element["powers_kva"]) * 1000
        for terminal, bus in enumerate(element["buses"]):
            for conductor in range(element["ncond"]):
                index = terminal * element["ncond"] + conductor
                node = int(element["nodes"][index])
                if node:
                    kcl[(_bus(bus), node)] += element["currents"][index]
        if name.startswith(("line.", "transformer.", "capacitor.", "reactor.", "fault.")):
            terminal_bases = np.repeat([current_bases[_bus(bus)] for bus in element["buses"]], element["ncond"])
            passive_error = max(passive_error, float(np.max(
                np.abs(element["currents"] - _yprim(dss, element) @ element["volts"]) / terminal_bases)))
    check("all_active_phase_node_kcl", max((abs(value) / current_bases[bus]
                                            for (bus, _), value in kcl.items()), default=0.0))
    check("active_passive_primitive_equations", passive_error)
    check("network_complex_power_balance", abs(power_sum) / sbase)
    fault = elements[receipt["fault_element"].lower()]
    if fault["enabled"]:
        voltage, current = complex(fault["volts"][0]), complex(fault["currents"][0])
        resistance = float(receipt["resistance_ohm"])
        check("fault_ohms_law", abs(current - voltage / resistance) / current_bases[_bus(receipt["hidden_bus"])])
        check("fault_resistive_power", abs(sum(fault["powers_kva"]) * 1000 - abs(voltage)**2 / resistance) / sbase)
    else:
        voltage, current = 0j, 0j
        check("fault_ohms_law", 0.0)
        check("fault_resistive_power", 0.0)
    if not receipt["restored"]:
        buses = [*receipt["original_endpoint_bus_names"], receipt["hidden_bus"]]
        assembled = np.zeros((9, 9), dtype=complex)
        for name in receipt["created_elements"]:
            if name == receipt["fault_element"]:
                continue
            element = elements[name.lower()]
            assembled += _grounded_stamp(element, _yprim(dss, element), buses)
        reduced = assembled[:6, :6] - assembled[:6, 6:] @ np.linalg.solve(assembled[6:, 6:], assembled[6:, :6])
        check("no_fault_split_full_abc_admittance", float(np.max(np.abs(reduced - _unpair(receipt["original_terminal_admittance_siemens"]))))
              * impedance_base_ohm(endpoint_bases[0], base))
    else:
        original = elements[receipt["original_element"].lower()]
        actual = _grounded_stamp(original, _yprim(dss, original), receipt["original_endpoint_bus_names"])
        check("restored_original_full_abc_admittance", float(np.max(np.abs(actual - _unpair(receipt["original_terminal_admittance_siemens"]))))
              * impedance_base_ohm(endpoint_bases[0], base))
    failed = [name for name, result in checks.items() if not result["passed"]]
    return {"contract": "full_phase_circuit_resistive_hif_equations_v1", "passed": not failed,
            "failed_checks": failed, "checks": checks, "active_element_count": len(active),
            "active_phase_node_count": len(kcl), "hidden_bus": receipt["hidden_bus"],
            "fault_enabled": fault["enabled"], "fault_current_a": [current.real, current.imag],
            "fault_voltage_v": [voltage.real, voltage.imag], "resistance_ohm": receipt["resistance_ohm"],
            "fault_real_power_w": float((voltage * current.conjugate()).real),
            "local_base_kv_ll": endpoint_bases[0],
            "normalization": "per_node_and_terminal_local_voltage_and_current_bases_shared_three_phase_power_base",
            "node_current_bases_a": current_bases}
