"""Isolated DSS contexts; compilation never edits a user's existing circuit.

OpenDSS solves a snapshot with a fixed-point current-injection iteration. When
a large generator sits behind a weak, heavily charged corridor (the IEEE118
345 kV 8-9-10 path) the operating point is a repelling fixed point of that
iteration: seeded exactly at it, the error roughly doubles per iteration and
OpenDSS reports convergence on the constant-impedance branch beyond Vmaxpu.
``compile_model`` therefore checks every device against its control law,
and ``solve`` does the same for a circuit whose compile needed the fallback
(other circuits keep the plain OpenDSS solve): constant-PQ devices (Model=1) must deliver their setpoint, and
voltage-regulated generators (Model=3) must deliver their kW with Q inside
[Minkvar, Maxkvar], holding the average phase magnitude at Vpu unless Q sits
at the limit on the matching side. Only when a device fails, they solve the
compiled circuit by Newton-Raphson (each linear element's own YPrim, the
source's own Norton current, exact device currents, one Q unknown and one
average-magnitude equation per regulated generator, active-set limits), write
that state back and require OpenDSS's own iteration to accept it. Circuits
OpenDSS already solves are returned exactly as before.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping
import weakref

import numpy as np
import opendssdirect

# Devices must meet their control law to this relative accuracy (floor 1 kVA);
# a device on its constant-Z fallback misses it by orders of magnitude.
SETPOINT_RELATIVE_TOLERANCE = 1e-6
# OpenDSS's own convergence tolerance; one further step then reaches the
# floating-point floor (~2e-14 relative against the stiff source impedance).
NEWTON_STEP_TOLERANCE = 1e-12
# Converging solves need at most ~7 iterations; a longer one hands a shorter step
# to the homotopy instead.
NEWTON_MAX_ITERATIONS = 15
HOMOTOPY_MIN_STEP = 1e-3
ACTIVE_SET_MAX_ROUNDS = 30
_LINEAR_KINDS = {"line", "transformer", "capacitor", "reactor", "fault", "vsource"}
_PQ_KINDS = {"load": 1.0, "generator": -1.0}
# Contexts whose compile needed the Newton fallback. Only these capture seeds
# and may fall back in ``solve``; every circuit OpenDSS solves natively keeps
# the plain solve path (reading node arrays before each solve of such circuits
# was seen to disturb later global-engine IEEE14 solves in the same process).
_NEWTON_CIRCUITS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _complex(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[0::2] + 1j * array[1::2]


def _node_voltages(dss) -> dict[str, complex]:
    names = [str(name).lower() for name in dss.Circuit.YNodeOrder()]
    values = _complex(dss.Circuit.YNodeVArray())
    if len(values) != len(names) or not np.isfinite(values).all():
        return {}
    return dict(zip(names, values))


def _property(dss, element: str, name: str) -> float:
    dss.Text.Command(f"? {element}.{name}")
    return float(dss.Text.Result())


def _regulated_settings(dss, element: str) -> dict[str, float]:
    """Model=3 generator: kW, setpoint (V line-to-neutral) and limits (var)."""
    dss.Generators.Name(element.split(".", 1)[1])
    kv, phases = float(dss.Generators.kV()), int(dss.Generators.Phases())
    return {"p_w": float(dss.Generators.kW()) * 1000.0, "q_var": float(dss.Generators.kvar()) * 1000.0,
            "vtarget_v": _property(dss, element, "Vpu") * kv * 1000.0 / (math.sqrt(3.0) if phases > 1 else 1.0),
            "qmax_var": _property(dss, element, "Maxkvar") * 1000.0,
            "qmin_var": _property(dss, element, "Minkvar") * 1000.0}


def _generators(dss) -> list[str]:
    names, index = [], dss.Generators.First()
    while index > 0:
        names.append(f"generator.{dss.Generators.Name()}")
        index = dss.Generators.Next()
    return names


def _constant_pq_devices(dss) -> list[dict[str, Any]]:
    """Enabled loads and generators with their setpoints (kVA consumed)."""
    devices = []
    for kind, collection in (("load", dss.Loads), ("generator", dss.Generators)):
        index = collection.First()
        while index > 0:
            name = f"{kind}.{collection.Name()}"
            dss.Circuit.SetActiveElement(name)
            if dss.CktElement.Enabled():
                devices.append({"name": name, "kind": kind, "model": int(collection.Model()),
                                "setpoint_kva": _PQ_KINDS[kind] * complex(collection.kW(), collection.kvar())})
            index = collection.Next()
    return devices


def constant_pq_setpoint_deviation(dss) -> dict[str, Any]:
    """Largest relative gap between a constant-PQ device's power and its setpoint."""
    worst = {"relative_deviation": 0.0, "element": None}
    for device in _constant_pq_devices(dss):
        if device["model"] != 1:
            continue
        dss.Circuit.SetActiveElement(device["name"])
        actual = complex(*np.sum(np.asarray(dss.CktElement.Powers(), dtype=float).reshape(-1, 2), axis=0))
        target = device["setpoint_kva"]
        relative = abs(actual - target) / max(abs(target), 1.0)
        if not math.isfinite(relative) or relative > worst["relative_deviation"]:
            worst = {"relative_deviation": relative if math.isfinite(relative) else math.inf,
                     "element": device["name"], "actual_kva": [actual.real, actual.imag],
                     "setpoint_kva": [target.real, target.imag]}
    worst["passed"] = worst["relative_deviation"] <= SETPOINT_RELATIVE_TOLERANCE
    return worst


def regulated_generator_deviation(dss) -> dict[str, Any]:
    """Check every Model=3 generator against P, its Q limits and V regulation.

    ``relative_deviation`` is the worst of: active-power error (relative to
    kW, floor 1 kW), limit excursion (relative to the limit span, floor 1
    kvar), and the average-magnitude error of a generator that is not at the
    limit on the side that explains it (relative to the setpoint).
    """
    rows, worst = [], 0.0
    for element in _generators(dss):
        dss.Circuit.SetActiveElement(element)
        if not dss.CktElement.Enabled():
            continue
        dss.Generators.Name(element.split(".", 1)[1])
        if int(dss.Generators.Model()) != 3:
            continue
        settings = _regulated_settings(dss, element)
        dss.Circuit.SetActiveElement(element)
        ncond = int(dss.CktElement.NumConductors())
        nodes = list(dss.CktElement.NodeOrder())[:ncond]
        powers = _complex(dss.CktElement.Powers())[:ncond] * 1000.0
        volts = _complex(dss.CktElement.Voltages())[:ncond]
        live = [k for k, node in enumerate(nodes) if node]
        generated = -complex(sum(powers[k] for k in live))
        average = float(np.mean([abs(volts[k]) for k in live]))
        p_error = abs(generated.real - settings["p_w"]) / max(abs(settings["p_w"]), 1e3)
        span = max(abs(settings["qmax_var"]), abs(settings["qmin_var"]), 1e3)
        tolerance = SETPOINT_RELATIVE_TOLERANCE * span
        excursion = max(0.0, generated.imag - settings["qmax_var"], settings["qmin_var"] - generated.imag) / span
        at_max = generated.imag >= settings["qmax_var"] - tolerance
        at_min = generated.imag <= settings["qmin_var"] + tolerance
        v_error = (average - settings["vtarget_v"]) / settings["vtarget_v"]
        explained = (v_error < 0 and at_max) or (v_error > 0 and at_min)
        regulation = 0.0 if explained else abs(v_error)
        deviation = max(p_error, excursion, regulation)
        worst = max(worst, deviation) if math.isfinite(deviation) else math.inf
        rows.append({"element": element, "relative_deviation": deviation,
                     "state": "at_qmax" if at_max else "at_qmin" if at_min else "regulating",
                     "average_voltage_pu_of_setpoint": average / settings["vtarget_v"],
                     "generated_kvar": generated.imag / 1000.0})
    return {"relative_deviation": worst, "passed": worst <= SETPOINT_RELATIVE_TOLERANCE,
            "generator_count": len(rows),
            "at_limit": sorted(row["element"] for row in rows if row["state"] != "regulating"),
            "worst": max(rows, key=lambda row: row["relative_deviation"]) if rows else None}


def device_control_deviation(dss) -> dict[str, Any]:
    constant, regulated = constant_pq_setpoint_deviation(dss), regulated_generator_deviation(dss)
    return {"passed": constant["passed"] and regulated["passed"],
            "constant_pq": constant, "regulated_generators": regulated}


def _write_voltages(dss, voltage: np.ndarray) -> None:
    """Set OpenDSS's node-voltage vector (ground entry first) without solving."""
    pointer = dss.YMatrix.VVector()
    pointer[0], pointer[1] = 0.0, 0.0
    for index, value in enumerate(voltage, 1):
        pointer[2 * index], pointer[2 * index + 1] = value.real, value.imag


def _circuit_equations(dss, seeds: Mapping[str, complex]) -> dict[str, Any]:
    """Linear admittance, source Norton current and devices on the Y node order.

    The seeds are written first, so every element quantity read below is
    evaluated at a finite state even after a diverged OpenDSS iteration.
    """
    dss.YMatrix.BuildYMatrixD(2, True)
    names = [str(name).lower() for name in dss.Circuit.YNodeOrder()]
    missing = [name for name in names if name not in seeds]
    if missing:
        raise RuntimeError(f"Newton fallback lacks seeds for nodes: {missing[:6]}")
    seed = np.asarray([complex(seeds[name]) for name in names], dtype=complex)
    if not np.isfinite(seed).all():
        raise RuntimeError("Newton fallback seeds must be finite")
    _write_voltages(dss, seed)
    index = {name: i for i, name in enumerate(names)}
    size = len(names)
    admittance = np.zeros((size, size), dtype=complex)
    injection = np.zeros(size, dtype=complex)
    devices, regulated, bands = [], [], []
    for element in dss.Circuit.AllElementNames():
        dss.Circuit.SetActiveElement(element)
        if not dss.CktElement.Enabled():
            continue
        kind = element.split(".", 1)[0].lower()
        ncond = int(dss.CktElement.NumConductors())
        buses = [str(bus).split(".", 1)[0].lower() for bus in dss.CktElement.BusNames()]
        rows = [index[f"{buses[k // ncond]}.{node}"] if node else None
                for k, node in enumerate(dss.CktElement.NodeOrder())]
        if kind in _PQ_KINDS:
            collection = dss.Loads if kind == "load" else dss.Generators
            collection.Name(element.split(".", 1)[1])
            live = [row for row in rows if row is not None]
            phase_base = float(collection.kV()) * 1000.0 / (math.sqrt(3.0) if len(live) > 1 else 1.0)
            band = (float(collection.Vminpu()) * phase_base, float(collection.Vmaxpu()) * phase_base)
            if kind == "generator" and int(collection.Model()) == 3:
                if int(dss.CktElement.NumTerminals()) != 1 or not live:
                    raise NotImplementedError(f"Newton fallback supports wye regulated generators only: {element}")
                regulated.append({"element": element, "rows": live, **_regulated_settings(dss, element)})
                bands.append((element, live, *band, phase_base))
                continue
            if int(collection.Model()) != 1 or len(live) != 1:
                raise NotImplementedError(f"Newton fallback supports single-phase-to-ground constant-PQ devices only: {element}")
            devices.append((live[0], _PQ_KINDS[kind] * complex(collection.kW(), collection.kvar()) * 1000.0))
            bands.append((element, live, *band, phase_base))
            continue
        if kind not in _LINEAR_KINDS:
            raise NotImplementedError(f"Newton fallback has no model for {element}")
        count = len(rows)
        primitive = _complex(dss.CktElement.YPrim()).reshape((count, count), order="F")
        live = [(k, row) for k, row in enumerate(rows) if row is not None]
        for k, row in live:
            for m, column in live:
                admittance[row, column] += primitive[k, m]
        if kind == "vsource":
            # The source's own model gives I_into = YPrim V - I_norton at any V.
            norton = primitive @ _complex(dss.CktElement.Voltages()) - _complex(dss.CktElement.Currents())
            for k, row in live:
                injection[row] += norton[k]
    return {"names": names, "seed": seed, "admittance": admittance, "injection": injection,
            "devices": devices, "regulated": regulated, "bands": bands}


def _reactive_from_kcl(equations: Mapping[str, Any], voltage: np.ndarray) -> np.ndarray:
    """Q each regulated generator must supply for KCL at ``voltage`` (var).

    Exact for any solved state, and never inconsistent with the voltage seed,
    unlike a stored Q from another state (e.g. restoring after a fault).
    Nodes shared by several regulated generators are split evenly.
    """
    other = equations["admittance"] @ voltage - equations["injection"]
    for row, power in equations["devices"]:
        other[row] += np.conj(power / voltage[row])
    share = np.zeros(len(voltage))
    for gen in equations["regulated"]:
        share[gen["rows"]] += 1.0
    return np.asarray([float(np.sum(voltage[gen["rows"]] * np.conj(other[gen["rows"]]) / share[gen["rows"]]).imag)
                       for gen in equations["regulated"]])


def _band_violation(equations: Mapping[str, Any], voltage: np.ndarray) -> str | None:
    """Name the worst device outside its constant-power band, where OpenDSS changes model."""
    worst, message = 0.0, None
    for element, rows, low, high, base in equations["bands"]:
        for row in rows:
            magnitude = abs(voltage[row])
            excess = max(low - magnitude, magnitude - high) / base
            if excess > worst:
                worst = excess
                message = (f"{element} at node {equations['names'][row]} is at {magnitude / base:.3f} pu, "
                           f"outside [{low / base:g}, {high / base:g}] pu")
    return message


def _equilibrated_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve with row/column max-norm scaling (V in volts, Q in var, I in amperes)."""
    rows = 1.0 / np.maximum(np.max(np.abs(matrix), axis=1), 1e-300)
    scaled = matrix * rows[:, None]
    columns = 1.0 / np.maximum(np.max(np.abs(scaled), axis=0), 1e-300)
    return np.linalg.solve(scaled * columns[None, :], rhs * rows) * columns


class _NewtonStall(RuntimeError):
    """One Newton solve failed to converge; the homotopy may retry a shorter step."""


def _newton_fixed_set(equations: Mapping[str, Any], voltage: np.ndarray, reactive: np.ndarray,
                      free: np.ndarray, active: np.ndarray, *, kcl_offset: np.ndarray | None = None,
                      regulation_offset: np.ndarray | None = None,
                      residual_only: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    """Newton for one active set; ``free`` marks regulated generators solving for Q.

    The offsets shift the equations to F(x) - offset (the Newton homotopy).
    ``residual_only`` returns the unshifted KCL mismatch and every regulated
    generator's average-magnitude error at the given state instead of solving.
    """
    position = {int(node): k for k, node in enumerate(active)}
    y = equations["admittance"][np.ix_(active, active)]
    injection = equations["injection"][active]
    n = len(active)
    pq_rows = np.asarray([position[int(row)] for row, _ in equations["devices"]], dtype=int)
    pq_conj = np.conj(np.asarray([power for _, power in equations["devices"]], dtype=complex))
    groups = [np.asarray([position[int(row)] for row in gen["rows"]], dtype=int) for gen in equations["regulated"]]
    powers = np.asarray([gen["p_w"] for gen in equations["regulated"]], dtype=float)
    targets = np.asarray([gen["vtarget_v"] for gen in equations["regulated"]], dtype=float)
    kcl_offset = np.zeros(n, dtype=complex) if kcl_offset is None else kcl_offset
    regulation_offset = np.zeros(len(groups)) if regulation_offset is None else regulation_offset
    unknown = np.flatnonzero(free)
    m = len(unknown)
    base = np.block([[y.real, -y.imag], [y.imag, y.real]])
    voltage = voltage[active].copy()
    reactive = reactive.copy()
    converged = False
    for iteration in range(1, NEWTON_MAX_ITERATIONS + 1):
        rows = [pq_rows] + [group for group in groups]
        coefficients = [pq_conj] + [np.full(len(group), -(p - 1j * q) / len(group)) for group, p, q
                                    in zip(groups, powers, reactive)]
        all_rows = np.concatenate(rows) if rows else np.zeros(0, dtype=int)
        all_conj = np.concatenate(coefficients) if coefficients else np.zeros(0, dtype=complex)
        v = voltage[all_rows]
        magnitude2 = v.real ** 2 + v.imag ** 2
        device_current = np.zeros(n, dtype=complex)
        np.add.at(device_current, all_rows, all_conj * v / magnitude2)
        mismatch = y @ voltage + device_current - injection
        if residual_only:
            return mismatch, np.asarray([np.mean(np.abs(voltage[group])) - target
                                         for group, target in zip(groups, targets)]), 0
        mismatch = mismatch - kcl_offset
        d_real = all_conj * (1.0 / magnitude2 - 2.0 * v.real * v / magnitude2 ** 2)
        d_imag = all_conj * (1j / magnitude2 - 2.0 * v.imag * v / magnitude2 ** 2)
        jacobian = np.zeros((2 * n + m, 2 * n + m))
        jacobian[:2 * n, :2 * n] = base
        np.add.at(jacobian, (all_rows, all_rows), d_real.real)
        np.add.at(jacobian, (all_rows + n, all_rows), d_real.imag)
        np.add.at(jacobian, (all_rows, all_rows + n), d_imag.real)
        np.add.at(jacobian, (all_rows + n, all_rows + n), d_imag.imag)
        regulation = np.zeros(m)
        for k, g in enumerate(unknown):
            group = groups[g]
            vg = voltage[group]
            dq = (1j / len(group)) * vg / np.abs(vg) ** 2
            jacobian[group, 2 * n + k] += dq.real
            jacobian[group + n, 2 * n + k] += dq.imag
            regulation[k] = np.mean(np.abs(vg)) - targets[g] - regulation_offset[g]
            jacobian[2 * n + k, group] = vg.real / (len(group) * np.abs(vg))
            jacobian[2 * n + k, group + n] = vg.imag / (len(group) * np.abs(vg))
        try:
            step = _equilibrated_solve(jacobian, -np.r_[mismatch.real, mismatch.imag, regulation])
        except np.linalg.LinAlgError as exc:
            raise _NewtonStall("singular Newton Jacobian") from exc
        delta = step[:n] + 1j * step[n:2 * n]
        voltage += delta
        reactive[unknown] += step[2 * n:]
        if not (np.isfinite(voltage).all() and np.isfinite(reactive).all()):
            raise _NewtonStall("Newton fallback diverged to nonfinite node voltages")
        if converged:
            return voltage, reactive, iteration
        # Q enters KCL directly, so a converged voltage pins it; its own step
        # floor (~1e-4 var at condition ~1e9) is not a meaningful criterion.
        converged = np.max(np.abs(delta) / np.maximum(np.abs(voltage), 1e-9)) <= NEWTON_STEP_TOLERANCE
    raise _NewtonStall("Newton fallback did not converge")


def _newton_solution(equations: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve the circuit, enforcing regulated-generator limits by an active set.

    A Newton homotopy H(x, t) = F(x) - (1 - t) F(x0) carries the seed x0 (the
    state before the edit) to t = 1. The first attempt is the full step t = 1,
    so a nearby solution is found exactly as by plain Newton; a stalled step
    is retried shorter. Limits are enforced at every accepted t, so a
    generator reaches its limit before the demand on it grows past what the
    limit allows. Failure to reach t = 1 means no operating point was found
    along the path from the seed.
    """
    seed = equations["seed"]
    y = equations["admittance"]
    device_rows = [row for row, _ in equations["devices"]] + [row for gen in equations["regulated"] for row in gen["rows"]]
    active = np.flatnonzero(np.any(y != 0, axis=1) | np.isin(np.arange(len(seed)), device_rows))
    regulated = equations["regulated"]
    reactive = np.asarray([gen["q_var"] for gen in regulated], dtype=float)
    if equations.get("reactive_from_kcl", True):
        reactive = _reactive_from_kcl(equations, seed)
    qmax = np.asarray([gen["qmax_var"] for gen in regulated], dtype=float)
    qmin = np.asarray([gen["qmin_var"] for gen in regulated], dtype=float)
    targets = np.asarray([gen["vtarget_v"] for gen in regulated], dtype=float)
    tolerance = SETPOINT_RELATIVE_TOLERANCE * np.maximum(np.maximum(np.abs(qmax), np.abs(qmin)), 1e3)
    # Start from the limit a generator already sits on (the reference solve's).
    clamp = np.where(reactive >= qmax - tolerance, 1, np.where(reactive <= qmin + tolerance, -1, 0))
    reactive = np.where(clamp > 0, qmax, np.where(clamp < 0, qmin, reactive))
    kcl0, regulation0, _ = _newton_fixed_set(equations, seed, reactive, clamp == 0, active, residual_only=True)
    state, t, step_size = seed.copy(), 0.0, 1.0
    iterations = rounds = homotopy_steps = 0
    while t < 1.0:
        target = min(1.0, t + step_size)
        trial_state, trial_reactive, trial_clamp = state.copy(), reactive.copy(), clamp.copy()
        try:
            for _ in range(ACTIVE_SET_MAX_ROUNDS):
                trial_reactive = np.where(trial_clamp > 0, qmax, np.where(trial_clamp < 0, qmin, trial_reactive))
                solved, trial_reactive, used = _newton_fixed_set(
                    equations, trial_state, trial_reactive, trial_clamp == 0, active,
                    kcl_offset=(1.0 - target) * kcl0, regulation_offset=(1.0 - target) * regulation0)
                iterations += used
                rounds += 1
                trial_state = trial_state.copy()
                trial_state[active] = solved
                averages = np.asarray([np.mean(np.abs(trial_state[gen["rows"]])) for gen in regulated])
                goals = targets + (1.0 - target) * regulation0
                changed = False
                for g in range(len(regulated)):
                    if trial_clamp[g] == 0 and trial_reactive[g] > qmax[g] + tolerance[g]:
                        trial_clamp[g], changed = 1, True
                    elif trial_clamp[g] == 0 and trial_reactive[g] < qmin[g] - tolerance[g]:
                        trial_clamp[g], changed = -1, True
                    elif trial_clamp[g] > 0 and averages[g] > goals[g] * (1 + 1e-10):
                        trial_clamp[g], changed = 0, True
                    elif trial_clamp[g] < 0 and averages[g] < goals[g] * (1 - 1e-10):
                        trial_clamp[g], changed = 0, True
                if not changed:
                    break
            else:
                raise _NewtonStall("Regulated-generator limit set did not settle")
        except _NewtonStall:
            step_size /= 4.0
            if step_size < HOMOTOPY_MIN_STEP:
                raise RuntimeError(f"No operating point found along the homotopy from the seed (reached t={t:.4g})")
            continue
        state, reactive, clamp, t = trial_state, trial_reactive, trial_clamp, target
        homotopy_steps += 1
        step_size = min(1.0, step_size * 2.0)
    violation = _band_violation(equations, state)
    if violation is not None:
        raise RuntimeError(f"No operating point inside the devices' constant-power voltage band: {violation}")
    return state, reactive, {"newton_iterations": iterations, "active_set_rounds": rounds,
                             "homotopy_steps": homotopy_steps,
                             "generators_at_limit": [gen["element"] for gen, side in zip(regulated, clamp) if side]}


def _accept_state(dss, equations: Mapping[str, Any], voltage: np.ndarray, reactive: np.ndarray) -> None:
    """Hand OpenDSS the solved state; its own iteration and control laws must agree."""
    for gen, q in zip(equations["regulated"], reactive):
        # Restating the limits keeps OpenDSS from re-deriving them from kW/PF.
        dss.Text.Command(f"Edit {gen['element']} kvar={q / 1000.0:.17g} "
                         f"Maxkvar={gen['qmax_var'] / 1000.0:.17g} Minkvar={gen['qmin_var'] / 1000.0:.17g}")
    dss.YMatrix.BuildYMatrixD(2, True)
    if [str(name).lower() for name in dss.Circuit.YNodeOrder()] != equations["names"]:
        raise RuntimeError("OpenDSS node order changed during the Newton fallback")
    _write_voltages(dss, voltage)
    dss.YMatrix.SolutionInitialized(True)
    dss.YMatrix.LoadsNeedUpdating(True)
    dss.Solution.Solve()
    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS did not accept the Newton operating state")
    check = device_control_deviation(dss)
    if not check["passed"]:
        raise RuntimeError(f"Newton operating state violates a device control law: {check}")


def _regulated_reactive(dss) -> dict[str, float]:
    """Present Q (var) of every Model=3 generator, captured before a solve moves it."""
    reactive = {}
    for element in _generators(dss):
        dss.Generators.Name(element.split(".", 1)[1])
        if int(dss.Generators.Model()) == 3:
            reactive[element] = float(dss.Generators.kvar()) * 1000.0
    return reactive


def solve_constant_pq(dss, seeds: Mapping[str, complex],
                      reactive_seeds: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Newton solve from named node-voltage (and regulated-Q) seeds, then OpenDSS acceptance."""
    equations = _circuit_equations(dss, seeds)
    if reactive_seeds is not None:
        equations["reactive_from_kcl"] = False
        for gen in equations["regulated"]:
            gen["q_var"] = float(reactive_seeds.get(gen["element"].lower(), gen["q_var"]))
    voltage, reactive, receipt = _newton_solution(equations)
    _accept_state(dss, equations, voltage, reactive)
    return {"method": "newton_raphson_on_compiled_yprim_then_opendss_acceptance", **receipt,
            "opendss_acceptance_iterations": int(dss.Solution.Iterations())}


def solve(dss) -> None:
    if dss not in _NEWTON_CIRCUITS:
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError("OpenDSS snapshot did not converge")
        return
    seeds = _node_voltages(dss)
    dss.Solution.Solve()
    if dss.Solution.Converged() and device_control_deviation(dss)["passed"]:
        return
    if not seeds:
        raise RuntimeError("OpenDSS snapshot did not converge to the device control laws and no seed is available")
    solve_constant_pq(dss, seeds)


def _reference_seeds(dss, master: Path) -> tuple[dict[str, complex], dict[str, float]]:
    """Balanced phasors and generator Q of the exporter's positive-sequence reference solve."""
    reference = json.loads((master.parent / "positive_sequence_reference.json").read_text(encoding="utf-8"))
    # Exporter naming: generator row i is Generator.gen_{i+1:03d}.
    reactive = {f"generator.gen_{i + 1:03d}": float(row[2]) * 1e6 for i, row in enumerate(reference["gen"])}
    by_bus = {f"b{int(row[0])}": (float(row[7]), float(row[8])) for row in reference["bus"]}
    seeds = {}
    for name in dss.Circuit.AllBusNames():
        bus = str(name).lower()
        if bus not in by_bus:
            continue
        dss.Circuit.SetActiveBus(bus)
        magnitude, angle = by_bus[bus]
        for node in dss.Bus.Nodes():
            seeds[f"{bus}.{node}"] = magnitude * float(dss.Bus.kVBase()) * 1000.0 * np.exp(
                1j * np.deg2rad(angle - 120.0 * (int(node) - 1)))
    return seeds, reactive


def compile_model(master: str | Path):
    path = Path(master).resolve(strict=True)
    dss = opendssdirect.NewContext()
    dss.Basic.AllowChangeDir(False)
    dss.Text.Command(f'Compile "{path}"')
    if dss.Solution.Converged() and device_control_deviation(dss)["passed"]:
        return dss
    if not (path.parent / "positive_sequence_reference.json").is_file():
        raise RuntimeError(f"OpenDSS did not converge to the device control laws: {path}")
    solve_constant_pq(dss, *_reference_seeds(dss, path))
    _NEWTON_CIRCUITS.add(dss)
    return dss


def redistribute_load(dss, registry, *, bus: int, delta: float) -> dict:
    """Change phase demand while preserving the bus's total complex power.

    delta=0 restores the generated reference. No phasors are edited; the
    modified circuit is solved to obtain the new measurements.
    """
    if not math.isfinite(delta) or not -1.0 < delta < 1.0:
        raise ValueError("delta must be finite and strictly between -1 and 1")
    loads = [row for row in registry["loads"] if row["bus"] == bus]
    if len(loads) != 3 or {row["phase"] for row in loads} != {1, 2, 3}:
        raise ValueError("Selected bus must have three generated single-phase loads")
    factors = {1: 1 + delta, 2: 1 - delta, 3: 1.0}
    before = [sum(row[key] for row in loads) for key in ("kw", "kvar")]
    after = [sum(row[key] * factors[row["phase"]] for row in loads) for key in ("kw", "kvar")]
    if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) for a, b in zip(before, after)):
        raise ValueError("Load redistribution requires a balanced reference phase split")
    commands = []
    for row in loads:
        factor = factors[row["phase"]]
        command = f"Edit {row['element']} kW={row['kw']*factor:.16g} kvar={row['kvar']*factor:.16g}"
        dss.Text.Command(command)
        commands.append(command)
    solve(dss)
    return {"bus": bus, "delta": delta, "phase_factors": factors,
            "before_kw_kvar": before, "after_kw_kvar": after, "commands": commands,
            "method": "physical_phase_load_redistribution_then_three_phase_solve"}
