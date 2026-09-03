"""Per-phase branch-current telemetry analysis for IEEE-14 diagnostics.

This module is pure numpy: it never touches OpenDSS.  It consumes the
``three_phase_branch_currents`` channel exported by
``IEEE_14_OpenDSS.export_measurement_series`` together with the existing
``three_phase_voltages`` channel and provides two physical analyses:

* **Unbalance-source localization.**  Kirchhoff's current law at every bus
  turns terminal currents into the per-phase current supplied by that bus's
  shunt elements.  A balanced constant-power load draws equal per-phase power
  regardless of voltage unbalance, so the bus whose per-phase shunt power is
  most uneven is the unbalance source.  Ranking by negative-sequence voltage
  alone cannot do this: negative-sequence voltage peaks at electrically weak
  buses, not at the injecting load.

* **Two-terminal HIF localization.**  A mid-span shunt fault draws current
  that enters the line from both ends, so the per-phase differential current
  (sum of the two terminal currents minus the modeled charging current)
  singles out both the faulted line and the faulted phase.  With the terminal
  voltages known, the fault position ``alpha`` is the point where the
  voltages computed from either end agree, and the fault resistance follows
  from the fault-point voltage and the differential current.

Sign convention: every terminal current is the current flowing *into* the
branch from that terminal, matching ``CktElement.Currents()`` in OpenDSS.
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

from IEEE_14_OpenDSS.constants import BRANCH_ORDER, BUS_ORDER

PHASES = ("A", "B", "C")
ROTATION = cmath.exp(2j * math.pi / 3.0)
BRANCH_CURRENT_CHANNEL = "three_phase_branch_currents"
BRANCH_CURRENT_SIGMA_KEY = "branch_current_sigma_pu"
#: Default per-component (real/imaginary) noise standard deviation assumed for
#: per-phase branch-current phasors in per-unit on the 100 MVA system base.
DEFAULT_BRANCH_CURRENT_SIGMA_PU = 1e-3
#: A line-differential above this many sigmas is treated as HIF-like evidence.
DIFFERENTIAL_DETECTION_SIGMAS = 6.0
#: Relative per-phase shunt-power spread floor: buses whose mean shunt power is
#: below this many per-unit are normalized against the floor, so an unloaded
#: bus cannot rank as an unbalance source through a 0/0 ratio.
DEFAULT_SHUNT_POWER_FLOOR_PU = 0.02
#: Fault positions this close to a terminal cannot be told apart from a gross
#: error on that terminal's current sensor using the line's own phasors.
ENDPOINT_AMBIGUITY_ALPHA = 0.02
S_BASE_MVA = 100.0
KV_LL_BASE = 1.0
TERMINAL_CURRENT_METHOD = "terminal_current_differential"
SHUNT_POWER_SPREAD_METHOD = "per_phase_shunt_power_spread"


# --------------------------------------------------------------------- basics


def phasor(magnitude: Any, angle_deg: Any) -> complex | None:
    try:
        mag = float(magnitude)
        ang = float(angle_deg)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(mag) and math.isfinite(ang)):
        return None
    return cmath.rect(mag, math.radians(ang))


def sequence_components(a: complex, b: complex, c: complex) -> tuple[complex, complex, complex]:
    """Return (zero, positive, negative) sequence components of an ABC set."""
    zero = (a + b + c) / 3.0
    positive = (a + ROTATION * b + ROTATION * ROTATION * c) / 3.0
    negative = (a + ROTATION * ROTATION * b + ROTATION * c) / 3.0
    return zero, positive, negative


def bus_number(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip().lower()
    if text.startswith("bus"):
        text = text[3:]
    elif text.startswith("b"):
        text = text[1:]
    text = text.split(".")[0]
    try:
        return int(text)
    except ValueError:
        return None


def _three_phasors(magnitudes: Any, angles: Any) -> list[complex] | None:
    if not isinstance(magnitudes, Sequence) or isinstance(magnitudes, (str, bytes)):
        return None
    if not isinstance(angles, Sequence) or isinstance(angles, (str, bytes)):
        return None
    if len(magnitudes) < 3 or len(angles) < 3:
        return None
    values = [phasor(magnitudes[index], angles[index]) for index in range(3)]
    if any(value is None for value in values):
        return None
    return [complex(value) for value in values]  # type: ignore[arg-type]


def voltage_rows_to_phasors(rows: Any) -> dict[int, list[complex]]:
    """Map 1-based bus number -> [Va, Vb, Vc] line-neutral phasors (pu)."""
    out: dict[int, list[complex]] = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return out
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        bus = bus_number(item.get("bus"))
        values = _three_phasors(item.get("vln_pu"), item.get("ang_deg"))
        if bus is None or values is None:
            continue
        out[bus] = values
    return out


def branch_current_rows_to_phasors(rows: Any) -> dict[int, dict[str, Any]]:
    """Map branch_row0 -> terminal current phasors (pu, into the branch)."""
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return out
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        row0 = item.get("branch_row0")
        if row0 is None:
            name = str(item.get("branch") or "").lower()
            matches = [index for index, elem in enumerate(BRANCH_ORDER) if elem.lower() == name]
            row0 = matches[0] if matches else None
        try:
            row0 = int(row0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        i_from = _three_phasors(item.get("i_from_pu"), item.get("ang_from_deg"))
        i_to = _three_phasors(item.get("i_to_pu"), item.get("ang_to_deg"))
        from_bus = bus_number(item.get("from_bus"))
        to_bus = bus_number(item.get("to_bus"))
        if i_from is None or i_to is None or from_bus is None or to_bus is None:
            continue
        out[row0] = {
            "branch": str(item.get("branch") or (BRANCH_ORDER[row0] if 0 <= row0 < len(BRANCH_ORDER) else row0)),
            "from_bus": from_bus,
            "to_bus": to_bus,
            "i_from": i_from,
            "i_to": i_to,
        }
    return out


def branch_current_rows_valid(rows: Any, *, expected_branches: int | None = None) -> bool:
    parsed = branch_current_rows_to_phasors(rows)
    if not parsed:
        return False
    if expected_branches is not None and len(parsed) != int(expected_branches):
        return False
    for item in parsed.values():
        for value in (*item["i_from"], *item["i_to"]):
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                return False
    return True


@lru_cache(maxsize=1)
def case14_line_parameters() -> dict[int, dict[str, Any]]:
    """Per-branch series impedance and total charging susceptance (pu)."""
    from pypower.api import case14  # type: ignore
    from pypower.idx_brch import BR_B, BR_R, BR_X, F_BUS, T_BUS, TAP  # type: ignore

    ppc = case14()
    branch = ppc["branch"]
    out: dict[int, dict[str, Any]] = {}
    for row0 in range(branch.shape[0]):
        out[row0] = {
            "r": float(branch[row0, BR_R]),
            "x": float(branch[row0, BR_X]),
            "b": float(branch[row0, BR_B]),
            "from_bus": int(branch[row0, F_BUS]),
            "to_bus": int(branch[row0, T_BUS]),
            "is_line": bool(float(branch[row0, TAP]) == 0.0),
            "dss_element": BRANCH_ORDER[row0] if row0 < len(BRANCH_ORDER) else None,
        }
    return out


def zbase_ohm(*, base_mva: float = S_BASE_MVA, kv_ll: float = KV_LL_BASE) -> float:
    return (float(kv_ll) * 1000.0) ** 2 / (float(base_mva) * 1e6)


# ------------------------------------------------------------ noise helpers


def add_branch_current_noise(
    rows: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
    sigma_pu: float,
) -> list[dict[str, Any]]:
    """Return a copy of the rows with Gaussian noise on each phasor component."""
    sigma = float(sigma_pu)
    noisy: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        if sigma > 0.0:
            for mag_key, ang_key in (("i_from_pu", "ang_from_deg"), ("i_to_pu", "ang_to_deg")):
                values = _three_phasors(row.get(mag_key), row.get(ang_key))
                if values is None:
                    continue
                perturbed = [
                    value + complex(rng.normal(0.0, sigma), rng.normal(0.0, sigma))
                    for value in values
                ]
                row[mag_key] = [float(abs(value)) for value in perturbed]
                row[ang_key] = [float(math.degrees(cmath.phase(value))) for value in perturbed]
        noisy.append(row)
    return noisy


def balanced_branch_current_control(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Balanced telemetry null: mean magnitude, phase-A angle, exact 120 degree spacing."""
    balanced: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        ok = True
        for mag_key, ang_key in (("i_from_pu", "ang_from_deg"), ("i_to_pu", "ang_to_deg")):
            magnitudes = row.get(mag_key)
            angles = row.get(ang_key)
            if (
                not isinstance(magnitudes, Sequence)
                or not isinstance(angles, Sequence)
                or len(magnitudes) != 3
                or len(angles) != 3
            ):
                ok = False
                break
            magnitude = float(sum(float(value) for value in magnitudes) / 3.0)
            phase_a = float(angles[0])
            row[mag_key] = [magnitude, magnitude, magnitude]
            row[ang_key] = [phase_a, phase_a - 120.0, phase_a + 120.0]
        if ok:
            balanced.append(row)
    return balanced


# ------------------------------------------------- unbalance localization


def bus_shunt_injections(
    voltages: Mapping[int, Sequence[complex]],
    currents: Mapping[int, Mapping[str, Any]],
) -> dict[int, list[complex]]:
    """Per-bus, per-phase current supplied by shunt elements (KCL over branches)."""
    injections: dict[int, list[complex]] = {
        bus_number(bus): [0j, 0j, 0j] for bus in BUS_ORDER  # type: ignore[misc]
    }
    for item in currents.values():
        for phase in range(3):
            injections.setdefault(item["from_bus"], [0j, 0j, 0j])[phase] += item["i_from"][phase]
            injections.setdefault(item["to_bus"], [0j, 0j, 0j])[phase] += item["i_to"][phase]
    return {bus: values for bus, values in injections.items() if bus in voltages}


def bus_shunt_power_unbalance(
    voltage_rows: Any,
    current_rows: Any,
    *,
    power_floor_pu: float = DEFAULT_SHUNT_POWER_FLOOR_PU,
    sigma_pu: float | None = None,
    detection_sigmas: float = DIFFERENTIAL_DETECTION_SIGMAS,
) -> list[dict[str, Any]]:
    """Rank buses by the relative spread of their per-phase shunt power.

    Returns one record per bus with both voltage and current evidence, sorted
    by significance and then ``phase_power_spread_rel`` descending.  A
    balanced constant-power shunt has zero spread regardless of the voltage
    unbalance it sees; the source of a load unbalance has spread of order the
    load's phase imbalance.

    When ``sigma_pu`` is given, each bus also gets a noise floor for its
    absolute spread: the shunt injection sums ``n`` terminal currents, so its
    per-component noise is ``sigma * sqrt(n)`` and the power spread noise is
    about ``|V| * sigma * sqrt(n)``.  Buses whose spread does not clear
    ``detection_sigmas`` times that floor are marked insignificant and ranked
    below every significant bus, which keeps unloaded buses (whose relative
    spread is noise over the power floor) from outranking real sources.
    """
    voltages = voltage_rows_to_phasors(voltage_rows)
    currents = branch_current_rows_to_phasors(current_rows)
    if not voltages or not currents:
        return []
    floor = max(float(power_floor_pu), 1e-9)
    incident: dict[int, int] = {}
    for item in currents.values():
        incident[item["from_bus"]] = incident.get(item["from_bus"], 0) + 1
        incident[item["to_bus"]] = incident.get(item["to_bus"], 0) + 1
    records: list[dict[str, Any]] = []
    for bus, injection in sorted(bus_shunt_injections(voltages, currents).items()):
        v = voltages[bus]
        s_phase = [v[phase] * injection[phase].conjugate() for phase in range(3)]
        p = np.asarray([value.real for value in s_phase], dtype=float)
        q = np.asarray([value.imag for value in s_phase], dtype=float)
        mean_p = float(np.mean(p))
        spread = float(np.max(np.abs(p - mean_p)))
        _, v1, v2 = sequence_components(*v)
        _, i1, i2 = sequence_components(*injection)
        noise_floor = None
        significant = True
        if sigma_pu is not None and float(sigma_pu) > 0.0:
            noise_floor = (
                float(detection_sigmas)
                * float(abs(v1))
                * float(sigma_pu)
                * math.sqrt(float(max(incident.get(bus, 1), 1)))
            )
            significant = bool(spread >= noise_floor)
        records.append(
            {
                "bus": int(bus),
                "phase_power_spread_rel": spread / max(abs(mean_p), floor),
                "phase_power_spread_pu": spread,
                "spread_noise_floor_pu": noise_floor,
                "significant": significant,
                "incident_branches": int(incident.get(bus, 0)),
                "per_phase_p_pu": [float(value) for value in p],
                "per_phase_q_pu": [float(value) for value in q],
                "mean_p_pu": mean_p,
                "negative_sequence_current_pu": float(abs(i2)),
                "positive_sequence_current_pu": float(abs(i1)),
                "vuf": float(abs(v2) / abs(v1)) if abs(v1) > 1e-12 else None,
            }
        )
    records.sort(
        key=lambda item: (bool(item["significant"]), float(item["phase_power_spread_rel"])),
        reverse=True,
    )
    return records


def unbalance_source_localization(
    voltage_rows: Any,
    current_rows: Any,
    *,
    top_k: int = 5,
    power_floor_pu: float = DEFAULT_SHUNT_POWER_FLOOR_PU,
    sigma_pu: float | None = None,
    detection_sigmas: float = DIFFERENTIAL_DETECTION_SIGMAS,
) -> dict[str, Any] | None:
    ranking = bus_shunt_power_unbalance(
        voltage_rows,
        current_rows,
        power_floor_pu=power_floor_pu,
        sigma_pu=sigma_pu,
        detection_sigmas=detection_sigmas,
    )
    if not ranking:
        return None
    top = ranking[0]
    second = float(ranking[1]["phase_power_spread_rel"]) if len(ranking) > 1 else 0.0
    return {
        "method": SHUNT_POWER_SPREAD_METHOD,
        "bus_1based": int(top["bus"]),
        "phase_power_spread_rel": float(top["phase_power_spread_rel"]),
        "phase_power_spread_pu": float(top["phase_power_spread_pu"]),
        "spread_noise_floor_pu": top["spread_noise_floor_pu"],
        "significant": bool(top["significant"]),
        "significant_bus_count": sum(1 for item in ranking if item["significant"]),
        "separation_ratio": float(top["phase_power_spread_rel"]) / max(second, 1e-9),
        "top_unbalance_source_buses": [
            {
                "rank": rank,
                "bus": int(item["bus"]),
                "phase_power_spread_rel": float(item["phase_power_spread_rel"]),
                "phase_power_spread_pu": float(item["phase_power_spread_pu"]),
                "significant": bool(item["significant"]),
                "per_phase_p_pu": item["per_phase_p_pu"],
                "negative_sequence_current_pu": item["negative_sequence_current_pu"],
            }
            for rank, item in enumerate(ranking[: max(1, int(top_k))], start=1)
        ],
    }


# ------------------------------------------------------- HIF localization


def _line_parameter(
    row0: int, line_parameters: Mapping[int, Mapping[str, Any]] | None
) -> Mapping[str, Any] | None:
    params = line_parameters if line_parameters is not None else case14_line_parameters()
    return params.get(int(row0))


def line_differential_phasors(
    voltage_rows: Any,
    current_rows: Any,
    *,
    line_parameters: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-line complex per-phase differential current, charging removed.

    ``differential[phase] = I_from + I_to - j(B/2)(V_from + V_to)``.  Keyed by
    ``branch_row0``; transformers are skipped.
    """
    voltages = voltage_rows_to_phasors(voltage_rows)
    currents = branch_current_rows_to_phasors(current_rows)
    out: dict[int, dict[str, Any]] = {}
    for row0, item in sorted(currents.items()):
        params = _line_parameter(row0, line_parameters)
        if params is None or not params.get("is_line"):
            continue
        v_from = voltages.get(item["from_bus"])
        v_to = voltages.get(item["to_bus"])
        charging_known = v_from is not None and v_to is not None
        b_total = float(params.get("b") or 0.0)
        differential: list[complex] = []
        for phase in range(3):
            total = item["i_from"][phase] + item["i_to"][phase]
            if charging_known:
                total -= 1j * (b_total / 2.0) * (v_from[phase] + v_to[phase])  # type: ignore[index]
            differential.append(complex(total))
        out[int(row0)] = {
            "branch_row0": int(row0),
            "line_index1": int(row0) + 1,
            "dss_element": params.get("dss_element") or item["branch"],
            "from_bus": int(item["from_bus"]),
            "to_bus": int(item["to_bus"]),
            "differential": differential,
            "charging_modeled": bool(charging_known),
        }
    return out


def _rank_differentials(
    phasors: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row0, item in sorted(phasors.items()):
        differential = [float(abs(value)) for value in item["differential"]]
        phase_index = int(np.argmax(differential))
        records.append(
            {
                "branch_row0": int(row0),
                "line_index1": int(item["line_index1"]),
                "dss_element": item["dss_element"],
                "from_bus": int(item["from_bus"]),
                "to_bus": int(item["to_bus"]),
                "differential_pu": differential,
                "score": float(differential[phase_index]),
                "phase": PHASES[phase_index],
                "charging_modeled": bool(item.get("charging_modeled", True)),
            }
        )
    records.sort(key=lambda record: float(record["score"]), reverse=True)
    return records


def line_differential_currents(
    voltage_rows: Any,
    current_rows: Any,
    *,
    line_parameters: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per-line, per-phase differential current with modeled charging removed.

    ``differential_pu[phase] = |I_from + I_to - j(B/2)(V_from + V_to)|``.
    A healthy line leaves only measurement noise; a mid-span shunt fault leaves
    the fault current on the faulted phase.  Transformers are skipped.
    """
    return _rank_differentials(
        line_differential_phasors(voltage_rows, current_rows, line_parameters=line_parameters)
    )


def _fault_point_mismatch(
    alpha: float,
    *,
    v_from: complex,
    v_to: complex,
    i_from: complex,
    i_to: complex,
    z: complex,
    b_total: float,
) -> tuple[float, complex, complex]:
    """Return (|Vx_from - Vx_to|, Vx, I_fault) for a fault fraction alpha."""
    y_from = 1j * b_total * alpha / 2.0
    y_to = 1j * b_total * (1.0 - alpha) / 2.0
    i_series_from = i_from - y_from * v_from
    i_series_to = i_to - y_to * v_to
    vx_from = v_from - alpha * z * i_series_from
    vx_to = v_to - (1.0 - alpha) * z * i_series_to
    vx = 0.5 * (vx_from + vx_to)
    i_fault = (i_series_from - y_from * vx) + (i_series_to - y_to * vx)
    return float(abs(vx_from - vx_to)), vx, i_fault


def two_terminal_hif_estimate(
    voltage_rows: Any,
    current_rows: Any,
    *,
    branch_row0: int,
    phase: str,
    line_parameters: Mapping[int, Mapping[str, Any]] | None = None,
    alpha_grid_size: int = 401,
    base_mva: float = S_BASE_MVA,
    kv_ll: float = KV_LL_BASE,
) -> dict[str, Any] | None:
    """Closed-form position and resistance of a mid-span shunt fault.

    The fault position is the fraction ``alpha`` from the from-bus at which
    the fault-point voltage computed from the from-end equals the one computed
    from the to-end.  Each segment keeps its share of the line charging.
    """
    voltages = voltage_rows_to_phasors(voltage_rows)
    currents = branch_current_rows_to_phasors(current_rows)
    item = currents.get(int(branch_row0))
    params = _line_parameter(int(branch_row0), line_parameters)
    if item is None or params is None or not params.get("is_line"):
        return None
    v_from = voltages.get(item["from_bus"])
    v_to = voltages.get(item["to_bus"])
    if v_from is None or v_to is None:
        return None
    phase_text = str(phase).strip().upper()
    if phase_text not in PHASES:
        return None
    phase_index = PHASES.index(phase_text)
    z = complex(float(params["r"]), float(params["x"]))
    if abs(z) <= 0.0:
        return None
    b_total = float(params.get("b") or 0.0)
    kwargs = {
        "v_from": v_from[phase_index],
        "v_to": v_to[phase_index],
        "i_from": item["i_from"][phase_index],
        "i_to": item["i_to"][phase_index],
        "z": z,
        "b_total": b_total,
    }
    grid = np.linspace(0.005, 0.995, max(int(alpha_grid_size), 3))
    errors = [_fault_point_mismatch(float(alpha), **kwargs)[0] for alpha in grid]
    best_index = int(np.argmin(errors))
    low = float(grid[max(best_index - 1, 0)])
    high = float(grid[min(best_index + 1, len(grid) - 1)])
    # Golden-section refinement inside the bracketing grid cells.
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc = _fault_point_mismatch(c, **kwargs)[0]
    fd = _fault_point_mismatch(d, **kwargs)[0]
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc = _fault_point_mismatch(c, **kwargs)[0]
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd = _fault_point_mismatch(d, **kwargs)[0]
    alpha = float(0.5 * (a + b))
    mismatch, vx, i_fault = _fault_point_mismatch(alpha, **kwargs)
    if abs(i_fault) <= 1e-15:
        return None
    impedance = vx / i_fault
    r_pu = float(impedance.real)
    z_base = zbase_ohm(base_mva=base_mva, kv_ll=kv_ll)
    # Self-consistency: the fault-point voltage mismatch relative to the
    # voltage drop the differential current produces along the line.  A real
    # shunt fault leaves noise-level mismatch; a bad current sensor cannot be
    # explained by any (alpha, R) and leaves a mismatch of order the drop.
    consistency_ratio = float(mismatch) / max(abs(z) * abs(i_fault), 1e-15)
    # A fault fitted at (or beyond) a terminal is indistinguishable, from this
    # line's phasors alone, from a gross error on that terminal's current
    # sensor: both leave a differential the other end cannot contradict.
    endpoint_ambiguous = bool(alpha <= ENDPOINT_AMBIGUITY_ALPHA or alpha >= 1.0 - ENDPOINT_AMBIGUITY_ALPHA)
    return {
        "method": TERMINAL_CURRENT_METHOD,
        "branch_row0": int(branch_row0),
        "line_index1": int(branch_row0) + 1,
        "dss_element": params.get("dss_element") or item["branch"],
        "from_bus": int(item["from_bus"]),
        "to_bus": int(item["to_bus"]),
        "phase": phase_text,
        "alpha_from_from_bus": alpha,
        "distance_percent_from_from_bus": 100.0 * alpha,
        "r_hif_pu": r_pu,
        "x_hif_pu": float(impedance.imag),
        "r_hif_ohm": r_pu * z_base,
        "i_hif_pu": float(abs(i_fault)),
        "fault_voltage_pu": float(abs(vx)),
        "fit_mismatch_pu": float(mismatch),
        "line_impedance_pu": float(abs(z)),
        "consistency_ratio": consistency_ratio,
        "endpoint_ambiguous": endpoint_ambiguous,
    }


def terminal_current_hif_localization(
    voltage_rows: Any,
    current_rows: Any,
    *,
    top_k: int = 5,
    sigma_pu: float = DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    detection_sigmas: float = DIFFERENTIAL_DETECTION_SIGMAS,
    line_parameters: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Line-level HIF localization payload in the three-phase NLM shape.

    ``top_hif_groups`` ranks lines by their largest per-phase differential
    current.  ``suspected_phase`` is the phase carrying that differential on
    the top line, and ``terminal_current_estimate`` is the closed-form position
    and resistance on that line and phase.
    """
    ranking = line_differential_currents(
        voltage_rows, current_rows, line_parameters=line_parameters
    )
    if not ranking:
        return None
    top = ranking[0]
    second_score = float(ranking[1]["score"]) if len(ranking) > 1 else 0.0
    # Noise floor: healthy lines carry roughly sqrt(2)*sigma per phasor sum.
    noise_floor = math.sqrt(2.0) * max(float(sigma_pu), 1e-12)
    detection_floor = float(detection_sigmas) * noise_floor
    groups = [
        {
            "rank": rank,
            "branch_row0": int(item["branch_row0"]),
            "line_index1": int(item["line_index1"]),
            "dss_element": item["dss_element"],
            "from_bus": int(item["from_bus"]),
            "to_bus": int(item["to_bus"]),
            "score": float(item["score"]),
            "phase": item["phase"],
            "differential_pu": [float(value) for value in item["differential_pu"]],
        }
        for rank, item in enumerate(ranking[: max(1, int(top_k))], start=1)
    ]
    estimate = two_terminal_hif_estimate(
        voltage_rows,
        current_rows,
        branch_row0=int(top["branch_row0"]),
        phase=str(top["phase"]),
        line_parameters=line_parameters,
    )
    phase_scores = {
        PHASES[index]: float(value) for index, value in enumerate(top["differential_pu"])
    }
    return {
        "success": True,
        "converged": True,
        "method": TERMINAL_CURRENT_METHOD,
        "top_hif_groups": groups,
        "suspected_phase": str(top["phase"]),
        "phase_scores": phase_scores,
        "max_line_differential_pu": float(top["score"]),
        "second_line_differential_pu": second_score,
        "separation_ratio": float(top["score"]) / max(second_score, noise_floor),
        "differential_detection_floor_pu": detection_floor,
        "differential_detected": bool(float(top["score"]) >= detection_floor),
        "branch_current_sigma_pu": float(sigma_pu),
        "terminal_current_estimate": estimate,
    }


def terminal_current_hif_localization_multiscan(
    scans: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
    sigma_pu: float = DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    detection_sigmas: float = DIFFERENTIAL_DETECTION_SIGMAS,
    line_parameters: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Line-level HIF localization from a persistent multi-scan window.

    A persistent fault draws a coherent differential phasor in every scan,
    while sensor noise is independent, so the complex differential is averaged
    across scans before ranking: the fault signal survives and the noise floor
    falls by ``sqrt(N)``.  Position and resistance are the per-scan closed-form
    medians on the winning line and phase.  Scans without both channels are
    skipped; a single usable scan degrades to the snapshot method.
    """
    usable: list[tuple[Mapping[str, Any], dict[int, dict[str, Any]]]] = []
    for scan in scans:
        if not isinstance(scan, Mapping):
            continue
        voltages = scan.get("three_phase_voltages")
        currents = scan.get(BRANCH_CURRENT_CHANNEL)
        if not voltages or not currents:
            continue
        phasors = line_differential_phasors(voltages, currents, line_parameters=line_parameters)
        if phasors:
            usable.append((scan, phasors))
    if not usable:
        return None
    if len(usable) == 1:
        scan, _ = usable[0]
        payload = terminal_current_hif_localization(
            scan.get("three_phase_voltages"),
            scan.get(BRANCH_CURRENT_CHANNEL),
            top_k=top_k,
            sigma_pu=sigma_pu,
            detection_sigmas=detection_sigmas,
            line_parameters=line_parameters,
        )
        if payload is not None:
            payload["scan_count"] = 1
        return payload

    count = len(usable)
    accumulated: dict[int, dict[str, Any]] = {}
    for _scan, phasors in usable:
        for row0, item in phasors.items():
            slot = accumulated.setdefault(
                row0,
                {**{key: value for key, value in item.items() if key != "differential"}, "differential": [0j, 0j, 0j], "count": 0},
            )
            for phase in range(3):
                slot["differential"][phase] += item["differential"][phase]
            slot["count"] += 1
    for slot in accumulated.values():
        slot["differential"] = [value / max(int(slot["count"]), 1) for value in slot["differential"]]
    ranking = _rank_differentials(accumulated)
    top = ranking[0]
    second_score = float(ranking[1]["score"]) if len(ranking) > 1 else 0.0
    noise_floor = math.sqrt(2.0) * max(float(sigma_pu), 1e-12) / math.sqrt(float(count))
    detection_floor = float(detection_sigmas) * noise_floor

    per_scan_estimates: list[dict[str, Any]] = []
    per_scan_top: list[int] = []
    per_scan_phase: list[str] = []
    for scan, phasors in usable:
        scan_ranking = _rank_differentials(phasors)
        per_scan_top.append(int(scan_ranking[0]["branch_row0"]))
        per_scan_phase.append(str(scan_ranking[0]["phase"]))
        estimate = two_terminal_hif_estimate(
            scan.get("three_phase_voltages"),
            scan.get(BRANCH_CURRENT_CHANNEL),
            branch_row0=int(top["branch_row0"]),
            phase=str(top["phase"]),
            line_parameters=line_parameters,
        )
        if estimate is not None:
            estimate = dict(estimate)
            estimate["scan_index"] = scan.get("scan_index")
            per_scan_estimates.append(estimate)
    aggregated: dict[str, Any] | None = None
    if per_scan_estimates:
        alphas = [float(item["alpha_from_from_bus"]) for item in per_scan_estimates]
        log_rs = [
            math.log(float(item["r_hif_pu"]))
            for item in per_scan_estimates
            if math.isfinite(float(item["r_hif_pu"])) and float(item["r_hif_pu"]) > 0.0
        ]
        reference = per_scan_estimates[0]
        aggregated = {
            "method": TERMINAL_CURRENT_METHOD,
            "branch_row0": int(top["branch_row0"]),
            "line_index1": int(top["line_index1"]),
            "dss_element": top["dss_element"],
            "from_bus": int(top["from_bus"]),
            "to_bus": int(top["to_bus"]),
            "phase": str(top["phase"]),
            "alpha_from_from_bus": float(np.median(alphas)),
            "distance_percent_from_from_bus": 100.0 * float(np.median(alphas)),
            "alpha_interval": [min(alphas), max(alphas)],
            "r_hif_pu": float(math.exp(np.median(log_rs))) if log_rs else None,
            "r_hif_pu_interval": [math.exp(min(log_rs)), math.exp(max(log_rs))] if log_rs else None,
            "r_hif_ohm": (
                float(math.exp(np.median(log_rs))) * zbase_ohm() if log_rs else None
            ),
            "i_hif_pu": float(top["score"]),
            "x_hif_pu": float(np.median([item["x_hif_pu"] for item in per_scan_estimates])),
            "fit_mismatch_pu": float(np.median([item["fit_mismatch_pu"] for item in per_scan_estimates])),
            "consistency_ratio": float(
                np.median([item["consistency_ratio"] for item in per_scan_estimates])
            ),
            "endpoint_ambiguous": bool(
                float(np.median(alphas)) <= ENDPOINT_AMBIGUITY_ALPHA
                or float(np.median(alphas)) >= 1.0 - ENDPOINT_AMBIGUITY_ALPHA
            ),
            "differential_detected": bool(float(top["score"]) >= detection_floor),
            "scan_count": len(per_scan_estimates),
            "per_scan": [
                {
                    "scan_index": item.get("scan_index"),
                    "alpha_from_from_bus": item["alpha_from_from_bus"],
                    "r_hif_pu": item["r_hif_pu"],
                    "i_hif_pu": item["i_hif_pu"],
                }
                for item in per_scan_estimates
            ],
        }
        del reference
    groups = [
        {
            "rank": rank,
            "branch_row0": int(item["branch_row0"]),
            "line_index1": int(item["line_index1"]),
            "dss_element": item["dss_element"],
            "from_bus": int(item["from_bus"]),
            "to_bus": int(item["to_bus"]),
            "score": float(item["score"]),
            "phase": item["phase"],
            "differential_pu": [float(value) for value in item["differential_pu"]],
        }
        for rank, item in enumerate(ranking[: max(1, int(top_k))], start=1)
    ]
    line_votes: dict[str, int] = {}
    for row0 in per_scan_top:
        line_votes[str(row0)] = line_votes.get(str(row0), 0) + 1
    phase_votes: dict[str, int] = {}
    for phase in per_scan_phase:
        phase_votes[phase] = phase_votes.get(phase, 0) + 1
    return {
        "success": True,
        "converged": True,
        "method": TERMINAL_CURRENT_METHOD,
        "aggregation": "coherent_mean_differential_across_scans",
        "scan_count": count,
        "top_hif_groups": groups,
        "suspected_phase": str(top["phase"]),
        "phase_scores": {
            PHASES[index]: float(value) for index, value in enumerate(top["differential_pu"])
        },
        "per_scan_line_votes": line_votes,
        "per_scan_phase_votes": phase_votes,
        "max_line_differential_pu": float(top["score"]),
        "second_line_differential_pu": second_score,
        "separation_ratio": float(top["score"]) / max(second_score, noise_floor),
        "differential_detection_floor_pu": detection_floor,
        "differential_detected": bool(float(top["score"]) >= detection_floor),
        "branch_current_sigma_pu": float(sigma_pu),
        "terminal_current_estimate": aggregated,
    }


def line_differential_null_test(
    voltage_rows: Any,
    current_rows: Any,
    *,
    sigma_pu: float = DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    detection_sigmas: float = DIFFERENTIAL_DETECTION_SIGMAS,
    line_parameters: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Explicit non-HIF null: no line carries a differential above the floor."""
    ranking = line_differential_currents(
        voltage_rows, current_rows, line_parameters=line_parameters
    )
    if not ranking:
        return None
    floor = float(detection_sigmas) * math.sqrt(2.0) * max(float(sigma_pu), 1e-12)
    top = ranking[0]
    return {
        "max_line_differential_pu": float(top["score"]),
        "max_differential_branch_row0": int(top["branch_row0"]),
        "max_differential_phase": str(top["phase"]),
        "differential_detection_floor_pu": floor,
        "hif_like_differential_present": bool(float(top["score"]) >= floor),
    }


__all__ = [
    "BRANCH_CURRENT_CHANNEL",
    "BRANCH_CURRENT_SIGMA_KEY",
    "DEFAULT_BRANCH_CURRENT_SIGMA_PU",
    "PHASES",
    "SHUNT_POWER_SPREAD_METHOD",
    "TERMINAL_CURRENT_METHOD",
    "add_branch_current_noise",
    "balanced_branch_current_control",
    "branch_current_rows_to_phasors",
    "branch_current_rows_valid",
    "bus_shunt_power_unbalance",
    "case14_line_parameters",
    "line_differential_currents",
    "line_differential_null_test",
    "line_differential_phasors",
    "sequence_components",
    "terminal_current_hif_localization",
    "terminal_current_hif_localization_multiscan",
    "two_terminal_hif_estimate",
    "unbalance_source_localization",
    "voltage_rows_to_phasors",
]
