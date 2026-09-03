"""Synthetic, KCL-consistent per-phase telemetry for tests and QA.

These builders produce ``three_phase_voltages`` and
``three_phase_branch_currents`` rows from the IEEE-14 line parameters without
OpenDSS.  They are deliberately simple radial constructions: enough physics
for the analysis code to have a known answer, not a power-flow solution.
Never use them as training data.
"""

from __future__ import annotations

import cmath
import math
from typing import Any, Mapping, Sequence

from IEEE_14_OpenDSS.constants import BRANCH_ORDER

from .branch_current_analysis import PHASES, case14_line_parameters

ROTATION = cmath.exp(2j * math.pi / 3.0)


def balanced_phasors(magnitude: float, angle_deg: float) -> list[complex]:
    reference = cmath.rect(float(magnitude), math.radians(float(angle_deg)))
    return [reference, reference * ROTATION**2, reference * ROTATION]


def _magnitude_angle(phasors: Sequence[complex]) -> tuple[list[float], list[float]]:
    magnitudes = [float(abs(value)) for value in phasors]
    angles = [float(math.degrees(cmath.phase(value))) for value in phasors]
    return magnitudes, angles


def voltage_rows(voltages: Mapping[int, Sequence[complex]]) -> list[dict[str, Any]]:
    rows = []
    for bus, phasors in sorted(voltages.items()):
        magnitudes, angles = _magnitude_angle(phasors)
        rows.append(
            {"bus": f"b{int(bus)}", "kvbase_ln": 0.577, "vln_pu": magnitudes, "ang_deg": angles}
        )
    return rows


def branch_current_rows(
    currents: Mapping[int, tuple[Sequence[complex], Sequence[complex]]],
) -> list[dict[str, Any]]:
    """Rows in the exporter schema from ``row0 -> (i_from, i_to)`` phasors."""
    params = case14_line_parameters()
    rows = []
    for row0, (i_from, i_to) in sorted(currents.items()):
        from_mag, from_ang = _magnitude_angle(i_from)
        to_mag, to_ang = _magnitude_angle(i_to)
        rows.append(
            {
                "branch": BRANCH_ORDER[int(row0)],
                "branch_row0": int(row0),
                "from_bus": f"b{params[int(row0)]['from_bus']}",
                "to_bus": f"b{params[int(row0)]['to_bus']}",
                "ibase_from_a": 57735.0,
                "ibase_to_a": 57735.0,
                "i_from_pu": from_mag,
                "ang_from_deg": from_ang,
                "i_to_pu": to_mag,
                "ang_to_deg": to_ang,
            }
        )
    return rows


def constant_power_load_current(
    voltages: Sequence[complex],
    total_pu: float,
    fractions: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
) -> list[complex]:
    """Per-phase current drawn by a constant-power load split across phases.

    ``I_ph = conj(S_ph / V_ph)`` with ``S_ph = 3 * total * fraction`` so the
    three fractions describe how the three-phase total divides among phases.
    """
    return [
        ((3.0 * float(total_pu) * float(fraction)) / voltages[index]).conjugate()
        for index, fraction in enumerate(fractions)
    ]


def synthetic_unbalance_rows(
    *,
    source_bus: int,
    split: Sequence[float],
    load_2_pu: float = 0.2,
    load_3_pu: float = 0.3,
    unbalanced_voltage_bus: int | None = None,
    voltage_phase_b_scale: float = 0.92,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Radial chain 1-2-3 (branch rows 0 and 2) with loads at buses 2 and 3.

    The load at ``source_bus`` (2 or 3) is split unevenly across phases; the
    other load stays balanced.  Each line carries its pi-model charging at
    both terminals, so the terminal currents satisfy KCL at every bus and the
    healthy-line differential is exactly the modeled charging.  Bus 1 acts as
    the source.  ``unbalanced_voltage_bus`` scales phase B of that bus's
    voltage so a voltage-unbalance gate (VUF) also fires there.
    """
    if source_bus not in (2, 3):
        raise ValueError("source_bus must be 2 or 3")
    params = case14_line_parameters()
    voltages = {
        1: balanced_phasors(1.06, 0.0),
        2: balanced_phasors(1.03, -4.0),
        3: balanced_phasors(1.01, -9.0),
    }
    if unbalanced_voltage_bus is not None:
        phasors = voltages[int(unbalanced_voltage_bus)]
        phasors[1] = phasors[1] * float(voltage_phase_b_scale)
    y12 = 1j * float(params[0]["b"]) / 2.0
    y23 = 1j * float(params[2]["b"]) / 2.0
    balanced = (1 / 3, 1 / 3, 1 / 3)
    i_load_2 = constant_power_load_current(
        voltages[2], load_2_pu, split if source_bus == 2 else balanced
    )
    i_load_3 = constant_power_load_current(
        voltages[3], load_3_pu, split if source_bus == 3 else balanced
    )
    # Line 2-3: the series current toward bus 3 feeds the load and the 3-end
    # charging; KCL at bus 3 then leaves exactly the load as shunt injection.
    i_series_23 = [i_load_3[k] + y23 * voltages[3][k] for k in range(3)]
    i_into_23_at_3 = [-i_series_23[k] + y23 * voltages[3][k] for k in range(3)]
    i_into_23_at_2 = [i_series_23[k] + y23 * voltages[2][k] for k in range(3)]
    # Bus 2 KCL: the bus-2 load is the only shunt element.
    i_into_12_at_2 = [-(i_load_2[k] + i_into_23_at_2[k]) for k in range(3)]
    i_series_12 = [y12 * voltages[2][k] - i_into_12_at_2[k] for k in range(3)]
    i_into_12_at_1 = [i_series_12[k] + y12 * voltages[1][k] for k in range(3)]
    currents = {0: (i_into_12_at_1, i_into_12_at_2), 2: (i_into_23_at_2, i_into_23_at_3)}
    return voltage_rows(voltages), branch_current_rows(currents)


def propagate_healthy_line(
    row0: int,
    *,
    v_from: Sequence[complex],
    i_series_from: Sequence[complex],
) -> tuple[list[complex], list[complex], list[complex]]:
    """Single pi-section line: returns ``(v_to, i_from_into, i_to_into)``."""
    params = case14_line_parameters()[int(row0)]
    z = complex(params["r"], params["x"])
    y = 1j * float(params["b"]) / 2.0
    v_to: list[complex] = []
    i_from_terminal: list[complex] = []
    i_to_terminal: list[complex] = []
    for index in range(3):
        vf = v_from[index]
        i_series = i_series_from[index]
        i_from_terminal.append(i_series + y * vf)
        vt = vf - z * i_series
        v_to.append(vt)
        i_to_terminal.append(-i_series + y * vt)
    return v_to, i_from_terminal, i_to_terminal


def propagate_faulted_line(
    row0: int,
    *,
    alpha: float,
    phase: str,
    r_hif_pu: float,
    v_from: Sequence[complex],
    i_series_from: Sequence[complex],
) -> tuple[list[complex], list[complex], list[complex], complex]:
    """Propagate from-end phasors through a mid-span shunt fault.

    Each segment is its own pi-section (series R+jX, half the segment's
    charging at each end) and the fault is a shunt resistor at the split.
    Returns ``(v_to, i_from_into_branch, i_to_into_branch, i_fault)``.
    """
    params = case14_line_parameters()[int(row0)]
    z = complex(params["r"], params["x"])
    b_total = float(params["b"])
    y_a = 1j * b_total * float(alpha) / 2.0
    y_b = 1j * b_total * (1.0 - float(alpha)) / 2.0
    phase_index = PHASES.index(str(phase).upper())
    v_to: list[complex] = []
    i_from_terminal: list[complex] = []
    i_to_terminal: list[complex] = []
    i_fault_out = 0j
    for index in range(3):
        vf = v_from[index]
        i_series = i_series_from[index]
        i_from_terminal.append(i_series + y_a * vf)
        vx = vf - float(alpha) * z * i_series
        i_arriving = i_series - y_a * vx
        i_fault = vx / float(r_hif_pu) if index == phase_index else 0j
        if index == phase_index:
            i_fault_out = i_fault
        i_series_b = i_arriving - i_fault - y_b * vx
        vt = vx - (1.0 - float(alpha)) * z * i_series_b
        i_leaving = i_series_b - y_b * vt
        v_to.append(vt)
        i_to_terminal.append(-i_leaving)
    return v_to, i_from_terminal, i_to_terminal, i_fault_out


def disjoint_healthy_line(row0: int) -> int:
    """A Line.* branch row that shares no bus with ``row0``."""
    params = case14_line_parameters()
    target = params[int(row0)]
    for candidate, item in sorted(params.items()):
        if candidate == int(row0) or not item["is_line"]:
            continue
        if {item["from_bus"], item["to_bus"]} & {target["from_bus"], target["to_bus"]}:
            continue
        return int(candidate)
    raise RuntimeError(f"No disjoint healthy line for branch row {row0}")


def synthetic_line_fault_rows(
    *,
    row0: int,
    alpha: float,
    phase: str,
    r_hif_pu: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], complex]:
    """Faulted line plus one healthy line on disjoint buses.

    Returns ``(voltage_rows, current_rows, i_fault)``.
    """
    params = case14_line_parameters()
    faulted = params[int(row0)]
    v_from = balanced_phasors(1.05, 0.0)
    v_to, i_from, i_to, i_fault = propagate_faulted_line(
        int(row0),
        alpha=alpha,
        phase=phase,
        r_hif_pu=r_hif_pu,
        v_from=v_from,
        i_series_from=balanced_phasors(0.6, -20.0),
    )
    healthy_row0 = disjoint_healthy_line(int(row0))
    healthy = params[healthy_row0]
    hv_from = balanced_phasors(1.04, -2.0)
    hv_to, hi_from, hi_to = propagate_healthy_line(
        healthy_row0,
        v_from=hv_from,
        i_series_from=balanced_phasors(0.4, -15.0),
    )
    voltages = {
        faulted["from_bus"]: v_from,
        faulted["to_bus"]: v_to,
        healthy["from_bus"]: hv_from,
        healthy["to_bus"]: hv_to,
    }
    currents = {int(row0): (i_from, i_to), healthy_row0: (hi_from, hi_to)}
    return voltage_rows(voltages), branch_current_rows(currents), i_fault


__all__ = [
    "balanced_phasors",
    "branch_current_rows",
    "constant_power_load_current",
    "disjoint_healthy_line",
    "propagate_faulted_line",
    "propagate_healthy_line",
    "synthetic_line_fault_rows",
    "synthetic_unbalance_rows",
    "voltage_rows",
]
