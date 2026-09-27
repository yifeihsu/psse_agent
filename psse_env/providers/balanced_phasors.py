"""PMU phasors from a root's true balanced state (suspicion-gated contract).

Under ``suspicion_gated_diagnostics`` phase-resolved phasors are requested
only on a balanced HIF suspicion, and the screen can fire on a root of any
family, so every alarmed root must carry phasors: a request that answered
"unavailable" on some roots would name the family.  OpenDSS families carry the
phasors their simulation produced; every other root gets the phasors of its
own true state, in the exact row format and per-unit conventions of the
OpenDSS exporter:

* the true state is the balanced WLS fit of the root's noise-free measurement
  vector on its true network (the parameter-error or true-topology case where
  the operator's model is wrong), which reproduces a power-flow solution;
* each phase is the positive-sequence value rotated by 0, -120 and +120
  degrees (the network is balanced; OpenDSS lines here have no mutual
  coupling and its transformers are wye-wye, so no phase shift is missing);
* bus voltages are line-to-neutral per unit on the bus base; terminal
  currents flow into the branch and are per unit on each terminal's base
  (``Yf V`` and ``Yt V``, the tap on the from side);
* one PMU sigma per rectangular component for every family.
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Mapping, Sequence

import numpy as np

from psse_env.providers.hif_screen import _fit, _Operator, _Problem, _ybus_dense

BALANCED_PHASOR_SOURCE = "true_balanced_state_positive_sequence_rotation_v1"
_ROTATIONS = (1.0 + 0.0j, complex(math.cos(-2 * math.pi / 3), math.sin(-2 * math.pi / 3)),
              complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)))


def _bus_kv(case: Mapping[str, Any]) -> dict[int, float]:
    """Line-to-line kV per external bus: the IEEE 14 study profile, else the case's BASE_KV."""
    bus = np.asarray(case["bus"], dtype=float)
    numbers = [int(value) for value in bus[:, 0]]
    if sorted(numbers) == list(range(1, 15)):
        from three_phase_model.voltage_bases import IEEE14_NOMINAL_KV

        return {number: float(IEEE14_NOMINAL_KV[number]) for number in numbers}
    return {int(row[0]): float(row[9]) if bus.shape[1] > 9 and float(row[9]) > 0 else 1.0 for row in bus}


def _branch_names(case: Mapping[str, Any]) -> list[str]:
    branch = np.asarray(case["branch"], dtype=float)
    bus_numbers = sorted(int(value) for value in np.asarray(case["bus"], dtype=float)[:, 0])
    if bus_numbers == list(range(1, 15)) and branch.shape[0] == 20:
        from IEEE_14_OpenDSS.constants import BRANCH_ORDER

        return list(BRANCH_ORDER)
    names = []
    for row in branch:
        kind = "Transformer" if float(row[8]) not in (0.0, 1.0) or float(row[9]) != 0.0 else "Line"
        names.append(f"{kind}.{int(row[0])}-{int(row[1])}")
    return names


def true_balanced_state(case: Mapping[str, Any], clean_measurements: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Bus magnitudes and angles (rad) of the noise-free vector on the true network."""
    from tools.lagrangian_port import _copy_result_to_internal

    internal = _copy_result_to_internal(case)
    operator = _Operator(internal["baseMVA"], internal["bus"], internal["branch"])
    z = np.asarray(clean_measurements, dtype=float)
    if z.size != operator.nz:
        raise ValueError(f"clean measurements carry {z.size} channels; the true case needs {operator.nz}")
    sigma = np.r_[np.full(operator.nb, 1e-3), np.full(operator.nz - operator.nb, 1e-2)]
    problem = _Problem(z, sigma, (), operator.nb, operator.ref)
    va0 = np.deg2rad(operator.bus[:, 8] - operator.bus[operator.ref, 8])
    fit = _fit(operator.base_model(), problem, [], va0, operator.bus[:, 7], max_iterations=100, tolerance=1e-11)
    if not fit.get("success"):
        raise ValueError("the true balanced state did not converge from the clean measurements")
    return np.asarray(fit["vm"], dtype=float), np.asarray(fit["va"], dtype=float)


def balanced_phasor_rows(case: Mapping[str, Any], vm: Sequence[float], va_rad: Sequence[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Clean three-phase voltage and branch-current rows of a balanced state."""
    from tools.lagrangian_port import _copy_result_to_internal

    internal = _copy_result_to_internal(case)
    bus_numbers = [int(value) for value in np.asarray(case["bus"], dtype=float)[:, 0]]
    external_branch = np.asarray(case["branch"], dtype=float)
    kv = _bus_kv(case)
    names = _branch_names(case)
    voltage = np.asarray(vm, dtype=float) * np.exp(1j * np.asarray(va_rad, dtype=float))
    _ybus, yf, yt, _f, _t = _ybus_dense(internal["baseMVA"], internal["bus"], internal["branch"])
    i_from = yf @ voltage
    i_to = yt @ voltage

    def phases(value: complex) -> tuple[list[float], list[float]]:
        rotated = [value * rotation for rotation in _ROTATIONS]
        return [float(abs(item)) for item in rotated], [float(math.degrees(np.angle(item))) for item in rotated]

    voltage_rows = []
    for position, number in enumerate(bus_numbers):
        magnitudes, angles = phases(voltage[position])
        voltage_rows.append({
            "bus": f"b{number}",
            "kvbase_ln": kv[number] / math.sqrt(3.0),
            "vln_pu": magnitudes,
            "ang_deg": angles,
        })
    current_rows = []
    for row0 in range(external_branch.shape[0]):
        from_bus, to_bus = int(external_branch[row0, 0]), int(external_branch[row0, 1])
        from_magnitudes, from_angles = phases(i_from[row0])
        to_magnitudes, to_angles = phases(i_to[row0])
        current_rows.append({
            "branch": names[row0],
            "branch_row0": row0,
            "from_bus": f"b{from_bus}",
            "to_bus": f"b{to_bus}",
            "ibase_from_a": 100e6 / (math.sqrt(3.0) * kv[from_bus] * 1e3),
            "ibase_to_a": 100e6 / (math.sqrt(3.0) * kv[to_bus] * 1e3),
            "i_from_pu": from_magnitudes,
            "ang_from_deg": from_angles,
            "i_to_pu": to_magnitudes,
            "ang_to_deg": to_angles,
        })
    return voltage_rows, current_rows


def phasor_rng(seed: int, scenario_id: str) -> np.random.Generator:
    """A per-root stream, so attaching phasors never shifts any other draw."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(str(scenario_id).encode("utf-8"))])


def noisy_phasors(
    voltage_rows: Sequence[Mapping[str, Any]], current_rows: Sequence[Mapping[str, Any]],
    rng: np.random.Generator, sigma_pu: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One independent sensor draw per rectangular component on every phasor."""
    from three_phase_nlm.branch_current_analysis import add_branch_current_noise
    from three_phase_nlm.measurement_noise import add_voltage_phasor_noise

    return (
        add_voltage_phasor_noise(voltage_rows, rng, float(sigma_pu)),
        add_branch_current_noise(current_rows, rng, float(sigma_pu)),
    )


__all__ = [
    "BALANCED_PHASOR_SOURCE", "balanced_phasor_rows", "noisy_phasors", "phasor_rng", "true_balanced_state",
]
