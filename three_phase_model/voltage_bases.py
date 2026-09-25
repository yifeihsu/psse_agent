"""Explicit user-selected voltage bases and local HIF unit conversion.

The canonical case may leave BASE_KV equal to zero. The named profile below is
an explicit model choice supplied for this experiment, not a claim that every
IEEE14/IEEE57 variant has these nominal voltages. The IEEE118 profile adopts the
BASE_KV the canonical case itself carries. No canonical source asset is edited.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np
from pypower.idx_brch import BR_STATUS, F_BUS, T_BUS, TAP, SHIFT
from pypower.idx_bus import BASE_KV, BUS_I


IEEE14_VOLTAGE_BASE_PROFILE_ID = "ieee14_nominal_69_13p8_18kv_v1"
IEEE14_NOMINAL_KV = {
    1: 69.0, 2: 69.0, 3: 69.0, 4: 69.0, 5: 69.0,
    6: 13.8, 7: 13.8, 8: 18.0, 9: 13.8, 10: 13.8, 11: 13.8,
    12: 13.8, 13: 13.8, 14: 13.8,
}
IEEE57_VOLTAGE_BASE_PROFILE_ID = "ieee57_reconstruction_138_69kv_v1"
IEEE57_RECONSTRUCTION_KV = {bus: 138.0 if bus <= 17 else 69.0 for bus in range(1, 58)}
IEEE118_VOLTAGE_BASE_PROFILE_ID = "ieee118_source_basekv_138_161_345kv_v1"
_IEEE118_345KV_BUSES = frozenset((8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81))
IEEE118_SOURCE_KV = {bus: 345.0 if bus in _IEEE118_345KV_BUSES else 161.0 if bus == 87 else 138.0
                     for bus in range(1, 119)}


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return value


def ieee14_voltage_base_profile() -> dict[str, Any]:
    """Return detached metadata for the selected line-to-line voltage map."""
    return {
        "profile_id": IEEE14_VOLTAGE_BASE_PROFILE_ID,
        "bus_base_kv_ll": dict(IEEE14_NOMINAL_KV),
        "voltage_units": "kV_line_to_line",
        "selection_basis": "explicit user-supplied experimental IEEE14 variant",
        "universal_ieee14_variant_claim": False,
        "canonical_source_modified": False,
    }


def ieee57_voltage_base_profile() -> dict[str, Any]:
    """Describe the selected physical reconstruction, not canonical ratings."""
    return {
        "profile_id": IEEE57_VOLTAGE_BASE_PROFILE_ID,
        "bus_base_kv_ll": dict(IEEE57_RECONSTRUCTION_KV),
        "voltage_units": "kV_line_to_line",
        "selection_basis": "explicit user-selected IEEE57 physical reconstruction: buses 1-17 at 138 kV, 18-57 at 69 kV",
        "reference_base_mva": 100.0,
        "canonical_nominal_voltage_claim": False,
        "universal_ieee57_variant_claim": False,
        "canonical_source_modified": False,
        "physicalization_scope": "supplies local physical units while retaining source per-unit network parameters",
    }


def ieee118_voltage_base_profile() -> dict[str, Any]:
    """Describe the case's own BASE_KV map: 345 kV core, one 161 kV bus, 138 kV rest.

    Two zero-tap branches join different source bases (86-87 at 138/161 kV and
    68-116 at 345/138 kV); the exporter realizes them as ideal-ratio
    transformers carrying the source charging at their endpoints.
    """
    return {
        "profile_id": IEEE118_VOLTAGE_BASE_PROFILE_ID,
        "bus_base_kv_ll": dict(IEEE118_SOURCE_KV),
        "voltage_units": "kV_line_to_line",
        "selection_basis": "canonical case BASE_KV column: buses 8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81 at 345 kV, "
                           "bus 87 at 161 kV, the other 106 buses at 138 kV",
        "reference_base_mva": 100.0,
        "canonical_nominal_voltage_claim": True,
        "universal_ieee118_variant_claim": False,
        "canonical_source_modified": False,
        "zero_tap_cross_voltage_branches": [[86, 87], [68, 116]],
        "physicalization_scope": "supplies local physical units while retaining source per-unit network parameters",
    }


def get_voltage_base_profile(name: str) -> dict[str, Any]:
    """Return detached metadata for an explicitly supported profile identity."""
    if name == IEEE14_VOLTAGE_BASE_PROFILE_ID:
        return ieee14_voltage_base_profile()
    if name == IEEE57_VOLTAGE_BASE_PROFILE_ID:
        return ieee57_voltage_base_profile()
    if name == IEEE118_VOLTAGE_BASE_PROFILE_ID:
        return ieee118_voltage_base_profile()
    raise ValueError(f"unsupported explicit voltage profile: {name!r}")


def _bus_matrix(case: Mapping[str, Any], *, voltage_map: Mapping[int, float] = IEEE14_NOMINAL_KV,
                system_name: str = "IEEE14") -> np.ndarray:
    if not isinstance(case, Mapping) or "bus" not in case:
        raise ValueError(f"An {system_name} case with a bus matrix is required")
    bus = np.asarray(case["bus"], dtype=float)
    if bus.ndim != 2 or bus.shape[1] <= BASE_KV or not np.isfinite(bus).all():
        raise ValueError("bus must be a finite matrix including BUS_I and BASE_KV")
    identifiers = bus[:, BUS_I]
    if not np.equal(identifiers, np.floor(identifiers)).all():
        raise ValueError(f"External {system_name} bus identifiers must be integers")
    ids = identifiers.astype(int).tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(voltage_map):
        raise ValueError(f"Selected voltage profile requires each external bus 1..{len(voltage_map)} exactly once")
    return bus


def apply_ieee14_voltage_bases(case: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a case and set BASE_KV by external BUS_I, without changing pu data."""
    bus = np.array(_bus_matrix(case), copy=True)
    configured = deepcopy(dict(case))
    previous = {int(row[BUS_I]): float(row[BASE_KV]) for row in bus}
    for row in bus:
        row[BASE_KV] = IEEE14_NOMINAL_KV[int(row[BUS_I])]
    configured["bus"] = bus
    configured["voltage_base_profile"] = {
        **ieee14_voltage_base_profile(), "original_bus_base_kv_ll": previous,
    }
    return configured


def apply_ieee57_voltage_bases(case: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an IEEE57 case and attach the explicit 138/69-kV reconstruction.

    Only BASE_KV is changed in the numeric case. Source per-unit branches,
    loads, dispatch and MVA base remain untouched, as does the canonical file.
    """
    bus = np.array(_bus_matrix(case, voltage_map=IEEE57_RECONSTRUCTION_KV, system_name="IEEE57"), copy=True)
    configured = deepcopy(dict(case))
    previous = {int(row[BUS_I]): float(row[BASE_KV]) for row in bus}
    for row in bus:
        row[BASE_KV] = IEEE57_RECONSTRUCTION_KV[int(row[BUS_I])]
    configured["bus"] = bus
    configured["voltage_base_profile"] = {
        **ieee57_voltage_base_profile(), "original_bus_base_kv_ll": previous,
    }
    return configured


def apply_ieee118_voltage_bases(case: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an IEEE118 case and attach its source-BASE_KV profile.

    A case whose nonzero BASE_KV disagrees with the profile is rejected rather
    than silently re-based; zero entries are filled from the profile.
    """
    bus = np.array(_bus_matrix(case, voltage_map=IEEE118_SOURCE_KV, system_name="IEEE118"), copy=True)
    previous = {int(row[BUS_I]): float(row[BASE_KV]) for row in bus}
    conflicts = sorted(number for number, kv in previous.items() if kv not in (0.0, IEEE118_SOURCE_KV[number]))
    if conflicts:
        raise ValueError(f"IEEE118 BASE_KV differs from the source profile at buses {conflicts}")
    configured = deepcopy(dict(case))
    for row in bus:
        row[BASE_KV] = IEEE118_SOURCE_KV[int(row[BUS_I])]
    configured["bus"] = bus
    configured["voltage_base_profile"] = {
        **ieee118_voltage_base_profile(), "original_bus_base_kv_ll": previous,
    }
    return configured


def impedance_base_ohm(kv_ll: float, base_mva: float) -> float:
    """Three-phase-system impedance base: Z_base = kV_LL**2 / MVA_base."""
    voltage = _positive(kv_ll, "kv_ll")
    power = _positive(base_mva, "base_mva")
    try:
        result = voltage**2 / power
    except OverflowError as exc:
        raise ValueError("Impedance base is not representable as a finite positive value") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Impedance base is not representable as a finite positive value")
    return result


def hif_resistance_spec(resistance_ohm: float, kv_ll: float, base_mva: float) -> dict[str, Any]:
    """Convert physical fault resistance using the faulted line's local base.

    Nominal current/power estimates assume one resistive line-to-ground fault
    at nominal phase voltage. They are scale estimates, not solved fault-point
    currents: network impedance, voltage sag and any arcing physics are absent.
    """
    resistance = _positive(resistance_ohm, "resistance_ohm")
    voltage = _positive(kv_ll, "kv_ll")
    power = _positive(base_mva, "base_mva")
    z_base = impedance_base_ohm(voltage, power)
    r_pu = resistance / z_base
    phase_voltage = voltage * 1000 / math.sqrt(3)
    current_base = power * 1e6 / (math.sqrt(3) * voltage * 1000)
    fault_current = phase_voltage / resistance
    current_pu = z_base / resistance
    single_phase_mw = phase_voltage * fault_current / 1e6
    derived = (r_pu, phase_voltage, current_base, fault_current, current_pu, single_phase_mw)
    if not all(math.isfinite(value) and value > 0 for value in derived):
        raise ValueError("Derived HIF base conversion is not finite and positive")
    return {
        "resistance_ohm": resistance, "resistance_pu": r_pu,
        "impedance_base_ohm": z_base, "kv_ll": voltage, "base_mva": power,
        "nominal_phase_voltage_v": phase_voltage, "current_base_a": current_base,
        "nominal_fault_current_a": fault_current, "nominal_fault_current_pu": current_pu,
        "nominal_single_phase_fault_power_mw": single_phase_mw,
        "phase_fault_model": "steady_state_single_phase_resistive_line_to_ground_surrogate",
        "approximation": "nominal phase voltage; ignores network voltage sag, location-dependent voltage and arcing",
        "voltage_base_scope": "local faulted-line line-to-line base; never substitute a system-wide 69 kV base",
    }


def ieee14_hif_branch_eligibility(case: Mapping[str, Any]) -> dict[str, Any]:
    """Identify active same-voltage Line branches under the explicit profile.

    The endpoint map is authoritative even if raw BASE_KV and TAP are zero.
    In particular branch 7--8 crosses 13.8/18 kV and is excluded despite its
    canonical TAP=0. Nonzero-tap transformers and inactive branches are excluded.
    """
    return _hif_branch_eligibility(case, ieee14_voltage_base_profile(), system_name="IEEE14")


def ieee57_hif_branch_eligibility(case: Mapping[str, Any]) -> dict[str, Any]:
    """Keep active same-local-voltage lines, preserving parallel row identity.

    The selected map governs eligibility even when source BASE_KV is zero.
    Cross-voltage endpoints, nonzero taps and phase shifts are not HIF lines.
    """
    return _hif_branch_eligibility(case, ieee57_voltage_base_profile(), system_name="IEEE57")


def ieee118_hif_branch_eligibility(case: Mapping[str, Any]) -> dict[str, Any]:
    """Same-voltage active zero-tap lines; 86-87 and 68-116 cross voltage bases."""
    return _hif_branch_eligibility(case, ieee118_voltage_base_profile(), system_name="IEEE118")


def _hif_branch_eligibility(case: Mapping[str, Any], profile: Mapping[str, Any], *, system_name: str) -> dict[str, Any]:
    voltage_map = profile["bus_base_kv_ll"]
    _bus_matrix(case, voltage_map=voltage_map, system_name=system_name)
    if "branch" not in case:
        raise ValueError(f"{system_name} branch matrix is required")
    branch = np.asarray(case["branch"], dtype=float)
    if branch.ndim != 2 or branch.shape[1] <= BR_STATUS or not np.isfinite(branch).all():
        raise ValueError("branch must be a finite matrix including endpoints, TAP and status")
    rows = []
    for row0, row in enumerate(branch):
        endpoints = row[[F_BUS, T_BUS]]
        if not np.equal(endpoints, np.floor(endpoints)).all() or any(int(bus) not in voltage_map for bus in endpoints):
            raise ValueError(f"Branch endpoints must name external buses in the selected {system_name} profile")
        from_bus, to_bus = (int(bus) for bus in endpoints)
        from_kv, to_kv = voltage_map[from_bus], voltage_map[to_bus]
        reasons = []
        if row[BR_STATUS] <= 0:
            reasons.append("inactive_branch")
        if from_kv != to_kv:
            reasons.append("cross_voltage_branch")
        if row[TAP] != 0:
            reasons.append("transformer_tap")
        if row[SHIFT] != 0:
            reasons.append("phase_shifting_branch")
        rows.append({"branch_row0": row0, "from_bus": from_bus, "to_bus": to_bus,
            "from_kv_ll": from_kv, "to_kv_ll": to_kv,
            "same_voltage": from_kv == to_kv, "kv_ll": from_kv if from_kv == to_kv else None,
            "active": bool(row[BR_STATUS] > 0), "raw_tap": float(row[TAP]),
            "eligible": not reasons, "exclusion_reasons": reasons,
            "reason": reasons[0] if reasons else None})
    return {"voltage_profile": profile["profile_id"],
            "eligible_branch_rows0": [row["branch_row0"] for row in rows if row["eligible"]],
            "excluded_branches": [row for row in rows if not row["eligible"]], "branch_rows": rows}


def eligible_ieee14_hif_branch_rows(case: Mapping[str, Any]) -> list[int]:
    return ieee14_hif_branch_eligibility(case)["eligible_branch_rows0"]


def eligible_ieee57_hif_branch_rows(case: Mapping[str, Any]) -> list[int]:
    return ieee57_hif_branch_eligibility(case)["eligible_branch_rows0"]


def eligible_ieee118_hif_branch_rows(case: Mapping[str, Any]) -> list[int]:
    return ieee118_hif_branch_eligibility(case)["eligible_branch_rows0"]



# Physical resistance classes for single-phase resistive HIF surrogates. The
# bounds encode the experiment's working vocabulary (moderately resistive,
# representative, weak, extreme); they are a research classification of
# physical ohms, not a standard. Lower bounds are inclusive, upper exclusive,
# except the final open class.
HIF_RESISTANCE_CLASSES_OHM: tuple[dict[str, Any], ...] = (
    {"name": "low_resistance_fault", "lower": 0.0, "upper": 50.0,
     "interpretation": "low or moderate fault resistance; not treated as an HIF"},
    {"name": "moderately_resistive", "lower": 50.0, "upper": 100.0,
     "interpretation": "moderately resistive fault (69 kV: ~400-800 A)"},
    {"name": "moderately_high_resistance", "lower": 100.0, "upper": 200.0,
     "interpretation": "moderately high resistance, entering HIF territory (69 kV: ~200-400 A)"},
    {"name": "representative_hif", "lower": 200.0, "upper": 500.0,
     "interpretation": "representative HIF (69 kV: ~80-200 A)"},
    {"name": "weak_hif", "lower": 500.0, "upper": 1000.0,
     "interpretation": "weak HIF (69 kV: ~40-80 A)"},
    {"name": "extreme_weak_hif", "lower": 1000.0, "upper": 5000.0,
     "interpretation": "extreme or very weak HIF; detection-limit population (69 kV: ~8-40 A)"},
    {"name": "near_open_circuit", "lower": 5000.0, "upper": None,
     "interpretation": "near-open-circuit downed conductor; a few amperes at 69 kV"},
)
HIF_DETECTION_LIMIT_BAND_OHM = (1000.0, 5000.0)


def hif_resistance_class(resistance_ohm: float) -> str:
    """Name the physical resistance class of one fault in ohms.

    The classification is voltage-agnostic by construction: the same ohms are
    a different per-unit severity at 13.8 kV than at 69 kV. Callers that need
    per-unit severity must report the local base alongside the class.
    """
    resistance = _positive(resistance_ohm, "resistance_ohm")
    for entry in HIF_RESISTANCE_CLASSES_OHM:
        upper = entry["upper"]
        if resistance >= entry["lower"] and (upper is None or resistance < upper):
            return str(entry["name"])
    raise ValueError("resistance_ohm is outside every classification interval")


def hif_resistance_classification_table() -> list[dict[str, Any]]:
    """Detached copy of the classification with the 69 kV / 100 MVA pu bounds."""
    z69 = impedance_base_ohm(69.0, 100.0)
    rows = []
    for entry in HIF_RESISTANCE_CLASSES_OHM:
        rows.append({**entry,
                     "lower_pu_69kv": entry["lower"] / z69,
                     "upper_pu_69kv": None if entry["upper"] is None else entry["upper"] / z69})
    return rows

__all__ = ["IEEE14_VOLTAGE_BASE_PROFILE_ID", "IEEE14_NOMINAL_KV", "ieee14_voltage_base_profile",
           "apply_ieee14_voltage_bases", "impedance_base_ohm", "hif_resistance_spec",
           "ieee14_hif_branch_eligibility", "eligible_ieee14_hif_branch_rows",
           "IEEE57_VOLTAGE_BASE_PROFILE_ID", "IEEE57_RECONSTRUCTION_KV", "ieee57_voltage_base_profile",
           "apply_ieee57_voltage_bases", "ieee57_hif_branch_eligibility", "eligible_ieee57_hif_branch_rows",
           "IEEE118_VOLTAGE_BASE_PROFILE_ID", "IEEE118_SOURCE_KV", "ieee118_voltage_base_profile",
           "apply_ieee118_voltage_bases", "ieee118_hif_branch_eligibility", "eligible_ieee118_hif_branch_rows",
           "get_voltage_base_profile",
           "HIF_RESISTANCE_CLASSES_OHM", "HIF_DETECTION_LIMIT_BAND_OHM", "hif_resistance_class",
           "hif_resistance_classification_table"]
