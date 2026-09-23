"""Physical HIF resistance units for the normalized IEEE-14 OpenDSS stack.

The checked-in ``IEEE_14_OpenDSS`` model is *normalized*: every bus is 1 kV
line-to-line on a 100 MVA base, so its impedance base is 0.01 ohm everywhere.
A consistently based per-unit solution does not depend on the voltage bases,
therefore a physical fault resistance ``R_ohm`` on a line whose actual voltage
is ``kV_local`` (69 kV for buses 1-5, 13.8 kV for 6-7 and 9-14, 18 kV for 8)
is realized exactly by

    R_pu        = R_ohm / (kV_local**2 / S_base_MVA)          (local base)
    R_model_ohm = R_pu * (1 kV**2 / S_base_MVA) = R_pu * 0.01  (what the DSS model needs)

The physical kilovolts must never reach the injector or the legacy NLM bridge:
those operate in the model's own 1 kV ohms. Only the ohm <-> pu conversion is
voltage-aware. ``1000 pu`` on the 69 kV base is 47.6 kOhm, not a 1000 ohm HIF.

Legacy corpora (before 2026-09-19) sampled ``r_hif_pu`` as system pu with no
voltage meaning and stored ``r_hif_ohm = r_hif_pu * 0.01`` (model ohms) and
``kv_ln = 0.577`` (model volts). Labels written by the reconfigured generator
carry ``resistance_units = "ohm_local_base"`` and the explicit fields below;
the ``label_*`` helpers read either generation without guessing.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping

import numpy as np

from IEEE_14_OpenDSS.constants import BRANCH_ORDER
from three_phase_model.voltage_bases import (
    HIF_DETECTION_LIMIT_BAND_OHM,
    IEEE14_NOMINAL_KV,
    IEEE14_VOLTAGE_BASE_PROFILE_ID,
    hif_resistance_class,
    impedance_base_ohm,
)

from .ieee14_adapter import ELIGIBLE_HIF_BRANCHES, parse_branch_endpoints

S_BASE_MVA = 100.0
#: Line-to-line voltage of the normalized DSS model. Never a physical value.
MODEL_KV_LL = 1.0
MODEL_KV_LN = MODEL_KV_LL / math.sqrt(3.0)
MODEL_ZBASE_OHM = MODEL_KV_LL**2 / S_BASE_MVA  # 0.01 ohm
VOLTAGE_BASE_PROFILE_ID = IEEE14_VOLTAGE_BASE_PROFILE_ID

RESISTANCE_UNITS_OHM_LOCAL_BASE = "ohm_local_base"
RESISTANCE_UNITS_PU_LEGACY = "pu_legacy_normalized_model"

#: Default ohm search box for the estimators: the recommended physical sweep
#: 50-5000 ohm (1.05-105 pu at 69 kV; 26-2625 pu at 13.8 kV).
DEFAULT_HIF_SEARCH_OHM = (50.0, 5000.0)
#: Historical system-pu search box, kept for bit-compatible replay of legacy runs.
LEGACY_HIF_SEARCH_PU = (5.0, 1000.0)
#: Main training population of the physical profile (2.1-21 pu at 69 kV).
MAIN_HIF_BAND_OHM = (100.0, 1000.0)
DETECTION_LIMIT_BAND_OHM = tuple(HIF_DETECTION_LIMIT_BAND_OHM)
#: Recommended logarithmic evaluation sweep, plus the extreme point.
EVALUATION_SWEEP_OHM = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)

VOLTAGE_STRATA = {"69kv": 69.0, "13p8kv": 13.8, "all_same_voltage": None}
BASIS_LOCAL_LINE = "local_line_kv_ll"
BASIS_FROM_BUS_CROSS_VOLTAGE = "from_bus_kv_ll_cross_voltage"
BASIS_EXPLICIT = "explicit_kv_ll"
BASIS_NORMALIZED_MODEL = "normalized_model_kv1"


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return result


# --------------------------------------------------------------------------- lines
def line_endpoint_kv(branch_row0: int) -> tuple[int, int, float, float]:
    """(from_bus, to_bus, from_kV_LL, to_kV_LL) of a BRANCH_ORDER row."""
    row0 = int(branch_row0)
    if row0 < 0 or row0 >= len(BRANCH_ORDER):
        raise IndexError(f"branch_row0={row0} outside BRANCH_ORDER")
    from_bus, to_bus = parse_branch_endpoints(BRANCH_ORDER[row0])
    if from_bus is None or to_bus is None:
        raise ValueError(f"Cannot parse endpoints of {BRANCH_ORDER[row0]!r}")
    return from_bus, to_bus, IEEE14_NOMINAL_KV[from_bus], IEEE14_NOMINAL_KV[to_bus]


def line_kv_ll_for_row0(branch_row0: int) -> float | None:
    """Local line-to-line kV of a same-voltage branch; ``None`` if it crosses voltages."""
    _, _, from_kv, to_kv = line_endpoint_kv(branch_row0)
    return from_kv if math.isclose(from_kv, to_kv) else None


def resolve_line_kv_ll(branch_row0: int, kv_ll: float | None = None) -> dict[str, Any]:
    """Voltage base used for ohm <-> pu conversion on one candidate branch.

    An explicit ``kv_ll`` wins. Otherwise a same-voltage line uses its own
    base. A cross-voltage branch (only ``Line.7-8``, 13.8/18 kV, among the
    ``Line.*`` rows) is not a physical midspan-HIF asset; it is still
    converted on its from-bus base so legacy corpora that used it keep
    replaying, and the receipt flags ``cross_voltage_branch``.
    """
    from_bus, to_bus, from_kv, to_kv = line_endpoint_kv(branch_row0)
    cross = not math.isclose(from_kv, to_kv)
    if kv_ll is not None:
        base, basis = _positive(kv_ll, "kv_ll"), BASIS_EXPLICIT
    elif cross:
        base, basis = from_kv, BASIS_FROM_BUS_CROSS_VOLTAGE
    else:
        base, basis = from_kv, BASIS_LOCAL_LINE
    return {
        "branch_row0": int(branch_row0), "dss_element": BRANCH_ORDER[int(branch_row0)],
        "from_bus": from_bus, "to_bus": to_bus, "from_kv_ll": from_kv, "to_kv_ll": to_kv,
        "kv_ll": base, "impedance_base_ohm": impedance_base_ohm(base, S_BASE_MVA),
        "resistance_basis": basis, "cross_voltage_branch": cross,
        "voltage_base_profile": VOLTAGE_BASE_PROFILE_ID,
    }


#: Same-voltage ``Line.*`` rows under the declared map (16 rows; excludes Line.7-8).
PHYSICAL_ELIGIBLE_HIF_BRANCHES: list[int] = [
    int(row0) for row0 in ELIGIBLE_HIF_BRANCHES if line_kv_ll_for_row0(row0) is not None
]
EXCLUDED_CROSS_VOLTAGE_BRANCHES: list[dict[str, Any]] = [
    {"branch_row0": int(row0), "dss_element": BRANCH_ORDER[row0],
     "from_kv_ll": line_endpoint_kv(row0)[2], "to_kv_ll": line_endpoint_kv(row0)[3],
     "reason": "cross_voltage_branch"}
    for row0 in ELIGIBLE_HIF_BRANCHES if line_kv_ll_for_row0(row0) is None
]


def eligible_rows_for_stratum(voltage_stratum: str = "all_same_voltage") -> list[int]:
    """Physical eligible rows restricted to one declared voltage stratum."""
    if voltage_stratum not in VOLTAGE_STRATA:
        raise ValueError(f"voltage_stratum must be one of {sorted(VOLTAGE_STRATA)}, got {voltage_stratum!r}")
    target = VOLTAGE_STRATA[voltage_stratum]
    return [row0 for row0 in PHYSICAL_ELIGIBLE_HIF_BRANCHES
            if target is None or math.isclose(line_kv_ll_for_row0(row0), target)]


def voltage_stratum_for_kv(kv_ll: float) -> str:
    for name, value in VOLTAGE_STRATA.items():
        if value is not None and math.isclose(float(kv_ll), value):
            return name
    return f"{float(kv_ll):g}kv"


# ---------------------------------------------------------------------- conversion
def local_pu_from_ohm(resistance_ohm: float, kv_ll: float, base_mva: float = S_BASE_MVA) -> float:
    return _positive(resistance_ohm, "resistance_ohm") / impedance_base_ohm(kv_ll, base_mva)


def ohm_from_local_pu(r_pu: float, kv_ll: float, base_mva: float = S_BASE_MVA) -> float:
    return _positive(r_pu, "r_pu") * impedance_base_ohm(kv_ll, base_mva)


def model_ohm_from_local_pu(r_pu: float) -> float:
    """Resistance the normalized 1 kV DSS model needs for a given pu value."""
    return _positive(r_pu, "r_pu") * MODEL_ZBASE_OHM


def model_ohm_from_physical_ohm(resistance_ohm: float, kv_ll: float) -> float:
    return model_ohm_from_local_pu(local_pu_from_ohm(resistance_ohm, kv_ll))


def physical_ohm_from_model_ohm(model_ohm: float, kv_ll: float) -> float:
    return ohm_from_local_pu(_positive(model_ohm, "model_ohm") / MODEL_ZBASE_OHM, kv_ll)


def hif_resistance_record(
    *, branch_row0: int, resistance_ohm: float | None = None, r_hif_pu: float | None = None,
    kv_ll: float | None = None,
) -> dict[str, Any]:
    """Complete, self-describing resistance record for one fault on one line.

    Exactly one of ``resistance_ohm`` (physical) or ``r_hif_pu`` (local-base
    pu) is required. Nominal current/power are scale estimates at nominal
    phase voltage; they ignore network sag and arcing.
    """
    if (resistance_ohm is None) == (r_hif_pu is None):
        raise ValueError("Supply exactly one of resistance_ohm or r_hif_pu")
    base = resolve_line_kv_ll(branch_row0, kv_ll)
    zbase = base["impedance_base_ohm"]
    if resistance_ohm is not None:
        ohm = _positive(resistance_ohm, "resistance_ohm")
        pu = ohm / zbase
    else:
        pu = _positive(r_hif_pu, "r_hif_pu")
        ohm = pu * zbase
    local_kv_ln_v = base["kv_ll"] * 1e3 / math.sqrt(3.0)
    return {
        **base,
        "resistance_ohm": ohm, "r_hif_pu": pu,
        "r_hif_model_ohm": model_ohm_from_local_pu(pu),
        "resistance_units": RESISTANCE_UNITS_OHM_LOCAL_BASE,
        "local_kv_ll": base["kv_ll"], "local_kv_ln": base["kv_ll"] / math.sqrt(3.0),
        "zbase_ohm": zbase, "model_kv_ll": MODEL_KV_LL, "model_kv_ln": MODEL_KV_LN,
        "model_zbase_ohm": MODEL_ZBASE_OHM,
        "nominal_fault_current_a": local_kv_ln_v / ohm,
        "nominal_fault_current_pu": 1.0 / pu,
        "nominal_single_phase_fault_power_mw": local_kv_ln_v**2 / ohm / 1e6,
        "resistance_class": hif_resistance_class(ohm),
        "voltage_stratum": voltage_stratum_for_kv(base["kv_ll"]),
        "model_ohm_convention": "R_model_ohm = r_hif_pu * 0.01 ohm (kV=1 normalized DSS model); "
                                "r_hif_pu = resistance_ohm / (local_kv_ll**2 / 100 MVA)",
    }


# ------------------------------------------------------------------ label helpers
def label_resistance_units(label: Mapping[str, Any] | None) -> str:
    units = (label or {}).get("resistance_units")
    return RESISTANCE_UNITS_OHM_LOCAL_BASE if units == RESISTANCE_UNITS_OHM_LOCAL_BASE else RESISTANCE_UNITS_PU_LEGACY


def label_model_ohm(label: Mapping[str, Any] | None) -> float | None:
    """Normalized-model ohms of a corpus label, whichever generation wrote it."""
    if not isinstance(label, Mapping):
        return None
    if label_resistance_units(label) == RESISTANCE_UNITS_OHM_LOCAL_BASE:
        value = label.get("r_hif_model_ohm")
        if value is None and label.get("r_hif_pu") is not None:
            return model_ohm_from_local_pu(float(label["r_hif_pu"]))
        return None if value is None else float(value)
    if label.get("r_hif_ohm") is not None:
        return float(label["r_hif_ohm"])
    if label.get("r_hif_pu") is not None:
        return model_ohm_from_local_pu(float(label["r_hif_pu"]))
    return None


def label_physical_ohm(label: Mapping[str, Any] | None, *, infer_legacy: bool = True) -> float | None:
    """Physical ohms of a label; legacy labels are inferred on the line's local base.

    Inference applies the declared 69/13.8/18 kV map to a legacy system-pu
    value. The result is what that legacy fault *would be* physically under
    this map; it was never a design input of the legacy corpus.
    """
    if not isinstance(label, Mapping):
        return None
    if label_resistance_units(label) == RESISTANCE_UNITS_OHM_LOCAL_BASE:
        value = label.get("resistance_ohm", label.get("r_hif_ohm"))
        return None if value is None else float(value)
    if not infer_legacy or label.get("r_hif_pu") is None or label.get("branch_row0") is None:
        return None
    base = resolve_line_kv_ll(int(label["branch_row0"]), label.get("local_kv_ll"))
    return float(label["r_hif_pu"]) * base["impedance_base_ohm"]


def label_local_kv_ll(label: Mapping[str, Any] | None) -> float | None:
    if not isinstance(label, Mapping):
        return None
    if label.get("local_kv_ll") is not None:
        return float(label["local_kv_ll"])
    if label.get("branch_row0") is None:
        return None
    return float(resolve_line_kv_ll(int(label["branch_row0"]))["kv_ll"])


# --------------------------------------------------------------------- search box
def resolve_resistance_search_box(
    *, branch_row0: int, r_hif_pu_min: float | None = None, r_hif_pu_max: float | None = None,
    r_hif_ohm_min: float | None = None, r_hif_ohm_max: float | None = None,
    kv_ll: float | None = None, default_ohm: tuple[float, float] | None = DEFAULT_HIF_SEARCH_OHM,
    default_pu: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Resolve the estimator's resistance search box on the candidate line's base.

    Precedence: an ohm box (both bounds) is converted with the local base; a pu
    box (both bounds) is taken as local-base pu; supplying both units is an
    error. With neither, ``default_ohm`` applies (or ``default_pu`` when
    ``default_ohm`` is None, for legacy replay). Returns both units, the base
    and its provenance so payloads can echo the box actually searched.
    """
    ohm_given = r_hif_ohm_min is not None or r_hif_ohm_max is not None
    pu_given = r_hif_pu_min is not None or r_hif_pu_max is not None
    if ohm_given and pu_given:
        raise ValueError("Supply the HIF resistance search box in ohms or in pu, not both")
    if ohm_given and (r_hif_ohm_min is None or r_hif_ohm_max is None):
        raise ValueError("Both r_hif_ohm_min and r_hif_ohm_max are required")
    if pu_given and (r_hif_pu_min is None or r_hif_pu_max is None):
        raise ValueError("Both r_hif_pu_min and r_hif_pu_max are required")
    base = resolve_line_kv_ll(branch_row0, kv_ll)
    zbase = base["impedance_base_ohm"]
    if ohm_given:
        lo, hi, source = float(r_hif_ohm_min), float(r_hif_ohm_max), "explicit_ohm"
        unit = "ohm"
    elif pu_given:
        lo, hi, source = float(r_hif_pu_min), float(r_hif_pu_max), "explicit_pu"
        unit = "pu"
    elif default_ohm is not None:
        lo, hi, source = float(default_ohm[0]), float(default_ohm[1]), "default_ohm"
        unit = "ohm"
    elif default_pu is not None:
        lo, hi, source = float(default_pu[0]), float(default_pu[1]), "default_pu"
        unit = "pu"
    else:
        raise ValueError("No HIF resistance search box available")
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi <= lo:
        raise ValueError(f"Require 0 < min < max for the HIF resistance search box ({unit})")
    if unit == "ohm":
        pu_min, pu_max, ohm_min, ohm_max = lo / zbase, hi / zbase, lo, hi
    else:
        pu_min, pu_max, ohm_min, ohm_max = lo, hi, lo * zbase, hi * zbase
    return {
        "r_hif_pu_min": pu_min, "r_hif_pu_max": pu_max,
        "r_hif_ohm_min": ohm_min, "r_hif_ohm_max": ohm_max,
        "kv_ll": base["kv_ll"], "impedance_base_ohm": zbase,
        "resistance_basis": base["resistance_basis"], "cross_voltage_branch": base["cross_voltage_branch"],
        "voltage_base_profile": VOLTAGE_BASE_PROFILE_ID, "box_source": source,
    }


__all__ = [
    "BASIS_EXPLICIT", "BASIS_FROM_BUS_CROSS_VOLTAGE", "BASIS_LOCAL_LINE", "BASIS_NORMALIZED_MODEL",
    "DEFAULT_HIF_SEARCH_OHM", "DETECTION_LIMIT_BAND_OHM", "EVALUATION_SWEEP_OHM",
    "EXCLUDED_CROSS_VOLTAGE_BRANCHES", "LEGACY_HIF_SEARCH_PU", "MAIN_HIF_BAND_OHM",
    "MODEL_KV_LL", "MODEL_KV_LN", "MODEL_ZBASE_OHM", "PHYSICAL_ELIGIBLE_HIF_BRANCHES",
    "RESISTANCE_UNITS_OHM_LOCAL_BASE", "RESISTANCE_UNITS_PU_LEGACY", "S_BASE_MVA",
    "VOLTAGE_BASE_PROFILE_ID", "VOLTAGE_STRATA",
    "eligible_rows_for_stratum", "hif_resistance_class", "hif_resistance_record",
    "label_local_kv_ll", "label_model_ohm", "label_physical_ohm", "label_resistance_units",
    "line_endpoint_kv", "line_kv_ll_for_row0", "local_pu_from_ohm", "model_ohm_from_local_pu",
    "model_ohm_from_physical_ohm", "ohm_from_local_pu", "physical_ohm_from_model_ohm",
    "resolve_line_kv_ll", "resolve_resistance_search_box", "voltage_stratum_for_kv",
]
