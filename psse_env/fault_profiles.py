"""Versioned synthetic fault settings and explicit measurement-error models.

The ``accuracy_*`` names identify sensitivity experiments using absolute
per-unit sensor standard deviations. They are not instrument accuracy classes,
percentage-of-reading specifications, or claims about field measurements.
Baseline Gaussian sampling remains the responsibility of existing generators.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np


_VUF_BINS = [
    {"name": "below_1pct", "lower": 0.0, "upper": 0.01, "lower_inclusive": True, "upper_inclusive": False},
    {"name": "1_to_2pct", "lower": 0.01, "upper": 0.02, "lower_inclusive": True, "upper_inclusive": False},
    {"name": "2_to_3pct", "lower": 0.02, "upper": 0.03, "lower_inclusive": True, "upper_inclusive": True},
    {"name": "above_3pct", "lower": 0.03, "upper": None, "lower_inclusive": False, "upper_inclusive": False},
]
_ENERGY_BINS = [
    {"name": "below_1", "lower": 0.0, "upper": 1.0, "lower_inclusive": True, "upper_inclusive": False},
    {"name": "1_to_9", "lower": 1.0, "upper": 9.0, "lower_inclusive": True, "upper_inclusive": False},
    {"name": "9_to_25", "lower": 9.0, "upper": 25.0, "lower_inclusive": True, "upper_inclusive": True},
    {"name": "above_25", "lower": 25.0, "upper": None, "lower_inclusive": False, "upper_inclusive": False},
]
_REVIEWED_V1 = {
    "schema_version": 1,
    "profile_id": "reviewed_v1",
    "field_accuracy_claim": False,
    "scope": "synthetic research settings; observational strata are not admission thresholds",
    "noise_profile_scope": "absolute per-unit standard deviations, not percentage accuracy or certified instrument classes",
    "noise_profiles": {
        "baseline": {"sigma_vm": 0.001, "sigma_pq": 0.01},
        "accuracy_005": {"sigma_vm": 0.001, "sigma_pq": 0.005},
        "accuracy_002": {"sigma_vm": 0.001, "sigma_pq": 0.002},
    },
    "hif": {
        "bands_pu": [[5.0, 10.0], [10.0, 20.0], [20.0, 40.0]],
        "stage_weights": {"early": [2, 2, 1], "full": [1, 1, 1]},
        "weak_band_pu": [20.0, 200.0],
        "location_fraction": [0.25, 0.75],
        "sampling": "weighted band selection, then uniform resistance within the selected band",
    },
    "parameter": {
        "gross_factor_bands": [[0.1, 0.5], [2.0, 5.0]],
        "moderate_factor_bands": [[0.7, 0.95], [1.05, 1.3]],
        "sampling": "equal-probability factor-band selection, then uniform factor within that band",
        "factor_direction": "caller must identify physical_actual/reference versus configured_model/reference",
    },
    "unbalance": {"vuf_definition": "abs(V_negative_sequence)/abs(V_positive_sequence)", "vuf_bins": _VUF_BINS},
    "harmonic": {
        "sensitivity_thd_fraction": [0.01, 0.05],
        "stress_thd_fraction": [0.1, 0.2],
    },
    "signal_energy": {
        "definition": "nonnegative signal-energy statistic supplied by the caller; no probability or degrees-of-freedom assertion",
        "bins": _ENERGY_BINS,
    },
    "measurement_chain": {
        "model": "S_measured = ct_gain * pt_gain * exp(j*(pt_angle_rad-ct_angle_rad)) * S",
        "gain_semantics": "positive multiplicative magnitude ratio",
        "angle_units": "radians",
        "sampled_parameter_ranges": None,
        "baseline_additive_noise_changed": False,
    },
}

# 69 kV line-to-line on the 100 MVA three-phase base: Zbase = 69**2 / 100 = 47.61 ohm.
# This is the main-population stratum only; 13.8 kV lines have Zbase = 1.9044 ohm,
# so the same ohms are 25 times larger in local pu there.
_Z_BASE_69KV_OHM = 69.0**2 / 100.0
_PHYSICAL_HIF_V1_SETTINGS = {
    "resistance_input_unit": "ohm",
    "main_range_ohm": [100.0, 1000.0],
    "bands_ohm": [[100.0, 200.0], [200.0, 500.0], [500.0, 1000.0]],
    "stage_weights": {"early": [2, 2, 1], "full": [1, 1, 1]},
    # Opt-in detection-limit cohort. Deliberately NOT a member of ``bands_ohm``:
    # curriculum weights and ``band=None`` draws are unchanged. Reachable only by
    # ``hif_resistance_ohm(band="detection_limit")`` or the explicit bounds.
    "detection_limit_band_ohm": [1000.0, 5000.0],
    "detection_limit_voltage_stratum_kv_ll": 69.0,
    "location_fraction": [0.25, 0.75],
    "evaluation_sweep_ohm": [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0],
    "extreme_evaluation_ohm": [5000.0],
    "main_voltage_stratum_kv_ll": 69.0,
    "evaluation_voltage_strata_kv_ll": [69.0, 13.8],
    "pu_equivalents_69kv": {
        "impedance_base_ohm": _Z_BASE_69KV_OHM,
        "conversion": "R_pu = R_ohm / (69 kV**2 / 100 MVA) = R_ohm / 47.61; valid for 69 kV lines only",
        "main_range_pu": [100.0 / _Z_BASE_69KV_OHM, 1000.0 / _Z_BASE_69KV_OHM],
        "bands_pu": [[low / _Z_BASE_69KV_OHM, high / _Z_BASE_69KV_OHM]
                     for low, high in ([100.0, 200.0], [200.0, 500.0], [500.0, 1000.0])],
        "detection_limit_band_pu": [1000.0 / _Z_BASE_69KV_OHM, 5000.0 / _Z_BASE_69KV_OHM],
        "evaluation_sweep_pu": [value / _Z_BASE_69KV_OHM for value in (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)],
        "extreme_evaluation_pu": [5000.0 / _Z_BASE_69KV_OHM],
        "note": (
            "Per-unit figures are derived labels on the 69 kV / 100 MVA base and are not the "
            "sampled quantity. 1000 pu at 69 kV is 47.6 kOhm (near-open circuit) and must not "
            "be described as a 1000 ohm HIF; 1000 ohm at 69 kV is 21.0 pu."
        ),
    },
    "voltage_stratification": (
        "Main pure-HIF sampling uses eligible 69 kV lines. Evaluate 13.8 kV lines separately; "
        "the same resistance in ohms is 25 times larger in pu at 13.8 kV than at 69 kV. "
        "Convert with the faulted line's local voltage and actual system MVA base."
    ),
    "eligible_branch_rule": "active same-voltage line with TAP=0; exclude cross-voltage endpoints even when TAP=0",
    "sampling": (
        "weighted main-band selection, then uniform resistance in ohms within that band; "
        "sweeps and the detection-limit band are evaluation-only and never drawn by the curriculum"
    ),
    "physical_model": "steady_state_single_phase_resistive_line_to_ground_surrogate",
}


def get_fault_profile(name: str = "reviewed_v1") -> dict[str, Any]:
    """Return an independent JSON-safe profile; unknown versions are errors."""
    if not isinstance(name, str) or name not in {"reviewed_v1", "ieee14_physical_hif_v1"}:
        raise ValueError(f"Unknown fault profile: {name!r}")
    profile = deepcopy(_REVIEWED_V1)
    if name == "ieee14_physical_hif_v1":
        # Keep the historical profile and its sampler independent of model imports.
        from three_phase_model.voltage_bases import (
            hif_resistance_classification_table, ieee14_voltage_base_profile,
        )

        voltage_profile = ieee14_voltage_base_profile()
        hif = deepcopy(_PHYSICAL_HIF_V1_SETTINGS)
        # Physical-ohm classes (lower inclusive, upper exclusive, final class open)
        # with their 69 kV pu bounds; a research vocabulary, not a standard.
        hif["resistance_classification_ohm"] = hif_resistance_classification_table()
        profile.update({
            "profile_id": name,
            "parent_profile_id": "reviewed_v1",
            "voltage_profile": voltage_profile["profile_id"],
            "voltage_base_profile": voltage_profile,
            "hif": hif,
        })
    return profile


def _count(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_real(value: float, name: str, *, nonnegative: bool = False, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if positive and result <= 0:
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _generator(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    return rng


def measurement_sigma(nb: int, nl: int, noise_profile: str = "baseline") -> np.ndarray:
    """Return standard deviations in [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt] order.

    There are ``3*nb + 4*nl`` entries. Estimator covariance is diag(sigma**2),
    not diag(sigma). This helper only provides weights; it does not draw noise.
    """
    nb = _count(nb, "nb", minimum=1)
    nl = _count(nl, "nl", minimum=0)
    profiles = _REVIEWED_V1["noise_profiles"]
    if not isinstance(noise_profile, str) or noise_profile not in profiles:
        raise ValueError(f"Unknown noise profile: {noise_profile!r}")
    setting = profiles[noise_profile]
    sigma = np.full(3 * nb + 4 * nl, setting["sigma_pq"], dtype=float)
    sigma[:nb] = setting["sigma_vm"]
    return sigma


def vuf_stratum(value: float) -> str:
    """VUF bins: [0,.01), [.01,.02), [.02,.03], (.03,infinity)."""
    value = _finite_real(value, "VUF", nonnegative=True)
    if value < 0.01:
        return "below_1pct"
    if value < 0.02:
        return "1_to_2pct"
    if value <= 0.03:
        return "2_to_3pct"
    return "above_3pct"


def signal_energy_stratum(J: float) -> str:
    """Descriptive J bins: [0,1), [1,9), [9,25], (25,infinity)."""
    value = _finite_real(J, "J", nonnegative=True)
    if value < 1:
        return "below_1"
    if value < 9:
        return "1_to_9"
    if value <= 25:
        return "9_to_25"
    return "above_25"


def parameter_factor(rng: np.random.Generator, cohort: str = "gross") -> float:
    """Draw one R or X multiplier; the caller chooses which parameter changes."""
    rng = _generator(rng)
    if not isinstance(cohort, str) or cohort not in {"gross", "moderate"}:
        raise ValueError("cohort must be 'gross' or 'moderate'")
    bands = _REVIEWED_V1["parameter"][f"{cohort}_factor_bands"]
    low, high = bands[int(rng.integers(len(bands)))]
    return float(rng.uniform(low, high))


def hif_resistance(
    rng: np.random.Generator, band: int | str | Sequence[float] | None = None, stage: str = "full",
) -> float:
    """Draw HIF resistance in pu without selecting a line, phase, or location.

    ``band`` accepts zero-based 0/1/2, exact profile bounds, named ``5_10``,
    ``10_20``, ``20_40``, or ``weak``. Only ``band=None`` uses curriculum
    weights; an explicitly selected band is sampled uniformly at either stage.
    The weak band is opt-in and is never drawn by the ordinary curriculum.
    """
    rng = _generator(rng)
    settings = _REVIEWED_V1["hif"]
    if not isinstance(stage, str) or stage not in settings["stage_weights"]:
        raise ValueError("stage must be 'early' or 'full'")
    bands = settings["bands_pu"]
    if band is None:
        weights = np.asarray(settings["stage_weights"][stage], dtype=float)
        bounds = bands[int(rng.choice(len(bands), p=weights / weights.sum()))]
    elif isinstance(band, Integral) and not isinstance(band, (bool, np.bool_)):
        if not 0 <= int(band) < len(bands):
            raise ValueError("HIF band index must be 0, 1, or 2")
        bounds = bands[int(band)]
    elif isinstance(band, str):
        named = {"5_10": bands[0], "10_20": bands[1], "20_40": bands[2], "weak": settings["weak_band_pu"]}
        if band not in named:
            raise ValueError(f"Unknown HIF band: {band!r}")
        bounds = named[band]
    elif isinstance(band, (list, tuple, np.ndarray)):
        raw = np.asarray(band, dtype=object)
        if raw.shape != (2,):
            raise ValueError("Explicit HIF band must have exactly two profile bounds")
        bounds = [_finite_real(value, "HIF band bound", positive=True) for value in raw]
        if bounds not in bands + [settings["weak_band_pu"]]:
            raise ValueError("Explicit HIF bounds must equal one of the reviewed profile bands")
    else:
        raise ValueError("HIF band must be None, an integer index, a band name, or exact profile bounds")
    return float(rng.uniform(*bounds))


def hif_resistance_ohm(
    rng: np.random.Generator, band: int | str | Sequence[float] | None = None, stage: str = "full",
) -> float:
    """Sample the physical profile's 100--1000 ohm curriculum or its opt-in cohorts.

    ``band`` accepts zero-based 0/1/2, exact profile bounds, or named
    ``100_200``, ``200_500``, ``500_1000``. An explicit band is uniform and
    independent of curriculum stage. The detection-limit cohort is opt-in like
    the reviewed profile's ``weak`` band: request it with ``band="detection_limit"``
    or the exact bounds ``[1000.0, 5000.0]``; it is never drawn by ``band=None``
    because it is not a member of ``bands_ohm`` and integer indices stay 0/1/2.
    Evaluation sweep/extreme resistances are explicit profile metadata and are
    never mixed into this sampler. A caller must select an eligible line and
    convert using that line's local base (1000 ohm is 21.0 pu at 69 kV, not
    1000 pu). The separate historical ``hif_resistance`` helper still returns pu.
    """
    rng = _generator(rng)
    settings = _PHYSICAL_HIF_V1_SETTINGS
    if not isinstance(stage, str) or stage not in settings["stage_weights"]:
        raise ValueError("stage must be 'early' or 'full'")
    bands = settings["bands_ohm"]
    detection_limit = settings["detection_limit_band_ohm"]
    if band is None:
        weights = np.asarray(settings["stage_weights"][stage], dtype=float)
        bounds = bands[int(rng.choice(len(bands), p=weights / weights.sum()))]
    elif isinstance(band, Integral) and not isinstance(band, (bool, np.bool_)):
        if not 0 <= int(band) < len(bands):
            raise ValueError("HIF ohm band index must be 0, 1, or 2")
        bounds = bands[int(band)]
    elif isinstance(band, str):
        named = dict(zip(("100_200", "200_500", "500_1000"), bands))
        named["detection_limit"] = detection_limit
        if band not in named:
            raise ValueError(f"Unknown HIF ohm band: {band!r}")
        bounds = named[band]
    elif isinstance(band, (list, tuple, np.ndarray)):
        raw = np.asarray(band, dtype=object)
        if raw.shape != (2,):
            raise ValueError("Explicit HIF ohm band must have exactly two profile bounds")
        bounds = [_finite_real(value, "HIF ohm band bound", positive=True) for value in raw]
        if bounds not in bands + [detection_limit]:
            raise ValueError("Explicit HIF ohm bounds must equal a physical profile main band or the detection-limit band")
    else:
        raise ValueError("HIF ohm band must be None, an integer index, a band name, or exact profile bounds")
    return float(rng.uniform(*bounds))


def _power_component(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.ndim > 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.number)
            or np.issubdtype(raw.dtype, np.complexfloating)):
        raise ValueError(f"{name} must be a nonempty real scalar or one-dimensional numeric array")
    result = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def measurement_chain_error(
    p: Any, q: Any, *, ct_gain: float = 1.0, pt_gain: float = 1.0,
    ct_angle_rad: float = 0.0, pt_angle_rad: float = 0.0,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Apply a coherent CT/PT error to paired real/reactive power readings.

    CT and PT gains are positive magnitude multipliers. Their angles are
    supplied phase errors in radians: S' = g_CT*g_PT*exp(j*(a_PT-a_CT))*S.
    Thus a phase error mixes P and Q; the helper never perturbs them as
    independent channels. It samples nothing and adds no Gaussian noise.
    P/Q must have equal scalar or one-dimensional shapes and are never mutated.
    No field-certified ranges are implied for the supplied gains or angles.
    """
    p_values, q_values = _power_component(p, "p"), _power_component(q, "q")
    if p_values.shape != q_values.shape:
        raise ValueError("p and q must have identical shapes")
    ct_gain = _finite_real(ct_gain, "ct_gain", positive=True)
    pt_gain = _finite_real(pt_gain, "pt_gain", positive=True)
    ct_angle_rad = _finite_real(ct_angle_rad, "ct_angle_rad")
    pt_angle_rad = _finite_real(pt_angle_rad, "pt_angle_rad")
    gain = ct_gain * pt_gain
    angle = pt_angle_rad - ct_angle_rad
    if not math.isfinite(gain) or not math.isfinite(angle):
        raise ValueError("Combined measurement-chain gain or angle is not finite")
    cosine, sine = math.cos(angle), math.sin(angle)
    with np.errstate(over="ignore", invalid="ignore"):
        p_measured = gain * (cosine * p_values - sine * q_values)
        q_measured = gain * (sine * p_values + cosine * q_values)
    if not np.all(np.isfinite(p_measured)) or not np.all(np.isfinite(q_measured)):
        raise ValueError("Measurement-chain output is not finite")
    if p_values.ndim == 0:
        return float(p_measured), float(q_measured)
    return p_measured, q_measured


__all__ = ["get_fault_profile", "measurement_sigma", "vuf_stratum", "signal_energy_stratum",
           "parameter_factor", "hif_resistance", "hif_resistance_ohm", "measurement_chain_error"]
