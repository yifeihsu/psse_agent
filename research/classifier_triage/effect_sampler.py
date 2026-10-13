"""Effect-targeted fault magnitude (2026-10-12).

A root's magnitude is drawn as the effect the balanced detector sees, not as a
physical size: the chi-square excess ``J - dof`` of the operator's WLS over the
excess of the alarm threshold at that network's degrees of freedom,

    ratio = (J - dof) / (chi2.ppf(1 - alpha, dof) - dof),

so ``ratio = 1`` is "just alarming" on IEEE 14 (dof 95, threshold excess 32)
and on IEEE 118 (dof 863, threshold excess 97) alike.  The physical knob is
solved for: the disturbed telemetry is ``z_reference + k * deviation`` where
``deviation`` is the noiseless change the fault makes and ``k`` scales it, and
``k`` is found by bisection (on the log scale, with one fixed sensor-noise
draw) until the ratio lands in the target band.  The implied physical
magnitude follows from the family's law (``implied_magnitude``): a meter
error's amplitude scales with ``k``, a harmonic source's THD with ``sqrt(k)``
(transducer errors are quadratic in the distortion), a fault's current with
``k`` (its resistance roughly with ``1/k``), an unbalance's phase split with
``k``.  The scaling is exact for meters and harmonics and first order for the
three-phase families, whose SCADA deviation is linear in the fault current
and in the unbalance delta for the magnitudes in use.

Why: the same physical recipe lands at very different detector magnitudes on
different networks (October 12: a harmonic root at 14 threshold-widths on
IEEE 14 and 4 on IEEE 118, an HIF at 2 and 4), so a classifier can read the
family off the size, and the size does not transfer.  One band for every
family and system removes that cue; the classifier is left with the shape of
the residual and multiplier pattern.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

CHI2_ALPHA = 0.01
#: The band of the study: log-uniform between "clearly alarming" and "strong", shared by every family.
DEFAULT_BAND = (1.5, 20.0)
#: Relative tolerance on the achieved ratio.
DEFAULT_TOLERANCE = 0.05
#: Bounds of the scale on the drawn physical magnitude; the implied magnitude must also stay physical.
DEFAULT_SCALE_BOUNDS = (0.02, 50.0)
MAX_ITERATIONS = 14

Solver = Callable[[np.ndarray], tuple[float, int] | None]


def threshold_excess(dof: int, alpha: float = CHI2_ALPHA) -> float:
    """Excess of the chi-square alarm threshold over the degrees of freedom."""
    dof = max(int(dof), 1)
    return float(chi2.ppf(1.0 - alpha, dof)) - float(dof)


def effect_ratio(objective: float, dof: int, alpha: float = CHI2_ALPHA) -> float:
    """The detector-relative magnitude of a solve: ``(J - dof) / threshold_excess(dof)``."""
    return (float(objective) - float(dof)) / threshold_excess(dof, alpha)


def draw_target(rng: np.random.Generator, band: Sequence[float] = DEFAULT_BAND) -> float:
    """One target ratio, log-uniform over the band."""
    low, high = (float(v) for v in band)
    if not (0.0 < low <= high):
        raise ValueError(f"band must satisfy 0 < low <= high, got {band!r}")
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def parse_band(text: str | Sequence[float] | None) -> tuple[float, float] | None:
    """``"1.5,20"`` or a pair -> (low, high); None stays None."""
    if text is None:
        return None
    if isinstance(text, str):
        parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
        values = tuple(float(p) for p in parts)
    else:
        values = tuple(float(v) for v in text)
    if len(values) != 2 or not (0.0 < values[0] <= values[1]):
        raise ValueError(f"an effect band is 'low,high' with 0 < low <= high, got {text!r}")
    return values[0], values[1]


def fit_scale(solve: Solver, reference: np.ndarray, deviation: np.ndarray, noise: np.ndarray, target: float, *,
              scale_bounds: Sequence[float] = DEFAULT_SCALE_BOUNDS, tolerance: float = DEFAULT_TOLERANCE,
              max_iterations: int = MAX_ITERATIONS, alpha: float = CHI2_ALPHA) -> dict[str, Any]:
    """Find the scale ``k`` whose solve of ``reference + k * deviation + noise`` has ratio ``target``.

    ``solve`` returns ``(objective, dof)`` or None when the WLS does not converge.  The ratio is
    monotone in ``k`` once the scaled deviation dominates the noise, which the lower bound is chosen
    to guarantee in practice; the search brackets the band first and reports ``reachable=False``
    when the bounds cannot reach it (the physical magnitude would be absurd, or the deviation is
    too weak).  Returns ``scale``, the achieved ``ratio``, ``objective``, ``dof``, ``iterations``,
    ``reachable`` and the solve at the bounds.
    """
    reference = np.asarray(reference, dtype=float)
    deviation = np.asarray(deviation, dtype=float)
    noise = np.asarray(noise, dtype=float)
    low, high = (float(v) for v in scale_bounds)
    if not (0.0 < low < high):
        raise ValueError("scale_bounds must be positive and ordered")
    if not np.isfinite(deviation).all() or float(np.max(np.abs(deviation))) == 0.0:
        return {"reachable": False, "reason": "no_deviation", "iterations": 0}

    cache: dict[float, tuple[float, int] | None] = {}

    def ratio_at(k: float) -> float | None:
        if k not in cache:
            cache[k] = solve(reference + k * deviation + noise)
        solved = cache[k]
        return None if solved is None else effect_ratio(solved[0], solved[1], alpha)

    # A bound where the WLS diverges (a scaled disturbance far beyond the model's range) is pulled
    # inwards geometrically until the solve converges; the band is then judged on that bracket.
    iterations = 0
    r_low = ratio_at(low)
    iterations += 1
    while r_low is None and low * 2.0 < high:
        low *= 2.0
        r_low = ratio_at(low)
        iterations += 1
    r_high = ratio_at(high)
    iterations += 1
    while r_high is None and high / 2.0 > low:
        high /= 2.0
        r_high = ratio_at(high)
        iterations += 1
    if r_low is None or r_high is None:
        return {"reachable": False, "reason": "wls_nonconvergence_at_bounds", "iterations": iterations,
                "ratio_at_bounds": [r_low, r_high], "bracket": [low, high]}
    if r_high < target * (1.0 - tolerance):
        return {"reachable": False, "reason": "too_weak_within_bounds", "iterations": iterations,
                "ratio_at_bounds": [r_low, r_high], "bracket": [low, high]}
    if r_low > target * (1.0 + tolerance):
        return {"reachable": False, "reason": "too_strong_within_bounds", "iterations": iterations,
                "ratio_at_bounds": [r_low, r_high], "bracket": [low, high]}
    lo, hi = math.log(low), math.log(high)
    best_k, best_ratio = None, None
    for _ in range(int(max_iterations)):
        mid = math.exp(0.5 * (lo + hi))
        ratio = ratio_at(mid)
        iterations += 1
        if ratio is None:
            # a non-converging interior solve: shrink towards the weaker side
            hi = math.log(mid)
            continue
        if best_ratio is None or abs(ratio - target) < abs(best_ratio - target):
            best_k, best_ratio = mid, ratio
        if abs(ratio - target) <= tolerance * target:
            break
        if ratio < target:
            lo = math.log(mid)
        else:
            hi = math.log(mid)
    if best_k is None:
        return {"reachable": False, "reason": "no_converging_interior_solve", "iterations": iterations,
                "ratio_at_bounds": [r_low, r_high]}
    objective, dof = cache[best_k]  # type: ignore[misc]
    return {"reachable": True, "scale": float(best_k), "ratio": float(best_ratio), "target": float(target),
            "objective": float(objective), "dof": int(dof), "iterations": iterations, "ratio_at_bounds": [r_low, r_high]}


def wls_solver(case: str | Mapping[str, Any], sigma: Sequence[float], metadata: Mapping[str, Any] | None = None) -> Solver:
    """The operator's balanced WLS as a ``Solver`` (``research.classifier_triage.data.wls_payload``)."""
    from research.classifier_triage.data import wls_payload

    declared = dict(metadata) if metadata else {"sigma_z": [float(v) for v in sigma]}

    def solve(z: np.ndarray) -> tuple[float, int] | None:
        try:
            payload = wls_payload({"case": case, "measurements": [float(v) for v in z], "metadata": declared})
        except Exception:
            return None
        if payload is None:
            return None
        return float(payload["objective"]), int(payload["dof"])

    return solve


# ----------------------------------------------------------- implied magnitudes

def implied_magnitude(family: str, label: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """The physical magnitude a scale implies for a family's drawn label (first order where noted)."""
    k = float(scale)
    out: dict[str, Any] = {"effect_scale": k}
    if family in ("measurement", "measurement_error", "mimic"):
        if "amplitude" in label:
            out["amplitude"] = float(label["amplitude"]) * k
        if "amplitudes" in label:
            out["amplitudes"] = [float(a) * k for a in label["amplitudes"]]
        out["law"] = "amplitude scales with k (exact)"
    elif family in ("harmonic", "harmonic_anomaly"):
        thd = label.get("thd_target", label.get("actual_thd"))
        if thd is not None:
            out["thd"] = float(thd) * math.sqrt(k)
        out["law"] = "THD scales with sqrt(k): the transducer errors are quadratic in the distortion (exact to second order)"
    elif family in ("hif", "high_impedance_fault"):
        for key in ("fault_current_pu", "nominal_fault_current_pu", "fault_current_a"):
            if label.get(key) is not None:
                out[key] = float(label[key]) * k
        for key in ("fault_power_mw", "nominal_single_phase_fault_power_mw"):
            if label.get(key) is not None:
                out[key] = float(label[key]) * k
        for key in ("r_hif_ohm", "resistance_ohm", "r_hif_pu", "resistance_pu"):
            if label.get(key) is not None:
                out[key] = float(label[key]) / k
        out["law"] = "fault current and power scale with k, resistance with 1/k (first order, Thevenin impedance << R)"
    elif family in ("three_phase_unbalance", "unbalance", "three_phase_imbalance"):
        split = label.get("load_split", {}) if isinstance(label.get("load_split"), Mapping) else {}
        fractions = split.get("fractions") if isinstance(split, Mapping) else None
        if isinstance(fractions, Mapping):
            out["fractions"] = {name: 1.0 / 3.0 + k * (float(value) - 1.0 / 3.0) for name, value in fractions.items()}
        out["law"] = "phase split deviates from equal shares by k times the drawn deviation (first order)"
    elif family in ("parameter", "parameter_error"):
        for key in ("r_factor", "x_factor"):
            if label.get(key) is not None:
                out[key] = float(label[key]) ** k
        out["law"] = "the factor's logarithm scales with k (exact when the case is re-solved, first order otherwise)"
    else:
        out["law"] = "no magnitude law for this family; deviation scaled by k"
    return out


def scale_phasor_rows(rows: Sequence[Mapping[str, Any]], clean_rows: Sequence[Mapping[str, Any]], scale: float,
                      fields: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """Disturbed phasor rows re-magnituded: clean + k * (disturbed - clean) on each (magnitude, angle) pair."""
    k = float(scale)
    out = []
    for row, clean in zip(rows, clean_rows):
        item = dict(row)
        for mag_key, ang_key in fields:
            mag = np.asarray(row[mag_key], dtype=float)
            ang = np.deg2rad(np.asarray(row[ang_key], dtype=float))
            mag0 = np.asarray(clean[mag_key], dtype=float)
            ang0 = np.deg2rad(np.asarray(clean[ang_key], dtype=float))
            disturbed = mag * np.exp(1j * ang)
            reference = mag0 * np.exp(1j * ang0)
            scaled = reference + k * (disturbed - reference)
            item[mag_key] = np.abs(scaled).tolist()
            item[ang_key] = np.rad2deg(np.angle(scaled)).tolist()
        out.append(item)
    return out


VOLTAGE_PHASOR_FIELDS = (("vln_pu", "ang_deg"),)
CURRENT_PHASOR_FIELDS = (("i_from_pu", "ang_from_deg"), ("i_to_pu", "ang_to_deg"))


def rescale_corpus_row(row: Mapping[str, Any], case: str, rng: np.random.Generator, target: float, *,
                       scale_bounds: Sequence[float] = DEFAULT_SCALE_BOUNDS, tolerance: float = DEFAULT_TOLERANCE,
                       solver: Solver | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """A waveform or meter corpus row re-magnituded to ``target`` with a fresh sensor-noise draw.

    The row carries ``z_true`` (the same operating point without the disturbance), ``z_clean``
    (the disturbance without sensor noise) and ``sigma_z``.  The returned row keeps the physical
    label, adds ``effect`` (scale, achieved ratio, implied magnitude) and, when clean phasor rows
    are present, re-magnitudes the phasors consistently.  ``(None, fit)`` when the band is
    unreachable within the scale bounds.
    """
    sigma = np.asarray(row["sigma_z"], dtype=float)
    reference = np.asarray(row["z_true"], dtype=float)
    disturbed = np.asarray(row["z_clean"], dtype=float)
    deviation = disturbed - reference
    noise = rng.standard_normal(sigma.size) * sigma
    solve = solver or wls_solver(case, sigma)
    fit = fit_scale(solve, reference, deviation, noise, target, scale_bounds=scale_bounds, tolerance=tolerance)
    if not fit.get("reachable"):
        return None, fit
    k = float(fit["scale"])
    out = dict(row)
    out["z_clean"] = (reference + k * deviation).tolist()
    out["z_obs"] = (reference + k * deviation + noise).tolist()
    scenario = str(row.get("scenario") or (row.get("label") or {}).get("error_type") or "")
    family = {"high_impedance_fault": "hif", "three_phase_imbalance": "three_phase_unbalance",
              "harmonic_anomaly": "harmonic", "measurement_error": "measurement"}.get(scenario, scenario)
    out["effect"] = {"target": float(target), **{k_: v for k_, v in fit.items() if k_ != "reachable"},
                     "implied": implied_magnitude(family, row.get("label") or {}, k), "sampler": "effect_targeted_v1"}
    for key, clean_key, fields in (("three_phase_voltages", "three_phase_voltages_clean", VOLTAGE_PHASOR_FIELDS),
                                   ("three_phase_branch_currents", "three_phase_branch_currents_clean", CURRENT_PHASOR_FIELDS)):
        if isinstance(row.get(key), list) and isinstance(row.get(clean_key), list) and len(row[key]) == len(row[clean_key]):
            out[key] = scale_phasor_rows(row[key], row[clean_key], k, fields)
    if isinstance(row.get("scans"), list):
        # a multi-scan window's scans are later re-noised by the generator from the single row; keep one
        out["scans"] = [dict(scan) for scan in row["scans"][:1]]
        out["scan_count"] = 1
    return out, fit
