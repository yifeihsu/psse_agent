"""Conservative meter-repair proposals conditioned on a diagnosed HIF.

These helpers consume predictions supplied by a caller; they neither estimate
HIF parameters nor certify that the supplied physical model is correct. Scores
are measurement-standardized discrepancies, NOT WLS normalized residuals or
chi-square statistics. Prediction envelopes are deterministic model-sensitivity
bounds, not confidence intervals and not a measurement covariance.

Original observations are never modified. A proposal passing the conditional
checks still requires independent physical-model validation and a truth audit
in an experiment. In particular, ``recovery_supported`` with no candidates only
means that this check found no remaining discrepancy.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


_SCORE_DEFINITION = "signed distance outside prediction envelope / measurement sigma; not WLS normalized residual"


def _vector(value: Sequence[float], name: str, length: int | None = None) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 1 or (length is not None and result.size != length):
        raise ValueError(f"{name} must be a one-dimensional vector of matching length")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _inputs(z: Sequence[float], predicted: Sequence[float], sigma_z: Sequence[float] | float):
    observed = _vector(z, "z")
    center = _vector(predicted, "prediction", observed.size)
    sigma = np.asarray(sigma_z, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(observed.size, float(sigma))
        if not np.isfinite(float(sigma_z)) or float(sigma_z) <= 0:
            raise ValueError("sigma_z must be finite and strictly positive")
    sigma = _vector(sigma, "sigma_z", observed.size)
    if np.any(sigma <= 0):
        raise ValueError("sigma_z must be finite and strictly positive")
    return observed, center, sigma


def _positive(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return parsed


def _candidate_budget(count: int, fraction: float, limit: int | None) -> int:
    fraction = _positive(fraction, "max_candidate_fraction")
    if fraction > 1:
        raise ValueError("max_candidate_fraction must be <= 1")
    budget = max(1, int(np.floor(count * fraction))) if count else 0
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 0:
            raise ValueError("max_candidate_count must be a nonnegative integer or None")
        budget = min(budget, int(limit))
    return budget


def _envelope_scores(z: np.ndarray, lower: np.ndarray, upper: np.ndarray, sigma: np.ndarray):
    return (np.maximum(z - upper, 0.0) + np.minimum(z - lower, 0.0)) / sigma


def _indices(mask: np.ndarray) -> list[int]:
    return np.flatnonzero(mask).tolist()


def diagnose_conditioned_meter_errors(
    z: Sequence[float],
    predicted_hif: Sequence[float],
    sigma_z: Sequence[float] | float,
    *,
    prediction_lower: Sequence[float] | None = None,
    prediction_upper: Sequence[float] | None = None,
    event_effect: Sequence[float] | None = None,
    detection_sigma: float = 5.0,
    max_envelope_width_sigma: float = 2.0,
    max_candidate_fraction: float = 0.10,
    max_candidate_count: int | None = None,
) -> dict[str, Any]:
    """Propose sparse meter repairs against a HIF-present forward prediction.

    A candidate exceeds ``detection_sigma`` beyond the supplied envelope. It
    is repairable only when the envelope's *full width* is at most
    ``max_envelope_width_sigma`` measurement sigmas. Replacements use the
    HIF-present center, retaining the physical effect on repaired channels.
    All other observations are retained exactly.

    The sparse-candidate budget rejects broad residual patterns that could
    indicate a wrong physical model. It is an explicit research assumption,
    not a proof that a sufficiently sparse discrepancy is a faulty meter.
    No replacements are applied to the returned proposal if any candidate is
    unsupported or the candidate budget is exceeded.
    """
    observed, center, sigma = _inputs(z, predicted_hif, sigma_z)
    threshold = _positive(detection_sigma, "detection_sigma")
    max_width = _positive(max_envelope_width_sigma, "max_envelope_width_sigma")
    budget = _candidate_budget(observed.size, max_candidate_fraction, max_candidate_count)
    if (prediction_lower is None) != (prediction_upper is None):
        raise ValueError("prediction_lower and prediction_upper must be supplied together")
    lower = center.copy() if prediction_lower is None else _vector(prediction_lower, "prediction_lower", observed.size)
    upper = center.copy() if prediction_upper is None else _vector(prediction_upper, "prediction_upper", observed.size)
    if np.any(lower > center) or np.any(center > upper):
        raise ValueError("prediction envelope must satisfy lower <= predicted_hif <= upper")
    effect = None if event_effect is None else _vector(event_effect, "event_effect", observed.size)

    scores = _envelope_scores(observed, lower, upper, sigma)
    widths = (upper - lower) / sigma
    candidates = np.abs(scores) > threshold
    unsupported = candidates & (widths > max_width)
    candidate_indices = _indices(candidates)
    reasons: list[str] = []
    if not observed.size:
        reasons.append("no_observable_channels")
    if len(candidate_indices) > budget:
        reasons.append("broad_residual_pattern_or_too_many_meter_errors")
    if np.any(unsupported):
        reasons.append("prediction_envelope_too_wide_for_candidate_repair")

    proposed = observed.copy()
    if not reasons:
        proposed[candidates] = center[candidates]
    remaining = _envelope_scores(proposed, lower, upper, sigma)
    if np.any(np.abs(remaining) > threshold):
        reasons.append("remaining_conditional_discrepancy")
    supported = not reasons
    repairs = [
        {"index0": index, "observed": float(observed[index]), "replacement": float(center[index]),
         "conditional_score": float(scores[index]), "envelope_width_sigma": float(widths[index])}
        for index in candidate_indices
    ] if supported else []

    return {
        "method": "hif_present_forward_prediction",
        "score_definition": _SCORE_DEFINITION,
        "envelope_interpretation": "deterministic sensitivity bounds; not a confidence interval or covariance",
        "detection_sigma": threshold,
        "max_envelope_width_sigma": max_width,
        "candidate_budget": budget,
        "candidate_indices": candidate_indices,
        "unsupported_candidate_indices": _indices(unsupported),
        "wide_envelope_indices": _indices(widths > max_width),
        "candidate_event_overlap_indices": [] if effect is None else _indices(candidates & (np.abs(effect) >= sigma)),
        "conditional_scores": scores.tolist(),
        "center_measurement_standardized_scores": ((observed - center) / sigma).tolist(),
        "prediction_envelope_width_sigma": widths.tolist(),
        "remaining_conditional_scores": remaining.tolist(),
        "proposed_measurements": proposed.tolist(),
        "replacements": repairs,
        "model_adequate": bool(observed.size and len(candidate_indices) <= budget and not np.any(unsupported)),
        "recovery_supported": supported,
        "ambiguous": bool(reasons or np.any(widths > max_width)),
        "failure_reasons": reasons,
        "physical_model_validated": False,
        "requires_independent_physical_model_validation": True,
        "no_remaining_discrepancy_evidence": bool(observed.size and not np.any(np.abs(remaining) > threshold)),
        "success_interpretation": "conditional proposal only; zero candidates means no discrepancy detected, not physical recovery",
    }


def diagnose_nonoverlap_meter_errors(
    z: Sequence[float],
    predicted_absent: Sequence[float],
    sigma_z: Sequence[float] | float,
    *,
    event_effect: Sequence[float],
    detection_sigma: float = 5.0,
    support_sigma: float = 1.0,
    max_candidate_fraction: float = 0.10,
    max_candidate_count: int | None = None,
) -> dict[str, Any]:
    """Conservative fallback under a nonoverlapping-impact assumption.

    Exclude channels whose predicted HIF effect is at least ``support_sigma``
    measurement sigmas. Compare retained channels to the *same-model* absent
    prediction, and propose replacements only there. The caller is responsible
    for ensuring that the absent prediction and event effect share one model
    and operating point.

    Excluded channels still receive a discrepancy check against absent+effect.
    An excursion there prevents all repair proposals because it could be an
    overlapping meter error or an inadequate physical prediction. Absence of
    an excursion does not prove nonoverlap: any known overlap must fail the
    experiment's separate truth audit. This method supplies no uncertainty
    envelope for the effect and explicitly relies on the independence premise.
    """
    observed, absent, sigma = _inputs(z, predicted_absent, sigma_z)
    effect = _vector(event_effect, "event_effect", observed.size)
    threshold = _positive(detection_sigma, "detection_sigma")
    support_threshold = _positive(support_sigma, "support_sigma")
    support = np.abs(effect) / sigma >= support_threshold
    retained = ~support
    budget = _candidate_budget(int(np.sum(retained)), max_candidate_fraction, max_candidate_count)
    absent_scores = (observed - absent) / sigma
    present_scores = (observed - absent - effect) / sigma
    candidates = retained & (np.abs(absent_scores) > threshold)
    unexplained_support = support & (np.abs(present_scores) > threshold)
    candidate_indices = _indices(candidates)
    reasons: list[str] = []
    if not np.any(retained):
        reasons.append("no_observable_channels_after_exclusion")
    if np.any(unexplained_support):
        reasons.append("overlap_or_insufficient_physical_model_evidence")
    if len(candidate_indices) > budget:
        reasons.append("broad_residual_pattern_or_too_many_meter_errors")
    proposed = observed.copy()
    if not reasons:
        proposed[candidates] = absent[candidates]
    remaining = np.where(retained, (proposed - absent) / sigma, (proposed - absent - effect) / sigma)
    if np.any(np.abs(remaining) > threshold):
        reasons.append("remaining_conditional_discrepancy")
    supported = not reasons
    repairs = [
        {"index0": index, "observed": float(observed[index]), "replacement": float(absent[index]),
         "conditional_score": float(absent_scores[index])}
        for index in candidate_indices
    ] if supported else []
    return {
        "method": "nonoverlap_assumption_fallback",
        "score_definition": "retained: (z - predicted_absent) / measurement sigma; excluded: (z - predicted_absent - event_effect) / measurement sigma; not WLS normalized residual",
        "detection_sigma": threshold,
        "support_sigma": support_threshold,
        "candidate_budget": budget,
        "candidate_indices": candidate_indices,
        "unsupported_candidate_indices": _indices(unexplained_support),
        "excluded_indices": _indices(support),
        "retained_indices": _indices(retained),
        "unexplained_excluded_indices": _indices(unexplained_support),
        "conditional_scores": np.where(retained, absent_scores, present_scores).tolist(),
        "remaining_conditional_scores": remaining.tolist(),
        "proposed_measurements": proposed.tolist(),
        "replacements": repairs,
        "recovery_supported": supported,
        "ambiguous": bool(reasons),
        "failure_reasons": reasons,
        "physical_model_validated": False,
        "assumes_nonoverlapping_impacts": True,
        "effect_uncertainty_accounted_for": False,
        "no_remaining_discrepancy_evidence": bool(np.any(retained) and not np.any(np.abs(remaining) > threshold)),
        "requires_independent_overlap_audit": True,
        "success_interpretation": "conditional proposal under independence assumption; exclusions and quiet scores do not prove recovery",
    }
