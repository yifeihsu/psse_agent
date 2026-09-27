"""HIF diagnostics of the suspicion-gated contract, from PMU snapshot phasors.

Under ``suspicion_gated_diagnostics`` phase-resolved phasors are requested only
on a balanced HIF suspicion, and every alarmed root carries them (generated
from its true state at one PMU sigma).  The HIF rungs therefore work from the
snapshot phasors and the operator's balanced model alone; nothing here reads
a simulator operating point, a scan window or a forward model that only HIF
windows would carry.

* ``zero_sequence_line_screen``: a phase-to-ground fault on a line leaves a
  zero-sequence differential current between the line's two terminals; a
  balanced model error (a line modeled in service that is out, a wrong
  charging susceptance) leaves a purely positive-sequence one.  The screen
  ranks lines by the zero-sequence part and calls the phasors HIF-like only
  when it clears a six-sigma floor.
* ``pmu_hif_estimate``: the closed-form two-terminal position and resistance
  on the localized line and phase, with a sensitivity range from re-running
  it on independently perturbed phasors at the declared sigma.
* ``balanced_hif_prediction``: the accepted fault as a shunt conductance
  ``G = 1 / (3 r_hif_pu)`` at ``alpha`` on a split copy of the balanced model
  (the equivalent the HIF screen fits; the 2026-09-26 study found its fitted G
  within 4% of this).  The state is fitted with the largest normalized
  residuals set aside one at a time, so a coexisting bad meter cannot pull its
  own replacement; the prediction, its effect on every operator channel, and
  a sensitivity envelope over the estimate's range feed the ordinary
  conditioned meter diagnosis.  SCADA voltage channels read one phase (phase
  A) while flows are three-phase totals, so a single-phase fault moves the
  voltage channels by its negative- and zero-sequence effect, which no
  balanced model carries; the acquired phasors measure that offset
  (``phase_voltage_offsets``) and it is removed before the balanced fit and
  restored in the prediction.
"""
from __future__ import annotations

import cmath
import math
from typing import Any, Mapping, Sequence

import numpy as np

from psse_env.providers.hif_screen import (
    GS,
    _fit,
    _h_full,
    _normalized_residuals,
    _Operator,
    _Problem,
    _ybus_dense,
)
from three_phase_nlm.branch_current_analysis import (
    DIFFERENTIAL_DETECTION_SIGMAS,
    PHASES,
    add_branch_current_noise,
    line_differential_phasors,
    two_terminal_hif_estimate,
)

#: Conditioning method recorded on the continuation ledger for this contract.
BALANCED_HIF_CONDITIONING_METHOD = "balanced_split_line_shunt_compensation"
#: The accepted-fit receipt of a PMU snapshot estimate.
PMU_HIF_FIT_METHOD = "pmu_two_terminal_snapshot"
_ROTATION = cmath.exp(2j * math.pi / 3.0)


def _sequence(values: Sequence[complex]) -> tuple[complex, complex, complex]:
    a, b, c = (complex(value) for value in values)
    zero = (a + b + c) / 3.0
    positive = (a + _ROTATION * b + _ROTATION * _ROTATION * c) / 3.0
    negative = (a + _ROTATION * _ROTATION * b + _ROTATION * c) / 3.0
    return zero, positive, negative


def zero_sequence_line_screen(
    voltage_rows: Any,
    current_rows: Any,
    *,
    sigma_pu: float,
    detection_sigmas: float = DIFFERENTIAL_DETECTION_SIGMAS,
    top_k: int = 5,
) -> dict[str, Any] | None:
    """Rank lines by the zero-sequence part of their terminal differential current.

    Each terminal phasor carries independent noise of ``sigma_pu`` per
    rectangular component, so a phase's differential has standard deviation
    ``sqrt(2) * sigma`` per component and its zero-sequence average
    ``sqrt(2 / 3) * sigma``.  The floor is ``detection_sigmas`` of that.
    The faulted phase is the one whose differential departs most from the
    balanced (positive-sequence) pattern.
    """
    differentials = line_differential_phasors(voltage_rows, current_rows)
    if not differentials:
        return None
    sigma = max(float(sigma_pu), 1e-12)
    floor = float(detection_sigmas) * math.sqrt(2.0 / 3.0) * sigma
    ranked = []
    for row0, item in differentials.items():
        zero, positive, negative = _sequence(item["differential"])
        balanced = [positive, positive * _ROTATION * _ROTATION, positive * _ROTATION]
        unbalanced = [abs(value - part) for value, part in zip(item["differential"], balanced)]
        phase_index = int(np.argmax(unbalanced))
        ranked.append({
            "branch_row0": int(row0),
            "line_index1": int(row0) + 1,
            "dss_element": item["dss_element"],
            "from_bus": int(item["from_bus"]),
            "to_bus": int(item["to_bus"]),
            "zero_sequence_pu": float(abs(zero)),
            "negative_sequence_pu": float(abs(negative)),
            "positive_sequence_pu": float(abs(positive)),
            "phase": PHASES[phase_index],
        })
    ranked.sort(key=lambda item: (-item["zero_sequence_pu"], item["branch_row0"]))
    top = ranked[0]
    second = ranked[1]["zero_sequence_pu"] if len(ranked) > 1 else 0.0
    return {
        "method": "zero_sequence_line_differential",
        "top_lines": [{**item, "rank": rank} for rank, item in enumerate(ranked[: max(1, int(top_k))], start=1)],
        "max_zero_sequence_pu": float(top["zero_sequence_pu"]),
        "second_zero_sequence_pu": float(second),
        "zero_sequence_floor_pu": floor,
        "hif_like": bool(top["zero_sequence_pu"] >= floor),
        "branch_current_sigma_pu": sigma,
    }


def pmu_hif_estimate(
    voltage_rows: Any,
    current_rows: Any,
    *,
    branch_row0: int,
    phase: str,
    sigma_pu: float,
    voltage_sigma_pu: float,
    draws: int = 24,
    seed: int = 20260927,
) -> dict[str, Any] | None:
    """Two-terminal closed form with a perturbation sensitivity range.

    The range comes from the estimate on ``draws`` copies of the measured
    phasors with fresh noise at the declared sigmas added on top.  It
    describes how noise moves the estimate; it is not a calibrated interval.
    """
    estimate = two_terminal_hif_estimate(voltage_rows, current_rows, branch_row0=int(branch_row0), phase=str(phase))
    if estimate is None:
        return None
    from three_phase_nlm.measurement_noise import add_voltage_phasor_noise

    rng = np.random.default_rng(int(seed) + 7919 * int(branch_row0) + PHASES.index(str(phase).upper()))
    alphas: list[float] = []
    resistances: list[float] = []
    for _ in range(max(int(draws), 0)):
        voltages = add_voltage_phasor_noise(voltage_rows, rng, float(voltage_sigma_pu))
        currents = add_branch_current_noise(current_rows, rng, float(sigma_pu))
        varied = two_terminal_hif_estimate(voltages, currents, branch_row0=int(branch_row0), phase=str(phase))
        if varied is None:
            continue
        if math.isfinite(varied["alpha_from_from_bus"]) and math.isfinite(varied["r_hif_pu"]) and varied["r_hif_pu"] > 0:
            alphas.append(float(varied["alpha_from_from_bus"]))
            resistances.append(float(varied["r_hif_pu"]))
    if alphas:
        estimate["sensitivity"] = {
            "draws": len(alphas),
            "alpha_range": [float(np.percentile(alphas, 5)), float(np.percentile(alphas, 95))],
            "r_hif_pu_range": [float(np.percentile(resistances, 5)), float(np.percentile(resistances, 95))],
            "interpretation": "5th-95th percentile over re-noised phasors; not a confidence interval",
        }
    return estimate


def pmu_fit_receipt(estimate: Mapping[str, Any]) -> dict[str, Any]:
    """Accepted-fit record the balanced conditioning reads back (no hashes)."""
    sensitivity = estimate.get("sensitivity") if isinstance(estimate.get("sensitivity"), Mapping) else {}
    alpha = float(estimate["alpha_from_from_bus"])
    resistance = float(estimate["r_hif_pu"])
    return {
        "method": PMU_HIF_FIT_METHOD,
        "success": True,
        "candidate_branch_row0": int(estimate["branch_row0"]),
        "estimated": {
            "alpha_from_from_bus": alpha,
            "r_hif_pu": resistance,
            "r_hif_ohm": float(estimate.get("r_hif_ohm", math.nan)),
            "phase": str(estimate.get("phase") or ""),
        },
        "uncertainty": {
            "alpha_range": list(sensitivity.get("alpha_range") or [alpha, alpha]),
            "r_hif_pu_range": list(sensitivity.get("r_hif_pu_range") or [resistance, resistance]),
        },
        "independent_of_current_scada": True,
    }


def phase_voltage_offsets(voltage_rows: Any, bus: np.ndarray) -> np.ndarray:
    """Per-bus |V_a| - |V_1| from acquired phasors, in the operator's bus order.

    The SCADA voltage channel reads phase A; the balanced model predicts the
    positive-sequence magnitude.  Buses without phasors get no offset.
    """
    from three_phase_nlm.branch_current_analysis import voltage_rows_to_phasors

    phasors = voltage_rows_to_phasors(voltage_rows)
    offsets = np.zeros(bus.shape[0])
    for position, number in enumerate(np.asarray(bus)[:, 0].astype(int)):
        values = phasors.get(int(number))
        if values is None:
            continue
        _zero, positive, _negative = _sequence(values)
        offsets[position] = abs(values[0]) - abs(positive)
    return offsets


def balanced_hif_prediction(
    base_mva: float,
    bus: np.ndarray,
    branch: np.ndarray,
    z: Sequence[float],
    sigma: Sequence[float],
    exact_indices: Sequence[int],
    fit: Mapping[str, Any],
    *,
    voltage_offsets: Sequence[float] | None = None,
    normalized_residual_threshold: float = 4.0,
    max_set_aside_fraction: float = 0.10,
    va0: Sequence[float] | None = None,
    vm0: Sequence[float] | None = None,
) -> dict[str, Any]:
    """HIF-present prediction of every operator channel from an accepted PMU fit.

    Returns the keys the conditioned meter diagnosis and the conditioned WLS
    use: ``predicted_hif_measurements``, ``measurement_effect`` (prediction
    minus the unfaulted model at the same bus voltages) and the envelope
    ``prediction_lower``/``prediction_upper`` over the estimate's range.
    """
    operator = _Operator(base_mva, bus, branch)
    shift = np.zeros(operator.nz)
    if voltage_offsets is not None:
        offsets = np.asarray(voltage_offsets, dtype=float).reshape(-1)
        if offsets.size != operator.nb:
            raise ValueError("voltage_offsets needs one entry per operator bus")
        shift[: operator.nb] = offsets
    # The balanced model is fitted on positive-sequence voltage channels.
    z_vector = np.asarray(z, dtype=float) - shift
    problem = _Problem(z_vector, np.asarray(sigma, dtype=float), tuple(sorted(int(i) for i in exact_indices)),
                       operator.nb, operator.ref)
    row0 = int(fit["candidate_branch_row0"])
    estimated = fit["estimated"]
    alpha = float(estimated["alpha_from_from_bus"])
    resistance = float(estimated["r_hif_pu"])
    if not (0.0 < alpha < 1.0) or not (math.isfinite(resistance) and resistance > 0.0):
        raise ValueError("an accepted HIF fit needs 0 < alpha < 1 and a positive resistance")
    if not 0 <= row0 < operator.nl:
        raise ValueError(f"HIF branch row {row0} is outside the operator model")
    if va0 is None or vm0 is None:
        va0 = np.deg2rad(operator.bus[:, 8] - operator.bus[operator.ref, 8])
        vm0 = operator.bus[:, 7]

    def model(a: float, r: float):
        split = operator.split_model(row0, float(min(max(a, 1e-3), 1.0 - 1e-3)))
        split.params = []
        split.bus[operator.nb, GS] = operator.base_mva / (3.0 * float(r))
        return split

    center = model(alpha, resistance)
    budget = max(1, int(max_set_aside_fraction * z_vector.size))
    set_aside: list[int] = []
    state = None
    for _ in range(budget + 1):
        state = _fit(center, problem, set_aside, va0, vm0)
        if not state.get("success"):
            raise ValueError("the HIF-conditioned balanced fit did not converge")
        residuals = _normalized_residuals(center, problem, state)
        worst = int(np.argmax(residuals))
        if residuals[worst] < float(normalized_residual_threshold) or len(set_aside) >= budget:
            break
        set_aside.append(worst)

    def predict(split, fitted) -> tuple[np.ndarray, np.ndarray]:
        full = _h_full(*_ybus_dense(split.base_mva, split.bus, split.branch), fitted["va"], fitted["vm"])
        present = np.where(split.sel < 0, 0.0, full[np.where(split.sel < 0, 0, split.sel)]) + shift
        unfaulted_model = operator.base_model()
        unfaulted = _h_full(
            *_ybus_dense(unfaulted_model.base_mva, unfaulted_model.bus, unfaulted_model.branch),
            fitted["va"][: operator.nb], fitted["vm"][: operator.nb],
        )
        return present, present - unfaulted

    predicted, effect = predict(center, state)
    uncertainty = fit.get("uncertainty") if isinstance(fit.get("uncertainty"), Mapping) else {}
    alphas = sorted({alpha, *[float(value) for value in uncertainty.get("alpha_range") or []]})
    resistances = sorted({resistance, *[float(value) for value in uncertainty.get("r_hif_pu_range") or [] if float(value) > 0]})
    vectors = [predicted]
    for a in (alphas[0], alphas[-1]):
        for r in (resistances[0], resistances[-1]):
            if a == alpha and r == resistance:
                continue
            varied = model(a, r)
            fitted = _fit(varied, problem, set_aside, state["va"][: operator.nb], state["vm"][: operator.nb])
            if not fitted.get("success"):
                raise ValueError("an HIF sensitivity refit did not converge")
            vectors.append(predict(varied, fitted)[0])
    stacked = np.vstack(vectors)
    return {
        "method": BALANCED_HIF_CONDITIONING_METHOD,
        "predicted_hif_measurements": predicted.tolist(),
        "measurement_effect": effect.tolist(),
        "prediction_lower": np.min(stacked, axis=0).tolist(),
        "prediction_upper": np.max(stacked, axis=0).tolist(),
        "state_fit_set_aside_indices": list(set_aside),
        "phase_voltage_offsets_applied": bool(np.any(shift != 0.0)),
        "branch_row0": row0,
        "alpha_from_from_bus": alpha,
        "shunt_conductance_pu": 1.0 / (3.0 * resistance),
    }


__all__ = [
    "BALANCED_HIF_CONDITIONING_METHOD", "PMU_HIF_FIT_METHOD", "balanced_hif_prediction",
    "phase_voltage_offsets", "pmu_fit_receipt", "pmu_hif_estimate", "zero_sequence_line_screen",
]
