"""Offline practical-scenario policy and paired measurement visibility audits.

These descriptors are audit labels, never GNN features. Physical relevance,
post-acquisition diagnostic strength, and balanced-input visibility are separate
axes. The numeric cutoffs are research cohort choices, not protection settings,
equipment guarantees, or declarations that low-signal faults are harmless.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.linalg import qr
from scipy.special import ndtr, ndtri

from .feature_schema import FAMILY_NAMES

POLICY_VERSION = "sft_practical_v1"
VISIBILITY_BANDS = ("below2", "boundary2to4", "visible_ge4")
ORACLE_LIMITATION = (
    "The Gaussian oracle knows this exact healthy parent, fault mean and common "
    "covariance. It is not an achievable-performance claim for unknown operating "
    "parents, composite fault classes or the trained GNN."
)


def _finite_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def _nonnegative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def paired_visibility(z_healthy: Any, z_fault: Any, sigma: Any, H: Any = None, *,
                      same_configured_model: bool = True,
                      false_positive_rate: float = .01) -> dict[str, Any]:
    """Audit paired *noiseless means* under the same diagonal Gaussian noise.

    ``sigma`` contains channel standard deviations, not variances. ``H`` is an
    optional unwhitened local balanced-measurement Jacobian at a specified
    reference state. A pivoted QR projects onto its column space without an
    inverse and also supports rank-deficient Jacobians. The residual energy is
    local linearized separation, not a realized noisy WLS objective.

    If configured models differ, z alone omits a changed model input. Return its
    numerical comparison under ``model_discrepancy_visibility`` and withhold the
    primary balanced-input band/oracle; d=0 then does not establish equal inputs.
    """
    healthy = _finite_vector(z_healthy, "z_healthy")
    fault = _finite_vector(z_fault, "z_fault")
    std = _finite_vector(sigma, "sigma")
    if healthy.shape != fault.shape or healthy.shape != std.shape or np.any(std <= 0):
        raise ValueError("paired means and positive sigma must have identical shapes")
    if not 0 < false_positive_rate < 1:
        raise ValueError("false_positive_rate must be between zero and one")
    if not isinstance(same_configured_model, bool):
        raise ValueError("same_configured_model must be explicitly boolean")
    delta = (fault - healthy) / std
    d = float(np.linalg.norm(delta))
    if not np.isfinite(d) or not np.isfinite(d * d):
        raise ValueError("whitened mean separation must be finite")
    measured = {
        "mean_separation_d": d,
        "mean_separation_d_squared": d * d,
        "maximum_channel_mean_separation_sigma": float(np.max(np.abs(delta))),
        "oracle_recall": float(ndtr(d - ndtri(1 - false_positive_rate))),
        "oracle_false_positive_rate": float(false_positive_rate),
        "rawband": "below2" if d < 2 else "boundary2to4" if d < 4 else "visible_ge4",
        "local_projected_residual_energy": None,
        "local_projected_residual_separation_d": None,
        "local_absorbed_state_energy": None,
        "local_jacobian_rank": None,
        "local_residual_degrees_of_freedom": None,
    }
    if H is not None:
        jacobian = np.asarray(H, dtype=np.float64)
        if jacobian.ndim != 2 or jacobian.shape[0] != healthy.size or not np.all(np.isfinite(jacobian)):
            raise ValueError("H must be a finite matrix with one row per measurement")
        whitened = jacobian / std[:, None]
        q, triangular, _ = qr(whitened, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(triangular))
        tolerance = (np.finfo(np.float64).eps * max(whitened.shape) * diagonal.max()
                     if diagonal.size else 0.0)
        rank = int(np.count_nonzero(diagonal > tolerance))
        coordinates = q[:, :rank].T @ delta
        residual = delta - q[:, :rank] @ coordinates
        energy = float(residual @ residual)
        measured.update(local_projected_residual_energy=energy,
            local_projected_residual_separation_d=float(np.sqrt(energy)),
            local_absorbed_state_energy=float(coordinates @ coordinates),
            local_jacobian_rank=rank, local_residual_degrees_of_freedom=int(healthy.size - rank))
    result = {
        "policy_version": POLICY_VERSION,
        "measurement_comparison": "paired_noiseless_means_common_diagonal_gaussian_covariance",
        "same_configured_model": same_configured_model,
        "oracle_limitation": ORACLE_LIMITATION,
        "local_projection_limitation": "A first-order reference-Jacobian audit; not an observed WLS alarm or full nonlinear identifiability test.",
        "physical_importance_inferred": False,
        **measured,
        "model_discrepancy_visibility": None,
    }
    if not same_configured_model:
        result.update({name: None for name in measured})
        result.update(rawband="not_applicable_configured_model_discrepancy",
            model_discrepancy_visibility={"measurement_only_comparison": measured,
                "reason": "Configured model inputs differ; the observation-mean comparison alone does not describe total allowed-input separation."})
    return result


def classify_scenario(families: Sequence[str], *, measurement_sigma_multiple: float | None = None,
                      parameter_physical_factors: Mapping[str, float] | None = None,
                      max_vuf: float | None = None, hif_injected: bool | None = None,
                      hif_phase_current_sigma: float | None = None,
                      hif_differential_current_sigma: float | None = None,
                      topology_changed_connectivity: bool | None = None,
                      visibility: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify research cohort membership independently of a WLS alarm.

    ``parameter_physical_factors`` maps R and/or X to *physical actual/reference*
    factors. It must not contain configured/reported multipliers or their
    reciprocal without conversion. ``max_vuf`` is max |V2|/|V1| (0.01 means 1%),
    not NEMA line-voltage magnitude deviation. ``hif_phase_current_sigma`` is
    acquired diagnostic-current significance under its declared noise model,
    never the balanced WLS residual or an unspecified current magnitude. The
    preferred ``hif_differential_current_sigma`` is the two-terminal residual
    divided by its standard deviation (sqrt(2) * 0.001 pu for independent
    0.001-pu terminal-current noise); its SFT criterion is >=6. It takes
    precedence over the legacy phase-current >=10 criterion when both exist.

    Missing evidence remains a boundary case. A mixed episode is core only when
    each represented family meets its own core rule. These are membership rules,
    not changes to the physical fault labels. Healthy controls always remain
    controls, independently of any noisy score or visibility diagnostic.
    """
    if isinstance(families, str) or len(set(families)) != len(families) or set(families) - set(FAMILY_NAMES):
        raise ValueError("families must be unique members of the five-family schema")
    measurement = _nonnegative(measurement_sigma_multiple, "measurement_sigma_multiple")
    vuf = _nonnegative(max_vuf, "max_vuf")
    current = _nonnegative(hif_phase_current_sigma, "hif_phase_current_sigma")
    differential = _nonnegative(hif_differential_current_sigma, "hif_differential_current_sigma")
    if hif_injected is not None and not isinstance(hif_injected, bool):
        raise ValueError("hif_injected must be boolean or None")
    if topology_changed_connectivity is not None and not isinstance(topology_changed_connectivity, bool):
        raise ValueError("topology_changed_connectivity must be boolean or None")
    factors = None
    if parameter_physical_factors is not None:
        factors = {str(key).upper(): float(value) for key, value in parameter_physical_factors.items()}
        if not factors or set(factors) - {"R", "X"} or len(factors) != len(parameter_physical_factors):
            raise ValueError("parameter_physical_factors must explicitly map R and/or X")
        if any(not np.isfinite(value) or value <= 0 for value in factors.values()):
            raise ValueError("physical R/X factors must be finite and positive")
    rules: dict[str, dict[str, Any]] = {}

    def record(family, cohort, reason, **details):
        rules[family] = {"cohort": cohort, "reason": reason, **details}

    if "measurement" in families:
        if measurement is None:
            cohort, reason = "boundary", "measurement_sigma_multiple_unavailable"
        elif 10 <= measurement <= 15:
            cohort, reason = "core", "measurement_10_to_15_sigma"
        elif 0 < measurement < 10:
            cohort, reason = "boundary", "measurement_below_10_sigma"
        else:
            cohort, reason = "out_of_scope", "measurement_outside_declared_sft_range"
        record("measurement", cohort, reason, measurement_sigma_multiple=measurement)
    if "parameter" in families:
        if factors is None:
            cohort, reason = "boundary", "physical_parameter_factors_unavailable"
        elif all(.1 <= value <= .5 or 2 <= value <= 5 for value in factors.values()):
            cohort, reason = "core", "physical_R_X_factors_in_sft_range"
        elif all(.1 <= value <= 5 for value in factors.values()) and any(value != 1 for value in factors.values()):
            cohort, reason = "boundary", "physical_parameter_change_below_sft_range"
        else:
            cohort, reason = "out_of_scope", "physical_parameter_factors_outside_sft_range_or_no_change"
        record("parameter", cohort, reason, physical_factors=factors,
               factor_convention="physical_actual_over_reference", configured_model_error_is_distinct=True)
    if "unbalance" in families:
        if vuf is None:
            cohort, reason, stratum = "boundary", "physical_vuf_unavailable", "unknown"
        elif vuf >= .01:
            cohort, reason = "core", "physical_vuf_at_least_1_percent"
            stratum = "strong_ge2_percent" if vuf >= .02 else "main_1_to_2_percent"
        else:
            cohort, reason, stratum = "boundary", "physical_vuf_below_1_percent", "below1_percent"
        record("unbalance", cohort, reason, maximum_negative_positive_voltage_ratio=vuf, physical_stratum=stratum,
               metric="abs_V2_over_abs_V1_not_NEMA_magnitude_unbalance")
    if "hif" in families:
        significance = differential if differential is not None else current
        threshold = 6 if differential is not None else 10
        criterion = ("two_terminal_differential_current_ge6_sigma" if differential is not None else
                     "legacy_phase_current_ge10_sigma" if current is not None else "unavailable")
        if hif_injected is False:
            cohort, reason = "out_of_scope", "hif_label_without_physical_injection"
        elif hif_injected is None:
            cohort, reason = "boundary", "physical_hif_injection_unconfirmed"
        elif significance is None:
            cohort, reason = "boundary", "acquired_phase_current_significance_unavailable"
        elif significance >= threshold:
            cohort, reason = "core", "injected_hif_meets_acquired_current_diagnostic_criterion"
        else:
            cohort, reason = "boundary", "injected_hif_below_acquired_current_diagnostic_criterion"
        record("hif", cohort, reason, physical_importance_preserved=hif_injected,
               acquired_phase_current_sigma=current,
               acquired_differential_current_sigma=differential,
               diagnostic_criterion=criterion, diagnostic_significance_sigma=significance,
               diagnostic_threshold_sigma=threshold if significance is not None else None,
               interpretation="Injected HIF remains physically positive regardless of balanced visibility or diagnostic significance; boundary does not mean harmless.")
    if "topology" in families:
        cohort = "core" if topology_changed_connectivity is True else "boundary" if topology_changed_connectivity is None else "out_of_scope"
        record("topology", cohort, "connectivity_change_required", changed_connectivity=topology_changed_connectivity)
    cohorts = [rule["cohort"] for rule in rules.values()]
    cohort = ("control" if not families else "out_of_scope" if "out_of_scope" in cohorts else
              "boundary" if "boundary" in cohorts else "core")
    band = visibility.get("rawband") if visibility is not None else None
    if band is not None and band not in (*VISIBILITY_BANDS, "not_applicable_configured_model_discrepancy"):
        raise ValueError("visibility must use a supported paired_visibility rawband")
    return {"policy_version": POLICY_VERSION, "families": list(families), "physical_cohort": cohort,
            "family_rules": rules, "balanced_screenability": band or "not_audited",
            "balanced_visible_ge4": band == "visible_ge4" if band in VISIBILITY_BANDS else None,
            "selection_uses_wls_alarm": False, "physical_membership_uses_balanced_visibility": False,
            "scope_note": "Research cohort choices only. Fault labels remain physical; low balanced-input visibility does not establish harmlessness."}
