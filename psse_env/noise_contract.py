"""Validate declared sensor noise and estimator weights without truth inputs.

This module checks configuration consistency only. It cannot verify that a
generator actually drew the declared noise, certify independence, or establish
population false-alarm calibration. Fitted predictions and noiseless references
are explicit roles so positive estimator weights cannot make them into noisy
sensor observations by accident.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_ROLES = {"sensor_observation", "noiseless_reference", "model_prediction"}
_DISTRIBUTIONS = {"gaussian", "none", "unknown"}
_REPRESENTATIONS = {"scalar", "complex_rectangular"}
_SEMANTICS = {"per_component", "complex_rms"}


def _sigma(value: Any, name: str, *, positive: bool) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.ndim > 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty scalar or one-dimensional sigma vector")
    if not np.all(np.isfinite(result)) or np.any(result <= 0 if positive else result < 0):
        limit = "strictly positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {limit}")
    return result


def _component_sigma(value: np.ndarray, semantics: str, representation: str) -> np.ndarray:
    if semantics not in _SEMANTICS:
        raise ValueError(f"Unsupported sigma semantics: {semantics!r}")
    if semantics == "complex_rms":
        if representation != "complex_rectangular":
            raise ValueError("complex_rms sigma requires complex_rectangular representation")
        # E[|epsilon|**2] = 2 * sigma_component**2 for equal independent
        # real and imaginary components. This is an explicit declaration.
        return value / np.sqrt(2.0)
    return value


def _json_sigma(value: np.ndarray | None) -> float | list[float] | None:
    if value is None:
        return None
    return float(value) if value.ndim == 0 else value.tolist()


def validate_noise_channel(
    *,
    channel: str,
    role: str,
    distribution: str,
    applied_sigma: float | Sequence[float] | None,
    estimator_sigma: float | Sequence[float],
    representation: str = "scalar",
    applied_sigma_semantics: str = "per_component",
    estimator_sigma_semantics: str = "per_component",
    require_matched_gaussian: bool = False,
) -> dict[str, Any]:
    """Normalize sigmas and optionally require matched Gaussian sensor noise.

    ``per_component`` means each scalar or each rectangular real/imaginary
    component. ``complex_rms`` means sqrt(E[|epsilon|**2]) under declared equal
    independent rectangular components, and is divided by sqrt(2). The units
    and ordering must already agree; this helper does not infer either.

    A scalar sigma broadcasts across the other vector. ``none`` requires zero
    applied sigma; ``unknown`` permits ``None`` and never passes strict mode.
    Noiseless references and model predictions never pass strict mode, even if
    their numerical sigma declarations happen to match estimator weights.
    """
    if not isinstance(channel, str) or not channel.strip():
        raise ValueError("channel must be a nonempty string")
    if role not in _ROLES:
        raise ValueError(f"Unsupported noise role: {role!r}")
    if distribution not in _DISTRIBUTIONS:
        raise ValueError(f"Unsupported noise distribution: {distribution!r}")
    if representation not in _REPRESENTATIONS:
        raise ValueError(f"Unsupported noise representation: {representation!r}")
    if not isinstance(require_matched_gaussian, bool):
        raise ValueError("require_matched_gaussian must be a boolean")
    estimator = _component_sigma(
        _sigma(estimator_sigma, "estimator_sigma", positive=True),
        estimator_sigma_semantics, representation,
    )
    # Validate semantics even when unknown source noise has no numeric sigma.
    _component_sigma(np.asarray(0.0), applied_sigma_semantics, representation)
    applied = None
    if applied_sigma is not None:
        applied = _component_sigma(
            _sigma(applied_sigma, "applied_sigma", positive=distribution == "gaussian"),
            applied_sigma_semantics, representation,
        )
        if distribution == "none" and np.any(applied != 0):
            raise ValueError("distribution='none' requires zero applied_sigma")
        if applied.ndim == estimator.ndim == 1 and applied.shape != estimator.shape:
            raise ValueError("applied_sigma and estimator_sigma shapes must match or one must be scalar")
        try:
            applied, estimator = [np.array(item, copy=True) for item in np.broadcast_arrays(applied, estimator)]
        except ValueError as exc:
            raise ValueError("applied_sigma and estimator_sigma shapes must match or one must be scalar") from exc
    elif distribution != "unknown":
        raise ValueError("applied_sigma=None is only permitted for distribution='unknown'")

    matches = bool(applied is not None and np.allclose(applied, estimator, rtol=1e-12, atol=0))
    reasons = []
    if role != "sensor_observation":
        reasons.append("not_sensor_observation")
    if distribution != "gaussian":
        reasons.append("distribution_not_gaussian")
    if applied is None:
        reasons.append("applied_sigma_unknown")
    elif not matches:
        reasons.append("applied_estimator_sigma_mismatch")
    matched = not reasons
    if require_matched_gaussian and not matched:
        raise ValueError(f"{channel}: matched Gaussian sensor noise required ({', '.join(reasons)})")
    return {
        "schema": "noise_channel_declaration_v1",
        "channel": channel, "role": role, "distribution": distribution,
        "representation": representation,
        "applied_sigma_semantics": applied_sigma_semantics,
        "estimator_sigma_semantics": estimator_sigma_semantics,
        "applied_sigma_per_component": _json_sigma(applied),
        "estimator_sigma_per_component": _json_sigma(estimator),
        "applied_matches_estimator": matches,
        "matched_gaussian": matched,
        "mismatch_reasons": reasons,
        "complex_rms_assumption": "equal independent real/imaginary components" if "complex_rms" in {applied_sigma_semantics, estimator_sigma_semantics} else None,
        "declaration_only": True,
        "source_draws_verified": False,
        "population_calibration_verified": False,
        "limitations": "Checks declared sigmas only; units, ordering, independence, actual draws, selection filtering, and residual-test calibration require separate evidence",
    }


def validate_shared_scada_covariance(
    root_sigma: Sequence[float], scans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require one positive SCADA sigma vector throughout a root's scans.

    Missing scan-level ``sigma_z`` inherits the root declaration; an explicit
    null or differing vector is rejected. This prevents a fit from consuming
    scan overrides while downstream meter decisions and WLS use root weights.
    The returned vector is a copy and inputs are never changed.
    """
    root = _sigma(root_sigma, "root_sigma", positive=True)
    if root.ndim != 1:
        raise ValueError("root_sigma must be a one-dimensional sigma vector")
    if not isinstance(scans, Sequence) or isinstance(scans, (str, bytes)) or not scans:
        raise ValueError("scans must be a nonempty sequence of scan mappings")
    sources = []
    for position, scan in enumerate(scans):
        if not isinstance(scan, Mapping):
            raise ValueError(f"scan {position} must be a mapping")
        if "z_obs" in scan:
            observed = np.asarray(scan["z_obs"], dtype=float)
            if observed.shape != root.shape or not np.all(np.isfinite(observed)):
                raise ValueError(f"scan {position} z_obs must be finite and match root_sigma shape")
        if "sigma_z" in scan:
            override = _sigma(scan["sigma_z"], f"scan {position} sigma_z", positive=True)
            if override.shape != root.shape or not np.allclose(override, root, rtol=1e-12, atol=0):
                raise ValueError(f"scan {position} sigma_z differs from shared root_sigma")
        sources.append({"scan_position0": position, "sigma_source": "scan" if "sigma_z" in scan else "root"})
    return {
        "schema": "shared_scada_covariance_declaration_v1",
        "resolved_sigma_z": root.tolist(), "scan_count": len(scans),
        "channel_count": int(root.size), "shared_covariance": True,
        "covariance_interpretation": "declared diagonal measurement covariance diag(sigma_z**2)",
        "scan_sigma_sources": sources,
        "declaration_only": True, "source_draws_verified": False,
        "population_calibration_verified": False,
    }


def resolve_state_measurement_noise(state: Mapping[str, Any], channel_count: int) -> dict[str, Any]:
    """Resolve declared operator covariance without changing physical readings.

    An absent declaration preserves legacy nominal behavior. New generators
    declare sigma_z explicitly. Zero variance is allowed only for separately
    declared exact structural constraints, never as a hidden numerical floor.
    The current balanced solver accepts diagonal stochastic covariance; reject
    a correlated projection rather than silently dropping its off-diagonals.
    """
    metadata = state.get("metadata") or {}
    operator = metadata.get("operator_noise") or {}
    for container, key in ((state, "sigma_z"), (metadata, "sigma_z"), (operator, "measurement_sigma")):
        if key in container and container[key] is None:
            raise ValueError(f"Explicit {key} cannot be null; omit only for a legacy nominal default")
    candidates = [value for value in (
        state.get("sigma_z"), metadata.get("sigma_z"), operator.get("measurement_sigma")
    ) if value is not None]
    if not candidates:
        return {"measurement_sigma": None, "exact_measurement_indices": [], "source": "legacy_nominal_default"}
    sigma = _sigma(candidates[0], "sigma_z", positive=False)
    if sigma.shape != (channel_count,):
        raise ValueError("sigma_z must match the active external measurement vector")
    for candidate in candidates[1:]:
        other = _sigma(candidate, "sigma_z", positive=False)
        if other.shape != sigma.shape or not np.allclose(other, sigma, rtol=1e-12, atol=0):
            raise ValueError("Conflicting measurement covariance declarations")
    exact_raw = metadata.get("structural_zero_indices", operator.get("structural_zero_indices", []))
    exact = []
    for item in exact_raw:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)) or not 0 <= int(item) < channel_count:
            raise ValueError("structural_zero_indices must contain valid external integer indices")
        exact.append(int(item))
    if len(set(exact)) != len(exact):
        raise ValueError("structural_zero_indices contains duplicates")
    if set(np.flatnonzero(sigma == 0)) != set(exact):
        raise ValueError("Zero measurement sigma must match declared exact structural constraints")
    covariance = operator.get("measurement_covariance", operator.get("covariance"))
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (channel_count, channel_count) or not np.isfinite(covariance).all():
            raise ValueError("operator covariance must match active measurement dimensions")
        if not np.allclose(covariance, np.diag(sigma**2), rtol=1e-10, atol=1e-16):
            raise ValueError("Balanced solver requires diagonal projected covariance; correlated rows need a full-covariance estimator")
    return {"measurement_sigma": sigma.tolist(), "exact_measurement_indices": sorted(exact),
            "source": "declared_sensor_covariance"}
