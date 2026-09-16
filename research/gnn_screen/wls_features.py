"""Current-snapshot float64 WLS evidence, using the shared balanced solver.

Version 1 supports a complete measurement vector and positive diagonal R.
Missing channels and constrained/partial WLS solves are explicitly unavailable.
No phase acquisition or physical labels are accepted by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from tools import lagrangian_port as lp
from .feature_schema import ScreenInputError


def default_measurement_sigma(nb: int, nl: int) -> np.ndarray:
    return np.r_[np.full(nb, 0.001), np.full(2 * nb + 4 * nl, 0.01)]


def configured_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist configured network fields; ignore simulator state and labels.

    Loads, dispatch, stored voltages/angles and generator records do not enter
    screening. A flat initialization and disabled zero-injection constraints
    make them unnecessary even as solver inputs.
    """
    try:
        base = float(case["baseMVA"])
        bus = np.array(case["bus"], dtype=np.float64, copy=True)
        branch = np.array(case["branch"], dtype=np.float64, copy=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ScreenInputError("Expected configured baseMVA, bus and branch arrays.") from exc
    if not np.isfinite(base) or base <= 0 or bus.ndim != 2 or branch.ndim != 2:
        raise ScreenInputError("Invalid configured case dimensions or system base.")
    if bus.shape[0] < 2 or branch.shape[0] < 1 or bus.shape[1] < 9 or branch.shape[1] < 11:
        raise ScreenInputError("Configured case requires bus/branch MATPOWER columns.")
    # Only the operator-visible network fields are relevant, including shunts.
    if not np.all(np.isfinite(bus[:, [0, 1, 4, 5]])) or not np.all(np.isfinite(branch[:, :11])):
        raise ScreenInputError("Nonfinite configured network values.")
    ids = bus[:, 0]
    if np.any(ids != ids.astype(np.int64)) or len(np.unique(ids)) != len(ids):
        raise ScreenInputError("External bus identifiers must be unique integers.")
    ends = branch[:, :2]
    if np.any(ends != ends.astype(np.int64)) or not np.all(np.isin(ends, ids)):
        raise ScreenInputError("Branch endpoints must refer to registered external buses.")
    if not np.all(np.isin(bus[:, 1], [1, 2, 3])) or np.count_nonzero(bus[:, 1] == 3) != 1:
        raise ScreenInputError("Exactly one reference and PQ/PV/reference bus types are required.")
    if not np.all(np.isin(branch[:, 10], [0, 1])):
        raise ScreenInputError("Configured branch status must be zero or one.")
    if np.any(np.hypot(branch[:, 2], branch[:, 3]) == 0) or np.any(branch[:, 8] < 0):
        raise ScreenInputError("Zero branch impedance or negative transformer tap is unsupported.")
    clean_bus = np.zeros((bus.shape[0], max(13, bus.shape[1])), dtype=np.float64)
    clean_bus[:, [0, 1, 4, 5]] = bus[:, [0, 1, 4, 5]]
    clean_bus[:, 7] = 1.0
    return {"baseMVA": base, "bus": clean_bus, "branch": branch[:, :13].copy(),
            "gen": np.empty((0, 21), dtype=np.float64)}


def state_measurements_and_jacobian(case: Mapping[str, Any], theta: np.ndarray,
                                    vm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return h and reference-column-reduced H at this estimated state."""
    internal = lp._copy_result_to_internal(dict(case))
    bus, branch = internal["bus"], internal["branch"]
    nb, nl = len(bus), len(branch)
    ybus, yf, yt = lp.make_ybus(internal["baseMVA"], bus, branch)
    v = vm * np.exp(1j * theta)
    h_full, sf, st = lp.make_jaco(np.r_[theta, vm], ybus, yf, yt, nb, nl,
                                branch[:, 0].astype(int), branch[:, 1].astype(int), v)
    inj = v * np.conj(ybus @ v)
    fitted = np.r_[vm, inj.real, inj.imag, sf.real, sf.imag, st.real, st.imag]
    ref = int(np.flatnonzero(bus[:, 1] == 3)[0])
    jacobian = np.delete(h_full.toarray(), ref, axis=1)
    return fitted.astype(np.float64), jacobian.astype(np.float64)


def residual_covariance(jacobian: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observable local covariance via Cholesky, never an explicit inverse."""
    h = np.asarray(jacobian, dtype=np.float64)
    r = np.asarray(variance, dtype=np.float64)
    if h.ndim != 2 or r.shape != (h.shape[0],) or not np.all(np.isfinite(h)):
        raise ScreenInputError("Invalid WLS Jacobian/covariance dimensions.")
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ScreenInputError("Diagonal covariance must be finite and positive.")
    a = h / np.sqrt(r[:, None])
    singular = np.linalg.svd(a, compute_uv=False)
    rank_tol = np.finfo(np.float64).eps * max(a.shape) * singular[0]
    if h.shape[0] <= h.shape[1] or len(singular) < h.shape[1] or singular[-1] <= rank_tol:
        raise ScreenInputError("Balanced WLS state is not observable with positive residual degrees of freedom.",
                               "inadequate_observability")
    try:
        factor = cho_factor(a.T @ a, lower=True, check_finite=True)
        solved = cho_solve(factor, h.T, check_finite=True)
    except np.linalg.LinAlgError as exc:
        raise ScreenInputError("WLS information matrix is singular.", "inadequate_observability") from exc
    omega = r - np.einsum("ij,ji->i", h, solved)
    # A large negative variance is an invalid numeric solve, not usable evidence.
    if np.any(omega < -1e-8 * r):
        raise ScreenInputError("Residual covariance is numerically unreliable.", "inadequate_observability")
    leverage = np.clip(1.0 - omega / r, 0.0, 1.0)
    return omega, leverage


def build_wls_features(case: Mapping[str, Any], z: Any, *, wls_details: Mapping[str, Any] | None = None,
                       measurement_sigma: Any = None, measurement_mask: Any = None,
                       max_it: int = 30, tol: float = 1e-8,
                       solver_settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    clean = configured_case(case)
    nb, nl = len(clean["bus"]), len(clean["branch"])
    try:
        observed = np.asarray(z, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ScreenInputError("z must contain only the complete numeric WLS measurement vector.") from exc
    count = 3 * nb + 4 * nl
    if observed.shape != (count,) or not np.all(np.isfinite(observed)):
        raise ScreenInputError("Complete finite z in Vm/Pinj/Qinj/Pf/Qf/Pt/Qt order is required.")
    if measurement_mask is not None:
        mask = np.asarray(measurement_mask)
        if mask.shape != observed.shape or not np.all(mask == 1):
            raise ScreenInputError("Partial measurement masks are unsupported by the shared WLS solver.")
    settings = dict(solver_settings or {})
    if set(settings) - {"max_it", "tol"}:
        raise ScreenInputError("Unsupported solver_settings; supported keys are max_it and tol.")
    max_it, tol = int(settings.get("max_it", max_it)), float(settings.get("tol", tol))
    if max_it < 1 or not np.isfinite(tol) or tol <= 0:
        raise ScreenInputError("Invalid WLS iteration settings.")
    sigma = (default_measurement_sigma(nb, nl) if measurement_sigma is None
             else np.asarray(measurement_sigma, dtype=np.float64))
    if sigma.shape != observed.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ScreenInputError("measurement_sigma must match z and be finite and positive.")
    if wls_details is None:
        # Reject an unobservable graph before the legacy sparse solver fails.
        _, initial_h = state_measurements_and_jacobian(clean, np.zeros(nb), np.ones(nb))
        residual_covariance(initial_h, sigma ** 2)
        try:
            details = lp.lagrangian_m_singlephase_details(observed, clean, 0, clean["bus"],
                        max_it=max_it, tol=tol, measurement_sigma=sigma)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            raise ScreenInputError(f"Balanced WLS failed: {exc}", "wls_failure") from exc
    else:
        details = dict(wls_details)
    if not details.get("success"):
        raise ScreenInputError("Balanced WLS did not converge.", "wls_failure")
    try:
        theta = np.asarray(details["theta_est_rad"], dtype=np.float64)
        vm = np.asarray(details["vm_est_pu"], dtype=np.float64)
        variance = np.asarray(details["measurement_variance_diag"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ScreenInputError("WLS details must expose the estimated state and actual solver covariance.") from exc
    if theta.shape != (nb,) or vm.shape != (nb,) or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(vm)) or np.any(vm <= 0):
        raise ScreenInputError("WLS estimated state is invalid.", "wls_failure")
    if variance.shape != observed.shape or (measurement_sigma is not None and not np.allclose(variance, sigma ** 2, rtol=1e-10, atol=0)):
        raise ScreenInputError("WLS covariance differs from the declared complete measurement covariance.")
    if details.get("covariance_kind", "diagonal") != "diagonal":
        raise ScreenInputError("Only diagonal measurement covariance is supported in version 1.")
    rows = np.asarray(details.get("measurement_rows", np.arange(count)))
    if not np.array_equal(rows, np.arange(count)):
        raise ScreenInputError("Partial/constrained WLS measurement rows are unsupported.")
    fitted, jacobian = state_measurements_and_jacobian(clean, theta, vm)
    residual = observed - fitted
    provided_residual = np.asarray(details.get("raw_residual", []), dtype=np.float64)
    if provided_residual.shape != residual.shape or not np.allclose(provided_residual, residual, rtol=1e-7, atol=1e-9):
        raise ScreenInputError("WLS details are not bound to this snapshot and configured model.")
    omega, leverage = residual_covariance(jacobian, variance)
    # Check supplied estimates actually describe a stationary WLS fit.
    step = cho_solve(cho_factor(jacobian.T @ (jacobian / variance[:, None])),
                     jacobian.T @ (residual / variance))
    if np.max(np.abs(step)) > max(5e-5, 5 * tol):
        raise ScreenInputError("WLS state has not converged for this snapshot.", "wls_failure")
    usable = omega > 1e-10 * variance
    signed = np.divide(residual, np.sqrt(np.maximum(omega, 0)),
                       out=np.zeros_like(residual), where=usable)
    objective = float(np.dot(residual, residual / variance))
    dof = count - jacobian.shape[1]
    return {"observed": observed.copy(), "fitted": fitted, "raw_residual": residual,
            "signed_normalized_residual": signed, "sigma": np.sqrt(variance),
            "variance": variance, "residual_covariance_diag": omega,
            "leverage": leverage, "available": np.ones(count, dtype=bool),
            "residual_usable": usable, "theta_est_rad": theta,
            "vm_est_pu": vm, "measurement_jacobian": jacobian,
            "wls_objective": objective, "n_measurements": count, "n_states": jacobian.shape[1],
            "dof": dof, "iterations": int(details.get("iterations", 0)),
            "solver_settings": dict(details.get("solver_settings", {}))}
