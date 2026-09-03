"""Shared branch-parameter Jacobian ``d h(x) / d [R_k, X_k]``.

Both the normalized-Lagrange-multiplier (NLM) parameter localizer in
``lagrangian_port`` and the multi-scan parameter corrector in
``correct_parameter_group_multi_scan_port`` need the sensitivity of the WLS
measurement model to one branch's series resistance and reactance.  Until
2026-09-03 the two modules carried separate hand-written copies and the NLM
copy had the wrong sign on the ``dP/db`` and ``dQ/dg`` terms and ignored
branch status, tap ratio, and phase shift.  This module is now the single
implementation; a finite-difference cross-check lives in
``tools/test_numerical_foundations.py``.

The measurement model is the MATPOWER/PYPOWER branch model used by
``make_ybus`` in the ports::

    Ys  = status / (R + jX)
    Yff = (Ys + j Bc/2) / |t|^2      Yft = -Ys / conj(t)
    Ytf = -Ys / t                    Ytt =  Ys + j Bc/2
    t   = tap * exp(j * shift)

with the measurement vector in MATLAB order
``[Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)]``.

The closed form is derived in complex notation.  With ``Ys = g + jb``::

    Sf = Vi * conj(Yff Vi + Yft Vj)
       -> dSf/dg = |Vi|^2/|t|^2 - Vi conj(Vj)/t
          dSf/db = -j |Vi|^2/|t|^2 + j Vi conj(Vj)/t
    St = Vj * conj(Ytf Vi + Ytt Vj)
       -> dSt/dg = |Vj|^2 - Vj conj(Vi)/conj(t)
          dSt/db = -j |Vj|^2 + j Vj conj(Vi)/conj(t)

Bus injections at the two terminals inherit the same derivatives, the line
charging ``Bc`` drops out, and the chain rule to ``(R, X)`` uses
``g = R/(R^2+X^2)``, ``b = -X/(R^2+X^2)``.  For a unit-tap, zero-shift line
this reduces to the textbook expressions ``dP_ij/db = -Vi Vj sin(d)`` and
``dQ_ij/dg = -Vi Vj sin(d)`` with ``d = theta_i - theta_j``.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping

import numpy as np

# MATPOWER / PYPOWER branch columns (0-based)
F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
TAP = 8
SHIFT = 9
BR_STATUS = 10

__all__ = [
    "branch_status",
    "branch_rx_jacobian",
    "branch_rx_jacobian_fd",
    "all_branch_rx_jacobian",
    "measurement_count",
]


def measurement_count(nb: int, nl: int) -> int:
    return 3 * int(nb) + 4 * int(nl)


def branch_status(branch_row: np.ndarray) -> float:
    """Branch status multiplier used by ``make_ybus`` (1.0 when the column is absent)."""
    row = np.asarray(branch_row, dtype=float).reshape(-1)
    if row.shape[0] <= BR_STATUS:
        return 1.0
    value = float(row[BR_STATUS])
    if not np.isfinite(value):
        return 0.0
    return value


def _complex_tap(branch_row: np.ndarray) -> complex:
    tap = float(branch_row[TAP])
    if abs(tap) < 1e-15:
        tap = 1.0
    return tap * np.exp(1j * np.pi / 180.0 * float(branch_row[SHIFT]))


def branch_rx_jacobian(
    case_int: Mapping[str, Any],
    line_idx: int,
    theta_full: np.ndarray,
    V_full: np.ndarray,
    *,
    method: str = "analytic",
    fd_rel_step: float = 1e-6,
    fd_abs_step: float = 1e-7,
) -> np.ndarray:
    """Return the ``(nz, 2)`` Jacobian of ``h(x)`` wrt ``[R_k, X_k]`` of branch ``line_idx``.

    Parameters
    ----------
    case_int
        Case with internal, consecutive 0-based bus numbering (``bus``,
        ``branch``, ``baseMVA``), as produced by the ports' ``_copy_*_to_internal``.
    line_idx
        0-based branch row.
    theta_full, V_full
        Full bus angle (rad) and magnitude (p.u.) vectors, length ``nb``.
    method
        ``"analytic"`` (default, exact for every MATPOWER branch type),
        ``"fd"`` (central finite differences on the full measurement model), or
        ``"auto"`` (alias of ``"analytic"``, kept for the corrector's API).

    An out-of-service branch (``BR_STATUS == 0``) has no influence on ``h`` and
    returns an all-zero matrix, matching the admittance model exactly.
    """
    method = str(method).lower()
    if method not in {"analytic", "fd", "auto"}:
        raise ValueError("method must be one of {'analytic', 'fd', 'auto'}")
    if method == "fd":
        return branch_rx_jacobian_fd(
            case_int, line_idx, theta_full, V_full, fd_rel_step=fd_rel_step, fd_abs_step=fd_abs_step
        )

    bus = np.asarray(case_int["bus"], dtype=float)
    branch = np.asarray(case_int["branch"], dtype=float)
    nb = bus.shape[0]
    nl = branch.shape[0]
    if not 0 <= int(line_idx) < nl:
        raise IndexError(f"line_idx={line_idx} outside valid range [0, {nl - 1}].")
    theta_full = np.asarray(theta_full, dtype=float).reshape(-1)
    V_full = np.asarray(V_full, dtype=float).reshape(-1)
    if theta_full.shape[0] != nb or V_full.shape[0] != nb:
        raise ValueError(f"theta_full and V_full must have length nb={nb}.")

    nz = measurement_count(nb, nl)
    H = np.zeros((nz, 2), dtype=float)

    row = branch[int(line_idx)]
    stat = branch_status(row)
    if stat == 0.0:
        return H

    r_k = float(row[BR_R])
    x_k = float(row[BR_X])
    denom = r_k**2 + x_k**2
    if denom < 1e-18:
        warnings.warn(
            f"Branch {int(line_idx) + 1} has R={r_k:.3e}, X={x_k:.3e}; parameter Jacobian is undefined "
            "and is returned as zero.",
            RuntimeWarning,
        )
        return H

    # g = R / (R^2 + X^2), b = -X / (R^2 + X^2)
    dg_dr = (x_k**2 - r_k**2) / denom**2
    db_dr = (2.0 * r_k * x_k) / denom**2
    dg_dx = (-2.0 * r_k * x_k) / denom**2
    db_dx = (x_k**2 - r_k**2) / denom**2

    i = int(row[F_BUS])
    j = int(row[T_BUS])
    t = _complex_tap(row)
    t_abs2 = float(np.real(t * np.conj(t)))
    Vi = V_full[i] * np.exp(1j * theta_full[i])
    Vj = V_full[j] * np.exp(1j * theta_full[j])
    Vi2 = float(V_full[i] ** 2)
    Vj2 = float(V_full[j] ** 2)

    # Derivatives of the complex terminal flows wrt series conductance g and
    # susceptance b (see module docstring).  ``stat`` scales Ys in make_ybus.
    dSf_dg = stat * (Vi2 / t_abs2 - Vi * np.conj(Vj) / t)
    dSf_db = stat * (-1j * Vi2 / t_abs2 + 1j * Vi * np.conj(Vj) / t)
    dSt_dg = stat * (Vj2 - Vj * np.conj(Vi) / np.conj(t))
    dSt_db = stat * (-1j * Vj2 + 1j * Vj * np.conj(Vi) / np.conj(t))

    dSf_dr = dSf_dg * dg_dr + dSf_db * db_dr
    dSf_dx = dSf_dg * dg_dx + dSf_db * db_dx
    dSt_dr = dSt_dg * dg_dr + dSt_db * db_dr
    dSt_dx = dSt_dg * dg_dx + dSt_db * db_dx

    # Injections at both terminals see the same flow derivatives.
    H[nb + i, 0] += dSf_dr.real
    H[nb + i, 1] += dSf_dx.real
    H[nb + j, 0] += dSt_dr.real
    H[nb + j, 1] += dSt_dx.real
    H[2 * nb + i, 0] += dSf_dr.imag
    H[2 * nb + i, 1] += dSf_dx.imag
    H[2 * nb + j, 0] += dSt_dr.imag
    H[2 * nb + j, 1] += dSt_dx.imag

    k = int(line_idx)
    H[3 * nb + k, 0] = dSf_dr.real
    H[3 * nb + k, 1] = dSf_dx.real
    H[3 * nb + nl + k, 0] = dSf_dr.imag
    H[3 * nb + nl + k, 1] = dSf_dx.imag
    H[3 * nb + 2 * nl + k, 0] = dSt_dr.real
    H[3 * nb + 2 * nl + k, 1] = dSt_dx.real
    H[3 * nb + 3 * nl + k, 0] = dSt_dr.imag
    H[3 * nb + 3 * nl + k, 1] = dSt_dx.imag
    return H


def _hx_internal(case_int: Mapping[str, Any], theta_full: np.ndarray, V_full: np.ndarray) -> np.ndarray:
    """Measurement function in MATLAB order using the ports' admittance model."""
    # Imported lazily to avoid a circular import at module load time.
    from tools.correct_parameter_group_multi_scan_port import calculate_hx, make_ybus

    bus = np.asarray(case_int["bus"], dtype=float)
    branch = np.asarray(case_int["branch"], dtype=float)
    base = float(case_int["baseMVA"])
    Ybus, Yf, Yt = make_ybus(base, bus, branch)
    return calculate_hx({"baseMVA": base, "bus": bus, "branch": branch}, theta_full, V_full, Ybus, Yf, Yt)


def branch_rx_jacobian_fd(
    case_int: Mapping[str, Any],
    line_idx: int,
    theta_full: np.ndarray,
    V_full: np.ndarray,
    *,
    fd_rel_step: float = 1e-6,
    fd_abs_step: float = 1e-7,
) -> np.ndarray:
    """Central finite-difference Jacobian of ``h`` wrt ``[R_k, X_k]``.

    The series admittance is a smooth rational function of ``(R, X)`` wherever
    ``R^2 + X^2 > 0``, so the central stencil straddles ``R = 0`` without any
    positivity clamp.  (The earlier corrector-local helper clamped the lower
    sample to a positivity floor, which produced a zero or negative stencil
    width for zero-resistance transformers and raised.)
    """
    bus = np.asarray(case_int["bus"], dtype=float)
    branch = np.asarray(case_int["branch"], dtype=float)
    nb = bus.shape[0]
    nl = branch.shape[0]
    if not 0 <= int(line_idx) < nl:
        raise IndexError(f"line_idx={line_idx} outside valid range [0, {nl - 1}].")
    nz = measurement_count(nb, nl)
    H = np.zeros((nz, 2), dtype=float)
    base_case = {"baseMVA": float(case_int["baseMVA"]), "bus": bus, "branch": branch}
    p0 = np.array([branch[line_idx, BR_R], branch[line_idx, BR_X]], dtype=float)
    for col, br_col in enumerate((BR_R, BR_X)):
        step = max(float(fd_rel_step) * abs(p0[col]), float(fd_abs_step))
        plus = np.array(branch, copy=True)
        minus = np.array(branch, copy=True)
        plus[line_idx, br_col] = p0[col] + step
        minus[line_idx, br_col] = p0[col] - step
        other = p0[1 - col]
        if (p0[col] - step) ** 2 + other**2 < 1e-18:
            # Degenerate stencil (both parameters ~0): fall back to a forward difference.
            minus[line_idx, br_col] = p0[col]
            width = step
        else:
            width = 2.0 * step
        h_plus = _hx_internal({**base_case, "branch": plus}, theta_full, V_full)
        h_minus = _hx_internal({**base_case, "branch": minus}, theta_full, V_full)
        H[:, col] = (h_plus - h_minus) / width
    return H


def all_branch_rx_jacobian(
    case_int: Mapping[str, Any],
    theta_full: np.ndarray,
    V_full: np.ndarray,
    *,
    method: str = "analytic",
) -> np.ndarray:
    """Return the ``(nz, 2*nl)`` Jacobian with columns ``[R_0, X_0, R_1, X_1, ...]``."""
    branch = np.asarray(case_int["branch"], dtype=float)
    bus = np.asarray(case_int["bus"], dtype=float)
    nl = branch.shape[0]
    nz = measurement_count(bus.shape[0], nl)
    Hp = np.zeros((nz, 2 * nl), dtype=float)
    for k in range(nl):
        Hp[:, 2 * k : 2 * k + 2] = branch_rx_jacobian(case_int, k, theta_full, V_full, method=method)
    return Hp
