from __future__ import annotations

from typing import Any, Dict, Tuple
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --- MATPOWER / PYPOWER-style column indices (0-based for Python) ---
# bus
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
VM = 7
VA = 8
REF = 3

# branch
F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
TAP = 8
SHIFT = 9
BR_STATUS = 10

_MISSING = object()


def _get_field(obj: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(obj, dict):
        if default is _MISSING:
            return obj[name]
        return obj.get(name, default)
    if default is _MISSING:
        return getattr(obj, name)
    return getattr(obj, name, default)


def _copy_case_to_internal(result: Any, baseMVA_override: float | None = None) -> Dict[str, np.ndarray]:
    """Copy a MATPOWER-like case/result and renumber buses internally to 0..nb-1."""
    bus = np.array(_get_field(result, "bus"), dtype=float, copy=True)
    branch = np.array(_get_field(result, "branch"), dtype=float, copy=True)
    baseMVA_case = _get_field(result, "baseMVA", None)

    if baseMVA_override is not None:
        baseMVA = float(baseMVA_override)
    elif baseMVA_case is not None:
        baseMVA = float(baseMVA_case)
    else:
        raise ValueError("baseMVA must be provided either in the case or as the baseMVA argument.")

    ext_bus_ids = bus[:, BUS_I].astype(int)
    int_map = {ext_id: k for k, ext_id in enumerate(ext_bus_ids)}

    bus[:, BUS_I] = np.arange(bus.shape[0], dtype=float)
    branch[:, F_BUS] = np.vectorize(int_map.__getitem__)(branch[:, F_BUS].astype(int))
    branch[:, T_BUS] = np.vectorize(int_map.__getitem__)(branch[:, T_BUS].astype(int))

    return {"baseMVA": baseMVA, "bus": bus, "branch": branch}


def make_ybus(baseMVA: float, bus: np.ndarray, branch: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """MATPOWER/PYPOWER-style Ybus, Yf, Yt."""
    nb = bus.shape[0]
    nl = branch.shape[0]

    stat = branch[:, BR_STATUS]
    Ys = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])
    Bc = stat * branch[:, BR_B]

    tap = np.ones(nl, dtype=complex)
    nonzero_tap = np.flatnonzero(branch[:, TAP] != 0.0)
    tap[nonzero_tap] = branch[nonzero_tap, TAP]
    tap *= np.exp(1j * np.pi / 180.0 * branch[:, SHIFT])

    Ytt = Ys + 1j * Bc / 2.0
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -Ys / np.conj(tap)
    Ytf = -Ys / tap

    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    f = branch[:, F_BUS].astype(int)
    t = branch[:, T_BUS].astype(int)

    Cf = sp.csr_matrix((np.ones(nl), (np.arange(nl), f)), shape=(nl, nb))
    Ct = sp.csr_matrix((np.ones(nl), (np.arange(nl), t)), shape=(nl, nb))

    rows = np.r_[np.arange(nl), np.arange(nl)]
    cols = np.r_[f, t]
    Yf = sp.csr_matrix((np.r_[Yff, Yft], (rows, cols)), shape=(nl, nb))
    Yt = sp.csr_matrix((np.r_[Ytf, Ytt], (rows, cols)), shape=(nl, nb))
    Ybus = Cf.T @ Yf + Ct.T @ Yt + sp.diags(Ysh, offsets=0, shape=(nb, nb), format="csr")

    return Ybus.tocsr(), Yf.tocsr(), Yt.tocsr()


def dSbus_dV_polar(Ybus: sp.spmatrix, V: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Partial derivatives of injected complex power wrt voltage angle and magnitude.
    Returns (dS_dVa, dS_dVm).
    """
    Ibus = Ybus @ V
    Vmag = np.abs(V)
    Vnorm = np.divide(V, Vmag, out=np.ones_like(V), where=Vmag > 0)

    diagV = sp.diags(V, 0, format="csr")
    diagVcI = sp.diags(np.conj(Ibus), 0, format="csr")
    diagVnorm = sp.diags(Vnorm, 0, format="csr")

    dS_dVm = diagV @ (Ybus @ diagVnorm).conj() + diagVcI @ diagVnorm
    dS_dVa = 1j * diagV @ (sp.diags(Ibus, 0, format="csr") - Ybus @ diagV).conj()
    return dS_dVa.tocsr(), dS_dVm.tocsr()


def dSbr_dV1(
    Yf: sp.spmatrix,
    Yt: sp.spmatrix,
    V: np.ndarray,
    nb: int,
    nl: int,
    f: np.ndarray,
    t: np.ndarray,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Direct Python translation of the MATLAB helper dSbr_dV1().
    Returns:
        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St
    """
    Yfc = Yf.conj().tocsr()
    Ytc = Yt.conj().tocsr()
    Vc = np.conj(V)

    Ifc = Yfc @ Vc
    Itc = Ytc @ Vc

    diagVf = sp.diags(V[f], 0, shape=(nl, nl), format="csr")
    diagVt = sp.diags(V[t], 0, shape=(nl, nl), format="csr")
    diagIfc = sp.diags(Ifc, 0, shape=(nl, nl), format="csr")
    diagItc = sp.diags(Itc, 0, shape=(nl, nl), format="csr")

    Vmag = np.abs(V)
    Vnorm = np.divide(V, Vmag, out=np.ones_like(V), where=Vmag > 0)

    diagVc = sp.diags(Vc, 0, shape=(nb, nb), format="csr")
    diagVnorm = sp.diags(Vnorm, 0, shape=(nb, nb), format="csr")

    CVf = sp.csr_matrix((V[f], (np.arange(nl), f)), shape=(nl, nb))
    CVnf = sp.csr_matrix((Vnorm[f], (np.arange(nl), f)), shape=(nl, nb))
    CVt = sp.csr_matrix((V[t], (np.arange(nl), t)), shape=(nl, nb))
    CVnt = sp.csr_matrix((Vnorm[t], (np.arange(nl), t)), shape=(nl, nb))

    dSf_dVa = 1j * (diagIfc @ CVf - diagVf @ Yfc @ diagVc)
    dSf_dVm = diagVf @ (Yf @ diagVnorm).conj() + diagIfc @ CVnf
    dSt_dVa = 1j * (diagItc @ CVt - diagVt @ Ytc @ diagVc)
    dSt_dVm = diagVt @ (Yt @ diagVnorm).conj() + diagItc @ CVnt

    Sf = V[f] * Ifc
    St = V[t] * Itc
    return (
        dSf_dVa.tocsr(),
        dSf_dVm.tocsr(),
        dSt_dVa.tocsr(),
        dSt_dVm.tocsr(),
        np.asarray(Sf).reshape(-1),
        np.asarray(St).reshape(-1),
    )


def make_jaco(
    Ybus: sp.spmatrix,
    Yf: sp.spmatrix,
    Yt: sp.spmatrix,
    nb: int,
    nl: int,
    fbus: np.ndarray,
    tbus: np.ndarray,
    V: np.ndarray,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build the measurement Jacobian in MATLAB row order [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]."""
    dSbus_dVa, dSbus_dVm = dSbus_dV_polar(Ybus, V)
    dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St = dSbr_dV1(Yf, Yt, V, nb, nl, fbus, tbus)

    j11 = dSbus_dVa.real
    j12 = dSbus_dVm.real
    j21 = dSbus_dVa.imag
    j22 = dSbus_dVm.imag
    j31 = dSf_dVa.real
    j32 = dSf_dVm.real
    j41 = dSt_dVa.real
    j42 = dSt_dVm.real
    j51 = dSf_dVa.imag
    j52 = dSf_dVm.imag
    j61 = dSt_dVa.imag
    j62 = dSt_dVm.imag
    j71 = sp.csr_matrix((nb, nb))
    j72 = sp.eye(nb, format="csr")

    J = sp.vstack(
        [
            sp.hstack([j71, j72], format="csr"),
            sp.hstack([j11, j12], format="csr"),
            sp.hstack([j21, j22], format="csr"),
            sp.hstack([j31, j32], format="csr"),
            sp.hstack([j51, j52], format="csr"),
            sp.hstack([j41, j42], format="csr"),
            sp.hstack([j61, j62], format="csr"),
        ],
        format="csr",
    )
    return J, Sf, St


def _solve_sparse(A: sp.csc_matrix, b: np.ndarray) -> np.ndarray:
    lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A")
    return lu.solve(b)


def _rcond_dense(A: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    condA = np.linalg.cond(A)
    if not np.isfinite(condA) or condA == 0.0:
        return 0.0
    return 1.0 / condA


def _branch_supports_analytic_param_jacobian(branch_row: np.ndarray) -> bool:
    """
    Analytic helper below matches the supplied MATLAB formula for a standard line
    model with unit tap ratio and zero phase shift. MATPOWER encodes "no tap"
    as TAP = 0, which is equivalent to 1.0 in makeYbus().
    """
    tap = float(branch_row[TAP])
    shift = float(branch_row[SHIFT])
    tap_effective = 1.0 if abs(tap) < 1e-15 else tap
    return abs(tap_effective - 1.0) < 1e-12 and abs(shift) < 1e-12


def expand_state_vector(x_scan_reduced: np.ndarray, nb: int, ref_bus: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand one scan's reduced state [theta_without_ref; V] back to full size.

    Parameters
    ----------
    x_scan_reduced : array_like, shape (2*nb - 1,)
    nb : int
    ref_bus : int
        0-based reference bus index.
    """
    x_scan_reduced = np.asarray(x_scan_reduced, dtype=float).reshape(-1)
    if x_scan_reduced.shape[0] != 2 * nb - 1:
        raise ValueError(f"Expected reduced state length {2*nb - 1}, got {x_scan_reduced.shape[0]}.")

    theta_full = np.zeros(nb, dtype=float)
    theta_keep = np.r_[np.arange(ref_bus), np.arange(ref_bus + 1, nb)]
    theta_full[theta_keep] = x_scan_reduced[: nb - 1]
    V_full = x_scan_reduced[nb - 1 :]
    return theta_full, V_full


def calculate_hx(
    case_int: Dict[str, np.ndarray],
    theta_full: np.ndarray,
    V_full: np.ndarray,
    Ybus: sp.spmatrix | None = None,
    Yf: sp.spmatrix | None = None,
    Yt: sp.spmatrix | None = None,
) -> np.ndarray:
    """Compute predicted measurements in MATLAB order [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]."""
    bus = case_int["bus"]
    branch = case_int["branch"]
    baseMVA = float(case_int["baseMVA"])
    nb = bus.shape[0]
    nl = branch.shape[0]
    fbus = branch[:, F_BUS].astype(int)
    tbus = branch[:, T_BUS].astype(int)

    if Ybus is None or Yf is None or Yt is None:
        Ybus, Yf, Yt = make_ybus(baseMVA, bus, branch)

    Vc = np.asarray(V_full, dtype=float) * np.exp(1j * np.asarray(theta_full, dtype=float))
    Ibus = Ybus @ Vc
    Sinj = Vc * np.conj(Ibus)
    Sf = Vc[fbus] * np.conj(Yf @ Vc)
    St = Vc[tbus] * np.conj(Yt @ Vc)

    hx = np.r_[
        np.asarray(V_full, dtype=float),
        np.asarray(Sinj).real.reshape(-1),
        np.asarray(Sinj).imag.reshape(-1),
        np.asarray(Sf).real.reshape(-1),
        np.asarray(Sf).imag.reshape(-1),
        np.asarray(St).real.reshape(-1),
        np.asarray(St).imag.reshape(-1),
    ]
    expected_nz = 3 * nb + 4 * nl
    if hx.shape[0] != expected_nz:
        raise RuntimeError(f"Predicted measurement vector has length {hx.shape[0]}, expected {expected_nz}.")
    return hx


def calculate_param_jacobian_for_line(
    mpc_case: Dict[str, np.ndarray],
    line_idx: int,
    x_state_vec_scan: np.ndarray,
    Ybus: sp.spmatrix | None = None,
    Yf: sp.spmatrix | None = None,
    Yt: sp.spmatrix | None = None,
) -> np.ndarray:
    """
    Analytic two-column Jacobian d h / d [R_k, X_k] for one target line.

    This is the direct Python adaptation of the MATLAB helper supplied later in
    the conversation. It assumes the case has already been converted to internal
    consecutive bus numbering and that `line_idx` is 0-based.

    Notes
    -----
    - Ybus/Yf/Yt are accepted for signature compatibility with MATLAB but are not
      needed in the closed-form expression.
    - The closed-form expression assumes the standard series branch model without
      off-nominal tap ratio or phase shift. For such branches, use the finite-
      difference fallback instead.
    """
    del Ybus, Yf, Yt  # signature compatibility only

    bus = np.asarray(mpc_case["bus"], dtype=float)
    branch = np.asarray(mpc_case["branch"], dtype=float)
    nb = bus.shape[0]
    nl = branch.shape[0]

    x_state_vec_scan = np.asarray(x_state_vec_scan, dtype=float).reshape(-1)
    if x_state_vec_scan.shape[0] != 2 * nb:
        raise ValueError(f"x_state_vec_scan must have length {2 * nb}, got {x_state_vec_scan.shape[0]}.")
    if not (0 <= line_idx < nl):
        raise IndexError(f"line_idx={line_idx} is outside valid range [0, {nl-1}].")

    nz_actual = 3 * nb + 4 * nl
    H_pg_line = np.zeros((nz_actual, 2), dtype=float)

    if int(round(branch[line_idx, BR_STATUS])) == 0:
        return H_pg_line

    if not _branch_supports_analytic_param_jacobian(branch[line_idx, :]):
        raise ValueError(
            "Analytic parameter Jacobian is only valid here for unit tap / zero shift branches. "
            "Use the finite-difference fallback for transformer-like branches."
        )

    theta_est_rad = x_state_vec_scan[:nb]
    V_est_pu = x_state_vec_scan[nb: 2 * nb]

    idx_fbus = int(branch[line_idx, F_BUS])
    idx_tbus = int(branch[line_idx, T_BUS])

    r_k = float(branch[line_idx, BR_R])
    x_k = float(branch[line_idx, BR_X])

    denom = r_k**2 + x_k**2
    if denom < 1e-9:
        warnings.warn(
            (
                f"Line {line_idx + 1} parameters R={r_k:.4e}, X={x_k:.4e} lead to a near-zero "
                "denominator. Returning a zero parameter Jacobian for stability."
            ),
            RuntimeWarning,
        )
        return H_pg_line

    Vi = float(V_est_pu[idx_fbus])
    Vj = float(V_est_pu[idx_tbus])
    dth_rad = float(theta_est_rad[idx_fbus] - theta_est_rad[idx_tbus])
    cosd = np.cos(dth_rad)
    sind = np.sin(dth_rad)

    # g_k = R_k / (R_k^2 + X_k^2)
    # b_k = -X_k / (R_k^2 + X_k^2)
    dg_dr = (x_k**2 - r_k**2) / (denom**2)
    db_dr = (2.0 * r_k * x_k) / (denom**2)
    dg_dx = (-2.0 * r_k * x_k) / (denom**2)
    db_dx = (x_k**2 - r_k**2) / (denom**2)

    # Flow from i to j on line k
    dP_ik_dgk = Vi**2 - Vi * Vj * cosd
    dP_ik_dbk = -Vi * Vj * sind
    dQ_ik_dgk = -Vi * Vj * sind
    dQ_ik_dbk = -Vi**2 + Vi * Vj * cosd

    # Flow from j to i on line k
    dP_jk_dgk = Vj**2 - Vi * Vj * cosd
    dP_jk_dbk = Vi * Vj * sind
    dQ_jk_dgk = Vi * Vj * sind
    dQ_jk_dbk = -Vj**2 + Vi * Vj * cosd

    # Chain rule wrt [R_k, X_k]
    dPi_dr = dP_ik_dgk * dg_dr + dP_ik_dbk * db_dr
    dPi_dx = dP_ik_dgk * dg_dx + dP_ik_dbk * db_dx
    dQi_dr = dQ_ik_dgk * dg_dr + dQ_ik_dbk * db_dr
    dQi_dx = dQ_ik_dgk * dg_dx + dQ_ik_dbk * db_dx

    dPj_dr = dP_jk_dgk * dg_dr + dP_jk_dbk * db_dr
    dPj_dx = dP_jk_dgk * dg_dx + dP_jk_dbk * db_dx
    dQj_dr = dQ_jk_dgk * dg_dr + dQ_jk_dbk * db_dr
    dQj_dx = dQ_jk_dgk * dg_dx + dQ_jk_dbk * db_dx

    # Measurement order: Vm (nb), Pinj (nb), Qinj (nb), Pf (nl), Qf (nl), Pt (nl), Qt (nl)
    # Vm rows remain zero.

    # P injections
    H_pg_line[nb + idx_fbus, 0] = dPi_dr
    H_pg_line[nb + idx_fbus, 1] = dPi_dx
    H_pg_line[nb + idx_tbus, 0] = dPj_dr
    H_pg_line[nb + idx_tbus, 1] = dPj_dx

    # Q injections
    H_pg_line[2 * nb + idx_fbus, 0] = dQi_dr
    H_pg_line[2 * nb + idx_fbus, 1] = dQi_dx
    H_pg_line[2 * nb + idx_tbus, 0] = dQj_dr
    H_pg_line[2 * nb + idx_tbus, 1] = dQj_dx

    # Line flows for this line only
    offset_pf = 3 * nb
    offset_qf = 3 * nb + nl
    offset_pt = 3 * nb + 2 * nl
    offset_qt = 3 * nb + 3 * nl

    H_pg_line[offset_pf + line_idx, 0] = dPi_dr
    H_pg_line[offset_pf + line_idx, 1] = dPi_dx
    H_pg_line[offset_qf + line_idx, 0] = dQi_dr
    H_pg_line[offset_qf + line_idx, 1] = dQi_dx
    H_pg_line[offset_pt + line_idx, 0] = dPj_dr
    H_pg_line[offset_pt + line_idx, 1] = dPj_dx
    H_pg_line[offset_qt + line_idx, 0] = dQj_dr
    H_pg_line[offset_qt + line_idx, 1] = dQj_dx

    return H_pg_line


def calculate_param_jacobian_for_line_fd(
    case_int: Dict[str, np.ndarray],
    line_idx: int,
    theta_full: np.ndarray,
    V_full: np.ndarray,
    *,
    fd_rel_step: float = 1e-6,
    min_param_value: float = 1e-9,
) -> np.ndarray:
    """
    Numerical two-column Jacobian of the measurement model wrt a single line's [R, X].

    This remains available as a robust fallback for branches with off-nominal
    taps or phase shifts, where the supplied analytic MATLAB expression does not
    directly match the full MATPOWER branch model.
    """
    base_case = {
        "baseMVA": float(case_int["baseMVA"]),
        "bus": np.array(case_int["bus"], copy=True),
        "branch": np.array(case_int["branch"], copy=True),
    }

    branch0 = base_case["branch"]
    p0 = np.array([branch0[line_idx, BR_R], branch0[line_idx, BR_X]], dtype=float)

    nz = 3 * base_case["bus"].shape[0] + 4 * base_case["branch"].shape[0]
    H_p = np.zeros((nz, 2), dtype=float)

    for col, br_col in enumerate((BR_R, BR_X)):
        step = fd_rel_step * max(abs(p0[col]), 1e-3)
        p_plus = p0[col] + step
        p_minus = max(p0[col] - step, min_param_value)

        case_plus = {
            "baseMVA": float(base_case["baseMVA"]),
            "bus": np.array(base_case["bus"], copy=True),
            "branch": np.array(base_case["branch"], copy=True),
        }
        case_minus = {
            "baseMVA": float(base_case["baseMVA"]),
            "bus": np.array(base_case["bus"], copy=True),
            "branch": np.array(base_case["branch"], copy=True),
        }

        case_plus["branch"][line_idx, br_col] = p_plus
        case_minus["branch"][line_idx, br_col] = p_minus

        Ybus_p, Yf_p, Yt_p = make_ybus(case_plus["baseMVA"], case_plus["bus"], case_plus["branch"])
        Ybus_m, Yf_m, Yt_m = make_ybus(case_minus["baseMVA"], case_minus["bus"], case_minus["branch"])

        hx_plus = calculate_hx(case_plus, theta_full, V_full, Ybus_p, Yf_p, Yt_p)
        hx_minus = calculate_hx(case_minus, theta_full, V_full, Ybus_m, Yf_m, Yt_m)

        denom = p_plus - p_minus
        if denom <= 0.0:
            raise RuntimeError("Finite-difference denominator for parameter Jacobian is non-positive.")
        H_p[:, col] = (hx_plus - hx_minus) / denom

    return H_p


def correct_parameter_group_multi_scan(
    mpc_with_error: Any,
    line_to_correct_idx: int,
    multi_scan_measurements_z: np.ndarray,
    initial_states_multi_scan: np.ndarray,
    R_variances_vec: np.ndarray,
    baseMVA: float | None,
    *,
    line_index_is_one_based: bool = True,
    max_iter_corr: int = 60,
    tol_corr: float = 5e-3,
    damping: float = 0.5,
    min_param_value: float = 1e-6,
    fd_rel_step: float = 1e-6,
    param_jacobian_method: str = "auto",
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Python translation of the MATLAB function `correct_parameter_group_multi_scan()`.

    Parameters
    ----------
    mpc_with_error : dict or object
        MATPOWER-like case with bus/branch/baseMVA.
    line_to_correct_idx : int
        Line index identifying the branch row to correct. By default this is
        interpreted as MATLAB-style 1-based indexing.
    multi_scan_measurements_z : ndarray, shape (nz_single, s)
        Measurement matrix whose k-th column is the full measurement vector for
        scan k in MATLAB order [Vm, Pinj, Qinj, Pf, Qf, Pt, Qt].
    initial_states_multi_scan : ndarray, shape (2*nb, s)
        Initial state guesses per scan, with the first nb rows equal to voltage
        magnitudes and the next nb rows equal to voltage angles in degrees.
    R_variances_vec : ndarray, shape (nz_single,)
        Per-measurement variances for a single scan.
    baseMVA : float or None
        Optional baseMVA override. If None, the value in the case is used.

    Keyword Parameters
    ------------------
    line_index_is_one_based : bool, default True
        Preserve MATLAB semantics for the branch row index.
    max_iter_corr : int, default 60
    tol_corr : float, default 5e-3
    damping : float, default 0.5
        Matches the MATLAB update `x_v_current = x_v_current + 0.5 * delta_x_v`.
    min_param_value : float, default 1e-6
        Positivity floor for corrected R and X.
    fd_rel_step : float, default 1e-6
        Relative perturbation size used by the finite-difference fallback.
    param_jacobian_method : {"auto", "analytic", "fd"}, default "auto"
        - "analytic": always use the closed-form Jacobian supplied later.
        - "fd": always use finite differences.
        - "auto": use the analytic Jacobian for standard unit-tap, zero-shift
          lines and fall back to finite differences otherwise.
    verbose : bool, default False
        If True, prints iteration diagnostics similar to the MATLAB script.

    Returns
    -------
    corrected_params_group, success_correction
        corrected_params_group is a length-2 array [R_est, X_est].
        success_correction is 1 if converged, else 0.
    """
    method = str(param_jacobian_method).lower()
    if method not in {"auto", "analytic", "fd"}:
        raise ValueError("param_jacobian_method must be one of {'auto', 'analytic', 'fd'}." )

    ppc = _copy_case_to_internal(mpc_with_error, baseMVA_override=baseMVA)
    bus = ppc["bus"]
    branch = ppc["branch"]
    baseMVA_use = float(ppc["baseMVA"])

    nb = bus.shape[0]
    nl = branch.shape[0]

    z_all = np.asarray(multi_scan_measurements_z, dtype=float)
    if z_all.ndim != 2:
        raise ValueError("multi_scan_measurements_z must be a 2-D array of shape (nz_single, n_scans).")
    nz_single, n_scans = z_all.shape

    x0_all = np.asarray(initial_states_multi_scan, dtype=float)
    if x0_all.shape != (2 * nb, n_scans):
        raise ValueError(
            f"initial_states_multi_scan must have shape {(2 * nb, n_scans)}, got {x0_all.shape}."
        )

    R_variances_vec = np.asarray(R_variances_vec, dtype=float).reshape(-1)
    expected_nz = 3 * nb + 4 * nl
    if nz_single != expected_nz:
        raise ValueError(
            f"Each scan must contain {expected_nz} measurements in full MATLAB order; got {nz_single}."
        )
    if R_variances_vec.shape[0] != nz_single:
        raise ValueError("R_variances_vec length must equal the number of measurements per scan.")
    if np.any(~np.isfinite(z_all)):
        raise ValueError("multi_scan_measurements_z must be finite; this port assumes full per-scan measurements.")
    if np.any(~np.isfinite(R_variances_vec)) or np.any(R_variances_vec <= 0.0):
        raise ValueError("R_variances_vec must contain finite, strictly positive variances.")

    line_idx = int(line_to_correct_idx)
    if line_index_is_one_based:
        line_idx -= 1
    if not (0 <= line_idx < nl):
        raise IndexError(f"line_to_correct_idx resolves to {line_idx}, outside valid range [0, {nl-1}].")

    success_correction = 0

    p_g_current = np.array(
        [branch[line_idx, BR_R], branch[line_idx, BR_X]],
        dtype=float,
    )
    corrected_params_group = p_g_current.copy()

    R_inv_single = sp.diags(1.0 / R_variances_vec, 0, shape=(nz_single, nz_single), format="csc")

    ref_candidates = np.flatnonzero(bus[:, BUS_TYPE].astype(int) == REF)
    if ref_candidates.size == 0:
        ref_bus = 0
        warnings.warn("No type-3 (slack) bus found; defaulting ref_bus=1.", RuntimeWarning)
    else:
        ref_bus = int(ref_candidates[0])
        if ref_candidates.size > 1:
            warnings.warn(
                f"Multiple type-3 buses found ({ref_candidates.size}); using the first one as reference.",
                RuntimeWarning,
            )

    n_states_per_scan = 2 * nb - 1
    N = n_scans * n_states_per_scan + 2

    x_states_all_scans_current = np.zeros(n_scans * n_states_per_scan, dtype=float)
    for k_scan in range(n_scans):
        v_init = x0_all[:nb, k_scan]
        a_init_deg = x0_all[nb: 2 * nb, k_scan]
        a_init_rad_full = np.deg2rad(a_init_deg)

        a_sub = np.delete(a_init_rad_full, ref_bus)
        x_k_scan = np.r_[a_sub, v_init]

        idx_start = k_scan * n_states_per_scan
        idx_end = (k_scan + 1) * n_states_per_scan
        x_states_all_scans_current[idx_start:idx_end] = x_k_scan

    x_v_current = np.r_[x_states_all_scans_current, p_g_current]

    if verbose:
        print(f"    Entering correct_parameter_group_multi_scan for line {line_to_correct_idx}...")
        print(f"    Starting iterative correction for line {line_to_correct_idx} with {n_scans} scans...")

    for iter_idx in range(1, max_iter_corr + 1):
        G_v = sp.lil_matrix((N, N), dtype=float)
        RHS = np.zeros(N, dtype=float)

        mpc_iter = {
            "baseMVA": baseMVA_use,
            "bus": np.array(bus, copy=True),
            "branch": np.array(branch, copy=True),
        }
        mpc_iter["branch"][line_idx, BR_R] = x_v_current[n_scans * n_states_per_scan + 0]
        mpc_iter["branch"][line_idx, BR_X] = x_v_current[n_scans * n_states_per_scan + 1]

        Ybus_iter, Yf_iter, Yt_iter = make_ybus(baseMVA_use, mpc_iter["bus"], mpc_iter["branch"])
        fbus_iter = mpc_iter["branch"][:, F_BUS].astype(int)
        tbus_iter = mpc_iter["branch"][:, T_BUS].astype(int)

        analytic_supported = _branch_supports_analytic_param_jacobian(mpc_iter["branch"][line_idx, :])
        if method == "analytic" and not analytic_supported:
            raise ValueError(
                "param_jacobian_method='analytic' was requested, but the target line has a non-unity tap or "
                "nonzero phase shift. Use 'auto' or 'fd' instead."
            )

        for k_scan in range(n_scans):
            idx_start = k_scan * n_states_per_scan
            idx_end = (k_scan + 1) * n_states_per_scan
            x_scan_k = x_v_current[idx_start:idx_end]

            theta_full, V_full = expand_state_vector(x_scan_k, nb, ref_bus)

            hx_k = calculate_hx(mpc_iter, theta_full, V_full, Ybus_iter, Yf_iter, Yt_iter)
            z_k = z_all[:, k_scan]
            delta_z_k = z_k - hx_k

            Vc_full = V_full * np.exp(1j * theta_full)
            J_full, _, _ = make_jaco(Ybus_iter, Yf_iter, Yt_iter, nb, nl, fbus_iter, tbus_iter, Vc_full)
            keep_cols = np.r_[np.arange(ref_bus), np.arange(ref_bus + 1, 2 * nb)]
            H_x_k = J_full[:, keep_cols].tocsc()

            if method == "fd" or (method == "auto" and not analytic_supported):
                H_p_k = calculate_param_jacobian_for_line_fd(
                    mpc_iter,
                    line_idx,
                    theta_full,
                    V_full,
                    fd_rel_step=fd_rel_step,
                    min_param_value=min_param_value,
                )
            else:
                H_p_k = calculate_param_jacobian_for_line(
                    mpc_iter,
                    line_idx,
                    np.r_[theta_full, V_full],
                    Ybus_iter,
                    Yf_iter,
                    Yt_iter,
                )

            Ht_W = H_x_k.T @ R_inv_single
            Hp_t_W = H_p_k.T @ R_inv_single

            Ht_W_dz = np.asarray(Ht_W @ delta_z_k).reshape(-1)
            Hp_t_W_dz = np.asarray(Hp_t_W @ delta_z_k).reshape(-1)

            row_s = slice(idx_start, idx_end)
            col_s = row_s
            row_p = slice(n_scans * n_states_per_scan, n_scans * n_states_per_scan + 2)
            col_p = row_p

            G_v[row_s, col_s] = G_v[row_s, col_s] + (Ht_W @ H_x_k)
            G_v[row_s, col_p] = G_v[row_s, col_p] + (Ht_W @ H_p_k)
            G_v[row_p, col_s] = G_v[row_p, col_s] + (Hp_t_W @ H_x_k)
            G_v[row_p, col_p] = G_v[row_p, col_p] + (Hp_t_W @ H_p_k)

            RHS[row_s] += Ht_W_dz
            RHS[row_p] += Hp_t_W_dz

        G_v = G_v.tocsc()
        rcond_Gv = _rcond_dense(G_v.toarray())
        if rcond_Gv < 1e-14:
            if verbose:
                print(f"    G_v is ill-conditioned at iteration {iter_idx}. Aborting correction.")
            success_correction = 0
            return corrected_params_group, success_correction

        x_prev = x_v_current.copy()
        delta_x_v = _solve_sparse(G_v, RHS)
        x_v_current = x_v_current + damping * delta_x_v

        idx_param = np.arange(n_scans * n_states_per_scan, n_scans * n_states_per_scan + 2)
        p_temp = x_v_current[idx_param]
        p_temp = np.maximum(p_temp, min_param_value)
        x_v_current[idx_param] = p_temp

        max_update = float(np.max(np.abs(x_v_current - x_prev)))
        p_g_current = x_v_current[idx_param].copy()
        corrected_params_group = p_g_current.copy()

        if verbose:
            print(
                f"    Iter {iter_idx}: max update={max_update:.2e}, "
                f"R_est={p_g_current[0]:.6g}, X_est={p_g_current[1]:.6g}"
            )

        if max_update < tol_corr:
            success_correction = 1
            if verbose:
                print(f"    Correction converged in {iter_idx} iterations.")
                print(f"    Exiting correct_parameter_group_multi_scan. success={success_correction}")
            return corrected_params_group, success_correction

    success_correction = 0
    if verbose:
        print(f"    Correction did NOT converge by iteration {max_iter_corr}. Returning last estimate.")
        print(f"    Exiting correct_parameter_group_multi_scan. success={success_correction}")
    return corrected_params_group, success_correction


__all__ = [
    "correct_parameter_group_multi_scan",
    "calculate_param_jacobian_for_line",
    "calculate_param_jacobian_for_line_fd",
    "expand_state_vector",
    "calculate_hx",
    "make_ybus",
    "make_jaco",
    "dSbus_dV_polar",
    "dSbr_dV1",
]
