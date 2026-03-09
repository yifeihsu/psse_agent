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

# gen
GEN_BUS = 0
PG = 1
QG = 2
GEN_STATUS = 7

_MISSING = object()


def _get_field(obj: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(obj, dict):
        if default is _MISSING:
            return obj[name]
        return obj.get(name, default)
    if default is _MISSING:
        return getattr(obj, name)
    return getattr(obj, name, default)


def _get_opt(options: Any, name: str, default: Any) -> Any:
    if options is None:
        return default
    if isinstance(options, dict):
        return options.get(name, default)
    return getattr(options, name, default)


def _copy_result_to_internal(result: Any) -> Dict[str, np.ndarray]:
    """Copy a MATPOWER-like result struct/dict and renumber buses internally to 0..nb-1."""
    bus = np.array(_get_field(result, "bus"), dtype=float, copy=True)
    branch = np.array(_get_field(result, "branch"), dtype=float, copy=True)
    gen = np.array(_get_field(result, "gen"), dtype=float, copy=True)
    baseMVA = float(_get_field(result, "baseMVA"))

    ext_bus_ids = bus[:, BUS_I].astype(int)
    int_map = {ext_id: k for k, ext_id in enumerate(ext_bus_ids)}

    bus[:, BUS_I] = np.arange(bus.shape[0], dtype=float)
    branch[:, F_BUS] = np.vectorize(int_map.__getitem__)(branch[:, F_BUS].astype(int))
    branch[:, T_BUS] = np.vectorize(int_map.__getitem__)(branch[:, T_BUS].astype(int))
    gen[:, GEN_BUS] = np.vectorize(int_map.__getitem__)(gen[:, GEN_BUS].astype(int))

    return {"baseMVA": baseMVA, "bus": bus, "branch": branch, "gen": gen}


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

    dSf_dV1 = 1j * (diagIfc @ CVf - diagVf @ Yfc @ diagVc)
    dSf_dV2 = diagVf @ (Yf @ diagVnorm).conj() + diagIfc @ CVnf
    dSt_dV1 = 1j * (diagItc @ CVt - diagVt @ Ytc @ diagVc)
    dSt_dV2 = diagVt @ (Yt @ diagVnorm).conj() + diagItc @ CVnt

    Sf = V[f] * Ifc
    St = V[t] * Itc
    return (
        dSf_dV1.tocsr(),
        dSf_dV2.tocsr(),
        dSt_dV1.tocsr(),
        dSt_dV2.tocsr(),
        np.asarray(Sf).reshape(-1),
        np.asarray(St).reshape(-1),
    )


def _rcond_dense(A: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    condA = np.linalg.cond(A)
    if not np.isfinite(condA) or condA == 0:
        return 0.0
    return 1.0 / condA


def _normalize_group_indices(indices: Any, n_full: int, *, one_based: bool) -> np.ndarray:
    if indices is None:
        return np.array([], dtype=int)
    arr = np.asarray(indices, dtype=int).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=int)
    if one_based:
        arr = arr - 1
    arr = arr[(arr >= 0) & (arr < n_full)]
    return np.unique(arr)


def _solve_gain(Gain: sp.csc_matrix, B: np.ndarray) -> np.ndarray:
    """Solve Gain * X = B using sparse LU with symmetric-min-degree-style ordering."""
    lu = spla.splu(Gain, permc_spec="MMD_AT_PLUS_A")
    return lu.solve(B)


def lagrangian_m_correct(
    z_in_full: np.ndarray,
    result: Any,
    ind: int,
    bus_data: np.ndarray,
    options: Any,
    R_variances_full_in: np.ndarray,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Python translation of the MATLAB function LagrangianM_correct().

    Parameters
    ----------
    z_in_full : array_like
        Full measurement vector in MATLAB ordering:
        [Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)].
        Invalid / unavailable entries should be NaN or outside (-998, 998),
        matching the MATLAB script.
    result : dict or object
        MATPOWER-like case/result with fields/keys: bus, branch, gen, baseMVA.
    ind : int
        Preserved from the MATLAB signature. It is not used by this script.
    bus_data : ndarray
        External bus data used to initialize angle (deg, column 9 in MATLAB) and
        Vm (column 8 in MATLAB).
    options : dict or object
        State-estimation options. Recognized fields:
          - enable_group_correction : bool
          - correction_group_full_indices : sequence of measurement indices
          - max_correction_iterations : int
          - correction_error_tolerance : float
          - group_indices_are_one_based : bool, default True to preserve MATLAB semantics
    R_variances_full_in : array_like
        Full vector of measurement variances in the same ordering as z_in_full.

    Returns
    -------
    lambdaN, success, r_norm, Omega, final_resid_raw, z_corrected_info

    Notes
    -----
    This port preserves several MATLAB behaviors intentionally:
      * invalid measurements are screened with the same sentinel logic;
      * each new correction cycle re-starts WLS from bus_data;
      * lambdaN is returned as zeros(nl), because the provided MATLAB script
        overwrites it that way at the end.
    """
    del ind  # unused in this MATLAB script

    z_in_full = np.asarray(z_in_full, dtype=float).reshape(-1)
    R_variances_full_in = np.asarray(R_variances_full_in, dtype=float).reshape(-1)
    bus_data = np.asarray(bus_data, dtype=float)

    if z_in_full.shape[0] != R_variances_full_in.shape[0]:
        raise ValueError("z_in_full and R_variances_full_in must have the same length.")

    ppc = _copy_result_to_internal(result)
    bus = ppc["bus"]
    branch = ppc["branch"]
    baseMVA = ppc["baseMVA"]

    ref_candidates = np.flatnonzero(bus[:, BUS_TYPE].astype(int) == REF)
    if ref_candidates.size == 0:
        ref = 0
        warnings.warn("No slack bus type 3 found. Using bus 1 as reference.", RuntimeWarning)
    else:
        ref = int(ref_candidates[0])
        if ref_candidates.size > 1:
            warnings.warn(
                f"Multiple slack buses found ({ref_candidates.size}). Using the first one.",
                RuntimeWarning,
            )

    eps_tol = 1e-4
    max_iter_wls = 20
    small_eps_omega = 1e-12

    nb = bus.shape[0]
    nl = branch.shape[0]
    fbus = branch[:, F_BUS].astype(int)
    tbus = branch[:, T_BUS].astype(int)

    z_corrected_info: Dict[str, Any] = {
        "applied_any_correction": False,
        "iterations_performed": 0,
        "skipped_reason": "Correction not enabled or no group specified by options.",
        "last_applied_error_norm": None,
        "last_corrected_global_indices": np.array([], dtype=int),
        "last_original_values": np.array([], dtype=float),
        "last_estimated_errors": np.array([], dtype=float),
        "last_corrected_values": np.array([], dtype=float),
    }

    valid_meas_mask = np.isfinite(z_in_full) & (z_in_full < 998.0) & (z_in_full > -998.0)
    z_active = z_in_full[valid_meas_mask].copy()

    if z_active.size == 0:
        raise ValueError("No valid measurements found in z_in_full.")

    R_variances_active = R_variances_full_in[valid_meas_mask].astype(float)
    if np.any(~np.isfinite(R_variances_active)) or np.any(R_variances_active <= 0.0):
        raise ValueError("Active measurement variances must be finite and strictly positive.")

    W_active_diag = 1.0 / R_variances_active
    W_active = sp.diags(W_active_diag, 0, format="csc")

    Ybus, Yf, Yt = make_ybus(baseMVA, bus, branch)

    nstate_vars = np.r_[np.arange(ref), np.arange(ref + 1, nb), np.arange(nb, 2 * nb)]

    enable_group_correction = bool(_get_opt(options, "enable_group_correction", False))
    group_indices_are_one_based = bool(_get_opt(options, "group_indices_are_one_based", True))
    correction_group_full_indices = _normalize_group_indices(
        _get_opt(options, "correction_group_full_indices", []),
        z_in_full.shape[0],
        one_based=group_indices_are_one_based,
    )

    if enable_group_correction:
        max_correction_iterations = int(_get_opt(options, "max_correction_iterations", 1))
        max_total_wls_runs = 1 + max_correction_iterations
        error_tol_for_correction_norm = float(_get_opt(options, "correction_error_tolerance", 1e-4))
    else:
        max_total_wls_runs = 1
        error_tol_for_correction_norm = 1e-4

    current_wls_run_number = 0
    perform_next_wls_run = True
    overall_wls_success_status = False

    H_active: sp.csc_matrix | None = None
    Gain: sp.csc_matrix | None = None
    hx_active: np.ndarray | None = None
    x_state: np.ndarray | None = None

    while perform_next_wls_run and current_wls_run_number < max_total_wls_runs:
        current_wls_run_number += 1

        initial_angles = np.deg2rad(bus_data[:, VA])
        initial_angles = initial_angles - initial_angles[ref]
        initial_voltages = bus_data[:, VM]
        x_state = np.r_[initial_angles, initial_voltages].astype(float)

        k_iter_wls = 0
        wls_converged_this_run = False

        while (not wls_converged_this_run) and (k_iter_wls < max_iter_wls):
            k_iter_wls += 1

            Vc_complex = x_state[nb: 2 * nb] * np.exp(1j * x_state[0:nb])
            hx_V = x_state[nb: 2 * nb]

            Ibus_complex = Ybus @ Vc_complex
            S_inj_complex = Vc_complex * np.conj(Ibus_complex)
            hx_Pinj = S_inj_complex.real
            hx_Qinj = S_inj_complex.imag

            _, _, _, _, Sf_complex, St_complex = dSbr_dV1(Yf, Yt, Vc_complex, nb, nl, fbus, tbus)
            hx_Pf = np.asarray(Sf_complex).real.reshape(-1)
            hx_Qf = np.asarray(Sf_complex).imag.reshape(-1)
            hx_Pt = np.asarray(St_complex).real.reshape(-1)
            hx_Qt = np.asarray(St_complex).imag.reshape(-1)

            hx_full_ordered = np.r_[hx_V, hx_Pinj, hx_Qinj, hx_Pf, hx_Qf, hx_Pt, hx_Qt]
            hx_active = hx_full_ordered[valid_meas_mask]

            dSbus_dVa_raw, dSbus_dVm_raw = dSbus_dV_polar(Ybus, Vc_complex)
            dSf_dVa_raw, dSf_dVm_raw, dSt_dVa_raw, dSt_dVm_raw, _, _ = dSbr_dV1(
                Yf, Yt, Vc_complex, nb, nl, fbus, tbus
            )

            H_V_Vang = sp.csr_matrix((nb, nb), dtype=float)
            H_V_Vmag = sp.eye(nb, format="csr")
            H_Pinj_Vang = dSbus_dVa_raw.real.tocsr()
            H_Pinj_Vmag = dSbus_dVm_raw.real.tocsr()
            H_Qinj_Vang = dSbus_dVa_raw.imag.tocsr()
            H_Qinj_Vmag = dSbus_dVm_raw.imag.tocsr()
            H_Pf_Vang = dSf_dVa_raw.real.tocsr()
            H_Pf_Vmag = dSf_dVm_raw.real.tocsr()
            H_Qf_Vang = dSf_dVa_raw.imag.tocsr()
            H_Qf_Vmag = dSf_dVm_raw.imag.tocsr()
            H_Pt_Vang = dSt_dVa_raw.real.tocsr()
            H_Pt_Vmag = dSt_dVm_raw.real.tocsr()
            H_Qt_Vang = dSt_dVa_raw.imag.tocsr()
            H_Qt_Vmag = dSt_dVm_raw.imag.tocsr()

            H_full_ordered = sp.vstack(
                [
                    sp.hstack([H_V_Vang, H_V_Vmag], format="csr"),
                    sp.hstack([H_Pinj_Vang, H_Pinj_Vmag], format="csr"),
                    sp.hstack([H_Qinj_Vang, H_Qinj_Vmag], format="csr"),
                    sp.hstack([H_Pf_Vang, H_Pf_Vmag], format="csr"),
                    sp.hstack([H_Qf_Vang, H_Qf_Vmag], format="csr"),
                    sp.hstack([H_Pt_Vang, H_Pt_Vmag], format="csr"),
                    sp.hstack([H_Qt_Vang, H_Qt_Vmag], format="csr"),
                ],
                format="csr",
            )

            H_active_temp = H_full_ordered[valid_meas_mask, :]
            H_active = H_active_temp[:, nstate_vars].tocsc()

            mismatch = z_active - hx_active
            Gain = (H_active.T @ W_active @ H_active).tocsc()

            gain_dense = Gain.toarray()
            if _rcond_dense(gain_dense) < 1e-14:
                wls_converged_this_run = False
                break

            rhs = np.asarray(H_active.T @ (W_active @ mismatch)).reshape(-1)
            dx = _solve_gain(Gain, rhs)

            x_state_temp_update = np.zeros(2 * nb, dtype=float)
            x_state_temp_update[nstate_vars] = dx
            x_state = x_state + x_state_temp_update
            x_state[ref] = 0.0

            if np.max(np.abs(dx)) < eps_tol:
                wls_converged_this_run = True
            elif k_iter_wls >= max_iter_wls:
                wls_converged_this_run = False
                break

        if not wls_converged_this_run:
            overall_wls_success_status = False
            perform_next_wls_run = False
            if current_wls_run_number == 1 and enable_group_correction:
                z_corrected_info["skipped_reason"] = "Initial WLS run failed to converge, correction not attempted."
            elif enable_group_correction and z_corrected_info["applied_any_correction"]:
                z_corrected_info["skipped_reason"] = (
                    f"WLS failed to converge after {z_corrected_info['iterations_performed']} correction iteration(s)."
                )
            continue

        overall_wls_success_status = True

        can_attempt_correction = (
            enable_group_correction
            and correction_group_full_indices.size > 0
            and current_wls_run_number < max_total_wls_runs
        )

        if can_attempt_correction:
            assert H_active is not None and Gain is not None and hx_active is not None

            H_dense = H_active.toarray()
            Ginv_Ht = _solve_gain(Gain, H_dense.T)
            temp_Omega = np.diag(R_variances_active) - H_dense @ Ginv_Ht
            temp_S_active = temp_Omega * W_active_diag[np.newaxis, :]

            global_indices_of_active_meas = np.flatnonzero(valid_meas_mask)
            is_member_of_group = np.isin(global_indices_of_active_meas, correction_group_full_indices)
            group_local_indices_in_active = np.flatnonzero(is_member_of_group)

            if group_local_indices_in_active.size > 0:
                r_s_raw = z_active - hx_active
                r_s_raw_group = r_s_raw[group_local_indices_in_active]
                S_ss = temp_S_active[np.ix_(group_local_indices_in_active, group_local_indices_in_active)]

                if _rcond_dense(S_ss) > 1e-12:
                    estimated_errors = np.linalg.solve(S_ss, r_s_raw_group)
                    norm_of_estimated_errors = float(np.linalg.norm(estimated_errors))

                    if norm_of_estimated_errors > error_tol_for_correction_norm:
                        original_group_values = z_active[group_local_indices_in_active].copy()
                        corrected_group_values = original_group_values - estimated_errors
                        z_active[group_local_indices_in_active] = corrected_group_values

                        z_corrected_info["applied_any_correction"] = True
                        z_corrected_info["iterations_performed"] += 1
                        z_corrected_info["last_applied_error_norm"] = norm_of_estimated_errors
                        corrected_global_indices = global_indices_of_active_meas[group_local_indices_in_active].copy()
                        if group_indices_are_one_based:
                            corrected_global_indices = corrected_global_indices + 1
                        z_corrected_info["last_corrected_global_indices"] = corrected_global_indices
                        z_corrected_info["last_original_values"] = original_group_values
                        z_corrected_info["last_estimated_errors"] = estimated_errors
                        z_corrected_info["last_corrected_values"] = corrected_group_values
                        z_corrected_info["skipped_reason"] = ""

                        perform_next_wls_run = True
                    else:
                        perform_next_wls_run = False
                        if not z_corrected_info["applied_any_correction"]:
                            z_corrected_info["skipped_reason"] = "Estimated errors below tolerance on first check."
                        else:
                            z_corrected_info["skipped_reason"] = (
                                f"Errors below tolerance after {z_corrected_info['iterations_performed']} correction(s)."
                            )
                else:
                    perform_next_wls_run = False
                    z_corrected_info["skipped_reason"] = "S_ss matrix ill-conditioned during correction attempt."
            else:
                perform_next_wls_run = False
                if not z_corrected_info["applied_any_correction"]:
                    z_corrected_info["skipped_reason"] = "Correction group empty in active measurements."
        else:
            perform_next_wls_run = False
            if enable_group_correction and current_wls_run_number >= max_total_wls_runs:
                if not z_corrected_info["applied_any_correction"]:
                    z_corrected_info["skipped_reason"] = (
                        "Max WLS runs reached; no effective correction applied or errors were small."
                    )
                elif not z_corrected_info["skipped_reason"]:
                    z_corrected_info["skipped_reason"] = "Max WLS runs reached."

    success = int(overall_wls_success_status)

    if success:
        assert H_active is not None and Gain is not None and hx_active is not None
        final_resid_raw = z_active - hx_active

        H_dense = H_active.toarray()
        Ginv_Ht = _solve_gain(Gain, H_dense.T)
        Omega = np.diag(R_variances_active) - H_dense @ Ginv_Ht

        diagOmega = np.diag(Omega).copy()
        near_zero_negative = (diagOmega < 0.0) & (diagOmega > -small_eps_omega * 10.0)
        diagOmega[near_zero_negative] = 0.0
        diagOmega[diagOmega <= 0.0] = small_eps_omega

        r_norm = np.abs(final_resid_raw) / np.sqrt(diagOmega)
    else:
        r_norm = np.array([], dtype=float)
        Omega = np.empty((0, 0), dtype=float)
        final_resid_raw = np.array([], dtype=float)
        if (not z_corrected_info["skipped_reason"]) and current_wls_run_number > 0:
            z_corrected_info["skipped_reason"] = "WLS process failed to converge."

    # The MATLAB script overwrites lambdaN at the very end, so preserve that behavior.
    lambdaN = np.zeros(nl, dtype=float)

    return lambdaN, success, r_norm, Omega, final_resid_raw, z_corrected_info
