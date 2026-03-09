
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _copy_result_to_internal(result: Any) -> Dict[str, np.ndarray]:
    """Copy a MATPOWER-like result struct/dict and renumber buses internally to 0..nb-1."""
    bus = np.array(_get_field(result, "bus"), dtype=float, copy=True)
    branch = np.array(_get_field(result, "branch"), dtype=float, copy=True)
    gen = np.array(_get_field(result, "gen"), dtype=float, copy=True)
    baseMVA = float(_get_field(result, "baseMVA"))

    ext_bus_ids = bus[:, BUS_I].astype(int)
    int_map = {ext_id: k for k, ext_id in enumerate(ext_bus_ids)}

    # Internal bus numbering expected by MATPOWER/PYPOWER helpers is consecutive.
    bus[:, BUS_I] = np.arange(bus.shape[0], dtype=float)
    branch[:, F_BUS] = np.vectorize(int_map.__getitem__)(branch[:, F_BUS].astype(int))
    branch[:, T_BUS] = np.vectorize(int_map.__getitem__)(branch[:, T_BUS].astype(int))
    gen[:, GEN_BUS] = np.vectorize(int_map.__getitem__)(gen[:, GEN_BUS].astype(int))

    return {"baseMVA": baseMVA, "bus": bus, "branch": branch, "gen": gen}


def make_sbus(baseMVA: float, bus: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """MATPOWER-like Sbus = generation - load, in p.u."""
    nb = bus.shape[0]
    sbus = -(bus[:, PD] + 1j * bus[:, QD]) / baseMVA

    on = gen[:, GEN_STATUS] > 0
    gbus = gen[on, GEN_BUS].astype(int)
    sgen = (gen[on, PG] + 1j * gen[on, QG]) / baseMVA
    np.add.at(sbus, gbus, sgen)
    return sbus


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


def make_jaco(
    x: np.ndarray,
    Ybus: sp.spmatrix,
    Yf: sp.spmatrix,
    Yt: sp.spmatrix,
    nb: int,
    nl: int,
    fbus: np.ndarray,
    tbus: np.ndarray,
    V: np.ndarray,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Builds the measurement Jacobian in the same row order as the MATLAB code."""
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


def _drop_rows_dense(v: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return v.copy()
    keep = np.ones(v.shape[0], dtype=bool)
    keep[rows] = False
    return v[keep]


def _split_rows_sparse(M: sp.spmatrix, rows: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return (kept_rows, picked_rows) from sparse matrix M."""
    if rows.size == 0:
        return M.tocsr(), sp.csr_matrix((0, M.shape[1]), dtype=M.dtype)

    keep = np.ones(M.shape[0], dtype=bool)
    keep[rows] = False
    return M[keep, :].tocsr(), M[rows, :].tocsr()


def lagrangian_m_singlephase(
    z: np.ndarray,
    result: Any,
    ind: int,
    bus_data: np.ndarray,
    *,
    zero_injection_tol: float | None = None,
    max_it: int = 20,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Python translation of the MATLAB function LagrangianM_singlephase().

    Parameters
    ----------
    z : array_like
        Measurement vector.
    result : dict or object
        MATPOWER-like case/result with fields/keys: bus, branch, gen, baseMVA.
    ind : int
        Preserved from the MATLAB signature. The original MATLAB only uses this
        in the final residual block; see note below.
    bus_data : ndarray
        External bus data used to initialize angle (deg) and Vm.
    zero_injection_tol : float or None
        If None, preserves the MATLAB script's current behavior (ZIn = []).
        If a float is provided, buses with |Sbus| < zero_injection_tol are
        treated as zero-injection buses.
    max_it : int
        Maximum WLS iterations.
    tol : float
        Convergence tolerance on max(abs(dx)).

    Returns
    -------
    lambdaN, success, r, lambda_vec, ea

    Notes
    -----
    The original MATLAB script contains two quirks that this port intentionally
    preserves unless you choose to modify them:
      1) zero-injection detection is disabled (`ZIn = []`);
      2) `ind == 1` is only applied in the final residual calculation, not in
         the WLS loop itself.
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    bus_data = np.asarray(bus_data, dtype=float)

    ppc = _copy_result_to_internal(result)
    bus = ppc["bus"]
    branch = ppc["branch"]
    gen = ppc["gen"]
    baseMVA = ppc["baseMVA"]

    ref_candidates = np.flatnonzero(bus[:, BUS_TYPE].astype(int) == REF)
    if ref_candidates.size != 1:
        raise ValueError(f"Expected exactly one reference bus, found {ref_candidates.size}.")
    ref = int(ref_candidates[0])

    # Ensure reference angle is zero (degrees in bus matrix).
    bus[:, VA] = bus[:, VA] - bus[ref, VA]

    nb = bus.shape[0]
    nl = branch.shape[0]
    fbus = branch[:, F_BUS].astype(int)
    tbus = branch[:, T_BUS].astype(int)

    # --- zero-injection handling ---
    ZI = make_sbus(baseMVA, bus, gen)
    if zero_injection_tol is None:
        zin = np.array([], dtype=int)  # exact MATLAB script behavior
    else:
        zin = np.flatnonzero(np.abs(ZI) < zero_injection_tol)
    nzi = zin.size

    # measurement covariance / weights
    Rdiag_full = np.r_[
        (0.001 ** 2) * np.ones(nb),
        (0.01 ** 2) * np.ones(2 * nb - 2 * nzi),
        (0.01 ** 2) * np.ones(4 * nl),
    ]
    W = sp.diags(1.0 / Rdiag_full, offsets=0, format="csc")

    # build Y-bus
    Ybus, Yf, Yt = make_ybus(baseMVA, bus, branch)

    # state initialization
    nstate = np.r_[np.arange(ref), np.arange(ref + 1, 2 * nb)]
    x = np.r_[np.zeros(nb), np.ones(nb)]
    x[0:nb] = bus_data[:, VA] * (np.pi / 180.0)
    x[nb:2 * nb] = bus_data[:, VM]
    x0 = x[np.r_[np.arange(ref), np.arange(ref + 1, nb), np.arange(nb, 2 * nb)]].copy()

    # rows removed by zero injections
    zi_rows = np.r_[nb + zin, 2 * nb + zin].astype(int)
    z_used = _drop_rows_dense(z, zi_rows)

    success = 1
    dxl = None
    H = None
    C = None
    Gain = None
    hx = None

    for _k in range(max_it + 1):
        Vc = x[nb:2 * nb] * np.exp(1j * x[0:nb])
        H_full, Sf, St = make_jaco(x, Ybus, Yf, Yt, nb, nl, fbus, tbus, Vc)

        # remove reference-angle column
        H_full = H_full[:, np.r_[np.arange(ref), np.arange(ref + 1, H_full.shape[1])]]

        # split off zero-injection rows
        H, C = _split_rows_sparse(H_full, zi_rows)

        Ibus = Ybus @ Vc
        Sinj = Vc * np.conj(Ibus)
        Pinj = Sinj.real
        Qinj = Sinj.imag

        Pf = np.asarray(Sf).reshape(-1).real
        Qf = np.asarray(Sf).reshape(-1).imag
        Pt = np.asarray(St).reshape(-1).real
        Qt = np.asarray(St).reshape(-1).imag

        hx_full = np.r_[x[nb:2 * nb], Pinj, Qinj, Pf, Qf, Pt, Qt]
        cx = hx_full[zi_rows] if zi_rows.size else np.zeros(0)
        hx = _drop_rows_dense(hx_full, zi_rows)

        if z_used.shape[0] != hx.shape[0]:
            raise ValueError(
                f"Measurement length mismatch after zero-injection removal: "
                f"len(z)={z_used.shape[0]}, len(hx)={hx.shape[0]}."
            )

        mH = H.shape[0]
        nC = C.shape[0]
        ns = 2 * nb - 1

        Gain = sp.bmat(
            [
                [sp.csc_matrix((ns, ns)), H.T @ W, C.T],
                [H, sp.eye(mH, format="csc"), sp.csc_matrix((mH, nC))],
                [C, sp.csc_matrix((nC, mH)), sp.csc_matrix((nC, nC))],
            ],
            format="csc",
        )

        mismatch = z_used - hx
        tk = np.r_[np.zeros(ns), mismatch, -cx]

        # MATLAB uses symamd + lu; use sparse LU with a symmetric-min-degree-style ordering
        lu = spla.splu(Gain, permc_spec="MMD_AT_PLUS_A")
        dxl = lu.solve(tk)
        dx = dxl[:ns]

        if np.max(np.abs(dx)) < tol:
            break

        x0 = x0 + dx
        x[nstate] = x0
    else:
        lambdaN = 10.0 * np.ones(nl)
        return lambdaN, 0, np.array([]), np.array([]), np.array([])

    # === parameter Jacobian wrt line r and x ===
    ntheta = np.r_[np.arange(ref), np.arange(ref + 1, nb)]
    theta_est = np.zeros(nb)
    theta_est[ntheta] = x0[: nb - 1]
    V_est = x0[nb - 1 :]

    Hp_full = np.zeros((3 * nb + 4 * nl, 2 * nl), dtype=float)

    for kk in range(nl):
        i = fbus[kk]
        j = tbus[kk]
        r_k = branch[kk, BR_R]
        x_k = branch[kk, BR_X]

        denom = r_k**2 + x_k**2
        g_ij = r_k / denom
        b_ij = -x_k / denom

        dg_dr = (x_k**2 - r_k**2) / (denom**2)
        db_dr = (2.0 * r_k * x_k) / (denom**2)
        dg_dx = (-2.0 * r_k * x_k) / (denom**2)
        db_dx = (x_k**2 - r_k**2) / (denom**2)

        Vi = V_est[i]
        Vj = V_est[j]
        dth = theta_est[i] - theta_est[j]
        cosd = np.cos(dth)
        sind = np.sin(dth)

        dPi_dg = Vi**2 - Vi * Vj * cosd
        dPi_db = Vi * Vj * sind

        dQi_dg = Vi * Vj * sind
        dQi_db = -Vi**2 + Vi * Vj * cosd

        dPj_dg = Vj**2 - Vi * Vj * cosd
        dPj_db = -Vi * Vj * sind

        dQj_dg = -Vi * Vj * sind
        dQj_db = -Vj**2 + Vi * Vj * cosd

        dPi_dr = dPi_dg * dg_dr + dPi_db * db_dr
        dPi_dx = dPi_dg * dg_dx + dPi_db * db_dx
        dPj_dr = dPj_dg * dg_dr + dPj_db * db_dr
        dPj_dx = dPj_dg * dg_dx + dPj_db * db_dx
        dQi_dr = dQi_dg * dg_dr + dQi_db * db_dr
        dQi_dx = dQi_dg * dg_dx + dQi_db * db_dx
        dQj_dr = dQj_dg * dg_dr + dQj_db * db_dr
        dQj_dx = dQj_dg * dg_dx + dQj_db * db_dx

        c0 = 2 * kk
        c1 = c0 + 1

        Hp_full[nb + i, c0] = dPi_dr
        Hp_full[nb + i, c1] = dPi_dx
        Hp_full[nb + j, c0] = dPj_dr
        Hp_full[nb + j, c1] = dPj_dx
        Hp_full[2 * nb + i, c0] = dQi_dr
        Hp_full[2 * nb + i, c1] = dQi_dx
        Hp_full[2 * nb + j, c0] = dQj_dr
        Hp_full[2 * nb + j, c1] = dQj_dx

        Hp_full[3 * nb + kk, c0] = dPi_dr
        Hp_full[3 * nb + kk, c1] = dPi_dx
        Hp_full[3 * nb + nl + kk, c0] = dQi_dr
        Hp_full[3 * nb + nl + kk, c1] = dQi_dx
        Hp_full[3 * nb + 2 * nl + kk, c0] = dPj_dr
        Hp_full[3 * nb + 2 * nl + kk, c1] = dPj_dx
        Hp_full[3 * nb + 3 * nl + kk, c0] = dQj_dr
        Hp_full[3 * nb + 3 * nl + kk, c1] = dQj_dx

    Cp = Hp_full[zi_rows, :] if zi_rows.size else np.zeros((0, 2 * nl))
    Hp = np.delete(Hp_full, zi_rows, axis=0) if zi_rows.size else Hp_full.copy()

    # Original MATLAB comment says: "If ind==1, also remove row nb from Hp"
    # but no code is present there. This port preserves the script as written.

    S = -np.vstack([W @ Hp, Cp]).T

    # Instead of forming inv(Gain), solve only the columns that correspond to the measurement block.
    ns = 2 * nb - 1
    mH = H.shape[0]
    n_total = Gain.shape[0]
    rhs = np.zeros((n_total, mH), dtype=float)
    rhs[ns : ns + mH, :] = np.eye(mH)
    lu_gain = spla.splu(Gain, permc_spec="MMD_AT_PLUS_A")
    X = lu_gain.solve(rhs)
    phi = X[ns:, :]  # equivalent to [E5; E8]

    R_used = np.diag(Rdiag_full)
    covu = phi @ R_used @ phi.T
    ea = S @ covu @ S.T

    lambda_vec = S @ dxl[ns:]
    tt = np.sqrt(np.clip(np.diag(ea), a_min=0.0, a_max=None))
    lambdaN = np.divide(lambda_vec, tt, out=np.full_like(lambda_vec, np.nan), where=tt > 0)

    # === final residual ===
    final_Vc = x[nb:2 * nb] * np.exp(1j * x[0:nb])
    Ibus_f = Ybus @ final_Vc
    Sinj_f = final_Vc * np.conj(Ibus_f)
    Pinj_f = Sinj_f.real
    Qinj_f = Sinj_f.imag
    _, Sf_f, St_f = make_jaco(x, Ybus, Yf, Yt, nb, nl, fbus, tbus, final_Vc)

    Pf_f = np.asarray(Sf_f).reshape(-1).real
    Qf_f = np.asarray(Sf_f).reshape(-1).imag
    Pt_f = np.asarray(St_f).reshape(-1).real
    Qt_f = np.asarray(St_f).reshape(-1).imag

    hx_full = np.r_[x[nb:2 * nb], Pinj_f, Qinj_f, Pf_f, Qf_f, Pt_f, Qt_f]
    hx_resid = _drop_rows_dense(hx_full, zi_rows)

    H_resid = H
    Rdiag_resid = Rdiag_full.copy()

    if ind == 1:
        # MATLAB uses hx_full(nb) = [] with 1-based indexing => remove Python row nb-1.
        # The original script does not remove the matching row from H/Rdiag in the WLS loop.
        # We remove it here only for the residual calculation to keep dimensions consistent.
        idx = nb - 1
        hx_resid = np.delete(hx_resid, idx)
        keep = np.ones(H_resid.shape[0], dtype=bool)
        keep[idx] = False
        H_resid = H_resid[keep, :]
        Rdiag_resid = np.delete(Rdiag_resid, idx)

    z_resid = z_used.copy()
    if ind == 1:
        z_resid = np.delete(z_resid, nb - 1)

    final_resid = z_resid - hx_resid

    Gres = (H_resid.T @ sp.diags(1.0 / Rdiag_resid, 0, format="csc") @ H_resid).tocsc()
    lu_res = spla.splu(Gres, permc_spec="MMD_AT_PLUS_A")
    Ginv_Ht = lu_res.solve(H_resid.T.toarray())
    proj_diag = np.sum(H_resid.toarray() * Ginv_Ht.T, axis=1)
    omega_diag = Rdiag_resid - proj_diag
    omega_diag = np.clip(omega_diag, a_min=np.finfo(float).eps, a_max=None)
    r = np.abs(final_resid) / np.sqrt(omega_diag)

    return lambdaN, success, r, lambda_vec, ea
