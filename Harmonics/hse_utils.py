import numpy as np
import math
import sys
import os

# Ensure we can import from the same package if needed, though this is a utility module.
# The user might have ieee14_verification in Harmonics package. 
# We assume this file is in d:\ps_llm_agent\Harmonics\hse_utils.py

# We need the scaling functions from ieee14_verification or duplicate them. 
# proper approach: import from ieee14_verification
from .ieee14_verification import scale_branch_params, _tap_complex, fundamental_bus_voltages

def build_ybus_harmonic(bus, branch, base_mva: float, h: int, r_model: str = "sqrt") -> np.ndarray:
    """
    Harmonic Ybus(h): frequency-scaled branch parameters + scaled bus shunt susceptance.
    """
    nb = bus.shape[0]
    Y = np.zeros((nb, nb), dtype=complex)

    # bus shunts (Gs + j Bs), scale Bs with h, Gs usually ignored or constant? 
    # Standard practice: Gs constant, Bs * h
    # MATPOWER bus cols: GS=5, BS=6 (0-indexed: 4, 5)
    Gs = bus[:, 4] / base_mva
    Bs = bus[:, 5] / base_mva
    Y[np.arange(nb), np.arange(nb)] += (Gs + 1j * (Bs * h))

    for br in branch:
        f = int(br[0]) - 1
        t = int(br[1]) - 1
        # MATPOWER branch cols: BR_R=3, BR_X=4, BR_B=5 (0-indexed: 2, 3, 4)
        r1, x1, b1 = float(br[2]), float(br[3]), float(br[4])
        r, x, b = scale_branch_params(r1, x1, b1, h, r_model)

        # TAP: ratio=9, angle=10 (0-indexed: 8, 9)
        tap = _tap_complex(float(br[8]), float(br[9]))
        
        y = 0.0 if (r == 0 and x == 0) else 1.0 / complex(r, x)
        ysh = 1j * b / 2.0

        Yff = (y + ysh) / (tap * np.conj(tap))
        Yft = -y / np.conj(tap)
        Ytt = (y + ysh)
        Ytf = -y / tap

        Y[f, f] += Yff
        Y[t, t] += Ytt
        Y[f, t] += Yft
        Y[t, f] += Ytf

    return Y

def estimate_single_source_injection_from_voltage(
    a: np.ndarray,         # complex vector (m,)
    v_meas: np.ndarray,    # complex vector (m,)
    sigma: np.ndarray      # real vector (m,)
) -> tuple:
    """
    Fit v_meas ≈ a * I (complex scalar) by WLS.
    Returns (I_hat_complex, SSE_weighted).
    """
    m = len(a)
    if m == 0:
        return 0.0 + 0.0j, float("inf")

    # Build real system: y = A x, x=[I_re, I_im]
    # Re(v)=Re(a)*Ire - Im(a)*Iim
    # Im(v)=Im(a)*Ire + Re(a)*Iim
    A = np.zeros((2 * m, 2), dtype=float)
    y = np.zeros(2 * m, dtype=float)

    A[:m, 0] = a.real
    A[:m, 1] = -a.imag
    A[m:, 0] = a.imag
    A[m:, 1] = a.real
    y[:m] = v_meas.real
    y[m:] = v_meas.imag

    # Row scaling by sigma
    sigma2 = np.concatenate([sigma, sigma])
    sigma2 = np.clip(sigma2, 1e-12, None)
    Aw = A / sigma2[:, None]
    yw = y / sigma2

    # Solve least squares
    x_hat, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    resid = yw - Aw @ x_hat
    sse = float(resid.T @ resid)

    I_hat = complex(x_hat[0], x_hat[1])
    return I_hat, sse

def harmonic_source_hse_single_source_scan(
    bus, branch, base_mva: float,
    harmonic_orders: list,
    Vh_meas_by_h: dict,         # h -> (buses0, Vmeas, sigma)
    slack_bus: int = 0,
    candidate_buses_1based: list | None = None,
    r_model: str = "sqrt",
    reg_eps: float = 1e-10
):
    """
    Returns:
      - best_bus_1based
      - ranking: list of dicts [{bus_1based, score}, ...]
      - I_source_hat: dict h -> complex injection at best bus
      - Vhat_by_h: dict h -> complex bus voltages (nb,) with slack harmonic voltage fixed to 0
    """
    nb = bus.shape[0]
    u = [i for i in range(nb) if i != slack_bus]  # unknown buses (non-slack)
    nu = len(u)
    bus_to_u = {bus_i: k for k, bus_i in enumerate(u)}

    if candidate_buses_1based is None:
        candidate0 = [i for i in range(nb) if i != slack_bus]
    else:
        candidate0 = [b - 1 for b in candidate_buses_1based if (b - 1) != slack_bus]

    candidate_u = [bus_to_u[i] for i in candidate0 if i in bus_to_u]

    # score per candidate (sum over harmonics)
    score = {cu: 0.0 for cu in candidate_u}
    Ihat_per_h_per_cand = {cu: {} for cu in candidate_u}

    # Precompute Zuu(h) per harmonic
    Zuu_by_h = {}

    for h in harmonic_orders:
        if h == 1:
            continue
        Yh = build_ybus_harmonic(bus, branch, base_mva, h, r_model=r_model)
        Yuu = Yh[np.ix_(u, u)] + reg_eps * np.eye(nu)
        # Zuu = inv(Yuu)
        Zuu = np.linalg.solve(Yuu, np.eye(nu))
        Zuu_by_h[h] = Zuu

        # measurements for this harmonic
        # Vh_meas_by_h structure check: (buses0, Vmeas, sigma)
        if h not in Vh_meas_by_h:
            continue
            
        buses0_in, Vmeas_in, sigma_in = Vh_meas_by_h[h]
        
        # Filter out slack bus measurements if any
        mask = [int(b) != slack_bus for b in buses0_in]
        buses0 = np.array(buses0_in)[mask]
        Vmeas = np.array(Vmeas_in)[mask]
        sigma = np.array(sigma_in)[mask]

        if len(buses0) == 0:
            continue
            
        meas_u = np.array([bus_to_u[b] for b in buses0], dtype=int)

        for cu in candidate_u:
            a = Zuu[meas_u, cu]  # transfer from injection at candidate bus to measured voltages
            I_hat, sse = estimate_single_source_injection_from_voltage(a, Vmeas, sigma)
            score[cu] += sse
            Ihat_per_h_per_cand[cu][h] = I_hat

    # pick best
    if not score:
        # Fallback if no valid candidates or measurements
        return None, [], {}, {}
        
    best_cu = min(score, key=score.get)
    best_bus0 = u[best_cu]
    best_bus_1based = best_bus0 + 1

    # ranking
    ranking = sorted(
        [{"bus_1based": (u[cu] + 1), "score": float(score[cu])} for cu in candidate_u],
        key=lambda d: d["score"]
    )

    # reconstruct states V(h) for best bus
    Vhat_by_h = {1: fundamental_bus_voltages(bus)}  # keep fundamental for THD calc if you want
    I_source_hat = {}

    for h in harmonic_orders:
        if h == 1:
            continue
        # IF we didn't calculate Zuu for this h (no measurements?), we can't reconstruct properly unless we rebuild Y
        if h in Zuu_by_h:
            Zuu = Zuu_by_h[h]
        else:
            # Rebuild Zuu if missing (e.g. no measurements for this harmonic but we want to simulate injection 0?)
            Yh = build_ybus_harmonic(bus, branch, base_mva, h, r_model=r_model)
            Yuu = Yh[np.ix_(u, u)] + reg_eps * np.eye(nu)
            Zuu = np.linalg.solve(Yuu, np.eye(nu))
        
        I_hat = Ihat_per_h_per_cand[best_cu].get(h, 0.0 + 0.0j)
        I_source_hat[h] = I_hat

        Iu = np.zeros(nu, dtype=complex)
        Iu[best_cu] = I_hat
        Vu = Zuu @ Iu

        Vh_full = np.zeros(nb, dtype=complex)
        Vh_full[slack_bus] = 0.0 + 0.0j
        for k, bus0 in enumerate(u):
            Vh_full[bus0] = Vu[k]

        Vhat_by_h[h] = Vh_full

    return best_bus_1based, ranking, I_source_hat, Vhat_by_h

def compute_thd_from_states(Vhat_by_h: dict, harmonic_orders: list, V1: np.ndarray) -> np.ndarray:
    nb = len(V1)
    thd = np.zeros(nb, dtype=float)
    for i in range(nb):
        num = 0.0
        for h in harmonic_orders:
            if h == 1:
                continue
            num += abs(Vhat_by_h.get(h, np.zeros(nb))[i]) ** 2
        thd[i] = math.sqrt(num) / max(abs(V1[i]), 1e-12)
    return thd
