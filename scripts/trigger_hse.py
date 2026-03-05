import os
import sys
import math
import json
import numpy as np
import matlab.engine

# Add project roots to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Harmonics')))

from Harmonics import ieee14_verification as h_ver
from Harmonics.ieee14_verification import (
    BUS, BRANCH, BASE_MVA, ABB_6PULSE_WITH_CHOKE,
    fundamental_bus_voltages, fundamental_load_currents,
    make_harmonic_current_injections, solve_all_harmonics,
    build_ybus,
    LegacyAnalogWattVarTransducer
)

# ----------------------------
# 0) (Small) helper: chi-square threshold
# ----------------------------
def chi2_threshold(dof: int, alpha: float = 0.05) -> float:
    """
    Return chi-square threshold for P(J <= thr) = 1 - alpha.
    Uses SciPy if available; otherwise Wilson-Hilferty approximation.
    """
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(1.0 - alpha, dof))
    except Exception:
        # Wilson-Hilferty approximation for chi-square quantile
        # q ≈ dof * (1 - 2/(9dof) + z*sqrt(2/(9dof)))^3
        # where z = N^{-1}(1-alpha)
        try:
            from math import erf, sqrt
            # approximate inverse CDF for normal using erfinv if available
            from math import erfcinv  # type: ignore
            z = math.sqrt(2) * erfcinv(2 * alpha)
        except Exception:
            # fallback z values for common alpha
            z = 1.6448536269514722 if abs(alpha - 0.05) < 1e-12 else 2.3263478740408408
        d = float(dof)
        return d * (1.0 - 2.0 / (9.0 * d) + z * math.sqrt(2.0 / (9.0 * d))) ** 3


# ----------------------------
# 1) Your measurement builder, modified to optionally return harmonic states
# ----------------------------
def build_full_harmonic_z(
    bus=BUS,
    branch=BRANCH,
    base_mva=BASE_MVA,
    harmonic_on=True,
    rng_seed_harmonic=42,
    rng_seed_noise=999,
    add_noise=True,
    return_harmonic_states=False
):
    """
    Builds full measurement vector z = [Vm; Pinj; Qinj; Pf; Qf; Pt; Qt]
    using harmonic injection model + legacy transducer.

    If return_harmonic_states=True, also returns (V_by_h, Iinj_by_h).
    """
    rng_h = np.random.default_rng(rng_seed_harmonic)
    rng_n = np.random.default_rng(rng_seed_noise)
    nb = bus.shape[0]
    nl = branch.shape[0]

    harmonics = [1, 5, 7, 11, 13, 17, 19]
    source_buses = [3]          # 1-based bus numbers
    spectrum = ABB_6PULSE_WITH_CHOKE

    # Fundamental baseline
    V1 = fundamental_bus_voltages(bus)
    Ybus1 = build_ybus(bus, branch, base_mva)
    I1_net = Ybus1 @ V1
    I1_load = fundamental_load_currents(bus, V1, base_mva)

    # Harmonic injections calibration to target THD at bus 3 (index 2)
    if harmonic_on:
        src_idx = [b - 1 for b in source_buses]
        Iinj_unit = make_harmonic_current_injections(nb, src_idx, spectrum, I1_load, rng_h, inj_scale=1.0)

        V_unit = solve_all_harmonics(bus, branch, harmonics, Iinj_unit, base_mva, slack_bus=0)
        thd_unit = h_ver.voltage_thd(V_unit, 2, harmonics)

        target_thd = 0.10  # <-- your abnormal level
        inj_scale = 0.0 if thd_unit < 1e-12 else (target_thd / thd_unit)

        Iinj_by_h = {h: inj_scale * Ivec for h, Ivec in Iinj_unit.items()}
    else:
        Iinj_by_h = {h: np.zeros(nb, dtype=complex) for h in harmonics if h != 1}

    # Fundamental injection (not used by HSE; used by your SCADA synthesis)
    Iinj_by_h[1] = I1_net

    # Solve all harmonics
    V_by_h = solve_all_harmonics(bus, branch, harmonics, Iinj_by_h, base_mva, slack_bus=0)

    # Branch terminal currents for all h (for SCADA flow synthesis)
    If_by_h, It_by_h = {}, {}

    def get_line_params(br, h, r_model="sqrt"):
        r1, x1, b1 = float(br[2]), float(br[3]), float(br[4])
        return h_ver.scale_branch_params(r1, x1, b1, h, r_model)

    for h in harmonics:
        If_vec = np.zeros(nl, dtype=complex)
        It_vec = np.zeros(nl, dtype=complex)
        Vh = V_by_h[h]

        for k, br in enumerate(branch):
            f = int(br[0]) - 1
            t = int(br[1]) - 1
            r, x, b = get_line_params(br, h)
            tap = h_ver._tap_complex(float(br[8]), float(br[9]))
            y = 0.0 if (r == 0 and x == 0) else 1.0 / complex(r, x)
            ysh = 1j * b / 2.0

            Yff = (y + ysh) / (tap * np.conj(tap))
            Yft = -y / np.conj(tap)
            Ytt = (y + ysh)
            Ytf = -y / tap

            If_vec[k] = Yff * Vh[f] + Yft * Vh[t]
            It_vec[k] = Ytf * Vh[f] + Ytt * Vh[t]

        If_by_h[h] = If_vec
        It_by_h[h] = It_vec

    # Legacy SCADA transducer
    transducer = LegacyAnalogWattVarTransducer(f0_hz=60.0)

    z_Vm, z_Pinj, z_Qinj, z_Pf, z_Qf, z_Pt, z_Qt = [], [], [], [], [], [], []

    # Vm: RMS magnitude of full waveform
    for i in range(nb):
        v_rms_pu = math.sqrt(sum(abs(V_by_h[h][i]) ** 2 for h in harmonics))
        noise = rng_n.normal(0, 0.001) if add_noise else 0.0
        z_Vm.append(v_rms_pu + noise)

    # Pinj/Qinj: legacy transducer applied to node injection current (simplified)
    for i in range(nb):
        Vpoint = {h: V_by_h[h][i] for h in harmonics}
        Ipoint = {h: Iinj_by_h.get(h, np.zeros(nb, complex))[i] for h in harmonics}
        P_pu, Q_pu = transducer.measure_PQ_pu(Vpoint, Ipoint)

        noise_p = rng_n.normal(0, 1.0) if add_noise else 0.0
        noise_q = rng_n.normal(0, 1.0) if add_noise else 0.0
        z_Pinj.append((P_pu * base_mva + noise_p) / base_mva)
        z_Qinj.append((Q_pu * base_mva + noise_q) / base_mva)

    # Pf/Qf/Pt/Qt: legacy transducer at branch terminals
    nl = branch.shape[0]
    for k in range(nl):
        f = int(branch[k, 0]) - 1
        t = int(branch[k, 1]) - 1

        Vpoint_f = {h: V_by_h[h][f] for h in harmonics}
        Ipoint_f = {h: If_by_h[h][k] for h in harmonics}
        Pf_pu, Qf_pu = transducer.measure_PQ_pu(Vpoint_f, Ipoint_f)

        noise_p = rng_n.normal(0, 1.0) if add_noise else 0.0
        noise_q = rng_n.normal(0, 1.0) if add_noise else 0.0
        z_Pf.append((Pf_pu * base_mva + noise_p) / base_mva)
        z_Qf.append((Qf_pu * base_mva + noise_q) / base_mva)

        Vpoint_t = {h: V_by_h[h][t] for h in harmonics}
        Ipoint_t = {h: It_by_h[h][k] for h in harmonics}
        Pt_pu, Qt_pu = transducer.measure_PQ_pu(Vpoint_t, Ipoint_t)

        noise_p = rng_n.normal(0, 1.0) if add_noise else 0.0
        noise_q = rng_n.normal(0, 1.0) if add_noise else 0.0
        z_Pt.append((Pt_pu * base_mva + noise_p) / base_mva)
        z_Qt.append((Qt_pu * base_mva + noise_q) / base_mva)

    z_full = z_Vm + z_Pinj + z_Qinj + z_Pf + z_Qf + z_Pt + z_Qt

    if return_harmonic_states:
        return z_full, V_by_h, Iinj_by_h
    return z_full


# ----------------------------
# 2) Your MATLAB WLS runner (unchanged)
# ----------------------------
def run_wls(z, case_name='case14'):
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    print("MATLAB engine started.")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    eng.addpath(eng.genpath(os.path.join(repo_root, 'mcp_server')), nargout=0)
    eng.addpath(eng.genpath(os.path.join(repo_root, 'Transmission')), nargout=0)

    eng.workspace['CaseName'] = case_name
    eng.eval("mpc = loadcase(CaseName);", nargout=0)
    eng.eval("bus_data = mpc.bus;", nargout=0)

    nb = int(eng.eval("size(mpc.bus, 1)"))
    nl = int(eng.eval("size(mpc.branch, 1)"))
    expected_len = 3 * nb + 4 * nl
    if len(z) != expected_len:
        eng.quit()
        raise ValueError(f"z length {len(z)} does not match expected {expected_len}")

    eng.workspace['Z'] = matlab.double(z)

    print("Running WLS...")
    eng.eval("[lambdaN, success, r, lambda_vec, ea] = LagrangianM_singlephase(Z, mpc, 0, bus_data);", nargout=0)

    success = float(eng.workspace['success'])
    r = np.array(eng.workspace['r']).flatten()
    lambdaN = np.array(eng.workspace['lambdaN']).flatten()

    # Your global statistic (assuming r is whitened/normalized residual)
    J = float(np.sum(r ** 2))

    print(f"WLS Success: {success}")
    print(f"Max |Residual|: {float(np.max(np.abs(r))):.3f}")
    print(f"Global Chi-Square J: {J:.2f}")

    eng.quit()
    return success, r, lambdaN, J


# ----------------------------
# 3) Harmonic-meter measurement synthesis (simulation only)
# ----------------------------
def simulate_harmonic_voltage_meter_measurements(
    V_by_h: dict,
    meter_buses_1based: list,
    harmonic_orders: list,
    sigma_v: float = 5e-4,
    rng_seed: int = 1234
):
    """
    Simulate harmonic voltage phasor measurements V_i(h) at selected buses.
    Returns dict: h -> (bus_idx0based_array, Vmeas_complex_array, sigma_array)
    """
    rng = np.random.default_rng(rng_seed)
    out = {}
    for h in harmonic_orders:
        if h == 1:
            continue
        buses0 = np.array([b - 1 for b in meter_buses_1based], dtype=int)
        Vtrue = np.array([V_by_h[h][i] for i in buses0], dtype=complex)

        # complex gaussian noise: E|e|^2 = sigma_v^2
        noise = (sigma_v / math.sqrt(2.0)) * (rng.standard_normal(len(buses0)) + 1j * rng.standard_normal(len(buses0)))
        Vmeas = Vtrue + noise

        out[h] = (buses0, Vmeas, np.full(len(buses0), sigma_v, dtype=float))
    return out


# ----------------------------
# 4) Build harmonic Ybus(h) (if your h_ver already has it, you can swap in)
# ----------------------------
def build_ybus_harmonic(bus, branch, base_mva: float, h: int, r_model: str = "sqrt") -> np.ndarray:
    """
    Harmonic Ybus(h): frequency-scaled branch parameters + scaled bus shunt susceptance.
    """
    nb = bus.shape[0]
    Y = np.zeros((nb, nb), dtype=complex)

    # bus shunts (Gs + j Bs), scale Bs with h
    Gs = bus[:, 4] / base_mva
    Bs = bus[:, 5] / base_mva
    Y[np.arange(nb), np.arange(nb)] += (Gs + 1j * (Bs * h))

    for br in branch:
        f = int(br[0]) - 1
        t = int(br[1]) - 1
        r1, x1, b1 = float(br[2]), float(br[3]), float(br[4])
        r, x, b = h_ver.scale_branch_params(r1, x1, b1, h, r_model)

        tap = h_ver._tap_complex(float(br[8]), float(br[9]))
        y = 0.0 if (r == 0 and x == 0) else 1.0 / complex(r, x)
        ysh = 1j * b / 2.0

        Yff = (y + ysh) / (tap * np.conj(tap))
        Ytt = (y + ysh)
        Yft = -y / np.conj(tap)
        Ytf = -y / tap

        Y[f, f] += Yff
        Y[t, t] += Ytt
        Y[f, t] += Yft
        Y[t, f] += Ytf

    return Y


# ----------------------------
# 5) HSE: single-source scan + harmonic state reconstruction
# ----------------------------
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
        buses0, Vmeas, sigma = Vh_meas_by_h[h]
        buses0 = [int(b) for b in buses0 if int(b) != slack_bus]  # ignore slack measurements here
        if len(buses0) == 0:
            continue
        meas_u = np.array([bus_to_u[b] for b in buses0], dtype=int)
        Vmeas = np.array([Vmeas[list(Vh_meas_by_h[h][0]).index(b)] for b in buses0], dtype=complex) \
            if len(buses0) != len(Vh_meas_by_h[h][0]) else Vmeas
        sigma = np.array([sigma[list(Vh_meas_by_h[h][0]).index(b)] for b in buses0], dtype=float) \
            if len(buses0) != len(Vh_meas_by_h[h][0]) else sigma

        for cu in candidate_u:
            a = Zuu[meas_u, cu]  # transfer from injection at candidate bus to measured voltages
            I_hat, sse = estimate_single_source_injection_from_voltage(a, Vmeas, sigma)
            score[cu] += sse
            Ihat_per_h_per_cand[cu][h] = I_hat

    # pick best
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
        Zuu = Zuu_by_h[h]
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
            num += abs(Vhat_by_h[h][i]) ** 2
        thd[i] = math.sqrt(num) / max(abs(V1[i]), 1e-12)
    return thd


# ----------------------------
# 6) End-to-end: trigger -> HSE
# ----------------------------
if __name__ == "__main__":
    # 1) Build SCADA measurements + get true harmonic states (simulation)
    z, V_by_h_true, Iinj_by_h_true = build_full_harmonic_z(
        harmonic_on=True,
        rng_seed_harmonic=42,
        rng_seed_noise=100,
        add_noise=True,
        return_harmonic_states=True
    )

    # 2) Run WLS-SE
    succ, r, lambdaN, J = run_wls(z)

    # 3) Trigger logic (example): global chi-square on J
    nb = BUS.shape[0]
    nl = BRANCH.shape[0]
    m = 3 * nb + 4 * nl
    n = 2 * nb - 1
    dof = m - n
    thr = chi2_threshold(dof, alpha=0.05)

    print(f"DOF={dof}, chi2(95%) thr={thr:.2f}")
    trigger_hse = (J > thr)

    if not trigger_hse:
        print("HSE not triggered (J below threshold).")
        sys.exit(0)

    print(">>> HSE TRIGGERED <<<")

    # 4) Simulate harmonic meters (choose where you have harmonic-capable sensors)
    #    You can pick PMU buses, critical buses, etc.
    harmonic_orders = [5, 7, 11, 13, 17, 19]
    meter_buses = [2, 3, 4, 5, 9, 14]  # 1-based (example)
    Vh_meas_by_h = simulate_harmonic_voltage_meter_measurements(
        V_by_h_true,
        meter_buses_1based=meter_buses,
        harmonic_orders=harmonic_orders,
        sigma_v=5e-4,
        rng_seed=2025
    )

    # 5) Run HSE (single-source scan)
    best_bus, ranking, I_source_hat, Vhat_by_h = harmonic_source_hse_single_source_scan(
        BUS, BRANCH, BASE_MVA,
        harmonic_orders=harmonic_orders,
        Vh_meas_by_h=Vh_meas_by_h,
        slack_bus=0,
        candidate_buses_1based=None,   # or restrict to top-K candidates from WLS residual mapping
        r_model="sqrt"
    )

    print(f"\nHSE candidate harmonic source bus = {best_bus}")
    print("Top-5 candidate buses by fit score:")
    for row in ranking[:5]:
        print(f"  bus {row['bus_1based']:>2d}  score={row['score']:.4e}")

    print("\nEstimated harmonic source injections (per harmonic):")
    for h in harmonic_orders:
        I_hat = I_source_hat.get(h, 0.0 + 0.0j)
        print(f"  h={h:>2d}: I_hat = {I_hat.real:+.4e} {I_hat.imag:+.4e}j (p.u.)")

    # 6) Optional: compute THD estimate map from reconstructed harmonic voltages
    V1 = fundamental_bus_voltages(BUS)
    thd_hat = compute_thd_from_states(Vhat_by_h, [1] + harmonic_orders, V1)
    print("\nEstimated THD_V per bus (from reconstructed V(h)):")
    for i, thd in enumerate(thd_hat, start=1):
        print(f"  bus {i:>2d}: THD={thd*100:6.2f}%")

    # 7) Save trace for your LLM-agent pipeline
    out = {
        "trigger": {"J": float(J), "chi2_thr": float(thr), "dof": int(dof), "triggered": bool(trigger_hse)},
        "hse": {
            "meter_buses": meter_buses,
            "harmonic_orders": harmonic_orders,
            "best_source_bus": int(best_bus),
            "ranking_top10": ranking[:10],
            "I_source_hat": {str(h): [float(I_source_hat[h].real), float(I_source_hat[h].imag)] for h in I_source_hat},
            "THD_hat_percent": [float(x * 100.0) for x in thd_hat]
        }
    }
    with open("hse_trace.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved HSE trace to hse_trace.json")

    # ----------------------------
    # 8) Performance Verification (Ground Truth Comparison)
    # ----------------------------
    print("\n--- HSE Performance Verification ---")
    
    # A) Source Identification
    true_source_bus = 3 
    print(f"True Source Bus: {true_source_bus}")
    print(f"Est Source Bus:  {best_bus}")
    if best_bus == true_source_bus:
        print(">> Identification: SUCCESS")
    else:
        print(f">> Identification: FAIL (Distance: {abs(best_bus - true_source_bus)})")

    # B) Injection Magnitude Error
    print("\nInjection Estimation Error (at Source Bus):")
    total_I_err = 0.0
    total_I_mag = 0.0
    for h in harmonic_orders:
        I_true = Iinj_by_h_true[h][true_source_bus - 1]
        I_est = I_source_hat.get(h, 0j)
        
        # We compare the estimated injection vector (sparse) against true vector
        # Since we identified the correct bus, we just compare the scalars
        err = abs(I_est - I_true)
        mag = abs(I_true)
        total_I_err += err
        total_I_mag += mag
        
        # relative error per harmonic
        rel_err = (err / mag * 100) if mag > 1e-9 else 0.0
        print(f"  h={h}: True={abs(I_true):.4f}, Est={abs(I_est):.4f}, Error={rel_err:.2f}%")
        
    avg_I_err = (total_I_err / total_I_mag * 100) if total_I_mag > 1e-9 else 0.0
    print(f">> Average Injection Magnitude Error: {avg_I_err:.2f}%")

    # C) Voltage State Reconstruction Error
    print("\nVoltage State Reconstruction Error (RMSE over all buses):")
    total_V_sse = 0.0
    total_V_ss = 0.0
    for h in harmonic_orders:
        V_true = V_by_h_true[h]
        V_est = Vhat_by_h[h]
        
        diff = V_est - V_true
        sse = np.sum(np.abs(diff)**2)
        ss = np.sum(np.abs(V_true)**2)
        
        total_V_sse += sse
        total_V_ss += ss
        
        rmse = math.sqrt(sse / len(V_true))
        rel_rmse = (math.sqrt(sse) / math.sqrt(ss) * 100) if ss > 1e-9 else 0.0
        print(f"  h={h}: RMSE={rmse:.5f} p.u. ({rel_rmse:.2f}%)")
        
    global_v_err = (math.sqrt(total_V_sse) / math.sqrt(total_V_ss) * 100) if total_V_ss > 1e-9 else 0.0
    print(f">> Global Voltage State Relative Error: {global_v_err:.2f}%")

    # D) THD Estimation Error
    print("\nTHD Estimation Error (per bus):")
    # compute true THD
    V1_true = fundamental_bus_voltages(BUS)
    thd_true = compute_thd_from_states(V_by_h_true, [1] + harmonic_orders, V1_true)
    
    max_thd_err = 0.0
    mean_thd_err = 0.0
    
    print(f"  {'Bus':>4} | {'True THD(%)':>12} | {'Est THD(%)':>12} | {'Error(%)':>10}")
    print(f"  {'-'*46}")
    
    for i in range(len(thd_true)):
        t_true = thd_true[i] * 100.0
        t_est = thd_hat[i] * 100.0
        err = abs(t_est - t_true)
        max_thd_err = max(max_thd_err, err)
        mean_thd_err += err
        print(f"  {i+1:4d} | {t_true:12.2f} | {t_est:12.2f} | {err:10.2f}")
        
    mean_thd_err /= len(thd_true)
    print(f"\n>> Max THD Error:  {max_thd_err:.2f}% (absolute percentage points)")
    print(f">> Mean THD Error: {mean_thd_err:.2f}% (absolute percentage points)")
