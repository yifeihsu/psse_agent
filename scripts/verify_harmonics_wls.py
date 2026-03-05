
import sys
import os
import math
import matlab.engine
import numpy as np
import json

# Add project roots to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Harmonics')))

from Harmonics import ieee14_verification as h_ver
from Harmonics.ieee14_verification import (
    BUS, BRANCH, BASE_MVA, ABB_6PULSE_WITH_CHOKE,
    fundamental_bus_voltages, fundamental_load_currents,
    make_harmonic_current_injections, solve_all_harmonics,
    branch_terminal_currents, build_ybus,
    LegacyAnalogWattVarTransducer
)


def build_full_harmonic_z(
    bus=BUS,
    branch=BRANCH,
    base_mva=BASE_MVA,
    harmonic_on=True,
    rng_seed_harmonic=42,
    rng_seed_noise=999,
    add_noise=True
):
    """
    Builds the full measurement vector z = [Vm; Pinj; Qinj; Pf; Qf; Pt; Qt]
    using the harmonic injection model and legacy transducer.
    """
    rng_h = np.random.default_rng(rng_seed_harmonic)
    rng_n = np.random.default_rng(rng_seed_noise)
    nb = bus.shape[0]
    nl = branch.shape[0]

    # 1. Harmonics Setup
    harmonics = [1, 5, 7, 11, 13, 17, 19]
    source_buses = [3]  
    spectrum = ABB_6PULSE_WITH_CHOKE
    
    # 2. Fundamental Solution (Baseline)
    V1 = fundamental_bus_voltages(bus)
    Ybus1 = build_ybus(bus, branch, base_mva)
    I1_net = Ybus1 @ V1
    
    I1_load = fundamental_load_currents(bus, V1, base_mva)
    
    # 3. Harmonic Injections
    if harmonic_on:
        src_idx = [b - 1 for b in source_buses]
        Iinj_unit = make_harmonic_current_injections(nb, src_idx, spectrum, I1_load, rng_h, inj_scale=1.0)
        
        V_unit = solve_all_harmonics(bus, branch, harmonics, Iinj_unit, base_mva, slack_bus=0)
        
        thd_unit = h_ver.voltage_thd(V_unit, 2, harmonics) 
        target_thd = 0.10
        inj_scale = 0.0 if thd_unit < 1e-12 else (target_thd / thd_unit)
        
        Iinj_by_h = {h: inj_scale * Ivec for h, Ivec in Iinj_unit.items()}
    else:
        Iinj_by_h = {}
        for h in harmonics:
            if h != 1:
                Iinj_by_h[h] = np.zeros(nb, dtype=complex)

    Iinj_by_h[1] = I1_net

    # 4. Solve all harmonics
    V_by_h = solve_all_harmonics(bus, branch, harmonics, Iinj_by_h, base_mva, slack_bus=0)
    
    # 5. Calculate Branch Currents for all h
    If_by_h = {}
    It_by_h = {}
    
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
            tap_ratio = float(br[8])
            tap_angle = float(br[9])
            tap = h_ver._tap_complex(tap_ratio, tap_angle)
            
            y = 0.0 if (r==0 and x==0) else 1.0/complex(r,x)
            ysh = 1j * b / 2.0
            
            Yff = (y + ysh) / (tap * np.conj(tap))
            Yft = -y / np.conj(tap)
            Ytt = (y + ysh)
            Ytf = -y / tap
            
            If_vec[k] = Yff * Vh[f] + Yft * Vh[t]
            It_vec[k] = Ytf * Vh[f] + Ytt * Vh[t]
            
        If_by_h[h] = If_vec
        It_by_h[h] = It_vec

    # 6. Apply Transducer
    transducer = LegacyAnalogWattVarTransducer(f0_hz=60.0)
    
    z_Vm = []
    z_Pinj = []
    z_Qinj = []
    z_Pf = []
    z_Qf = []
    z_Pt = []
    z_Qt = []
    
    for i in range(nb):
        sum_sq = 0.0
        for h in harmonics:
            sum_sq += abs(V_by_h[h][i])**2
        v_rms_pu = math.sqrt(sum_sq)
        
        noise = rng_n.normal(0, 0.001) if add_noise else 0.0
        meas = v_rms_pu + noise
        z_Vm.append(meas)
        
    for i in range(nb):
        Vpoint = {}
        Ipoint = {}
        for h in harmonics:
            v_val = V_by_h.get(h, 0j)
            i_val = Iinj_by_h.get(h, 0j)
            Vpoint[h] = v_val[i] if hasattr(v_val, '__getitem__') else v_val
            Ipoint[h] = i_val[i] if hasattr(i_val, '__getitem__') else i_val
        
        P_pu, Q_pu = transducer.measure_PQ_pu(Vpoint, Ipoint)
        
        noise_p = rng_n.normal(0, 1.0) if add_noise else 0.0
        noise_q = rng_n.normal(0, 1.0) if add_noise else 0.0
        
        P_meas = P_pu * base_mva + noise_p
        Q_meas = Q_pu * base_mva + noise_q
        
        z_Pinj.append(P_meas / base_mva)
        z_Qinj.append(Q_meas / base_mva)
        
    for k in range(nl):
        f = int(branch[k, 0]) - 1
        t = int(branch[k, 1]) - 1
        
        # From side
        Vpoint_f = {}
        Ipoint_f = {}
        for h in harmonics:
            v_val = V_by_h.get(h, 0j)
            i_val = If_by_h.get(h, 0j)
            Vpoint_f[h] = v_val[f] if hasattr(v_val, '__getitem__') else v_val
            Ipoint_f[h] = i_val[k] if hasattr(i_val, '__getitem__') else i_val
            
        Pf_pu, Qf_pu = transducer.measure_PQ_pu(Vpoint_f, Ipoint_f)
        
        noise_p = rng_n.normal(0, 1.0) if add_noise else 0.0
        noise_q = rng_n.normal(0, 1.0) if add_noise else 0.0
        
        Pf_meas = Pf_pu * base_mva + noise_p
        Qf_meas = Qf_pu * base_mva + noise_q
        z_Pf.append(Pf_meas / base_mva)
        z_Qf.append(Qf_meas / base_mva)
        
        # To side
        Vpoint_t = {}
        Ipoint_t = {}
        for h in harmonics:
            v_val = V_by_h.get(h, 0j)
            i_val = It_by_h.get(h, 0j)
            Vpoint_t[h] = v_val[t] if hasattr(v_val, '__getitem__') else v_val
            Ipoint_t[h] = i_val[k] if hasattr(i_val, '__getitem__') else i_val
            
        Pt_pu, Qt_pu = transducer.measure_PQ_pu(Vpoint_t, Ipoint_t)
        
        noise_p = rng_n.normal(0, 1.0) if add_noise else 0.0
        noise_q = rng_n.normal(0, 1.0) if add_noise else 0.0
        
        Pt_meas = Pt_pu * base_mva + noise_p
        Qt_meas = Qt_pu * base_mva + noise_q
        z_Pt.append(Pt_meas / base_mva)
        z_Qt.append(Qt_meas / base_mva)
    
    z_full = z_Vm + z_Pinj + z_Qinj + z_Pf + z_Qf + z_Pt + z_Qt
    return z_full

def run_wls(z, case_name='case14'):
    import matlab.engine
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    print("MATLAB engine started.")
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    eng.addpath(eng.genpath(os.path.join(repo_root, 'mcp_server')), nargout=0)
    eng.addpath(eng.genpath(os.path.join(repo_root, 'Transmission')), nargout=0)
    
    eng.workspace['CaseName'] = case_name
    eng.eval("mpc = loadcase(CaseName);", nargout=0)
    eng.eval("bus_data = mpc.bus;", nargout=0)
    
    # Check expected length
    nb = int(eng.eval("size(mpc.bus, 1)"))
    nl = int(eng.eval("size(mpc.branch, 1)"))
    expected_len = 3*nb + 4*nl
    if len(z) != expected_len:
        print(f"Error: z length {len(z)} does not match expected {expected_len}")
        return False, [], []

    z_mat = matlab.double(z)
    eng.workspace['Z'] = z_mat
    
    print("Running WLS...")
    eng.eval("[lambdaN, success, r, lambda_vec, ea] = LagrangianM_singlephase(Z, mpc, 0, bus_data);", nargout=0)
    
    success = eng.workspace['success']
    try:
        r = np.array(eng.workspace['r']).flatten()
        lambdaN = np.array(eng.workspace['lambdaN']).flatten()
    except Exception as e:
        print(f"Error retrieving r/lambdaN: {e}")
        r = []
        lambdaN = []

    
    print(f"WLS Success: {success}")
    if len(r) > 0:
        max_abs_r = np.max(np.abs(r))
        # Calculate Global Chi-Square Statistic J
        # r are normalized residuals (r_i = residual / std_dev)
        J = float(np.sum(r**2))
        print(f"Max |Residual|: {max_abs_r}")
        print(f"Global Chi-Square J: {J}")
    else:
        J = 0.0

    eng.quit()
    return success, r, lambdaN, J

if __name__ == "__main__":
    try:
        # --- Signal-to-Noise Check ---
        print("\n--- Signal-to-Noise Analysis ---")
        # Same seed for both ensures alignment, add_noise=False ignores noise seed
        z0 = np.array(build_full_harmonic_z(harmonic_on=False, rng_seed_harmonic=42, rng_seed_noise=100, add_noise=False)) 
        zH = np.array(build_full_harmonic_z(harmonic_on=True,  rng_seed_harmonic=42, rng_seed_noise=100, add_noise=False)) 
        dz = zH - z0
        
        nb = 14
        nl = 20
        idx_Vm = slice(0, nb)
        idx_Pinj = slice(nb, 2*nb)
        idx_Qinj = slice(2*nb, 3*nb)
        idx_Pf = slice(3*nb, 3*nb+nl)
        
        print("Max |Δz| from Harmonics (Noiseless):")
        print(f"  Vm:   {np.max(np.abs(dz[idx_Vm])):.6f} p.u.")
        print(f"  Pinj: {np.max(np.abs(dz[idx_Pinj])):.6f} p.u.")
        print(f"  Qinj: {np.max(np.abs(dz[idx_Qinj])):.6f} p.u.")
        print(f"  Pf:   {np.max(np.abs(dz[idx_Pf])):.6f} p.u.")

        # --- Case A: No harmonics, no bad data ---
        print("\n--- Case A: No harmonics, no bad data ---")
        # Use specific noise seed
        zA = build_full_harmonic_z(harmonic_on=False, rng_seed_harmonic=42, rng_seed_noise=100, add_noise=True)
        succA, rA, _, JA = run_wls(zA)
        max_rA = np.max(np.abs(rA)) if len(rA) else 0.0
        print(f"A max|r| = {max_rA}")
        print(f"A J = {JA}")

        # --- Case B: Harmonics (5% THD), no bad data ---
        print("\n--- Case B: Harmonics (5% THD), no bad data ---")
        # Use SAME noise seed as A
        zB = build_full_harmonic_z(harmonic_on=True, rng_seed_harmonic=42, rng_seed_noise=100, add_noise=True)
        succB, rB, _, JB = run_wls(zB)
        max_rB = np.max(np.abs(rB)) if len(rB) else 0.0
        print(f"B max|r| = {max_rB}")
        print(f"B J = {JB}")

        # --- Case C: No harmonics + injected bad data ---
        print("\n--- Case C: No harmonics + injected bad data ---")
        zC = list(zA)
        bad_idx = 3*nb + 0 
        zC[bad_idx] += 0.1 # 10 sigma error
        
        succC, rC, _, JC = run_wls(zC)
        max_rC = np.max(np.abs(rC)) if len(rC) else 0.0
        print(f"C max|r| = {max_rC}")
        print(f"C J = {JC}")
        anomalies_C = np.where(np.abs(rC) > 3.0)[0]
        if bad_idx in anomalies_C:
            print("SUCCESS: Injected bad data detected.")

        # --- Case D: Harmonics + injected bad data ---
        print("\n--- Case D: Harmonics + injected bad data ---")
        zD = list(zB)
        zD[bad_idx] += 0.1
        succD, rD, _, JD = run_wls(zD)
        max_rD = np.max(np.abs(rD)) if len(rD) else 0.0
        print(f"D max|r| = {max_rD}")
        print(f"D J = {JD}")
        
        # Save Summary Results
        summary = {
            "signal_to_noise": {
                "max_delta_Vm": float(np.max(np.abs(dz[idx_Vm]))),
                "max_delta_Pf": float(np.max(np.abs(dz[idx_Pf])))
            },
            "Case_A_check": {"max_r": float(max_rA), "J": float(JA)},
            "Case_B_harmonics": {"max_r": float(max_rB), "J": float(JB)},
            "Case_C_bad_data": {"max_r": float(max_rC), "J": float(JC)},
            "Case_D_combined": {"max_r": float(max_rD), "J": float(JD)}
        }
        with open("wls_verification_suite_results.json", "w") as f:
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
