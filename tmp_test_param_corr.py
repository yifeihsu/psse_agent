import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mcp_server')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Transmission')))

from matpower_server import _param_correction_json
from generate_measurements import load_case, solve_ac_opf, compute_measurements_pu, make_index_map, base_gaussian_noise

def run_test():
    print("=== Testing pure Python _param_correction_json ===")
    
    # 1. Load the base case
    try:
        ppc_base = load_case("14")
    except Exception as e:
        print("Failed to load case 14:", e)
        return
        
    nb = ppc_base["bus"].shape[0]
    nl = ppc_base["branch"].shape[0]
    idx_map = make_index_map(nb, nl)
    
    # 2. Select a line to corrupt: Line index 0 (1-based -> 1, between bus 1 and 2)
    target_line_0based = 0
    target_line_1based = 1
    
    true_R = ppc_base["branch"][target_line_0based, 2]
    true_X = ppc_base["branch"][target_line_0based, 3]
    
    print(f"Target Line {target_line_1based}: True R={true_R:.5f}, True X={true_X:.5f}")
    
    # Generate 3 "true" scans (different OPF conditions) by perturbing loads slightly
    n_scans = 3
    z_scans = []
    initial_states = []
    
    rng = np.random.default_rng(42)
    sigmas = {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2}
    
    for k in range(n_scans):
        # Perturb loads randomly by +/- 20%
        ppc_scan = {}
        for key in ppc_base:
            if isinstance(ppc_base[key], np.ndarray):
                ppc_scan[key] = np.copy(ppc_base[key])
            else:
                ppc_scan[key] = ppc_base[key]
                
        ppc_scan["bus"][:, 2] *= rng.uniform(0.8, 1.2, size=nb) # Pd
        ppc_scan["bus"][:, 3] *= rng.uniform(0.8, 1.2, size=nb) # Qd
        
        solved = solve_ac_opf(ppc_scan)
        z_true = compute_measurements_pu(solved)
        
        # Add realistic noise
        z_obs = z_true + base_gaussian_noise(z_true, idx_map, sigmas, rng)
        z_scans.append(z_obs.tolist())
        
        # Record initial states for the solver (we'll just use the solved states + a tiny bit of noise as a starting point)
        v_mag = solved["bus"][:, 7]
        v_ang_deg = solved["bus"][:, 8]
        # Add small perturbations to initial state guess so the solver has to do *some* work
        v_mag_guess = v_mag + rng.normal(0, 0.01, size=nb)
        v_ang_deg_guess = v_ang_deg + rng.normal(0, 1.0, size=nb)
        initial_states.append(np.r_[v_mag_guess, v_ang_deg_guess].tolist())

    # 3. Corrupt the base case that we will feed to the estimator
    ppc_corrupted = {"baseMVA": ppc_base["baseMVA"], "bus": np.copy(ppc_base["bus"]), "branch": np.copy(ppc_base["branch"]), "gen": np.copy(ppc_base["gen"])}
    wrong_R = true_R * 2.5
    wrong_X = true_X * 0.5
    ppc_corrupted["branch"][target_line_0based, 2] = wrong_R
    ppc_corrupted["branch"][target_line_0based, 3] = wrong_X
    print(f"Corrupted Initial Guess passed to estimator: R={wrong_R:.5f}, X={wrong_X:.5f}")
    
    # Save corrupted case to a temporary file
    import tempfile
    def _write_mpc_to_file(mpc_dict, filepath):
        with open(filepath, 'w') as f:
            f.write("function mpc = case_temp\n")
            f.write(f"mpc.baseMVA = {mpc_dict['baseMVA']};\n")
            
            f.write("mpc.bus = [\n")
            for row in mpc_dict["bus"]:
                f.write("  " + " ".join(f"{x:.6f}" for x in row) + ";\n")
            f.write("];\n")
            
            f.write("mpc.branch = [\n")
            for row in mpc_dict["branch"]:
                f.write("  " + " ".join(f"{x:.6f}" for x in row) + ";\n")
            f.write("];\n")
            
            f.write("mpc.gen = [\n")
            for row in mpc_dict["gen"]:
                f.write("  " + " ".join(f"{x:.6f}" for x in row) + ";\n")
            f.write("];\n")
            
    temp_case = tempfile.NamedTemporaryFile(suffix=".m", delete=False)
    temp_case.close()
    _write_mpc_to_file(ppc_corrupted, temp_case.name)
    
    print("\nExecuting Python Multi-Scan Parameter Estimator...")
    try:
        res = _param_correction_json(
            eng=None, 
            case_path=temp_case.name, 
            line_index=target_line_1based,
            z_scans=z_scans,
            initial_states=initial_states
        )
        
        print(f"Success: {res.get('success')}")
        if res.get('success'):
            r_est, x_est = res.get('corrected_params', [0, 0])
            print(f"Estimated Corrections: R={r_est:.5f}, X={x_est:.5f}")
            print(f"Absolute Error: dR={abs(true_R - r_est):.5f}, dX={abs(true_X - x_est):.5f}")
        else:
            print("Estimator failed:", res.get('error'))
            
    except Exception as e:
        print("Error during execution:", e)
        import traceback
        traceback.print_exc()
        
    finally:
        os.remove(temp_case.name)

if __name__ == "__main__":
    run_test()
