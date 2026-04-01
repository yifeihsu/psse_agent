import sys
import os
import json
import numpy as np

# Load local tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Transmission')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mcp_server')))

from generate_measurements import (
    load_case,
    solve_ac_opf,
    compute_measurements_pu,
    make_index_map,
    base_gaussian_noise
)
from matpower_server import _wls_json

def run_test():
    print("=== Generating 'Noisy' Measurement Series ===")
    
    # 1. Load MATPOWER case and find ground truth
    try:
        ppc_base = load_case("14")
    except Exception as e:
        print("Failed to load case 14 via PYPOWER:", e)
        return
        
    solved = solve_ac_opf(ppc_base)
    if solved is None:
        print("AC OPF failed to solve case 14.")
        return
        
    # 2. Extract measurements (clean, purely mathematical)
    z_true = compute_measurements_pu(solved)
    
    nb = ppc_base["bus"].shape[0]
    nl = ppc_base["branch"].shape[0]
    idx_map = make_index_map(nb, nl)
    
    # 3. Apply base Gaussian noise 
    # (sigma values aligned with WLS weight matrix W: Vm 1e-3; Inj 1e-2; Flow 1e-2)
    rng = np.random.default_rng(42)  # Seed for reproducibility
    sigmas = {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2}
    z_obs = z_true + base_gaussian_noise(z_true, idx_map, sigmas, rng)
    
    z_list = z_obs.tolist()
    print(f"Buses: {nb}, Branches: {nl}, Total Measurements: {len(z_list)}")
    
    # 3. Call Pure Python WLS Migration
    case_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mcp_server', 'case14.m'))
    print(f"\n=== Running Pure Python WLS via _wls_json ===")
    
    try:
        res = _wls_json(None, case_path, z_list)
        
        if not res.get("success"):
            print("WLS Failed!")
            print(res)
            return

        r = np.array(res["r"])
        lambdaN = np.array(res["lambdaN"])
        
        print("\n--- RESULTS ---")
        print("WLS Success: TRUE")
        print(f"Max Normalized Residual (r): {np.max(np.abs(r)):.6f}")
        print(f"Mean Normalized Residual (r): {np.mean(np.abs(r)):.6f}")
        
        print(f"\nMax Normalized Lambda: {np.max(np.abs(lambdaN)):.6f}")
        
        if np.max(np.abs(r)) < 1.0:
            print("\nCONCLUSION: SUCCESS. The pure Python WLS perfectly matches the clean physics model with zero residual errors.")
        else:
            print("\nCONCLUSION: WARNING. Residuals are unexpectedly high for a clean measurement series.")
            
    except Exception as e:
        print("Error during _wls_json:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
