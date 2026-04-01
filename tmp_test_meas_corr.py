import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mcp_server')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Transmission')))

from matpower_server import _meas_correction_json, _wls_json
from generate_measurements import load_case, solve_ac_opf, compute_measurements_pu, make_index_map, base_gaussian_noise

def run_test():
    print("=== Testing pure Python _meas_correction_json with Noise ===")\
    
    try:
        ppc_base = load_case("14")
    except Exception as e:
        print("Failed to load case 14:", e)
        return
        
    solved = solve_ac_opf(ppc_base)
    z_true = compute_measurements_pu(solved)
    
    nb = ppc_base["bus"].shape[0]
    nl = ppc_base["branch"].shape[0]
    idx_map = make_index_map(nb, nl)
    
    # Apply realistic Gaussian noise
    rng = np.random.default_rng(123)
    sigmas = {"vm": 1e-3, "inj": 1e-2, "flow": 1e-2}
    z_obs = z_true + base_gaussian_noise(z_true, idx_map, sigmas, rng)
    
    # Inject a large gross error into measurement index 10
    z_obs[10] += 0.5
    
    z_list = z_obs.tolist()
    case_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mcp_server', 'case14.m'))
    
    print("\n--- 1. Testing WLS BEFORE Correction ---")
    try:
        res_uncorrected = _wls_json(None, case_path, z_list)
        r_uncorrected = np.array(res_uncorrected['r'])
        print(f"Max r_norm (Uncorrected): {np.max(np.abs(r_uncorrected)):.6f}")
    except Exception as e:
        print("Failed to run uncorrected WLS:", e)

    print("\n--- 2. Testing WLS AFTER Correction ---")
    print("Calling _meas_correction_json with suspect_group=[11]...")
    try:
        res = _meas_correction_json(
            eng=None, 
            case_path=case_path, 
            z_list=z_list,
            suspect_group=[11], # 1-based index corresponding to Python 0-based index 10
            enable_correction=True,
            max_correction_iterations=2,
            error_tolerance=1e-3
        )
        
        print(f"Success: {res.get('success')}")
        print(f"Applied Correction: {res.get('applied_any_correction')}")
        if 'r_norm' in res:
            r = np.array(res['r_norm'])
            print(f"Max r_norm (Corrected): {np.max(r):.6f}")
        
        
        print("\n--- 3. Testing WLS with CORRECTED Values ---")
        if res.get('success') and 'corrected_measurements' in res:
            # Overwrite the original z_list with the corrected estimations
            for item in res['corrected_measurements']:
                idx = item['index0']
                corrected_val = item['corrected']
                z_list[idx] = corrected_val
                
            res_final = _wls_json(None, case_path, z_list)
            if res_final.get('success'):
                r_final = np.array(res_final['r'])
                print(f"Max r_norm (Final System): {np.max(np.abs(r_final)):.6f}")
                print(f"Mean r_norm (Final System): {np.mean(np.abs(r_final)):.6f}")
            else:
                print("Failed to run final WLS:", res_final.get('error'))
            
    except Exception as e:
        print("Error during _meas_correction_json:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
