import sys
import os
import json
import numpy as np

# Add paths to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mcp_server')))

from scripts.trigger_hse import (
    build_full_harmonic_z, 
    simulate_harmonic_voltage_meter_measurements, 
    BUS, BRANCH, BASE_MVA
)
from mcp_server.matpower_server import _run_hse_logic

def test_hse_integration():
    print("Generating test data from trigger_hse logic...")
    # 1. Generate ground truth
    z, V_by_h_true, Iinj_by_h_true = build_full_harmonic_z(
        harmonic_on=True,
        rng_seed_harmonic=42,
        rng_seed_noise=100,
        add_noise=True,
        return_harmonic_states=True
    )
    
    harmonic_orders = [5, 7, 11, 13, 17, 19]
    meter_buses = [2, 3, 4, 5, 9, 14]  # 1-based
    
    Vh_meas_by_h = simulate_harmonic_voltage_meter_measurements(
        V_by_h_true,
        meter_buses_1based=meter_buses,
        harmonic_orders=harmonic_orders,
        sigma_v=5e-4,
        rng_seed=2025
    )
    
    # 2. Convert to tool input format
    harmonic_measurements = []
    for h, (buses0, Vmeas, sigma) in Vh_meas_by_h.items():
        for i, b0 in enumerate(buses0):
            v = Vmeas[i]
            rec = {
                "h": int(h),
                "bus": int(b0) + 1, # 1-based
                "V_real": float(v.real),
                "V_imag": float(v.imag),
                "sigma": float(sigma[i])
            }
            harmonic_measurements.append(rec)
            
    print(f"Prepared {len(harmonic_measurements)} measurements.")
    
    # 3. Call the internal logic directly
    print("Calling _run_hse_logic...")
    
    result = _run_hse_logic(
        case_path='case14',
        harmonic_measurements=harmonic_measurements,
        harmonic_orders=harmonic_orders,
        slack_bus=0
    )
    
    # 4. Verification
    if not result.get("success"):
        print("Tool failed:", result.get("error"))
        print(result.get("traceback"))
        sys.exit(1)
        
    best_candidate = result.get("best_candidate_bus_1based")
    print(f"Tool identified best bus: {best_candidate}")
    
    top3 = result.get("ranking_top10", [])[:3]
    print("Top 3 candidates:", top3)
    
    expected_source = 3
    if best_candidate == expected_source:
        print("SUCCESS: Tool correctly identified Bus 3.")
    else:
        print(f"FAILURE: Expected {expected_source}, got {best_candidate}")
        sys.exit(1)

    # Check THD
    est_thd = result.get("estimated_thd_percent", {})
    thd_bus3 = est_thd.get("3")
    print(f"Estimated THD at Bus 3: {thd_bus3:.2f}%")
    if abs(thd_bus3 - 9.98) > 0.1:
         print("WARNING: THD estimate mismatch (expected ~9.98%)")

if __name__ == "__main__":
    test_hse_integration()
