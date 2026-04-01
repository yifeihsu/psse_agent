import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mcp_server')))
from matpower_server import wls_from_path, run_hse_from_path

print("Testing pure Python wls_from_path...")
import numpy as np
import json

# case14 has 14 buses and 20 lines. Length of z is 3*14 + 4*20 = 122
z = np.ones(122).tolist()

try:
    res = wls_from_path(case_path="case14", z=z)
    print("WLS Result:", {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()})
except Exception as e:
    print("WLS Error:", e)

print("\nTesting pure Python run_hse_from_path...")
try:
    measurements = [
        {"h": 5, "bus": 2, "Vm": 1.0, "Va_deg": 0.0, "sigma": 0.001}
    ]
    res_hse = run_hse_from_path(
        case_path="case14", 
        harmonic_measurements=measurements, 
        harmonic_orders=[5], 
        slack_bus=0
    )
    print("HSE Result:", res_hse)
except Exception as e:
    print("HSE Error:", e)

