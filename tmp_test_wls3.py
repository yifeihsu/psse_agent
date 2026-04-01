import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mcp_server')))
from matpower_server import _wls_json, _run_hse_logic

print("Testing _wls_json directly...")
import numpy as np
import json

# case14 has 14 buses and 20 lines. Length of z is 3*14 + 4*20 = 122
z = np.ones(122).tolist()
case_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mcp_server', 'case14.m'))

try:
    res = _wls_json(None, case_path, z)
    print("WLS Result:", {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()})
except Exception as e:
    print("WLS Error:", e)

print("\nTesting _run_hse_logic directly...")
try:
    measurements = [
        {"h": 5, "bus": 2, "Vm": 1.0, "Va_deg": 0.0, "sigma": 0.001}
    ]
    res_hse = _run_hse_logic(
        case_path=case_path, 
        harmonic_measurements=measurements, 
        harmonic_orders=[5], 
        slack_bus=0
    )
    print("HSE Result:", res_hse)
except Exception as e:
    print("HSE Error:", e)

    import traceback
    traceback.print_exc()

