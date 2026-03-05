
import sys
import os
import json
import numpy as np
import requests
import pandapower as pp
from pypower.api import case14

# Add repo root to path
sys.path.append(os.getcwd())

from Transmission import nodebreaker_pp14 as nb
from Transmission.generate_measurements import _nb_to_operator_z
from Transmission.nb_to_matpower import build_corrected_busbranch_and_export

endpoint = "http://127.0.0.1:3929/tools"

def mcp_call(name: str, args: dict) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    headers = {"Accept": "application/json, text/event-stream"}
    r = requests.post(endpoint, json=payload, headers=headers, timeout=90)
    r.raise_for_status()
    res = r.json().get("result", {})
    sc = res.get("structuredContent")
    if isinstance(sc, dict):
        return sc
    # Fallback parsing
    texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
    for t in texts:
        try:
            return json.loads(t)
        except:
            pass
    return {}

def verify_local():
    cb_name = 'CB_2R2_2R3'
    desired_status = False # Open -> Topology Error
    
    print(f"Generating z_corr locally for {cb_name} -> {desired_status}...")
    
    # 1. Build NB net locally (uses updated nodebreaker_pp14 with taps)
    status_map = {cb_name: desired_status}
    try:
        net, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123(status_map=status_map)
    except Exception as e:
        print(f"Error building NB net: {e}")
        import traceback
        with open("traceback.txt", "w") as f:
            traceback.print_exc(file=f)
        return
    
    # 2. Run PF
    pp.runpp(net, calculate_voltage_angles=True, init="dc")
    
    # 3. Generate z_corr
    ppc_base = case14()
    z_corr = _nb_to_operator_z(net, line_idx, trafo_idx, ppc_base)
    print(f"Generated z_corr with length {len(z_corr)}")
    
    # --- Compare with Standard Case14 Measurements ---
    from pypower.api import runpf, ppoption
    from Transmission.generate_measurements import compute_measurements_pu
    
    ppc_std = case14()
    opt = ppoption(VERBOSE=0, OUT_ALL=0)
    res_std, success = runpf(ppc_std, opt)
    if success:
        z_std = compute_measurements_pu(res_std)
        print(f"\n--- Comparing z_corr vs z_std (Length {len(z_std)}) ---")
        
        # z structure: Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)
        n_bus = res_std['bus'].shape[0]
        n_branch = res_std['branch'].shape[0]
        
        diff = np.abs(z_corr - z_std)
        max_diff = np.max(diff)
        max_idx = np.argmax(diff)
        print(f"Max difference: {max_diff:.4f} at index {max_idx}")
        
        # Identify what max_idx corresponds to
        if max_idx < n_bus:
            print(f"Max diff is Vm at Bus {max_idx+1}")
        elif max_idx < 2*n_bus:
            print(f"Max diff is Pinj at Bus {max_idx - n_bus + 1}")
        elif max_idx < 3*n_bus:
            print(f"Max diff is Qinj at Bus {max_idx - 2*n_bus + 1}")
        elif max_idx < 3*n_bus + n_branch:
            print(f"Max diff is Pf at Branch {max_idx - 3*n_bus}")
        elif max_idx < 3*n_bus + 2*n_branch:
            print(f"Max diff is Qf at Branch {max_idx - 3*n_bus - n_branch}")
        elif max_idx < 3*n_bus + 3*n_branch:
            print(f"Max diff is Pt at Branch {max_idx - 3*n_bus - 2*n_branch}")
        else:
            print(f"Max diff is Qt at Branch {max_idx - 3*n_bus - 3*n_branch}")

        print(f"z_corr[{max_idx}] = {z_corr[max_idx]:.4f}")
        print(f"z_std[{max_idx}] = {z_std[max_idx]:.4f}")

        if max_diff > 0.01:
            print("Large differences found:")
            # Vm
            for i in range(n_bus):
                if diff[i] > 0.01:
                    print(f"Vm Bus {i+1}: {z_corr[i]:.4f} vs {z_std[i]:.4f} (diff {diff[i]:.4f})")
            
            # Specific check for Bus 6
            print(f"Vm Bus 6: {z_corr[5]:.4f} vs {z_std[5]:.4f}")
            print(f"Qinj Bus 6: {z_corr[5+2*n_bus]:.4f} vs {z_std[5+2*n_bus]:.4f}")
            
            offset = n_bus
            # Pinj
            for i in range(n_bus):
                if diff[offset+i] > 0.1:
                    print(f"Pinj Bus {i+1}: {z_corr[offset+i]:.4f} vs {z_std[offset+i]:.4f} (diff {diff[offset+i]:.4f})")
            
            offset += n_bus
            # Qinj
            for i in range(n_bus):
                if diff[offset+i] > 0.1:
                    print(f"Qinj Bus {i+1}: {z_corr[offset+i]:.4f} vs {z_std[offset+i]:.4f} (diff {diff[offset+i]:.4f})")
            
            offset += n_bus
            # Flows
            # Pf
            for i in range(n_branch):
                if diff[offset+i] > 0.1:
                    print(f"Pf Branch {i}: {z_corr[offset+i]:.4f} vs {z_std[offset+i]:.4f} (diff {diff[offset+i]:.4f})")
            
            offset += n_branch
            # Qf
            for i in range(n_branch):
                if diff[offset+i] > 0.1:
                    print(f"Qf Branch {i}: {z_corr[offset+i]:.4f} vs {z_std[offset+i]:.4f} (diff {diff[offset+i]:.4f})")
            
            offset += n_branch
            # Pt
            for i in range(n_branch):
                if diff[offset+i] > 0.1:
                    print(f"Pt Branch {i}: {z_corr[offset+i]:.4f} vs {z_std[offset+i]:.4f} (diff {diff[offset+i]:.4f})")

            offset += n_branch
            # Qt
            for i in range(n_branch):
                if diff[offset+i] > 0.1:
                    print(f"Qt Branch {i}: {z_corr[offset+i]:.4f} vs {z_std[offset+i]:.4f} (diff {diff[offset+i]:.4f})")
    
    # 4. Build Corrected Case (uses updated nb_to_matpower with taps)
    filename_mat = "Transmission/case14_topology_verify_local.mat"
    build_corrected_busbranch_and_export(cb_name, desired_status=desired_status, filename_mat=filename_mat)
    
    # 5. Run WLS using the server (but passing our correct z and correct case)
    print("Running WLS via server...")
    # Try passing absolute path
    abs_path = os.path.abspath(filename_mat)
    wls = mcp_call("wls_from_path", {"case_path": abs_path, "z": z_corr.tolist()})
    
    print("WLS Response Keys:", wls.keys())
    if "error" in wls:
        print("WLS Error:", wls["error"])
    
    r = np.array(wls.get("r", []), float)
    if r.size:
        print(f"WLS success: {wls.get('success')}")
        print(f"Max |r_norm|: {np.nanmax(np.abs(r)):.3f}")
    else:
        print("WLS returned no residuals.")
        print(wls)

if __name__ == "__main__":
    verify_local()
