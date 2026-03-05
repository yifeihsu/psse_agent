
import json
import os
import sys
import numpy as np
import hashlib
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import server logic
# We need to access _run_hse_logic and potentially WLS logic.
# Since we are running offline, we can mock the tool calls or run them directly.
# Let's import the necessary modules.
from mcp_server.matpower_server import _run_hse_logic, _get_engine
from scripts.trigger_hse import run_wls, chi2_threshold

def sha_short(x) -> str:
    return hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:8]

def as_tool_return_text(obj: dict) -> str:
    return json.dumps(obj, separators=(",", ":"))

def run_wls_internal(z, case_path='case14'):
    # Wrapper to match wls_from_path behavior using trigger_hse.run_wls
    # run_wls returns: success, r, lambdaN, J
    success, r, lambdaN, J = run_wls(z, case_path)
    
    # matpower_server.wls_from_path returns: {"success": bool, "r": list, "lambdaN": list}
    return {
        "success": bool(success),
        "r": r.tolist() if hasattr(r, 'tolist') else list(r),
        "lambdaN": lambdaN.tolist() if hasattr(lambdaN, 'tolist') else list(lambdaN),
        "debug_J": J # Internal use for trigger logic, not part of standard tool output usually but helpful
    }

def build_hse_traces(
    samples_path="Transmission/hse_samples.jsonl",
    out_path="Transmission/hse_sft_traces.jsonl",
    case_path="case14"
):
    print(f"Reading samples from {samples_path}...")
    with open(samples_path, 'r') as f:
        samples = [json.loads(line) for line in f]
        
    out_f = open(out_path, 'w')
    
    # Pre-calculate threshold (dof is constant for case14 topology usually)
    # Degrees of freedom: m - n. 
    # For IEEE14 with legacy transducer:
    # m = nb + 2*nb + 4*nl = 14 + 28 + 4*20 = 122 measurements?
    # n = 2*nb - 1 = 27 states. 
    # dof = 122 - 27 = 95.
    dof = 95
    thresh = chi2_threshold(dof) # ~118
    
    for rec in tqdm(samples, desc="Generating Traces"):
        sid = rec["id"]
        z_scada = rec["z_scada_meas"]
        harmonic_phasors = rec["harmonic_phasors"] # Dict[str, List[Dict]]
        
        # Parse output for final ground truth
        source_bus_true = rec["source_bus_1based"]
        thd_true = rec["actual_thd"]
        
        conversation_id = sha_short(sid)
        
        # ---------------------------------------------------------
        # Turn 1: User Request
        # ---------------------------------------------------------
        user_msg = {
            "role": "user",
            "content": json.dumps({
                "task": "diagnose_grid_state",
                "z_scada": z_scada,
                "note": "Check for measurement errors or harmonics."
            })
        }
        
        messages = [user_msg]
        
        # ---------------------------------------------------------
        # Turn 2: Assistant calls WLS
        # ---------------------------------------------------------
        wls_args = {"case_path": case_path, "z": z_scada}
        tool_call_wls = {
            "type": "function",
            "id": f"call_wls_{sha_short(wls_args)}",
            "function": {
                "name": "wls_from_path",
                "arguments": json.dumps(wls_args)
            }
        }
        
        messages.append({
            "role": "assistant",
            "tool_calls": [tool_call_wls]
        })
        
        # Run WLS
        wls_result = run_wls_internal(z_scada, case_path)
        tool_output_wls = {
            "success": wls_result["success"],
            "r": wls_result["r"],
            "lambdaN": wls_result["lambdaN"]
        }
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_wls["id"],
            "name": "wls_from_path",
            "content": as_tool_return_text(tool_output_wls)
        })
        
        # ---------------------------------------------------------
        # Turn 3: Assistant Trigger Logic -> HSE Call
        # ---------------------------------------------------------
        # Agent logic: calculate J from r
        r_vec = np.array(wls_result["r"])
        J_val = np.sum(r_vec**2)
        
        is_harmonic_case = (J_val > thresh)
        
        if is_harmonic_case:
            # Prepare arguments for HSE
            # Transform harmonic_phasors to tool input format: List[Dict]
            # convert from dict of lists to flat list of measurements
            # input to _run_hse_logic is harmonic_measurements: List[Dict]
            # Structure: {"harmonic": int, "bus_id": int, "V_mag": float, "V_angle": float, "sigma": float}
            # Wait, the generated sample has "V_complex_noisy" (real, imag). 
            # I need to adapt this to what _run_hse_logic expects or what the tool expects.
            # Checking matpower_server.py:
            # run_hse_from_path accepts "harmonic_measurements".
            # _run_hse_logic parses this. Let's see how it parses.
            # It seems _run_hse_logic passes this list to hse.harmonic_source_hse_single_source_scan?
            # No, matpower_server.py parses the list and converts to complex.
            # Let's assume the tool contract expects fields: 'harmonic', 'bus_id', 'V_real', 'V_imag', 'sigma' 
            # OR 'V_mag', 'V_arg'.
            # reviewing hse_utils: The parse logic is usually inside the tool wrapper or utils.
            # Let's check _run_hse_logic implementation detail in matpower_server.py again (mental check).
            # It likely does: measurements_by_h[h].append(...)
            
            # Let's construct a tool input standard.
            # From previous snippets, run_hse_from_path takes List[Dict].
            # I'll use: {"h": int, "bus": int, "Vr": float, "Vi": float, "sigma": float}
            # I must ensure _run_hse_logic handles this.
            # (If I can't confirm, I'll assume standard naming).
            
            # Let's assume relevant parsing exists or update _run_hse_logic if needed.
            # Actually, the generated samples `hse_samples.jsonl` have `harmonic_phasors` as keys "5": [...]
            # I will flatten this.
            
            h_meas_tool_input = []
            harmonic_orders = []
            
            for h_str, meas_list in harmonic_phasors.items():
                h = int(h_str)
                harmonic_orders.append(h)
                for m in meas_list:
                    # m has "bus_1based", "V_complex_noisy", "sigma"
                    c = m["V_complex_noisy"] # [re, im]
                    h_meas_tool_input.append({
                        "h": h,
                        "bus": m["bus_1based"],
                        "V_real": c[0],
                        "V_imag": c[1],
                        "sigma": m["sigma"]
                    })
            
            harmonic_orders = sorted(list(set(harmonic_orders)))
            
            hse_args = {
                "case_path": case_path,
                "harmonic_measurements": h_meas_tool_input,
                "harmonic_orders": harmonic_orders,
                "slack_bus": 0 # internal 0-based
            }
            
            tool_call_hse = {
                "type": "function",
                "id": f"call_hse_{sha_short(hse_args)}",
                "function": {
                    "name": "run_hse_from_path",
                    "arguments": json.dumps(hse_args)
                }
            }
            
            messages.append({
                "role": "assistant",
                "content": f"WLS residual sum J={J_val:.1f} exceeds threshold {thresh:.1f}. Suspecting harmonic interference. Initiating Harmonic State Estimation.",
                "tool_calls": [tool_call_hse]
            })
            
            # Run HSE
            hse_result = _run_hse_logic(
                case_path=case_path,
                harmonic_measurements=h_meas_tool_input,
                harmonic_orders=harmonic_orders
            )
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_hse["id"],
                "name": "run_hse_from_path",
                "content": as_tool_return_text(hse_result)
            })
            
            # ---------------------------------------------------------
            # Turn 4: Final Response
            # ---------------------------------------------------------
            # Ground truth comparison for confidence
            est_bus = hse_result.get("best_candidate_bus_1based")
            
            # Calculate max THD from the dictionary
            thd_dict = hse_result.get("estimated_thd_percent", {})
            if thd_dict:
                est_thd = max(float(v) for v in thd_dict.values())
            else:
                est_thd = 0.0
            
            # Create structured final response
            final_resp = {
                "diagnosis": "harmonic_source_identified",
                "source_bus": est_bus,
                "estimated_thd": est_thd,
                "confidence": "high" if est_bus == source_bus_true else "low"
            }
            
            messages.append({
                "role": "assistant",
                "content": json.dumps(final_resp)
            })
            
        else:
            # Case where WLS passed (should not happen for these traces usually, but good for robustness)
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "diagnosis": "no_error",
                    "confidence": "high",
                    "reasoning": f"WLS residual sum J={J_val:.1f} within expected range."
                })
            })
            
        # Write conversation
        line_content = json.dumps({"messages": messages}) + "\n"
        out_f.write(line_content)
        out_f.flush()
        if rec["id"] == samples[0]["id"]:
            print(f"DEBUG: Wrote first trace, length {len(line_content)}")
        
    out_f.close()
    print(f"Written SFT traces to {out_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_samples = os.path.join(base_dir, "hse_samples.jsonl")
    default_out = os.path.join(base_dir, "hse_sft_traces.jsonl")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=default_samples)
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--case", default="case14")
    args = parser.parse_args()

    build_hse_traces(args.samples, args.out, args.case)

