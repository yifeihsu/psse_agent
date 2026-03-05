
import json
import os
import re
import argparse
from typing import Any, Dict, List, Optional
import requests
import numpy as np

try:
    from pypower.api import case14, case118
except ImportError:
    print("pypower not found. Please install it to run this script.")
    exit(1)

# Configuration
ENDPOINT = "http://127.0.0.1:3929/tools"

def mcp_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to call MCP tool via HTTP endpoint."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    headers = {"Accept": "application/json"}
    try:
        r = requests.post(ENDPOINT, json=payload, headers=headers, timeout=120) 
        r.raise_for_status()
        data = r.json()
        
        if isinstance(data, dict) and "error" in data:
             err = data.get("error")
             return {"success": False, "error": str(err)}

        res = data.get("result", {}) if isinstance(data, dict) else {}
        # Parse content
        # print(f"DEBUG RAW: {json.dumps(res, default=str)[:200]}") # UNCOMMENT TO DEBUG
        content_boxes = res.get("content", [])
        for c in content_boxes:
            if c.get("type") == "text":
                text = c.get("text", "")
                # print(f"DEBUG TEXT: {text[:200]}") # UNCOMMENT TO DEBUG
                # Try parsing JSON from text
                try:
                    return json.loads(text)
                except:
                    # heuristic for common json inside markdown
                    m = re.search(r"\{[\s\S]*\}", text)
                    if m:
                        try:
                            return json.loads(m.group(0))
                        except:
                            pass
        return {}
    except Exception as e:
        print(f"MCP Call Failed: {e}")
        return {"success": False, "error": str(e)}

def _load_param_samples(samples_path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("scenario") == "parameter_error":
                    recs.append(rec)
            except:
                pass
    return recs

def _load_base_branch(case_name: str) -> np.ndarray:
    """
    Load base branch matrix from PYPOWER so indexing/ordering matches dataset generation.

    `meta.json` stores case like "case14" / "case118".
    """
    case_key = str(case_name).strip().lower()
    if case_key in ("14", "case14"):
        ppc = case14()
    elif case_key in ("118", "case118"):
        ppc = case118()
    else:
        raise ValueError(f"Unsupported case={case_name!r}; expected 'case14' or 'case118'.")
    return np.array(ppc["branch"], dtype=float)

def main():
    parser = argparse.ArgumentParser(description="Quick check for parameter error correction")
    parser.add_argument("--input-dir", default="out_sft_measurements", help="Directory with samples.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Analyze only first N samples")
    args = parser.parse_args()
    
    samples_path = os.path.join(args.input_dir, "samples.jsonl")
    meta_path = os.path.join(args.input_dir, "meta.json")
    
    if not os.path.exists(samples_path):
        print(f"Error: {samples_path} not found.")
        return

    # Load base case name from meta
    case_name = "case14"
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            case_name = meta.get("case", "case14")
        except:
            pass

    print(f"Loading base branch data for {case_name} using PYPOWER...")
    # NOTE: This matches Transmission/generate_measurements.py ordering/values.
    base_branch = _load_base_branch(case_name)
    
    print(f"Loading parameter samples from {samples_path}...")
    recs = _load_param_samples(samples_path)
    print(f"Found {len(recs)} parameter_error samples. Analyzing first {args.limit}...")
    
    success_count = 0
    total_processed = 0

    for i, rec in enumerate(recs):
        if i >= args.limit:
            break
        
        total_processed += 1
        label = rec.get("label", {})
        line_idx = label.get("line_row") # 0-based
        
        if line_idx is None:
            print(f"Skipping sample {i}: No line_row in label")
            continue
            
        z_scans = rec.get("z_scans")
        initial_states = rec.get("initial_states")
        
        # Determine Truth
        # line_idx is 0-based row index in ppc['branch']
        # Columns: BR_R=2, BR_X=3 (0-based)
        true_r = base_branch[line_idx, 2]
        true_x = base_branch[line_idx, 3]
        
        # Determine Initial (Errored) Guesses seen by tool
        # The tool loads the "case_path" which is usually just case14 on disk?
        # NO. The tool implementation `_param_correction_json` loads `case_path`.
        # Is `case_path` the *original* case14 or the *errored* case?
        # In `build_sft_traces.py`, we see:
        #   "case_path": case_path (which is "case14")
        # BUT `generate_measurements.py` modifies the branch in memory but doesn't save a specific errored file for parameter errors?
        # Wait. If `case14` on disk is clean, and the tool loads `case14`, then the tool starts with the *Clean* parameters?
        # If so, correction is moot?
        # Let's check `correct_parameters_from_path` implementation note:
        # "Assumes the case already contains the erroneous parameters; this tool estimates corrected [R, X]"
        # If we pass "case14", and case14 is clean, then the tool starts at Truth.
        # However, the measurements `z_scans` are generated from the *Errored* model (where R/X are wrong).
        # So the "Initial Guess" is Truth (if case14 is clean on disk).
        # The solver tries to fit `z_scans` (which imply Error) using the model.
        # This is actually "Inverse Parameter" problem.
        # Usually, we assume the model in DB is wrong (has Initial Guess), and reality (Measurements) implies True.
        # Ideally: Initial Guess = Wrong Value. Truth = Value that matches Z.
        # But here:
        #   Z comes from `ppc_error` (where R = R_base * factor).
        #   Tool loads `ppc_disk` (where R = R_base).
        #   So Initial Guess = R_base.
        #   Target (Reality) = R_base * factor.
        #   So the tool should move from R_base -> (R_base * factor).
        
        # Wait, usually "Parameter Error" means:
        #   Database has Value A. Reality is Value B.
        #   We want to update Database to Value B.
        #   Here, `z_scans` reflect Reality (which has the "error factor" applied in generation).
        #   So "Reality" = R_base * factor.
        #   "Database" = R_base.
        #   So we expect the Estimated Param to converge to (R_base * factor).
        
        # Let's verify this interpretation.
        r_factor = label.get("r_factor", 1.0)
        x_factor = label.get("x_factor", 1.0)
        
        target_r = true_r * r_factor
        target_x = true_x * x_factor
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Line Index: {line_idx} ({label.get('from_bus')}->{label.get('to_bus')})")
        print(f"Base Values:  R={true_r:.4f}, X={true_x:.4f}")
        print(f"Reality (Target): R={target_r:.4f}, X={target_x:.4f} (Factors: {r_factor:.2f}, {x_factor:.2f})")
        
        # Call correction tool
        # Note: If the tool loads clean case14, it starts at Base Values.
        # IMPORTANT: MCP tool expects 1-based branch row index (MATLAB).
        # Our dataset label uses 0-based `line_row` (PYPOWER/Python).
        line_index_1based = int(line_idx) + 1

        print(f"Calling correct_parameters_from_path (line_index={line_index_1based})...")
        result = mcp_call("correct_parameters_from_path", {
            "case_path": case_name,
            "line_index": line_index_1based,
            "z_scans": z_scans,
            "initial_states": initial_states,
            "R_variances_full": None
        })
        
        if result.get("success"):
            params = result.get("corrected_params", [])
            est_r, est_x = params[0], params[1]
            print(f"  [SUCCESS] Converged.")
            print(f"  Estimated:    R={est_r:.4f}, X={est_x:.4f}")

            meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
            if meta:
                mf = meta.get("from_bus")
                mt = meta.get("to_bus")
                if mf is not None and mt is not None:
                    expected_f = label.get("from_bus")
                    expected_t = label.get("to_bus")
                    if (expected_f is not None and int(mf) != int(expected_f)) or (
                        expected_t is not None and int(mt) != int(expected_t)
                    ):
                        print(
                            f"  [WARNING] Tool metadata bus mismatch: meta={int(mf)}->{int(mt)} "
                            f"vs label={expected_f}->{expected_t} (check line_index mapping)."
                        )
            
            # Error check
            err_r = abs(est_r - target_r)
            err_x = abs(est_x - target_x)
            print(f"  Error:        R_err={err_r:.4f}, X_err={err_x:.4f}")
            
            if est_r < 0 or est_x < 0:
                print("  [WARNING] Negative parameter detected!")
                
            success_count += 1
        else:
            print("  [FAILURE] Tool did not converge.")
            # Print estimates anyway if available
            params = result.get("corrected_params", [])
            if params:
                est_r, est_x = params[0], params[1]
                print(f"  Last Estimate: R={est_r:.4f}, X={est_x:.4f}")
            print(f"  Error/Meta: {result}")

if __name__ == "__main__":
    main()
