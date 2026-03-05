import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import requests

endpoint = "http://127.0.0.1:3929/tools"


def _load_topology_samples(samples_path: str) -> List[Dict[str, Any]]:
    """Return all sample records with scenario == 'topology_error'."""
    recs = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                candidate = json.loads(line)
            except Exception:
                continue
            if candidate.get("scenario") == "topology_error":
                recs.append(candidate)
    if not recs:
        # Just print warning, don't crash
        print(f"Warning: No topology_error samples found in {samples_path}")
    return recs

def mcp_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    headers = {"Accept": "application/json, text/event-stream"}
    try:
        r = requests.post(endpoint, json=payload, headers=headers, timeout=90)
        r.raise_for_status()
        data = r.json()
        # DEBUG PRINTS
        # DEBUG PRINTS
        # print(f"DEBUG: mcp_call({name}) raw response: {json.dumps(data, default=str)[:500]}...") 
        if isinstance(data, dict) and "error" in data:
             # print(f"DEBUG: mcp_call error found: {data['error']}")
             pass
        
        # End Debug
        # JSON-RPC error path (FastMCP returns this at the top level)
        if isinstance(data, dict) and "error" in data:
            err = data.get("error") or {}
            if isinstance(err, dict):
                return {
                    "success": False,
                    "error": err.get("message") or str(err),
                    "code": err.get("code"),
                    "data": err.get("data"),
                }
            return {"success": False, "error": str(err)}

        res = data.get("result", {}) if isinstance(data, dict) else {}
        sc = res.get("structuredContent")
        if isinstance(sc, dict):
            return sc
        texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
        for t in texts:
            t2 = t.strip()
            if t2.startswith("```"):
                t2 = re.sub(r"^```[a-zA-Z0-9_\-]*\n|\n```$", "", t2)
            m = re.search(r"\{[\s\S]*\}", t2)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            try:
                return json.loads(t2)
            except Exception:
                continue
        # Only raise if text exists but fails parsing. If empty content, return empty?
        # Standardize empty return as failure indicator or let it raise?
        # debug: print(f"Raw texts: {texts}")
        return {} 
    except Exception as e:
        print(f"MCP Call Failed: {e}")
        return {}



def process_sample(rec: Dict[str, Any], idx: int, case_path: str = "case14") -> None:
    z: List[float] = rec["z_obs"]
    scenario = rec.get("scenario")
    lab: Dict[str, Any] = rec.get("label", {})
    cb_name: Optional[str] = lab.get("cb_name")
    ns = lab.get("new_status", False)
    if isinstance(ns, str):
        injected_status = (ns.lower() in ("closed", "true", "1"))
    else:
        injected_status = bool(ns)
    # The debug script sets target status explicitly.
    # If the error is 'open' (False), we want to inject 'True' (Correct it).
    desired_status = not injected_status

    print(f"\n--- Sample {idx+1} ---")
    print(f"Scenario: {scenario}")
    print(f"CB name: {cb_name!r}")
    print(f"Injected error status (new_status): {injected_status}")
    print(f"Desired (healthy) status for correction: {desired_status}")

    # 1) WLS on original measurements (Detection Step) (existing logic)
    try:
        wls = mcp_call("wls_from_path", {"case_path": case_path, "z": z})
        # Check if WLS returned a valid result or empty (crash indicator)
        if not wls:
              print("WLS returned empty response (likely crashed). Treating as HIGH Error (Detection Successful).")
              r = np.array([])
        else:
            r = np.array(wls.get("r", []), float)
            if r.size:
                print(f"WLS success: {wls.get('success')}")
                print(f"Max |r_norm| before correction: {np.nanmax(np.abs(r)):.3f}")
            else:
                print("WLS returned no residuals; payload keys:", list(wls.keys()))
                if "error" in wls:
                    print("WLS Error:", wls["error"])
    except Exception as e:
        print(f"WLS Step 1 Failed (likely crashed): {e}")
        print("Treating as HIGH Error (Detection Successful). Proceeding to correction...")

    # 2) Topology correction verification using PRE-GENERATED model
    # The dataset now contains 'corrected_model_path' and 'z_true_full_model' (which is the actual measurement for the corrected state)
    
    model_path = rec.get("corrected_model_path")
    z_corrected = rec.get("z_true_full_model")

    if not model_path or not z_corrected:
        print("Skipping verification: 'corrected_model_path' or 'z_true_full_model' not found in sample.")
        # Fallback to old logic? No, user requested specific workflow.
        return

    if not os.path.exists(model_path):
        print(f"Skipping verification: Model file not found at {model_path}")
        return

    print(f"Verifying using pre-generated corrected model: {os.path.basename(model_path)}")
    
    # 3) Verification: Run WLS using the CUSTOM MODEL FILE + ACTUAL MEASUREMENTS
    try:
        # We use z_corrected (which is z_true_full_model from generator) 
        # as the measurement vector to verify against the corrected model.
        wls2 = mcp_call("wls_from_path", {"case_path": model_path, "z": z_corrected})
        
        r2 = np.array(wls2.get("r", []), float)
        if r2.size:
            print(f"WLS-after (corrected model & z) success: {wls2.get('success')}")
            print(f"Max |r_norm| after correction: {np.nanmax(np.abs(r2)):.3f}")
        else:
            print("Post-correction WLS returned no residuals; payload keys:", list(wls2.keys()))
            if "error" in wls2:
                print("Post-correction WLS Error:", wls2["error"])
    except Exception as e:
        print(f"Verification WLS failed: {e}")



def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="out_sft_measurements", help="Directory containing samples.jsonl and meta.json")
    args = parser.parse_args()

    input_dir = args.input_dir
    meta_path = os.path.join(input_dir, "meta.json")
    samples_path = os.path.join(input_dir, "samples.jsonl")

    # global case
    meta = json.load(open(meta_path, encoding="utf-8"))
    case = meta["case"]

    print(f"Loading samples from {samples_path}...")
    recs = _load_topology_samples(samples_path)
    print(f"Found {len(recs)} topology error samples.")
    for i, rec in enumerate(recs):
        process_sample(rec, i, case_path=case)


if __name__ == "__main__":
    main()
