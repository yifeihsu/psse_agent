"""
Build SFT-ready tool-use traces for the diagnostic agent.

- Reads scenarios from the generated dataset (measurement_error / parameter_error / negative).
- Calls the MCP tool implemented in mcp_server/matpower_server.py (or --mock to stub):
    - wls_from_path(case_path: str, z: List[float]) -> {success, r, lambdaN, ...}
- Emits OpenAI-style chat JSONL with a single tool call followed by a structured final decision.
- Keeps labels strictly structured (no chain-of-thought).

Requirements: requests, numpy, pandas, tqdm
"""

import os, json, time, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# --------- MCP client (JSON-RPC over HTTP or WebSocket HTTP-bridge) ---------

def mcp_call_tool(endpoint: str, name: str, arguments: dict, timeout=60):
    """
    Minimal MCP JSON-RPC call for FastMCP HTTP bridge.
    Expects endpoint to handle method "tools/call" with params {name, arguments} and return
    {"result": {"isError": bool, "content": [{"type":"text","text":"{...json...}"}]}}.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1e6) % 2**31,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments}
    }
    headers = {"Accept": "application/json, text/event-stream"}
    r = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("result", {"isError": True, "content": [{"type": "text", "text": "Invalid MCP response"}]})

# --------- utilities ---------

def sha_short(x) -> str:
    return hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:8]

def as_tool_return_text(obj: dict) -> str:
    """Tool messages should be plain text; we use JSON string for structure."""
    return json.dumps(obj, separators=(",", ":"))

# --------- main builder ---------

def build_sft(
    samples_path="out_sft_measurements/samples.jsonl",
    meta_path="out_sft_measurements/meta.json",
    case_name: str | None = None,
    mcp_endpoint="http://localhost:3929/tools",   # set to your FastMCP HTTP endpoint
    out_path="sft_with_tools.jsonl",
    mock=False,
    seed=7,
    add_no_error=50,
    with_correction=True,
    corr_max_iter=2,
    corr_tol=1e-3
):
    """
    add_no_error: number of "no_error" negative controls synthesized from clean measurements (small noise only)
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    out = open(out_path, "w")
    meta = json.load(open(meta_path))
    idx_map = {k: slice(v[0], v[1]) for k, v in meta["index_map"].items()}
    # Determine the case to use with MCP (prefer meta to ensure alignment)
    case_path = meta.get("case") if case_name in (None, "auto") else case_name

    # load all samples into memory (ok for ~1k)
    samples = [json.loads(l) for l in open(samples_path)]
    # optionally add a small set of negative controls (no gross errors)
    if add_no_error > 0:
        negs = []
        for s in samples:
            if s["scenario"] == "measurement_error" and s["label"]["subtype"] in ("channel_bias","channel_scale"):
                # turn this one into "no_error" by undoing the bias/scale on z_obs (weak heuristic)
                z_obs = np.array(s["z_obs"], dtype=float)
                ch = s["label"]["channel"]
                sl = idx_map[ch]
                lab = s["label"].copy()
                if "bias" in lab:
                    z_obs[sl] = z_obs[sl] - lab["bias"]
                if "scale" in lab:
                    z_obs[sl] = z_obs[sl] / lab["scale"]
                rec = {
                    "id": f"ne_{rng.randrange(10**12)}",
                    "scenario": "no_error",
                    "z_obs": z_obs.tolist(),
                    "z_true": z_obs.tolist(),
                    "label": {"error_type": "no_error"},
                    "op_point": s["op_point"]
                }
                negs.append(rec)
                if len(negs) >= add_no_error:
                    break
        samples.extend(negs)

    # build conversations
    n_skipped = 0
    for rec in tqdm(samples, desc="Building SFT traces"):
        z_obs = rec["z_obs"]
        sid = rec["id"]
        scenario = rec["scenario"]

        # 1) system prompt: agent contract + output schema
        system_prompt = (
            "You are a power-system diagnostic agent. "
            "You have MCP tools: `wls_from_path` (state estimation with bad-data indicators).\n"
            "Procedure: (1) call `wls_from_path` on the provided snapshot; (2) inspect residuals `r` and normalized Lagrange multipliers `lambdaN`; "
            "(3) produce a STRICT JSON decision with keys: "
            "{has_error:boolean, error_family:'measurement_error'|'parameter_error'|'topology_error'|'no_error', "
            "decision_basis:{r_topk:number[], lambda_topk:number[]}, "
            "suspect_location:{...}, "
            "recommended_tool:null, confidence:number}.\n"
            "Never include chain-of-thought, only the final JSON."
        )

        # 2) user message - reference paths or inline small arrays; use inline for simplicity
        user_content = {
            "case": case_path,
            "z_obs": z_obs,        # per-unit vector with order: Vm, Pinj, Qinj, Pf, Qf, Pt, Qt
            "meta_hint": {"nb": meta["nb"], "nl": meta["nl"]},
            "note": "Run WLS (wls_from_path) and decide."
        }

        # 3) assistant -> tool call
        tool_call = {
            "type": "function",
            "id": f"call_wls_{sha_short(sid)}",
            "function": {
                "name": "wls_from_path",
                "arguments": json.dumps({"case_path": case_path, "z": z_obs})
            }
        }

        # 4) tool result (real MCP call or mock)
        if mock:
            # simple mock: magnify lambda when label says parameter_error; else magnify residual of the channel/index
            if scenario == "parameter_error":
                lam = np.zeros(meta["nl"] * 2)
                i = rec["label"]["line_row"]
                lam[2*i:2*i+2] = [5.0, 6.0]  # big values
                tool_payload = {"success": True, "r": [0.1], "lambdaN": lam.tolist()}
            elif scenario == "measurement_error":
                # residual vector matches measurement dimension (3*nb + 4*nl)
                r = np.zeros(meta["nb"]*3 + meta["nl"]*4)
                lab = rec["label"]
                ch = lab["channel"]
                sl = idx_map[ch]
                # spike one element for outlier; or lift the whole channel
                if lab["subtype"] == "single_gross_outlier" and "index" in lab:
                    r[lab["index"]] = 6.0
                else:
                    r[sl.start:sl.stop] = 3.5
                tool_payload = {"success": True, "r": r.tolist(), "lambdaN": [0.2]*(meta["nl"]*2)}
            elif scenario == "topology_error":
                # Topology mismatch generally causes widespread residuals. Emulate with elevated flow residuals.
                m = meta["nb"]*3 + meta["nl"]*4
                r = np.zeros(m)
                pf = idx_map["Pf"]; qf = idx_map["Qf"]; pt = idx_map["Pt"]; qt = idx_map["Qt"]
                r[pf.start:pf.stop] = 4.0
                r[qf.start:qf.stop] = 3.8
                r[pt.start:pt.stop] = 3.6
                r[qt.start:qt.stop] = 3.4
                tool_payload = {"success": True, "r": r.tolist(), "lambdaN": [0.3]*(meta["nl"]*2)}
            else:  # no_error
                tool_payload = {"success": True, "r": [0.1], "lambdaN": [0.1]*(meta["nl"]*2)}
        else:
            try:
                mcp_ret = mcp_call_tool(
                    mcp_endpoint,
                    "wls_from_path",
                    {"case_path": case_path, "z": z_obs}
                )
                # extract text from MCP 'content' and parse as JSON
                tool_payload = (
                    mcp_ret.get("structuredContent")
                    if isinstance(mcp_ret.get("structuredContent"), dict)
                    else None
                )
                if tool_payload is None:
                    text_blobs = [c.get("text","") for c in mcp_ret.get("content", []) if c.get("type") == "text"]
                    tool_payload = json.loads(text_blobs[0]) if text_blobs else {"success": False}
            except Exception as e:
                n_skipped += 1
                continue

        # Summarize WLS output to save tokens (top-k only)
        tool_payload_slim = {"success": bool(tool_payload.get("success", True))}
        try:
            r_vec0 = np.asarray(tool_payload.get("r", []), dtype=float)
            lam0 = np.asarray(tool_payload.get("lambdaN", []), dtype=float)
            nb = int(meta["nb"]); nl = int(meta["nl"]); nz = 3*nb + 4*nl
            if r_vec0.size == nz:
                sigma = np.empty(nz, dtype=float)
                sigma[:nb] = 0.01; sigma[nb:3*nb] = 0.005; sigma[3*nb:3*nb+4*nl] = 0.002
                rn = r_vec0 / sigma
            else:
                rn = r_vec0
            # top-5 indices by absolute value
            if rn.size:
                topk_idx = np.argsort(-np.abs(rn))[:5].tolist()
                tool_payload_slim["r_topk"] = [[int(i), float(rn[int(i)])] for i in topk_idx]
                tool_payload_slim["r_len"] = int(r_vec0.size)
            if lam0.size:
                topk_l = np.argsort(-np.abs(lam0))[:5].tolist()
                tool_payload_slim["lambda_topk"] = [[int(i), float(lam0[int(i)])] for i in topk_l]
                tool_payload_slim["lambdaN_len"] = int(lam0.size)
        except Exception:
            pass

        # 5) optional correction calls (after detection logic, only when we can act without extra input)
        extra_msgs = []
        if with_correction and not mock and scenario == "measurement_error":
            # Choose the single index with largest |normalized residual| as suspect_group.
            # Normalize residuals using the same default variances as the server tool:
            # Vm: (0.01)^2, Pinj/Qinj: (0.005)^2, flows (Pf/Qf/Pt/Qt): (0.002)^2.
            r_vec = np.asarray(tool_payload.get("r", []), dtype=float)
            sg = None
            if r_vec.size:
                try:
                    nb = int(meta["nb"])  # buses
                    nl = int(meta["nl"])  # branches
                    nz = 3*nb + 4*nl
                    if r_vec.size == nz:
                        sigma = np.empty(nz, dtype=float)
                        sigma[:nb] = 0.01
                        sigma[nb:3*nb] = 0.005
                        sigma[3*nb:3*nb+4*nl] = 0.002
                        r_norm = r_vec / sigma
                        k = int(np.nanargmax(np.abs(r_norm)))
                        sg = [k]
                    else:
                        # fallback to raw residual magnitude if sizes mismatch
                        k = int(np.nanargmax(np.abs(r_vec)))
                        sg = [k]
                except Exception:
                    sg = None
            # Fallbacks when residuals are unavailable or above failed
            if sg is None:
                lab = rec.get("label", {})
                if isinstance(lab.get("index"), int):
                    sg = [int(lab["index"])]
                elif isinstance(lab.get("indices"), list):
                    sg = [int(i) for i in lab["indices"]]
                else:
                    ch = lab.get("channel")
                    if ch in idx_map:
                        sl = idx_map[ch]
                        sg = list(range(sl.start, sl.stop))
                if sg is None:
                    sg = []

            corr_call = {
                "type": "function",
                "id": f"call_corr_meas_{sha_short(sid)}",
                "function": {
                    "name": "correct_measurements_from_path",
                    "arguments": json.dumps({
                        "case_path": case_path,
                        "z": z_obs,
                        "suspect_group": sg,
                        "enable_correction": True,
                        "max_correction_iterations": int(corr_max_iter),
                        "error_tolerance": float(corr_tol)
                    })
                }
            }
            # Attempt the MCP call; if it fails, still emit a visible failure payload
            try:
                corr_ret = mcp_call_tool(
                    mcp_endpoint,
                    "correct_measurements_from_path",
                    {"case_path": case_path, "z": z_obs, "suspect_group": sg,
                     "enable_correction": True, "max_correction_iterations": int(corr_max_iter),
                     "error_tolerance": float(corr_tol), "R_variances_full": None}
                )
                corr_payload = (
                    corr_ret.get("structuredContent")
                    if isinstance(corr_ret.get("structuredContent"), dict)
                    else None
                )
                if corr_payload is None:
                    _texts = [c.get("text", "") for c in corr_ret.get("content", []) if c.get("type") == "text"]
                    corr_payload = json.loads(_texts[0]) if _texts else {"success": False}
            except Exception as e:
                corr_payload = {"success": False, "error": str(e)}

            # Slim correction payload: include corrected_measurements and r_norm summary
            corr_payload_slim = {"success": bool(corr_payload.get("success", True))}
            try:
                cms = corr_payload.get("corrected_measurements") or []
                # keep up to 5 entries
                corr_payload_slim["corrected_measurements"] = cms[:5]
                rn = np.asarray(corr_payload.get("r_norm", []), dtype=float)
                if rn.size:
                    corr_payload_slim["r_norm_max_abs"] = float(np.nanmax(np.abs(rn)))
            except Exception:
                pass

            extra_msgs.extend([
                {"role": "assistant", "tool_calls": [corr_call]},
                {"role": "tool", "tool_call_id": corr_call["id"], "name": "correct_measurements_from_path",
                 "content": as_tool_return_text(corr_payload_slim)}
            ])

            # If we received a corrected value, substitute and verify with a second WLS
            try:
                # primary: use our k if defined
                subst_entry = None
                try:
                    # k is set in r_norm computation above when r matched nz
                    k
                except NameError:
                    k = None
                cms = corr_payload.get("corrected_measurements") or []
                if k is not None:
                    for e in cms:
                        if int(e.get("index0", -1)) == int(k):
                            subst_entry = e; break
                if subst_entry is None and cms:
                    # fallback: pick by largest |estimated_error|
                    best = None; best_abs = -1
                    for e in cms:
                        try:
                            v = abs(float(e.get("estimated_error", 0.0)))
                            if v > best_abs:
                                best_abs = v; best = e
                        except Exception:
                            continue
                    subst_entry = best
                if subst_entry:
                    z2 = list(z_obs)
                    idx0 = int(subst_entry.get("index0"))
                    val = float(subst_entry.get("corrected"))
                    if 0 <= idx0 < len(z2):
                        z2[idx0] = val
                        wls2_call = {
                            "type": "function",
                            "id": f"call_wls_verify_{sha_short(sid)}",
                            "function": {
                                "name": "wls_from_path",
                                "arguments": json.dumps({"case_path": case_path, "z": z2})
                            }
                        }
                        wls2_ret = mcp_call_tool(mcp_endpoint, "wls_from_path", {"case_path": case_path, "z": z2})
                        wls2_payload = (
                            wls2_ret.get("structuredContent") if isinstance(wls2_ret.get("structuredContent"), dict) else None
                        )
                        if wls2_payload is None:
                            _tb = [c.get("text","") for c in wls2_ret.get("content", []) if c.get("type") == "text"]
                            wls2_payload = json.loads(_tb[0]) if _tb else {"success": False}
                        # slim summary
                        wls2_slim = {"success": bool(wls2_payload.get("success", True))}
                        try:
                            r2 = np.asarray(wls2_payload.get("r", []), dtype=float)
                            if r2.size == nz:
                                sigma = np.empty(nz, dtype=float)
                                sigma[:nb] = 0.01; sigma[nb:3*nb] = 0.005; sigma[3*nb:3*nb+4*nl] = 0.002
                                r2n = r2 / sigma
                            else:
                                r2n = r2
                            if r2n.size:
                                topk2 = np.argsort(-np.abs(r2n))[:5].tolist()
                                wls2_slim["r_topk"] = [[int(i), float(r2n[int(i)])] for i in topk2]
                                wls2_slim["r_len"] = int(r2.size)
                        except Exception:
                            pass
                        extra_msgs.extend([
                            {"role": "assistant", "tool_calls": [wls2_call]},
                            {"role": "tool", "tool_call_id": wls2_call["id"], "name": "wls_from_path",
                             "content": as_tool_return_text(wls2_slim)}
                        ])
            except Exception:
                pass

        # 6) assistant final content (GROUND TRUTH as target)
        if scenario == "parameter_error":
            lab = rec["label"]
            final = {
                "has_error": True,
                "error_family": "parameter_error",
                "decision_basis": {
                    "r_topk": [],  # optional: fill during training time if desired
                    "lambda_topk": []  # same
                },
                "suspect_location": {
                    "line_row": lab["line_row"],
                    "from_bus": lab["from_bus"],
                    "to_bus": lab["to_bus"]
                },
                "recommended_tool": "correct_parameters_from_path",
                "confidence": 0.99
            }
        elif scenario == "measurement_error":
            lab = rec["label"]
            final = {
                "has_error": True,
                "error_family": "measurement_error",
                "decision_basis": {"r_topk": [], "lambda_topk": []},
                "suspect_location": {
                    "channel": lab["channel"],
                    **({ "index": lab["index"] } if "index" in lab else {}),
                },
                "recommended_tool": "correct_measurements_from_path",
                "confidence": 0.99
            }
        elif scenario == "topology_error":
            lab = rec["label"]
            final = {
                "has_error": True,
                "error_family": "topology_error",
                "decision_basis": {"r_topk": [], "lambda_topk": []},
                "suspect_location": {
                    "substation": lab.get("substation"),
                    "cb_name": lab.get("cb_name")
                },
                "recommended_tool": None,
                "confidence": 0.99
            }
        else:  # no_error
            final = {
                "has_error": False,
                "error_family": "no_error",
                "decision_basis": {"r_topk": [], "lambda_topk": []},
                "suspect_location": {},
                "recommended_tool": None,
                "confidence": 0.98
            }

        # Bundle into a single conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content)},
            {"role": "assistant", "tool_calls": [tool_call]},
            {
              "role": "tool",
              "tool_call_id": tool_call["id"],
              "name": "wls_from_path",
              "content": as_tool_return_text(tool_payload_slim)
            },
        ]
        messages.extend(extra_msgs)
        messages.append({"role": "assistant", "content": json.dumps(final)})
        convo = {"messages": messages}
        out.write(json.dumps(convo) + "\n")

    out.close()
    print(f"Wrote SFT file: {out_path}")
    if n_skipped:
        print(f"Skipped {n_skipped} examples due to MCP errors/timeouts.")

    # Optional: split train/valid/test with deterministic shuffle
    rows = [json.loads(l) for l in open(out_path)]
    rng.shuffle(rows)
    n = len(rows)
    a, b = int(0.8*n), int(0.9*n)
    for name, chunk in [("train", rows[:a]), ("valid", rows[a:b]), ("test", rows[b:])]:
        with open(f"split_{name}.jsonl", "w") as f:
            for r in chunk: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="out_sft_measurements/samples.jsonl")
    p.add_argument("--meta", default="out_sft_measurements/meta.json")
    p.add_argument("--case", default="auto", choices=["auto","case14","case118"], help="MATPOWER case name; 'auto' uses value in meta.json")
    p.add_argument("--endpoint", default="http://localhost:3929/tools")
    p.add_argument("--out", default="sft_with_tools.jsonl")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--no-error", type=int, default=50)
    p.add_argument("--no-correction", action="store_true")
    p.add_argument("--corr-iters", type=int, default=2)
    p.add_argument("--corr-tol", type=float, default=1e-3)
    args = p.parse_args()

    build_sft(args.samples, args.meta, None if args.case == "auto" else args.case,
              args.endpoint, args.out, args.mock, 7, args.no_error,
              with_correction=(not args.no_correction),
              corr_max_iter=args.corr_iters,
              corr_tol=args.corr_tol)
