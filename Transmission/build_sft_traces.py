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
            "You have MCP tools: `wls_from_path` (state estimation with bad-data indicators) and `run_hse_from_path` (harmonic state estimation).\n"
            "Procedure: (1) call `wls_from_path` on the provided snapshot; (2) inspect residuals `r` and normalized Lagrange multipliers `lambdaN`; "
            "If the global residual J is significantly elevated without a clear single bad measurement, suspect harmonics and call `run_hse_from_path` with the available harmonic measurements.\n"
            "(3) produce a STRICT JSON decision with keys: "
            "{has_error:boolean, error_family:'measurement_error'|'parameter_error'|'topology_error'|'three_phase_imbalance'|'harmonic_anomaly'|'no_error', "
            "decision_basis:{r_topk:number[], lambda_topk:number[]}, "
            "suspect_location:{...}, "
            "recommended_tool:null, confidence:number}.\n"
            "If three-phase imbalance is suspected, request 3ϕ substation voltage measurements before finalizing.\n"
            "Never include chain-of-thought, only the final JSON."
        )

        # 2) user message - reference paths or inline small arrays; use inline for simplicity
        user_content = {
            "case": case_path,
            "z_obs": z_obs,        # per-unit vector with order: Vm, Pinj, Qinj, Pf, Qf, Pt, Qt
            "meta_hint": {"nb": meta["nb"], "nl": meta["nl"]},
            "note": "Run WLS (wls_from_path) and decide."
        }
        if scenario == "three_phase_imbalance":
            user_content["note"] = (
                "This snapshot is a 1ϕ-equivalent operator z vector (phase-A Vm + 3ϕ totals). "
                "If imbalance is suspected, request 3ϕ VLN voltage measurements from substations."
            )
            user_content["has_three_phase_voltage_measurements"] = True

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
            elif scenario == "three_phase_imbalance":
                # Emulate widespread flow residuals (model mismatch) with moderate voltage/injection residuals.
                m = meta["nb"] * 3 + meta["nl"] * 4
                r = np.zeros(m)
                pf = idx_map["Pf"]; qf = idx_map["Qf"]; pt = idx_map["Pt"]; qt = idx_map["Qt"]
                vm = idx_map["Vm"]; pinj = idx_map["Pinj"]; qinj = idx_map["Qinj"]
                r[vm.start:vm.stop] = 2.5
                r[pinj.start:pinj.stop] = 2.0
                r[qinj.start:qinj.stop] = 2.0
                r[pf.start:pf.stop] = 4.2
                r[qf.start:qf.stop] = 4.0
                r[pt.start:pt.stop] = 3.8
                r[qt.start:qt.stop] = 3.6
                tool_payload = {"success": True, "r": r.tolist(), "lambdaN": [0.2] * (meta["nl"] * 2)}
            elif scenario == "harmonic_anomaly":
                # Elevated global residual (J), but no single massive spike > 6
                m = meta["nb"] * 3 + meta["nl"] * 4
                r = np.random.normal(0, 0.5, m)
                vm = idx_map["Vm"]
                r[vm.start:vm.stop] = np.random.normal(1.5, 0.5, vm.stop - vm.start) # moderate voltage stress
                # Make J roughly 150-300
                r *= (200.0 / np.sum(r**2))**0.5
                tool_payload = {"success": True, "r": r.tolist(), "lambdaN": [0.1] * (meta["nl"] * 2)}
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

        # 5) optional correction calls (after detection logic, only when we can act without extra input)
        extra_msgs = []
        if scenario == "three_phase_imbalance":
            # In this workflow, the operator requests 3ϕ voltage measurements from substations.
            three_phase = rec.get("three_phase_voltages")
            if isinstance(three_phase, list) and three_phase:
                extra_msgs.extend(
                    [
                        {
                            "role": "assistant",
                            "content": (
                                "The residual pattern is consistent with a possible three-phase imbalance. "
                                "Please provide three-phase (A/B/C) substation voltage measurements (VLN magnitude/angle per bus) "
                                "so I can proceed with three-phase state estimation / imbalance assessment."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "three_phase_voltages": three_phase,
                                    "note": "Per-bus 3ϕ VLN voltage measurements (pu) from substations.",
                                }
                            ),
                        },
                    ]
                )
        if with_correction and not mock and scenario == "measurement_error":
            # Choose the single index with largest |normalized residual| as suspect_group.
            # Note: wls_from_path returns normalized residuals already (field 'r').
            r_vec = np.asarray(tool_payload.get("r", []), dtype=float)
            sg = None
            if r_vec.size:
                try:
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
            extra_msgs.extend([
                {"role": "assistant", "tool_calls": [corr_call]},
                {"role": "tool", "tool_call_id": corr_call["id"], "name": "correct_measurements_from_path",
                 "content": as_tool_return_text(corr_payload)}
            ])

            # If we received a corrected value, substitute and verify with a second WLS
            try:
                # primary: use our k if defined
                subst_entry = None
                # k is set above
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
                        # emit full payload (including ea)
                        extra_msgs.extend([
                            {"role": "assistant", "tool_calls": [wls2_call]},
                            {"role": "tool", "tool_call_id": wls2_call["id"], "name": "wls_from_path",
                             "content": as_tool_return_text(wls2_payload)}
                        ])
            except Exception:
                pass
        elif with_correction and not mock and scenario == "parameter_error":
            # Use multi-scan data (if available) to call parameter correction tool
            lab = rec.get("label", {})
            line_row = int(lab.get("line_row", -1))
            z_scans = rec.get("z_scans")
            init_states = rec.get("initial_states")
            if line_row >= 0 and isinstance(z_scans, list) and isinstance(init_states, list):
                # prefer 1-based line index for MATLAB; tool accepts 0/1-based
                line_index = line_row + 1
                param_call = {
                    "type": "function",
                    "id": f"call_corr_param_{sha_short(sid)}",
                    "function": {
                        "name": "correct_parameters_from_path",
                        "arguments": json.dumps({
                            "case_path": case_path,
                            "line_index": line_index,
                            "z_scans": z_scans,
                            "initial_states": init_states,
                            "R_variances_full": None
                        })
                    }
                }
                try:
                    param_ret = mcp_call_tool(
                        mcp_endpoint,
                        "correct_parameters_from_path",
                        {
                            "case_path": case_path,
                            "line_index": line_index,
                            "z_scans": z_scans,
                            "initial_states": init_states,
                            "R_variances_full": None
                        }
                    )
                    param_payload = (
                        param_ret.get("structuredContent")
                        if isinstance(param_ret.get("structuredContent"), dict)
                        else None
                    )
                    if param_payload is None:
                        _texts = [c.get("text", "") for c in param_ret.get("content", []) if c.get("type") == "text"]
                        param_payload = json.loads(_texts[0]) if _texts else {"success": False}
                except Exception as e:
                    param_payload = {"success": False, "error": str(e)}

                extra_msgs.extend([
                    {"role": "assistant", "tool_calls": [param_call]},
                    {"role": "tool", "tool_call_id": param_call["id"], "name": "correct_parameters_from_path",
                     "content": as_tool_return_text(param_payload)}
                ])
        elif with_correction and not mock and scenario == "topology_error":
            lab = rec.get("label", {})
            cb_name = lab.get("cb_name")
            desired_status = not bool(lab.get("new_status", False))  # flip back
            if cb_name:
                topo_call = {
                    "type": "function",
                    "id": f"call_corr_topo_{sha_short(sid)}",
                    "function": {
                        "name": "correct_topology_from_path",
                        "arguments": json.dumps({
                            "case_path": case_path,
                            "cb_name": cb_name,
                            "desired_status": desired_status
                        })
                    }
                }
                try:
                    topo_ret = mcp_call_tool(
                        mcp_endpoint,
                        "correct_topology_from_path",
                        {"case_path": case_path, "cb_name": cb_name, "desired_status": desired_status}
                    )
                    topo_payload = (
                        topo_ret.get("structuredContent")
                        if isinstance(topo_ret.get("structuredContent"), dict)
                        else None
                    )
                    if topo_payload is None:
                        _texts = [c.get("text", "") for c in topo_ret.get("content", []) if c.get("type") == "text"]
                        topo_payload = json.loads(_texts[0]) if _texts else {"success": False}
                except Exception as e:
                    topo_payload = {"success": False, "error": str(e)}

                extra_msgs.extend([
                    {"role": "assistant", "tool_calls": [topo_call]},
                    {"role": "tool", "tool_call_id": topo_call["id"], "name": "correct_topology_from_path",
                     "content": as_tool_return_text(topo_payload)}
                ])

                # Optional: re-run WLS on corrected z if provided
                try:
                    # NEW LOGIC: Prefer pre-generated corrected model and measurements
                    case_path_verify = case_path
                    z_verify = None
                    
                    if "corrected_model_path" in rec and "z_true_full_model" in rec:
                        case_path_verify = rec["corrected_model_path"]
                        z_verify = rec["z_true_full_model"]
                    elif "z_corrected" in topo_payload:
                        # Fallback to tool output
                        z_verify = topo_payload.get("z_corrected")
                        
                    if isinstance(z_verify, list):
                        wls2_call = {
                            "type": "function",
                            "id": f"call_wls_verify_topo_{sha_short(sid)}",
                            "function": {
                                "name": "wls_from_path",
                                "arguments": json.dumps({"case_path": case_path_verify, "z": z_verify})
                            }
                        }
                        wls2_ret = mcp_call_tool(mcp_endpoint, "wls_from_path", {"case_path": case_path_verify, "z": z_verify})
                        wls2_payload = (
                            wls2_ret.get("structuredContent")
                            if isinstance(wls2_ret.get("structuredContent"), dict)
                            else None
                        )
                        if wls2_payload is None:
                            _tb = [c.get("text", "") for c in wls2_ret.get("content", []) if c.get("type") == "text"]
                            wls2_payload = json.loads(_tb[0]) if _tb else {"success": False}
                        extra_msgs.extend([
                            {"role": "assistant", "tool_calls": [wls2_call]},
                            {"role": "tool", "tool_call_id": wls2_call["id"], "name": "wls_from_path",
                             "content": as_tool_return_text(wls2_payload)}
                        ])
                except Exception:
                    pass
        elif scenario == "harmonic_anomaly":
            # the agent sees the elevated J, suspects harmonics, calls run_hse_from_path
            # we check if harmonic measurements are provided in the trace
            h_meas = rec.get("harmonic_measurements", [])
            hse_call = {
                "type": "function",
                "id": f"call_hse_{sha_short(sid)}",
                "function": {
                    "name": "run_hse_from_path",
                    "arguments": json.dumps({
                        "case_path": case_path,
                        "harmonic_measurements": h_meas
                    })
                }
            }
            if mock:
                lab = rec.get("label", {})
                source_bus = lab.get("source_bus", 3)
                thd = lab.get("thd_target", 10.0)
                hse_payload = {
                    "success": True,
                    "best_candidate": source_bus,
                    "max_thd": thd,
                    "source_injection_magnitude": 0.05,
                    "candidates_tested": [3, 4, 9],
                    "notes": "Harmonic source identified."
                }
            else:
                try:
                    hse_ret = mcp_call_tool(
                        mcp_endpoint,
                        "run_hse_from_path",
                        {"case_path": case_path, "harmonic_measurements": h_meas}
                    )
                    hse_payload = (
                        hse_ret.get("structuredContent")
                        if isinstance(hse_ret.get("structuredContent"), dict)
                        else None
                    )
                    if hse_payload is None:
                        _texts = [c.get("text", "") for c in hse_ret.get("content", []) if c.get("type") == "text"]
                        hse_payload = json.loads(_texts[0]) if _texts else {"success": False}
                except Exception as e:
                    hse_payload = {"success": False, "error": str(e)}

            extra_msgs.extend([
                {"role": "assistant", "tool_calls": [hse_call]},
                {"role": "tool", "tool_call_id": hse_call["id"], "name": "run_hse_from_path",
                 "content": as_tool_return_text(hse_payload)}
            ])

        # 6) assistant final content (GROUND TRUTH as target)
        if scenario == "parameter_error":
            lab = rec["label"]
            final = {
                "has_error": True,
                "error_family": "parameter_error",
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
                "suspect_location": {
                    "substation": lab.get("substation"),
                    "cb_name": lab.get("cb_name")
                },
                "recommended_tool": None,
                "confidence": 0.99
            }
        elif scenario == "three_phase_imbalance":
            lab = rec.get("label", {})
            final = {
                "has_error": True,
                "error_family": "three_phase_imbalance",
                "suspect_location": {
                    "unbalance_bus": lab.get("unbalance_bus"),
                },
                "recommended_tool": None,
                "confidence": 0.95
            }
        elif scenario == "harmonic_anomaly":
            lab = rec.get("label", {})
            final = {
                "has_error": True,
                "error_family": "harmonic_anomaly",
                "suspect_location": {
                    "source_bus": lab.get("source_bus"),
                },
                "recommended_tool": None,
                "confidence": 0.95
            }
        else:  # no_error
            final = {
                "has_error": False,
                "error_family": "no_error",
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
              "content": as_tool_return_text(tool_payload)
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
