import json, re, requests, numpy as np

endpoint = "http://127.0.0.1:3929/tools"
meta = json.load(open("out_sft_measurements/meta.json"))
case = meta["case"]

# Prefer a measurement_error sample in lines 500..600, else fall back to first measurement_error, else first sample
z = None
rec = None
with open("out_sft_measurements/samples.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
    # Try window 500..600
    for i in range(min(600, len(lines))):
        if 500 <= i < 600:
            try:
                rec = json.loads(lines[i])
                if rec.get("scenario") == "measurement_error":
                    z = rec["z_obs"]
                    break
            except Exception:
                pass
    if z is None:
        # Try any measurement_error
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
                if rec.get("scenario") == "measurement_error":
                    z = rec["z_obs"]
                    break
            except Exception:
                pass
    if z is None:
        # Fallback to first valid row with z_obs
        for line in lines:
            try:
                rec = json.loads(line)
                if "z_obs" in rec:
                    z = rec["z_obs"]
                    break
            except Exception:
                pass
if z is None:
    raise RuntimeError("Could not find a valid sample with z_obs")

def mcp_call(name, args):
    payload = {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":args}}
    # FastMCP HTTP bridge expects an Accept header; without it some setups return 406
    headers = {"Accept": "application/json, text/event-stream"}
    r = requests.post(endpoint, json=payload, headers=headers, timeout=90)
    r.raise_for_status()
    res = r.json().get("result", {})
    sc = res.get("structuredContent")
    if isinstance(sc, dict):
        return sc
    texts = [c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"]
    # Try to find a JSON object in any text blob
    for t in texts:
        # Strip code fences if present
        t2 = t.strip()
        if t2.startswith("```"):
            t2 = re.sub(r"^```[a-zA-Z0-9_\-]*\n|\n```$", "", t2)
        # Extract the first {...} JSON object
        m = re.search(r"\{[\s\S]*\}", t2)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # Try raw
        try:
            return json.loads(t2)
        except Exception:
            continue
    raise RuntimeError(f"No JSON content from MCP. Raw: {texts[:1]}")

"""Run WLS to compute residuals and select the suspect by largest normalized residual."""
wls = mcp_call("wls_from_path", {"case_path": case, "z": z})
r = np.array(wls["r"], float)
nb, nl = int(meta["nb"]), int(meta["nl"])
nz = 3*nb + 4*nl
r_norm = r
k = int(np.nanargmax(np.abs(r_norm)))
print(f"Suspect index (0-based): {k}  | max |r_norm| before: {np.nanmax(np.abs(r_norm)):.3f}")

# Identify channel and local index for readability
chan_name = None; chan_local = None
for ch, (s, e) in meta.get("index_map", {}).items():
    if s <= k < e:
        chan_name = ch
        chan_local = k - s
        break
if chan_name is not None:
    print(f"Channel: {chan_name}[{chan_local}]")

# 2) Correction
# Server tool accepts 0- or 1-based; to be robust for MATLAB indexing use 1-based here
corr = mcp_call("correct_measurements_from_path", {
    "case_path": case,
    "z": z,
    "suspect_group": [k + 1],
    "enable_correction": True,
    "max_correction_iterations": 2,
    "error_tolerance": 1e-3,
    "R_variances_full": None
})
r_after = np.array(corr.get("r_norm", []), float)
print("Correction success:", corr.get("success"))
if r_after.size:
    print(f"Max |r_norm| after:  {np.nanmax(np.abs(r_after)):.3f}")
print("Returned keys:", list(corr.keys()))

# Print ground truth and corrected measurement value for the suspect index
z_true = rec.get("z_true") if isinstance(rec, dict) else None
obs_val = z[k] if k < len(z) else None
gt_val = z_true[k] if isinstance(z_true, list) and k < len(z_true) else None
zci = corr.get("z_corrected_info", {}) or {}
cidx_raw = zci.get("last_corrected_global_indices") or []
orig_vals_raw = zci.get("last_original_values") or []
corr_vals_raw = zci.get("last_corrected_values") or []
est_errs_raw = zci.get("last_estimated_errors") or []

def _flatten(seq):
    if isinstance(seq, (list, tuple)):
        for x in seq:
            yield from _flatten(x)
    elif seq is not None:
        yield seq

# Flatten and normalize types
try:
    cidx_list = [float(x) for x in _flatten(cidx_raw)]
except Exception:
    cidx_list = []
try:
    corr_vals = [float(x) for x in _flatten(corr_vals_raw)]
except Exception:
    corr_vals = []
try:
    orig_vals = [float(x) for x in _flatten(orig_vals_raw)]
except Exception:
    orig_vals = []
try:
    est_errs = [float(x) for x in _flatten(est_errs_raw)]
except Exception:
    est_errs = []

# Normalize corrected indices (1-based -> 0-based)
idx0s = []
try:
    idx0s = [int(round(i)) - 1 for i in cidx_list]
except Exception:
    idx0s = []

# Find position j in tool arrays corresponding to our suspect k
pos = None
try:
    pos = idx0s.index(k)
except Exception:
    pos = None

# If our k wasn't corrected, optionally pick the tool's strongest correction
fallback_j = None
if pos is None and len(idx0s) > 0:
    try:
        if isinstance(est_errs, list) and len(est_errs) == len(idx0s):
            # choose by largest |estimated error|
            fallback_j = int(np.nanargmax(np.abs(np.array(est_errs, float))))
        else:
            fallback_j = 0
    except Exception:
        fallback_j = 0

print("Observed z[k]:", obs_val)
print("Ground truth z_true[k]:", gt_val if gt_val is not None else "N/A")
if pos is not None and pos < len(corr_vals):
    print("Corrected value (tool at k):", corr_vals[pos])
    if pos < len(orig_vals):
        print("Original value (tool group at k):", orig_vals[pos])
elif fallback_j is not None and fallback_j < len(corr_vals):
    print(f"Tool corrected different index {idx0s[fallback_j]} (0-based); using that for substitution.")
    print("Corrected value (tool):", corr_vals[fallback_j])
    if fallback_j < len(orig_vals):
        print("Original value (tool group):", orig_vals[fallback_j])
else:
    # If the tool didn't report any index, show first few reported for reference
    if isinstance(cidx_list, list) and len(cidx_list) > 0:
        preview = min(5, len(cidx_list))
        print(f"Tool reported corrected indices (1-based): {cidx_list[:preview]} ...")
        if len(corr_vals) >= preview:
            print(f"Their corrected values preview: {corr_vals[:preview]}")

print("z_corrected_info keys:", list(zci.keys()))

# 3) Substitute a corrected value and re-run WLS to verify
subst_j = None
subst_k = None
if pos is not None and pos < len(corr_vals):
    subst_j = pos
    subst_k = k
elif fallback_j is not None and fallback_j < len(corr_vals) and fallback_j < len(idx0s):
    subst_j = fallback_j
    subst_k = idx0s[fallback_j]

if subst_j is not None and subst_k is not None and 0 <= subst_k < len(z) and subst_j < len(corr_vals):
    try:
        z2 = list(z)
        z2[subst_k] = float(corr_vals[subst_j])
        wls2 = mcp_call("wls_from_path", {"case_path": case, "z": z2})
        r2 = np.array(wls2["r"], float)  # already normalized
        r2_norm = r2
        print("Re-run WLS after substitution:")
        print(f"  Substituted index (0-based): {subst_k}")
        print(f"  Max |r_norm| after substitution: {np.nanmax(np.abs(r2_norm)):.3f}")
        before = abs(r_norm[subst_k]) if subst_k < len(r_norm) else float('nan')
        after = abs(r2_norm[subst_k]) if subst_k < len(r2_norm) else float('nan')
        print(f"  |r_norm[subst_k]| before -> after: {before:.3f} -> {after:.3f}")
    except Exception as e:
        print("Re-run WLS failed:", str(e))
else:
    print("No corrected index available for substitution; skipping WLS re-run.")