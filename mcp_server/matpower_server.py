# matpower_mcp_server_fixed.py
from __future__ import annotations

import os
import json
import tempfile
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP

mcp = FastMCP("MATPOWER Power Flow (FastMCP v2)")

# This server is fully ported to pure Python for the grid-estimation functions used here.
# The old MATLAB-engine comments are removed because they were stale and the WLS/correction
# tools now call Python implementations directly.

# ---------- Helper: write case text to a temp .m ----------
def _parse_matpower_case(case_text: str) -> Dict[str, Any]:
    import re
    import numpy as np
    
    # 1. Extract baseMVA
    base_mva_match = re.search(r'mpc\.baseMVA\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?);', case_text)
    baseMVA = float(base_mva_match.group(1)) if base_mva_match else 100.0
    
    # 2. Extract matrices (bus, gen, branch)
    def extract_matrix(name: str) -> np.ndarray:
        pattern = rf'mpc\.{name}\s*=\s*\[(.*?)\];'
        match = re.search(pattern, case_text, re.DOTALL)
        if not match:
            return np.array([])
            
        matrix_str = match.group(1)
        rows = []
        for line in matrix_str.split('\n'):
            line = line.split('%')[0].strip()
            if not line:
                continue
            # Replace MATLAB semicolons/commas with spaces for uniform string splitting
            line = line.replace(';', ' ').replace(',', ' ')
            row_vals = [float(val) for val in line.split() if val]
            if row_vals:
                rows.append(row_vals)
        return np.array(rows)

    bus = extract_matrix('bus')
    gen = extract_matrix('gen')
    branch = extract_matrix('branch')

    if bus.size == 0 or branch.size == 0 or gen.size == 0:
        raise ValueError("Failed to parse mpc.bus, mpc.branch, or mpc.gen from MATPOWER text.")

    return {
        "baseMVA": baseMVA,
        "bus": bus,
        "gen": gen,
        "branch": branch
    }

# ---------- Helper: write case text to a temp .m ----------
def _write_case_text(case_text: str, case_name: str) -> str:
    safe = "".join(c for c in case_name if c.isalnum() or c == "_") or "case_tmp"
    tmpdir = tempfile.mkdtemp(prefix="matpower_case_")
    fpath = os.path.join(tmpdir, f"{safe}.m")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(case_text)
    return fpath

# ---------- (lazy) imports for topology correction helpers ----------
def _lazy_import_topology_helpers():
    """
    Import node-breaker helpers on demand. Returns tuple (nb_module, load_case_fn, nb_to_operator_fn).
    """
    try:
        import sys as _sys
        import os as _os
        repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), os.pardir))
        if repo_root not in _sys.path:
            _sys.path.append(repo_root)
        from Transmission import nodebreaker_pp14 as nb  # type: ignore
        from Transmission.generate_measurements import load_case as _gm_load_case, _nb_to_operator_z  # type: ignore
        import pandapower as pp  # type: ignore
        return nb, _gm_load_case, _nb_to_operator_z, pp
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Topology helpers not available: {e}") from e



# ---------- WLS (LagrangianM_singlephase Port) ----------
def _load_python_case(case_path: str) -> Dict[str, Any]:
    """Helper to load a case file (.m) using our regex parser."""
    import os
    if not os.path.isfile(case_path):
        # Try to resolve built-in cases if needed, otherwise string
        # e.g case14.m from mcp_server/ 
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        
        # Check Transmission first just in case
        paths_to_try = [
            os.path.join(repo_root, "mcp_server", f"{case_path}.m"),
            os.path.join(repo_root, "mcp_server", f"{case_path}"),
            os.path.join(repo_root, "Transmission", f"{case_path}.m"),
            os.path.join(repo_root, "Transmission", f"{case_path}")
        ]
        
        found = False
        for p in paths_to_try:
            if os.path.isfile(p):
                case_path = p
                found = True
                break
                
        if not found:
             raise FileNotFoundError(f"Case file not found: {case_path}")

    with open(case_path, 'r', encoding='utf-8') as f:
        case_text = f.read()
    return _parse_matpower_case(case_text)


def _wls_json(case_path: str, z_list: List[float]) -> Dict[str, Any]:  # pragma: no cover
    """
    Python equivalent of LagrangianM_singlephase(z, result, ind, bus_data).
    Bypasses MATLAB engine completely.
    """
    import sys
    import numpy as np

    # Import Python port of the WLS logic
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
        
    try:
        from tools import lagrangian_port
    except ImportError as e:
        raise ImportError(f"Lagrangian port not found: {e}") from e

    # Parse case file into dictionary format
    ppc = _load_python_case(case_path)
    bus_data = ppc["bus"]
    
    # Run WLS using purely Python
    z_arr = np.array(z_list, dtype=float)
    
    nb = ppc["bus"].shape[0]
    nl = ppc["branch"].shape[0]
    expN = 3*nb + 4*nl
    if len(z_arr) != expN:
        raise ValueError(f"WLS input error: |z|={len(z_arr)}, expected {expN} (=3*nb + 4*nl).")
        
    try:
        lambdaN, success, r, lambda_vec, ea = lagrangian_port.lagrangian_m_singlephase(
            z=z_arr,
            result=ppc,
            ind=0,
            bus_data=bus_data,
        )
        
        # Format identical to the previous MATLAB response
        r_list = r.tolist() if isinstance(r, np.ndarray) else list(r)
        return {
            "success": bool(success),
            "lambdaN": lambdaN.tolist(),
            "r": r_list,
            "global_residual_sum": float(np.sum(np.asarray(r_list, dtype=float) ** 2)),
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ---------- Measurement Error Correction (LagrangianM_correct) ----------
def _meas_correction_json(
    case_path: str,
    z_list: List[float],
    *,
    suspect_group: List[int] | None = None,
    enable_correction: bool = True,
    max_correction_iterations: int = 2,
    error_tolerance: float = 1e-3,
    R_variances_full: List[float] | None = None,
) -> Dict[str, Any]:  # pragma: no cover
    """
    Measurement-error correction using Python lagrangian_correct_port.py

    Mirrors the WLS+NLM tool flow:
    - Loads the case via pure Python regex loadcase
    - Expects full measurement vector ordered as [Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)]
    - Optionally accepts a suspect group of global indices for grouped correction

    Returns a dict with keys: success, r_norm, resid_raw, lambdaN, and optionally Omega or Omega_shape,
    plus z_corrected_info (as produced by the Python routine).
    """
    import sys
    import numpy as np

    # Import Python port of the correct logic
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
        
    try:
        from tools import lagrangian_correct_port
    except ImportError as e:
        raise ImportError(f"Lagrangian correct port not found: {e}") from e

    # Parse case file into dictionary format
    ppc = _load_python_case(case_path)
    bus_data = ppc["bus"]
    nb = ppc["bus"].shape[0]
    nl = ppc["branch"].shape[0]
    nz = 3*nb + 4*nl

    if len(z_list) != nz:
        raise ValueError(f"Correction input error: |z|={len(z_list)}, expected {nz} (=3*nb + 4*nl).")
        
    z_arr = np.array(z_list, dtype=float)
    
    # Training traces pass 0-based global measurement indices. Convert here so the underlying
    # routine still receives the 1-based convention it expects.
    suspect_group_0 = [] if suspect_group is None else [int(i) for i in suspect_group]
    suspect_group_1 = []
    for idx0 in suspect_group_0:
        if idx0 < 0 or idx0 >= nz:
            raise ValueError(f"suspect_group index {idx0} outside valid range [0, {nz-1}]")
        suspect_group_1.append(idx0 + 1)

    options = {
        "enable_group_correction": enable_correction,
        "correction_group_full_indices": suspect_group_1,
        "max_correction_iterations": max_correction_iterations,
        "correction_error_tolerance": error_tolerance,
        "group_indices_are_one_based": True,
    }
    
    if R_variances_full is None:
        R_variances_full_arr = np.zeros(nz, dtype=float)
        R_variances_full_arr[0:nb] = (0.001)**2
        R_variances_full_arr[nb:3*nb] = (0.01)**2
        R_variances_full_arr[3*nb:3*nb+4*nl] = (0.01)**2
    elif len(R_variances_full) != nz:
        raise ValueError(f"R_variances length={len(R_variances_full)} mismatch expected {nz}")
    else:
        R_variances_full_arr = np.array(R_variances_full, dtype=float)

    try:
        lambdaN, success, r_norm, Omega, final_resid_raw, z_corrected_info = \
            lagrangian_correct_port.lagrangian_m_correct(
                z_in_full=z_arr,
                result=ppc,
                ind=0,
                bus_data=bus_data,
                options=options,
                R_variances_full_in=R_variances_full_arr,
            )
            
        S = {
            "success": bool(success),
            "lambdaN": lambdaN.tolist() if isinstance(lambdaN, np.ndarray) else list(lambdaN),
            "r_norm": r_norm.tolist() if isinstance(r_norm, np.ndarray) else list(r_norm),
            "resid_raw": final_resid_raw.tolist() if isinstance(final_resid_raw, np.ndarray) else list(final_resid_raw),
            "z_corrected_info": z_corrected_info,
        }
        
        if Omega.size <= 4000:
            S["Omega"] = Omega.tolist() if isinstance(Omega, np.ndarray) else list(Omega)
        else:
            S["Omega_shape"] = list(Omega.shape)
            
        # Post-process JSON to add a flattened, normalized summary of corrected measurements
        def _flatten(x):
            if isinstance(x, list):
                out = []
                for v in x:
                    out.extend(_flatten(v))
                return out
            elif isinstance(x, np.ndarray):
                return x.flatten().tolist()
            return [x] if x is not None else []
            
        zci = z_corrected_info or {}
        idxs = _flatten(zci.get("last_corrected_global_indices") or [])
        orig_vals = _flatten(zci.get("last_original_values") or [])
        corr_vals = _flatten(zci.get("last_corrected_values") or [])
        err_vals = _flatten(zci.get("last_estimated_errors") or [])
        
        corrected = []
        n = min(len(idxs), len(corr_vals))
        for j in range(n):
            try:
                idx1 = int(round(float(idxs[j])))
                idx0 = idx1 - 1
            except Exception:
                continue
            rec = {
                "index1": idx1,
                "index0": idx0,
                "corrected": float(corr_vals[j]) if j < len(corr_vals) else None,
                "original": float(orig_vals[j]) if j < len(orig_vals) else None,
                "estimated_error": float(err_vals[j]) if j < len(err_vals) else None,
            }
            corrected.append(rec)
            
        S["corrected_measurements"] = corrected
        S["applied_any_correction"] = bool(zci.get("applied_any_correction", False))
        S["iterations_performed"] = int(zci.get("iterations_performed", 0))
        S["suspect_group_zero_based"] = suspect_group_0
        
        return S

    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ---------- Parameter Error Correction (multi-scan) ----------
def _param_correction_json(
    case_path: str,
    line_index: int,
    z_scans: List[List[float]],
    initial_states: List[List[float]],
    R_variances_full: List[float] | None = None,
) -> Dict[str, Any]:  # pragma: no cover
    """
    Correct a single line's parameters [R, X] using multiple measurement snapshots (Pure Python).
    """
    import sys
    import numpy as np
    
    # Import Python port of the multi-scan param correction
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
        
    try:
        import tools.correct_parameter_group_multi_scan_port as param_port
    except ImportError as e:
        raise ImportError(f"Parameter correction port not found: {e}") from e

    # Parse case file into dictionary format
    ppc = _load_python_case(case_path)
    nb = ppc["bus"].shape[0]
    nl = ppc["branch"].shape[0]
    nz = 3*nb + 4*nl
    baseMVA = float(ppc["baseMVA"])

    line_idx = int(line_index)
    
    # Normalize z_scans to matrix: shape (nz, s)
    if not z_scans or not isinstance(z_scans[0], (list, tuple)):
        raise ValueError("z_scans must be a 2D list: scans x nz or nz x scans")
    s_dim0 = len(z_scans)
    s_dim1 = len(z_scans[0])
    if s_dim0 == nz:
        # already (nz x s)
        z_mat = np.array(z_scans, dtype=float)
        s = s_dim1
    elif s_dim1 == nz:
        # (s x nz) -> (nz x s)
        z_mat = np.array(z_scans, dtype=float).T
        s = s_dim0
    else:
        raise ValueError(f"z_scans shape not compatible with nz={nz} (got {s_dim0}x{s_dim1})")

    # Normalize initial_states to shape (2*nb, s)
    if not initial_states or not isinstance(initial_states[0], (list, tuple)):
        raise ValueError("initial_states must be a 2D list: scans x 2*nb or 2*nb x scans")
    ist_dim0 = len(initial_states)
    ist_dim1 = len(initial_states[0])
    two_nb = 2 * nb
    if ist_dim0 == two_nb:
        ist_mat = np.array(initial_states, dtype=float)
        s_states = ist_dim1
    elif ist_dim1 == two_nb:
        ist_mat = np.array(initial_states, dtype=float).T
        s_states = ist_dim0
    else:
        raise ValueError(f"initial_states shape not compatible with 2*nb={two_nb} (got {ist_dim0}x{ist_dim1})")
    if s_states != s:
        raise ValueError(f"number of scans mismatch between z_scans (s={s}) and initial_states (s={s_states})")

    # Provide variances vector
    if R_variances_full is None:
        R_variances = np.zeros(nz, dtype=float)
        R_variances[0:nb] = (0.001)**2
        R_variances[nb:3*nb] = (0.01)**2
        R_variances[3*nb:3*nb+4*nl] = (0.01)**2
    else:
        if len(R_variances_full) != nz:
            raise ValueError(f"R_variances_full length {len(R_variances_full)} != expected nz {nz}")
        R_variances = np.array(R_variances_full, dtype=float)

    try:
        corrected_params, success = param_port.correct_parameter_group_multi_scan(
            mpc_with_error=ppc,
            line_to_correct_idx=line_idx,
            multi_scan_measurements_z=z_mat,
            initial_states_multi_scan=ist_mat,
            R_variances_vec=R_variances,
            baseMVA=baseMVA,
            line_index_is_one_based=True, # preserve original signature semantics
        )
        
        # 1-based indexing for internal retrieval (following MATLAB expectations)
        safe_line_idx = line_idx - 1 if line_idx > 0 else 0
        from_bus = float(ppc["branch"][safe_line_idx, 0])
        to_bus = float(ppc["branch"][safe_line_idx, 1])

        S = {
            "success": bool(success),
            "corrected_params": corrected_params.tolist() if isinstance(corrected_params, np.ndarray) else list(corrected_params),
            "meta": {
                "line_index": line_idx,
                "from_bus": from_bus,
                "to_bus": to_bus,
                "nb": nb,
                "nl": nl,
                "scans": s
            }
        }
        return S

    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}




# ---------- Tools: Measurement Error Correction ----------
@mcp.tool(name="correct_measurements_from_path")
def correct_measurements_from_path(
    *,
    case_path: str,
    z: List[float],
    suspect_group: List[int] | None = None,
    enable_correction: bool = True,
    max_correction_iterations: int = 2,
    error_tolerance: float = 1e-3,
    R_variances_full: List[float] | None = None,
) -> Dict[str, Any]:
    """
    Grouped measurement-error correction using Transmission/LagrangianM_correct.m.

    Behavior
    - Mirrors WLS+NLM tool semantics: loads the case via loadcase, checks vector length,
      and invokes the MATLAB routine. Accepts optional suspect group indices
      (0-based or 1-based) and basic correction parameters.

    Inputs
    - case_path: MATPOWER case name/path (e.g., 'case14').
    - z: Full measurement vector ordered as [Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)].
    - suspect_group (optional): list of 0-based global indices indicating the group to correct.
    - enable_correction (default True), max_correction_iterations (default 2), error_tolerance (default 1e-3).
    - R_variances_full (optional): full variance vector; defaults applied if omitted.

    Returns
    - success: boolean
    - r_norm: normalized residuals (kept ordering)
    - resid_raw: raw residuals
    - lambdaN: normalized multipliers
    - Omega or Omega_shape: KKT/Gain information (omitted if too large)
    - z_corrected_info: diagnostic struct with correction details
    """
    try:
        return _meas_correction_json(
            case_path,
            z,
            suspect_group=suspect_group,
            enable_correction=enable_correction,
            max_correction_iterations=max_correction_iterations,
            error_tolerance=error_tolerance,
            R_variances_full=R_variances_full,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(name="correct_measurements_from_text")
def correct_measurements_from_text(
    *,
    case_name: str,
    case_text: str,
    z: List[float],
    suspect_group: List[int] | None = None,
    enable_correction: bool = True,
    max_correction_iterations: int = 2,
    error_tolerance: float = 1e-3,
    R_variances_full: List[float] | None = None,
) -> Dict[str, Any]:
    """
    Same as correct_measurements_from_path, but accepts inline case.m text.
    'case_name' must match the function name inside the .m.
    """
    path = _write_case_text(case_text, case_name)
    return correct_measurements_from_path(
        case_path=str(path),
        z=z,
        suspect_group=suspect_group,
        enable_correction=enable_correction,
        max_correction_iterations=max_correction_iterations,
        error_tolerance=error_tolerance,
        R_variances_full=R_variances_full,
    )


# ---------- Tools: Parameter Error Correction (multi-scan) ----------
@mcp.tool(name="correct_parameters_from_path")
def correct_parameters_from_path(
    *,
    case_path: str,
    line_index: int,
    z_scans: List[List[float]],
    initial_states: List[List[float]],
    R_variances_full: List[float] | None = None,
) -> Dict[str, Any]:
    """
    Correct a single line's parameters [R, X] using multiple measurement snapshots (multi-scan ASE).

    Behavior
    - Follows existing tool conventions: loads the case via loadcase and calls
      Transmission/correct_parameter_group_multi_scan.m.
    - Assumes the case already contains the erroneous parameters; this tool estimates corrected [R, X]
      for branch row `line_index` using multiple measurement snapshots.

    Inputs
    - case_path: MATPOWER case name/path (e.g., 'case14').
    - line_index: **1-based** MATPOWER branch row index.
      If you have a 0-based Python index `line_row`, pass `line_row + 1`.
    - z_scans: Measurement snapshots. Each scan is a full vector ordered as
      [Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)]. Provide either
      shape (s x nz) [preferred] or (nz x s) - the tool normalizes to (nz x s).
    - initial_states: Initial states per scan as [V(1..nb); angle_deg(1..nb)] for each scan.
      Provide either shape (s x 2*nb) [preferred] or (2*nb x s) — normalized internally.
    - R_variances_full (optional): Full variances (length nz); defaults will be applied if omitted.

    Returns
    - success: boolean
    - corrected_params: [R_est, X_est]
    - meta: {line_index, from_bus, to_bus, nb, nl, scans}

    Notes
    - The calling agent should request measurement snapshots from the user after detecting a parameter error;
      this tool does not synthesize scans.
    """
    return _param_correction_json(
        case_path,
        line_index,
        z_scans,
        initial_states,
        R_variances_full,
    )


@mcp.tool(name="correct_parameters_from_text")
def correct_parameters_from_text(
    *,
    case_name: str,
    case_text: str,
    line_index: int,
    z_scans: List[List[float]],
    initial_states: List[List[float]],
    R_variances_full: List[float] | None = None,
) -> Dict[str, Any]:
    """
    Same as correct_parameters_from_path, but accepts inline case.m text.
    'case_name' must match the function name inside the .m.
    """
    path = _write_case_text(case_text, case_name)
    return _param_correction_json(
        path,
        line_index,
        z_scans,
        initial_states,
        R_variances_full,
    )

# ---------- Tools: Harmonic State Estimation ----------
def _lazy_import_hse_utils():
    try:
        import sys as _sys
        import os as _os
        repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
        if repo_root not in _sys.path:
            _sys.path.append(repo_root)
        from Harmonics import hse_utils  # type: ignore
        from Harmonics import ieee14_verification as h_ver_global # fallback if needed to set constants
        return hse_utils
    except Exception as e:
        raise ImportError(f"HSE utils not available: {e}") from e


def _run_hse_logic(
    case_path: str,
    harmonic_measurements: List[Dict[str, Any]],
    harmonic_orders: List[int],
    slack_bus: int = 0
) -> Dict[str, Any]:
    """Internal implementation of HSE logic."""
    import numpy as np
    import math
    try:
        hse = _lazy_import_hse_utils()
        
        # 1. Load Case Data (bus, branch, baseMVA) via pure Python regex parser
        ppc = _load_python_case(case_path)
        
        bus_mat = np.array(ppc["bus"])
        branch_mat = np.array(ppc["branch"])
        base_mva = float(ppc["baseMVA"])
        
        # 2. Parse Measurements
        # hse_utils expects: Vh_meas_by_h[h] = (buses0, Vmeas, sigma)
        Vh_meas_by_h = {}
        
        # Group by harmonic
        grouped = {}
        for m in harmonic_measurements:
            h = int(m["h"])
            if h not in grouped:
                grouped[h] = {"buses": [], "V": [], "sigma": []}
            
            b_idx = int(m["bus"]) - 1 # to 0-based
            
            if "V_real" in m and "V_imag" in m:
                v_complex = complex(m["V_real"], m["V_imag"])
            else:
                deg = float(m.get("Va_deg", 0.0))
                rad = math.radians(deg)
                mag = float(m["Vm"])
                v_complex = complex(mag * math.cos(rad), mag * math.sin(rad))
                
            sigma = float(m.get("sigma", 1e-4))
            
            grouped[h]["buses"].append(b_idx)
            grouped[h]["V"].append(v_complex)
            grouped[h]["sigma"].append(sigma)
            
        for h, data in grouped.items():
            Vh_meas_by_h[h] = (
                np.array(data["buses"], dtype=int),
                np.array(data["V"], dtype=complex),
                np.array(data["sigma"], dtype=float)
            )
            
        # 3. Running HSE
        best_bus, ranking, I_source_hat, Vhat_by_h = hse.harmonic_source_hse_single_source_scan(
            bus=bus_mat,
            branch=branch_mat,
            base_mva=base_mva,
            harmonic_orders=harmonic_orders,
            Vh_meas_by_h=Vh_meas_by_h,
            slack_bus=slack_bus,
            candidate_buses_1based=None
        )
        
        # 4. THD Calculation (need fundamental voltages)
        Vm = bus_mat[:, 7]
        Va = np.radians(bus_mat[:, 8])
        V1 = Vm * (np.cos(Va) + 1j * np.sin(Va))
        
        thd_est = hse.compute_thd_from_states(Vhat_by_h, [1] + harmonic_orders, V1)
        
        # 5. Serialize Output
        def c2l(c): return [float(c.real), float(c.imag)]
        
        return {
            "success": True,
            "best_candidate_bus_1based": int(best_bus) if best_bus else None,
            "ranking_top10": ranking[:10] if ranking else [],
            "estimated_injections": {str(h): c2l(val) for h, val in I_source_hat.items()},
            "estimated_thd_percent": {str(i+1): float(t*100) for i, t in enumerate(thd_est)}
        }
        
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _infer_harmonic_orders(harmonic_measurements: List[Dict[str, Any]]) -> List[int]:
    orders = sorted({int(m["h"]) for m in harmonic_measurements if int(m.get("h", 0)) > 1})
    if not orders:
        raise ValueError("Could not infer harmonic_orders from harmonic_measurements.")
    return orders


@mcp.tool(name="run_hse_from_path")
def run_hse_from_path(
    *,
    case_path: str,
    harmonic_measurements: List[Dict[str, Any]],
    harmonic_orders: List[int] | None = None,
    slack_bus: int = 0,
) -> Dict[str, Any]:
    """
    Run Harmonic State Estimation (HSE) to identify a single harmonic source.
    
    Inputs:
    - case_path: MATPOWER case name/path (e.g. 'case14').
    - harmonic_measurements: List of measurements. Each item is a dict:
        { "h": int, "bus": int, "Vm": float, "Va_deg": float, "sigma": float }
      where 'bus' is 1-based index (MATPOWER convention), Vm is magnitude, Va_deg is phase in degrees.
      OR provided as direct complex if "V_real" and "V_imag" are keys.
    - harmonic_orders (optional): List of harmonics to consider (e.g. [5, 7, 11, ...]).
      If omitted, the server infers them from the `h` fields inside harmonic_measurements.
    - slack_bus: 0-based index of slack bus (default 0).

    Returns:
    - success: bool
    - full_ranking: List of {bus_1based, score} sorted by likelihood (lower score = better).
    - best_candidate: bus_1based of likely source.
    - estimated_injections: {h: [real, imag]} at best bus.
    - est_thd: {bus_1based: thd_percent} estimated across network.
    """
    orders = harmonic_orders if harmonic_orders else _infer_harmonic_orders(harmonic_measurements)
    return _run_hse_logic(case_path, harmonic_measurements, orders, slack_bus)

@mcp.tool(name="correct_topology_from_path")
def correct_topology_from_path(
    *,
    case_path: str,
    cb_name: str,
    desired_status: bool | str = False,
) -> Dict[str, Any]:
    """
    Flip or set a circuit-breaker status in pocket substations, run PF, and return corrected measurements.

    Inputs
    - case_path: MATPOWER case name/path (e.g., 'case14').
    - cb_name: CB identifier as exposed by nodebreaker_pp14 (e.g., 'CB_1_N1_N2').
    - desired_status: True/closed or False/open (string 'open'/'closed' also accepted).

    Returns
    - success: boolean
    - z_corrected: corrected measurement vector (operator order) if success
    - cb_name, old_status, new_status
    - error/skipped_reason if not successful
    """
    try:
        nb, load_case_fn, nb_to_operator_fn, pp = _lazy_import_topology_helpers()
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": str(e)}

    # normalize desired_status
    ds = desired_status
    if isinstance(ds, str):
        ds = ds.strip().lower()
        if ds in ("open", "false", "0"):
            ds_bool = False
        elif ds in ("closed", "true", "1"):
            ds_bool = True
        else:
            return {"success": False, "error": f"invalid desired_status={desired_status}"}
    else:
        ds_bool = bool(ds)

    # Build NB net with the requested CB status
    try:
        status_map = {cb_name: ds_bool}
        net, sec_bus, cb_idx, line_idx, trafo_idx = nb.build_nb_ieee14_pocket123(status_map=status_map)
    except Exception as e:
        return {"success": False, "error": f"build_nb failed: {e}"}

    # Run PF with a few fallbacks for robustness
    pf_errors: List[str] = []
    for kwargs in (
        {"init": "dc", "max_iteration": 20},
        {"init": "flat", "max_iteration": 40},
        {"init": "results", "max_iteration": 40},
    ):
        try:
            pp.runpp(net, calculate_voltage_angles=True, **kwargs)
            break
        except Exception as err:  # pragma: no cover - diagnostic path
            pf_errors.append(f"{kwargs}: {err}")
            continue
    else:
        return {"success": False, "error": f"PF failed: {' | '.join(pf_errors)}"}

    # Map PB measurements to operator z order
    try:
        # case_path may be 'case14' or '14'
        case_name_only = case_path.replace("case", "") if isinstance(case_path, str) else case_path
        ppc_base = load_case_fn(str(case_name_only))
        z_corr = nb_to_operator_fn(net, line_idx, trafo_idx, ppc_base)
    except Exception as e:
        return {"success": False, "error": f"mapping failed: {e}"}

    return {
        "success": True,
        "z_corrected": z_corr.tolist(),
        "cb_name": cb_name,
        "old_status": None,  # not tracked here
        "new_status": ds_bool,
    }

@mcp.tool(name="wls_from_path")
def wls_from_path(*, case_path: str, z: List[float]) -> Dict[str, Any]:
    """
    Weighted least-squares state estimation with normalized Lagrange multipliers (WLS+NLM).

    Behavior
    - Loads the MATPOWER case (name/path) and runs the pure-Python WLS port.
    - Expects a full measurement vector ordered as [Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)].

    Inputs
    - case_path: Path or case name resolvable by the server loader (e.g., 'case14').
    - z: Full measurement vector length 3*nb + 4*nl.

    Returns
    - success: boolean
    - r: normalized residual vector (same ordering as inputs)
    - lambdaN: normalized multipliers (typically length 2*nl)
    Note: large diagnostic arrays (EA, lambda_vec) are omitted to reduce payload size.
    """
    return _wls_json(case_path, z)

@mcp.tool(name="wls_from_text")
def wls_from_text(*, case_name: str, case_text: str, z: List[float]) -> Dict[str, Any]:
    """
    Same as wls_from_path, but accepts inline case.m contents.
    'case_name' must match the function name inside the .m file.
    """
    path = _write_case_text(case_text, case_name)
    return _wls_json(path, z)

if __name__ == "__main__":
    # Bind to a stable HTTP port so clients (build_sft_traces.py) can call reliably
    mcp.run(transport="http", host="127.0.0.1", port=3929)


