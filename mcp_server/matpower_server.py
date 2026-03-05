# matpower_mcp_server_fixed.py
from __future__ import annotations

import os
import json
import tempfile
import threading
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP

mcp = FastMCP("MATPOWER Power Flow (FastMCP v2)")

# NOTE ABOUT MATLAB ENGINE (Python 3.13):
# Importing matlab.engine can crash CPython 3.13 on some setups.
# To keep the server robust, we avoid importing matlab.engine at module import time by
# lazily importing it inside _get_engine(). The WLS tools use the Python engine.

# ---------- (Optional) MATLAB engine support (disabled by default) ----------
_ENG = None
_ENG_LOCK = threading.Lock()
_MATPOWER_READY = False
_CONSTANTS_DEFINED = False

def _get_engine(startup_options: str | None = None):  # pragma: no cover
    """Lazy-import and start MATLAB engine if available.
    """
    global _ENG
    with _ENG_LOCK:
        if _ENG is None:
            # Lazy import to avoid interpreter crash on some Python versions
            import matlab.engine  # type: ignore
            _ENG = matlab.engine.start_matlab(startup_options or "")
        return _ENG

# ---------- MATPOWER path handling (non-invasive) ----------
def _ensure_matpower_visible(eng) -> None:  # pragma: no cover
    """Ensure MATLAB sees both user .m files and MATPOWER functions.

    Always add the repo paths first so helpers like LagrangianM_singlephase.m
    are discoverable even when MATPOWER is already on the path. Then ensure
    runpf is available (via existing path or $MATPOWER_PATH).
    """
    global _MATPOWER_READY
    if _MATPOWER_READY:
        return

    # Always add server and repo root (Transmission/*.m) so user functions are visible
    server_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(server_dir, os.pardir))
    try:
        eng.addpath(eng.genpath(server_dir), nargout=0)
        eng.addpath(eng.genpath(repo_root), nargout=0)
    except Exception:
        pass

    # If MATPOWER not visible yet, try MATPOWER_PATH
    if not eng.which("runpf"):
        env = os.environ.get("MATPOWER_PATH")
        if env:
            try:
                eng.addpath(eng.genpath(env), nargout=0)
            except Exception:
                pass

    if not eng.which("runpf"):
        raise RuntimeError(
            "MATPOWER not found in MATLAB path. Either add it to MATLAB's path, "
            "or set MATPOWER_PATH env var before starting the server."
        )

    _MATPOWER_READY = True


def _ensure_constants(eng) -> None:  # pragma: no cover
    global _CONSTANTS_DEFINED
    if _CONSTANTS_DEFINED:
        return
    eng.eval("define_constants;", nargout=0)
    _CONSTANTS_DEFINED = True

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

# ---------- Core: run PF in MATLAB and return JSON (no sparse crossing) ----------
def _pf_json(eng, case_path: str, *, dc: bool) -> Dict[str, Any]:  # pragma: no cover
    _ensure_matpower_visible(eng)  # as in your fixed server

    # Use MATLAB‑valid variable names
    eng.workspace["CasePath"] = case_path
    eng.workspace["IsDC"] = float(1 if dc else 0)

    _ensure_constants(eng)
    eng.eval("""
        mpopt = mpoption('verbose', 0, 'out.all', 0);
        if IsDC ~= 0
            mpopt = mpoption(mpopt, 'model', 'DC');  % DC power flow option
        end

        [res, ~] = runpf(CasePath, mpopt);  % AC (or DC if set above)

        S = struct();
        S.success    = res.success;
        S.et         = res.et;
        S.iterations = res.iterations;
        S.baseMVA    = res.baseMVA;

        % Ensure exported arrays are FULL (avoid sparse crossing into Python)
        S.buses    = full([res.bus(:, BUS_I), res.bus(:, VM), res.bus(:, VA)]);
        S.branches = full([res.branch(:, F_BUS), res.branch(:, T_BUS), ...
                           res.branch(:, PF), res.branch(:, PT), ...
                           res.branch(:, QF), res.branch(:, QT)]);
        S.gens     = full([res.gen(:, GEN_BUS), res.gen(:, PG), res.gen(:, QG)]);

        JsonOut = jsonencode(S);
    """, nargout=0)

    json_str = eng.workspace["JsonOut"]  # fetch as Python str
    return json.loads(json_str)

def _opf_json(eng, case_path: str) -> Dict[str, Any]:  # pragma: no cover
    """Run AC OPF via runopf and return JSON with objective, LMPs, KKT multipliers, flows, etc."""
    _ensure_matpower_visible(eng)
    eng.workspace["CasePath"] = case_path
    _ensure_constants(eng)  # provides BUS/GEN/BRANCH column indices, incl. LAM_P, MU_*
    eng.eval(r"""
        mpopt = mpoption('verbose', 0, 'out.all', 0);  % quiet run :contentReference[oaicite:2]{index=2}
        [res, success] = runopf(CasePath, mpopt);      % AC OPF by default :contentReference[oaicite:3]{index=3}

        S = struct();
        S.success = success;
        S.et      = res.et;
        S.f       = res.f;             % objective value
        S.baseMVA = res.baseMVA;

        % Export key results (use FULL to avoid sparse crossing)
        % Buses: [BUS_I, VM, VA, LAM_P, LAM_Q]
        S.buses = full([res.bus(:, BUS_I), res.bus(:, VM), res.bus(:, VA), ...
                        res.bus(:, LAM_P), res.bus(:, LAM_Q)]);

        % Generators: [GEN_BUS, PG, QG, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN]
        S.gens = full([res.gen(:, GEN_BUS), res.gen(:, PG), res.gen(:, QG), ...
                       res.gen(:, MU_PMAX), res.gen(:, MU_PMIN), ...
                       res.gen(:, MU_QMAX), res.gen(:, MU_QMIN)]);

        % Branches: [F_BUS, T_BUS, PF, QF, PT, QT, MU_SF, MU_ST]
        S.branches = full([res.branch(:, F_BUS), res.branch(:, T_BUS), ...
                           res.branch(:, PF), res.branch(:, QF), ...
                           res.branch(:, PT), res.branch(:, QT), ...
                           res.branch(:, MU_SF), res.branch(:, MU_ST)]);

        JsonOut = jsonencode(S);
    """, nargout=0)
    return json.loads(eng.workspace["JsonOut"])

# ---------- WLS (your LagrangianM_singlephase) ----------
def _wls_json(eng, case_path: str, z_list: List[float]) -> Dict[str, Any]:  # pragma: no cover
    """
    Call user's LagrangianM_singlephase(z, result, ind, bus_data).
    Assumes the function is visible on MATLAB path (check with which()).
    """
    _ensure_matpower_visible(eng)

    # Ensure function is reachable
    if not eng.which("LagrangianM_singlephase"):
        raise RuntimeError(
            "LagrangianM_singlephase.m not found on MATLAB path. "
            "Add it to the path (addpath) or place it alongside the server."
        )

    # Put inputs in MATLAB workspace
    eng.workspace["CasePath"] = case_path
    # z as 1-by-N double row (function normalizes orientation internally)
    import matlab as ml
    eng.workspace["Z"] = ml.double([float(x) for x in z_list])
    eng.workspace["Ind"] = 0

    # Build case, then call the user function; return JSON
    eng.eval(r"""
        mpc = loadcase(CasePath);                        % load case (struct)
        bus_data = mpc.bus;                              % initial Vm/Va (cols VM=8, VA=9)

        % Basic dimension check vs expected ordering used by your code:
        nb = size(mpc.bus,1); nl = size(mpc.branch,1);
        expN = 3*nb + 4*nl;  % [Vm(nb), Pinj(nb), Qinj(nb), Pf/Qf/Pt/Qt(4nl)]
        if numel(Z) ~= expN
            error("WLS input error: |z|=%d, expected %d (=3*nb + 4*nl).", numel(Z), expN);
        end

        [lambdaN, success, r, lambda_vec, ea] = LagrangianM_singlephase(Z, mpc, Ind, bus_data);

        S = struct();
        S.success    = success;
        S.lambdaN    = full(lambdaN);
        S.r          = full(r); % NOTE: this is the normalized residual
        % S.lambda_vec = full(lambda_vec);   % (omitted to reduce payload size)
        % S.ea         = full(ea);           % (omitted to reduce payload size)
        JsonOut = jsonencode(S);
    """, nargout=0)
    return json.loads(eng.workspace["JsonOut"]) 



# ---------- Measurement Error Correction (LagrangianM_correct) ----------
def _meas_correction_json(
    eng,  # matlab.engine.MatlabEngine
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
    Measurement-error correction using Transmission/LagrangianM_correct.m

    Mirrors the WLS+NLM tool flow:
    - Loads the case via loadcase (no OPF)
    - Expects full measurement vector ordered as [Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)]
    - Optionally accepts a suspect group of global indices for grouped correction

    Returns a dict with keys: success, r_norm, resid_raw, lambdaN, and optionally Omega or Omega_shape,
    plus z_corrected_info (as produced by the MATLAB routine).
    """
    _ensure_matpower_visible(eng)

    if not eng.which("LagrangianM_correct"):
        raise RuntimeError(
            "LagrangianM_correct.m not found on MATLAB path. "
            "Ensure Transmission/ is on path or place the file alongside the server."
        )

    # Put inputs in MATLAB workspace
    eng.workspace["CasePath"] = case_path
    import matlab as ml  # type: ignore
    eng.workspace["Z"] = ml.double([float(x) for x in z_list])
    if suspect_group:
        sg = [int(i) for i in suspect_group]
        if any(i == 0 for i in sg):
            sg = [i + 1 for i in sg]
        eng.workspace["SuspectGroup"] = ml.double(sg)
    else:
        eng.workspace["SuspectGroup"] = ml.double([])
    eng.workspace["EnableCorr"] = float(1 if enable_correction else 0)
    eng.workspace["MaxCorrIter"] = float(max_correction_iterations)
    eng.workspace["ErrTol"] = float(error_tolerance)

    # Load case, check dimensions, build defaults, call MATLAB function
    eng.eval(r"""
        mpc = loadcase(CasePath);
        bus_data = mpc.bus;
        nb = size(mpc.bus, 1); nl = size(mpc.branch, 1);
        nz = 3*nb + 4*nl;
        if numel(Z) ~= nz
            error('Correction input error: |z|=%d, expected %d (=3*nb + 4*nl).', numel(Z), nz);
        end
    """, nargout=0)

    # Provide R variances
    if R_variances_full is None:
        eng.eval(r"""
            R_variances_full = zeros(nz, 1);
            R_variances_full(1:nb) = (0.001)^2;              % Vm
            R_variances_full(nb+1:3*nb) = (0.01)^2;        % P/Q inj
            R_variances_full(3*nb+1:3*nb+4*nl) = (0.01)^2; % Pf/Qf/Pt/Qt
        """, nargout=0)
    else:
        eng.workspace["R_variances_full_py"] = ml.double([float(x) for x in R_variances_full])
        eng.eval(r"""
            if numel(R_variances_full_py) ~= nz
                error('R_variances length=%d mismatch expected %d', numel(R_variances_full_py), nz);
            end
            R_variances_full = R_variances_full_py(:);
        """, nargout=0)

    eng.eval(r"""
        [lambdaN, success_final, r_norm, Omega, final_resid_raw, z_corrected_info] = ...
            LagrangianM_correct(Z(:), mpc, 0, bus_data, struct( ...
                'enable_group_correction', (EnableCorr ~= 0), ...
                'correction_group_full_indices', SuspectGroup(:)', ...
                'max_correction_iterations', MaxCorrIter, ...
                'correction_error_tolerance', ErrTol), ...
                R_variances_full(:));

        S = struct();
        S.success = logical(success_final);
        S.lambdaN = full(lambdaN(:)');
        S.r_norm = full(r_norm(:)');
        S.resid_raw = full(final_resid_raw(:)');
        if numel(Omega) <= 4000
            S.Omega = full(Omega);
        else
            S.Omega_shape = size(Omega);
        end
        try
            S.z_corrected_info = z_corrected_info;
        catch
            S.z_corrected_info = struct();
        end
        JsonOut = jsonencode(S);
    """, nargout=0)

    # Post-process JSON to add a flattened, normalized summary of corrected measurements
    obj = json.loads(eng.workspace["JsonOut"])  # type: ignore
    def _flatten(x):
        if isinstance(x, list):
            out = []
            for v in x:
                out.extend(_flatten(v))
            return out
        return [x] if x is not None else []
    zci = obj.get("z_corrected_info") or {}
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
    obj["corrected_measurements"] = corrected
    if isinstance(zci, dict):
        obj["applied_any_correction"] = bool(zci.get("applied_any_correction", False))
        obj["iterations_performed"] = int(zci.get("iterations_performed", 0))
    return obj


# ---------- Parameter Error Correction (multi-scan) ----------
def _param_correction_json(
    eng,
    case_path: str,
    line_index: int,
    z_scans: List[List[float]],
    initial_states: List[List[float]],
    R_variances_full: List[float] | None = None,
) -> Dict[str, Any]:  # pragma: no cover
    """
    Correct a single line's parameters [R, X] using multiple measurement snapshots.

    Mirrors existing tool patterns:
    - Loads the case via loadcase (expects the case already contains the erroneous R/X).
    - Expects multiple measurement scans, each a full-length vector ordered as
      [Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)].
    - Initial states per scan are provided as [V(1..nb); angle_deg(1..nb)] for each scan.

    Inputs
    - case_path: MATPOWER case name/path (e.g., 'case14').
    - line_index: **1-based** MATPOWER branch row index of the line to correct.
      If you have a 0-based Python index `line_row`, pass `line_row + 1`.
    - z_scans: Measurement snapshots. Shape can be either
        (s x nz) => s scans of length nz, or (nz x s) => already column-stacked. We normalize to (nz x s).
    - initial_states: Initial [V; angle_deg] per scan. Shape can be
        (s x 2*nb) or (2*nb x s). We normalize to (2*nb x s).
    - R_variances_full (optional): Full-length measurement variances; if omitted, defaults are used.

    Returns
    - success: boolean convergence flag from correct_parameter_group_multi_scan
    - corrected_params: [R_est, X_est]
    - meta: {line_index, from_bus, to_bus, nb, nl, scans}
    """
    _ensure_matpower_visible(eng)

    # Verify MATLAB routine availability
    if not eng.which("correct_parameter_group_multi_scan"):
        raise RuntimeError(
            "correct_parameter_group_multi_scan.m not found on MATLAB path. "
            "Ensure Transmission/ is on path or place the file alongside the server."
        )

    # Load case, determine sizes
    eng.workspace["CasePath"] = case_path
    _ensure_constants(eng)
    eng.eval(r"""
        mpc = loadcase(CasePath);
        nb = size(mpc.bus, 1); nl = size(mpc.branch, 1);
        nz = 3*nb + 4*nl;
        baseMVA = mpc.baseMVA;
        F_BUS = 1; T_BUS = 2;  % define_constants would do this as well
    """, nargout=0)

    # Fetch sizes to condition Python-side shaping logic
    nb = int(eng.eval("nb"))  # type: ignore
    nl = int(eng.eval("nl"))  # type: ignore
    nz = int(eng.eval("nz"))  # type: ignore

    # Normalize line index to 1-based for MATLAB
    line_idx = int(line_index)
    if line_idx == 0:
        line_idx = 1
    if line_idx < 1:
        raise ValueError("line_index must be >= 1 (or 0 to indicate first line)")
    if line_idx > nl:
        raise ValueError(f"line_index={line_idx} exceeds number of branches nl={nl}")

    # Normalize z_scans to shape (nz x s)
    if not z_scans or not isinstance(z_scans[0], (list, tuple)):
        raise ValueError("z_scans must be a 2D list: scans x nz or nz x scans")
    s_dim0 = len(z_scans)
    s_dim1 = len(z_scans[0])
    if s_dim0 == nz:
        # already (nz x s)
        z_mat_rows = z_scans
        s = s_dim1
    elif s_dim1 == nz:
        # (s x nz) -> transpose to (nz x s)
        s = s_dim0
        z_mat_rows = [[float(z_scans[row][col]) for row in range(s)] for col in range(nz)]
    else:
        raise ValueError(f"z_scans shape not compatible with nz={nz} (got {s_dim0}x{s_dim1})")

    # Normalize initial_states to shape (2*nb x s)
    if not initial_states or not isinstance(initial_states[0], (list, tuple)):
        raise ValueError("initial_states must be a 2D list: scans x 2*nb or 2*nb x scans")
    ist_dim0 = len(initial_states)
    ist_dim1 = len(initial_states[0])
    two_nb = 2 * nb
    if ist_dim0 == two_nb:
        ist_rows = initial_states
        s_states = ist_dim1
    elif ist_dim1 == two_nb:
        s_states = ist_dim0
        ist_rows = [[float(initial_states[row][col]) for row in range(s_states)] for col in range(two_nb)]
    else:
        raise ValueError(f"initial_states shape not compatible with 2*nb={two_nb} (got {ist_dim0}x{ist_dim1})")
    if s_states != s:
        raise ValueError(f"number of scans mismatch between z_scans (s={s}) and initial_states (s={s_states})")

    # Provide variables to MATLAB workspace
    import matlab as ml  # type: ignore
    eng.workspace["LineIdx"] = float(line_idx)
    eng.workspace["ZScans"] = ml.double(z_mat_rows)  # rows=nz, cols=s
    eng.workspace["InitStates"] = ml.double(ist_rows)  # rows=2*nb, cols=s

    # Provide variances vector
    if R_variances_full is None:
        eng.eval(r"""
            R_variances_full = zeros(nz, 1);
            % Defaults aligned with Transmission/generate_measurements.py and main_pe_correction.m
            R_variances_full(1:nb) = (0.001)^2;             % Vm
            R_variances_full(nb+1:3*nb) = (0.01)^2;         % P/Q inj
            R_variances_full(3*nb+1:3*nb+4*nl) = (0.01)^2;  % Pf/Qf/Pt/Qt
        """, nargout=0)
    else:
        if len(R_variances_full) != nz:
            raise ValueError(f"R_variances_full length {len(R_variances_full)} != expected nz {nz}")
        eng.workspace["RVariances"] = ml.double([float(x) for x in R_variances_full])
        eng.eval("R_variances_full = RVariances(:);", nargout=0)

    # Invoke the MATLAB routine and package results
    eng.eval(r"""
        % Extract branch info for metadata
        from_bus = mpc.branch(LineIdx, 1); to_bus = mpc.branch(LineIdx, 2);

        [corrected_params_group, success_correction] = correct_parameter_group_multi_scan( ...
            mpc, LineIdx, ZScans, InitStates, R_variances_full, baseMVA);

        S = struct();
        S.success = logical(success_correction);
        S.corrected_params = full(corrected_params_group(:)'); % [R_est, X_est]
        S.meta = struct('line_index', LineIdx, 'from_bus', from_bus, 'to_bus', to_bus, ...
                        'nb', nb, 'nl', nl, 'scans', size(ZScans,2));
        JsonOut = jsonencode(S);
    """, nargout=0)

    return json.loads(eng.workspace["JsonOut"])  # type: ignore

# ---------- Minimal tools  ----------
@mcp.tool(name="run_pf_from_path")
def run_pf_from_path(*, case_path: str) -> Dict[str, Any]:  # pragma: no cover
    """
    AC power flow for a MATPOWER case file (.m or .mat). 'case_path' may be an absolute path,
    relative path, or a bare case name if it's on MATLAB's path.
    """
    eng = _get_engine()
    return _pf_json(eng, case_path, dc=False)

@mcp.tool(name="run_pf_from_text")
def run_pf_from_text(*, case_name: str, case_text: str) -> Dict[str, Any]:  # pragma: no cover
    """
    AC power flow from raw case.m text. 'case_name' must match the function name inside the .m.
    """
    eng = _get_engine()
    path = _write_case_text(case_text, case_name)
    return _pf_json(eng, path, dc=False)

@mcp.tool(name="run_dcpf_from_path")
def run_dcpf_from_path(*, case_path: str) -> Dict[str, Any]:  # pragma: no cover
    """DC power flow for a MATPOWER case file."""
    eng = _get_engine()
    return _pf_json(eng, case_path, dc=True)

@mcp.tool(name="run_dcpf_from_text")
def run_dcpf_from_text(*, case_name: str, case_text: str) -> Dict[str, Any]:  # pragma: no cover
    """DC power flow from raw case.m text."""
    eng = _get_engine()
    path = _write_case_text(case_text, case_name)
    return _pf_json(eng, path, dc=True)

# ---------- Tools: OPF ----------
@mcp.tool(name="run_opf_from_path")
def run_opf_from_path(*, case_path: str) -> Dict[str, Any]:  # pragma: no cover
    """
    Run AC OPF on a MATPOWER case (.m or .mat).
    'case_path' may be an absolute/relative path or a bare case name on MATLAB's path.
    """
    eng = _get_engine()
    return _opf_json(eng, case_path)

@mcp.tool(name="run_opf_from_text")
def run_opf_from_text(*, case_name: str, case_text: str) -> Dict[str, Any]:  # pragma: no cover
    """
    Run AC OPF from raw case.m text. 'case_name' must match the function name inside the .m.
    """
    eng = _get_engine()
    path = _write_case_text(case_text, case_name)
    return _opf_json(eng, path)


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
    - suspect_group (optional): list of global indices indicating the group to correct.
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
        eng = _get_engine()
        return _meas_correction_json(
            eng,
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
    eng = _get_engine()
    return _param_correction_json(
        eng,
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
    eng = _get_engine()
    path = _write_case_text(case_text, case_name)
    return _param_correction_json(
        eng,
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
        eng = _get_engine()
        
        # 1. Load Case Data (BUS, BRANCH, BASEMVA) via MATLAB
        eng.workspace["CasePath"] = case_path
        _ensure_constants(eng)
        eng.eval(r"""
            mpc = loadcase(CasePath);
            bus = mpc.bus;
            branch = mpc.branch;
            baseMVA = mpc.baseMVA;
            % Ensure full matrices
            bus = full(bus);
            branch = full(branch);
        """, nargout=0)
        
        bus_mat = np.array(eng.workspace["bus"])
        branch_mat = np.array(eng.workspace["branch"])
        base_mva = float(eng.workspace["baseMVA"])
        
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


@mcp.tool(name="run_hse_from_path")
def run_hse_from_path(
    *,
    case_path: str,
    harmonic_measurements: List[Dict[str, Any]],
    harmonic_orders: List[int],
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
    - harmonic_orders: List of harmonics to consider (e.g. [5, 7, 11, ...]).
    - slack_bus: 0-based index of slack bus (default 0).

    Returns:
    - success: bool
    - full_ranking: List of {bus_1based, score} sorted by likelihood (lower score = better).
    - best_candidate: bus_1based of likely source.
    - estimated_injections: {h: [real, imag]} at best bus.
    - est_thd: {bus_1based: thd_percent} estimated across network.
    """
    return _run_hse_logic(case_path, harmonic_measurements, harmonic_orders, slack_bus)

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
    - Loads the MATPOWER case (name/path) and runs the MATLAB routine LagrangianM_singlephase via matlab.engine.
    - Expects a full measurement vector ordered as [Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)].

    Inputs
    - case_path: Path or case name resolvable by MATLAB (e.g., 'case14').
    - z: Full measurement vector length 3*nb + 4*nl.

    Returns
    - success: boolean
    - r: normalized residual vector (same ordering as inputs)
    - lambdaN: normalized multipliers (typically length 2*nl)
    Note: large diagnostic arrays (EA, lambda_vec) are omitted to reduce payload size.
    """
    eng = _get_engine()
    return _wls_json(eng, case_path, z)

@mcp.tool(name="wls_from_text")
def wls_from_text(*, case_name: str, case_text: str, z: List[float]) -> Dict[str, Any]:
    """
    Same as wls_from_path, but accepts inline case.m contents.
    'case_name' must match the function name inside the .m file.
    """
    eng = _get_engine()
    path = _write_case_text(case_text, case_name)
    return _wls_json(eng, path, z)

if __name__ == "__main__":
    # Bind to a stable HTTP port so clients (build_sft_traces.py) can call reliably
    mcp.run(transport="http", host="127.0.0.1", port=3929)


