# matpower_mcp_server_fixed.py
from __future__ import annotations

import os
import json
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastmcp import FastMCP

mcp = FastMCP("MATPOWER Power Flow (FastMCP v2)")

# NOTE ABOUT MATLAB ENGINE (Python 3.13):
# Importing matlab.engine can crash CPython 3.13 on some setups.
# To keep the server robust, we avoid importing matlab.engine at module import time
# and provide a CLI-based fallback for the WLS tool using `matlab -batch`.

# ---------- (Optional) MATLAB engine support (disabled by default) ----------
_ENG = None
_ENG_LOCK = threading.Lock()

def _get_engine(startup_options: str | None = None):  # pragma: no cover
    """Lazy-import and start MATLAB engine if available.
    Avoid calling this on Python versions where the engine crashes.
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
    """Use MATPOWER already on path if present; otherwise try $MATPOWER_PATH; else error."""
    global _MATPOWER_READY
    if _MATPOWER_READY:
        return
    if eng.which("runpf"):  # already visible to this MATLAB process
        _MATPOWER_READY = True
        return
    env = os.environ.get("MATPOWER_PATH")
    if env:
        eng.addpath(eng.genpath(env), nargout=0)
    # Also ensure server dir is on path (user functions)
    server_dir = os.path.abspath(os.path.dirname(__file__))
    eng.addpath(eng.genpath(server_dir), nargout=0)
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
        S.r          = full(r);
        S.lambda_vec = full(lambda_vec);
        S.ea         = full(ea);
        JsonOut = jsonencode(S);
    """, nargout=0)
    return json.loads(eng.workspace["JsonOut"])



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

# ---------- Tools: WLS ----------
def _wls_cli_json(case_path: str, z_list: List[float]) -> Dict[str, Any]:
    """Run WLS via MATLAB CLI using wls_cli_main.m

    Writes input JSON to a temp file, calls MATLAB with -batch to run wls_cli_main,
    and parses the output JSON.
    """
    # Prepare IO files
    tmpdir = Path(tempfile.mkdtemp(prefix="wls_cli_"))
    in_path = tmpdir / "in.json"
    out_path = tmpdir / "out.json"

    with in_path.open("w", encoding="utf-8") as f:
        json.dump({"case_path": case_path, "z": [float(x) for x in z_list]}, f)

    # Compose MATLAB batch command
    server_dir = Path(__file__).parent.resolve()
    # Use forward slashes for MATLAB and escape quotes
    sd = str(server_dir).replace("\\", "/")
    ip = str(in_path).replace("\\", "/")
    op = str(out_path).replace("\\", "/")
    batch_cmd = (
        f"try, addpath(genpath('{sd}')); wls_cli_main('{ip}','{op}'); catch e, disp(getReport(e,'extended')); exit(1); end"
    )

    # Launch MATLAB in batch mode
    env = os.environ.copy()
    try:
        proc = subprocess.run(
            ["matlab", "-batch", batch_cmd],
            cwd=str(server_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        return {"success": False, "error": "MATLAB executable not found in PATH"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "MATLAB batch run timed out"}

    if proc.returncode != 0:
        return {
            "success": False,
            "error": "MATLAB returned non-zero exit code",
            "stderr": proc.stderr[-2000:],
            "stdout": proc.stdout[-2000:],
        }

    # Read output JSON
    if not out_path.exists():
        return {
            "success": False,
            "error": "Output JSON not produced by MATLAB",
            "stdout": proc.stdout[-2000:],
        }
    try:
        with out_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"success": False, "error": f"Failed to parse output JSON: {e}"}


@mcp.tool(name="wls_from_path")
def wls_from_path(*, case_path: str, z: List[float]) -> Dict[str, Any]:
    """
    WLS state estimation + bad-data post-processing using LagrangianM_singlephase.
    Uses MATLAB CLI fallback for robustness across Python versions.

    Args:
      case_path : path or case name for mpc
      z         : measurement vector in the expected order (see notes)
    """
    # Prefer CLI approach to avoid matlab.engine import issues on Python 3.13+
    return _wls_cli_json(case_path, z)

@mcp.tool(name="wls_from_text")
def wls_from_text(*, case_name: str, case_text: str, z: List[float]) -> Dict[str, Any]:
    """
    Same as wls_from_path, but loads case from raw .m text ('case_name' must match function name).
    Uses MATLAB CLI fallback.
    """
    path = _write_case_text(case_text, case_name)
    return _wls_cli_json(path, z)

if __name__ == "__main__":
    # Bind to a stable HTTP port so clients (build_sft_traces.py) can call reliably
    mcp.run(transport="http", host="127.0.0.1", port=3929)
