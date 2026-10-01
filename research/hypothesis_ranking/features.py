"""Policy-visible WLS ledger features and the offline screen analysis of one state.

Everything here is computed from what the agent may see on a state: the
operator model, the balanced SCADA vector, its declared covariance and
structural zeros, the balanced WLS solve, and the balanced screen report.
Truth never enters; the dataset builder attaches it separately.
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

from mcp_server.matpower_server import _load_python_case, _wls_json
from psse_env.noise_contract import resolve_state_measurement_noise
from psse_env.providers.hif_screen import BR_STATUS, SHIFT, TAP, default_hif_lines, screen_hif
from tools.lagrangian_port import _copy_result_to_internal

CHANNEL_BLOCKS = ("Vm", "P", "Q", "Pf", "Qf", "Pt", "Qt")
#: The provider's residual-versus-multiplier dominance rule (matpower.py).
DOMINANCE_RATIO = 1.2
DEFAULT_CHI2_ALPHA = 0.01
DEFAULT_NORMALIZED_RESIDUAL_THRESHOLD = 4.0


def channel_type(index: int, nb: int, nl: int) -> str:
    """Measurement block of external channel ``index`` in ``[Vm, P, Q, Pf, Qf, Pt, Qt]`` order."""
    bounds = (nb, 2 * nb, 3 * nb, 3 * nb + nl, 3 * nb + 2 * nl, 3 * nb + 3 * nl, 3 * nb + 4 * nl)
    for block, upper in zip(CHANNEL_BLOCKS, bounds):
        if index < upper:
            return block
    raise IndexError(f"channel {index} outside a {nb}-bus {nl}-branch vector")


def channel_asset(index: int, nb: int, nl: int) -> int:
    """Bus row (Vm, P, Q) or branch row (flows) that channel ``index`` measures."""
    block = channel_type(index, nb, nl)
    if block in ("Vm", "P", "Q"):
        return index % nb
    offset = {"Pf": 3 * nb, "Qf": 3 * nb + nl, "Pt": 3 * nb + 2 * nl, "Qt": 3 * nb + 3 * nl}[block]
    return index - offset


def flow_channel_index(block: str, branch_row0: int, nb: int, nl: int) -> int:
    offset = {"Pf": 3 * nb, "Qf": 3 * nb + nl, "Pt": 3 * nb + 2 * nl, "Qt": 3 * nb + 3 * nl}[block]
    return offset + int(branch_row0)


def candidate_lines(case: Mapping[str, Any]) -> tuple[list[int], str]:
    """Candidate HIF lines the screen would use, with the rule that produced them.

    ``default_hif_lines`` knows IEEE 14 by its numbering and otherwise needs
    base kV on both ends; canonical cases with zero base kV (case57) give no
    lines, which is what production would do today.  The study falls back to
    every in-service untapped branch so the HIF class can be evaluated there;
    the rule name records which one applied.
    """
    lines = default_hif_lines(case)
    if lines:
        return lines, "default_hif_lines"
    branch = np.asarray(case["branch"], dtype=float)
    fallback = [
        k for k in range(branch.shape[0])
        if branch[k, BR_STATUS] > 0 and branch[k, TAP] in (0.0, 1.0) and branch[k, SHIFT] == 0.0
    ]
    return fallback, "all_in_service_untapped_branches"


def wls_features(
    payload: Mapping[str, Any], nb: int, nl: int, *,
    chi2_alpha: float = DEFAULT_CHI2_ALPHA,
    normalized_residual_threshold: float = DEFAULT_NORMALIZED_RESIDUAL_THRESHOLD,
    top_k: int = 6,
) -> dict[str, Any]:
    """The WLS ledger as the policy sees it, plus the ranked residual and multiplier evidence."""
    residuals = np.asarray(payload.get("r") or [], dtype=float)
    lambdas = np.asarray(payload.get("lambdaN") or [], dtype=float)
    statistic = float(payload["global_residual_sum"])
    dof = int(payload["dof"])
    threshold = float(chi2.ppf(1.0 - chi2_alpha, max(dof, 1)))
    abs_r = np.abs(residuals)
    max_r = float(abs_r.max()) if abs_r.size else 0.0
    # lambda_layout "per_branch_R_X_interleaved": rows 2k and 2k+1 are branch k's R and X multipliers.
    if lambdas.size == 2 * nl:
        lam_r, lam_x = np.abs(lambdas[0::2]), np.abs(lambdas[1::2])
        per_branch = np.maximum(lam_r, lam_x)
        which = np.where(lam_r >= lam_x, "R", "X")
    elif lambdas.size == nl:
        per_branch = np.abs(lambdas)
        which = np.array(["?"] * nl)
    else:
        per_branch = np.zeros(nl)
        which = np.array(["?"] * nl)
    max_l = float(per_branch.max()) if per_branch.size else 0.0
    ranked_branches = [int(k) for k in np.argsort(-per_branch)]
    top_l = per_branch[ranked_branches[0]] if ranked_branches else 0.0
    runner_l = per_branch[ranked_branches[1]] if len(ranked_branches) > 1 else 0.0
    chi_alarm = statistic >= threshold
    residual_alarm = max_r >= normalized_residual_threshold
    return {
        "success": True,
        "chi_square_statistic": statistic,
        "chi_square_dof": dof,
        "chi_square_alpha": chi2_alpha,
        "chi_square_threshold": threshold,
        "chi_square_ratio": statistic / threshold if threshold > 0 else None,
        "chi_square_alarm": bool(chi_alarm),
        "max_normalized_residual": max_r,
        "normalized_residual_threshold": normalized_residual_threshold,
        "normalized_residual_alarm": bool(residual_alarm),
        "alarm": bool(chi_alarm or residual_alarm),
        "remaining_anomaly_score": max(statistic / threshold, max_r / normalized_residual_threshold),
        "count_abs_residual_gt3": int(np.sum(abs_r > 3.0)),
        "count_abs_residual_gt4": int(np.sum(abs_r > 4.0)),
        "top_residuals": [
            {"index": int(i), "channel": channel_type(int(i), nb, nl), "asset_row0": channel_asset(int(i), nb, nl),
             "value": float(residuals[i])}
            for i in np.argsort(-abs_r)[:top_k] if abs_r[i] > 0.0
        ],
        "max_abs_branch_multiplier": max_l,
        "top_branch_multipliers": [
            {"branch_row0": int(k), "parameter": str(which[k]), "value": float(per_branch[k])}
            for k in ranked_branches[:top_k] if per_branch[k] > 0.0
        ],
        "branch_multiplier_ranking": ranked_branches[:top_k],
        "branch_ranking_dominance_ratio": (float(top_l / runner_l) if runner_l > 0.0 else None),
        "measurement_dominant": bool(max_r > DOMINANCE_RATIO * max_l),
        "branch_dominant": bool(max_l > DOMINANCE_RATIO * max_r),
        "iterations": payload.get("iterations"),
    }


def json_safe(value: Any) -> Any:
    """Plain JSON types; non-finite floats become None."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def analyze_state(
    record: Mapping[str, Any], *,
    chi2_alpha: float = DEFAULT_CHI2_ALPHA,
    normalized_residual_threshold: float = DEFAULT_NORMALIZED_RESIDUAL_THRESHOLD,
    run_screen: bool = True,
    lines: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Balanced WLS on one state and, when it alarms, the balanced screen.

    ``record`` carries ``case`` (a case name or path), ``measurements`` and the
    execution ``metadata`` a strict root would carry (``sigma_z``,
    ``structural_zero_indices``, ``operator_noise``).  The result holds the
    WLS ledger features and the full screen report with its offline fields.
    """
    case_name = str(record["case"])
    z = [float(value) for value in record["measurements"]]
    noise = resolve_state_measurement_noise({"metadata": record.get("metadata") or {}}, len(z))
    sigma = noise["measurement_sigma"]
    exact = list(noise["exact_measurement_indices"])
    started = time.perf_counter()
    payload = _wls_json(case_name, z, measurement_sigma=sigma, exact_measurement_indices=exact or None)
    out: dict[str, Any] = {"wls_seconds": time.perf_counter() - started}
    if not payload.get("success"):
        out["wls"] = {"success": False, "alarm": None, "error": str(payload.get("error"))}
        return out
    case = _load_python_case(case_name)
    nb = int(np.asarray(case["bus"]).shape[0])
    nl = int(np.asarray(case["branch"]).shape[0])
    wls = wls_features(payload, nb, nl, chi2_alpha=chi2_alpha, normalized_residual_threshold=normalized_residual_threshold)
    out["wls"] = wls
    out["nb"], out["nl"] = nb, nl
    if not (run_screen and wls["alarm"]):
        return out
    internal = _copy_result_to_internal(case)
    if lines is None:
        lines, rule = candidate_lines(case)
    else:
        lines, rule = [int(k) for k in lines], "explicit"
    sigma_vector = sigma if sigma is not None else [0.001] * nb + [0.01] * (len(z) - nb)
    started = time.perf_counter()
    try:
        report = screen_hif(
            internal["baseMVA"], internal["bus"], internal["branch"], z, sigma_vector, exact,
            lines=lines, va0=payload.get("theta_est_rad"), vm0=payload.get("vm_est_pu"),
            dof=int(payload["dof"]) if payload.get("dof") is not None else None,
        )
    except Exception as exc:  # a screen failure is recorded, never a decision
        report = {"status": "screen_error", "suspected": False, "error": f"{type(exc).__name__}: {exc}",
                  "rounds": [], "final": None, "unexplained": None}
    out["screen_seconds"] = time.perf_counter() - started
    report["candidate_line_rule"] = rule
    out["screen"] = json_safe(report)
    return out
