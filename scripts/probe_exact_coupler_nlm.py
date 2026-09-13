"""Standalone feasibility experiment; does not modify the production estimator.

Run from the repository root:
  python scripts/probe_exact_coupler_nlm.py --output output/coupler_nlm_probe.json

The same expanded physical section model serves every status hypothesis.
There are 70 voltage nodes, 13 explicit coupler P/Q pairs, and 26 exact
operational constraints. Nullspace elimination enforces them without epsilon.
Covariance is the local first-order propagation of analog Gaussian noise, not
a claim of exact finite-sample calibration under an incorrect nonlinear model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.linalg import null_space, solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2
from threadpoolctl import threadpool_limits
from pypower.api import ppoption, runpf
from psse_env.systems import resolve_system
from logical_topology.inventory import build_inventory, process_topology
from logical_topology.measurements import (
    build_measurement_inventory, expected_measurements, sample_measurements,
)
from logical_topology.estimation import _MeasurementModel, estimate


def multipliers(C, A, e, U):
    """KKT multipliers and covariance for objective (1/2)||h_w-z_w||^2.

    C is independent, linear, and exactly imposed. U spans range(A null(C)).
    KKT gives lambda = -(C C.T)^-1 C A.T e at a stationary point.
    For unit-covariance whitened analog noise eta, residual response is
    -(I-U U.T) eta, hence lambda response D=(C C.T)^-1 C A.T(I-U U.T).
    Sigma_lambda = D D.T. No unconstrained inverse of A.T A is required.
    """
    B = np.linalg.solve(C @ C.T, C @ A.T)
    lam = -B @ e
    D = B - (B @ U) @ U.T
    covariance = D @ D.T
    return lam, covariance, D


def constrained_fit(case, inventory, statuses, observations, sensors):
    # All-open compilation retains every physical section voltage. Supplying
    # all-closed statuses to the measurement model retains every flow state.
    # This is only the unconstrained equation template; actual statuses are C.
    expanded = dict(statuses)
    expanded.update({c["device_id"]: 0 for c in inventory["couplers"]})
    all_flows = dict(statuses)
    all_flows.update({c["device_id"]: 1 for c in inventory["couplers"]})
    processed = process_topology(case, inventory, expanded)
    model = _MeasurementModel(case, inventory, all_flows, processed, sensors)
    ncb = len(inventory["couplers"])
    C = np.zeros((2*ncb, model.nstate))
    labels = []
    angle_column = {int(row): index for index, row in enumerate(model.angle_rows)}
    for i, cb in enumerate(inventory["couplers"]):
        if statuses[cb["device_id"]] == 1:
            for node_key, sign in (("node_a", 1), ("node_b", -1)):
                row = int(processed["node_to_row0"][cb[node_key]])
                if row != model.ref:
                    C[2*i, angle_column[row]] = sign
                C[2*i+1, model.vm_start+row] = sign
            labels.extend(["angle_equality_rad", "magnitude_equality_pu"])
        else:
            C[2*i, model.voltage_states+i] = 1
            C[2*i+1, model.voltage_states+ncb+i] = 1
            labels.extend(["active_flow_zero_pu", "reactive_flow_zero_pu"])
    mask = np.asarray(sensors["available_mask"], dtype=bool)
    R = np.asarray(sensors["covariance"])[np.ix_(mask, mask)]
    L = np.linalg.cholesky(R)
    z = np.asarray([v for v, available in zip(observations["values"], mask) if available])
    def whiten(value):
        return solve_triangular(L, value, lower=True, check_finite=False)
    N = null_space(C)
    initial = np.zeros(model.nstate)
    initial[model.vm_start:model.voltage_states] = 1
    result = least_squares(
        lambda u: whiten(model.evaluate(N @ u)[0][mask]-z), N.T @ initial,
        jac=lambda u: whiten(model.evaluate(N @ u)[1][mask]) @ N,
        method="trf", x_scale="jac", ftol=1e-11, xtol=1e-11, gtol=1e-8,
        max_nfev=100,
    )
    x = N @ result.x
    h, H = model.evaluate(x)
    e = whiten(h[mask]-z)
    A = whiten(H[mask])
    U, singular, _ = np.linalg.svd(A @ N, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-9))
    U = U[:, :rank]
    lam, covariance, D = multipliers(C, A, e, U)
    sd = np.sqrt(np.maximum(np.diag(covariance), 0))
    if rank != N.shape[1] or np.any(sd <= 1e-10):
        raise ValueError("Unobservable effective state or non-testable multiplier")
    nlm = lam / sd
    variance = np.diag(R)-np.sum((L @ U)**2, axis=1)
    normalized_residual = np.abs(z-h[mask])/np.sqrt(np.maximum(variance, np.maximum(np.diag(R)*1e-12, 1e-16)))
    J = float(e @ e)
    threshold = float(chi2.ppf(.95, int(mask.sum())-rank))
    ranking = []
    for i, cb in enumerate(inventory["couplers"]):
        pair = slice(2*i, 2*i+2)
        ranking.append({
            "device_id": cb["device_id"], "reported_status": statuses[cb["device_id"]],
            "constraint_kinds": labels[pair], "lambda": lam[pair].tolist(),
            "lambda_sd": sd[pair].tolist(), "normalized_multipliers": nlm[pair].tolist(),
            "score_max_absolute_nlm": float(np.max(np.abs(nlm[pair]))),
            "joint_two_constraint_statistic": float(lam[pair] @ np.linalg.solve(covariance[pair, pair], lam[pair])),
        })
    ranking.sort(key=lambda row: (-row["score_max_absolute_nlm"], row["device_id"]))

    # Independent direct KKT sensitivity check, and constraint-order invariance.
    G = A.T @ A
    K = np.block([[G, C.T], [C, np.zeros((len(C), len(C)))]])
    response = np.linalg.solve(K, np.vstack((A.T, np.zeros((len(C), len(e))))))
    kkt_D = response[model.nstate:]
    p = np.arange(len(C))[::-1]
    lam_p, covariance_p, _ = multipliers(C[p], A, e, U)
    inverse = np.argsort(p)
    ordered_nlm = lam_p[inverse]/np.sqrt(np.diag(covariance_p)[inverse])
    baseline = estimate(case, inventory, statuses, observations, sensors)
    if not baseline["converged"]:
        raise ValueError("Contracted comparison did not converge")
    report = {
        "expanded_state_dimension": model.nstate,
        "exact_constraint_rank": int(np.linalg.matrix_rank(C)),
        "effective_state_dimension": N.shape[1], "weighted_tangent_jacobian_rank": rank,
        "available_measurement_count": int(mask.sum()), "chi_square_dof": int(mask.sum())-rank,
        "converged": bool(result.success), "function_evaluations": result.nfev,
        "solver_message": result.message,
        "constraint_max_absolute": float(np.max(np.abs(C @ x))),
        "tangent_gradient_max_absolute": float(np.max(np.abs((A @ N).T @ e))),
        "kkt_stationarity_max_absolute": float(np.max(np.abs(A.T @ e+C.T @ lam))),
        "kkt_stationarity_relative_to_gradient_scale": float(np.max(np.abs(A.T @ e+C.T @ lam))/max(1., np.linalg.norm(A, ord=2)*np.linalg.norm(e))),
        "minimum_voltage_magnitude_pu": float(min(x[model.vm_start:model.voltage_states])),
        "maximum_voltage_magnitude_pu": float(max(x[model.vm_start:model.voltage_states])),
        "J": J, "chi_square_threshold": threshold,
        "max_normalized_residual": float(max(normalized_residual)),
        "plausible": bool(result.success and rank == N.shape[1] and J < threshold and max(normalized_residual) < 4),
        "contracted_estimator_J": baseline["wls_objective"],
        "contracted_estimator_max_normalized_residual": baseline["max_normalized_residual"],
        "contracted_estimator_plausible": baseline["plausible"],
        "J_absolute_difference_from_contracted": abs(J-baseline["wls_objective"]),
        "prediction_max_absolute_difference_from_contracted_pu": float(np.max(np.abs(h-baseline["predicted_values"]))),
        "nlm_max_absolute_difference_after_constraint_reversal": float(np.max(np.abs(nlm-ordered_nlm))),
        "covariance_max_relative_difference_from_direct_kkt": float(np.max(np.abs(covariance-kkt_D @ kkt_D.T))/np.max(np.abs(covariance))),
        "multiplier_noise_response_max_relative_difference_from_direct_kkt": float(np.max(np.abs(D-kkt_D))/np.max(np.abs(D))),
        "minimum_multiplier_covariance_eigenvalue": float(np.linalg.eigvalsh(covariance)[0]),
        "coupler_ranking": ranking,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/exact_coupler_nlm_probe.json")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        parser.error(f"Output already exists: {target}; choose a fresh path")
    target.parent.mkdir(parents=True, exist_ok=True)
    case = resolve_system("case57").load_case()
    inventory = build_inventory("case57")
    sensors = build_measurement_inventory(inventory, "direct")
    normal = dict(inventory["normal_statuses"])
    opts = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10)
    rows = []
    worlds = []
    for true_open_bus in (None, 4, 12):
        truth_statuses = dict(normal)
        if true_open_bus is not None:
            truth_statuses[f"case57:bus:{true_open_bus}:coupler"] = 0
        processed = process_topology(case, inventory, truth_statuses)
        physical, success = runpf(processed["case"], opts)
        world = {"true_open_bus": true_open_bus, "physical_pf_converged": bool(success),
                 "physical_connected": processed["connectivity"]["connected"]}
        worlds.append(world)
        if not success:
            continue
        noiseless = expected_measurements(case, inventory, truth_statuses, physical, sensors)
        seed = 90200+(true_open_bus or 0)
        observations = sample_measurements(noiseless, sensors, seed=seed, noise=True)
        # Physical truth selects fixed experimental fixtures only. Ranking and
        # candidate choice below receive only the reported model and raw z/R.
        error_buses = (None, 4, 12) if true_open_bus is None else (None, true_open_bus)
        for error_bus in error_buses:
            reported = dict(truth_statuses)
            error_device = f"case57:bus:{error_bus}:coupler" if error_bus is not None else None
            if error_device:
                reported[error_device] = 1-reported[error_device]
            result = constrained_fit(case, inventory, reported, observations, sensors)
            row = {"true_open_bus": true_open_bus, "error_device_for_evaluation_only": error_device,
                   "scenario": "healthy" if error_bus is None else "split" if true_open_bus is None else "merging",
                   "noise_seed": seed, "result": result}
            if error_device:
                row["true_error_rank_for_evaluation_only"] = next(i+1 for i, r in enumerate(result["coupler_ranking"]) if r["device_id"] == error_device)
            # Apply no mutation: evaluate the top-ranked alternative only when
            # the current model has a residual-consistency alarm.
            if not result["plausible"]:
                device = result["coupler_ranking"][0]["device_id"]
                alternative = dict(reported)
                alternative[device] = 1-alternative[device]
                fitted = estimate(case, inventory, alternative, observations, sensors)
                row["top_ranked_alternative_test"] = {
                    "device_id": device, "desired_status": alternative[device],
                    "J": fitted["wls_objective"], "max_normalized_residual": fitted["max_normalized_residual"],
                    "plausible": fitted["plausible"], "converged": fitted["converged"],
                    "observable": fitted["observable"],
                    "matches_truth_for_evaluation_only": alternative == truth_statuses,
                }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("scenario", "true_open_bus", "error_device_for_evaluation_only")}, ensure_ascii=True),
                  "J", round(result["J"], 5), "rank", row.get("true_error_rank_for_evaluation_only"), flush=True)
    errors = [r for r in rows if r["scenario"] != "healthy"]
    healthy = [r for r in rows if r["scenario"] == "healthy"]
    checks = {
        "seven_prespecified_reported_models_completed": len(rows) == 7,
        "all_converged_and_effective_observable": all(r["result"]["converged"] and r["result"]["weighted_tangent_jacobian_rank"] == 139 for r in rows),
        "constraints_within_1e_minus_12": all(r["result"]["constraint_max_absolute"] < 1e-12 for r in rows),
        "objectives_match_contracted_within_1e_minus_6": all(r["result"]["J_absolute_difference_from_contracted"] < 1e-6 for r in rows),
        "predictions_match_contracted_within_1e_minus_6_pu": all(r["result"]["prediction_max_absolute_difference_from_contracted_pu"] < 1e-6 for r in rows),
        "covariance_matches_independent_kkt_within_1e_minus_10_relative": all(r["result"]["covariance_max_relative_difference_from_direct_kkt"] < 1e-10 for r in rows),
        "constraint_reordering_preserves_nlm_within_1e_minus_10": all(r["result"]["nlm_max_absolute_difference_after_constraint_reversal"] < 1e-10 for r in rows),
        "all_three_healthy_controls_plausible": len(healthy) == 3 and all(r["result"]["plausible"] for r in healthy),
        "all_four_erroneous_models_rejected": len(errors) == 4 and all(not r["result"]["plausible"] for r in errors),
        "all_four_errors_rank_first": len(errors) == 4 and all(r["true_error_rank_for_evaluation_only"] == 1 for r in errors),
        "all_four_top_ranked_alternatives_fit_and_match_truth": len(errors) == 4 and all(r["top_ranked_alternative_test"]["plausible"] and r["top_ranked_alternative_test"]["matches_truth_for_evaluation_only"] for r in errors),
    }
    sources = ["logical_topology/estimation.py", "logical_topology/inventory.py", "logical_topology/measurements.py"]
    output = {
        "contract": "standalone_exact_coupler_nlm_feasibility_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed single-noise-draw direct-sensor PF fixtures for buses 4 and 12; feasibility, not performance or false-positive calibration. No production edits or integration.",
        "physical_world_limitations": "Canonical dispatch with runpf; no OPF redispatch or operating-limit admission. Fresh physical worlds, not existing audit corpus.",
        "covariance_scope": "Exact algebra for local linearized analog-noise propagation under hard independent linear constraints; not exact nonlinear finite-sample significance.",
        "hypothesis_solver": "165 expanded states minus 26 exact linear constraints = 139 effective states. Unit voltage/zero angle/zero flow initialization, no truth state initialization.",
        "bounds_difference": "The nullspace probe is unconstrained in voltage magnitude; existing estimator uses 0.2 to 2.0 bounds. Observed extrema are reported to check inactivity.",
        "ranking_scope": "Max absolute normalized operational-constraint multiplier among all 13 couplers. Branch statuses held fixed; truth used only for fixture generation and evaluation labels.",
        "acceptance_scope": "Top-ranked alternative only receives ordinary WLS plausibility verification; no full rival scan, calibration certificate, or automatic model correction is claimed.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verification": checks, "all_checks_pass": all(checks.values()),
        "worlds": worlds, "rows": rows,
    }
    target.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Saved", target, flush=True)
    if not all(checks.values()):
        raise SystemExit("Feasibility verification failed; inspect saved JSON")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
