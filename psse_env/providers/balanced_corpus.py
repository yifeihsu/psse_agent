"""Fresh balanced telemetry for the existing Round0ScenarioGenerator.

This physical-source stage admits solved OPF operating windows independently
of WLS detectability or expert success. Its clean Gaussian draws are never
conditioned on a chi-square test. Downstream training admission is separate.
Truth cases and labels are offline artifacts; no solved initial-state vectors
are supplied to the policy or parameter estimator.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from copy import deepcopy

from pypower.idx_brch import BR_R, BR_X, PF, PT, QF, QT
from pypower.idx_bus import BUS_I, PD, QD, VM, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, PMAX, PMIN, QG, QMAX, QMIN

from Transmission.generate_measurements import (
    DEFAULT_SIGMAS,
    MEASUREMENT_ORDER,
    apply_measurement_error,
    apply_parameter_error_oneline,
    base_gaussian_noise,
    compute_measurements_pu,
    make_index_map,
    scale_loads,
    sigma_vector,
    solve_ac_opf,
    write_ppc_as_matpower_m,
)
from psse_env.systems import resolve_system


BALANCED_CORPUS_FAMILIES = ("no_error", "measurement_error", "parameter_error")
_PARAMETER_FACTOR_RANGES = [(0.1, 0.5), (2.0, 5.0)]
_POWER_BALANCE_TOLERANCE_PU = 1e-4
_BOUND_TOLERANCE_PU = 1e-5


class PhysicalAdmissionError(ValueError):
    """A solved source failed one independently checked physical contract."""

    def __init__(self, reason: str, metrics: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.metrics = dict(metrics or {})


def validate_balanced_solution(solved: Mapping[str, Any], measurements: np.ndarray) -> dict[str, Any]:
    """Compare telemetry with OPF injections/flow columns and enforce bounds.

The injection check uses generator-minus-load arithmetic, independent of the
admittance calculation used to generate telemetry. This is an arithmetic
cross-check of one solver's output, not a second simulation engine.
"""
    if not bool(solved.get("success")):
        raise PhysicalAdmissionError("opf_not_successful")
    bus = np.asarray(solved["bus"], dtype=float)
    gen = np.asarray(solved["gen"], dtype=float)
    branch = np.asarray(solved["branch"], dtype=float)
    base_mva = float(solved["baseMVA"])
    z = np.asarray(measurements, dtype=float)
    nb, nl = len(bus), len(branch)
    if z.shape != (3 * nb + 4 * nl,):
        raise PhysicalAdmissionError("measurement_dimension_mismatch")
    if not all(np.isfinite(a).all() for a in (bus, gen, branch, z)) or not np.isfinite(base_mva) or base_mva <= 0:
        raise PhysicalAdmissionError("nonfinite_or_invalid_solution")
    if branch.shape[1] <= QT:
        raise PhysicalAdmissionError("solver_branch_flows_missing")
    bus_rows = {int(bus_id): row for row, bus_id in enumerate(bus[:, BUS_I])}
    if len(bus_rows) != nb:
        raise PhysicalAdmissionError("duplicate_bus_id")
    injection = -(bus[:, PD] + 1j * bus[:, QD]) / base_mva
    active_gen = gen[gen[:, GEN_STATUS] > 0]
    for generator in active_gen:
        row = bus_rows.get(int(generator[GEN_BUS]))
        if row is None:
            raise PhysicalAdmissionError("generator_bus_missing")
        injection[row] += (generator[PG] + 1j * generator[QG]) / base_mva
    idx = make_index_map(nb, nl)
    injection_error = float(np.max(np.abs(z[idx["Pinj"]] + 1j * z[idx["Qinj"]] - injection)))
    flow_error = max(
        float(np.max(np.abs(z[idx[key]] - branch[:, column] / base_mva)))
        for key, column in (("Pf", PF), ("Qf", QF), ("Pt", PT), ("Qt", QT))
    )
    vm_error = float(np.max(np.abs(z[idx["Vm"]] - bus[:, VM])))
    voltage_violation = float(max(0.0, np.max(bus[:, VMIN] - bus[:, VM]), np.max(bus[:, VM] - bus[:, VMAX])))
    generator_violation = 0.0
    if len(active_gen):
        generator_violation = float(max(
            0.0,
            np.max(active_gen[:, PMIN] - active_gen[:, PG]),
            np.max(active_gen[:, PG] - active_gen[:, PMAX]),
            np.max(active_gen[:, QMIN] - active_gen[:, QG]),
            np.max(active_gen[:, QG] - active_gen[:, QMAX]),
        ) / base_mva)
    metrics = {
        "vm_min_pu": float(np.min(bus[:, VM])),
        "vm_max_pu": float(np.max(bus[:, VM])),
        "voltage_bound_violation_pu": voltage_violation,
        "generator_bound_violation_pu": generator_violation,
        "generator_load_injection_max_error_pu": injection_error,
        "stored_branch_flow_max_error_pu": flow_error,
        "voltage_measurement_max_error_pu": vm_error,
        "power_balance_tolerance_pu": _POWER_BALANCE_TOLERANCE_PU,
        "bound_tolerance_pu": _BOUND_TOLERANCE_PU,
    }
    if voltage_violation > _BOUND_TOLERANCE_PU or generator_violation > _BOUND_TOLERANCE_PU:
        raise PhysicalAdmissionError("operating_bounds_violated", metrics)
    if max(injection_error, flow_error, vm_error) > _POWER_BALANCE_TOLERANCE_PU:
        raise PhysicalAdmissionError("telemetry_physics_mismatch", metrics)
    return metrics


def _source_id(spec: Any, seed: int, family: str, attempt: int, solved: Mapping[str, Any]) -> str:
    payload = {
        "system": spec.case_id, "base_case_hash": spec.base_case_hash,
        "seed": seed, "family": family, "attempt": attempt,
        "baseMVA": float(solved["baseMVA"]),
        "bus": np.asarray(solved["bus"]).tolist(),
        "gen": np.asarray(solved["gen"]).tolist(),
        "branch": np.asarray(solved["branch"]).tolist(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return f"balanced_{digest[:24]}"


# ------------------------------------------------- effect-targeted magnitudes

def _effect_targeted_meters(z_true, z_obs, label, solver, rng, band, scale_bounds):
    """Scale a row's gross meter amplitudes until the WLS excess lands in the band (exact)."""
    from research.classifier_triage.effect_sampler import draw_target, fit_scale, implied_magnitude

    z_true = np.asarray(z_true, dtype=float)
    z_obs = np.asarray(z_obs, dtype=float)
    gross = np.zeros_like(z_true)
    if "indices" in label:
        for index, amplitude in zip(label["indices"], label["amplitudes"]):
            gross[int(index)] = float(amplitude)
    else:
        gross[int(label["index"])] = float(label["amplitude"])
    noise = z_obs - z_true - gross
    target = draw_target(rng, band)
    fit = fit_scale(solver, z_true, gross, noise, target, scale_bounds=scale_bounds)
    if not fit.get("reachable"):
        return z_obs, {**dict(label), "effect_failure": fit.get("reason")}, None
    k = float(fit["scale"])
    new_label = dict(label)
    if "indices" in label:
        new_label["amplitudes"] = [float(a) * k for a in label["amplitudes"]]
    else:
        new_label["amplitude"] = float(label["amplitude"]) * k
    new_label["drawn_amplitude_scale"] = k
    effect = {"target": target, **{key: value for key, value in fit.items() if key != "reachable"},
              "implied": implied_magnitude("measurement", label, k), "sampler": "effect_targeted_v1"}
    return z_true + noise + k * gross, new_label, effect


def _effect_targeted_parameter(nominal_case, label, noise, solver, rng, band, scale_bounds):
    """Re-solve the true network with the factor's logarithm scaled until the WLS excess lands in the band.

    Returns ``(true_case, solved, z_true, z_obs, label, admission, effect)`` or None.  The scale
    bounds are tightened so the implied factors stay within [0.02, 50]; every candidate state is
    the true network's own OPF, so the corpus keeps consistent physical cases.
    """
    import math

    from research.classifier_triage.effect_sampler import draw_target, effect_ratio, implied_magnitude

    noise = np.asarray(noise, dtype=float)
    row = int(label["line_row"])
    r_factor, x_factor = float(label["r_factor"]), float(label["x_factor"])
    low, high = (float(v) for v in scale_bounds)
    for factor in (r_factor, x_factor):
        if factor != 1.0:
            limit = math.log(50.0 if factor > 1.0 else 0.02) / math.log(factor)
            high = min(high, limit)
    if not (low < high):
        return None
    target = draw_target(rng, band)
    cache: dict[float, tuple[Any, Any, np.ndarray, dict[str, Any]] | None] = {}

    def solve_case(k: float):
        if k in cache:
            return cache[k]
        case = deepcopy(nominal_case)
        br = case["branch"]
        br[row, BR_R] = max(1e-6, br[row, BR_R] * (r_factor ** k))
        br[row, BR_X] = max(1e-6, br[row, BR_X] * (x_factor ** k))
        try:
            solved_k = solve_ac_opf(case)
            if solved_k is None:
                raise RuntimeError("opf")
            z_k = compute_measurements_pu(solved_k)
            admission_k = validate_balanced_solution(solved_k, z_k)
        except Exception:
            cache[k] = None
            return None
        cache[k] = (case, solved_k, np.asarray(z_k, dtype=float), admission_k)
        return cache[k]

    def ratio_at(k: float):
        state = solve_case(k)
        if state is None:
            return None
        solved_k = solver(state[2] + noise)
        return None if solved_k is None else effect_ratio(solved_k[0], solved_k[1])

    r_low, r_high = ratio_at(low), ratio_at(high)
    if r_low is None or r_high is None or r_high < target * 0.9 or r_low > target * 1.1:
        return None
    lo, hi = math.log(low), math.log(high)
    best = None
    for iteration in range(10):
        mid = math.exp(0.5 * (lo + hi))
        ratio = ratio_at(mid)
        if ratio is None:
            hi = math.log(mid)
            continue
        if best is None or abs(ratio - target) < abs(best[1] - target):
            best = (mid, ratio, iteration + 3)
        if abs(ratio - target) <= 0.1 * target:
            break
        if ratio < target:
            lo = math.log(mid)
        else:
            hi = math.log(mid)
    if best is None:
        return None
    k, ratio, iterations = best
    case, solved_k, z_k, admission_k = solve_case(k)
    new_label = {**dict(label), "r_factor": r_factor ** k, "x_factor": x_factor ** k, "drawn_r_factor": r_factor,
                 "drawn_x_factor": x_factor, "drawn_factor_log_scale": k}
    effect = {"target": target, "scale": k, "ratio": ratio, "iterations": iterations, "ratio_at_bounds": [r_low, r_high],
              "implied": implied_magnitude("parameter", label, k), "sampler": "effect_targeted_v1",
              "resolved": "true-network OPF at the implied factors"}
    return case, solved_k, z_k, z_k + noise, new_label, admission_k, effect


def build_balanced_corpus(
    output_dir: str | Path,
    *,
    system: str = "case57",
    seed: int = 20260910,
    counts: Mapping[str, int] | None = None,
    load_scale_range: tuple[float, float] = (0.80, 1.0),
    num_scans: int = 3,
    max_attempt_multiplier: int = 10,
    effect_band: tuple[float, float] | None = None,
    effect_scale_bounds: tuple[float, float] = (0.02, 50.0),
) -> dict[str, Any]:
    """Generate fresh clean, single/multiple meter, and one-line R/X sources.

    ``counts`` counts independent raw OPF windows, not scans or eventual
    rollouts. Meter subtypes alternate over accepted windows. Parameter scans
    are independent noise draws around the same solved true network. Exhausted
    physical admission returns a durable manifest with ``complete=False``;
    OPF nonconvergence is never labeled mathematical infeasibility.

    ``effect_band`` (2026-10-12) draws each error's magnitude as the effect the
    operator's WLS sees (research.classifier_triage.effect_sampler): a meter
    error's gross amplitudes are scaled exactly, a line error's R/X factors
    are re-solved (the factor's logarithm is scaled and the true network's
    OPF re-run) until the chi-square excess relative to the alarm threshold's
    excess lands in the band; rows the band cannot reach within
    ``effect_scale_bounds`` are rejected and counted.
    """
    requested_input = dict(counts if counts is not None else {family: 8 for family in BALANCED_CORPUS_FAMILIES})
    unknown = set(requested_input) - set(BALANCED_CORPUS_FAMILIES)
    if unknown:
        raise ValueError(f"Unsupported balanced corpus families: {sorted(unknown)}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in requested_input.values()):
        raise ValueError("counts must contain nonnegative integers")
    for name, value in (("num_scans", num_scans), ("max_attempt_multiplier", max_attempt_multiplier)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    lo, hi = (float(value) for value in load_scale_range)
    if not np.isfinite([lo, hi]).all() or not 0 < lo <= hi:
        raise ValueError("load_scale_range must be finite, positive, and ordered")
    spec = resolve_system(system)
    requested = {family: requested_input.get(family, 0) for family in BALANCED_CORPUS_FAMILIES}
    ppc_base = spec.load_case()
    idx_map = make_index_map(spec.nb, spec.nl)
    sigma = sigma_vector(idx_map, DEFAULT_SIGMAS)
    if not np.array_equal(sigma, np.asarray(spec.measurement_sigma(), dtype=float)):
        raise ValueError("System sigma must match the deployed fixed WLS covariance")
    out = Path(output_dir).resolve()
    if (out / "samples.jsonl").exists() or (out / "meta.json").exists():
        raise FileExistsError(f"Corpus output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    base_case_path = write_ppc_as_matpower_m(ppc_base, out / "base_case.m", "base_case")
    corpus_path = out / "samples.jsonl"
    meta_path = out / "meta.json"
    rejection_path = out / "rejections.jsonl"
    attempted = {family: 0 for family in BALANCED_CORPUS_FAMILIES}
    accepted = {family: 0 for family in BALANCED_CORPUS_FAMILIES}
    rejected = {family: 0 for family in BALANCED_CORPUS_FAMILIES}
    rejected_records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    effect_rng = np.random.default_rng(seed + 7919)
    effect_solver = None
    if effect_band is not None:
        from research.classifier_triage.effect_sampler import parse_band, wls_solver

        effect_band = parse_band(effect_band)
        effect_solver = wls_solver(spec.case_id, sigma)
    with corpus_path.open("w", encoding="utf-8") as stream:
        for family in BALANCED_CORPUS_FAMILIES:
            budget = max_attempt_multiplier * requested[family]
            while accepted[family] < requested[family] and attempted[family] < budget:
                attempted[family] += 1
                alpha = float(rng.uniform(lo, hi))
                true_case = scale_loads(ppc_base, alpha)
                label: dict[str, Any] = {"error_type": family}
                if family == "parameter_error":
                    true_case, label = apply_parameter_error_oneline(
                        true_case, rng, _PARAMETER_FACTOR_RANGES, _PARAMETER_FACTOR_RANGES,
                    )
                rejection = {"scenario": family, "attempt": attempted[family], "load_scale": alpha, "label": label}
                try:
                    solved = solve_ac_opf(true_case)
                except Exception as exc:
                    solved = None
                    rejection.update(reason="opf_solver_exception", detail=f"{type(exc).__name__}: {exc}")
                if solved is None:
                    rejection.setdefault("reason", "opf_nonconvergence")
                    rejected[family] += 1
                    rejected_records.append(rejection)
                    continue
                try:
                    z_true = compute_measurements_pu(solved)
                    admission = validate_balanced_solution(solved, z_true)
                except PhysicalAdmissionError as exc:
                    rejection.update(reason=exc.reason, metrics=exc.metrics)
                    rejected[family] += 1
                    rejected_records.append(rejection)
                    continue
                source_id = _source_id(spec, seed, family, attempted[family], solved)
                effect: dict[str, Any] | None = None
                if family == "measurement_error":
                    subtype = ("single_gross_outlier", "multi_gross_outliers")[accepted[family] % 2]
                    z_obs, label = apply_measurement_error(z_true, idx_map, rng, subtype=subtype)
                    if effect_band is not None:
                        z_obs, label, effect = _effect_targeted_meters(
                            z_true, z_obs, label, effect_solver, effect_rng, effect_band, effect_scale_bounds,
                        )
                        if effect is None:
                            rejection.update(reason="effect_band_unreachable", detail=label.get("effect_failure"))
                            rejected[family] += 1
                            rejected_records.append(rejection)
                            continue
                else:
                    z_obs = z_true + base_gaussian_noise(z_true, idx_map, DEFAULT_SIGMAS, rng)
                    if family == "parameter_error" and effect_band is not None:
                        targeted = _effect_targeted_parameter(
                            scale_loads(ppc_base, alpha), label, z_obs - z_true, effect_solver, effect_rng,
                            effect_band, effect_scale_bounds,
                        )
                        if targeted is None:
                            rejection.update(reason="effect_band_unreachable")
                            rejected[family] += 1
                            rejected_records.append(rejection)
                            continue
                        true_case, solved, z_true, z_obs, label, admission, effect = targeted
                solved_path = write_ppc_as_matpower_m(
                    solved, out / "physical_cases" / f"{source_id}.m", source_id,
                )
                record: dict[str, Any] = {
                    "id": source_id, "source_realization_id": source_id,
                    "scenario": family, "network_case": spec.case_id,
                    "base_case_hash": spec.base_case_hash,
                    "configured_case_path": spec.case_path,
                    "canonical_case_artifact_path": str(base_case_path),
                    "physical_case_path": str(solved_path),
                    "z_true": z_true.tolist(), "z_obs": z_obs.tolist(),
                    "label": label, "op_point": {"load_scale": alpha},
                    "sigmas": dict(DEFAULT_SIGMAS),
                    "sigma_z": sigma.tolist(),
                    "physical_validation": {"passed": True, **admission},
                }
                if effect is not None:
                    record["effect"] = effect
                if family == "parameter_error":
                    record["sigma_z_scans"] = sigma.tolist()
                    record["z_scans"] = [
                        (z_true + rng.standard_normal(spec.nz) * sigma).tolist()
                        for _ in range(num_scans)
                    ]
                    record["scan_contract"] = "independent_noise_same_opf_operating_window_v1"
                    parameter_path = write_ppc_as_matpower_m(
                        true_case, out / "cases_parameter_error" / f"{source_id}.m", source_id,
                    )
                    record["parameter_error_case_path"] = str(parameter_path)
                    record["correction_case_path"] = str(parameter_path)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                accepted[family] += 1
    with rejection_path.open("w", encoding="utf-8") as stream:
        for rejection in rejected_records:
            stream.write(json.dumps(rejection, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "schema_version": "balanced_physical_corpus_v1", "system": spec.to_manifest(),
        "runtime": {package: version(package) for package in ("numpy", "scipy", "PYPOWER")},
        "network_case": spec.case_id, "base_case_hash": spec.base_case_hash,
        "seed": int(seed), "load_scale_range": [lo, hi], "num_scans": num_scans,
        "effect_band": list(effect_band) if effect_band is not None else None,
        "effect_scale_bounds": list(effect_scale_bounds) if effect_band is not None else None,
        "max_attempt_multiplier": max_attempt_multiplier,
        "measurement_order": list(MEASUREMENT_ORDER), "sigmas": dict(DEFAULT_SIGMAS),
        "nz": spec.nz, "state_count": spec.state_count,
        "requested": requested, "attempted": attempted, "accepted": accepted, "rejected": rejected,
        "counts": {"requested": requested, "attempted": attempted, "accepted": accepted, "rejected": rejected},
        "complete": accepted == requested,
        "corpus_path": str(corpus_path), "artifact_dir": str(out), "meta_path": str(meta_path),
        "canonical_case_artifact_path": str(base_case_path),
        "rejection_path": str(rejection_path), "rejections": rejected_records,
        "physical_source_contract": "changed_true_network_configured_base_model_v1",
        "clean_noise_chi_square_filtered": False,
        "raw_admission_uses_expert_or_wls": False,
        "physical_validation": {
            "all_accepted_passed": True,
            "power_balance_tolerance_pu": _POWER_BALANCE_TOLERANCE_PU,
            "bound_tolerance_pu": _BOUND_TOLERANCE_PU,
            "checks": ["opf_success", "voltage_bounds", "generator_bounds", "generator_minus_load_injections", "solver_branch_flows"],
        },
        "scan_contract": "independent_noise_same_opf_operating_window_v1",
        "initial_states": "omitted; downstream estimators must use observable initialization",
        "scope": "balanced no_error, measurement_error, parameter_error; no extended-physics synthesis",
        "solver_failure_interpretation": "nonconvergence is not proof of infeasibility",
        "solver_compatibility": "empty branch-constraint vector/Hessian shapes normalized without adding thermal limits",
        "thermal_limits_interpretation": "Uses canonical case bounds; equipment capability is not independently validated",
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
