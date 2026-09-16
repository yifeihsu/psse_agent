"""Fresh physical IEEE-14 screening corpus with independent operating parents.

All labels use the same normalized OpenDSS realization and registry exporter.
The physical model is a fundamental-frequency, fixed-PQ snapshot; HIF means a
steady-state phase-to-ground resistor, not an arcing/harmonic simulator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from psse_env.systems import resolve_system
from Transmission.generate_measurements import solve_ac_opf
from three_phase_model.exporter import export_model, load_assumptions
from three_phase_model.runtime import compile_model, solve
from three_phase_model.measurements import extract_measurements
from three_phase_model.disturbances import (audit_disturbed_circuit,
    eligible_hif_branch_rows, inject_midspan_hif)
from .dataset import content_hash, file_sha256, jsonable, write_json
from .feature_schema import MEASUREMENT_CONVENTION
from .wls_features import configured_case, default_measurement_sigma, state_measurements_and_jacobian

SPLITS = ("train", "validation", "calibration", "test")
SEVERITIES = ("weak", "intermediate", "strong")


def connected(case: dict[str, Any]) -> bool:
    buses = set(map(int, case["bus"][:, 0]))
    seen = {min(buses)}
    edges = case["branch"][case["branch"][:, 10] == 1, :2].astype(int)
    while True:
        expanded = seen | {int(t) for f, t in edges if int(f) in seen} | {int(f) for f, t in edges if int(t) in seen}
        if expanded == seen:
            return seen == buses
        seen = expanded


def status_candidates(case: dict[str, Any]) -> list[int]:
    """Only identity-preserving branch toggles with connected configured graph."""
    candidates = []
    for row in range(len(case["branch"])):
        candidate = copy.deepcopy(case)
        candidate["branch"][row, 10] = 1 - candidate["branch"][row, 10]
        if connected(candidate):
            candidates.append(row)
    return candidates


def sample_parent(rng: np.random.Generator, ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    case = resolve_system("case14").load_case()
    # Known physical network variation also supplies correct-config controls.
    # Model values alone must not identify all parameter/status error samples.
    network_factors = rng.uniform(.80, 1.20, size=(len(case["branch"]), 2))
    case["branch"][:, 2:4] *= network_factors
    global_load = float(rng.uniform(.80, 1.20))
    spatial = np.clip(np.exp(rng.normal(0, .14, size=len(case["bus"]))), .72, 1.35)
    case["bus"][:, 2:4] *= (global_load * spatial)[:, None]
    dispatch_factor = float(rng.uniform(.70, 1.30))
    case["gen"][1:, 1] *= dispatch_factor
    voltage_delta = rng.normal(0, .006, len(case["gen"]))
    case["gen"][:, 5] += voltage_delta
    for row in case["gen"]:
        case["bus"][case["bus"][:, 0] == row[0], 7] = row[5]
    cost_factors = rng.uniform(.75, 1.25, size=len(case["gen"]))
    case["gencost"][:, 4:] *= cost_factors[:, None]
    physical_open_row = None
    if ordinal % 4 == 0:
        candidates = [r for r in status_candidates(case) if case["branch"][r, 8] == 0]
        physical_open_row = int(rng.choice(candidates))
        case["branch"][physical_open_row, 10] = 0
    return case, {"global_load_factor": global_load, "bus_load_factors": spatial.tolist(),
        "non_slack_dispatch_factor": dispatch_factor, "generator_voltage_delta_pu": voltage_delta.tolist(),
        "generator_cost_factors": cost_factors.tolist(),
        "known_branch_rx_factors": network_factors.tolist(), "physical_open_branch_row0": physical_open_row,
        "setpoint_randomization_scope": "initialization_only_before_constrained_opf",
        "pv_control_equivalence": False}


def solve_operating_parent(case: dict) -> tuple[dict, dict]:
    """Accept only a converged source OPF within existing case operating bounds."""
    solved = solve_ac_opf(case)
    if solved is None:
        raise RuntimeError("Constrained source AC OPF did not converge")
    bus, gen = solved["bus"], solved["gen"]
    violations = {"voltage_pu": float(max(0., np.max(bus[:, 7] - bus[:, 11]), np.max(bus[:, 12] - bus[:, 7]))),
        "generator_p_mw": float(max(0., np.max(gen[:, 1] - gen[:, 8]), np.max(gen[:, 9] - gen[:, 1]))),
        "generator_q_mvar": float(max(0., np.max(gen[:, 2] - gen[:, 3]), np.max(gen[:, 4] - gen[:, 2])))}
    if violations["voltage_pu"] > 1e-5 or max(violations["generator_p_mw"], violations["generator_q_mvar"]) > .01:
        raise RuntimeError(f"Source OPF exceeds operating bounds: {violations}")
    physical = copy.deepcopy(case)
    for name, columns in (("bus", 13), ("branch", 13), ("gen", 21)):
        physical[name] = np.asarray(solved[name])[:, :columns].copy()
    return physical, {"success": True, "method": "Transmission.generate_measurements.solve_ac_opf",
        "reference_q_limits_enforced": True, "maximum_bound_violations": violations,
        "realized_generator_p_mw": gen[:, 1].tolist(), "realized_generator_q_mvar": gen[:, 2].tolist(),
        "realized_bus_vm_pu": bus[:, 7].tolist(), "objective": float(solved["f"]),
        "thermal_constraint_scope": "existing RATE_A only; zero ratings remain unconstrained"}


def redistribute_phases(dss: Any, registry: dict, *, bus: int, phase: int, delta: float) -> dict:
    """Cyclic load redistribution preserving total P and Q, all three orientations."""
    if phase not in (1, 2, 3) or not 0 < delta < 1:
        raise ValueError("phase must be 1/2/3 and delta in (0, 1)")
    rows = [r for r in registry["loads"] if r["bus"] == bus]
    if len(rows) != 3 or {r["phase"] for r in rows} != {1, 2, 3}:
        raise ValueError("unbalance needs a balanced three-phase load")
    factors = {phase: 1 + delta, phase % 3 + 1: 1 - delta, (phase + 1) % 3 + 1: 1.0}
    commands = []
    for row in rows:
        factor = factors[row["phase"]]
        command = f"Edit {row['element']} kW={row['kw'] * factor:.16g} kvar={row['kvar'] * factor:.16g}"
        dss.Text.Command(command)
        commands.append(command)
    solve(dss)
    before = np.asarray([[r["kw"], r["kvar"]] for r in rows]).sum(axis=0)
    after = np.asarray([[r["kw"], r["kvar"]] for r in rows]) * np.asarray([factors[r["phase"]] for r in rows])[:, None]
    if not np.allclose(after.sum(axis=0), before, atol=1e-9, rtol=1e-12):
        raise RuntimeError("Unbalance changed total bus demand")
    return {"kind": "phase_load_redistribution", "bus": bus, "increased_phase": phase,
        "delta": delta, "phase_factors": factors, "total_kw_kvar": before.tolist(), "commands": commands}


def physical_snapshot(build: dict, *, hif: dict | None = None, unbalance: dict | None = None) -> tuple[np.ndarray, dict]:
    dss = compile_model(Path(build["output_dir"]) / "Master.dss")
    disturbances = []
    if unbalance:
        disturbances.append(redistribute_phases(dss, build["registry"], **unbalance))
    receipt = None
    if hif:
        receipt = inject_midspan_hif(dss, build["registry"], build["assumptions"], **hif)
        audit = audit_disturbed_circuit(dss, receipt, build["registry"], build["assumptions"])
        if not audit["passed"]:
            raise RuntimeError(f"HIF physical audit failed: {audit['failed_checks']}")
        disturbances.append({"kind": "hif", "receipt": receipt, "audit": audit})
    telemetry = extract_measurements(dss, build["registry"], build["assumptions"],
        branch_overrides=receipt["branch_overrides"] if receipt else None)
    z = np.asarray(telemetry["measurement_vector"], dtype=float)
    if z.shape != (122,) or not np.isfinite(z).all():
        raise RuntimeError("Physical export must contain 122 finite external channels")
    kcl = float(telemetry["max_kcl_mismatch_pu"])
    if kcl > 1e-7:
        raise RuntimeError(f"External bus KCL audit failed: {kcl}")
    vuf = max(float(row["vln_sequence_pu"][2]) / float(row["vln_sequence_pu"][1])
              for row in telemetry["three_phase_voltages"])
    return z, {"engine": dss.Basic.Version(), "converged": True,
        "external_kcl_max_mismatch_pu": kcl, "maximum_voltage_negative_positive_ratio": vuf,
        "disturbances": disturbances,
        "exporter": "three_phase_model.measurements.extract_measurements:measurement_vector"}


def healthy_audit(build: dict, z: np.ndarray) -> dict:
    reference = build["reference"]
    fitted, _ = state_measurements_and_jacobian(configured_case(reference),
        np.deg2rad(reference["bus"][:, 8]), reference["bus"][:, 7])
    maximum = float(np.max(np.abs(z - fitted)))
    sigma = default_measurement_sigma(14, 20)
    discrepancy = float(np.dot((z - fitted) / sigma, (z - fitted) / sigma))
    if maximum > 1e-5:
        raise RuntimeError(f"Healthy physical export disagrees with balanced equations: {maximum}")
    return {"passed": True, "maximum_balanced_equation_error_pu": maximum,
            "squared_sigma_scaled_discrepancy": discrepancy, "tolerance_pu": 1e-5,
            "shunt_handling": "measured generator-minus-load injections; bus shunts remain in configured Ybus"}


def competing_variant(case: dict, z: np.ndarray, *, family: str,
                      severity: str, rng: np.random.Generator) -> tuple[dict, np.ndarray, dict]:
    configured, observed = copy.deepcopy(case), z.copy()
    level = SEVERITIES.index(severity)
    if family == "measurement":
        channel = int(rng.integers(len(z)))
        bias_sigma = (3., 7., 15.)[level] * float(rng.choice([-1, 1]))
        observed[channel] += bias_sigma * default_measurement_sigma(14, 20)[channel]
        metadata = {"channel_index0": channel, "additive_bias_sigma": bias_sigma}
    elif family == "parameter":
        row = int(rng.choice(np.flatnonzero(case["branch"][:, 10])))
        column = int(rng.choice([2, 3]))
        if column == 2 and case["branch"][row, column] == 0:
            column = 3
        multiplier = float(1 + rng.choice([-1, 1]) * (.07, .16, .30)[level])
        configured["branch"][row, column] *= multiplier
        metadata = {"branch_row0": row, "parameter_column0": column,
            "configured_multiplier": multiplier, "physical_model_unchanged": True}
    elif family == "topology":
        candidates = status_candidates(case)
        # A physically open line is intentionally reported closed in half of
        # eligible parents; the remainder report an actual closed line open.
        opened = [r for r in candidates if case["branch"][r, 10] == 0]
        row = int(rng.choice(opened if opened and rng.random() < .5 else candidates))
        configured["branch"][row, 10] = 1 - configured["branch"][row, 10]
        metadata = {"branch_row0": row, "true_status": int(case["branch"][row, 10]),
            "configured_status": int(configured["branch"][row, 10]),
            "physical_model_unchanged": True, "configured_graph_connected": connected(configured)}
    else:
        raise ValueError(f"Unsupported competing family: {family}")
    return configured, observed, {"kind": family, **metadata}


def generate_corpus(output_dir: str | Path, *, parents_by_split: dict[str, int], seed: int = 20260916,
                    noise_replicates: int = 4, healthy_calibration_replicates: int = 100,
                    variants_per_family: int = 3,
                    healthy_replicates_by_split: dict[str, int] | None = None) -> dict:
    """Write physical noiseless means; the dataset loader draws all sensor noise.

    Existing outputs are refused. Failures are explicit in generation_report.json
    and failed_variants.jsonl; a failed healthy physical audit aborts generation.
    """
    if set(parents_by_split) - set(SPLITS) or any(n < 0 for n in parents_by_split.values()):
        raise ValueError("Invalid split parent counts")
    if min(noise_replicates, healthy_calibration_replicates, variants_per_family) < 1:
        raise ValueError("Replicate and variant counts must be positive")
    healthy_replicates = {split: noise_replicates for split in SPLITS}
    healthy_replicates["calibration"] = healthy_calibration_replicates
    healthy_replicates.update(healthy_replicates_by_split or {})
    if set(healthy_replicates) != set(SPLITS) or min(healthy_replicates.values()) < 1:
        raise ValueError("Invalid healthy replicate counts")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sigma = default_measurement_sigma(14, 20).tolist()
    write_json(out / "measurement_sigma.json", sigma)
    rng = np.random.default_rng(seed)
    counts: Counter = Counter()
    failures = []
    parent_rejections = []
    audits = []
    implementation = [Path(__file__), *sorted((Path(__file__).parents[2] / "three_phase_model").glob("*.py"))]
    source_hashes = {str(p.relative_to(Path(__file__).parents[2])): file_sha256(p) for p in implementation}
    rows_count = 0
    ordinal = 0
    manifest = out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for split in SPLITS:
            for split_ordinal in range(int(parents_by_split.get(split, 0))):
                parent_id = f"ieee14_{seed}_{split}_{split_ordinal:05d}"
                parent_seed = int(rng.integers(0, 2**63 - 1))
                parent_rng = np.random.default_rng(parent_seed)
                for attempt in range(20):
                    proposed_case, operating_point = sample_parent(parent_rng, ordinal)
                    try:
                        physical_case, opf_audit = solve_operating_parent(proposed_case)
                        operating_point["source_opf"] = opf_audit
                        operating_point["accepted_attempt_index0"] = attempt
                        break
                    except (RuntimeError, ValueError) as exc:
                        parent_rejections.append({"parent_id": parent_id, "attempt_index0": attempt,
                            "reason": str(exc), "proposed_operating_point": operating_point})
                        write_json(out / "rejected_parent_attempts.json", parent_rejections)
                else:
                    raise RuntimeError(f"No feasible operating parent after 20 source attempts: {parent_id}")
                parent_dir = out / "parents" / parent_id
                build = export_model(physical_case, parent_dir / "model", case_id="case14",
                    assumptions=load_assumptions("normalized_diagonal"),
                    source_provenance={"parent_id": parent_id, "seed": parent_seed, "operating_point": operating_point})
                base_case = configured_case(physical_case)
                healthy_z, baseline_metadata = physical_snapshot(build)
                audit = healthy_audit(build, healthy_z)
                write_json(parent_dir / "healthy_audit.json", audit)
                write_json(parent_dir / "source_opf_audit.json", opf_audit)
                audits.append(audit)
                known_case_path = parent_dir / "configured_case.json"
                write_json(known_case_path, base_case)
                variants: list[tuple[str, str, dict, dict | None, dict | None, list[str]]] = []
                variants.append(("healthy", "healthy", {}, None, None, []))
                if split != "calibration":
                    hif_rows = eligible_hif_branch_rows(build["registry"])
                    load_buses = sorted({row["bus"] for row in build["registry"]["loads"]})
                    for index in range(variants_per_family):
                        severity = SEVERITIES[index % 3]
                        hif = {"branch_row0": int(parent_rng.choice(hif_rows)),
                            "phase": int(parent_rng.integers(1, 4)), "alpha": float(parent_rng.uniform(.15, .85)),
                            "resistance_pu": float((120., 24., 5.)[index % 3] * np.exp(parent_rng.uniform(-.2, .2)))}
                        unbalance = {"bus": int(parent_rng.choice(load_buses)),
                            "phase": int(parent_rng.integers(1, 4)), "delta": float((.08, .30, .65)[index % 3])}
                        variants += [(f"hif_{index}", severity, {}, hif, None, ["hif"]),
                            (f"unbalance_{index}", severity, {}, None, unbalance, ["unbalance"]),
                            (f"measurement_{index}", severity, {"family": "measurement"}, None, None, ["measurement"]),
                            (f"parameter_{index}", severity, {"family": "parameter"}, None, None, ["parameter"])]
                    variants.append(("topology", "strong", {"family": "topology"}, None, None, ["topology"]))
                    first_hif = next(v[3] for v in variants if v[0] == "hif_0")
                    split_z, split_physics = physical_snapshot(build, hif={**first_hif, "enabled": False})
                    split_audit = healthy_audit(build, split_z)
                    split_audit["maximum_unsplit_measurement_difference_pu"] = float(np.max(np.abs(split_z - healthy_z)))
                    split_audit["physics"] = split_physics
                    write_json(parent_dir / "disabled_hif_split_control_audit.json", split_audit)
                    # Mixed variants reuse paired physical fault realizations.
                    mids = next(v for v in variants if v[0] == f"hif_{min(1, variants_per_family-1)}")
                    midu = next(v for v in variants if v[0] == f"unbalance_{min(1, variants_per_family-1)}")
                    variants += [("hif_measurement", "intermediate", {"family": "measurement"}, mids[3], None, ["hif", "measurement"]),
                        ("unbalance_parameter", "intermediate", {"family": "parameter"}, None, midu[4], ["unbalance", "parameter"]),
                        ("hif_unbalance", "intermediate", {}, mids[3], midu[4], ["hif", "unbalance"]),
                        ("hif_topology", "intermediate", {"family": "topology"}, mids[3], None, ["hif", "topology"])]
                for variant_name, severity, competing, hif, unbalance, families in variants:
                    try:
                        z, physics = ((healthy_z.copy(), copy.deepcopy(baseline_metadata))
                            if not (hif or unbalance) else physical_snapshot(build, hif=hif, unbalance=unbalance))
                        configured = base_case
                        if competing:
                            configured, z, corruption = competing_variant(base_case, z, severity=severity,
                                rng=parent_rng, **competing)
                            physics["configured_or_measurement_corruption"] = corruption
                        case_path = known_case_path
                        if competing.get("family") in ("parameter", "topology"):
                            case_path = parent_dir / f"{variant_name}_configured_case.json"
                            write_json(case_path, configured)
                        metadata_path = parent_dir / f"{variant_name}_physical_audit.json"
                        write_json(metadata_path, physics)
                        row = {"case": case_path.relative_to(out).as_posix(), "z": z.tolist(),
                            "parent_id": parent_id, "split": split, "families": families,
                            "severity": severity, "window_id": f"{parent_id}:{variant_name}",
                            "measurement_kind": "noiseless_mean", "measurement_convention": MEASUREMENT_CONVENTION,
                            "measurement_sigma": "measurement_sigma.json",
                            "noise_seed": int(parent_rng.integers(0, 2**63 - 1)),
                            "noise_replicates": healthy_replicates[split] if not families else noise_replicates,
                            "offline_metadata": {"physical_audit_path": metadata_path.relative_to(out).as_posix(),
                                "parent_seed": parent_seed, "physical_case_hash": build["manifest"]["base_case_hash"],
                                "physical_source": "normalized_diagonal_opendss_fixed_pq_snapshot",
                                "physical_scope": "fundamental_frequency_resistive_hif_and_load_unbalance",
                                "measurement_source": baseline_metadata["exporter"], "variant": variant_name,
                                "operating_point": operating_point,
                                "maximum_voltage_negative_positive_ratio": physics["maximum_voltage_negative_positive_ratio"],
                                "hif_phase": hif["phase"] if hif else None,
                                "unbalance_increased_phase": unbalance["phase"] if unbalance else None,
                                "affected_phase": ("hif_" + str(hif["phase"]) + "_unbalance_" + str(unbalance["phase"]))
                                    if hif and unbalance else str((hif or unbalance or {}).get("phase", "none")),
                                "full_provenance_hash": content_hash(physics)}}
                        handle.write(json.dumps(jsonable(row), allow_nan=False, separators=(",", ":")) + "\n")
                        counts[f"{split}:rows"] += 1
                        counts[f"{split}:noise_windows"] += row["noise_replicates"]
                        for family in families or ["healthy"]:
                            counts[f"{split}:{family}"] += 1
                        rows_count += 1
                    except (RuntimeError, ValueError) as exc:
                        failures.append({"parent_id": parent_id, "split": split,
                            "variant": variant_name, "families": families, "error": str(exc)})
                handle.flush()
                counts[f"{split}:parents"] += 1
                ordinal += 1
                print(json.dumps({"event": "parent_complete", "parent_id": parent_id,
                    "completed_parents": ordinal, "rows": rows_count, "failed_variants": len(failures),
                    "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
    (out / "failed_variants.jsonl").write_text("".join(json.dumps(v) + "\n" for v in failures), encoding="utf-8")
    write_json(out / "rejected_parent_attempts.json", parent_rejections)
    report = {"schema": "physical_wls_screen_corpus_v1", "manifest": str(manifest), "seed": seed,
        "parents_by_split": parents_by_split, "noise_replicates": noise_replicates,
        "healthy_replicates_by_split": healthy_replicates,
        "counts": dict(counts), "failed_variants": failures, "rejected_parent_attempts": parent_rejections,
        "manifest_sha256": file_sha256(manifest), "implementation_sha256": source_hashes,
        "healthy_max_balanced_equation_error_pu": max((a["maximum_balanced_equation_error_pu"] for a in audits), default=None),
        "healthy_max_squared_sigma_scaled_discrepancy": max((a["squared_sigma_scaled_discrepancy"] for a in audits), default=None),
        "measurement_convention": MEASUREMENT_CONVENTION, "elapsed_seconds": time.monotonic()-started,
        "severity_definitions": {"hif_resistance_pu": {"weak": 120, "intermediate": 24, "strong": 5,
            "multiplicative_jitter": "exp(uniform(-0.2,0.2))"},
            "unbalance_delta": dict(zip(SEVERITIES, [.08, .30, .65])),
            "measurement_bias_abs_sigma": dict(zip(SEVERITIES, [3, 7, 15])),
            "parameter_relative_abs_error": dict(zip(SEVERITIES, [.07, .16, .30])),
            "topology": "one connected configured status toggle; strong is a family-specific label",
            "mixtures": "intermediate physical settings with intermediate competing corruption",
            "comparability": "severity describes physical settings within family, not equal detectability across families"},
        "limitations": ["Normalized diagonal phase impedance realization, not measured phase network parameters",
            "OPF-valid baseline with source P/Q/voltage limits; post-disturbance fixed-PQ replay has no PV regulation",
            "HIF is a steady-state resistor; no arcing, waveform, harmonic, or dynamic accuracy claim",
            "Independent randomized operating parents, with related physical/noise variants grouped",
            "Single IEEE-14 system; no IEEE-57 transfer evidence"]}
    write_json(out / "generation_report.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    for split, default in zip(SPLITS, (100, 30, 100, 50)):
        parser.add_argument(f"--{split}-parents", type=int, default=default)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--noise-replicates", type=int, default=4)
    parser.add_argument("--healthy-calibration-replicates", type=int, default=100)
    parser.add_argument("--healthy-validation-replicates", type=int)
    parser.add_argument("--healthy-test-replicates", type=int)
    parser.add_argument("--variants-per-family", type=int, default=3)
    args = parser.parse_args(argv)
    report = generate_corpus(args.output_dir,
        parents_by_split={split: getattr(args, f"{split}_parents") for split in SPLITS},
        seed=args.seed, noise_replicates=args.noise_replicates,
        healthy_calibration_replicates=args.healthy_calibration_replicates,
        variants_per_family=args.variants_per_family,
        healthy_replicates_by_split={split: value for split, value in
            (("validation", args.healthy_validation_replicates), ("test", args.healthy_test_replicates)) if value is not None})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
