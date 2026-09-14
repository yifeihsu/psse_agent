"""Independently audit a finished IEEE57 disturbance run and replay its circuits.

This validates saved physics, coverage, provenance, and acquisition boundaries.
It does not rerun the diagnostic algorithms or turn a correct candidate into
an accepted parameter estimate. Replays use only offline injection receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROFILES = {"exact", "noisy_nominal", "noisy_precision_sensitivity"}
OBSERVATION_KEYS = {"measurement_vector", "three_phase_voltages", "three_phase_branch_currents"}
VOLTAGE_KEYS = {"external_bus", "bus", "row0", "vln_pu_rect"}
CURRENT_KEYS = {"asset_id", "branch_row0", "from_bus", "to_bus", "i_from_pu_rect", "i_to_pu_rect"}
REQUIRED_PHYSICS = {
    "healthy": ("physics",),
    "unbalance": ("physics", "restoration", "restoration_physics"),
    "hif": ("physics", "split_null", "split_null_physics", "split_primitive_audit",
            "fault_physics", "fault_removed", "fault_removed_physics", "restored_primitive_audit",
            "restoration", "restoration_physics"),
}


def _json(path: Path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under(root: Path, value: str) -> Path:
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes run directory: {value}")
    return path


def _finite_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    original = np.asarray(value, dtype=object)
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float, np.integer, np.floating))
           for item in original.flat):
        raise ValueError("Measurement arrays must contain JSON numbers, not booleans or strings")
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite array with shape {shape}, got {array.shape}")
    return array


def _validate_observations(payload, registry, assumptions):
    """Strict schemas and exact registry bindings exclude hidden/truth aliases."""
    if set(payload) != PROFILES:
        raise ValueError("Observation profiles differ from the acquisition contract")
    buses, branches = registry["buses"], registry["branches"]
    count = 3 * len(buses) + 4 * len(branches)
    arrays = {}
    for profile, observed in payload.items():
        if set(observed) != OBSERVATION_KEYS:
            raise ValueError("Unexpected observation fields, including possible privileged metadata")
        z = _finite_array(observed["measurement_vector"], (count,))
        if len(observed["three_phase_voltages"]) != len(buses) or len(observed["three_phase_branch_currents"]) != len(branches):
            raise ValueError("Observation bus/branch count disagrees with the external registry")
        for expected, row in zip(buses, observed["three_phase_voltages"]):
            if set(row) != VOLTAGE_KEYS or any(row[key] != expected[source] for key, source in (
                ("external_bus", "external_bus"), ("bus", "dss_bus"), ("row0", "row0")
            )):
                raise ValueError("Voltage observations contain unknown keys, hidden bus aliases, or wrong ordering")
            _finite_array(row["vln_pu_rect"], (3, 2))
        for expected, row in zip(branches, observed["three_phase_branch_currents"]):
            if set(row) != CURRENT_KEYS or any(row[key] != expected[key] for key in ("asset_id", "branch_row0", "from_bus", "to_bus")):
                raise ValueError("Current observations contain unknown keys, hidden element aliases, or wrong ordering")
            for key in ("i_from_pu_rect", "i_to_pu_rect"):
                _finite_array(row[key], (3, 2))
        arrays[profile] = {
            "z": z,
            "v": np.asarray([r["vln_pu_rect"] for r in observed["three_phase_voltages"]]),
            "if": np.asarray([r["i_from_pu_rect"] for r in observed["three_phase_branch_currents"]]),
            "it": np.asarray([r["i_to_pu_rect"] for r in observed["three_phase_branch_currents"]]),
        }
    if not np.array_equal(arrays["noisy_nominal"]["z"], arrays["noisy_precision_sensitivity"]["z"]):
        raise ValueError("Precision sensitivity unexpectedly changes the separate SCADA noise realization")
    # The experiment explicitly pairs standardized phasor draws across profiles.
    for key in ("v", "if", "it"):
        sigma_key = "voltage_sigma_pu" if key == "v" else "current_sigma_pu"
        ratio = assumptions["nominal"][sigma_key] / assumptions["precision_sensitivity"][sigma_key]
        discrepancy = arrays["noisy_nominal"][key] - arrays["exact"][key] - ratio * (arrays["noisy_precision_sensitivity"][key] - arrays["exact"][key])
        if float(np.max(np.abs(discrepancy))) > 2e-13:
            raise ValueError("Paired precision-profile phasor noise draws are inconsistent")
    return arrays["exact"]


def audit_artifacts(output_dir: str | Path, *, report_path: str | Path | None = None,
                    extended_replays: bool = False) -> dict:
    root = Path(output_dir).resolve(strict=True)
    destination = Path(report_path).resolve() if report_path else root / "artifact_audit.json"
    if destination.exists():
        raise FileExistsError(f"Choose a new report path to retain earlier audit evidence: {destination}")
    config = _json(root / "experiment_config.json")
    receipt = _json(root / "run_receipt.json")  # only finished runs have this receipt
    failures, model_reports, replays, maxima = [], [], [], {}

    def check(condition, code, detail):
        if not condition:
            failures.append({"code": code, "detail": detail})

    expected_models = {f"{name}_{scale:03d}" for name in ("normalized_diagonal", "coupled_sensitivity") for scale in (80, 100)}
    check({m["model_id"] for m in receipt["models"]} == expected_models and len(receipt["models"]) == 4,
          "model_coverage", "Expected diagonal/coupled models at both 0.8 and 1.0 load")
    check(receipt.get("all_sources_unchanged_during_run") is True and config["source_before"] == receipt["source_after"],
          "run_source_integrity", "Recorded implementation hashes changed during the run")
    source_now = {name: _sha(_under(REPO, name)) for name in config["source_before"]}
    current_matches = source_now == config["source_before"]
    snapshot_root = root / "implementation_snapshot"
    snapshot_hashes = {name: _sha(snapshot_root / name) if (snapshot_root / name).is_file() else None
                       for name in config["source_before"]}
    snapshot_matches = snapshot_hashes == config["source_before"]
    check(current_matches or snapshot_matches, "implementation_reproducibility",
          "Neither current source nor preserved implementation snapshot matches the recorded run")
    scenario_ids, disturbance_roots, healthy_roots = set(), set(), set()
    global_counts, physics_counts = Counter(), Counter()
    parameter_counts = {profile: Counter() for profile in PROFILES}
    parameter_errors = defaultdict(list)
    observation_count = 0
    artifact_hashes = {"experiment_config.json": _sha(root / "experiment_config.json"), "run_receipt.json": _sha(root / "run_receipt.json")}
    for model in sorted(receipt["models"], key=lambda value: value["model_id"]):
        model_id = model["model_id"]
        result_path = _under(root, model["results_path"])
        directory = result_path.parent
        registry = _json(directory / "model/asset_registry.json")
        assumptions = _json(directory / "model/assumptions.json")
        source_case = _json(directory / "model/source_case.json")
        manifest = _json(directory / "model/build_manifest.json")
        expected_case_hash = hashlib.sha256(json.dumps(source_case, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        check(expected_case_hash == registry["base_case_hash"] == manifest["base_case_hash"], "source_case_hash", model_id)
        for name, digest in manifest["files_sha256"].items():
            check(_sha(_under(directory / "model", name)) == digest, "model_file_hash", f"{model_id}/{name}")
        check(_json(directory / "balanced_validation.json").get("passed") is True, "baseline_validation", model_id)
        rows = _json(result_path)
        counts = Counter(row["family"] for row in rows)
        eligible_lines = {b["branch_row0"] for b in registry["branches"] if b["status"] == 1 and b["dss_element"].startswith("Line.")}
        load_buses = {row["bus"] for row in registry["loads"]}
        smoke = config["preset"] == "smoke"
        expected = {"healthy": 3 if smoke else 100, "hif": 9 if smoke else 3 * len(eligible_lines),
                    "unbalance": 6 if smoke else 2 * len(load_buses)}
        check(dict(counts) == expected, "family_counts", {"model": model_id, "expected": expected, "actual": dict(counts)})
        check(model["row_count"] == len(rows) and model["scenario_count"] == counts["hif"] + counts["unbalance"], "model_receipt_counts", model_id)
        if not smoke:
            hif_pairs = [(r["truth"]["branch_row0"], r["truth"]["phase"]) for r in rows if r["family"] == "hif"]
            unbalance_pairs = [(r["truth"]["bus"], r["truth"]["delta"]) for r in rows if r["family"] == "unbalance"]
            check(Counter(hif_pairs) == Counter((line, phase) for line in eligible_lines for phase in (1, 2, 3)), "line_phase_coverage", model_id)
            check(Counter(unbalance_pairs) == Counter((bus, delta) for bus in load_buses for delta in config["unbalance_deltas"]), "load_delta_coverage", model_id)
        master = directory / "model/Master.dss"
        for row in rows:
            label, family = row["scenario_id"], row["family"]
            for profile in PROFILES:
                parameter_counts[profile]["hif_cases"] += family == "hif"
            check(label not in scenario_ids, "duplicate_scenario_id", label)
            scenario_ids.add(label)
            check("execution_failure" not in row, "execution_failure", label)
            truth = {"family": "healthy"} if family == "healthy" else row["truth"]
            fingerprint = hashlib.sha256(json.dumps({"case_hash": registry["base_case_hash"], "assumptions": assumptions,
                                                     "disturbance": truth}, sort_keys=True).encode()).hexdigest()
            check(row.get("physical_root_fingerprint") == fingerprint, "physical_root_fingerprint", label)
            if family == "healthy":
                healthy_roots.add(fingerprint)
            else:
                check(fingerprint not in disturbance_roots, "duplicate_disturbance_root", label)
                disturbance_roots.add(fingerprint)
            for key in REQUIRED_PHYSICS.get(family, ()):
                evidence = row.get(key, {})
                check(evidence.get("passed") is True, "required_physics", f"{label}/{key}")
                physics_counts[key] += evidence.get("passed") is True
                for name, values in evidence.get("checks", {}).items():
                    check(values.get("passed") is True, "physics_subcheck", f"{label}/{key}/{name}")
                    error = values.get("max_error", values.get("max_error_pu"))
                    limit = values.get("limit", values.get("tolerance_pu", values.get("tolerance")))
                    if error is not None:
                        check(isinstance(error, (float, int)) and math.isfinite(error) and error >= 0,
                              "physics_numeric_evidence", f"{label}/{key}/{name}")
                        if limit is not None:
                            check(error <= limit, "physics_numeric_limit", f"{label}/{key}/{name}")
                        maxima[f"{key}/{name}"] = max(maxima.get(f"{key}/{name}", 0), error)
            if not row.get("observations_path"):
                # Failed physical admission is deliberately retained by the
                # generator. Report it as an audit failure instead of crashing
                # or quietly counting the absent acquisition as checked.
                check(False, "missing_observations", label)
                continue
            observations_path = _under(root, row["observations_path"])
            check(_sha(observations_path) == row["observations_sha256"], "observation_sha256", label)
            with gzip.open(observations_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            try:
                _validate_observations(payload, registry, config["phase_profiles"])
            except (ValueError, KeyError, TypeError) as exc:
                check(False, "observation_allowlist_or_noise", f"{label}: {exc}")
            observation_count += 1
            for profile in PROFILES:
                candidate = row.get("phase_diagnostics", {}).get(profile, {}).get("hif_candidate")
                parameter_counts[profile]["hif_cases_with_observations"] += family == "hif"
                if not candidate:
                    continue
                correct = (family == "hif" and candidate.get("branch_row0") == truth["branch_row0"]
                           and candidate.get("asset_id") == truth["asset_id"] and candidate.get("phase") == truth["phase"])
                accepted = candidate.get("parameter_estimates_accepted") is True
                parameter_counts[profile]["emitted_hif_candidates"] += 1
                parameter_counts[profile]["correct_branch_and_phase"] += correct
                parameter_counts[profile]["accepted_parameter_candidates"] += accepted
                parameter_counts[profile]["accepted_correct_branch_and_phase"] += accepted and correct
                parameter_counts[profile]["accepted_wrong_branch_phase_or_family"] += accepted and not correct
                check(isinstance(candidate.get("parameter_estimates_accepted"), bool), "parameter_acceptance_flag", f"{label}/{profile}")
                alpha, resistance = candidate.get("alpha_estimate"), candidate.get("resistance_pu_estimate")
                if accepted:
                    alpha_sigma, resistance_sigma = candidate.get("alpha_sigma_linearized"), candidate.get("resistance_sigma_linearized_pu")
                    imaginary = candidate.get("resistance_imaginary_pu")
                    values = (alpha, resistance, alpha_sigma, resistance_sigma, imaginary)
                    finite = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
                    check(finite and 0 < alpha < 1 and resistance > 0 and 0 <= alpha_sigma <= .1
                          and 0 <= resistance_sigma <= .25 * resistance
                          and alpha - 3 * alpha_sigma > 0 and alpha + 3 * alpha_sigma < 1
                          and abs(imaginary) <= max(.2 * resistance, 3 * resistance_sigma)
                          and candidate.get("distance_observable") is True and candidate.get("shunt_hypothesis_consistent") is True,
                          "accepted_parameter_bounds", f"{label}/{profile}")
                if correct and all(isinstance(v, (int, float)) and math.isfinite(v) for v in (alpha, resistance)):
                    parameter_errors[profile].append({"scenario_id": label, "accepted": accepted,
                        "resistance_true_pu": truth["resistance_pu"], "alpha_absolute_error": abs(alpha-truth["alpha"]),
                        "resistance_relative_error": abs(resistance/truth["resistance_pu"]-1)})
            if family != "healthy":
                injection = row["injection"]
                if family == "hif":
                    for key in ("alpha", "phase", "resistance_pu", "branch_row0", "asset_id"):
                        check(injection[key] == truth[key], "hif_receipt_truth_binding", f"{label}/{key}")
                    expected_resistance = truth["resistance_pu"] * assumptions["base_kv_ll"]**2 / assumptions["base_mva"]
                    check(math.isclose(injection["resistance_ohm"], expected_resistance, rel_tol=1e-12), "resistance_base", label)
                    check(injection["fault_enabled"] is True and injection["restored"] is False,
                          "fault_receipt_state", label)
                    check(row["physics"]["enabled_fault_count"] == 1, "enabled_fault_count", label)
                    check(truth["alpha"] in config["hif_alpha"] and truth["resistance_pu"] in config["hif_resistance_pu"], "fault_parameter_design", label)
                else:
                    check(injection["bus"] == truth["bus"] and injection["delta"] == truth["delta"], "unbalance_receipt_binding", label)
                replay_path = _under(root, row["replay_path"])
                expected_replay = ('! Fresh IEEE57 fundamental-frequency research scenario\n'
                    f'Redirect "{master.as_posix()}"\n' + "\n".join(injection.get("commands", []))
                    + (f"\nEdit {injection['fault_element']} Enabled=yes" if family == "hif" else "") + "\nSolve\n")
                check(replay_path.read_text() == expected_replay, "replay_receipt_binding", label)
        global_counts.update(counts)
        artifact_hashes[str(result_path.relative_to(root))] = _sha(result_path)
        # Base charged line exercises Cmatrix restoration; optional additions
        # cover all phases and all resistance groups without hundreds of replays.
        selected = []

        def select_replay(predicate, label):
            found = next((row for row in rows if row.get("replay_path") and predicate(row)), None)
            if found is None:
                check(False, "missing_standalone_replay", f"{model_id}/{label}")
            else:
                selected.append(found)

        select_replay(lambda row: row["family"] == "hif", "hif")
        select_replay(lambda row: row["family"] == "unbalance", "unbalance")
        if extended_replays:
            for phase in (1, 2, 3):
                select_replay(lambda row: row["family"] == "hif" and row["truth"]["phase"] == phase, f"phase{phase}")
            for resistance in config["hif_resistance_pu"]:
                select_replay(lambda row: row["family"] == "hif" and row["truth"]["resistance_pu"] == resistance, f"Rpu{resistance}")
        if not smoke and model_id == "coupled_sensitivity_080":
            select_replay(lambda row: row["family"] == "hif" and row["truth"]["branch_row0"] == 42
                          and row["truth"]["alpha"] == .2 and row["truth"]["phase"] == 2
                          and row["truth"]["resistance_pu"] == 10.0,
                          "previously_failed_new_node_initialization_regression")
        selected = list({r["scenario_id"]: r for r in selected}.values())
        for row in selected:
            from three_phase_model.runtime import compile_model
            from three_phase_model.measurements import extract_measurements
            from three_phase_model.disturbances import audit_disturbed_circuit
            from scripts.validate_ieee57_disturbances import circuit_physics
            engine = compile_model(_under(root, row["replay_path"]))
            telemetry = extract_measurements(engine, registry, assumptions, branch_overrides=row["injection"].get("branch_overrides"))
            with gzip.open(_under(root, row["observations_path"]), "rt", encoding="utf-8") as stream:
                saved = json.load(stream)["exact"]
            errors = {"measurement_vector": float(np.max(np.abs(np.asarray(telemetry["measurement_vector"]) - saved["measurement_vector"])))}
            for family, fields in (("three_phase_voltages", ("vln_pu_rect",)), ("three_phase_branch_currents", ("i_from_pu_rect", "i_to_pu_rect"))):
                for field in fields:
                    errors[field] = float(np.max(np.abs(np.asarray([r[field] for r in telemetry[family]]) - np.asarray([r[field] for r in saved[family]]))))
            passed = errors["measurement_vector"] <= 1e-8 and all(v <= 1e-10 for key, v in errors.items() if key != "measurement_vector")
            record = {"scenario_id": row["scenario_id"], "family": row["family"], "converged": bool(engine.Solution.Converged()),
                      "max_absolute_errors_pu": errors, "bit_exact_numeric_channels": all(v == 0 for v in errors.values()),
                      "passed": passed, "replay_sha256": _sha(_under(root, row["replay_path"]))}
            record["circuit_physics"] = circuit_physics(engine, registry, assumptions)
            record["passed"] &= record["circuit_physics"]["passed"]
            record["previously_failed_initialization_case"] = (model_id == "coupled_sensitivity_080" and row["family"] == "hif"
                                                               and row["truth"]["branch_row0"] == 42 and row["truth"]["phase"] == 2)
            if row["family"] == "hif":
                record["fault_physics"] = audit_disturbed_circuit(engine, row["injection"], registry, assumptions)
                record["passed"] &= record["fault_physics"]["passed"]
            check(record["passed"], "standalone_replay", row["scenario_id"])
            replays.append(record)
        model_reports.append({"model_id": model_id, "counts": dict(counts), "source_case_hash": expected_case_hash,
                              "external_buses": len(registry["buses"]), "eligible_hif_lines": len(eligible_lines),
                              "load_buses": len(load_buses), "replay_count": len(selected)})
        print(f"Audited {model_id}: {len(rows)} result rows and {len(selected)} standalone replays", flush=True)
    check(len(healthy_roots) == 4, "healthy_root_count", len(healthy_roots))
    check(not (healthy_roots & disturbance_roots), "root_overlap", "Healthy and disturbance fingerprints overlap")
    check(len(disturbance_roots) == global_counts["hif"] + global_counts["unbalance"], "unique_disturbance_count", len(disturbance_roots))
    def error_summary(values):
        if not values:
            return {"count": 0}
        return {"count": len(values), **{name: {"median": float(np.median([v[name] for v in values])),
                    "p95": float(np.quantile([v[name] for v in values], .95)), "maximum": max(v[name] for v in values)}
                    for name in ("alpha_absolute_error", "resistance_relative_error")}}
    parameters = {profile: {"counts": dict(parameter_counts[profile]),
                    "errors_given_correct_branch_and_phase": error_summary(parameter_errors[profile]),
                    "errors_given_correct_branch_phase_and_accepted_parameters": error_summary([v for v in parameter_errors[profile] if v["accepted"]]),
                    "accepted_by_true_resistance_pu": {str(resistance): error_summary([v for v in parameter_errors[profile]
                        if v["accepted"] and v["resistance_true_pu"] == resistance]) for resistance in config["hif_resistance_pu"]}}
                  for profile in sorted(PROFILES)}
    import opendssdirect
    result = {"contract": "ieee57_disturbance_artifact_independent_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "passed": not failures, "failures": failures, "output_dir": str(root), "preset": config["preset"],
              "script_sha256": _sha(Path(__file__)), "python_version": sys.version, "numpy_version": np.__version__,
              "opendssdirect_version": opendssdirect.__version__, "opendss_engine_version": opendssdirect.Basic.Version(),
              "row_counts": dict(global_counts), "healthy_physical_roots": len(healthy_roots),
              "distinct_disturbance_roots": len(disturbance_roots), "acquisition_files_audited": observation_count,
              "acquisition_profiles_per_file": 3, "physics_pass_counts": dict(physics_counts),
              "maximum_recorded_physics_errors": maxima, "models": model_reports, "replays": replays,
              "artifact_sha256": artifact_hashes, "source_sha256_current": source_now,
              "current_implementation_matches_run": current_matches,
              "preserved_implementation_snapshot_matches_run": snapshot_matches,
              "implementation_snapshot_sha256": snapshot_hashes,
              "parameter_acceptance_audit": parameters,
              "parameter_interpretation": "Alpha is fraction of equivalent series impedance, not geographic distance. Accepted estimates pass declared linearized uncertainty gates; error summaries are conditional on correctly selected branch and phase, not empirical confidence coverage.",
              "replay_tolerances_pu": {"operator": 1e-8, "phasor_component": 1e-10},
              "scope": "All recorded physics/coverage/input-schema/hash checks plus selected independent OpenDSS replays. No rerun of diagnosis, parameter-fit acceptance, or population calibration."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--extended-replays", action="store_true")
    args = parser.parse_args(argv)
    result = audit_artifacts(args.output_dir, report_path=args.report_path, extended_replays=args.extended_replays)
    print(json.dumps({"passed": result["passed"], "row_counts": result["row_counts"],
                      "acquisition_files_audited": result["acquisition_files_audited"], "replays": len(result["replays"]),
                      "failure_count": len(result["failures"])}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
