"""IEEE57 physical HIF sweep: 5% chi-square primary, 1% same-observation audit.

The selected reconstruction uses 138 kV at buses 1--17 and 69 kV at 18--57.
All other sweep mechanics are shared with the backward-compatible IEEE14 runner.
The disjoint-shard runner and merger also serve IEEE118
(scripts/audit_ieee118_hif_physical_sweep.py).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import audit_ieee14_hif_physical_sweep as sweep


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_rows(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            sweep._write_row(handle, row)


def validate_population_coverage(config, cases, controls, groups, observations):
    """Require the entire declared grid, including every available noisy view.

    Physical failures still occupy their case/control cells. They require no
    fictitious observation. WLS failures are actual attempted observations and
    must remain present. This check is independent of alarm outcomes.
    """
    parent_ids = [f"parent_{i:02d}_load_{scale:g}" for i, scale in enumerate(config["load_scales"])]
    branches, phases = config["selected_branch_rows0"], config["phases"]
    expected_cases, expected_controls, expected_groups, expected_observations = {}, {}, {}, set()
    for parent_index, parent in enumerate(parent_ids):
        healthy_id = parent + "_healthy"
        expected_controls[healthy_id] = {"kind": "healthy", "parent_id": parent}
        for branch in branches:
            split_id = f"{parent}_line_{branch:02d}_split"
            expected_controls[split_id] = {"kind": "no_fault_split", "parent_id": parent, "branch_row0": branch}
            for phase in phases:
                active = []
                for resistance in config["resistances_ohm"]:
                    case_id = f"{parent}_line_{branch:02d}_phase_{phase}_R_{resistance:g}"
                    expected_cases[case_id] = {"kind": "hif", "parent_id": parent, "branch_row0": branch,
                                              "phase": phase, "resistance_ohm": resistance}
                    if case_id in cases and cases[case_id]["physical_success"]:
                        active.append(case_id)
                for replica in range(config["noise_replicates"]):
                    group_id = f"{parent}_line_{branch:02d}_phase_{phase}_noise_{replica}"
                    entropy = [config["seed"], parent_index, branch, phase, replica]
                    expected_groups[group_id] = entropy
                    observed_ids = list(active)
                    for control_id in (healthy_id, split_id):
                        if control_id in controls and controls[control_id]["physical_success"]:
                            observed_ids.append(control_id)
                    for case_id in observed_ids:
                        expected_observations.update((case_id, group_id, profile) for profile in config["noise"]["sigma_z"])
    for label, actual, expected in (("physical case", cases, expected_cases), ("physical control", controls, expected_controls),
                                    ("noise group", groups, expected_groups), ("noisy observation", observations, expected_observations)):
        missing, unexpected = set(expected)-set(actual), set(actual)-set(expected)
        if missing or unexpected:
            raise ValueError(f"Incomplete {label} cross-product: missing={len(missing)}, unexpected={len(unexpected)}")
    for actual, expected in ((cases, expected_cases), (controls, expected_controls)):
        for key, fields in expected.items():
            if any(actual[key].get(field) != value for field, value in fields.items()):
                raise ValueError(f"Physical record identity disagrees with declared grid: {key}")
    measurement_count = config["measurement_count"]
    for group_id, entropy in expected_groups.items():
        row = groups[group_id]
        if row["seed_sequence_entropy"] != entropy:
            raise ValueError("Noise-group seed/parent/branch/phase identity mismatch")
        unit, _ = sweep.standard_noise(*entropy, size=measurement_count)
        if not sweep.np.array_equal(unit, row["unit_noise"]) or sweep.numeric_hash(unit) != row["unit_noise_sha256"]:
            raise ValueError("Noise-group realization differs from its declared seed")
    return {"passed": True, "operating_parent_count": len(parent_ids),
        "physical_case_count": len(cases), "physical_control_count": len(controls),
        "healthy_physical_controls": len(parent_ids), "no_fault_split_physical_controls": len(parent_ids)*len(branches),
        "noise_group_count": len(groups), "noisy_observation_count": len(observations),
        "fault_noisy_observations": sum(key[0] in cases for key in observations),
        "control_noisy_observations": sum(key[0] in controls for key in observations),
        "observation_expectation": "complete_profile_cross_product_for_every_available_physical_mean_and_matching_noise_group",
        "physical_failures_preserved": sum(not row["physical_success"] for row in cases.values()),
        "wls_failures_preserved": sum(not row["wls"]["success"] for row in observations.values())}


def validate_saved_population(directory):
    """Read-only post-run coverage check; never rewrites collection artifacts."""
    directory = Path(directory)
    config = json.loads((directory/"experiment_config.json").read_text())
    indexed = {}
    for name in ("cases", "controls", "noise_groups", "wls_observations"):
        values = {}
        for row in _read_rows(directory/f"{name}.jsonl"):
            key = ((row["case_id"], row["noise_group_id"], row["noise_profile"]) if name == "wls_observations" else
                   row["noise_group_id"] if name == "noise_groups" else row["case_id"])
            if key in values:
                raise ValueError(f"Duplicate saved {name} key: {key}")
            values[key] = row
        indexed[name] = values
    result = validate_population_coverage(config, indexed["cases"], indexed["controls"],
                                          indexed["noise_groups"], indexed["wls_observations"])
    result.update(contract="physical_hif_post_run_coverage_validation_v1",
        source_role="read_only_post_run_validator; original_collection_snapshots_are_preserved",
        collection_merger_sha256=config["implementation_sha256"].get("scripts/audit_ieee57_hif_physical_sweep.py"),
        post_run_validator_sha256=_file_hash(Path(__file__)),
        audited_files_sha256={name: _file_hash(directory/name) for name in
            ("experiment_config.json", "summary.json", "cases.jsonl", "controls.jsonl", "noise_groups.jsonl", "wls_observations.jsonl")})
    return result


def merge_shards(output, shard_directories):
    """Merge disjoint line shards; repeated operating parents remain two parents.

    Equality of duplicated physical controls, parent records and model hashes
    is mandatory. A noisy observation is identified by case/noise-group/profile,
    so paired healthy observations at different lines remain distinct windows.
    Original shard artifacts are read only.
    """
    output = Path(output).resolve()
    directories = [Path(path).resolve() for path in shard_directories]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Distinct completed shard directories are required")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("experiment_config.json", "summary.json", "cases.jsonl", "controls.jsonl",
                 "wls_observations.jsonl", "noise_groups.jsonl", "parent_provenance.json"):
        if (output/name).exists():
            raise FileExistsError(f"Refusing to replace an existing merged artifact: {output/name}")
    cases, controls, observations, groups, parents, parent_models = {}, {}, {}, {}, {}, {}
    configs, receipts, selected = [], [], []

    def unique(mapping, key, value, *, allow_identical=False):
        if key in mapping:
            if allow_identical and mapping[key] == value:
                return
            raise ValueError(f"Duplicate or inconsistent shard record: {key}")
        mapping[key] = value

    for directory in directories:
        config = json.loads((directory/"experiment_config.json").read_text())
        receipt = json.loads((directory/"summary.json").read_text())
        if not receipt["complete"]:
            raise ValueError(f"Incomplete physical population in shard {directory}")
        comparable = {key: value for key, value in config.items()
                      if key not in ("created_utc", "selected_branch_rows0", "expected_physical_hif_cases")}
        if configs:
            first = {key: value for key, value in configs[0].items()
                     if key not in ("created_utc", "selected_branch_rows0", "expected_physical_hif_cases")}
            if comparable != first:
                raise ValueError("Shard configuration/noise/source mismatch")
        if set(selected) & set(config["selected_branch_rows0"]):
            raise ValueError("Shard branch rows must be disjoint")
        selected.extend(config["selected_branch_rows0"])
        configs.append(config)
        receipts.append(receipt)
        for row in _read_rows(directory/"cases.jsonl"):
            if row["branch_row0"] not in config["selected_branch_rows0"]:
                raise ValueError("Physical case escapes its declared branch shard")
            unique(cases, row["case_id"], row)
        for row in _read_rows(directory/"controls.jsonl"):
            unique(controls, row["case_id"], row, allow_identical=row["kind"] == "healthy")
        for row in _read_rows(directory/"noise_groups.jsonl"):
            unique(groups, row["noise_group_id"], row)
        for row in _read_rows(directory/"wls_observations.jsonl"):
            unique(observations, (row["case_id"], row["noise_group_id"], row["noise_profile"]), row)
        for parent in receipt["parents"]:
            parent_id = parent["parent_id"]
            unique(parents, parent_id, parent, allow_identical=True)
            model = directory/"parents"/parent_id
            hashes = {name: _file_hash(model/name) for name in
                ("source_case.json", "positive_sequence_reference.json", "asset_registry.json", "assumptions.json")
                if (model/name).is_file()}
            if parent["success"] and len(hashes) != 4:
                raise ValueError(f"Successful parent model provenance missing: {model}")
            if parent_id not in parent_models:
                parent_models[parent_id] = {"files_sha256": hashes, "shard_model_directories": []}
            elif hashes != parent_models[parent_id]["files_sha256"]:
                raise ValueError(f"Duplicated parent physical model differs: {parent_id}")
            parent_models[parent_id]["shard_model_directories"].append(str(model))
    config = deepcopy(configs[0])
    config["selected_branch_rows0"] = sorted(selected)
    expected = len(selected) * len(config["load_scales"]) * len(config["phases"]) * len(config["resistances_ohm"])
    config["expected_physical_hif_cases"] = expected
    config["parallel"] = {"shard_count": len(directories), "shards": [str(path) for path in directories],
        "partition": "disjoint_branch_rows_each_with_identical_ordered_load_parents_and_seed",
        "duplicate_parent_policy": "require_identical_records_and_physical_model_hashes_then_count_once"}
    coverage = validate_population_coverage(config, cases, controls, groups, observations)
    if len(cases) != expected:
        raise ValueError(f"Merged physical population incomplete: {len(cases)} versus {expected}")
    if len(parents) != len(config["load_scales"]):
        raise ValueError("Merged parent identities differ from the declared operating parents")
    expected_groups = len(selected) * len(config["load_scales"]) * len(config["phases"]) * config["noise_replicates"]
    if len(groups) != expected_groups:
        raise ValueError("Merged standardized-noise groups are incomplete")
    means = {key: row.get("mean_measurement_vector") for key, row in {**cases, **controls}.items()}
    for row in observations.values():
        if row["case_id"] not in means or means[row["case_id"]] is None or row["noise_group_id"] not in groups:
            raise ValueError("Merged observation has no available physical mean/noise group")
        # Bind the merged record to exactly the tested noisy observation.
        observed = sweep.np.asarray(means[row["case_id"]]) + sweep.np.asarray(groups[row["noise_group_id"]]["unit_noise"]) * sweep.np.asarray(config["noise"]["sigma_z"][row["noise_profile"]])
        if sweep.numeric_hash(observed) != row["observed_sha256"]:
            raise ValueError("Merged observation hash disagrees with its paired physical mean/noise")
    case_rows, obs_rows, parent_rows, control_rows = list(cases.values()), list(observations.values()), list(parents.values()), list(controls.values())
    summary, aggregates = sweep.summarize(case_rows, obs_rows, parent_rows,
        expected_physical_cases=expected, noise_replicates=config["noise_replicates"], contract=config["contract"])
    for alpha in config["wls"].get("comparison_chi_square_alphas", []):
        key = f"{alpha:g}"
        compared = [{**row, "wls": {**row["wls"], **sweep.rethreshold_wls(row["wls"], alpha)}} for row in obs_rows]
        _, counts = sweep.summarize(case_rows, compared, parent_rows,
            expected_physical_cases=expected, noise_replicates=config["noise_replicates"], contract=config["contract"])
        summary.setdefault("chi_square_alpha_comparisons", {})[key] = counts
        suffix = "alpha_" + key.replace(".", "p")
        sweep._write_csv(output/f"detection_by_voltage_resistance_{suffix}.csv", [row for row in counts if row["kind"] == "hif"])
        sweep._write_csv(output/f"controls_summary_{suffix}.csv", [row for row in counts if row["kind"] != "hif"])
    summary.update(parents=parent_rows, operating_parent_count=len(parents), independent_noise_group_count=len(groups),
        population_coverage_validation=coverage,
        healthy_physical_controls=sum(row["kind"] == "healthy" for row in control_rows),
        no_fault_split_physical_controls=sum(row["kind"] == "no_fault_split" for row in control_rows),
        failed_physical_controls=[row["case_id"] for row in control_rows if not row["physical_success"]],
        implementation_unchanged_during_run=all(row["implementation_unchanged_during_run"] for row in receipts),
        shard_count=len(directories), shard_wall_seconds=[row["wall_seconds"] for row in receipts],
        provenance="parent_provenance.json and original shard experiment_config/implementation_snapshot")
    sweep.write_json(output/"experiment_config.json", config)
    sweep.write_json(output/"parent_provenance.json", parent_models)
    for name, rows in (("cases", case_rows), ("controls", control_rows), ("wls_observations", obs_rows), ("noise_groups", list(groups.values()))):
        _write_rows(output/f"{name}.jsonl", rows)
    sweep._write_csv(output/"detection_by_voltage_resistance.csv", [row for row in aggregates if row["kind"] == "hif"])
    sweep._write_csv(output/"controls_summary.csv", [row for row in aggregates if row["kind"] != "hif"])
    sweep._write_csv(output/"physical_cases.csv", [{key: value for key, value in row.items()
        if key not in ("physical_audit", "split_initialization", "pu_equivalence", "mean_measurement_vector")} for row in case_rows])
    sweep.write_json(output/"summary.json", summary)
    return summary


def _run_shard(job):
    return sweep.run_sweep(job["output"], **job["settings"])


def run_parallel_sweep(output, *, workers=8, **settings):
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"A fresh output directory is required: {output}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    settings.setdefault("system", "case57")
    settings.setdefault("seed", 20260919)
    if workers == 1:
        return sweep.run_sweep(output, **settings)
    _, _, eligible = sweep.configure_system(settings["system"], settings.get("voltage_profile"))
    selected = settings.get("branch_rows")
    selected = eligible if selected is None else list(selected)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(eligible):
        raise ValueError("Distinct eligible branch rows are required")
    partitions = [part.tolist() for part in sweep.np.array_split(selected, min(workers, len(selected)))]
    jobs = [{"output": str(output/"shards"/f"shard_{index:02d}"),
             "settings": {**settings, "branch_rows": partition}} for index, partition in enumerate(partitions)]
    output.mkdir(parents=True)
    sweep.write_json(output/"parallel_plan.json", {"workers": len(jobs), "blas_threads_per_worker": 1, "jobs": jobs})
    print(f"Starting {len(jobs)} disjoint branch shards; each retains the same parent order and seed", flush=True)
    failures = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(_run_shard, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                print(sweep._json({"shard_complete": job["output"], "physical_cases": result["recorded_physical_hif_cases"],
                    "physical_failed": result["physical_hif_failed"], "wls_failed": result["fault_wls_failures"],
                    "elapsed_seconds": time.perf_counter()-started}), flush=True)
            except Exception as exc:
                failures.append({"shard": job["output"], "error": f"{type(exc).__name__}: {exc}"})
    if failures:
        sweep.write_json(output/"parallel_run_receipt.json", {"complete": False, "failures": failures,
            "wall_seconds": time.perf_counter()-started, "original_shard_artifacts_preserved": True})
        raise RuntimeError(f"{len(failures)} shard processes failed; partial artifacts were preserved")
    try:
        summary = merge_shards(output, [job["output"] for job in jobs])
    except Exception as exc:
        sweep.write_json(output/"parallel_run_receipt.json", {"complete": False, "merge_error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter()-started, "original_shard_artifacts_preserved": True})
        raise
    summary["wall_seconds"] = time.perf_counter()-started
    sweep.write_json(output/"summary.json", summary)
    sweep.write_json(output/"parallel_run_receipt.json", {"complete": summary["complete"], "workers": len(jobs),
        "wall_seconds": summary["wall_seconds"], "operating_parent_count": summary["operating_parent_count"],
        "physical_cases": summary["recorded_physical_hif_cases"], "noisy_observations": summary["all_noisy_observations"],
        "original_shard_artifacts_preserved": True})
    return summary


def main(argv=None, *, default_system="case57", default_seed=20260919):
    parser = sweep.argument_parser(default_system=default_system, default_seed=default_seed)
    parser.add_argument("--workers", type=int, default=8)
    args = vars(parser.parse_args(argv))
    output, workers = args.pop("output_dir"), args.pop("workers")
    args["comparison_alphas"] = args.pop("comparison_chi_square_alphas")
    if args.get("generator_control") is None:
        args.pop("generator_control", None)
    summary = run_parallel_sweep(output, workers=workers, **args)
    return 0 if (summary["complete"] and not summary["physical_hif_failed"]
        and not summary["failed_physical_controls"] and not summary["fault_wls_failures"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
