"""Generate independent practical-corpus shards and merge their audit ledgers.

Each worker owns a distinct directory, seed, operating parents and DSS process.
Merged manifests retain all parent/window IDs and use paths relative to their
new root. Original physical audit bytes remain intact, preserving provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import copy
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import time
from typing import Any

import numpy as np

from .dataset import file_sha256, write_json

SPLITS = ("train", "validation", "calibration", "test")


def shard_plan(parents_by_split: dict[str, int], *, seed: int, workers: int) -> list[dict[str, Any]]:
    """Distribute each split deterministically without dividing any parent."""
    if set(parents_by_split) - set(SPLITS) or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                                                  for n in parents_by_split.values()):
        raise ValueError("parents_by_split requires nonnegative integer counts for known splits")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    total = sum(parents_by_split.values())
    if not total:
        raise ValueError("at least one physical parent is required")
    count = min(workers, total)
    children = np.random.SeedSequence(seed).spawn(count)
    plans = [{"shard_index": index, "seed": int(child.generate_state(1, dtype=np.uint32)[0]),
              "seed_sequence_spawn_key": list(child.spawn_key),
              "parents_by_split": {split: parents_by_split.get(split, 0) // count +
                  int(index < parents_by_split.get(split, 0) % count) for split in SPLITS}}
             for index, child in enumerate(children)]
    if len({plan["seed"] for plan in plans}) != count:
        raise RuntimeError("Child seed collision")
    return plans


def _worker_init() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    global _native_threads
    _native_threads = threadpool_limits(limits=1)


def _generate_shard(job: dict[str, Any]) -> dict[str, Any]:
    log_path = Path(job["log_path"])
    with log_path.open("w", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        from .practical_corpus import generate_corpus
        return generate_corpus(job["output_dir"], **job["arguments"])


def _prefixed_path(value: str, prefix: str) -> str:
    # Absolute externally supplied references stay absolute. Generated shard
    # references are relative and may not escape the shard directory.
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return value
    path = PurePosixPath(value.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError(f"Shard-relative artifact reference escapes its source: {value}")
    return (PurePosixPath(prefix) / path).as_posix()


def _rewrite_metadata_paths(value: Any, prefix: str) -> Any:
    if isinstance(value, list):
        return [_rewrite_metadata_paths(item, prefix) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key.endswith("_path") and isinstance(item, str):
            result[key] = _prefixed_path(item, prefix)
        elif key.endswith("_paths") and isinstance(item, list) and all(isinstance(path, str) for path in item):
            result[key] = [_prefixed_path(path, prefix) for path in item]
        else:
            result[key] = _rewrite_metadata_paths(item, prefix)
    return result


def rewrite_manifest_row(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Change artifact references only; numeric inputs, labels and seeds stay exact."""
    result = copy.deepcopy(row)
    for field in ("case", "z", "measurement_sigma"):
        if isinstance(result.get(field), str):
            result[field] = _prefixed_path(result[field], prefix)
    if "offline_metadata" in result:
        result["offline_metadata"] = _rewrite_metadata_paths(result["offline_metadata"], prefix)
        # Referenced audit files retain their own shard-relative references.
        # Keeping their bytes untouched also keeps full_provenance_hash valid.
        result["offline_metadata"]["source_shard_root"] = prefix
    return result


def merge_shards(output_dir: str | Path, reports: list[dict[str, Any]], *,
                 plan: list[dict[str, Any]], seed: int, elapsed_seconds: float) -> dict[str, Any]:
    """Merge in shard-index order, including every rejection and physical failure."""
    out = Path(output_dir).resolve()
    if len(reports) != len(plan) or not reports:
        raise ValueError("One completed report is required for every shard")
    common = reports[0]
    invariant_keys = ("policy_version", "implementation_sha256", "noise_replicates",
        "healthy_replicates_by_split", "attempt_cap_per_slot", "main_minimum_paired_distance", "profiles")
    profile_keys = ("scenario_profile", "noise_profile", "curriculum_stage", "profile_identity", "profile_identity_sha256")
    auxiliary_names = {cohort: Path(info["path"]).name for cohort, info in common.get("auxiliary_manifests", {}).items()}
    for report in reports[1:]:
        for key in invariant_keys:
            if report[key] != common[key]:
                raise ValueError(f"Cannot merge shards with different {key}")
        for key in profile_keys:
            if report.get(key) != common.get(key):
                raise ValueError(f"Cannot merge shards with different {key}")
        if {cohort: Path(info["path"]).name for cohort, info in report.get("auxiliary_manifests", {}).items()} != auxiliary_names:
            raise ValueError("Cannot merge shards with different auxiliary cohort manifests")
    counts, merged_counts = Counter(), Counter()
    parent_owner, parent_splits, windows = {}, {}, set()
    source_reports = []
    event_histogram, outcome_histogram, rejection_histogram, failure_histogram = Counter(), Counter(), Counter(), Counter()
    for item, report in zip(plan, reports):
        if report["seed"] != item["seed"] or report["parents_by_split"] != item["parents_by_split"]:
            raise ValueError("Shard report identity does not match its deterministic plan")
        counts.update(report["counts"])
    manifest_specs = [("manifest.jsonl", "main"), ("boundary_manifest.jsonl", "boundary"),
                      *[(filename, cohort) for cohort, filename in auxiliary_names.items()]]
    for filename, scope in manifest_specs:
        with (out / filename).open("w", encoding="utf-8") as destination:
            for item, report in zip(plan, reports):
                index = item["shard_index"]
                prefix = f"shards/shard_{index:03d}"
                source = out / prefix / filename
                with source.open(encoding="utf-8-sig") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        parent, split = row["parent_id"], row["split"]
                        if scope in auxiliary_names and split not in {"validation", "test"}:
                            raise ValueError(f"Evaluation-only auxiliary cohort {scope} cannot contain split={split}")
                        if parent in parent_owner and parent_owner[parent] != index:
                            raise ValueError(f"Parent identity is shared by different shards: {parent}")
                        if parent in parent_splits and parent_splits[parent] != split:
                            raise ValueError(f"Physical parent leaks across splits: {parent}")
                        window = (parent, row["window_id"])
                        if window in windows:
                            raise ValueError(f"Repeated physical measurement window: {window}")
                        parent_owner[parent], parent_splits[parent] = index, split
                        windows.add(window)
                        rewritten = rewrite_manifest_row(row, prefix)
                        destination.write(json.dumps(rewritten, separators=(",", ":"), allow_nan=False) + "\n")
                        merged_counts[f"{scope}:{split}:rows"] += 1
                        merged_counts[f"{scope}:{split}:noise_windows"] += int(row.get("noise_replicates", 1))
                        for family in row["families"] or ["healthy"]:
                            merged_counts[f"{scope}:{split}:{family}"] += 1
    manifest_scopes = {scope for _, scope in manifest_specs}
    receipt_counts = Counter({key: value for key, value in counts.items()
                             if key.split(":", 1)[0] in manifest_scopes})
    if receipt_counts != merged_counts:
        raise ValueError("Merged manifest counts disagree with shard receipts")
    failure_count = ledger_count = 0
    with (out / "proposal_ledger.jsonl").open("w", encoding="utf-8") as ledger, \
         (out / "physical_failures.jsonl").open("w", encoding="utf-8") as failures:
        for item, report in zip(plan, reports):
            prefix = f"shards/shard_{item['shard_index']:03d}"
            with (out / prefix / "proposal_ledger.jsonl").open(encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = _rewrite_metadata_paths(json.loads(line), prefix)
                    text = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                    ledger.write(text)
                    ledger_count += 1
                    event_histogram[record.get("event", "unspecified")] += 1
                    if record.get("outcome"):
                        outcome_histogram[record["outcome"]] += 1
                    rejection_histogram.update(record.get("scenario_policy", {}).get("rejection_reasons", []))
                    if record.get("outcome") == "simulation_failure" or record.get("event") == "parent_source_rejected":
                        failure_count += 1
                        failure_histogram[record.get("reason", "unspecified")] += 1
                        failures.write(text)
            report_path = out / prefix / "generation_report.json"
            source_reports.append({**item, "shard_root": prefix,
                "generation_report": report_path.relative_to(out).as_posix(),
                "generation_report_sha256": file_sha256(report_path),
                "manifest_sha256": report["manifest_sha256"],
                "boundary_manifest_sha256": report["boundary_manifest_sha256"],
                "auxiliary_manifest_sha256": {cohort: info["sha256"] for cohort, info in report.get("auxiliary_manifests", {}).items()},
                "implementation_sha256": report["implementation_sha256"]})
    totals = {split: sum(item["parents_by_split"][split] for item in plan) for split in SPLITS}
    observed_parents = Counter(parent_splits.values())
    if any(observed_parents.get(split, 0) != expected for split, expected in totals.items()):
        raise ValueError("Merged parent counts disagree with the shard plan")
    report = {**{key: copy.deepcopy(common[key]) for key in invariant_keys},
        **{key: copy.deepcopy(common[key]) for key in profile_keys if key in common},
        "schema": "parallel_practical_physical_wls_screen_corpus_v1", "seed": seed,
        "seed_derivation": "numpy.random.SeedSequence(root_seed).spawn(shard_count), one uint32 state per child; uniqueness verified",
        "parents_by_split": totals, "shard_count": len(plan), "source_reports": source_reports,
        "manifest": str(out / "manifest.jsonl"), "boundary_manifest": str(out / "boundary_manifest.jsonl"),
        "proposal_ledger": str(out / "proposal_ledger.jsonl"), "counts": dict(counts),
        "manifest_sha256": file_sha256(out / "manifest.jsonl"),
        "boundary_manifest_sha256": file_sha256(out / "boundary_manifest.jsonl"),
        "proposal_ledger_sha256": file_sha256(out / "proposal_ledger.jsonl"),
        "wrapper_implementation_sha256": file_sha256(__file__),
        "elapsed_seconds": elapsed_seconds,
        "summed_shard_elapsed_seconds": sum(report["elapsed_seconds"] for report in reports),
        "healthy_max_balanced_equation_error_pu": max((report["healthy_max_balanced_equation_error_pu"]
            for report in reports if report["healthy_max_balanced_equation_error_pu"] is not None), default=None),
        "intended_main_slots_per_noncalibration_parent": common["intended_main_slots_per_noncalibration_parent"],
        "proposal_audit": {"records": ledger_count, "event_histogram": dict(event_histogram),
            "outcome_histogram": dict(outcome_histogram), "rejection_reason_histogram": dict(rejection_histogram)},
        "physical_failures": {"count": failure_count, "reason_histogram": dict(failure_histogram),
            "ledger": str(out / "physical_failures.jsonl")},
        "admission_uses_wls_alarm": any(report["admission_uses_wls_alarm"] for report in reports),
        "admission_uses_noisy_or_learned_scores": any(report["admission_uses_noisy_or_learned_scores"] for report in reports),
        "reported_model_stays_identical_within_parent": all(report["reported_model_stays_identical_within_parent"] for report in reports),
        "physical_audit_path_base": "physical_audit_path is relative to merged root; internal references inside unchanged audit files use offline_metadata.source_shard_root",
        "limitations": common["limitations"]}
    if auxiliary_names:
        report["schema"] = "parallel_practical_physical_wls_screen_corpus_v2"
        report["auxiliary_manifests"] = {cohort: {"path": str(out / filename),
            "sha256": file_sha256(out / filename), "training_eligible": False,
            "parent_splits": ["validation", "test"],
            "row_count": sum(counts.get(f"{cohort}:{split}:rows", 0) for split in SPLITS),
            "rows_by_split": {split: counts.get(f"{cohort}:{split}:rows", 0) for split in SPLITS}}
            for cohort, filename in auxiliary_names.items()}
        report["positive_boundary_storage"] = common.get("positive_boundary_storage")
        report["exact_wls_audit_used_for_admission"] = False
    write_json(out / "generation_report.json", report)
    return report


def generate_parallel(output_dir: str | Path, *, parents_by_split: dict[str, int], seed: int = 20260918,
                      workers: int = 4, noise_replicates: int = 2,
                      healthy_calibration_replicates: int = 80,
                      healthy_replicates_by_split: dict[str, int] | None = None,
                      attempt_cap: int = 24, scenario_profile: str = "legacy_v1",
                      noise_profile: str = "baseline", stage: str = "full") -> dict[str, Any]:
    if scenario_profile not in {"legacy_v1", "reviewed_v1", "ieee14_physical_hif_v1"} or noise_profile not in {"baseline", "accuracy_005", "accuracy_002"}:
        raise ValueError("Unsupported scenario_profile or noise_profile")
    if stage not in {"early", "full"} or (scenario_profile == "legacy_v1" and stage != "full"):
        raise ValueError("early curriculum requires reviewed_v1; stage must be early or full")
    plan = shard_plan(parents_by_split, seed=seed, workers=workers)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "shards").mkdir()
    write_json(out / "shard_plan.json", {"seed": seed, "workers": workers, "shards": plan,
        "scenario_profile": scenario_profile, "noise_profile": noise_profile, "stage": stage})
    jobs = [{"output_dir": str(out / "shards" / f"shard_{item['shard_index']:03d}"),
             "log_path": str(out / "shards" / f"shard_{item['shard_index']:03d}.log"),
             "arguments": {"parents_by_split": item["parents_by_split"], "seed": item["seed"],
                 "noise_replicates": noise_replicates,
                 "healthy_calibration_replicates": healthy_calibration_replicates,
                 "healthy_replicates_by_split": healthy_replicates_by_split, "attempt_cap": attempt_cap,
                 "scenario_profile": scenario_profile, "noise_profile": noise_profile, "stage": stage}}
            for item in plan]
    started = time.monotonic()
    reports: list[Any] = [None] * len(plan)
    with ProcessPoolExecutor(max_workers=min(workers, len(plan)), initializer=_worker_init) as executor:
        pending = {executor.submit(_generate_shard, job): item["shard_index"] for item, job in zip(plan, jobs)}
        for future in as_completed(pending):
            index = pending[future]
            reports[index] = future.result()
            print(json.dumps({"event": "shard_complete", "shard_index": index,
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    return merge_shards(out, reports, plan=plan, seed=seed, elapsed_seconds=time.monotonic() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    for split, default in zip(SPLITS, (384, 96, 128, 128)):
        parser.add_argument(f"--{split}-parents", type=int, default=default)
    parser.add_argument("--noise-replicates", type=int, default=2)
    parser.add_argument("--healthy-calibration-replicates", type=int, default=80)
    parser.add_argument("--healthy-validation-replicates", type=int)
    parser.add_argument("--healthy-test-replicates", type=int)
    parser.add_argument("--attempt-cap", type=int, default=24)
    parser.add_argument("--scenario-profile", choices=("legacy_v1", "reviewed_v1", "ieee14_physical_hif_v1"), default="legacy_v1")
    parser.add_argument("--noise-profile", choices=("baseline", "accuracy_005", "accuracy_002"), default="baseline")
    parser.add_argument("--stage", choices=("early", "full"), default="full")
    args = parser.parse_args(argv)
    report = generate_parallel(args.output_dir, parents_by_split={s: getattr(args, f"{s}_parents") for s in SPLITS},
        workers=args.workers, seed=args.seed, noise_replicates=args.noise_replicates,
        healthy_calibration_replicates=args.healthy_calibration_replicates, attempt_cap=args.attempt_cap,
        healthy_replicates_by_split={s: n for s, n in (("validation", args.healthy_validation_replicates),
            ("test", args.healthy_test_replicates)) if n is not None},
        scenario_profile=args.scenario_profile, noise_profile=args.noise_profile, stage=args.stage)
    print(json.dumps({"manifest": report["manifest"], "counts": report["counts"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
