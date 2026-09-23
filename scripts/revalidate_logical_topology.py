"""Re-audit an immutable logical-topology corpus under the current comparison guard.

This script never generates physical worlds or measurements. Verified old
numerical fits may be reused; every candidate comparison/certificate is rebuilt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from logical_topology.fit_cache import VerifiedEstimatorCache, VerifiedSourceRun, file_sha256
from logical_topology.scenarios import write_json
from scripts.validate_logical_topology import SOURCE_FILES, summarize


INPUT_DIRECTORIES = ("inventories", "sensors", "observations", "model_inputs", "physical_audit")
INPUT_FILES = ("manifest.json", "config.json")
REVALIDATION_SOURCES = list(dict.fromkeys([*SOURCE_FILES, "scripts/revalidate_logical_topology.py",
                                         "logical_topology/fit_cache.py", "logical_topology/calibration.py"]))


def copy_corpus_inputs(source_run, output_dir) -> dict:
    """Copy only input artifacts, checking bytes without touching the old run."""
    source = Path(source_run).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("The verified output must be separate from the source run")
    if output.exists():
        raise FileExistsError(f"Verified output already exists: {output}")
    source_corpus = source / "corpus"
    files = []
    for directory in INPUT_DIRECTORIES:
        folder = source_corpus / directory
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Required source input directory is missing or linked: {directory}")
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError("Input copying does not follow filesystem links")
            if path.is_file():
                files.append(path)
    files.extend(source_corpus / name for name in INPUT_FILES)
    hashes = {str(path.relative_to(source_corpus)).replace("\\", "/"): file_sha256(path) for path in sorted(files)}
    (output / "corpus").mkdir(parents=True)
    for directory in INPUT_DIRECTORIES:
        shutil.copytree(source_corpus / directory, output / "corpus" / directory)
    for name in INPUT_FILES:
        shutil.copyfile(source_corpus / name, output / "corpus" / name)
    for relative, digest in hashes.items():
        if file_sha256(output / "corpus" / relative) != digest or file_sha256(source_corpus / relative) != digest:
            raise RuntimeError(f"Corpus input changed during verified copying: {relative}")
    receipt = {"contract": "immutable_logical_corpus_copy_v1", "source_run": str(source),
               "input_file_count": len(hashes), "files_sha256": hashes,
               "old_audits_copied": False, "source_bytes_preserved": True, "copied_bytes_match": True}
    write_json(output / "input_copy_manifest.json", receipt)
    return receipt


@lru_cache(maxsize=2)
def _source_run(path):
    return VerifiedSourceRun(path)


def revalidate_batch(job):
    from threadpoolctl import threadpool_limits
    from logical_topology.audit import evaluate_scenario
    if "estimator" not in inspect.signature(evaluate_scenario).parameters:
        raise RuntimeError("Revalidation requires the audited optional-estimator integration; frozen core was not edited")
    source = _source_run(job["source_run"])
    output = Path(job["output_dir"])
    results = []
    with threadpool_limits(limits=1):
        for row in job["rows"]:
            started, cache = time.perf_counter(), None
            try:
                cache = VerifiedEstimatorCache(source, row) if row["physical_admission"]["admitted"] else None
                report = evaluate_scenario(output / "corpus", row, max_pairs=job["max_pairs"],
                                           scan_pairs=True, estimator=cache)
            except Exception as exc:
                import traceback
                report = {"scenario_id": row["scenario_id"], "physical_admitted": row["physical_admission"]["admitted"],
                          "audit_execution_failure": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            if cache is not None:
                provenance = cache.receipt()
                path = output / "corpus" / "fit_reuse" / f"{row['scenario_id']}.json"
                write_json(path, provenance)
                report["numeric_fit_provenance"] = {
                    **{key: provenance[key] for key in ("indexed_exact_inputs", "lookups", "reused_fits",
                                                        "fresh_estimator_calls", "fresh_numerical_solves", "numerical_sources_match")},
                    "path": str(path.relative_to(output)).replace("\\", "/"),
                    "sha256": file_sha256(path), "old_decisions_or_certificates_reused": False,
                }
            report.update(family=row["family"], direction=row["direction"], layout=row["layout"],
                          measurement_profile=row["measurement_profile"], parent_physical_root=row["parent_physical_root"],
                          load_scale=row["load_scale"], elapsed_seconds=time.perf_counter()-started)
            write_json(output / "corpus" / "row_audits" / f"{row['scenario_id']}.json", report)
            results.append(report)
    return results


def _write_report(output, summary, provenance):
    total = summary["overall"]
    lines = ["# Revalidation of the immutable logical-topology corpus", "",
             f"This run re-audited {total['planned']} retained scenario/deployment rows from the unchanged source corpus; {total['physically_admitted']} were physically admitted. It did not regenerate measurements, redispatch generation, or rerun physical OPF/PF.", "",
             f"Numerical work: **{provenance['reused_fits']} verified historical fit returns**, **{provenance['fresh_estimator_calls']} new estimator calls**, and **{provenance['fresh_numerical_solves']} new numerical solves**. Historical fit reuse required exact semantic input keys and matching numerical-source hashes. Old candidate decisions and certificates were not reused.", "",
             "Candidate comparisons and correction certificates were rebuilt with the current post-selection separation guard. Absolute chi-square/normalized-residual plausibility alone is insufficient for applying a topology change. The predeclared false-rejection budget is shared across rival models and both testing routes. The common-relaxation comparison is asymptotic and assumes regular Gaussian models; this is not an empirical or exact finite-sample false-correction guarantee.", "",
             "| Family/profile | Planned | Admitted | Exact final statuses | False CB corrections |",
             "|---|---:|---:|---:|---:|"]
    for name, group in summary["groups"].items():
        lines.append(f"| {name} | {group['planned']} | {group['physically_admitted']} | {group['exact_status_recovery']} | {group['false_corrections']} |")
    lines += ["", "Exact final statuses include correctly retained healthy configurations. They do not claim recovery of analog or parameter overlays, or learned-policy performance.", "",
              f"Audit execution failures: {total['audit_execution_failures']}. Healthy-CB preservation failures: {total['healthy_cb_preservation_failures']}. Fixed-evidence or parameter preservation failures: {total['fixed_evidence_or_parameter_preservation_failures']}.", "",
              "[Manifest](corpus/manifest.json) · [Summary](summary.json) · [Input copy hashes](input_copy_manifest.json) · [Run/source receipt](run_receipt.json)", "",
              "Per-row compact audits are in `corpus/row_audits`; fresh comparison scans/certificates are in `corpus/audits`; offline numerical reuse receipts are in `corpus/fit_reuse`. The original run remains unchanged.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.max_pairs < 0:
        parser.error("workers must be 1..4 and max-pairs must be nonnegative")
    from logical_topology.audit import evaluate_scenario
    if "estimator" not in inspect.signature(evaluate_scenario).parameters:
        raise RuntimeError("Optional estimator integration is not installed; do not edit the still-frozen core")
    source = VerifiedSourceRun(args.source_run)
    output = args.output_dir.resolve()
    source_before = {name: file_sha256(REPO / name) for name in REVALIDATION_SOURCES}
    copy_receipt = copy_corpus_inputs(source.root, output)
    for name, digest in source_before.items():
        destination = output / "implementation_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
        if file_sha256(destination) != digest:
            raise RuntimeError(f"Implementation changed while snapshotting {name}")
    config = {"contract": "fixed_corpus_comparison_guard_revalidation_v1", "source_run": str(source.root),
              "created_utc": datetime.now(timezone.utc).isoformat(), "workers": args.workers, "max_pairs": args.max_pairs,
              "physical_worlds_or_measurements_regenerated": False, "cache_uses_old_decisions": False,
              "cache_numerical_sources_match_at_start": source.sources_match_now()}
    write_json(output / "run_config.json", config)
    rows = source.manifest["rows"]
    jobs = [{"source_run": str(source.root), "output_dir": str(output), "max_pairs": args.max_pairs,
             "rows": rows[index:index+4]} for index in range(0, len(rows), 4)]
    results = []
    if args.workers == 1:
        for job in jobs:
            results.extend(revalidate_batch(job))
            print(f"Revalidated {len(results)}/{len(rows)} rows", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(revalidate_batch, job) for job in jobs]
            for future in as_completed(futures):
                results.extend(future.result())
                print(f"Revalidated {len(results)}/{len(rows)} rows", flush=True)
    results.sort(key=lambda row: row["scenario_id"])
    summary = summarize(source.manifest, results)
    provenance = {key: sum((row.get("numeric_fit_provenance") or {}).get(key, 0) for row in results)
                  for key in ("lookups", "reused_fits", "fresh_estimator_calls", "fresh_numerical_solves")}
    summary.update(revalidation=config, numerical_fit_provenance=provenance,
                   statistical_scope="absolute fit plus current finite-family post-selection separation guard; asymptotic common-relaxation assumptions apply")
    write_json(output / "audit_results.json", results)
    write_json(output / "summary.json", summary)
    source_after = {name: file_sha256(REPO / name) for name in REVALIDATION_SOURCES}
    old_unchanged = all(file_sha256(source.corpus / name) == digest for name, digest in copy_receipt["files_sha256"].items())
    copied_unchanged = all(file_sha256(output / "corpus" / name) == digest for name, digest in copy_receipt["files_sha256"].items())
    receipt = {"config": config, "source_before": source_before, "source_after": source_after,
               "numerical_versions": source.current_versions,
               "cache_historical_version_attestation": source.version_attestation,
               "all_sources_unchanged_during_run": source_before == source_after,
               "all_rows_audited": summary["all_rows_audited"], "source_corpus_unchanged": old_unchanged,
               "copied_corpus_unchanged": copied_unchanged, "input_copy_manifest_sha256": file_sha256(output / "input_copy_manifest.json"),
               "manifest_sha256": file_sha256(output / "corpus" / "manifest.json"),
               "summary_sha256": file_sha256(output / "summary.json"), "numerical_fit_provenance": provenance}
    write_json(output / "run_receipt.json", receipt)
    _write_report(output, summary, provenance)
    print(json.dumps({"overall": summary["overall"], "numerical_fit_provenance": provenance}, indent=2), flush=True)
    return 0 if (summary["all_rows_audited"] and not summary["overall"]["audit_execution_failures"]
                 and old_unchanged and copied_unchanged and source_before == source_after) else 2


if __name__ == "__main__":
    raise SystemExit(main())
