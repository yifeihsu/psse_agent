"""Build and audit logical IEEE topology-error scenarios without teacher filtering."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from logical_topology.scenarios import build_corpus, write_json
from logical_topology.fit_cache import numerical_versions


SOURCE_FILES = [
    "scripts/validate_logical_topology.py", "logical_topology/__init__.py", "logical_topology/inventory.py",
    "logical_topology/measurements.py", "logical_topology/estimation.py",
    "logical_topology/runtime.py", "logical_topology/calibration.py", "logical_topology/fit_cache.py",
    "logical_topology/scenarios.py", "logical_topology/audit.py",
    "psse_env/systems/registry.py", "Transmission/generate_measurements.py",
    "mcp_server/case14.m", "mcp_server/case57.m", "mcp_server/case118.m",
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_batch(job):
    from threadpoolctl import threadpool_limits
    from logical_topology.audit import evaluate_scenario
    with threadpool_limits(limits=1):
        result = []
        for row in job["rows"]:
            started = time.perf_counter()
            try:
                audit = evaluate_scenario(job["corpus_dir"], row, max_pairs=job["max_pairs"], scan_pairs=True)
            except Exception as exc:
                # Execution failure is recorded, never converted into a clean
                # topology or removed from the physically valid population.
                import traceback
                audit = {"scenario_id": row["scenario_id"], "physical_admitted": row["physical_admission"]["admitted"],
                         "audit_execution_failure": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            audit.update(scenario_id=row["scenario_id"], family=row["family"], direction=row["direction"],
                         layout=row["layout"], measurement_profile=row["measurement_profile"],
                         parent_physical_root=row["parent_physical_root"], load_scale=row["load_scale"],
                         elapsed_seconds=time.perf_counter()-started)
            write_json(Path(job["corpus_dir"])/"row_audits"/f"{row['scenario_id']}.json", audit)
            result.append(audit)
        return result


def group_summary(rows):
    admitted = [row for row in rows if row.get("physical_admitted")]
    wls = [row.get("initial_estimation") or {} for row in admitted]
    return {"planned": len(rows), "physically_admitted": len(admitted),
            "audit_execution_failures": sum("audit_execution_failure" in row for row in rows),
            "initial_wls_converged": sum(result.get("converged") is True for result in wls),
            "initial_chi_square_alarms": sum(result.get("chi_square_alarm") is True for result in wls),
            "initial_normalized_residual_alarms": sum(result.get("normalized_residual_alarm") is True for result in wls),
            "initial_normalized_residual_only_alarms": sum(result.get("normalized_residual_alarm") is True
                and result.get("chi_square_alarm") is False for result in wls),
            "runtime_decisions": dict(Counter(row.get("runtime_decision", "not_executed") for row in admitted)),
            "corrections_applied": sum(bool(row.get("correction_applied")) for row in admitted),
            "exact_status_recovery": sum((row.get("status_audit") or {}).get("exact_status_recovery") is True for row in admitted),
            "false_corrections": sum((row.get("status_audit") or {}).get("false_correction_count", 0) for row in admitted),
            "healthy_cb_preservation_failures": sum((row.get("status_audit") or {}).get("healthy_cb_preserved") is False for row in admitted),
            "fixed_evidence_or_parameter_preservation_failures": sum((row.get("preservation") or {}).get("passed") is False for row in admitted)}


def summarize(manifest, rows):
    family_profile = defaultdict(list)
    for row in rows:
        family_profile[(row["family"], row["measurement_profile"])].append(row)
    split_counts = Counter(row["split"] for row in manifest["rows"])
    structural_counts = Counter(row["structural_split"] for row in manifest["rows"])
    family_splits = defaultdict(Counter)
    for row in manifest["rows"]:
        if row["physical_admission"]["admitted"]:
            family_splits[row["structural_split"]][row["family"]] += 1
    by_world = defaultdict(set)
    by_structural_world = defaultdict(set)
    for row in manifest["rows"]:
        by_world[row["parent_physical_root"]].add(row["split"])
        by_structural_world[row["parent_physical_root"]].add(row["structural_split"])
    split_safe = all(len(values) == 1 for values in [*by_world.values(), *by_structural_world.values()])
    expected = {row["scenario_id"] for row in manifest["rows"]}
    actual = {row["scenario_id"] for row in rows}
    return {"contract": "logical_topology_error_audit_summary_v1", "all_rows_audited": len(rows) == len(expected) and actual == expected,
            "overall": group_summary(rows), "groups": {f"{family}/{profile}": group_summary(group)
                for (family, profile), group in sorted(family_profile.items())},
            "physical_world_count": manifest["physical_world_count"],
            "physical_worlds_admitted": manifest["physical_worlds_admitted"],
            "physical_rejection_reasons": dict(Counter(row["physical_admission"].get("reason")
                for row in manifest["rows"] if not row["physical_admission"]["admitted"])),
            "default_split_counts": dict(split_counts), "structural_split_counts": dict(structural_counts),
            "structural_split_family_coverage": {key: dict(value) for key, value in family_splits.items()},
            "parent_operating_roots_do_not_cross_splits": split_safe,
            "teacher_or_wls_success_used_to_filter_physical_corpus": False,
            "status_identification_scope": "current model, all single-CB alternatives, plus declared complete or explicitly budget-limited two-CB alternatives",
            "statistical_scope": "rank-aware Gaussian WLS goodness of fit and normalized residuals plus a finite-family comparison guard with shared method budgets; conditional asymptotic separation within enumerated hypotheses, not an exact nonlinear false-correction guarantee",
            "training_readiness": "not_certified; review family coverage and collect distinct operating worlds where structural holdout reserves shared parents"}


def write_report(output, manifest, summary, config):
    total = summary["overall"]
    lines = ["# Logical topology-error implementation and executed audit", "",
        f"System: **{config['system']}**. The physical source uses AC OPF followed by AC PF and bounds/equation checks. {total['planned']} scenario/deployment rows were planned; {total['physically_admitted']} were physically admitted from {summary['physical_worlds_admitted']}/{summary['physical_world_count']} true operating worlds.", "",
        "## Implemented electrical semantics", "",
        "- Each canonical branch has one logical asset-status CB. Status zero removes its complete two-terminal admittance, including charging. Parallel circuits retain distinct immutable identities.",
        "- Each eligible synthetic substation has two frozen sections and one ideal coupler. Closed switches contract sections; open switches keep separate buses. No tiny impedance, unrelated-bus merge, automatic extra slack, or hidden load shedding is used.",
        "- Inclusion/exclusion and split/merging directions refer to true versus reported status. Correctly open branches and couplers are healthy controls.",
        "- Raw section injections are obtained from their assigned physical generators and loads. Closed-coupler P/Q flows are fitted nuisance variables, not additional sensors. Open-coupler flows are exactly zero. Candidate fits use the same physical sensor identities, covariance, and available observations.",
        "- Indirect even/odd deployments mask half the branches before targets are selected. Unavailable values are None at the operator boundary. Shared available channels use matched noise draws across profiles.", "",
        "## Executed outcomes", "", "| Family/profile | Planned | Physical admission | Exact final statuses | False CB corrections |", "|---|---:|---:|---:|---:|"]
    for key, group in summary["groups"].items():
        lines.append(f"| {key} | {group['planned']} | {group['physically_admitted']} | {group['exact_status_recovery']} | {group['false_corrections']} |")
    lines += ["", "Exact final statuses include healthy configurations correctly retained. They do not imply that measurement/parameter overlays were repaired. Ambiguous, weak, unobservable, unknown-status and wrong-model numerical-failure outcomes remain in the detailed audits.", "",
        f"Audit execution failures: {total['audit_execution_failures']}. Fixed observation/covariance/parameter preservation failures: {total['fixed_evidence_or_parameter_preservation_failures']}. Healthy-CB preservation failures: {total['healthy_cb_preservation_failures']}.", "",
        "WLS checks chi-square at alpha=0.05 using available measurements minus the actual Jacobian rank, and maximum normalized residual at 4.0. A candidate must be observable, converged and pass both tests. The current model is kept when plausible. A change also requires a complete declared search, exactly one plausible candidate, and a comparison guard separating it from every rival. Gaussian zero-flow and common-relaxation comparison routes share a predeclared family budget; unresolved or insufficiently separated alternatives prevent a correction certificate. The common-relaxation argument is conditional and asymptotic, not an exact nonlinear false-correction guarantee.", "",
        "The raw section formulation accounts for model complexity: opening one coupler adds two voltage states and removes two closed-coupler flow variables. Identifiability is assessed with raw measurements and rank-aware statistics rather than selecting the lowest residual from differently aggregated datasets.", "",
        "Physical rejection reasons (scenario/deployment counts): " + json.dumps(summary["physical_rejection_reasons"], sort_keys=True) + ".", "",
        "## Data and audit boundaries", "",
        "The fixed logical inventory, true status vector, and initial reported status vector are distinct. Runtime inspection/testing/application receives only the reported model and observed sensors. Truth enters the offline feasibility and outcome audit. Topology correction does not rerun PF, redispatch, regenerate observations, or reset other current statuses/parameters.",
        "The main source corpus is never filtered by rule-expert success or wrong-topology WLS convergence. Every planned case has a physical-admission record; true configurations that fail admission remain in the manifest outside the main operating population. Solver nonconvergence is not a proof of physical infeasibility.",
        "Noise, measurement-profile and mixed-error derivatives share a parent physical operating root. Both ordinary and structural holdout views preserve those groups. Reserving a held-out asset moves its entire shared parent to structural test, which can reduce training-family coverage; the views are not a ready-to-train certificate.", "",
        "## Inspect and reproduce", "", "- [Scenario manifest](corpus/manifest.json)", "- [Summary](summary.json)",
        "- [Source/run receipt](run_receipt.json)",
        "- `corpus/row_audits/*.json` contains compact outcomes; `corpus/audits/*.json.gz` contains full candidate evidence.",
        "- `corpus/model_inputs`, `inventories`, `sensors`, and `observations` are explicit reusable inputs to `LogicalTopologyRuntime`.", "",
        "```powershell", f"python scripts/validate_logical_topology.py --output-dir output/logical_topology_new --system {config['system']} --preset {config['preset']} --load-scales {' '.join(str(v) for v in config['load_scales'])} --workers 4", "```", "",
        "This is a logical-switch research testbed. It does not reconstruct IEEE57 physical switchgear, model one-end-open energized lines, or establish learned-policy performance. The IEEE14 detailed/logical bridge is exact in the normal state; its detailed terminal-switch faults have intentionally different semantics.", "",
        "Electrical reference: [MATPOWER whole-branch status/charging implementation](https://matpower.org/docs/ref/matpower7.1/lib/makeYbus.html).", ""]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system", choices=("case14", "case57", "case118"), default="case57")
    parser.add_argument("--load-scales", type=float, nargs="+", default=[.8, 1.0])
    parser.add_argument("--preset", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between one and four")
    max_pairs = args.max_pairs if args.max_pairs is not None else (32 if args.preset == "smoke" else 5000)
    if max_pairs < 0:
        parser.error("max-pairs must be nonnegative")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_before = {name: _sha(REPO/name) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        destination = output/"implementation_snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, destination)
        if _sha(destination) != source_before[name]:
            raise RuntimeError(f"Implementation changed while snapshotting {name}")
    config = {"system": args.system, "preset": args.preset, "seed": args.seed, "load_scales": args.load_scales,
              "max_pairs": max_pairs, "workers": args.workers, "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output/"run_config.json", config)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        manifest = build_corpus(output/"corpus", system=args.system, load_scales=args.load_scales,
                                seed=args.seed, smoke=args.preset == "smoke")
    rows = manifest["rows"]
    # Small batches keep failures durable and load-balance expensive masked or
    # two-error fits. Evidence stays isolated inside each scenario runtime.
    jobs = [{"corpus_dir": str(output/"corpus"), "max_pairs": max_pairs, "rows": rows[start:start+4]}
            for start in range(0, len(rows), 4)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(audit_batch, job) for job in jobs]
        for future in as_completed(futures):
            results.extend(future.result())
            print(f"Audited {len(results)}/{len(rows)} scenario/deployment rows", flush=True)
    results.sort(key=lambda row: row["scenario_id"])
    summary = summarize(manifest, results)
    write_json(output/"audit_results.json", results)
    write_json(output/"summary.json", summary)
    after = {name: _sha(REPO/name) for name in SOURCE_FILES}
    write_json(output/"run_receipt.json", {"config": config, "source_before": source_before, "source_after": after,
        "numerical_versions": numerical_versions(),
        "all_sources_unchanged_during_run": after == source_before, "all_rows_audited": summary["all_rows_audited"],
        "manifest_sha256": _sha(output/"corpus"/"manifest.json"), "summary_sha256": _sha(output/"summary.json")})
    write_report(output, manifest, summary, config)
    print(json.dumps(summary["overall"], indent=2), flush=True)
    return 0 if summary["all_rows_audited"] and not summary["overall"]["audit_execution_failures"] and after == source_before else 2


if __name__ == "__main__":
    raise SystemExit(main())
