"""Bounded direct-sensor branch-status bridge through the existing expert stack.

Select the first physically admitted asset in each of four declared families
before executing any policy. No generator admission, observations, core runtime,
or teacher is changed. Couplers and masked/correlated sensors are out of scope.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from functools import partial
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from logical_topology.inventory import process_topology
from psse_env.dagger.evaluator import ClosedLoopRolloutEvaluator
from psse_env.dagger.release_factories import (
    EXPERT_POLICY_IDENTITY, deterministic_case_loader,
    observable_expert_policy_factory, production_environment_factory,
)
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.providers.matpower import _render_matpower_case
from scripts.validate_balanced_transfer import collect_source_receipt, summarize_episode

FAMILIES = ("healthy_closed", "exclusion", "inclusion", "healthy_outage")


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def _source(root, relative, hashes):
    path = (root/relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("corpus reference leaves the declared source directory")
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def _asset_key(row):
    devices = row["error_device_ids"] or [key for key, value in row["true_statuses"].items() if value == 0]
    indices = [int(device.split(":branch:")[1].split(":")[0]) for device in devices]
    return (min(indices, default=0), float(row["load_scale"]), row["scenario_id"])


def run_probe(corpus_dir, output_dir):
    corpus = Path(corpus_dir).resolve(strict=True)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    manifest = _source(corpus, "manifest.json", hashes)
    eligible = [row for row in manifest["rows"] if row["layout"] == "branch_status"
                and row["measurement_profile"] == "direct"
                and row["physical_admission"]["admitted"] is True]
    selected = []
    for family in FAMILIES:
        rows = sorted((row for row in eligible if row["family"] == family), key=_asset_key)
        if not rows:
            raise ValueError(f"no physically admitted direct branch case for {family}")
        selected.append(rows[0])
    selection = {"contract": "predeclared_physics_only_four_family_branch_bridge_v1",
                 "selection_rule": "first canonical asset then load scale then scenario id in each declared family; physical admission only",
                 "teacher_results_used_for_selection": False,
                 "families": list(FAMILIES),
                 "source_scenario_ids": [row["scenario_id"] for row in selected],
                 "source_parent_physical_worlds": sorted({row["parent_physical_root"] for row in selected})}
    _write(out/"selection_before_policy.json", selection)
    cases = out/"cases"
    cases.mkdir()
    envelopes = []
    conversions = []
    for row in selected:
        execution = row["execution"]
        inventory = _source(corpus, execution["inventory_path"], hashes)
        sensors = _source(corpus, execution["measurement_inventory_path"], hashes)
        observations = _source(corpus, execution["observations_path"], hashes)
        base = _source(corpus, execution["base_case_path"], hashes)
        physical = _source(corpus, row["physical_audit_path"], hashes)
        if inventory["couplers"] or len(inventory["nodes"]) != 57 or len(sensors["records"]) != 491:
            raise ValueError("legacy bridge requires the branch-only IEEE57 physical inventory")
        expected_kinds = [kind for kind, count in (("Vm",57),("Pinj",57),("Qinj",57),("Pf",80),("Qf",80),("Pt",80),("Qt",80)) for _ in range(count)]
        covariance = np.asarray(sensors["covariance"])
        target_covariance = np.diag([.001**2 if kind == "Vm" else .01**2 for kind in expected_kinds])
        if (not all(sensors["available_mask"]) or any(value is None for value in observations["values"])
            or [record["kind"] for record in sensors["records"]] != expected_kinds
            or not np.array_equal(covariance, target_covariance)
            or observations["sensor_inventory_hash"] != sensors["sensor_inventory_hash"]
            or observations["sensor_ids"] != [record["sensor_id"] for record in sensors["records"]]
            or physical["admitted"] is not True):
            raise ValueError("legacy bridge cannot change masks, covariance, row order, or physical admission")
        reported = process_topology(base, inventory, row["model_statuses"])["case"]
        clean = process_topology(base, inventory, row["true_statuses"])["case"]
        tag = row["scenario_id"][:16]
        reported_path, clean_path = cases/f"reported_{tag}.m", cases/f"clean_{tag}.m"
        reported_path.write_text(_render_matpower_case(reported, f"reported_{tag}"), encoding="utf-8")
        clean_path.write_text(_render_matpower_case(clean, f"clean_{tag}"), encoding="utf-8")
        faults = []
        for asset in inventory["branches"]:
            device = asset["device_id"]
            if row["model_statuses"][device] != row["true_statuses"][device]:
                faults.append({"branch_row0": asset["row0"], "line_index1": asset["row0"]+1,
                               "expected_status": row["true_statuses"][device], "reported_status": row["model_statuses"][device]})
        values = list(observations["values"])
        legacy = {"scenario_id": f"logical_bridge_{tag}", "scenario_family": "topology" if faults else "no_error",
                  "network_case": "case57", "error_cardinality": len(faults), "source_tier": "physics_synthesized",
                  "source_realization_id": row["parent_physical_root"], "scenario_admission_mode": "physical",
                  "case": str(reported_path), "measurements": values, "clean_case": str(clean_path),
                  "clean_measurements": list(values), "metadata": {}, "true_topology_errors": faults,
                  "true_measurement_errors": [], "true_parameter_errors": []}
        envelope = partition_release_scenario_v1(legacy, split="development")
        if envelope["execution"]["measurements"] != values:
            raise RuntimeError("conversion changed the fixed observations")
        envelopes.append(envelope)
        conversions.append({"bridge_scenario_id": legacy["scenario_id"], "source_scenario_id": row["scenario_id"],
                            "source_family": row["family"], "source_parent_physical_root": row["parent_physical_root"],
                            "fixed_observation_sha256": hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest(),
                            "reported_case_sha256": hashlib.sha256(reported_path.read_bytes()).hexdigest(),
                            "clean_case_sha256": hashlib.sha256(clean_path.read_bytes()).hexdigest(),
                            "physical_admission_passed": True})
    _write(out/"scenarios.json", {"standard_success": envelopes})
    _write(out/"conversion_receipt.json", {"sources_sha256": hashes, "conversions": conversions})
    before = collect_source_receipt(phase="before_evaluation")
    for relative in ("scripts/probe_logical_branch_legacy.py", "logical_topology/inventory.py"):
        before["sha256"][relative] = hashlib.sha256((REPO/relative).read_bytes()).hexdigest()
    def progress(record):
        if record.get("event") == "episode_complete":
            print(json.dumps(record), flush=True)
    evaluator = ClosedLoopRolloutEvaluator(
        env_factory=partial(production_environment_factory, chi2_alpha=.05, normalized_residual_threshold=4.0),
        policy_factory=observable_expert_policy_factory, case_loader=deterministic_case_loader,
        max_steps=40, seed=20260911, required_suites=("standard_success",),
        minimum_suites=1, minimum_episodes_per_suite=4, minimum_roots_per_suite=4,
        require_release_environment=True,
        expected_policy_identity={"explicit_policy_identity": EXPERT_POLICY_IDENTITY, "model_id": None, "model_revision": None},
        require_policy_identity=True, progress_callback=progress,
    )
    with threadpool_limits(limits=1):
        evaluation = evaluator.evaluate({"standard_success": envelopes}).as_dict()
    after = collect_source_receipt(phase="after_evaluation")
    for relative in ("scripts/probe_logical_branch_legacy.py", "logical_topology/inventory.py"):
        after["sha256"][relative] = hashlib.sha256((REPO/relative).read_bytes()).hexdigest()
    episodes = [summarize_episode(row) for row in evaluation["suite_metrics"]["episodes"]]
    kinds = {row["bridge_scenario_id"]: row["source_family"] for row in conversions}
    for episode in episodes:
        episode["source_family"] = kinds[episode["scenario_id"]]
    report = {"contract": "bounded_logical_branch_existing_pipeline_bridge_v1",
              "scope": "four preselected branch-only direct491 cases through existing transactional environment, providers, observable rule expert and strict audit",
              "couplers_supported_by_this_bridge": False, "masked_or_correlated_sensors_supported": False,
              "training_performed": False, "selection": selection, "conversions": conversions,
              "sources_sha256": hashes, "runtime_source_attestation": {"before": before, "after": after,
              "source_hashes_matched": before["sha256"] == after["sha256"]},
              "source_data_unchanged": all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in hashes.items()),
              "episodes": episodes, "evaluation": evaluation}
    _write(out/"expert_bridge_report.json", report)
    print(json.dumps({"output": str(out/"expert_bridge_report.json"), "task_successes": sum(row["truth_audited_task_success"] for row in episodes),
                      "episodes": [{key: row[key] for key in ("source_family", "truth_audited_task_success", "terminal_outcome", "false_commit_count", "false_finalization_count", "evaluator_error")} for row in episodes]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.corpus_dir, args.output_dir)
    errors = any(row["evaluator_error"] or not row["truth_audited_task_success_evidence_known"] for row in result["episodes"])
    raise SystemExit(2 if errors or not result["source_data_unchanged"] or not result["runtime_source_attestation"]["source_hashes_matched"] else 0)
