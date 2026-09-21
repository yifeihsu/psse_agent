"""Deterministic shards preserve physical lineage, immutable audits and ledgers."""
import copy
import json
from pathlib import Path

import pytest

from research.gnn_screen.dataset import file_sha256, load_manifest, write_json
from research.gnn_screen.generate_practical_parallel import (generate_parallel, merge_shards,
                                                            rewrite_manifest_row, shard_plan)


def test_split_counts_and_independent_child_seeds_are_reproducible():
    counts = {"train": 384, "validation": 96, "calibration": 128, "test": 128}
    plan = shard_plan(counts, seed=20260918, workers=4)
    assert plan == shard_plan(counts, seed=20260918, workers=4)
    assert len({item["seed"] for item in plan}) == 4
    assert all(item["parents_by_split"] == {"train": 96, "validation": 24, "calibration": 32, "test": 32} for item in plan)
    uneven = shard_plan({"train": 11, "test": 5}, seed=12, workers=3)
    assert sum(item["parents_by_split"]["train"] for item in uneven) == 11
    assert sum(item["parents_by_split"]["test"] for item in uneven) == 5


def test_rewrite_changes_only_path_references_and_retains_inline_measurements():
    row = {"case": "parents/p/case.json", "z": [1.1, -2.3], "measurement_sigma": "sigma.json",
        "parent_id": "p", "window_id": "p:hif", "noise_seed": 55, "families": ["hif"],
        "offline_metadata": {"physical_audit_path": "parents/p/physical.json", "full_provenance_hash": "original",
            "nested": {"actual_model_path": "parents/p/model", "classification": "boundary"},
            "reason": "do not rewrite this string with a/slash"}}
    original = copy.deepcopy(row)
    rewritten = rewrite_manifest_row(row, "shards/shard_001")
    assert row == original
    assert rewritten["case"] == "shards/shard_001/parents/p/case.json"
    assert rewritten["z"] == row["z"]
    assert rewritten["offline_metadata"]["physical_audit_path"] == "shards/shard_001/parents/p/physical.json"
    assert rewritten["offline_metadata"]["source_shard_root"] == "shards/shard_001"
    assert rewritten["offline_metadata"]["full_provenance_hash"] == "original"
    assert rewritten["offline_metadata"]["reason"] == row["offline_metadata"]["reason"]
    row["z"] = "observations.json"
    assert rewrite_manifest_row(row, "shards/shard_001")["z"] == "shards/shard_001/observations.json"


def _fake_shard(root, item, *, parent=None):
    index = item["shard_index"]
    directory = root / "shards" / f"shard_{index:03d}"
    directory.mkdir(parents=True)
    parent = parent or f"parent-{item['seed']}"
    row = {"case": "case.json", "z": [1.0], "measurement_sigma": "sigma.json", "parent_id": parent,
           "window_id": f"{parent}:healthy", "split": "train", "families": [], "noise_replicates": 2,
           "offline_metadata": {"physical_audit_path": "audit.json", "full_provenance_hash": "immutable"}}
    (directory / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (directory / "boundary_manifest.jsonl").write_text("", encoding="utf-8")
    write_json(directory / "audit.json", {"actual_physical_model_path": "model", "original": True})
    ledger = [{"event": "proposal", "parent_id": parent, "outcome": "valid_positive_below_main_criteria",
               "scenario_policy": {"rejection_reasons": ["physical_core_rule_not_met", "paired_mean_distance_below_main_margin"]}},
              {"event": "proposal", "parent_id": parent, "outcome": "simulation_failure", "reason": "voltage-envelope fallback"}]
    (directory / "proposal_ledger.jsonl").write_text("\n".join(json.dumps(row) for row in ledger) + "\n", encoding="utf-8")
    report = {"seed": item["seed"], "parents_by_split": item["parents_by_split"], "policy_version": "test_policy",
        "implementation_sha256": {"scenario_policy.py": "same-source"}, "noise_replicates": 2,
        "healthy_replicates_by_split": {"train": 2}, "attempt_cap_per_slot": 8, "main_minimum_paired_distance": 5,
        "profiles": {"parameter": "physical_actual_over_reported_parent"},
        "counts": {"main:train:rows": 1, "main:train:noise_windows": 2, "main:train:healthy": 1,
                   "train:parents": 1, "proposal:train:below_main": 1, "proposal:train:simulation_failure": 1},
        "manifest_sha256": file_sha256(directory / "manifest.jsonl"),
        "boundary_manifest_sha256": file_sha256(directory / "boundary_manifest.jsonl"), "elapsed_seconds": 1.,
        "healthy_max_balanced_equation_error_pu": 1e-10, "intended_main_slots_per_noncalibration_parent": 18,
        "admission_uses_wls_alarm": False, "admission_uses_noisy_or_learned_scores": False,
        "reported_model_stays_identical_within_parent": True, "limitations": ["test only"]}
    write_json(directory / "generation_report.json", report)
    return report


def test_merge_preserves_all_rejections_failure_receipts_and_source_audit_bytes(tmp_path):
    plan = shard_plan({"train": 2}, seed=17, workers=2)
    reports = [_fake_shard(tmp_path, item) for item in plan]
    originals = [file_sha256(tmp_path / "shards" / f"shard_{i:03d}" / "audit.json") for i in range(2)]
    result = merge_shards(tmp_path, reports, plan=plan, seed=17, elapsed_seconds=1.5)
    rows = [json.loads(line) for line in Path(result["manifest"]).read_text().splitlines()]
    assert [row["parent_id"] for row in rows] == [f"parent-{item['seed']}" for item in plan]
    assert result["counts"]["main:train:noise_windows"] == 4
    assert result["proposal_audit"]["records"] == 4
    assert result["proposal_audit"]["rejection_reason_histogram"] == {
        "physical_core_rule_not_met": 2, "paired_mean_distance_below_main_margin": 2}
    assert result["physical_failures"]["count"] == 2
    assert result["physical_failures"]["reason_histogram"] == {"voltage-envelope fallback": 2}
    for i, original in enumerate(originals):
        assert file_sha256(tmp_path / "shards" / f"shard_{i:03d}" / "audit.json") == original


def test_merge_rejects_cross_shard_parent_aliases(tmp_path):
    plan = shard_plan({"train": 2}, seed=17, workers=2)
    reports = [_fake_shard(tmp_path, item, parent="same-parent") for item in plan]
    with pytest.raises(ValueError, match="different shards"):
        merge_shards(tmp_path, reports, plan=plan, seed=17, elapsed_seconds=1)


def test_merge_rejects_mixed_review_or_covariance_profiles(tmp_path):
    plan = shard_plan({"train": 2}, seed=23, workers=2)
    reports = [_fake_shard(tmp_path, item) for item in plan]
    for report in reports:
        report.update(scenario_profile="reviewed_v1", noise_profile="baseline", curriculum_stage="full")
    reports[1]["noise_profile"] = "accuracy_005"
    with pytest.raises(ValueError, match="different noise_profile"):
        merge_shards(tmp_path, reports, plan=plan, seed=23, elapsed_seconds=1)


def test_actual_two_worker_corpus_is_readable_by_existing_manifest_loader(tmp_path):
    # This is a real physical-generation integration check, not a detection test.
    result = generate_parallel(tmp_path / "physical", parents_by_split={"train": 2}, seed=841202609,
                               workers=2, noise_replicates=1, healthy_calibration_replicates=2, attempt_cap=8)
    rows = load_manifest(result["manifest"])
    assert {row["parent_id"] for row in rows} == {
        f"practical_ieee14_{item['seed']}_train_00000" for item in result["source_reports"]}
    assert len(rows) == result["counts"]["main:train:rows"]
    assert all(row["case"]["baseMVA"] == 100 for row in rows)
    assert all(row["measurement_sigma"] == rows[0]["measurement_sigma"] for row in rows)
    root = Path(result["manifest"]).parent
    for row in rows:
        metadata = row["offline_metadata"]
        assert (root / metadata["physical_audit_path"]).is_file()
        audit = json.loads((root / metadata["physical_audit_path"]).read_text())
        if "actual_physical_model_path" in audit:
            assert (root / metadata["source_shard_root"] / audit["actual_physical_model_path"]).is_dir()


def test_reviewed_parallel_keeps_evaluation_manifests_and_profile_identity(tmp_path):
    result = generate_parallel(tmp_path / "reviewed", parents_by_split={"validation": 2}, seed=184201,
        workers=2, noise_replicates=1, healthy_calibration_replicates=1, attempt_cap=2,
        scenario_profile="reviewed_v1", noise_profile="accuracy_002", stage="full")
    assert result["schema"] == "parallel_practical_physical_wls_screen_corpus_v2"
    assert result["scenario_profile"] == "reviewed_v1" and result["noise_profile"] == "accuracy_002"
    weak = result["auxiliary_manifests"]["weak_hif_evaluation"]
    rows = load_manifest(weak["path"])
    assert weak["row_count"] == len(rows) == 2
    assert all(row["split"] == "validation" and not row["offline_metadata"]["training_eligible"] for row in rows)
    for row in load_manifest(result["manifest"]):
        assert row["measurement_sigma"][0] == .001 and row["measurement_sigma"][14] == .002
        assert row["offline_metadata"]["profile_identity_sha256"] == result["profile_identity_sha256"]
    for info in result["auxiliary_manifests"].values():
        assert file_sha256(info["path"]) == info["sha256"]
