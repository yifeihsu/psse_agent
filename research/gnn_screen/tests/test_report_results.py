"""Reporting checks catch leakage, mismatched artifacts, and denominator errors."""
import copy
import json

import pytest

from research.gnn_screen.dataset import make_labels
from research.gnn_screen.evaluate import summarize
from research.gnn_screen.report_results import audit_artifacts, generation_coverage, main, render_report


def artifacts():
    predictions = []
    for parent, family, wls, score in [
        ("test-a", [], False, .1), ("test-b", [], False, .2),
        ("test-a", ["hif"], False, .8), ("test-a", ["hif"], False, .2),
        ("test-b", ["unbalance"], True, .9),
    ]:
        predictions.append({"parent_id": parent, "window_id": str(len(predictions)),
            "labels": make_labels(family), "phase_score": score, "anomaly_score": score,
            "family_scores": [score] * 5, "wls_alarm": wls,
            "severity": "weak", "offline_metadata": {}})
    calibration = {"model_id": "model", "checkpoint_sha256": "sha", "phase_threshold": .5,
        "anomaly_threshold": .5, "calibration_parent_ids": ["cal"], "healthy_count": 100,
        "healthy_parent_count": 1, "requested_healthy_false_trigger_rate": .01,
        "wls_comparator": {"rule": "J >= threshold", "chi2_alpha": .01,
                           "normalized_residual_threshold": None}}
    evaluation = summarize(predictions, calibration, [True] * 5, bootstrap_replicates=10)
    evaluation.update({"calibration": calibration, "model_id": "model", "checkpoint_sha256": "sha",
        "test_parent_ids": ["test-a", "test-b"], "predictions": predictions,
        "invalid_test_windows": [], "screen_availability": 1,
        "phase_trigger_recall_counting_unavailable_as_untriggered": 2 / 3})
    training = {"parent_splits": {"train": "train", "val": "validation", "cal": "calibration",
                                  "test-a": "test", "test-b": "test"},
        "selected": {"model_id": "model", "seed": 0, "epoch": 3},
        "valid_graph_counts": {"train": 3, "validation": 3, "calibration": 100, "test": 5}}
    return evaluation, training


def test_report_uses_conditional_wls_miss_denominator_and_parent_count():
    evaluation, training = artifacts()
    text, audit = render_report(evaluation, training, bootstrap_replicates=10)
    rate = audit["configured_wls_negative_phase_recall"]
    assert rate["count"] == 2 and rate["parents"] == 1 and rate["rate"] == .5
    assert "missed by configured WLS | 50.00% | 2 | 1" in text
    assert "GNN phase recall | 66.67% | 3 | 2" in text
    assert "does not establish a zero population false-trigger rate" in text
    assert "not phase-detection successes" in text
    assert audit["phase_composition"]["hif"]["gnn_phase_recall"]["count"] == 2
    assert audit["phase_composition"]["unbalance"]["gnn_phase_recall"]["count"] == 1
    assert "hif | GNN | 50.00% | 2 | 1" in text


@pytest.mark.parametrize("bad_split", ["train", "validation", "calibration"])
def test_report_rejects_overlapping_or_reassigned_test_parents(bad_split):
    evaluation, training = artifacts()
    training["parent_splits"]["test-a"] = bad_split
    with pytest.raises(ValueError, match="overlap|assignment"):
        audit_artifacts(evaluation, training)


def test_report_rejects_checkpoint_mismatch_and_bad_prediction_counts():
    evaluation, training = artifacts()
    bad = copy.deepcopy(evaluation)
    bad["calibration"]["checkpoint_sha256"] = "other-checkpoint"
    with pytest.raises(ValueError, match="mismatch"):
        audit_artifacts(bad, training)
    evaluation["valid_test_windows"] += 1
    with pytest.raises(ValueError, match="prediction count"):
        audit_artifacts(evaluation, training)


def test_report_exposes_matched_wls_and_family_head_metrics_separately():
    evaluation, training = artifacts()
    rate = {"count": 5, "parents": 2, "rate": .8, "parent_bootstrap_ci95": [.5, 1]}
    evaluation["matched_budget_wls"] = {"phase_recall": rate, "healthy_false_trigger_rate": rate,
        "gnn_recall_among_matched_wls_misses": rate}
    evaluation["by_family"]["hif"].update({"family_threshold": .7, "family_recall": rate,
                                           "family_healthy_trigger_rate": rate})
    text, _ = render_report(evaluation, training, bootstrap_replicates=0)
    assert "Matched WLS phase recall | 80.00%" in text
    assert "hif | recall | 0.7000 | 80.00%" in text
    assert "combined probability of one or more heads" in text


def test_report_cli_writes_durable_markdown_and_audit_json(tmp_path):
    evaluation, training = artifacts()
    evaluation_path = tmp_path / "evaluation.json"
    training_path = tmp_path / "training.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    training_path.write_text(json.dumps(training), encoding="utf-8")
    report_path = tmp_path / "report.md"
    audit_path = tmp_path / "audit.json"
    main(["--evaluation", str(evaluation_path), "--training-report", str(training_path),
          "--output", str(report_path), "--summary-json", str(audit_path), "--bootstrap-replicates", "0"])
    assert "held-out simulation pilot" in report_path.read_text(encoding="utf-8")
    assert json.loads(audit_path.read_text(encoding="utf-8"))["test_parent_count"] == 2


def test_practical_generation_reports_real_failures_and_selection_coverage():
    evaluation, training = artifacts()
    generation = {
        "schema": "parallel_practical_physical_wls_screen_corpus_v1", "policy_version": "sft_practical_v1",
        "seed": 17, "main_minimum_paired_distance": 5, "attempt_cap_per_slot": 8,
        "parents_by_split": {"train": 2, "test": 2},
        "profiles": {"hif_main": "Rpu 5-40", "hif_challenge": "SFT Rpu 20-200"},
        "counts": {"main:train:rows": 20, "main:train:noise_windows": 40, "main:train:healthy": 2,
            "boundary:test:rows": 7, "boundary:test:noise_windows": 14, "test:unfilled_slots": 3},
        "physical_failures": {"count": 4, "reason_histogram": {"physical divergence": 4}, "ledger": "failures.jsonl"},
        "proposal_audit": {"records": 80, "event_histogram": {"proposal": 74},
            "outcome_histogram": {"valid_positive_below_main_criteria": 15, "simulation_failure": 4},
            "rejection_reason_histogram": {"paired_mean_distance_below_main_margin": 12}},
        "admission_uses_wls_alarm": False, "admission_uses_noisy_or_learned_scores": False,
        "reported_model_stays_identical_within_parent": True,
        "source_reports": [{"shard_index": 0, "seed": 731, "shard_root": "shards/shard_000",
                            "generation_report": "shards/shard_000/generation_report.json"}],
    }
    text, audit = render_report(evaluation, training, generation=generation, bootstrap_replicates=0)
    assert "Failed physical proposals or parent-source attempts | 4" in text
    assert "Main minimum paired noiseless mean distance | 5" in text
    assert "Unfilled slots | 3" in text
    assert "main | train | 20 | 40 | 2" in text
    assert "boundary | test | 7 | 14 | 0" in text
    assert "valid_positive_below_main_criteria | 15" in text
    assert "paired_mean_distance_below_main_margin | 12" in text
    assert "hif_challenge | SFT Rpu 20-200" in text
    assert "shards/shard_000/generation_report.json" in text
    assert "not the fitted WLS residual statistic" in text
    assert "histograms overlap" in text
    assert "Physical simulation failures are excluded" in text
    assert audit["generation_coverage"]["physical_failure_count"] == 4


def test_legacy_generation_failure_reporting_and_unknown_counts_remain_honest():
    evaluation, training = artifacts()
    generation = {"schema": "physical_wls_screen_corpus_v1", "failed_variants": [{"variant": "one"}, {"variant": "two"}]}
    text, audit = render_report(evaluation, training, generation=generation, bootstrap_replicates=0)
    assert "Failed physical variants | 2" in text
    assert audit["generation_coverage"]["physical_failure_count"] == 2
    assert "Main minimum paired" not in text
    text, _ = render_report(evaluation, generation={}, bootstrap_replicates=0)
    assert "Failed physical variants / proposals | not available" in text


def test_serial_practical_failure_counts_include_source_rejections():
    result = generation_coverage({"schema": "practical_physical_wls_screen_corpus_v1",
        "counts": {"proposal:train:simulation_failure": 5, "proposal:test:simulation_failure": 2,
                   "parent_source_rejections": 1}, "parents_by_split": {"train": 2, "test": 1}})
    assert result["physical_failure_count"] == 8
    assert result["unfilled_slot_count"] == 0
