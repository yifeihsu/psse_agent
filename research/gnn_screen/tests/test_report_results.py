"""Reporting checks catch leakage, mismatched artifacts, and denominator errors."""
import copy
import json

import pytest

from research.gnn_screen.dataset import make_labels
from research.gnn_screen.evaluate import summarize
from research.gnn_screen.report_results import audit_artifacts, main, render_report


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
