"""Held-out family thresholds and matched-budget detector comparison."""
from research.gnn_screen.dataset import make_labels
from research.gnn_screen.evaluate import summarize


def test_matched_budget_and_wls_missed_phase_are_distinct_populations():
    rows = []
    for idx, (families, score, wls_score, wls_alarm) in enumerate([
        ([], .1, .3, False), ([], .3, .8, False),
        (["hif"], .9, .7, False), (["unbalance"], .4, 1.3, True),
        (["parameter"], .8, 1.4, True),
    ]):
        rows.append({"parent_id": f"p{idx}", "labels": make_labels(families),
                     "phase_score": score, "anomaly_score": score,
                     "family_scores": [score] * 5, "wls_score": wls_score,
                     "wls_alarm": wls_alarm, "severity": "weak", "offline_metadata": {}})
    report = summarize(rows, {"phase_threshold": .5, "anomaly_threshold": .5,
                       "family_thresholds": {"hif": .6}, "matched_wls_threshold": 1.2},
                       [True] * 5, bootstrap_replicates=0)
    assert report["phase_recall"]["rate"] == .5
    assert report["phase_recall_among_wls_misses"]["count"] == 1
    assert report["phase_recall_among_wls_misses"]["rate"] == 1
    assert report["matched_budget_wls"]["phase_recall"]["rate"] == .5
    assert report["matched_budget_wls"]["paired_phase_recall_gain_over_wls"]["rate"] == 0
    assert report["by_family"]["hif"]["family_recall"]["rate"] == 1
    assert report["by_family"]["hif"]["family_false_positive_rate"]["rate"] == .25
    assert report["by_family_and_severity_wls"]["hif/weak"]["rate"] == 0
