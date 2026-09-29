"""Preserve configured residual decisions through study and policy boundaries."""



from psse_env.actions import RUN_WLS
from psse_env.dagger.dataset_builder import (
    _compact_last_tool_output,
    _compact_last_verification,
    summarize_history,
)


ACTION = {"tool": RUN_WLS, "arguments": {"state_id": "active"}}
STATE_HASH = "a" * 64


def _metrics(statistic=100.0, max_residual=5.0, *, enabled=True):
    chi_alarm = statistic >= 200.0
    nr_alarm = enabled and max_residual >= 4.0
    return {
        "state_id": "active", "state_hash": STATE_HASH,
        "evidence_source": "deployment_wls:lagrangian_port",
        "chi_square_statistic": statistic, "chi_square_threshold": 200.0,
        "max_normalized_residual": max_residual,
        "normalized_residual_threshold": 4.0 if enabled else None,
        "normalized_residual_alarm": nr_alarm, "chi_square_alarm": chi_alarm,
        "chi_square_alpha": 0.05, "chi_square_ratio": statistic / 200.0,
        "anomaly_detection_rule": (
            "chi_square_or_normalized_residual" if enabled else "chi_square_only"
        ),
        "remaining_anomaly_score": max(statistic / 200.0, max_residual / 4.0) if enabled else statistic / 200.0,
        "no_material_anomaly_remaining": not (chi_alarm or nr_alarm),
        "globally_resolved": not (chi_alarm or nr_alarm),
    }


def test_policy_compaction_retains_decision_rule_thresholds_and_composite_score():
    metrics = _metrics()
    output = {"tool_metrics": metrics, "execution_status": "success"}
    summaries = [
        _compact_last_tool_output(output)["observable_metrics"],
        _compact_last_verification(metrics),
        summarize_history([{"action": ACTION, "tool_output": output}])[0]["observable_metrics"],
    ]
    for summary in summaries:
        for key in (
            "normalized_residual_threshold", "normalized_residual_alarm", "chi_square_alarm",
            "chi_square_alpha", "chi_square_ratio", "anomaly_detection_rule", "remaining_anomaly_score",
        ):
            assert summary[key] == metrics[key]
