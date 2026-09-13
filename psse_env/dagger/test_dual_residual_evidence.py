"""Preserve configured residual decisions through study and policy boundaries."""

import copy

import pytest

from psse_env.actions import RUN_WLS
from psse_env.dagger.dataset_builder import (
    _compact_last_tool_output,
    _compact_last_verification,
    summarize_history,
)
from psse_env.dagger.evaluator import objective_tool_evidence
from psse_env.dagger.study_metrics import (
    StudyEvidenceError,
    _OBJECTIVE_TOOL_EVIDENCE_FIELDS,
    _residual_certificate,
    _validated_objective_tool_evidence,
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


def _certificate(metrics):
    evidence = objective_tool_evidence(ACTION, {"tool_metrics": metrics})
    return _validated_objective_tool_evidence(
        evidence, action=ACTION, execution_status="success",
        runtime_state_hash=STATE_HASH, field="evidence",
    )


@pytest.mark.parametrize(
    "statistic,max_residual,expected",
    [(100.0, 5.0, False), (250.0, 3.0, False), (100.0, 3.0, True), (100.0, 4.0, False), (200.0, 3.0, False)],
)
def test_dual_residual_certificate_uses_both_configured_tests(statistic, max_residual, expected):
    evidence = _certificate(_metrics(statistic, max_residual))
    assert _residual_certificate(evidence) == (True, expected, None)


def test_old_certificate_schema_remains_exactly_unchanged():
    metrics = _metrics(enabled=False)
    legacy = {key: value for key, value in metrics.items() if key in _OBJECTIVE_TOOL_EVIDENCE_FIELDS}
    # This field existed in old policy output but was never a narrow certificate field.
    legacy["remaining_anomaly_score"] = 0.5
    evidence = _certificate(legacy)
    assert set(evidence) == _OBJECTIVE_TOOL_EVIDENCE_FIELDS
    assert _residual_certificate(evidence) == (True, True, None)
    assert _residual_certificate(_certificate(metrics)) == (True, True, None)


@pytest.mark.parametrize("field,value", [
    ("normalized_residual_alarm", False), ("chi_square_alarm", True),
    ("chi_square_ratio", 0.9), ("chi_square_alpha", 0.0),
    ("normalized_residual_threshold", -4.0), ("anomaly_detection_rule", "chi_square_only"),
    ("globally_resolved", True), ("no_material_anomaly_remaining", True),
])
def test_contradictory_new_certificates_fail_closed(field, value):
    evidence = _certificate(_metrics())
    evidence[field] = value
    with pytest.raises(StudyEvidenceError):
        _residual_certificate(evidence)


def test_declared_dual_rule_without_threshold_is_unevaluable():
    evidence = _certificate(_metrics())
    evidence.pop("normalized_residual_threshold")
    assert _residual_certificate(evidence) == (False, None, "final_normalized_residual_threshold_missing")


def test_extra_or_mistyped_optional_certificate_fields_fail_validation():
    for field, value in (("unexpected", 0), ("normalized_residual_alarm", "false"), ("anomaly_detection_rule", [])):
        metrics = _metrics()
        evidence = objective_tool_evidence(ACTION, {"tool_metrics": metrics})
        evidence[field] = value
        with pytest.raises(StudyEvidenceError):
            _validated_objective_tool_evidence(
                evidence, action=ACTION, execution_status="success",
                runtime_state_hash=STATE_HASH, field="evidence",
            )


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


def test_exported_optional_evidence_is_independent_of_provider_output():
    metrics = _metrics()
    evidence = _certificate(metrics)
    snapshot = copy.deepcopy(evidence)
    metrics["normalized_residual_threshold"] = 8.0
    assert evidence == snapshot
