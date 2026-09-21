from __future__ import annotations

import json

import numpy as np
import pytest

from three_phase_nlm.conditioned_meter_recovery import (
    diagnose_conditioned_meter_errors,
    diagnose_nonoverlap_meter_errors,
)


def test_physical_prediction_retains_same_channel_meter_error_and_physical_effect():
    baseline = np.linspace(0.5, 1.5, 20)
    effect = np.zeros(20)
    effect[3] = 0.04
    prediction = baseline + effect
    observed = prediction.copy()
    observed[3] += 0.10
    observed[7] += 0.003  # Preserve ordinary noise verbatim.
    result = diagnose_conditioned_meter_errors(observed, prediction, 0.01, event_effect=effect)
    assert result["recovery_supported"]
    assert result["candidate_indices"] == [3]
    assert result["candidate_event_overlap_indices"] == [3]
    assert result["proposed_measurements"][3] == prediction[3]
    assert result["proposed_measurements"][3] != baseline[3]
    assert np.array_equal(np.array(result["proposed_measurements"])[np.arange(20) != 3], observed[np.arange(20) != 3])
    assert result["conditional_scores"][3] == pytest.approx(10)
    assert not result["physical_model_validated"]
    json.dumps(result, allow_nan=False)


def test_deterministic_envelope_uses_distance_beyond_edge_and_rejects_wide_repair():
    center = np.zeros(20)
    lower, upper = center - 1, center + 1
    observed = center.copy()
    observed[2] = 7
    narrow = diagnose_conditioned_meter_errors(observed, center, 1, prediction_lower=lower, prediction_upper=upper)
    assert narrow["conditional_scores"][2] == 6
    assert narrow["recovery_supported"]
    lower[2] = -2
    wide = diagnose_conditioned_meter_errors(observed, center, 1, prediction_lower=lower, prediction_upper=upper)
    assert not wide["recovery_supported"]
    assert wide["unsupported_candidate_indices"] == [2]
    assert wide["proposed_measurements"] == observed.tolist()
    assert wide["replacements"] == []


def test_wrong_physical_model_with_broad_residuals_cannot_be_repaired_as_meters():
    observed = np.ones(20) * 7
    result = diagnose_conditioned_meter_errors(observed, np.zeros(20), 1)
    assert not result["recovery_supported"]
    assert "broad_residual_pattern_or_too_many_meter_errors" in result["failure_reasons"]
    assert result["proposed_measurements"] == observed.tolist()
    assert not result["model_adequate"]


def test_fallback_repairs_retained_channel_and_preserves_excluded_observations():
    baseline = np.linspace(0.5, 1.5, 20)
    effect = np.zeros(20)
    effect[3] = 0.04
    observed = baseline + effect
    observed[9] += 0.10
    result = diagnose_nonoverlap_meter_errors(observed, baseline, 0.01, event_effect=effect)
    assert result["recovery_supported"]
    assert result["candidate_indices"] == [9]
    assert result["excluded_indices"] == [3]
    assert result["proposed_measurements"][3] == observed[3]
    assert result["proposed_measurements"][9] == baseline[9]
    assert result["assumes_nonoverlapping_impacts"]
    assert result["requires_independent_overlap_audit"]
    json.dumps(result, allow_nan=False)


def test_fallback_overlap_blocks_even_an_otherwise_repairable_nonoverlap_error():
    baseline = np.zeros(20)
    effect = np.zeros(20)
    effect[3] = 4
    observed = baseline + effect
    observed[3] += 10
    observed[9] += 10
    result = diagnose_nonoverlap_meter_errors(observed, baseline, 1, event_effect=effect)
    assert not result["recovery_supported"]
    assert result["candidate_indices"] == [9]
    assert result["unexplained_excluded_indices"] == [3]
    assert result["unsupported_candidate_indices"] == [3]
    assert "overlap_or_insufficient_physical_model_evidence" in result["failure_reasons"]
    assert result["proposed_measurements"] == observed.tolist()
    assert result["replacements"] == []


def test_no_observable_channels_never_reports_supported_recovery():
    empty = diagnose_conditioned_meter_errors([], [], 1)
    assert not empty["recovery_supported"]
    result = diagnose_nonoverlap_meter_errors([4, 4], [0, 0], 1, event_effect=[4, 4])
    assert not result["recovery_supported"]
    assert result["retained_indices"] == []
    assert result["failure_reasons"] == ["no_observable_channels_after_exclusion"]
    assert not result["no_remaining_discrepancy_evidence"]


def test_threshold_is_strict_and_quiet_result_is_not_physical_validation():
    result = diagnose_conditioned_meter_errors([5, -5], [0, 0], 1)
    assert result["candidate_indices"] == []
    assert result["recovery_supported"]
    assert result["no_remaining_discrepancy_evidence"]
    assert result["requires_independent_physical_model_validation"]
    assert not result["physical_model_validated"]


def test_inputs_are_not_mutated_by_either_helper():
    z = np.zeros(20)
    z[4] = 10
    center = np.zeros(20)
    sigma = np.ones(20)
    lower = center - 0.1
    upper = center + 0.1
    effect = np.zeros(20)
    inputs = [z, center, sigma, lower, upper, effect]
    copies = [value.copy() for value in inputs]
    diagnose_conditioned_meter_errors(z, center, sigma, prediction_lower=lower, prediction_upper=upper, event_effect=effect)
    diagnose_nonoverlap_meter_errors(z, center, sigma, event_effect=effect)
    for actual, original in zip(inputs, copies):
        assert np.array_equal(actual, original)


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), [1, 0]])
def test_invalid_sigmas_are_rejected(sigma):
    with pytest.raises(ValueError, match="sigma_z"):
        diagnose_conditioned_meter_errors([0, 0], [0, 0], sigma)


def test_shape_and_envelope_errors_cannot_silently_broadcast():
    with pytest.raises(ValueError, match="matching length"):
        diagnose_conditioned_meter_errors([0, 0], [0], 1)
    with pytest.raises(ValueError, match="supplied together"):
        diagnose_conditioned_meter_errors([0], [0], 1, prediction_lower=[-1])
    with pytest.raises(ValueError, match="lower <= predicted_hif <= upper"):
        diagnose_conditioned_meter_errors([0], [0], 1, prediction_lower=[1], prediction_upper=[2])
    with pytest.raises(ValueError, match="finite"):
        diagnose_nonoverlap_meter_errors([0], [0], 1, event_effect=[float("nan")])
