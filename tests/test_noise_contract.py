from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from psse_env.noise_contract import validate_noise_channel, validate_shared_scada_covariance


def _channel(**overrides):
    kwargs = dict(channel="voltage_phasors", role="sensor_observation", distribution="gaussian",
                  applied_sigma=0.005, estimator_sigma=0.005, representation="complex_rectangular")
    kwargs.update(overrides)
    return validate_noise_channel(**kwargs)


def test_matched_sensor_declarations_do_not_certify_actual_draws_or_calibration():
    receipt = _channel(require_matched_gaussian=True)
    assert receipt["matched_gaussian"]
    assert receipt["applied_matches_estimator"]
    assert receipt["declaration_only"]
    assert not receipt["source_draws_verified"]
    assert not receipt["population_calibration_verified"]
    json.dumps(receipt, allow_nan=False)


def test_complex_rms_to_component_conversion_exposes_harmonic_weight_mismatch():
    mismatch = _channel(applied_sigma=1e-4, estimator_sigma=1e-4, applied_sigma_semantics="complex_rms")
    assert mismatch["applied_sigma_per_component"] == pytest.approx(1e-4 / np.sqrt(2))
    assert not mismatch["matched_gaussian"]
    with pytest.raises(ValueError, match="applied_estimator_sigma_mismatch"):
        _channel(applied_sigma=1e-4, estimator_sigma=1e-4, applied_sigma_semantics="complex_rms", require_matched_gaussian=True)
    matched = _channel(applied_sigma=1e-4, estimator_sigma=1e-4 / np.sqrt(2), applied_sigma_semantics="complex_rms", require_matched_gaussian=True)
    assert matched["matched_gaussian"]


def test_both_sigmas_can_be_declared_complex_rms():
    receipt = _channel(applied_sigma_semantics="complex_rms", estimator_sigma_semantics="complex_rms")
    assert receipt["matched_gaussian"]
    assert receipt["estimator_sigma_per_component"] == pytest.approx(0.005 / np.sqrt(2))


@pytest.mark.parametrize("role", ["noiseless_reference", "model_prediction"])
def test_reference_or_prediction_never_passes_as_gaussian_sensor_data(role):
    receipt = _channel(role=role)
    assert receipt["applied_matches_estimator"]
    assert not receipt["matched_gaussian"]
    with pytest.raises(ValueError, match="not_sensor_observation"):
        _channel(role=role, require_matched_gaussian=True)


def test_noiseless_legacy_channel_and_unknown_noise_are_explicitly_uncalibrated():
    quiet = _channel(distribution="none", applied_sigma=0)
    assert not quiet["matched_gaussian"]
    assert quiet["applied_sigma_per_component"] == 0
    unknown = _channel(distribution="unknown", applied_sigma=None)
    assert unknown["applied_sigma_per_component"] is None
    assert "applied_sigma_unknown" in unknown["mismatch_reasons"]
    for kwargs in ({"distribution": "none", "applied_sigma": 0}, {"distribution": "unknown", "applied_sigma": None}):
        with pytest.raises(ValueError, match="matched Gaussian sensor noise required"):
            _channel(**kwargs, require_matched_gaussian=True)


def test_scalar_broadcasting_and_input_immutability():
    sigma = np.full(4, 0.005)
    before = sigma.copy()
    receipt = _channel(estimator_sigma=sigma, require_matched_gaussian=True)
    assert receipt["applied_sigma_per_component"] == sigma.tolist()
    receipt["estimator_sigma_per_component"][0] = 99
    np.testing.assert_array_equal(sigma, before)


@pytest.mark.parametrize("kwargs", [
    {"applied_sigma": float("nan")}, {"estimator_sigma": 0},
    {"applied_sigma": -0.1}, {"applied_sigma": []},
    {"applied_sigma": [[0.005]]}, {"applied_sigma": [0.005, 0.005], "estimator_sigma": [0.005] * 3},
    {"applied_sigma": [0.005], "estimator_sigma": [0.005] * 3},
    {"distribution": "none", "applied_sigma": 0.005},
    {"distribution": "gaussian", "applied_sigma": None},
    {"representation": "scalar", "applied_sigma_semantics": "complex_rms"},
    {"applied_sigma_semantics": "polar_magnitude"},
])
def test_invalid_or_ambiguous_sigma_declarations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _channel(**kwargs)


def test_shared_covariance_inherits_root_or_accepts_identical_scan_weights():
    sigma = np.array([0.001, 0.01])
    scans = [{"z_obs": [1.0, 0.4]}, {"z_obs": [1.1, 0.5], "sigma_z": sigma.copy()}]
    original = copy.deepcopy(scans)
    receipt = validate_shared_scada_covariance(sigma, scans)
    assert receipt["resolved_sigma_z"] == sigma.tolist()
    assert receipt["scan_count"] == 2
    assert receipt["scan_sigma_sources"] == [
        {"scan_position0": 0, "sigma_source": "root"}, {"scan_position0": 1, "sigma_source": "scan"}]
    assert scans[0] == original[0]
    np.testing.assert_array_equal(scans[1]["sigma_z"], original[1]["sigma_z"])
    assert not receipt["source_draws_verified"]
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("scans", [
    [{"sigma_z": [0.001, 0.1]}], [{"sigma_z": [0.001]}],
    [{"sigma_z": [0.001, 0]}], [{"sigma_z": None}],
    [{"sigma_z": [0.001, float("inf")]}], [{"sigma_z": 0.001}],
    [{"z_obs": [1]}], [{"z_obs": [1, float("nan")]}], [],
])
def test_scan_drift_invalid_values_and_mismatched_layout_fail(scans):
    with pytest.raises(ValueError):
        validate_shared_scada_covariance([0.001, 0.01], scans)


@pytest.mark.parametrize("sigma", [[], 0.01, [0.001, -0.01], [0.001, float("nan")]])
def test_invalid_root_covariance_is_rejected(sigma):
    with pytest.raises(ValueError):
        validate_shared_scada_covariance(sigma, [{}])
