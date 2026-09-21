from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Harmonics.ieee14_verification import BASE_MVA, BRANCH, BUS
from Transmission import generate_measurements as classical
from Transmission import generate_multi_error_measurements as mixed
from Transmission.generate_hse_traces import build_trace


def test_zero_harmonic_clean_scada_matches_fundamental_including_all_branch_flows():
    trace = build_trace(9, 0.0, 20260916)
    expected = classical.compute_measurements_pu({
        "bus": BUS.copy(), "branch": BRANCH.copy(), "baseMVA": BASE_MVA,
    })
    np.testing.assert_allclose(trace["z_scada_true"], expected, rtol=0, atol=1e-11)
    assert np.max(np.abs(expected[42:])) > 0.1  # This control used to have zero branch flows.


def test_fresh_harmonic_noise_matches_serialized_component_and_scada_sigmas():
    scada_noise, phasor_noise = [], []
    for seed in range(100, 140):
        trace = build_trace(9, 0.15, seed)
        scada_noise.extend((np.array(trace["z_scada_meas"]) - trace["z_scada_true"]) / trace["sigma_z"])
        for rows in trace["harmonic_phasors"].values():
            for item in rows:
                assert item["sigma_semantics"] == "per_component"
                assert item["sigma"] == pytest.approx(1e-4 / np.sqrt(2))
                phasor_noise.extend((np.array(item["V_complex_noisy"]) - item["V_complex_true"]) / item["sigma"])
    for noise in (scada_noise, phasor_noise):
        assert abs(np.mean(noise)) < 0.06
        assert 0.95 < np.std(noise) < 1.05


def test_classic_harmonic_adapter_keeps_component_semantics():
    row = classical.make_harmonic_anomaly_record(np.random.default_rng(19))
    assert len(row["sigma_z"]) == 122
    assert row["sigma_z"][:14] == [0.001] * 14
    assert row["sigma_z"][14:] == [0.01] * 108
    for item in row["harmonic_measurements"]:
        assert item["sigma_semantics"] == "per_component"
        assert item["sigma"] == pytest.approx(item["sigma_complex_rms"] / np.sqrt(2))


def test_mixed_meter_overlay_preserves_one_original_gaussian_draw(monkeypatch, tmp_path):
    case = classical.load_case("14")
    index_map = classical.make_index_map(14, 20)
    sigma = classical.sigma_vector(index_map)
    truth = classical.compute_measurements_pu(case)
    observed = truth + np.random.default_rng(32).normal(0, sigma)
    base = {"z_true": truth.tolist(), "z_obs": observed.tolist(),
            **classical.scada_noise_fields(index_map),
            "label": {"error_type": "parameter_error"}, "op_point": {}}
    monkeypatch.setattr(mixed, "_make_component_factories", lambda *_args, **_kwargs: {"parameter_error": lambda: base})
    row = mixed.make_multi_error_record(np.random.default_rng(33), case, index_map, tmp_path,
                                       ["measurement", "parameter"], scans=2,
                                       load_scale_min=0.9, load_scale_max=1.0)
    np.testing.assert_array_equal(row["sigma_z"], sigma)
    restored = np.array(row["z_obs"])
    label = next(item for item in row["label"]["errors"] if item["error_type"] == "measurement_error")
    if "index" in label:
        restored[label["index"]] -= label["amplitude"]
    else:
        restored[label["indices"]] -= label["amplitudes"]
    np.testing.assert_allclose(restored, observed, rtol=0, atol=1e-15)
    snapshot = row["verification_snapshots"]["post_measurement_correction"]
    np.testing.assert_array_equal(snapshot["z_obs"], observed)
    np.testing.assert_array_equal(snapshot["sigma_z"], sigma)
    assert row["verification_snapshots"]["post_parameter_correction"]["sigma_z_policy"] == "preserve_current_sigma_z"


def test_physical_parameter_topology_verification_has_full_noise_and_own_layout(tmp_path):
    case = classical.load_case("14")
    index_map = classical.make_index_map(14, 20)
    row = mixed.make_multi_error_record(np.random.default_rng(20260916), case, index_map, Path(tmp_path),
                                       ["parameter", "topology"], scans=2,
                                       load_scale_min=0.9, load_scale_max=1.0)
    assert row is not None
    noise = (np.array(row["z_true_full_model"]) - row["z_clean_full_model"]) / row["sigma_z_full_model"]
    assert 0.8 < np.std(noise) < 1.3  # Reject the previous half-sigma draw.
    assert "not independent" in row["verification_noise_dependency"]
    assert len(row["sigma_z_scans"]) == len(row["z_scans"][0])
    assert len(row["sigma_z"]) == len(row["z_obs"]) == 122
    for snapshot in row["verification_snapshots"].values():
        assert len(snapshot["sigma_z"]) == len(snapshot["z_obs"])
        np.testing.assert_array_equal(snapshot["sigma_z"], row["sigma_z_full_model"])


def test_materialized_verification_snapshot_rejects_missing_or_wrong_covariance():
    for sigma in (None, [0.001], [0.001, 0]):
        with pytest.raises(ValueError, match="matching positive sigma_z"):
            mixed._stage_snapshot(case_path=None, remaining_families=[], note="test",
                                  z_obs=[1.0, 0.2], sigma_z=sigma)
