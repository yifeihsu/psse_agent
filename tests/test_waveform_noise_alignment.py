from copy import deepcopy

import numpy as np
import pytest

from three_phase_nlm.measurement_noise import (
    add_scada_noise, add_voltage_phasor_noise, align_legacy_waveform_row,
    generated_noise_contract, scada_noise_sigma, scaled_sensor_sigma,
)


def _voltages():
    return [{"bus": "b1", "vln_pu": [1.03, 1.02, 1.01], "ang_deg": [4.0, -116.0, 124.0]}]


def _scan(index=0):
    return {"scan_index": index, "z_obs": [0.1] * 122, "z_clean": [0.09] * 122,
            "three_phase_voltages": _voltages(),
            "three_phase_branch_currents": [{"preserve": "existing noisy currents"}],
            "branch_current_sigma_pu": .001}


def _legacy_hif():
    return {**_scan(), "sigma_z": scada_noise_sigma().tolist(), "scans": [_scan(), _scan(1)]}


def _hif_meta(scale=1):
    return {"hif": {"generation": {"noise_scale": scale},
                    "branch_current_measurements": {"branch_current_sigma_pu": .001}}}


def _unbalance_meta():
    return {"imbalance": {"three_phase_branch_current_measurements": {
        "applied_noise_sigma_pu": .001, "branch_current_sigma_pu": .001}}}


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_standard_sensor_generation_rejects_nonpositive_or_nonfinite_noise(scale):
    with pytest.raises(ValueError):
        scada_noise_sigma(scale)
    with pytest.raises(ValueError):
        scaled_sensor_sigma(.005, scale, field="voltage")


def test_scaled_applied_noise_matches_published_sigmas_empirically():
    rng = np.random.default_rng(314159)
    sigma = scada_noise_sigma(1.7)
    clean = np.linspace(-2, 2, 122)
    standardized = np.asarray([(np.asarray(add_scada_noise(clean, rng, sigma)) - clean) / sigma
                               for _ in range(500)])
    assert abs(standardized.mean()) < .02
    assert abs(standardized.std() - 1) < .02
    rows = _voltages() * 3000
    before = deepcopy(rows)
    phase_sigma = scaled_sensor_sigma(.005, 1.7, field="phase_voltage")
    noisy = add_voltage_phasor_noise(rows, rng, phase_sigma)
    def phasors(values):
        return np.asarray([r["vln_pu"] for r in values]) * np.exp(
            1j * np.deg2rad(np.asarray([r["ang_deg"] for r in values])))
    delta = (phasors(noisy) - phasors(rows)) / phase_sigma
    assert rows == before
    for component in (delta.real, delta.imag):
        assert abs(component.mean()) < .04
        assert abs(component.std() - 1) < .04
    contract = generated_noise_contract(sigma, noise_scale=1.7, three_phase_sigma=phase_sigma,
                                        branch_current_sigma_pu=.0017)
    assert all(c["matched_gaussian"] for c in contract["channels"].values())


def test_legacy_hif_adds_only_missing_voltage_noise_and_fixes_scaled_weights():
    source = _legacy_hif()
    before = deepcopy(source)
    aligned = align_legacy_waveform_row(source, "hif", np.random.default_rng(22), legacy_metadata=_hif_meta(2))
    assert source == before
    assert aligned["z_obs"] == source["z_obs"]
    assert aligned["three_phase_branch_currents"] == source["three_phase_branch_currents"]
    assert aligned["three_phase_voltages"] != source["three_phase_voltages"]
    assert aligned["three_phase_voltages_clean"] == source["three_phase_voltages"]
    assert aligned["sigma_z"] == scada_noise_sigma(2).tolist()
    assert aligned["three_phase_sigma"] == .01
    assert aligned["branch_current_sigma_pu"] == .002
    for scan, original in zip(aligned["scans"], source["scans"]):
        assert scan["z_obs"] == original["z_obs"]
        assert scan["three_phase_branch_currents"] == original["three_phase_branch_currents"]
        assert scan["sigma_z"] == aligned["sigma_z"]
    assert aligned["three_phase_voltages"] == aligned["scans"][0]["three_phase_voltages"]


def test_known_voltage_only_hif_noise_does_not_invent_current_telemetry():
    source = _legacy_hif()
    for entry in [source, *source["scans"]]:
        entry.pop("three_phase_branch_currents")
        entry.pop("branch_current_sigma_pu")
    metadata = {"hif": {"generation": {"noise_scale": 1}}}
    result = align_legacy_waveform_row(source, "hif", np.random.default_rng(6), legacy_metadata=metadata)
    assert result["three_phase_sigma"] == .005
    assert result["z_obs"] == source["z_obs"]
    assert "three_phase_branch_currents" not in result
    assert "branch_current_sigma_pu" not in result
    assert set(result["noise_contract"]["channels"]) == {"scada", "three_phase_voltages"}


def test_legacy_unbalance_adds_scada_voltage_noise_and_preserves_current_noise():
    source = _scan()
    before = deepcopy(source)
    aligned = align_legacy_waveform_row(source, "three_phase_unbalance", np.random.default_rng(32),
                                       legacy_metadata=_unbalance_meta())
    assert source == before
    assert aligned["z_obs"] != source["z_obs"]
    assert aligned["z_clean"] == source["z_obs"]
    assert aligned["three_phase_voltages"] != source["three_phase_voltages"]
    assert aligned["three_phase_branch_currents"] == source["three_phase_branch_currents"]
    assert aligned["sigma_z"] == scada_noise_sigma().tolist()


def test_new_rows_are_idempotent_and_do_not_draw_noise_twice():
    rng = np.random.default_rng(45)
    source = align_legacy_waveform_row(_legacy_hif(), "hif", rng, legacy_metadata=_hif_meta())
    rng_before = deepcopy(rng.bit_generator.state)
    aligned = align_legacy_waveform_row(source, "hif", rng)
    assert aligned == source
    assert rng.bit_generator.state == rng_before
    aligned["scans"][0]["z_obs"][0] = 999
    assert source["scans"][0]["z_obs"][0] != 999


def test_explicit_stress_noise_rows_pass_without_legacy_upgrade():
    source = _legacy_hif()
    source["three_phase_sigma"] = .005
    for scan in source["scans"]:
        scan["three_phase_sigma"] = .005
        scan["sigma_z"] = source["sigma_z"].copy()
    rng = np.random.default_rng(54)
    rng_before = deepcopy(rng.bit_generator.state)
    assert align_legacy_waveform_row(source, "hif", rng) == source
    assert rng.bit_generator.state == rng_before


def test_unknown_legacy_noise_or_explicit_weight_drift_is_rejected():
    with pytest.raises(ValueError, match="source meta.json"):
        align_legacy_waveform_row(_legacy_hif(), "hif", np.random.default_rng(3))
    with pytest.raises(ValueError, match="noise_scale"):
        align_legacy_waveform_row(_legacy_hif(), "hif", np.random.default_rng(3), legacy_metadata={"hif": {}})
    meta = _unbalance_meta()
    meta["imbalance"]["three_phase_branch_current_measurements"]["applied_noise_sigma_pu"] = .002
    with pytest.raises(ValueError, match="sigmas differ"):
        align_legacy_waveform_row(_scan(), "unbalance", np.random.default_rng(3), legacy_metadata=meta)
    aligned = align_legacy_waveform_row(_legacy_hif(), "hif", np.random.default_rng(3), legacy_metadata=_hif_meta())
    aligned["three_phase_sigma"] = .007
    with pytest.raises(ValueError, match="matched Gaussian"):
        align_legacy_waveform_row(aligned, "hif", np.random.default_rng(3))


def test_generator_validation_rejects_false_gaussian_or_mismatching_weights(tmp_path):
    from Transmission.generate_measurements_imbalance import generate_dataset as imbalance
    with pytest.raises(ValueError, match="noise_scale"):
        imbalance(out_dir=str(tmp_path / "zero"), n_imbalance=0, n_no_error=0, seed=1,
                  load_scale_min=.9, load_scale_max=1.1, dirichlet_alpha=3, noise_scale=0)
    with pytest.raises(ValueError, match="must equal"):
        imbalance(out_dir=str(tmp_path / "mismatch"), n_imbalance=0, n_no_error=0, seed=1,
                  load_scale_min=.9, load_scale_max=1.1, dirichlet_alpha=3,
                  branch_current_noise_pu=.002, branch_current_sigma_pu=.001)
    assert not (tmp_path / "zero").exists()
    assert not (tmp_path / "mismatch").exists()
