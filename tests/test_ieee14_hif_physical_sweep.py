"""Sweep completeness, exact noisy observations, and retained failures."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import audit_ieee14_hif_physical_sweep as sweep


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def actual_sweep(tmp_path_factory):
    path = tmp_path_factory.mktemp("sweep") / "new_run"
    result = sweep.run_sweep(path, load_scales=(1.,), branch_rows=(0, 10),
        phases=(1,), resistances_ohm=(100., 500.), seed=18)
    return path, result


def test_actual_sweep_keeps_quiet_faults_and_explicit_physical_denominators(actual_sweep):
    path, result = actual_sweep
    assert result["complete"]
    assert result["expected_physical_hif_cases"] == result["recorded_physical_hif_cases"] == 4
    assert result["physical_hif_succeeded"] == 4
    assert result["fault_noisy_observations_attempted"] == 12
    assert result["control_noisy_observations_attempted"] == 12
    assert result["healthy_physical_controls"] == 1
    assert result["no_fault_split_physical_controls"] == 2
    assert result["parents_passed"] == 1
    faults = _rows(path/"cases.jsonl")
    assert {row["local_base_kv_ll"] for row in faults} == {69., 13.8}
    assert all(row["pu_equivalence"]["passed"] for row in faults)
    for row in faults:
        assert row["fault_current_a"] == pytest.approx(row["fault_voltage_ln_v"] / row["resistance_ohm"])
        assert row["fault_power_mw"] == pytest.approx(row["fault_current_a"]**2 * row["resistance_ohm"] / 1e6)
    observed = _rows(path/"wls_observations.jsonl")
    assert any(row["kind"] == "hif" and row["wls"]["alarm"] is False for row in observed)
    assert any(row["kind"] == "hif" and row["wls"]["alarm"] is True for row in observed)
    for metrics in result["parents"][0]["noiseless_wls"].values():
        assert metrics["success"] and not metrics["alarm"]


def test_saved_noise_and_sigma_reconstruct_every_tested_observation(actual_sweep):
    path, _ = actual_sweep
    means = {row["case_id"]: row["mean_measurement_vector"] for name in ("cases", "controls")
             for row in _rows(path/f"{name}.jsonl")}
    groups = {row["noise_group_id"]: row for row in _rows(path/"noise_groups.jsonl")}
    assert len(groups) == 2  # independent line groups, not 12 independent accuracy/R trials
    config = json.loads((path/"experiment_config.json").read_text())
    observed = _rows(path/"wls_observations.jsonl")
    for row in observed:
        group = groups[row["noise_group_id"]]
        regenerated, entropy = sweep.standard_noise(*group["seed_sequence_entropy"])
        assert entropy == group["seed_sequence_entropy"]
        np.testing.assert_array_equal(regenerated, group["unit_noise"])
        sigma = np.asarray(config["noise"]["sigma_z"][row["noise_profile"]])
        tested_z = np.asarray(means[row["case_id"]]) + regenerated * sigma
        assert sweep.numeric_hash(tested_z) == row["observed_sha256"]
        assert row["sigma_vm_pu"] == .001
        assert row["sigma_power_pu"] == sigma[14]
        assert row["wls"]["alarm"] == (row["wls"]["J"] >= row["wls"]["chi_square_threshold"]
            or row["wls"]["max_normalized_residual"] >= 4)
    assert config["wls"]["chi_square_alpha"] == .01


def test_physical_failure_stays_in_expected_population_without_negative_wls(tmp_path, monkeypatch):
    original = sweep._snapshot

    def fail_one_resistance(build, *, hif=None):
        if hif and hif.get("enabled", True) and hif.get("resistance_ohm") == 500:
            raise RuntimeError("deliberate unavailable physical solve")
        return original(build, hif=hif)

    monkeypatch.setattr(sweep, "_snapshot", fail_one_resistance)
    path = tmp_path/"failed_case"
    result = sweep.run_sweep(path, load_scales=(1.,), branch_rows=(0,),
        phases=(1,), resistances_ohm=(100., 500.))
    assert result["complete"]
    assert result["recorded_physical_hif_cases"] == 2
    assert result["physical_hif_succeeded"] == result["physical_hif_failed"] == 1
    assert result["fault_noisy_observations_expected"] == 6
    assert result["fault_noisy_observations_attempted"] == 3
    failed = [row for row in _rows(path/"cases.jsonl") if not row["physical_success"]]
    assert len(failed) == 1 and "deliberate unavailable" in failed[0]["error"]
    assert all(row["case_id"] != failed[0]["case_id"] for row in _rows(path/"wls_observations.jsonl"))


def test_failed_wls_is_unavailable_never_quiet_or_zero_energy(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unobservable fixture")
    monkeypatch.setattr(sweep, "build_wls_features", fail)
    row = sweep.wls_audit({}, [], [])
    assert row["success"] is False and row["alarm"] is None
    assert "J" not in row and "unobservable fixture" in row["error"]


def test_output_must_be_fresh_and_existing_artifact_is_unchanged(tmp_path):
    existing = tmp_path/"old_receipt"
    existing.mkdir()
    (existing/"receipt.json").write_text('{"historical":true}')
    with pytest.raises(FileExistsError, match="fresh"):
        sweep.run_sweep(existing)
    assert (existing/"receipt.json").read_text() == '{"historical":true}'


@pytest.mark.parametrize("kwargs", [{"branch_rows": (13,)}, {"noise_replicates": 0},
    {"load_scales": (0.,)}, {"resistances_ohm": (-1.,)}, {"phases": (4,)}])
def test_invalid_plan_rejected_before_writing(tmp_path, kwargs):
    target = tmp_path/"invalid"
    with pytest.raises(ValueError):
        sweep.run_sweep(target, **kwargs)
    assert not target.exists()
