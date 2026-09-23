"""IEEE57 dimensions/local voltage bases and same-observation gate comparison."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from scipy.stats import chi2

from scripts import audit_ieee14_hif_physical_sweep as sweep
from scripts import audit_ieee57_hif_physical_sweep as entry57
from three_phase_model.voltage_bases import IEEE14_VOLTAGE_BASE_PROFILE_ID, IEEE57_VOLTAGE_BASE_PROFILE_ID


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def actual57(tmp_path_factory):
    output = tmp_path_factory.mktemp("physical57") / "sweep"
    result = sweep.run_sweep(output, system="case57", load_scales=(.8,),
        branch_rows=(0, 28), phases=(2,), resistances_ohm=(100.,), seed=19)
    return output, result


def test_real57_has_491_channels_378_dof_and_both_local_resistance_bases(actual57):
    output, result = actual57
    assert result["complete"] and result["physical_hif_succeeded"] == 2
    assert result["physical_hif_failed"] == result["fault_wls_failures"] == 0
    config = json.loads((output/"experiment_config.json").read_text())
    assert config["contract"] == "ieee57_physical_hif_unfiltered_sweep_v1"
    assert (config["bus_count"], config["branch_count"], config["measurement_count"]) == (57, 80, 491)
    assert config["voltage_profile"] == IEEE57_VOLTAGE_BASE_PROFILE_ID
    assert len(config["eligible_branch_rows0"]) == 63
    assert config["wls"]["chi_square_alpha"] == .05
    assert config["wls"]["comparison_chi_square_alphas"] == [.01]
    for row in _rows(output/"cases.jsonl"):
        assert len(row["mean_measurement_vector"]) == 491
        assert row["local_base_kv_ll"] == (138 if row["branch_row0"] == 0 else 69)
        assert row["resistance_pu"] == pytest.approx(100 / (row["local_base_kv_ll"]**2 / 100))
        assert row["fault_current_a"] == pytest.approx(row["fault_voltage_ln_v"] / 100)
        assert row["pu_equivalence"]["passed"]
    parent = json.loads((output/"parents"/"parent_00_load_0.8"/"source_case.json").read_text())
    assert {int(row[0]): row[9] for row in parent["bus"]} == {bus: 138 if bus <= 17 else 69 for bus in range(1, 58)}
    for row in _rows(output/"wls_observations.jsonl"):
        metrics = row["wls"]
        assert metrics["dof"] == 378
        assert metrics["chi_square_threshold"] == pytest.approx(chi2.ppf(.95, 378))
        strict = metrics["chi_square_comparisons"]["0.01"]
        assert strict["chi_square_threshold"] == pytest.approx(chi2.ppf(.99, 378))
        assert strict["same_fitted_observation"]
        assert strict["alarm"] == (metrics["J"] >= strict["chi_square_threshold"] or metrics["max_normalized_residual"] >= 4)
        if strict["alarm"]:
            assert metrics["alarm"]


def test_generic_noise_sigma_and_same_observation_hashes(actual57):
    output, _ = actual57
    config = json.loads((output/"experiment_config.json").read_text())
    groups = {row["noise_group_id"]: row for row in _rows(output/"noise_groups.jsonl")}
    means = {row["case_id"]: row["mean_measurement_vector"] for name in ("cases", "controls")
             for row in _rows(output/f"{name}.jsonl")}
    for profile, values in config["noise"]["sigma_z"].items():
        assert len(values) == 491
        assert values[:57] == [.001] * 57
        assert values[57:] == [{"baseline": .01, "accuracy_005": .005, "accuracy_002": .002}[profile]] * 434
    for row in _rows(output/"wls_observations.jsonl"):
        group = groups[row["noise_group_id"]]
        unit, _ = sweep.standard_noise(*group["seed_sequence_entropy"], size=491)
        np.testing.assert_array_equal(unit, group["unit_noise"])
        observed = np.asarray(means[row["case_id"]]) + unit * np.asarray(config["noise"]["sigma_z"][row["noise_profile"]])
        assert sweep.numeric_hash(observed) == row["observed_sha256"]
    assert (output/"detection_by_voltage_resistance_alpha_0p01.csv").is_file()
    assert (output/"controls_summary_alpha_0p01.csv").is_file()


def test_secondary_alpha_reuses_one_fit_and_does_not_relax_normalized_gate(monkeypatch):
    calls = []

    def fit(*args, **kwargs):
        calls.append(True)
        return {"wls_objective": 430., "signed_normalized_residual": np.array([3.]), "dof": 378, "iterations": 4}

    monkeypatch.setattr(sweep, "build_wls_features", fit)
    result = sweep.wls_audit({}, [], [], chi_square_alpha=.05, comparison_alphas=(.01,))
    assert calls == [True]
    assert result["alarm"]
    assert not result["chi_square_comparisons"]["0.01"]["alarm"]
    result["max_normalized_residual"] = 4.
    assert sweep.rethreshold_wls(result, .01)["alarm"]
    assert calls == [True]


def test_failed_fit_remains_unavailable_under_secondary_alpha():
    failed = sweep.rethreshold_wls({"success": False, "alarm": None, "error": "failed"}, .01)
    assert failed["success"] is False and failed["alarm"] is None
    assert "J" not in failed and "chi_square_threshold" not in failed


def test_profiles_cannot_cross_systems_and_original_case_stays_unmodified(tmp_path):
    before = sweep.resolve_system("case57").load_case()["bus"].copy()
    configured, _, _ = sweep.configure_system("case57")
    np.testing.assert_array_equal(before, sweep.resolve_system("case57").load_case()["bus"])
    assert np.all(configured["bus"][:, 9] > 0)
    for system, profile in (("case57", IEEE14_VOLTAGE_BASE_PROFILE_ID), ("case14", IEEE57_VOLTAGE_BASE_PROFILE_ID)):
        target = tmp_path/system
        with pytest.raises(ValueError, match="incompatible"):
            sweep.run_sweep(target, system=system, voltage_profile=profile)
        assert not target.exists()


def test_cli_system_defaults_preserve14_and_select57(monkeypatch, tmp_path):
    calls = []

    def run(output, **kwargs):
        calls.append(kwargs)
        return {"complete": True, "physical_hif_failed": 0, "failed_physical_controls": [], "fault_wls_failures": 0}

    monkeypatch.setattr(sweep, "run_sweep", run)
    monkeypatch.setattr(entry57, "run_parallel_sweep", run)
    assert sweep.main(["--output-dir", str(tmp_path/"14")]) == 0
    assert entry57.main(["--output-dir", str(tmp_path/"57")]) == 0
    assert calls[0]["system"] == "case14" and calls[0]["seed"] == 20260918
    assert calls[1]["system"] == "case57" and calls[1]["seed"] == 20260919
    assert calls[1]["load_scales"] == [.8, 1.]
    assert calls[1]["phases"] == [1, 2, 3]
    assert calls[1]["resistances_ohm"] == list(sweep.RESISTANCES_OHM)
    assert calls[1]["chi_square_alpha"] is None  # system-specific .05 resolved by runner
    assert calls[1]["comparison_alphas"] is None  # .01 same-fit comparison resolved by runner
    assert calls[1]["workers"] == 8


def _partition_actual_fixture(actual57, path):
    original, result = actual57
    controls = _rows(original/"controls.jsonl")
    directories = []
    for index, branch in enumerate((0, 28)):
        directory = path/f"shard{index}"
        directory.mkdir(parents=True)
        config = json.loads((original/"experiment_config.json").read_text())
        config.update(selected_branch_rows0=[branch], expected_physical_hif_cases=1)
        sweep.write_json(directory/"experiment_config.json", config)
        receipt = dict(result, recorded_physical_hif_cases=1, expected_physical_hif_cases=1)
        sweep.write_json(directory/"summary.json", receipt)
        for name in ("cases", "controls", "wls_observations", "noise_groups"):
            rows = _rows(original/f"{name}.jsonl")
            if name == "noise_groups":
                rows = [row for row in rows if row["seed_sequence_entropy"][2] == branch]
            else:
                rows = [row for row in rows if row.get("branch_row0") == branch
                        or name == "controls" and row["kind"] == "healthy"]
            entry57._write_rows(directory/f"{name}.jsonl", rows)
        shutil.copytree(original/"parents", directory/"parents")
        directories.append(directory)
    return directories


def test_merge_deduplicates_parents_but_preserves_distinct_paired_observations(actual57, tmp_path):
    shards = _partition_actual_fixture(actual57, tmp_path/"parts")
    output = tmp_path/"merged"
    result = entry57.merge_shards(output, shards)
    assert result["complete"] and result["physical_hif_succeeded"] == 2
    assert result["operating_parent_count"] == result["healthy_physical_controls"] == 1
    assert result["no_fault_split_physical_controls"] == result["independent_noise_group_count"] == 2
    assert result["fault_noisy_observations_attempted"] == 6
    assert result["control_noisy_observations_attempted"] == 12
    assert result["by_voltage_resistance_and_noise"] == actual57[1]["by_voltage_resistance_and_noise"]
    assert result["chi_square_alpha_comparisons"] == actual57[1]["chi_square_alpha_comparisons"]
    assert len(_rows(output/"controls.jsonl")) == 3
    provenance = json.loads((output/"parent_provenance.json").read_text())
    assert len(provenance["parent_00_load_0.8"]["shard_model_directories"]) == 2


@pytest.mark.parametrize("corruption", ["overlapping_branches", "changed_seed", "changed_parent_mean", "changed_observation_hash",
    "missing_fault_observation", "missing_healthy_observation", "missing_split_control"])
def test_merge_rejects_noise_parent_or_observation_inconsistency(actual57, tmp_path, corruption):
    shards = _partition_actual_fixture(actual57, tmp_path/"parts")
    if corruption in ("overlapping_branches", "changed_seed"):
        path = shards[1]/"experiment_config.json"
        config = json.loads(path.read_text())
        config["selected_branch_rows0" if corruption == "overlapping_branches" else "seed"] = [0] if corruption == "overlapping_branches" else 123
        sweep.write_json(path, config)
    elif corruption == "changed_parent_mean":
        path = shards[1]/"controls.jsonl"
        rows = _rows(path)
        next(row for row in rows if row["kind"] == "healthy")["mean_measurement_vector"][0] += .01
        path.write_text("\n".join(sweep._json(row) for row in rows)+"\n")
    elif corruption == "missing_split_control":
        path = shards[1]/"controls.jsonl"
        rows = [row for row in _rows(path) if row["kind"] != "no_fault_split"]
        path.write_text("\n".join(sweep._json(row) for row in rows)+"\n")
    else:
        path = shards[1]/"wls_observations.jsonl"
        rows = _rows(path)
        if corruption.startswith("missing_"):
            kind = "hif" if corruption == "missing_fault_observation" else "healthy"
            rows.remove(next(row for row in rows if row["kind"] == kind))
        else:
            rows[0]["observed_sha256"] = "invalid"
        path.write_text("\n".join(sweep._json(row) for row in rows)+"\n")
    with pytest.raises(ValueError):
        entry57.merge_shards(tmp_path/"merged", shards)
    assert all((path/"cases.jsonl").is_file() for path in shards)
