from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from logical_topology.audit import evaluate_scenario
from logical_topology.estimation import estimate
from logical_topology.fit_cache import (NUMERICAL_SOURCES, REPO, VerifiedEstimatorCache,
                                        VerifiedSourceRun, file_sha256, legacy_hash)
from logical_topology.inventory import build_inventory
from logical_topology.measurements import build_measurement_inventory, expected_measurements, sample_measurements
from logical_topology.scenarios import solve_true_world, write_json
from psse_env.systems import resolve_system
from scripts.revalidate_logical_topology import copy_corpus_inputs


@pytest.fixture(scope="module")
def source_runs(tmp_path_factory):
    inventory = build_inventory("case14", split_buses=[])
    true = dict(inventory["normal_statuses"])
    physical = solve_true_world(resolve_system("case14").load_case(), inventory, true)
    assert physical["admitted"]
    sensors = build_measurement_inventory(inventory)
    observations = sample_measurements(expected_measurements(physical["operating_case"], inventory, true,
                                                            physical["solution"], sensors), sensors, noise=False)
    roots = {}
    for overlay in (False, True):
        source = tmp_path_factory.mktemp("old_fit_source")
        corpus = source / "corpus"
        reported = {**true, inventory["branches"][0]["device_id"]: 0}
        case = deepcopy(physical["operating_case"])
        case["branch"][0, 10] = 0
        parameter_error = None
        if overlay:
            parameter_error = {"branch_row0": 1, "factor": 2., "true_r_x": case["branch"][1, 2:4].tolist()}
            case["branch"][1, 2:4] *= 2
        for relative, payload in (("inventories/model.json", inventory), ("sensors/model.json", sensors),
                                  ("observations/model.json", observations), ("model_inputs/model.json", case),
                                  ("physical_audit/world.json", {**physical, "true_statuses": true})):
            write_json(corpus / relative, payload)
        row = {"scenario_id": "sample", "family": "topology+parameter" if overlay else "exclusion",
               "direction": "true_closed_model_open", "layout": "branch_status", "measurement_profile": "direct",
               "load_scale": 1., "parent_physical_root": "world", "physical_root_fingerprint": "scenario",
               "split": "development", "structural_split": "development", "hypothesis_cardinality": 1,
               "true_statuses": true, "physical_admission": {"admitted": True, "physics": physical["physics"]},
               "physical_audit_path": "physical_audit/world.json",
               "execution": {"inventory_path": "inventories/model.json", "measurement_inventory_path": "sensors/model.json",
                             "observations_path": "observations/model.json", "base_case_path": "model_inputs/model.json",
                             "current_statuses": reported}}
        if parameter_error:
            row["parameter_error"] = parameter_error
        manifest = {"rows": [row], "scenario_count": 1, "physical_world_count": 1, "physical_worlds_admitted": 1}
        write_json(corpus / "manifest.json", manifest)
        write_json(corpus / "config.json", {"system": "case14", "smoke": True})
        compact = evaluate_scenario(corpus, row)
        write_json(corpus / "row_audits" / "sample.json", compact)
        names = (*NUMERICAL_SOURCES, "logical_topology/runtime.py", "logical_topology/audit.py")
        hashes = {name: file_sha256(REPO / name) for name in names}
        for name in names:
            destination = source / "implementation_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
        write_json(source / "run_receipt.json", {"source_before": hashes, "source_after": hashes,
                                                "all_sources_unchanged_during_run": True, "all_rows_audited": True,
                                                "manifest_sha256": file_sha256(corpus / "manifest.json")})
        roots[overlay] = (source, row, case, inventory, sensors, observations, physical)
    return roots


def test_exact_numeric_fit_reuse_normalizes_arrays_and_authoritative_statuses(source_runs):
    path, row, case, inventory, sensors, observations, _ = source_runs[False]
    source = VerifiedSourceRun(path)
    calls = []
    cache = VerifiedEstimatorCache(source, row, estimator=lambda *args, **kwargs: calls.append(args))
    as_lists = json.loads(json.dumps({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in case.items()}))
    # These flags are immaterial as input seeds: the complete status vector
    # authoritatively replaces every BR_STATUS entry before compilation.
    for branch in as_lists["branch"]:
        branch[10] = 0
    result = cache(as_lists, inventory, row["true_statuses"], observations, sensors)
    assert result["numerical_fit_execution"]["reused"]
    assert not result["numerical_fit_execution"]["fresh_solve"]
    assert result["plausible"]
    assert calls == []
    assert "offline_true" not in json.dumps(result)
    receipt = cache.receipt()
    assert receipt["reused_fits"] == 1
    assert not receipt["old_candidate_decisions_or_certificates_reused"]
    assert receipt["lookups_provenance"][0]["source_fit_slots"]


def test_changed_values_covariance_parameters_statuses_or_budget_require_new_fits(source_runs):
    path, row, case, inventory, sensors, observations, _ = source_runs[False]
    calls = []

    def fresh(*args, **kwargs):
        calls.append(deepcopy((args, kwargs)))
        return estimate(*args, **kwargs)

    cache = VerifiedEstimatorCache(VerifiedSourceRun(path), row, estimator=fresh)
    changed_case = deepcopy(case)
    changed_case["branch"][1, 2] *= 1.01
    assert not cache(changed_case, inventory, row["true_statuses"], observations, sensors)["numerical_fit_execution"]["reused"]
    changed_obs = deepcopy(observations)
    changed_obs["values"][0] += .001
    assert not cache(case, inventory, row["true_statuses"], changed_obs, sensors)["numerical_fit_execution"]["reused"]
    changed_sensors = deepcopy(sensors)
    changed_sensors["covariance"][0][1] = changed_sensors["covariance"][1][0] = 1e-7
    changed_sensors["sensor_inventory_hash"] = legacy_hash({key: value for key, value in changed_sensors.items() if key != "sensor_inventory_hash"})
    changed_obs = {**observations, "sensor_inventory_hash": changed_sensors["sensor_inventory_hash"]}
    assert not cache(case, inventory, row["true_statuses"], changed_obs, changed_sensors)["numerical_fit_execution"]["reused"]
    changed_status = {**row["execution"]["current_statuses"], inventory["branches"][2]["device_id"]: 0}
    assert not cache(case, inventory, changed_status, observations, sensors)["numerical_fit_execution"]["reused"]
    assert not cache(case, inventory, row["true_statuses"], observations, sensors, max_nfev=101)["numerical_fit_execution"]["reused"]
    assert len(calls) == 5 and cache.receipt()["fresh_estimator_calls"] == 5


def test_true_parameter_fit_cannot_substitute_for_corrupted_current_model(source_runs):
    path, row, case, inventory, sensors, observations, physical = source_runs[True]
    cache = VerifiedEstimatorCache(VerifiedSourceRun(path), row)
    wrong_parameters = cache(case, inventory, row["true_statuses"], observations, sensors)
    physical_parameters = cache(physical["operating_case"], inventory, row["true_statuses"], observations, sensors)
    assert wrong_parameters["numerical_fit_execution"]["reused"]
    assert physical_parameters["numerical_fit_execution"]["reused"]
    assert wrong_parameters["plausible"] is False
    assert physical_parameters["plausible"] is True
    assert wrong_parameters["numerical_fit_execution"]["semantic_input_sha256"] != physical_parameters["numerical_fit_execution"]["semantic_input_sha256"]


def test_changed_numerical_source_disables_reuse_and_mutated_old_input_is_rejected(tmp_path, source_runs):
    path, row, case, inventory, sensors, observations, _ = source_runs[False]
    repo = tmp_path / "changed_repo"
    for name in NUMERICAL_SOURCES:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    source = VerifiedSourceRun(path, repo_root=repo)
    cache = VerifiedEstimatorCache(source, row)
    with (repo / "logical_topology/estimation.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# A changed numerical implementation must miss the old cache.\n")
    result = cache(case, inventory, row["true_statuses"], observations, sensors)
    assert not result["numerical_fit_execution"]["reused"]
    assert cache.receipt()["lookups_provenance"][-1]["reason"] == "numerical_source_hash_changed"
    damaged = tmp_path / "damaged_source"
    shutil.copytree(path, damaged)
    obs_path = damaged / "corpus" / row["execution"]["observations_path"]
    changed = json.loads(obs_path.read_text())
    changed["values"][0] += .2
    write_json(obs_path, changed)
    with pytest.raises(ValueError, match="Source execution input changed"):
        VerifiedEstimatorCache(VerifiedSourceRun(damaged), row)


def test_input_copy_excludes_old_audits_and_refuses_overwrite(tmp_path, source_runs):
    source = source_runs[False][0]
    before = {str(path.relative_to(source)): file_sha256(path) for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "verified"
    receipt = copy_corpus_inputs(source, output)
    assert not (output / "corpus/audits").exists()
    assert not (output / "corpus/row_audits").exists()
    assert receipt["source_bytes_preserved"]
    for name, digest in receipt["files_sha256"].items():
        assert file_sha256(output / "corpus" / name) == digest
    assert before == {str(path.relative_to(source)): file_sha256(path) for path in source.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        copy_corpus_inputs(source, output)
    with pytest.raises(ValueError, match="separate"):
        copy_corpus_inputs(source, source / "must_not_modify_old_run")
