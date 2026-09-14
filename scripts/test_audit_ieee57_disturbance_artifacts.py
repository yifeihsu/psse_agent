"""Negative tests for the independent artifact acquisition boundary."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.audit_ieee57_disturbance_artifacts import (
    _finite_array,
    _under,
    _validate_observations,
    audit_artifacts,
)


def _fixture():
    registry = {"buses": [{"external_bus": index + 1, "row0": index, "dss_bus": f"b{index+1}"} for index in range(2)],
                "branches": [{"asset_id": "case:branch:1", "branch_row0": 0, "from_bus": 1, "to_bus": 2}]}
    observed = {"measurement_vector": [1.0] * 10,
                "three_phase_voltages": [{**bus, "vln_pu_rect": [[1.0, 0.0]] * 3} for bus in registry["buses"]],
                "three_phase_branch_currents": [{**registry["branches"][0], "i_from_pu_rect": [[0.1, 0.0]] * 3,
                                                  "i_to_pu_rect": [[-0.1, 0.0]] * 3}]}
    for row in observed["three_phase_voltages"]:
        row["bus"] = row.pop("dss_bus")
    profiles = {"nominal": {"voltage_sigma_pu": 1e-4, "current_sigma_pu": 1e-3},
                "precision_sensitivity": {"voltage_sigma_pu": 1e-5, "current_sigma_pu": 1e-4}}
    return {name: copy.deepcopy(observed) for name in ("exact", "noisy_nominal", "noisy_precision_sensitivity")}, registry, profiles


def test_acquisition_allowlist_accepts_fixed_external_bindings_without_mutation():
    payload, registry, profiles = _fixture()
    before = copy.deepcopy(payload)
    arrays = _validate_observations(payload, registry, profiles)
    assert arrays["z"].shape == (10,)
    assert payload == before


@pytest.mark.parametrize("kind", ["truth_key", "hidden_bus", "hidden_line", "extra_injection", "wrong_order", "wrong_phase_shape"])
def test_acquisition_rejects_truth_hidden_aliases_and_wrong_mapping(kind):
    payload, registry, profiles = _fixture()
    exact = payload["exact"]
    if kind == "truth_key":
        exact["truth"] = {"phase": 1}
    elif kind == "hidden_bus":
        exact["three_phase_voltages"][0]["bus"] = "hif_b0001_001_bus"
    elif kind == "hidden_line":
        exact["three_phase_branch_currents"][0]["asset_id"] = "Line.hif_b0001_from"
    elif kind == "extra_injection":
        exact["three_phase_branch_currents"][0]["fault_resistance"] = 0.1
    elif kind == "wrong_order":
        exact["three_phase_voltages"].reverse()
    else:
        exact["three_phase_voltages"][0]["vln_pu_rect"].append([1.0, 0.0])
    with pytest.raises(ValueError):
        _validate_observations(payload, registry, profiles)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "1.0"])
def test_acquisition_requires_finite_numeric_measurements(invalid):
    with pytest.raises(ValueError):
        _finite_array([1.0, invalid], (2,))


def test_precision_sensitivity_must_share_scada_and_standardized_phase_noise():
    payload, registry, profiles = _fixture()
    payload["noisy_precision_sensitivity"]["measurement_vector"][0] += 0.01
    with pytest.raises(ValueError, match="SCADA"):
        _validate_observations(payload, registry, profiles)
    payload, registry, profiles = _fixture()
    payload["noisy_nominal"]["three_phase_voltages"][0]["vln_pu_rect"][0][0] += 0.001
    with pytest.raises(ValueError, match="Paired precision"):
        _validate_observations(payload, registry, profiles)


def test_paths_cannot_escape_the_audited_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "external.txt").write_text("outside")
    with pytest.raises(ValueError, match="escapes"):
        _under(run, "../external.txt")


def test_incomplete_run_cannot_mint_an_audit_receipt(tmp_path):
    (tmp_path / "experiment_config.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        audit_artifacts(tmp_path)
    assert not (tmp_path / "artifact_audit.json").exists()


def test_existing_audit_evidence_is_not_overwritten(tmp_path):
    report = tmp_path / "artifact_audit.json"
    report.write_text("existing audit evidence")
    with pytest.raises(FileExistsError):
        audit_artifacts(tmp_path)
    assert report.read_text() == "existing audit evidence"
