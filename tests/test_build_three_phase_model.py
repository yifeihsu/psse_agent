"""Exercise the complete model build and its reference-gated disturbance stage."""
import json
from unittest.mock import patch

from scripts.build_three_phase_model import build_and_validate


def test_complete_model_build_with_circuit_unbalance_and_restoration(tmp_path):
    out = tmp_path / "model"
    result = build_and_validate(out)
    assert result["passed"]
    assert result["counts"]["external_phase_node_count"] == 171
    assert result["reports"]["unbalance"]["maximum_negative_sequence_voltage_ratio"] > .01
    assert result["reports"]["restoration"]["maximum_voltage_rectangular_error_pu"] < 1e-8
    manifest = json.loads((out / "build_manifest.json").read_text())
    assert manifest["validation_performed"] and manifest["validation_passed"]
    assert len(json.loads((out / "measurements.json").read_text())["measurement_vector"]) == 491
    assert (out / "UnbalanceExample.dss").is_file()


def test_failed_independent_reference_prevents_disturbance_generation(tmp_path):
    out = tmp_path / "model"
    with patch("scripts.build_three_phase_model.compare_matpower_reference", return_value={"passed": False}), patch(
        "scripts.build_three_phase_model.redistribute_load"
    ) as redistribute:
        result = build_and_validate(out, matpower_reference_dir=tmp_path)
    assert not result["passed"]
    redistribute.assert_not_called()
    manifest = json.loads((out / "build_manifest.json").read_text())
    assert not manifest["validation_passed"]
    assert not (out / "unbalance_measurements.json").exists()
    assert "unbalance_validation_report.json" not in manifest["validation_files"]


def test_failed_baseline_prevents_reference_or_disturbance_generation(tmp_path):
    with patch("scripts.build_three_phase_model.validate_model", return_value={
        "passed": False, "checks": {"deliberate_rejection": {"passed": False}},
        "failed_checks": ["deliberate_rejection"],
    }), patch("scripts.build_three_phase_model.redistribute_load") as redistribute, patch(
        "scripts.build_three_phase_model.compare_matpower_reference"
    ) as cross:
        result = build_and_validate(tmp_path / "model", matpower_reference_dir=tmp_path)
    assert not result["passed"]
    redistribute.assert_not_called()
    cross.assert_not_called()
