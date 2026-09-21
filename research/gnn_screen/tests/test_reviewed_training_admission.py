"""Admission checks only: no optimizer, neural training, or corpus mutation."""
from __future__ import annotations

import copy
import json

import pytest

from mcp_server.matpower_server import _load_python_case
from research.gnn_screen.dataset import MEASUREMENT_CONVENTION, content_hash, jsonable
from research.gnn_screen import train as training


def _row(*, fault=True, parent="train-parent", split="train"):
    row = {"case": jsonable(_load_python_case("case14")), "z": [1.] * 14 + [0.] * 108,
        "measurement_sigma": [.001] * 14 + [.01] * 108, "measurement_kind": "observed",
        "noise_replicates": 1, "parent_id": parent, "window_id": f"{parent}:{'fault' if fault else 'healthy'}",
        "families": ["hif"] if fault else [], "split": split,
        "measurement_convention": MEASUREMENT_CONVENTION,
        "offline_metadata": {"scenario_profile": "reviewed_v1"}}
    row["offline_metadata"]["training_admission"] = {
        "contract": training.REVIEWED_TRAINING_ADMISSION_CONTRACT,
        "eligible": True, "kind": "fault_actionable" if fault else "healthy_completion",
        "scope": "executed_expert_prefix_only", "wls_alarm": fault, "expert_valid": True,
        "action_executed": True, "execution_success": True, "safe_finalize": not fault,
        "component_checks_passed": True,
        "thresholds": {"chi_square_alpha": .01, "normalized_residual": 4.},
        "measurement_sha256": content_hash(row["z"]), "case_sha256": content_hash(row["case"]),
        "sigma_sha256": content_hash(row["measurement_sigma"]),
    }
    return row


def _manifest(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_verified_observed_training_and_unfiltered_weak_evaluation(tmp_path):
    fault, healthy = _row(), _row(fault=False)
    weak = _row(parent="weak-test-parent", split="test")
    weak.update(measurement_kind="noiseless_mean", noise_replicates=5, noise_seed=88)
    weak["offline_metadata"].pop("training_admission")
    path = _manifest(tmp_path, [fault, healthy, weak])
    before = path.read_bytes()
    result = training.validate_reviewed_training_admission(path)
    assert result["reviewed_training_windows_checked"] == 2
    assert result["reviewed_fault_windows"] == result["reviewed_healthy_windows"] == 1
    assert not result["nontraining_rows_filtered"] and not result["full_physical_repair_validated"]
    assert path.read_bytes() == before


def test_training_entrypoint_rejects_raw_reviewed_means_before_graph_or_model_work(tmp_path, monkeypatch):
    fault = _row()
    fault.update(measurement_kind="noiseless_mean", noise_seed=88)
    path = _manifest(tmp_path, [fault, _row(fault=False)])
    monkeypatch.setattr(training, "prepare_corpus", lambda *args, **kwargs: pytest.fail("graph building must not run"))
    with pytest.raises(ValueError, match="fixed observed replica"):
        training.train(path, tmp_path / "never_created")
    assert not (tmp_path / "never_created").exists()


@pytest.mark.parametrize("field,value", [
    ("eligible", False), ("expert_valid", False), ("action_executed", False),
    ("execution_success", False), ("wls_alarm", False),
    ("scope", "full_physical_repair"), ("contract", "unverified"),
])
def test_false_or_ambiguous_prefix_evidence_is_rejected(tmp_path, field, value):
    fault = _row()
    fault["offline_metadata"]["training_admission"][field] = value
    with pytest.raises(ValueError):
        training.validate_reviewed_training_admission(_manifest(tmp_path, [fault, _row(fault=False)]))


@pytest.mark.parametrize("which", ["measurement", "case", "sigma"])
def test_observed_values_model_and_covariance_remain_bound(tmp_path, which):
    fault = _row()
    if which == "measurement":
        fault["z"][20] += .1
    elif which == "case":
        fault["case"]["branch"][0][2] *= 1.1
    else:
        fault["measurement_sigma"][20] *= 2
    with pytest.raises(ValueError, match=f"{which}_sha256"):
        training.validate_reviewed_training_admission(_manifest(tmp_path, [fault, _row(fault=False)]))


def test_global_alarm_is_not_enough_for_mixed_faults(tmp_path):
    fault = _row()
    fault["families"] = ["hif", "measurement"]
    fault["offline_metadata"]["training_admission"]["component_checks_passed"] = False
    with pytest.raises(ValueError, match="all component checks"):
        training.validate_reviewed_training_admission(_manifest(tmp_path, [fault, _row(fault=False)]))


@pytest.mark.parametrize("field,value", [("wls_alarm", True), ("safe_finalize", False)])
def test_healthy_controls_need_quiet_safe_completion(tmp_path, field, value):
    healthy = _row(fault=False)
    healthy["offline_metadata"]["training_admission"][field] = value
    with pytest.raises(ValueError, match="healthy control"):
        training.validate_reviewed_training_admission(_manifest(tmp_path, [_row(), healthy]))


@pytest.mark.parametrize("fault", [True, False])
def test_reviewed_selection_cannot_be_claimed_trainable_without_faults_and_controls(tmp_path, fault):
    with pytest.raises(ValueError, match="nonempty fault and healthy"):
        training.validate_reviewed_training_admission(_manifest(tmp_path, [_row(fault=fault)]))


def test_legacy_input_does_not_acquire_new_admission_requirement(tmp_path):
    row = _row()
    row["offline_metadata"] = {"scenario_profile": "legacy_v1"}
    row.update(measurement_kind="noiseless_mean", noise_seed=91)
    result = training.validate_reviewed_training_admission(_manifest(tmp_path, [row]))
    assert result["reviewed_training_windows_checked"] == 0


def test_parent_group_cannot_cross_training_and_evaluation(tmp_path):
    with pytest.raises(ValueError, match="parent leakage"):
        training.validate_reviewed_training_admission(_manifest(tmp_path, [_row(), _row(fault=False, split="validation")]))


def test_observed_noise_group_must_be_lineage_not_fresh_noise_input(tmp_path):
    fault = _row()
    fault["noise_group_id"] = "source-paired-group"
    with pytest.raises(ValueError, match="no post-selection noise redraw"):
        training.validate_reviewed_training_admission(_manifest(tmp_path, [fault, _row(fault=False)]))
