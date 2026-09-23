from __future__ import annotations

import copy
import json
import sys
from argparse import Namespace

import numpy as np
import pytest

from scripts import evaluate_hif_measurement_recovery as experiment


def test_observable_scan_removes_truth_and_does_not_alias_original():
    source = {
        "scan_index": 1, "z_obs": [1.0, 2.0], "op_point": {"load_scale": 0.9},
        "three_phase_branch_currents": [{"observed": [1.0]}],
        "z_clean": [9.0, 9.0], "z_true": [8.0, 8.0],
        "three_phase_branch_currents_clean": [{"hidden": [9.0]}],
        "label": {"branch_row0": 12}, "hidden_truth": {"target": 0},
        "nlm_diagnostic": {"detected_top1": True},
    }
    result = experiment.observable_scan(source)
    assert set(result) == {"scan_index", "z_obs", "op_point", "three_phase_branch_currents"}
    result["z_obs"][0] = 77.0
    result["op_point"]["load_scale"] = 1.2
    result["three_phase_branch_currents"][0]["observed"][0] = 5.0
    assert source["z_obs"] == [1.0, 2.0]
    assert source["op_point"]["load_scale"] == 0.9
    assert source["three_phase_branch_currents"][0]["observed"] == [1.0]


def test_audit_rejects_off_target_write_despite_successful_target_repair():
    original = np.array([1.0, 2.0, 3.0])
    active = original.copy()
    active[1] += 10.0
    proposed = original.copy()
    proposed[2] += 1e-12  # Even a small unsolicited write violates preservation.
    decision = {"proposed_measurements": proposed, "recovery_supported": True,
                "candidate_indices": [1, 2]}
    audited = experiment.audit_decision(decision, active, original, original, np.ones(3), 1)
    assert audited["target_error_to_noiseless_sigma"] == 0
    assert audited["off_target_write_count"] == 1
    assert not audited["exact_write_support"]
    assert not audited["success"]


def test_audit_does_not_accept_quiet_but_unrepaired_true_target():
    original = np.zeros(3)
    active = original.copy()
    active[1] = 10.0
    decision = {"proposed_measurements": active.copy(), "recovery_supported": True,
                "candidate_indices": []}
    audited = experiment.audit_decision(decision, active, original, original, np.ones(3), 1)
    assert not audited["success"]
    assert audited["target_error_to_noiseless_sigma"] == 10.0
    assert audited["off_target_write_count"] == 0


def test_summary_keeps_failed_or_incomplete_roots_in_planned_denominators():
    roots = [
        {"root_key": "failed", "fit_success": False, "error": "fit_failed"},
        {"root_key": "partial", "fit_success": True, "error": "prediction_audit_failed"},
    ]
    summary = experiment.summarize(roots, [], {"persistent_targets": 2})
    assert summary["attempted_root_count"] == 2
    assert summary["expected_transient_trace_count"] == 488
    assert summary["unexecuted_transient_traces_counted_as_failure"] == 488
    assert summary["expected_persistent_trace_count"] == 4
    assert summary["unexecuted_persistent_traces_counted_as_failure"] == 4
    assert summary["event_only_evaluated_root_count"] == 0
    assert summary["event_only_false_alarm_root_count"] == 0


def _run_stubbed_pilot(tmp_path, monkeypatch, *, fail_first_persistent=False):
    size = 122
    original = np.ones(size)
    effect = np.zeros(size)
    effect[3] = 0.1  # Observable max-effect target; min-effect target is channel zero.
    scans = []
    for scan_index in range(4):
        scans.append({
            "scan_index": scan_index, "z_obs": (original + 0.1 * scan_index).tolist(),
            "z_clean": (original + 0.1 * scan_index).tolist(),
            "three_phase_voltages": [{"bus": 1, "vln_pu": [1, 1, 1]}],
            "three_phase_branch_currents": [{"branch_row0": 0, "current": [1, 2, 3]}],
            "three_phase_branch_currents_clean": [{"hidden": True}],
            "op_point": {"load_scale": 1.0 + scan_index * 0.05},
            "topology_id": "fixture", "branch_current_sigma_pu": 0.001,
        })
    row = {"id": "fixture", "scans": scans, "sigma_z": [0.01] * size,
           "label": {"split_ratio": 0.4, "r_hif_pu": 100, "branch_row0": 0, "phase": "A"},
           "z_true": [-1000] * size}
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(row) + "\n")
    fits = []
    predictions = []

    def fake_fit(history, sigma, args):
        fits.append(copy.deepcopy(history))
        if fail_first_persistent and len(fits) == 2:
            return {"success": False, "error": "deliberate_stress_failure"}
        return {"success": True, "candidate_branch_row0": 0, "parameter_identifiable": True,
                "estimated": {"alpha_from_from_bus": 0.4, "r_hif_pu": 100, "phase": "A"}}

    def fake_prediction(fit, target, snapshot_id):
        predictions.append(copy.deepcopy(target))
        return {"predicted_hif_measurements": original.tolist(),
                "predicted_base_measurements": (original - effect).tolist(),
                "measurement_effect": effect.tolist(), "prediction_lower": original.tolist(),
                "prediction_upper": original.tolist()}

    monkeypatch.setattr(experiment, "fit_history", fake_fit)
    # This fixture exercises trial/repair logic with algebraic arrays, not a
    # generated sensor distribution. Preparation has separate strict tests.
    monkeypatch.setattr(experiment, "prepare_trial_row", lambda row, **kwargs: (
        copy.deepcopy(row), {"version": experiment.NOISE_PREPARATION_VERSION, "fixture": "logic_only_mock"}))
    monkeypatch.setattr(experiment, "prediction", fake_prediction)
    monkeypatch.setattr(experiment, "wls_metrics", lambda *args, **kwargs: {"no_material_anomaly": True})
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *args, **kwargs: "test-head\n")
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["evaluate_hif_measurement_recovery.py", "--samples", str(samples),
                        "--output-dir", str(output), "--history-scans", "3", "--persistent-targets", "2"])
    experiment.main()
    root_dir = next((output / "roots").iterdir())
    traces = [json.loads(line) for line in (root_dir / "traces.jsonl").read_text().splitlines()]
    return row, fits, predictions, traces, json.loads((output / "summary.json").read_text())


def test_persistent_stress_corrupts_every_fitting_scan_and_active_snapshot(tmp_path, monkeypatch):
    row, fits, predictions, traces, summary = _run_stubbed_pilot(tmp_path, monkeypatch)
    assert len(fits) == 3
    for history in fits:
        assert [scan["scan_index"] for scan in history] == [1, 2, 3]
        for scan in history:
            assert "z_clean" not in scan
            assert "three_phase_branch_currents_clean" not in scan
    for fit_index, target_index in [(1, 3), (2, 0)]:
        for actual, initial in zip(fits[fit_index], fits[0]):
            delta = np.array(actual["z_obs"]) - initial["z_obs"]
            assert np.flatnonzero(delta).tolist() == [target_index]
            assert delta[target_index] == pytest.approx(0.1)
            assert actual["three_phase_branch_currents"] == initial["three_phase_branch_currents"]
    assert all(target["scan_index"] == 0 for target in predictions)
    assert all(target["z_obs"] == row["scans"][0]["z_obs"] for target in predictions)
    persistent = [trace for trace in traces if trace["mode"] == "persistent"]
    assert [trace["measurement_index"] for trace in persistent] == [3, 0]
    assert all(trace["physical_model"]["changed_indices"] == [trace["measurement_index"]] for trace in persistent)
    assert all(trace["physical_model"]["success"] for trace in persistent)
    assert len([trace for trace in traces if trace["mode"] == "transient"]) == 244
    assert summary["attempted_root_count"] == 1
    assert summary["physical_fault_removed"] is False


def test_failed_persistent_target_remains_in_denominator_and_next_target_runs(tmp_path, monkeypatch):
    _, fits, _, traces, summary = _run_stubbed_pilot(tmp_path, monkeypatch, fail_first_persistent=True)
    assert len(fits) == 3
    persistent = [trace for trace in traces if trace["mode"] == "persistent"]
    assert len(persistent) == 2
    assert persistent[0]["status"] == "failed"
    assert persistent[0]["true_overlap"]
    assert not persistent[0]["physical_model"]["success"]
    assert persistent[0]["physical_model"]["off_target_write_count"] == 0
    assert not persistent[0]["diagnosis_correct"]
    assert persistent[1]["status"] == "completed"
    assert persistent[1]["physical_model"]["success"]
    assert summary["strata"]["persistent"]["trace_count"] == 2
    assert summary["strata"]["persistent"]["physical_model_meter_recovery"] == 1


def test_real_dss_branch_net_measurement_convention_preserves_physical_capacitor():
    dss = pytest.importorskip("opendssdirect")
    from pypower.api import case14
    from three_phase_nlm.hif_parameter_estimator import _resolve_model_dir, _simulate_base

    op_point = {"load_scale": 0.91, "bus_load_scales": {"b2": 0.97}}
    original_op_point = copy.deepcopy(op_point)
    canonical_case = case14()
    original_case_bus = canonical_case["bus"].copy()
    baseline = _simulate_base(_resolve_model_dir(None, "case14"), op_point=op_point)
    observed = np.asarray(baseline["z"], dtype=float)
    original_observed = observed.copy()
    sigma = np.r_[np.full(14, 0.001), np.full(108, 0.01)]
    original_sigma = sigma.copy()

    def physical_capacitors():
        result = {}
        for name in dss.Capacitors.AllNames():
            dss.Capacitors.Name(name)
            result[name] = dss.Capacitors.kvar()
        return result

    original_capacitors = physical_capacitors()
    assert original_capacitors and sum(original_capacitors.values()) > 0
    args = Namespace(chi_square_alpha=0.05, detection_sigma=5.0)
    canonical = experiment.wls_metrics(observed, sigma, args, exported_injections=False,
                                       input_role="noiseless_model_prediction")
    branch_net = experiment.wls_metrics(observed, sigma, args, input_role="noiseless_model_prediction")

    assert canonical["converged"] and branch_net["converged"]
    assert canonical["chi_square_alarm"]
    assert not canonical["no_material_anomaly"]
    assert branch_net["no_material_anomaly"]
    assert not branch_net["chi_square_alarm"]
    assert not branch_net["normalized_residual_alarm"]
    assert branch_net["measurement_convention"] == "branch_net_injections_including_capacitors"
    assert branch_net["input_role"] == "noiseless_model_prediction"
    assert "not_postrepair" in branch_net["statistical_interpretation"]
    assert canonical["measurement_convention"] == "canonical_matpower_injections"
    np.testing.assert_array_equal(observed, original_observed)
    np.testing.assert_array_equal(sigma, original_sigma)
    np.testing.assert_array_equal(canonical_case["bus"], original_case_bus)
    np.testing.assert_array_equal(case14()["bus"], original_case_bus)
    assert op_point == original_op_point
    assert physical_capacitors() == original_capacitors


def test_hif_fitting_rejects_covariance_override_before_using_telemetry():
    import pytest
    from argparse import Namespace
    with pytest.raises(ValueError, match="sigma"):
        experiment.fit_history([{"z_obs": [1., 2.], "sigma_z": [.1, .2]}],
                               np.asarray([.1, .1]), Namespace())


def test_wls_rejects_ambiguous_statistic_role():
    import pytest
    from argparse import Namespace
    with pytest.raises(ValueError, match="input role"):
        experiment.wls_metrics(np.ones(122), np.ones(122),
                               Namespace(chi_square_alpha=.05, detection_sigma=5.),
                               input_role="clean")


def _legacy_preparation_fixture():
    sigma = [.001] * 14 + [.01] * 108
    scans = [{
        "scan_index": index, "z_obs": [1.0] * 122, "z_clean": [.99] * 122,
        "three_phase_voltages": [{"bus": "b1", "vln_pu": [1., 1., 1.], "ang_deg": [0., -120., 120.]}],
        "three_phase_branch_currents": [{"preserved": "already noisy sensor data"}],
        "branch_current_sigma_pu": .001, "op_point": {"load_scale": 1.0},
    } for index in range(3)]
    row = {"id": "preparation-fixture", **copy.deepcopy(scans[0]), "scans": scans, "sigma_z": sigma,
           "label": {"branch_row0": 10}, "z_true": [77.] * 122}
    metadata = {"hif": {"generation": {"noise_scale": 1.},
                         "branch_current_measurements": {"branch_current_sigma_pu": .001}}}
    return row, metadata


def test_noise_preparation_is_deterministic_label_independent_and_no_redraw():
    source, metadata = _legacy_preparation_fixture()
    before = copy.deepcopy(source)
    prepared, receipt = experiment.prepare_trial_row(source, source_metadata=metadata, seed=19)
    repeated, repeated_receipt = experiment.prepare_trial_row(source, source_metadata=metadata, seed=19)
    assert prepared == repeated and receipt == repeated_receipt and source == before
    assert prepared["three_phase_voltages"] != source["three_phase_voltages"]
    assert prepared["z_obs"] == source["z_obs"]
    assert prepared["three_phase_branch_currents"] == source["three_phase_branch_currents"]
    assert all(scan["three_phase_sigma"] == .005 for scan in prepared["scans"])
    changed_truth = copy.deepcopy(source)
    changed_truth["label"]["branch_row0"] = 19
    for scan in changed_truth["scans"]:
        scan["z_clean"] = [1000.] * 122
    altered, altered_receipt = experiment.prepare_trial_row(changed_truth, source_metadata=metadata, seed=19)
    assert [experiment.observable_scan(s) for s in altered["scans"]] == [experiment.observable_scan(s) for s in prepared["scans"]]
    assert altered_receipt["prepared_observable_scans_sha256"] == receipt["prepared_observable_scans_sha256"]
    contracted, contracted_receipt = experiment.prepare_trial_row(prepared, source_metadata=None, seed=29)
    assert contracted == prepared
    assert contracted_receipt["provenance"] == "existing_noise_contract"


def test_explicit_weights_without_applied_noise_evidence_are_not_a_contract():
    source, _ = _legacy_preparation_fixture()
    source["three_phase_sigma"] = .005
    for scan in source["scans"]:
        scan["three_phase_sigma"] = .005
    with pytest.raises(ValueError, match="applied-noise contract"):
        experiment.prepare_trial_row(source, source_metadata=None, seed=19)


def test_new_preparation_cannot_reuse_historical_configuration(tmp_path, monkeypatch):
    _run_stubbed_pilot(tmp_path, monkeypatch)
    config_path = tmp_path / "out" / "config.json"
    config = json.loads(config_path.read_text())
    assert config["noise_preparation_version"] == experiment.NOISE_PREPARATION_VERSION
    assert config["noise_preparation_seed"] == 20260916
    config.pop("noise_preparation_version")
    config_path.write_text(json.dumps(config))
    historical = config_path.read_bytes()
    with pytest.raises(ValueError, match="new output directory"):
        experiment.main()
    assert config_path.read_bytes() == historical
