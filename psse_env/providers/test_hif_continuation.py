from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from mcp_server.matpower_server import _load_python_case
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers import hif_continuation as continuation
from psse_env.providers.scenario_generator import build_measurement_vector


def _bound_metadata():
    sensors = {"three_phase_voltages": [{"bus": "b1", "vln_pu": [1., 1., 1.]}],
               "three_phase_branch_currents": [{"observable": 1}]}
    scan = {"scan_index": 0, "op_point": {"load_scale": .91, "bus_load_scales": {"b2": .97}}, **sensors}
    return {"hif_runtime": copy.deepcopy(sensors), "hif_scan_window": {"scans": [scan]},
            "measurement_convention": {"shunt_convention": "ybus"}}


def _fit():
    return {"success": True, "candidate_branch_row0": 2,
            "estimated": {"phase": "A", "alpha_from_from_bus": .45, "r_hif_pu": 10., "resistance_model": "shared"},
            "uncertainty": {"near_best_alpha_interval": [.4, .5], "near_best_r_hif_pu_interval": [9., 11.]}}


def _state_and_prediction(monkeypatch, *, fault=True, wide=False):
    absent = build_measurement_vector(_load_python_case("case14"))
    sigma = np.r_[np.full(14, .001), np.full(108, .01)]
    effect = np.zeros(122)
    effect[25] = .2  # HIF and meter error deliberately share this channel.
    center = absent + effect
    prediction = {"predicted_hif_measurements": center.tolist(), "predicted_base_measurements": absent.tolist(),
                  "measurement_effect": effect.tolist(), "prediction_lower": center.tolist(), "prediction_upper": center.tolist()}
    if wide:
        prediction["prediction_lower"][0] -= .01
        prediction["prediction_upper"][0] += .01
    observed = center.copy()
    observed[25] += .1 if fault else 0.
    state = {"state_id": "s0", "state_hash": "h0", "case": "case14", "measurements": observed.tolist(),
             "metadata": {"sigma_z": sigma.tolist()}, "policy_observation": {
                 "unresolved_signatures": ["hif_suspected_zero_sequence"],
                 "explained_anomalies": [{"family": "hif", "detail": {"conditioning_fit": _fit()}}]}}
    monkeypatch.setattr("psse_env.providers.matpower.conditioned_prediction", lambda state, cache: copy.deepcopy(prediction))
    return state, prediction


def test_auxiliary_binding_ignores_fault_labels_and_stale_scada():
    metadata = _bound_metadata()
    expected = continuation.current_scan(metadata)
    metadata["hif_runtime"]["z_obs"] = [999.] * 122
    metadata["hif_scan_window"]["scans"][0].update(z_obs=[-999.] * 122, clean=[777.] * 122, label={"target": 77})
    assert continuation.current_scan(metadata) == expected
    assert expected["op_point"]["bus_load_scales"] == {"b2": .97}


def test_ambiguous_or_missing_acquisition_is_not_guessed():
    metadata = _bound_metadata()
    duplicate = copy.deepcopy(metadata["hif_scan_window"]["scans"][0])
    duplicate["scan_index"] = 1
    metadata["hif_scan_window"]["scans"].append(duplicate)
    with pytest.raises(ValueError, match="one auxiliary-bound"):
        continuation.current_scan(metadata)
    with pytest.raises(ValueError):
        continuation.current_scan({"hif_runtime": {"load_scale": 1.}})


@pytest.mark.parametrize("index", [None, True, -1, 0.5])
def test_invalid_scan_identity_fails_closed(index):
    metadata = _bound_metadata()
    metadata["hif_scan_window"]["scans"][0]["scan_index"] = index
    with pytest.raises(ValueError, match="unique nonnegative integer"):
        continuation.current_scan(metadata)


def test_duplicate_scan_id_fails_closed():
    metadata = _bound_metadata()
    duplicate = copy.deepcopy(metadata["hif_scan_window"]["scans"][0])
    duplicate["three_phase_branch_currents"] = [{"other": 2}]
    metadata["hif_scan_window"]["scans"].append(duplicate)
    with pytest.raises(ValueError, match="unique nonnegative integer"):
        continuation.current_scan(metadata)


def test_effect_subtraction_preserves_same_channel_meter_fault_and_raw_state(monkeypatch):
    state, prediction = _state_and_prediction(monkeypatch)
    original = copy.deepcopy(state)
    provider = MatpowerDeploymentProviders(evidence_profile="auxiliary_diagnostics", normalized_residual_threshold=4.)
    solved = provider._solve(state)
    assert state == original
    assert solved["wls_measurements"][25] - prediction["predicted_base_measurements"][25] == pytest.approx(.1)
    metrics = provider.run_wls(state)
    assert metrics["hif_conditioning"]["status"] == "ready"
    assert metrics["hif_conditioning"]["remaining_meter_candidate_indices"] == [25]
    assert metrics["hif_conditioning"]["overlapping_candidate_indices"] == [25]
    assert not metrics["no_material_anomaly_remaining"]
    assert metrics["chi_square_alarm"] or metrics["normalized_residual_alarm"]
    context = provider.get_measurement_context(state)
    action = context["supported_corrections"][0]
    correction = provider.correct_measurements(state, action)
    assert correction["modification"]["measurement_updates"] == {25: prediction["predicted_hif_measurements"][25]}
    candidate = copy.deepcopy(state)
    candidate["measurements"][25] = correction["modification"]["measurement_updates"][25]
    assert provider.run_wls(candidate)["no_material_anomaly_remaining"]
    assert state == original


def test_pure_hif_control_has_no_meter_writes(monkeypatch):
    state, _ = _state_and_prediction(monkeypatch, fault=False)
    provider = MatpowerDeploymentProviders(evidence_profile="auxiliary_diagnostics", normalized_residual_threshold=4.)
    assert provider.run_wls(state)["no_material_anomaly_remaining"]
    assert provider.get_measurement_context(state)["supported_corrections"] == []
    output = provider.correct_measurements(state, {"arguments": {"suspect_group": [25]}})
    assert output["execution_status"] == "failure"


def test_wide_prediction_cannot_hide_an_error_or_authorize_repair(monkeypatch):
    state, _ = _state_and_prediction(monkeypatch, wide=True)
    provider = MatpowerDeploymentProviders(evidence_profile="auxiliary_diagnostics", )
    output = provider.run_wls(state)
    assert output["hif_conditioning"]["status"] == "unavailable"
    assert not output["globally_resolved"]
    assert provider.get_measurement_context(state)["supported_corrections"] == []
    assert provider.correct_measurements(state, {"arguments": {"suspect_group": [25]}})["execution_status"] == "failure"


@pytest.mark.parametrize("arguments", [{"suspect_group": [24]}, {"measurement_updates": {25: 0.}}, {"suspect_group": []}])
def test_unsupported_or_explicit_writes_cannot_bypass_conditioning(monkeypatch, arguments):
    state, _ = _state_and_prediction(monkeypatch)
    output = MatpowerDeploymentProviders(evidence_profile="auxiliary_diagnostics", ).correct_measurements(state, {"arguments": arguments})
    assert output["execution_status"] == "failure"
    assert "modification" not in output


def test_prediction_requires_independent_fit_current_model_and_ybus():
    fit = continuation.fit_receipt(_fit(), "case14", independent=False)
    state = {"case": "case14", "metadata": _bound_metadata(), "policy_observation": {
        "explained_anomalies": [{"family": "hif", "detail": {"conditioning_fit": fit}}]}}
    with pytest.raises(ValueError, match="independent"):
        continuation.conditioned_prediction(state, {})
    fit["independent_of_current_scada"] = True
    fit["case_sha256"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        continuation.conditioned_prediction(state, {})
    fit["case_sha256"] = continuation.case_fingerprint("case14")
    fit["acquisition_sha256"] = continuation.acquisition_fingerprint(state["metadata"])
    state["metadata"]["measurement_convention"]["shunt_convention"] = "legacy_injection"
    with pytest.raises(ValueError, match="ybus"):
        continuation.conditioned_prediction(state, {})


def test_multiscan_excludes_current_scada_from_fit(monkeypatch):
    metadata = _bound_metadata()
    scan = metadata["hif_scan_window"]["scans"][0]
    scan["z_obs"] = [99.] * 122
    for index in (1, 2, 3):
        history = copy.deepcopy(scan)
        history.update(scan_index=index, z_obs=[float(index)] * 122,
                       three_phase_branch_currents=[{"observable": index + 10}])
        metadata["hif_scan_window"]["scans"].append(history)
    provider = MatpowerDeploymentProviders(evidence_profile="auxiliary_diagnostics", )
    captured = {}
    def fit(**kwargs):
        captured.update(kwargs)
        return {**_fit(), "fit": {"weighted_residual_norm": 1., "residual_reduction_vs_no_hif": .5}}
    monkeypatch.setattr(provider, "_memoized_hif_multiscan", fit)
    result = provider.estimate_hif_multiscan({"case": "case14", "metadata": metadata},
        {"arguments": {"candidate_branch_row0": 2}})
    assert [s["scan_index"] for s in captured["scans"]] == [1, 2, 3]
    assert result["anomaly_explanation"]["detail"]["conditioning_fit"]["independent_of_current_scada"]


def test_new_auxiliary_acquisition_invalidates_old_fit():
    metadata = _bound_metadata()
    fit = continuation.fit_receipt(_fit(), "case14", independent=True, metadata=metadata)
    state = {"case": "case14", "metadata": metadata, "policy_observation": {
        "explained_anomalies": [{"family": "hif", "detail": {"conditioning_fit": fit}}]}}
    metadata["hif_runtime"]["three_phase_branch_currents"] = [{"observable": 2}]
    metadata["hif_scan_window"]["scans"][0]["three_phase_branch_currents"] = [{"observable": 2}]
    with pytest.raises(ValueError, match="auxiliary acquisition change"):
        continuation.conditioned_prediction(state, {})


def test_forward_model_change_invalidates_fit(monkeypatch):
    metadata = _bound_metadata()
    fit = continuation.fit_receipt(_fit(), "case14", independent=True, metadata=metadata)
    state = {"case": "case14", "metadata": metadata, "policy_observation": {
        "explained_anomalies": [{"family": "hif", "detail": {"conditioning_fit": fit}}]}}
    monkeypatch.setattr(continuation, "model_fingerprint", lambda _: "changed-model")
    with pytest.raises(ValueError, match="forward-model change"):
        continuation.conditioned_prediction(state, {})


def test_unconfigured_local_detector_requires_handoff(monkeypatch):
    state, _ = _state_and_prediction(monkeypatch, fault=False)
    metrics = MatpowerDeploymentProviders(evidence_profile="auxiliary_diagnostics", ).run_wls(state)
    assert metrics["hif_conditioning"]["status"] == "unavailable"
    assert "normalized_residual_test_not_configured" in metrics["hif_conditioning"]["failure_reasons"]
    assert not metrics["no_material_anomaly_remaining"]


def test_model_fingerprint_includes_uppercase_dss_and_sorts_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("three_phase_nlm.hif_parameter_estimator._resolve_model_dir", lambda *_: tmp_path)
    # Deliberately create in a different order from the canonical path order.
    contents = {"zeta.dss": b"Redirect nested/IEEE14Lines.DSS\n",
                "nested/IEEE14Lines.DSS": b"New Line.original Bus1=b1 Bus2=b2\n",
                "Master.DsS": b"Redirect zeta.dss\n",
                "alpha.dss": b"! lowercase first letter\n",
                "Zulu.DSS": b"! uppercase first letter\n"}
    for relative, data in contents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (tmp_path / "not_a_file.DSS").mkdir()
    (tmp_path / "notes.txt").write_text("not a circuit file")
    expected = hashlib.sha256()
    for relative in sorted(contents):
        expected.update(relative.encode())
        expected.update(b"\0")
        expected.update(contents[relative])
    original = continuation.model_fingerprint(str(tmp_path))
    assert original == expected.hexdigest()
    (tmp_path / "nested/IEEE14Lines.DSS").write_bytes(b"New Line.changed Bus1=b1 Bus2=b3\n")
    assert continuation.model_fingerprint(str(tmp_path)) != original
