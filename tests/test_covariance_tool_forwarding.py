from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mcp_server import matpower_server as server
from Transmission import build_sft_traces as traces
from trace_protocol import hydrate_tool_arguments, round_tool_arguments, round_tool_result_payload, round_user_payload


def test_public_wls_wrappers_forward_exact_covariance(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_wls_json", lambda case, z, **kwargs: calls.append((case, z, kwargs)) or {"success": True})
    monkeypatch.setattr(server, "_write_case_text", lambda *_: "derived.m")
    sigma = [0.00123456789, 0.0]
    server.wls_from_path.fn(case_path="case14", z=[1, 0], measurement_sigma=sigma, exact_measurement_indices=[1])
    server.wls_from_text.fn(case_name="derived", case_text="unused", z=[1, 0], measurement_sigma=sigma, exact_measurement_indices=[1])
    assert [item[0] for item in calls] == ["case14", "derived.m"]
    assert all(item[2] == {"measurement_sigma": sigma, "exact_measurement_indices": [1]} for item in calls)


def test_public_meter_wrappers_forward_variance_and_constraints(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_meas_correction_json", lambda case, z, **kwargs: calls.append(kwargs) or {"success": True})
    monkeypatch.setattr(server, "_write_case_text", lambda *_: "derived.m")
    kwargs = dict(z=[1, 0], suspect_group=[0], R_variances_full=[1e-8, 0], exact_measurement_indices=[1])
    assert server.correct_measurements_from_path.fn(case_path="case14", **kwargs)["success"]
    assert server.correct_measurements_from_text.fn(case_name="derived", case_text="unused", **kwargs)["success"]
    assert all(item["R_variances_full"] == [1e-8, 0] and item["exact_measurement_indices"] == [1] for item in calls)


def test_public_single_scan_hif_preserves_all_sensor_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_estimate_hif_location_magnitude_logic", lambda **kwargs: calls.append(kwargs) or {"success": True})
    server.estimate_hif_location_magnitude_from_path.fn(
        case_path="case14", candidate_branch_row0=1, sigma_z=[0.004, 0.04],
        three_phase_sigma=0.007, branch_current_sigma_pu=0.002,
    )
    assert calls[0]["sigma_z"] == [0.004, 0.04]
    assert calls[0]["three_phase_sigma"] == 0.007
    assert calls[0]["branch_current_sigma_pu"] == 0.002


@pytest.mark.parametrize("name,extra,expected", [
    ("wls_from_path", {}, {"measurement_sigma": [0.002, 0], "exact_measurement_indices": [1]}),
    ("correct_measurements_from_path", {"suspect_group": [0]}, {"R_variances_full": [4e-6, 0], "exact_measurement_indices": [1]}),
    ("correct_parameters_from_path", {"line_index": 1, "z_scans": [[1, 0]]}, {"R_variances_full": [4e-6, 0]}),
    ("estimate_hif_location_magnitude_from_path", {"candidate_branch_row0": 0, "three_phase_sigma": 0.007, "branch_current_sigma_pu": 0.003},
     {"sigma_z": [0.002, 0], "three_phase_sigma": 0.007, "branch_current_sigma_pu": 0.003}),
])
def test_local_dispatch_forwards_declared_weights(monkeypatch, name, extra, expected):
    calls = []
    monkeypatch.setattr(server, name, SimpleNamespace(fn=lambda **kwargs: calls.append(kwargs) or {"success": True}))
    result = traces.call_tool_json("unused", name, {"case_path": "case14", "z": [1, 0], "sigma_z": [0.002, 0], "exact_measurement_indices": [1], **extra})
    assert result["success"]
    for key, value in expected.items():
        assert calls[0][key] == value


def test_hydration_uses_new_snapshot_covariance_and_rejects_stale_weights():
    messages = [{"role": "user", "content": json.dumps({"case_path": "case14", "z_obs": [1, 2], "sigma_z": [.001, .01]})}]
    snapshot = traces.make_verification_snapshot_payload("derived", [1, 2, 0], "new layout", "post_topology_correction",
                                                        sigma_z=[.003, .02, 0], exact_measurement_indices=[2])
    hidden = {"snapshot_context": snapshot}
    wls, _ = hydrate_tool_arguments("wls_from_path", {}, messages, hidden)
    assert wls["measurement_sigma"] == [.003, .02, 0]
    assert wls["exact_measurement_indices"] == [2]
    assert wls["z"] == [1, 2, 0]
    correction, _ = hydrate_tool_arguments("correct_measurements_from_path", {"suspect_group": [0]}, messages, hidden)
    assert correction["R_variances_full"] == [9e-6, 4e-4, 0]
    with pytest.raises(ValueError, match="conflicts"):
        hydrate_tool_arguments("wls_from_path", {"measurement_sigma": [.001, .01]}, messages, hidden)


def test_parameter_context_uses_scan_covariance_not_current_root_weights():
    context = traces.make_parameter_followup_payload({"z_scans": [[1, 2]], "initial_states": [],
        "sigma_z": [.001, .01], "sigma_z_scans": [.002, .03]}, "case14")
    args, _ = hydrate_tool_arguments("correct_parameters_from_path", {"line_index": 1}, [], {"parameter_context": context})
    assert args["R_variances_full"] == [4e-6, 9e-4]


def test_hif_context_removes_clean_audit_arrays_and_hydrates_noise():
    source = {"id": "hif", "z_obs": [1, 2], "sigma_z": [.002, .03], "three_phase_sigma": .007,
              "branch_current_sigma_pu": .004, "scans": [{"scan_index": 0, "z_obs": [1, 2],
              "z_clean": [9, 9], "z_absent_clean": [8, 8], "three_phase_voltages_clean": ["hidden"],
              "three_phase_branch_currents_clean": ["hidden"], "label": {"phase": "A"}}]}
    context = traces.make_hif_context_payload(source, "case14")
    assert not any("clean" in key or "label" in key for key in context["scans"][0])
    assert context["scans"][0]["three_phase_sigma"] == .007
    for tool in ("estimate_hif_location_magnitude_from_path", "estimate_hif_location_magnitude_multiscan_from_path"):
        args, _ = hydrate_tool_arguments(tool, {"candidate_branch_row0": 0}, [], {"hif_context": context})
        assert args["sigma_z"] == [.002, .03]
        assert args["branch_current_sigma_pu"] == .004
        if tool == "estimate_hif_location_magnitude_from_path":
            assert args["three_phase_sigma"] == .007


def test_covariance_serialization_never_rounds_small_weights_to_zero():
    component_sigma = 1e-4 / np.sqrt(2)
    payload = {"sigma_z": [component_sigma], "R_variances_full": [component_sigma**2],
               "harmonic_measurements": [{"sigma": component_sigma, "V_real": 0.123456789}]}
    for formatter in (round_user_payload, round_tool_arguments, round_tool_result_payload):
        result = formatter(payload)
        assert result["sigma_z"] == payload["sigma_z"]
        assert result["R_variances_full"] == payload["R_variances_full"]
        assert result["harmonic_measurements"][0]["sigma"] == component_sigma
        assert result["harmonic_measurements"][0]["V_real"] == 0.123457


def test_parameter_verification_preserves_noisy_observation(monkeypatch, tmp_path):
    index_map = {"Vm": [0, 14], "Pinj": [14, 28], "Qinj": [28, 42], "Pf": [42, 62], "Qf": [62, 82], "Pt": [82, 102], "Qt": [102, 122]}
    sigma = [.002] * 14 + [.02] * 108
    observed = [1.001] * 122
    record = {"id": "parameter", "scenario": "parameter_error", "z_true": [1.] * 122, "z_obs": observed,
              "sigma_z": sigma, "sigma_z_scans": sigma, "z_scans": [observed], "initial_states": [[1.] * 28],
              "label": {"line_row": 0, "from_bus": 1, "to_bus": 2}, "correction_case_path": "case14"}
    meta = {"case": "case14", "nb": 14, "nl": 20, "index_map": index_map, "branch_info": [], "baseMVA": 100.}
    monkeypatch.setattr(traces, "load_sample_sources", lambda _: (meta, [record]))
    monkeypatch.setattr(traces, "call_backend_tool", lambda _endpoint, name, *_args, **_kwargs:
                        {"success": True} if name == "correct_parameters_from_path" else
                        {"success": True, "r": [1.] * 122, "lambdaN": [1.] * 40, "wls_objective": 122.})
    captured = []

    def capture(*args, **kwargs):
        captured.append((args, kwargs))
        raise RuntimeError("verification captured")

    monkeypatch.setattr(traces, "make_verification_snapshot_payload", capture)
    config = traces.BuilderConfig(samples_path=Path("unused"), meta_path=Path("unused"), imbalance_samples_path=None,
        imbalance_meta_path=None, hif_samples_path=None, hif_meta_path=None, hardening_source_samples_path=None,
        case_name=None, endpoint="unused", out_path=tmp_path / "trace.jsonl", analysis_out_path=None, mock=False,
        seed=1, add_no_error=0, with_correction=True, corr_max_iter=2, corr_tol=.001,
        allow_hif_metadata_fallback=False, hardening_examples=0)
    with pytest.raises(RuntimeError, match="verification captured"):
        traces.build_sft(config)
    assert captured[0][0][1] == observed
    assert captured[0][0][1] != record["z_true"]
    assert captured[0][1]["sigma_z"] == sigma


def test_noiseless_topology_prediction_is_not_observation_even_with_a_sigma_label():
    with pytest.raises(traces.UnsupportedTopologyVerification, match="noiseless_model_prediction"):
        traces.verified_topology_snapshot(
            {"z_obs": [1., 2.], "sigma_z": [.001, .01], "measurement_role": "noiseless_model_prediction"},
            current_z_obs=[1.001, 2.1], current_sigma_z=[.001, .01],
        )


def test_topology_model_only_update_retains_noise_and_remaining_fault_with_verified_ids():
    original = [1.0012, 2.174]  # Ordinary noise and an independent large meter error.
    sigma = [.001, .01]
    ids = ["Vm:bus1", "Pf:line1:from"]
    snapshot = {"z_obs_policy": "preserve_current_z_obs", "measurement_channel_ids": ids}
    kept = traces.verified_topology_snapshot(snapshot, current_z_obs=original,
        current_sigma_z=sigma, current_channel_ids=ids)
    assert kept["z_obs"] == original
    assert kept["sigma_z"] == sigma
    assert kept["original_observations_preserved"]
    for unverifiable in (None, ["Vm:different_bus", "Pf:line1:from"]):
        with pytest.raises(traces.UnsupportedTopologyVerification, match="identity_unverified"):
            traces.verified_topology_snapshot(snapshot, current_z_obs=original,
                current_sigma_z=sigma, current_channel_ids=unverifiable)


@pytest.mark.parametrize("with_explicit_noisy_snapshot", [False, True])
def test_topology_trace_rejects_solver_fallback_but_accepts_noisy_snapshot(monkeypatch, tmp_path, with_explicit_noisy_snapshot):
    index_map = {"Vm": [0, 14], "Pinj": [14, 28], "Qinj": [28, 42], "Pf": [42, 62], "Qf": [62, 82], "Pt": [82, 102], "Qt": [102, 122]}
    sigma = [.001] * 14 + [.01] * 108
    original = [1.001] * 122
    original[40] += .10  # A solver replacement would erase this remaining error.
    record = {"id": "topology", "scenario": "topology_error", "z_true": [1.] * 122,
              "z_obs": original, "sigma_z": sigma,
              "label": {"cb_name": "CB_test", "old_status": "closed", "new_status": "open"}}
    noisy_verification = [1.002] * 122
    noisy_verification[40] += .10
    if with_explicit_noisy_snapshot:
        record["verification_snapshots"] = {"post_topology_correction": {
            "case_path": "case14", "z_obs": noisy_verification, "sigma_z": sigma,
        }}
    meta = {"case": "case14", "nb": 14, "nl": 20, "index_map": index_map, "branch_info": [], "baseMVA": 100.}
    monkeypatch.setattr(traces, "load_sample_sources", lambda _: (meta, [record]))
    calls = []

    def backend(_endpoint, name, _arguments, _messages, hidden_context, **_kwargs):
        calls.append((name, hidden_context.get("snapshot_context")))
        if name == "correct_topology_from_path":
            return {"success": True, "z_corrected": [1.] * 122,
                    "z_corrected_role": "noiseless_model_prediction", "z_corrected_noise_applied": False}
        return {"success": True, "r": [1.] * 122, "lambdaN": [1.] * 40, "wls_objective": 122.}

    monkeypatch.setattr(traces, "call_backend_tool", backend)
    monkeypatch.setattr(traces, "build_final_target", lambda *_args, **_kwargs: {"evidence": {"global_metrics": {}}})
    monkeypatch.setattr(traces, "rejection_reason", lambda _: None)
    config = traces.BuilderConfig(samples_path=Path("unused"), meta_path=Path("unused"), imbalance_samples_path=None,
        imbalance_meta_path=None, hif_samples_path=None, hif_meta_path=None, hardening_source_samples_path=None,
        case_name=None, endpoint="unused", out_path=tmp_path / "trace.jsonl", analysis_out_path=tmp_path / "analysis.jsonl", mock=False,
        seed=1, add_no_error=0, with_correction=True, corr_max_iter=2, corr_tol=.001,
        allow_hif_metadata_fallback=False, hardening_examples=0)
    traces.build_sft(config)
    assert record["z_obs"] == original
    if with_explicit_noisy_snapshot:
        assert len([item for item in calls if item[0] == "wls_from_path"]) == 2
        verified = next(context for name, context in calls if name == "wls_from_path" and context is not None)
        assert verified["z_obs"] == noisy_verification
        assert verified["sigma_z"] == sigma
        assert config.out_path.read_text().strip()
    else:
        assert len([item for item in calls if item[0] == "wls_from_path"]) == 1
        assert not config.out_path.read_text().strip()
        analysis = config.analysis_out_path.read_text()
        assert "topology_verification_noisy_observation_or_covariance_missing" in analysis


def _waveform_source_config(tmp_path, row, metadata):
    samples = tmp_path / "samples.jsonl"
    meta = tmp_path / "meta.json"
    samples.write_text(json.dumps(row) + "\n")
    meta.write_text(json.dumps(metadata))
    return SimpleNamespace(samples_path=samples, meta_path=meta, seed=20260916,
                           imbalance_samples_path=None, imbalance_meta_path=None,
                           hif_samples_path=None, hif_meta_path=None)


def _legacy_voltage_only_hif():
    sigma = [.001] * 14 + [.01] * 108
    voltage = [{"bus": "b1", "vln_pu": [1.03, 1.02, 1.01], "ang_deg": [0., -120., 120.]}]
    scan = {"scan_index": 0, "z_obs": [.1] * 122, "three_phase_voltages": voltage}
    return {"id": "legacy_hif", "scenario": "high_impedance_fault", "sigma_z": sigma,
            "z_obs": scan["z_obs"], "three_phase_voltages": voltage, "scans": [scan]}


def test_sft_waveform_boundary_rejects_unknown_or_weight_only_noise(tmp_path):
    source = _legacy_voltage_only_hif()
    config = _waveform_source_config(tmp_path, source, {})
    original = config.samples_path.read_bytes()
    with pytest.raises(ValueError, match="Regenerate.*supported original source meta"):
        traces.load_sample_sources(config)
    assert config.samples_path.read_bytes() == original
    source["three_phase_sigma"] = .005
    config = _waveform_source_config(tmp_path, source, {})
    with pytest.raises(ValueError, match="weighting sigmas alone"):
        traces.load_sample_sources(config)


def test_sft_known_legacy_metadata_upgrade_is_reproducible_and_preserves_existing_noise(tmp_path):
    source = _legacy_voltage_only_hif()
    config = _waveform_source_config(tmp_path, source, {"hif": {"generation": {"noise_scale": 1.}}})
    original_bytes = config.samples_path.read_bytes()
    _, first = traces.load_sample_sources(config)
    _, second = traces.load_sample_sources(config)
    assert first == second
    assert first[0]["z_obs"] == source["z_obs"]
    assert first[0]["three_phase_voltages"] != source["three_phase_voltages"]
    assert first[0]["noise_contract"]["schema"] == "generated_sensor_noise_v1"
    assert first[0]["noise_alignment"]["added_noise"] == ["three_phase_voltages"]
    assert config.samples_path.read_bytes() == original_bytes


def test_sft_fresh_declared_waveform_rows_are_not_noised_again(tmp_path):
    from three_phase_nlm.measurement_noise import align_legacy_waveform_row

    source = align_legacy_waveform_row(_legacy_voltage_only_hif(), "hif", np.random.default_rng(44),
                                       legacy_metadata={"hif": {"generation": {"noise_scale": 1.}}})
    config = _waveform_source_config(tmp_path, source, {})
    _, rows = traces.load_sample_sources(config)
    assert rows == [source]


@pytest.mark.parametrize("scenario", ["harmonic_anomaly", "multi_error"])
def test_sft_unknown_legacy_harmonic_covariance_is_rejected_in_loader_and_context(tmp_path, scenario):
    source = {"id": "legacy_harmonic", "scenario": scenario,
              "harmonic_measurements": [{"bus": 2, "h": 5, "V_real": .01, "V_imag": .02, "sigma": 1e-4}]}
    config = _waveform_source_config(tmp_path, source, {})
    with pytest.raises(ValueError, match="Regenerate legacy harmonic"):
        traces.load_sample_sources(config)
    with pytest.raises(ValueError, match="sigma_semantics"):
        traces.make_harmonic_followup_payload(source, "case14")


def test_fresh_harmonic_component_sigmas_pass_unchanged_and_rms_converts_once(tmp_path):
    rms = {"bus": 2, "h": 5, "V_real": .01, "V_imag": .02,
           "sigma": 1e-4, "sigma_semantics": "complex_rms"}
    source = {"id": "harmonic", "scenario": "multi_error", "harmonic_measurements": [rms]}
    config = _waveform_source_config(tmp_path, source, {})
    source_bytes = config.samples_path.read_bytes()
    _, rows = traces.load_sample_sources(config)
    component = rows[0]["harmonic_measurements"][0]
    assert component["sigma"] == 1e-4 / np.sqrt(2)
    assert component["sigma_semantics"] == "per_component"
    assert rms["sigma"] == 1e-4
    assert config.samples_path.read_bytes() == source_bytes
    context = traces.make_harmonic_followup_payload(rows[0], "case14")
    assert context["harmonic_measurements"][0]["sigma"] == component["sigma"]
    normalized_again = traces.normalized_generated_harmonic_measurements(context["harmonic_measurements"])
    assert normalized_again == context["harmonic_measurements"]
    fresh = {"bus": 2, "h": 5, "V_real": .01, "V_imag": .02,
             "sigma": 1e-4 / np.sqrt(2), "sigma_semantics": "per_component", "sigma_complex_rms": 1e-4}
    assert traces.normalized_generated_harmonic_measurements([fresh]) == [fresh]
