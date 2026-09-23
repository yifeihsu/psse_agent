"""wls_gated_diagnostics: no flags or truth reach the agent; auxiliary tools open only behind a balanced WLS alarm."""
from __future__ import annotations

from copy import deepcopy

import pytest

from mcp_server.matpower_server import _load_python_case
from psse_env.actions import (
    ESTIMATE_HIF_FROM_PATH, ESTIMATE_HIF_MULTISCAN_FROM_PATH, GET_HARMONIC_CONTEXT,
    GET_THREE_PHASE_CONTEXT, RUN_ALTERNATIVE_TEST, RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH, RUN_WLS,
)
from psse_env.dagger.dataset_builder import system_prompt_for_observation, tool_schemas_for_observation
from psse_env.dagger.protocol_bridge import CANONICAL_TO_INTERNAL_TOOL, unified_tool_schemas
from psse_env.evidence_profile import (
    AUXILIARY_EVIDENCE_PROFILE, DEFAULT_EVIDENCE_PROFILE, GATED_DIAGNOSTIC_TOOLS,
    PRECOMPUTED_DIAGNOSIS_FIELDS, SCADA_ONLY_PROFILE, WLS_GATED_DISABLED_TOOLS, WLS_GATED_PROFILE,
    sanitize_gated_execution, sanitize_gated_metadata, sanitize_gated_observation,
)
from psse_env.oracle.process_validity import ProcessValidityOracle, current_wls_alarm
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import build_measurement_vector
from psse_env.transactional_env import TransactionalPSSEEnv
from three_phase_nlm.measurement_noise import generated_noise_contract

SIGMA = [.001] * 14 + [.01] * 108
PMU_SIGMA = 1e-4
GATE_CODES = {"diagnostics_require_wls_alarm", "tool_disabled_by_evidence_profile",
              "evidence_profile_tool_unavailable", "evidence_unavailable_in_scada_only_profile"}


def _measurements(*, anomalous: bool) -> list[float]:
    z = build_measurement_vector(_load_python_case("case14")).tolist()
    if anomalous:
        z[40] += .2
    return z


def _balanced_voltages(z):
    return [{"bus": f"b{index + 1}", "vln_pu": [float(z[index])] * 3, "ang_deg": [0., -120., 120.]}
            for index in range(14)]


def _branch_currents():
    return [{"branch": "Line.1-2", "branch_row0": 0, "from_bus": "b1", "to_bus": "b2",
             "i_from_pu": [1., 1., 1.], "ang_from_deg": [0., -120., 120.],
             "i_to_pu": [1., 1., 1.], "ang_to_deg": [180., 60., -60.]}]


def _scenario(*, anomalous: bool = True, currents: bool = True):
    z = _measurements(anomalous=anomalous)
    voltages = _balanced_voltages(z)
    metadata = {
        "sigma_z": SIGMA,
        "noise_contract": generated_noise_contract(
            SIGMA, noise_scale=1., three_phase_sigma=PMU_SIGMA, branch_current_sigma_pu=PMU_SIGMA),
        "parameter_scans": {"z_scans": [z, z], "sigma_z": SIGMA, "initial_states": [[7.] * 28],
                            "op_point": {"load_scale": 7.}, "labels": {"true_line": 3}},
        "three_phase_voltages": voltages, "three_phase_sigma": PMU_SIGMA,
        "harmonic_measurements": [{"bus": 3, "h": 5, "Vm": .02, "Va_deg": 0., "sigma": 1e-4}],
        "hif_runtime": {"z_obs": z, "scan_index": 0, "op_point": {"load_scale": .8}, "sigma_z": SIGMA,
                        "three_phase_sigma": PMU_SIGMA, "three_phase_voltages": voltages,
                        "load_scale": .8, "label": {"phase": "A"}},
        "hif_scan_window": {"scan_window_path": "w", "sigma_z": SIGMA, "three_phase_sigma": PMU_SIGMA,
                            "scans": [{"scan_index": 0, "z_obs": z, "three_phase_voltages": voltages,
                                       "op_point": {"load_scale": .8}, "label": {"phase": "A"},
                                       "three_phase_voltages_clean": voltages, "z_clean": z}],
                            "window_metadata": {"operating_point_mode": "diverse"}},
        "nlm_diagnostic": {"top_hif_groups": [{"branch_row0": 3}]},
        "faulted_model_dir": "C:/truth/faulted", "pristine_model_dir": None,
        "op_point": {"load_scale": .8}, "label": {"family": "hif"},
        "reported_breaker_status": {"CB3": False}, "substation_telemetry": {"reported_statuses": {"CB3": False}},
    }
    if currents:
        metadata["three_phase_branch_currents"] = _branch_currents()
        metadata["branch_current_sigma_pu"] = PMU_SIGMA
        metadata["hif_runtime"]["three_phase_branch_currents"] = _branch_currents()
        metadata["hif_runtime"]["branch_current_sigma_pu"] = PMU_SIGMA
    return {
        "scenario_id": "gated_root", "scenario_family": "measurement+hif", "case": "case14", "measurements": z,
        "unresolved_signatures": ["hif_suspected_zero_sequence"],
        "remaining_anomaly_score": 999., "no_material_anomaly_remaining": True,
        "semantic_field_provenance": {"unresolved_signatures": "deployment_sensor:waveform_capture",
                                      "remaining_anomaly_score": "deployment_sensor:waveform_capture",
                                      "no_material_anomaly_remaining": "deployment_sensor:waveform_capture"},
        "oracle_action_hints": [{"tool": "correct_parameters", "arguments": {"line_index": 3}}],
        "hidden_truth": {"true_hif_errors": [{"branch_row0": 3, "phase": "A"}]},
        "metadata": metadata,
    }


def _alarm_ledger(state_id="s0", state_hash="hash0", *, chi=True, residual=False):
    return {"state_id": state_id, "state_hash": state_hash, "successful": True,
            "evidence_source": "deployment_wls:lagrangian_port",
            "chi_square_alarm": chi, "normalized_residual_alarm": residual}


def _provider_snapshot(*, alarm: bool, currents: bool = True, declare_sigma: bool = True):
    scenario = _scenario(currents=currents)
    metadata = deepcopy(scenario["metadata"])
    if not declare_sigma:
        for block in (metadata, metadata["hif_runtime"], metadata["hif_scan_window"]):
            block.pop("three_phase_sigma", None)
            block.pop("branch_current_sigma_pu", None)
            block.pop("noise_contract", None)
    return {
        "case": "case14", "state_id": "s0", "state_hash": "hash0",
        "measurements": scenario["measurements"], "metadata": metadata,
        "policy_observation": {
            "active_state_id": "s0", "candidate_state_id": None, "unresolved_signatures": [],
            "fresh_context_evidence": {"wls": _alarm_ledger()} if alarm else {"wls": {"successful": False}},
        },
    }


class RecordingProvider:
    provider_kind = "deployment"

    def __init__(self, kind, *, alarm=True):
        self.kind, self.alarm, self.inputs = kind, alarm, []

    def __call__(self, state, action=None):
        self.inputs.append(deepcopy(state))
        binding = {"state_id": state["state_id"], "state_hash": state["state_hash"]}
        if self.kind == "wls":
            return {**binding, "evidence_source": "deployment_wls:fixture", "wls_objective": 10.,
                    "remaining_anomaly_score": 2. if self.alarm else 0., "no_material_anomaly_remaining": not self.alarm,
                    "chi_square_alarm": self.alarm, "normalized_residual_alarm": False,
                    "unresolved_signatures": ["wls_residual_outlier index=40 channel=Qinj"] if self.alarm else []}
        if self.kind == "failing":
            return {**binding, "execution_status": "failure", "error_code": "three_phase_context_failure"}
        if self.kind == "three_phase":
            return {**binding, "evidence_source": "deployment_context:three_phase_measurements",
                    "context_tool": GET_THREE_PHASE_CONTEXT, "request_attempted": True,
                    "three_phase_context_status": "available",
                    "available_evidence_channels": ["three_phase_voltages", "three_phase_branch_currents"],
                    "measurement_status": {"three_phase_voltages": "available", "three_phase_branch_currents": "available"}}
        return {**binding, "evidence_source": "deployment_context:fixture"}


def _real_env():
    providers = MatpowerDeploymentProviders(chi2_alpha=.01, normalized_residual_threshold=4.)
    return providers, TransactionalPSSEEnv(**providers.env_kwargs())


# --------------------------------------------------------------------- profile


def test_research_default_is_the_gated_profile():
    assert DEFAULT_EVIDENCE_PROFILE == WLS_GATED_PROFILE
    assert TransactionalPSSEEnv().evidence_profile == WLS_GATED_PROFILE
    assert MatpowerDeploymentProviders().env_kwargs()["evidence_profile"] == WLS_GATED_PROFILE
    assert RUN_ALTERNATIVE_TEST in WLS_GATED_DISABLED_TOOLS
    assert GATED_DIAGNOSTIC_TOOLS == {GET_THREE_PHASE_CONTEXT, GET_HARMONIC_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH,
                                      RUN_HSE_FROM_PATH, ESTIMATE_HIF_FROM_PATH, ESTIMATE_HIF_MULTISCAN_FROM_PATH}


# ----------------------------------------------------------------------- reset


def test_reset_keeps_streams_but_drops_seeded_signatures_hints_truth_and_precomputed_diagnoses():
    scenario = _scenario()
    original = deepcopy(scenario)
    env = TransactionalPSSEEnv()
    state = env.reset(scenario)
    assert state["unresolved_signatures"] == []
    assert state["remaining_anomaly_score"] is None
    assert state["no_material_anomaly_remaining"] is False
    assert state["explained_anomalies"] == []
    assert env.get_policy_observation().available_evidence == []
    stored = env.store.get_state(state["active_state_id"])
    metadata = stored["metadata"]
    assert metadata["evidence_profile"] == WLS_GATED_PROFILE
    for key in ("three_phase_voltages", "three_phase_branch_currents", "three_phase_sigma", "branch_current_sigma_pu",
                "harmonic_measurements", "substation_telemetry", "reported_breaker_status", "hif_runtime",
                "hif_scan_window", "parameter_scans", "noise_contract", "sigma_z"):
        assert key in metadata, key
    assert metadata["three_phase_sigma"] == PMU_SIGMA
    assert set(metadata["noise_contract"]["channels"]) == {"scada", "three_phase_voltages", "three_phase_branch_currents"}
    assert set(metadata["parameter_scans"]) == {"z_scans", "sigma_z"}
    for key in (*PRECOMPUTED_DIAGNOSIS_FIELDS, "op_point", "label", "scenario_id", "hidden_truth",
                "unresolved_signatures", "oracle_action_hints", "scenario_family"):
        assert key not in metadata, key
    assert "label" not in metadata["hif_runtime"]
    assert metadata["hif_runtime"]["three_phase_sigma"] == PMU_SIGMA
    scan = metadata["hif_scan_window"]["scans"][0]
    assert set(scan) == {"scan_index", "z_obs", "three_phase_voltages", "op_point"}
    assert "window_metadata" not in metadata["hif_scan_window"]
    assert "hidden_truth" not in stored
    assert stored["measurements"] == scenario["measurements"]
    assert scenario == original
    observation = env.get_policy_observation().as_dict()
    assert observation["evidence_profile"] == WLS_GATED_PROFILE
    assert observation["unresolved_signatures"] == []
    assert observation["available_evidence"] == []


def test_gated_sanitizers_strip_truth_from_every_level():
    execution = sanitize_gated_execution(_scenario())
    for key in ("hidden_truth", "scenario_family", "oracle_action_hints", "unresolved_signatures",
                "remaining_anomaly_score", "no_material_anomaly_remaining", "semantic_field_provenance"):
        assert key not in execution
    assert execution["evidence_profile"] == WLS_GATED_PROFILE
    assert execution["metadata"]["three_phase_voltages"]
    metadata = sanitize_gated_metadata({"nlm_diagnostic": {}, "faulted_model_dir": "x", "hse_summary": {},
                                        "three_phase_sigma": PMU_SIGMA, "label": {}})
    assert set(metadata) == {"three_phase_sigma", "evidence_profile"}
    observation = sanitize_gated_observation({
        "unresolved_signatures": ["hif_suspected_line_differential", "wls_residual_outlier index=1"],
        "explained_anomalies": [{"family": "three_phase_unbalance", "detail": {"bus_1based": 9}}],
        "fresh_context_evidence": {"three_phase": {"request_attempted": True}, "hif_conditioning": {"status": "ready"}},
        "nlm_summary": {"top_hif_groups": []}, "hif_summary": {"estimated": {"alpha": .4}},
        "hidden_truth": {"x": 1}, "label": "hif", "nlm_diagnostic": {}, "op_point": {}, "gnn_screen": {},
        "true_hif_errors": [], "semantic_field_provenance": {"unresolved_signatures": "deployment_diagnostic:nlm"},
    })
    assert observation["unresolved_signatures"] == ["hif_suspected_line_differential", "wls_residual_outlier index=1"]
    assert observation["explained_anomalies"][0]["family"] == "three_phase_unbalance"
    assert set(observation["fresh_context_evidence"]) == {"three_phase", "hif_conditioning"}
    assert observation["nlm_summary"] == {"top_hif_groups": []} and observation["hif_summary"]["estimated"]["alpha"] == .4
    for key in ("hidden_truth", "label", "nlm_diagnostic", "op_point", "gnn_screen", "true_hif_errors"):
        assert key not in observation
    assert observation["semantic_field_provenance"] == {"unresolved_signatures": "deployment_diagnostic:nlm"}


def test_truth_labels_and_precomputed_diagnoses_do_not_affect_execution():
    first = _scenario()
    second = deepcopy(first)
    second.update(scenario_family="harmonic", unresolved_signatures=["harmonic_distortion"],
                  remaining_anomaly_score=-88., no_material_anomaly_remaining=False,
                  oracle_action_hints=[{"tool": "correct_measurements", "arguments": {"suspect_group": [9]}}],
                  hidden_truth={"true_hif_errors": [{"branch_row0": 15, "phase": "C"}]})
    second["metadata"].update(nlm_diagnostic={"top_hif_groups": [{"branch_row0": 15}]}, faulted_model_dir="C:/other",
                              op_point={"load_scale": 1.3}, label={"family": "harmonic"}, hse_summary={"bus": 4})
    second["metadata"]["parameter_scans"]["initial_states"] = [[999.]]
    second["metadata"]["hif_runtime"]["label"] = {"phase": "C"}
    second["metadata"]["hif_scan_window"]["scans"][0]["label"] = {"phase": "C"}
    second["metadata"]["hif_scan_window"]["window_metadata"] = {"operating_point_mode": "fixed"}
    hashes, outputs = [], []
    for scenario in (first, second):
        providers, env = _real_env()
        env.reset(scenario)
        active = env.store.active_state_id
        hashes.append(env.store.state_hash(active))
        _, output = env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
        assert output["execution_status"] == "success", output
        outputs.append(output["tool_metrics"])
    assert hashes[0] == hashes[1]
    assert outputs[0] == outputs[1]
    assert outputs[0]["chi_square_alarm"] or outputs[0]["normalized_residual_alarm"]


# ------------------------------------------------------------------------ gate


def test_gated_tools_are_rejected_before_an_alarm_and_admitted_after():
    providers, env = _real_env()
    state = env.reset(_scenario(currents=False))
    active = state["active_state_id"]
    for tool in sorted(GATED_DIAGNOSTIC_TOOLS):
        _, output = env.step({"tool": tool, "arguments": {"state_id": active, "candidate_branch_row0": 0}})
        assert output["execution_status"] == "failure"
        assert output["error_code"] == "diagnostics_require_wls_alarm", (tool, output)
        assert output["valid_next_actions"] == [{"tool": RUN_WLS, "arguments": {"state_id": active}}]
        direct = env.dispatch_valid_action({"tool": tool, "arguments": {"state_id": active, "candidate_branch_row0": 0}})
        assert direct["error_code"] == "diagnostics_require_wls_alarm"
    assert env.get_policy_observation().available_evidence == []
    _, wls = env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    assert wls["execution_status"] == "success", wls
    assert wls["tool_metrics"]["chi_square_alarm"] or wls["tool_metrics"]["normalized_residual_alarm"]
    assert env._current_wls_alarm(active)
    _, acquired = env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
    assert acquired["execution_status"] == "success", acquired
    assert acquired["tool_metrics"]["three_phase_context_status"] == "available"
    assert acquired["tool_metrics"]["available_evidence_channels"] == ["three_phase_voltages"]
    # The root's persistent scan window is revealed by this acquisition only.
    assert env.get_policy_observation().available_evidence == ["three_phase_voltages", "hif_scan_window"]
    _, screened = env.step({"tool": RUN_THREE_PHASE_NLM_FROM_PATH, "arguments": {"state_id": active}})
    assert screened["execution_status"] == "success", screened
    assert screened["tool_metrics"]["nlm_summary"]["diagnostic_classification"] in {"balanced_three_phase", "unresolved"}
    # Spectra are absent on this root: the request is admitted, the answer is "unavailable".
    _, spectra = env.step({"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": active}})
    assert spectra["execution_status"] == "success", spectra
    assert spectra["tool_metrics"]["harmonic_context_status"] in {"available", "unavailable"}


def test_quiet_wls_keeps_gated_tools_closed():
    providers, env = _real_env()
    state = env.reset(_scenario(anomalous=False, currents=False))
    active = state["active_state_id"]
    _, wls = env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    assert wls["execution_status"] == "success", wls
    assert not wls["tool_metrics"]["chi_square_alarm"] and not wls["tool_metrics"]["normalized_residual_alarm"]
    _, output = env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
    assert output["error_code"] == "diagnostics_require_wls_alarm"
    assert env.get_policy_observation().available_evidence == []


def test_disabled_tool_is_rejected_before_and_after_an_alarm():
    providers, env = _real_env()
    state = env.reset(_scenario(currents=False))
    active = state["active_state_id"]
    for _ in range(2):
        _, output = env.step({"tool": RUN_ALTERNATIVE_TEST, "arguments": {"state_id": active}})
        assert output["execution_status"] == "failure"
        assert output["error_code"] == "tool_disabled_by_evidence_profile"
        assert env.dispatch_valid_action({"tool": RUN_ALTERNATIVE_TEST, "arguments": {"state_id": active}})["error_code"] == "evidence_profile_tool_unavailable"
        env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    with pytest.raises(ValueError, match="unavailable under evidence_profile"):
        env._operator_escalation_audit({"tool": RUN_ALTERNATIVE_TEST, "arguments": {"state_id": active}})


def test_process_gate_reads_alarm_flags_not_wls_signatures():
    oracle = ProcessValidityOracle()

    def state(ledger, signatures=()):
        return {"evidence_profile": WLS_GATED_PROFILE, "active_state_id": "s0", "candidate_state_id": None,
                "has_open_candidate": False, "unresolved_signatures": list(signatures),
                "fresh_context_evidence": {"wls": ledger}, "available_evidence": [], "tried_action_signatures": []}

    action = {"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": "s0"}}
    # A chi-square alarm with no minted wls_* signature is enough.
    assert oracle.check(state(_alarm_ledger(chi=True)), action)["process_valid"]
    assert oracle.check(state(_alarm_ledger(chi=False, residual=True)), action)["process_valid"]
    # wls_* signatures without an alarm flag are not the gate.
    quiet = oracle.check(state(_alarm_ledger(chi=False), ["wls_residual_outlier index=40"]), action)
    assert quiet["error_code"] == "diagnostics_require_wls_alarm"
    assert quiet["valid_next_actions"] == [{"tool": RUN_WLS, "arguments": {"state_id": "s0"}}]
    # The ledger must be bound to the active state's exact contents.
    unbound = deepcopy(_alarm_ledger()); unbound["state_hash"] = ""
    assert oracle.check(state(unbound), action)["error_code"] == "diagnostics_require_wls_alarm"
    other = _alarm_ledger(state_id="s9")
    assert oracle.check(state(other), action)["error_code"] == "diagnostics_require_wls_alarm"
    assert oracle.check(state({"successful": False}), action)["error_code"] == "diagnostics_require_wls_alarm"
    # Disabled tools stay disabled; scada_only refuses the gated tools outright.
    assert oracle.check(state(_alarm_ledger()), {"tool": RUN_ALTERNATIVE_TEST, "arguments": {"state_id": "s0"}})["error_code"] == "tool_disabled_by_evidence_profile"
    scada = {**state(_alarm_ledger()), "evidence_profile": SCADA_ONLY_PROFILE}
    assert oracle.check(scada, action)["error_code"] == "tool_disabled_by_evidence_profile"
    # An inconclusive candidate is gated on its own verification flags.
    candidate = {**state({"successful": False}), "candidate_state_id": "c1", "has_open_candidate": True,
                 "has_verified_candidate": True, "candidate_disposition": "INCONCLUSIVE",
                 "last_verification": {"state_id": "c1", "chi_square_alarm": True}}
    assert current_wls_alarm(candidate, "c1")
    assert oracle.check(candidate, {"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": "c1"}})["process_valid"]
    candidate["last_verification"]["chi_square_alarm"] = False
    assert oracle.check(candidate, {"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": "c1"}})["error_code"] == "diagnostics_require_wls_alarm"


@pytest.mark.parametrize("method", ["get_three_phase_context", "get_harmonic_context", "run_hse",
                                   "run_three_phase_nlm", "estimate_hif", "estimate_hif_multiscan"])
def test_provider_gate_refuses_without_alarm_and_admits_with_alarm(method):
    provider = MatpowerDeploymentProviders()
    closed = getattr(provider, method)(_provider_snapshot(alarm=False), {"arguments": {"candidate_branch_row0": 0}})
    assert closed["execution_status"] == "failure"
    assert closed["error_code"] == "diagnostics_require_wls_alarm"
    signature_only = _provider_snapshot(alarm=False)
    signature_only["policy_observation"]["unresolved_signatures"] = ["wls_residual_outlier index=40"]
    assert getattr(provider, method)(signature_only, {"arguments": {"candidate_branch_row0": 0}})["error_code"] == "diagnostics_require_wls_alarm"
    admitted = getattr(provider, method)(_provider_snapshot(alarm=True), {"arguments": {"candidate_branch_row0": 0}})
    assert admitted.get("error_code") not in GATE_CODES, admitted


def test_provider_three_phase_context_reports_coverage_after_alarm():
    provider = MatpowerDeploymentProviders()
    result = provider.get_three_phase_context(_provider_snapshot(alarm=True), {})
    assert result.get("execution_status", "success") == "success"
    assert result["three_phase_context_status"] == "available"
    assert result["available_evidence_channels"] == ["three_phase_voltages", "three_phase_branch_currents"]


# -------------------------------------------------------------------- channels


def test_channels_are_listed_only_after_acquisition():
    wls, three_phase = RecordingProvider("wls"), RecordingProvider("three_phase")
    env = TransactionalPSSEEnv(wls_runner=wls, evidence_providers={GET_THREE_PHASE_CONTEXT: three_phase})
    state = env.reset(_scenario())
    active = state["active_state_id"]
    assert env.get_policy_observation().available_evidence == []
    env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    assert env.get_policy_observation().available_evidence == []
    env.context_flags.update(has_fresh_parameter_context=True, parameter_context_state_id=active)
    assert env.get_policy_observation().available_evidence == ["parameter_scans"]
    env.context_flags.update(has_fresh_topology_context=True, topology_context_state_id=active)
    assert env.get_policy_observation().available_evidence == ["parameter_scans", "substation_telemetry"]
    _, output = env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
    assert output["execution_status"] == "success", output
    # An answered acquisition retires the balanced contexts taken while it
    # was pending (they must be taken again), so only the acquired phasor
    # channels remain listed until the contexts are refreshed.  The root's
    # persistent scan window is published with the phasors it extends.
    acquired = ["three_phase_voltages", "three_phase_branch_currents", "hif_scan_window"]
    channels = env.get_policy_observation().available_evidence
    assert channels == acquired
    env.context_flags.update(has_fresh_parameter_context=True, parameter_context_state_id=active,
                             has_fresh_topology_context=True, topology_context_state_id=active)
    channels = env.get_policy_observation().available_evidence
    assert channels == ["parameter_scans", "substation_telemetry", *acquired]
    for never in ("hif_runtime", "nlm_diagnostic", "harmonic_measurements"):
        assert never not in channels
    env.context_flags.update(has_fresh_parameter_context=False, has_fresh_topology_context=False)
    assert env.get_policy_observation().available_evidence == acquired


def test_scan_window_channel_is_bound_to_the_acquisition_and_to_the_root():
    # A root without a scan window never lists one, before or after acquisition.
    wls, three_phase = RecordingProvider("wls"), RecordingProvider("three_phase")
    scenario = _scenario()
    del scenario["metadata"]["hif_scan_window"]
    env = TransactionalPSSEEnv(wls_runner=wls, evidence_providers={GET_THREE_PHASE_CONTEXT: three_phase})
    active = env.reset(scenario)["active_state_id"]
    env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
    assert env.get_policy_observation().available_evidence == ["three_phase_voltages", "three_phase_branch_currents"]
    # A root with a window lists nothing until the acquisition succeeded on the active state.
    env = TransactionalPSSEEnv(wls_runner=wls, evidence_providers={GET_THREE_PHASE_CONTEXT: three_phase})
    active = env.reset(_scenario())["active_state_id"]
    env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    assert "hif_scan_window" not in env.get_policy_observation().available_evidence
    env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
    assert "hif_scan_window" in env.get_policy_observation().available_evidence
    # A failed re-acquisition on the same state retires the listing again.
    three_phase.kind = "failing"
    env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
    assert env.get_policy_observation().available_evidence == []


def test_unmeetable_acquisition_obligation_does_not_block_balanced_corrections():
    correction = {"tool": "correct_measurements",
                  "arguments": {"state_id": None, "measurement_updates": {"40": 0.1}}}
    # No auxiliary provider at all: the WLS alarm opens no obligation the
    # environment could discharge, so the correction reaches its executor gate.
    env = TransactionalPSSEEnv(wls_runner=RecordingProvider("wls"))
    active = env.reset(_scenario())["active_state_id"]
    env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    correction["arguments"]["state_id"] = active
    _, output = env.step(correction)
    assert output["error_code"] == "correction_executor_missing", output
    assert env._configured_evidence_tools() == []
    # A configured three-phase provider makes the acquisition obligation real.
    env = TransactionalPSSEEnv(wls_runner=RecordingProvider("wls"),
                               evidence_providers={GET_THREE_PHASE_CONTEXT: RecordingProvider("three_phase")})
    active = env.reset(_scenario())["active_state_id"]
    env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    correction["arguments"]["state_id"] = active
    _, output = env.step(correction)
    assert output["error_code"] == "correction_route_not_actionable"
    assert output["error_detail"] == "measurement_three_phase_evidence_request_pending"
    assert env._configured_evidence_tools() == [GET_THREE_PHASE_CONTEXT]
    # Hand-built oracle states without the declaration keep every obligation.
    oracle = ProcessValidityOracle()
    state = {"evidence_profile": WLS_GATED_PROFILE, "active_state_id": "s0", "candidate_state_id": None,
             "has_open_candidate": False, "unresolved_signatures": ["wls_residual_outlier index=40"],
             "fresh_context_evidence": {"wls": _alarm_ledger()}, "available_evidence": [], "tried_action_signatures": []}
    action = {"tool": "correct_measurements", "arguments": {"state_id": "s0", "measurement_updates": {"40": 0.1}}}
    assert oracle.check(state, action)["error_detail"] == "measurement_harmonic_evidence_request_pending"
    assert oracle.check({**state, "configured_evidence_tools": []}, action)["process_valid"]
    assert oracle.check({**state, "configured_evidence_tools": [GET_THREE_PHASE_CONTEXT]}, action)["error_detail"] == "measurement_three_phase_evidence_request_pending"


def test_admitted_provider_receives_gated_payload_without_truth():
    wls, three_phase = RecordingProvider("wls"), RecordingProvider("three_phase")
    env = TransactionalPSSEEnv(wls_runner=wls, evidence_providers={GET_THREE_PHASE_CONTEXT: three_phase})
    state = env.reset(_scenario())
    active = state["active_state_id"]
    env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    _, output = env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
    assert output["execution_status"] == "success", output
    assert len(three_phase.inputs) == 1
    seen = three_phase.inputs[0]
    assert seen["evidence_profile"] == WLS_GATED_PROFILE
    assert "hidden_truth" not in seen
    metadata = seen["metadata"]
    assert metadata["three_phase_voltages"] and metadata["three_phase_branch_currents"]
    assert metadata["three_phase_sigma"] == PMU_SIGMA and metadata["branch_current_sigma_pu"] == PMU_SIGMA
    for key in (*PRECOMPUTED_DIAGNOSIS_FIELDS, "label", "op_point", "scenario_id"):
        assert key not in metadata, key
    assert "label" not in metadata["hif_runtime"]
    assert all("label" not in scan and "z_clean" not in scan for scan in metadata["hif_scan_window"]["scans"])
    observation = seen["policy_observation"]
    assert observation["evidence_profile"] == WLS_GATED_PROFILE
    assert observation["unresolved_signatures"] == ["wls_residual_outlier index=40 channel=Qinj"]
    assert observation["explained_anomalies"] == []
    assert observation["fresh_context_evidence"]["wls"]["chi_square_alarm"] is True
    for seen_wls in wls.inputs:
        assert seen_wls["evidence_profile"] == WLS_GATED_PROFILE
        assert "nlm_diagnostic" not in seen_wls["metadata"]


def test_missing_diagnostic_provider_fails_closed_after_alarm():
    wls = RecordingProvider("wls")
    env = TransactionalPSSEEnv(wls_runner=wls)
    state = env.reset(_scenario())
    active = state["active_state_id"]
    env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    _, output = env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
    assert output["execution_status"] == "failure"
    assert output["error_code"] == "evidence_provider_missing"


def test_minted_signatures_and_explanations_are_kept_under_the_gated_profile():
    env = TransactionalPSSEEnv()
    state = env.reset(_scenario())
    active = state["active_state_id"]
    env._apply_minted_signatures(RUN_THREE_PHASE_NLM_FROM_PATH, {"minted_signatures": ["hif_suspected_line_differential"]})
    assert "hif_suspected_line_differential" in env.current_state()["unresolved_signatures"]
    env._record_anomaly_explanation(RUN_THREE_PHASE_NLM_FROM_PATH, active, {
        "evidence_source": "deployment_diagnostic:sequence_voltage_unbalance",
        "anomaly_explanation": {"family": "three_phase_unbalance", "kind": "voltage_unbalance_confirmed",
                                "detail": {"bus_1based": 9}}})
    explained = env.current_state()["explained_anomalies"]
    assert explained and explained[0]["family"] == "three_phase_unbalance"
    assert "three_phase_unbalance localized_by_diagnostic" in env.current_state()["unresolved_signatures"]


# ---------------------------------------------------------------- tool schemas


def test_tool_schemas_exclude_only_disabled_tools_and_prompt_states_the_gate():
    every_tool = unified_tool_schemas()
    names = {CANONICAL_TO_INTERNAL_TOOL.get(str(t["function"]["name"]), str(t["function"]["name"])) for t in every_tool}
    gated = tool_schemas_for_observation(every_tool, {"evidence_profile": WLS_GATED_PROFILE})
    gated_names = {CANONICAL_TO_INTERNAL_TOOL.get(str(t["function"]["name"]), str(t["function"]["name"])) for t in gated}
    assert gated_names == names - WLS_GATED_DISABLED_TOOLS
    assert GATED_DIAGNOSTIC_TOOLS <= gated_names
    prompt = system_prompt_for_observation("base", {"evidence_profile": WLS_GATED_PROFILE})
    assert prompt.startswith("base") and "wls_gated_diagnostics" in prompt
    assert "chi-square or normalized-residual alarm" in prompt and "unavailable" in prompt


# ----------------------------------------------------------------- estimators


def test_estimator_uses_the_declared_phasor_sigma_and_never_a_default(monkeypatch):
    captured = {}

    def fake_logic(**kwargs):
        captured.clear(); captured.update(kwargs)
        return {"success": False, "error": "captured"}

    monkeypatch.setattr("psse_env.providers.matpower._estimate_hif_location_magnitude_logic", fake_logic)
    provider = MatpowerDeploymentProviders()
    action = {"arguments": {"candidate_branch_row0": 0}}
    result = provider.estimate_hif(_provider_snapshot(alarm=True), action)
    assert result["error_code"] == "hif_estimation_failure"
    assert captured["three_phase_sigma"] == PMU_SIGMA
    assert captured["branch_current_sigma_pu"] == PMU_SIGMA
    # Declared only through the noise contract.
    contract_only = _provider_snapshot(alarm=True)
    for block in (contract_only["metadata"], contract_only["metadata"]["hif_runtime"], contract_only["metadata"]["hif_scan_window"]):
        block.pop("three_phase_sigma", None); block.pop("branch_current_sigma_pu", None)
    captured.clear()
    provider.estimate_hif(contract_only, action)
    assert captured["three_phase_sigma"] == PMU_SIGMA and captured["branch_current_sigma_pu"] == PMU_SIGMA
    # No declaration anywhere: fail closed, the estimator is never invoked.
    captured.clear()
    undeclared = provider.estimate_hif(_provider_snapshot(alarm=True, declare_sigma=False), action)
    assert undeclared["execution_status"] == "failure"
    assert undeclared["error_code"] in {"three_phase_sigma_undeclared", "branch_current_sigma_pu_undeclared"}
    assert captured == {}
    # The permissive historical profile keeps its explicit legacy default.
    legacy = MatpowerDeploymentProviders(evidence_profile=AUXILIARY_EVIDENCE_PROFILE)
    state = _provider_snapshot(alarm=True, declare_sigma=False)
    state.pop("policy_observation")
    legacy.estimate_hif(state, action)
    assert captured["three_phase_sigma"] == 5e-3 and captured["branch_current_sigma_pu"] == 1e-3


def test_nlm_uses_declared_current_sigma_and_never_the_stored_diagnosis(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("strict NLM must not replay a stored diagnostic or a faulted model")

    monkeypatch.setattr("psse_env.providers.matpower._run_three_phase_nlm_logic", forbidden)
    provider = MatpowerDeploymentProviders()
    # Phasors present with an undeclared current sigma: fail closed.
    undeclared = provider.run_three_phase_nlm(_provider_snapshot(alarm=True, declare_sigma=False), {"arguments": {}})
    assert undeclared["error_code"] == "branch_current_sigma_pu_undeclared"
    # Phasors present and declared: computed from the measurements.  The
    # synthetic currents omit line charging, so at the 1e-4 floor the screen
    # may classify the line as HIF-like and return the terminal-current
    # summary instead of the null test; either shape carries the floor that
    # the declared sigma sets (6 * sqrt(2) * sigma).
    computed = provider.run_three_phase_nlm(_provider_snapshot(alarm=True), {"arguments": {}})
    assert computed.get("execution_status", "success") == "success", computed
    assert computed["evidence_source"] in {
        "deployment_diagnostic:sequence_voltage_unbalance+branch_currents",
        "deployment_diagnostic:terminal_current_differential",
    }
    summary = computed["nlm_summary"]
    floor = (summary.get("line_differential_null") or {}).get("differential_detection_floor_pu",
                                                                 summary.get("differential_detection_floor_pu"))
    from three_phase_nlm.branch_current_analysis import line_differential_null_test
    snapshot = _provider_snapshot(alarm=True)
    rows = snapshot["metadata"]["three_phase_voltages"], snapshot["metadata"]["three_phase_branch_currents"]
    declared_floor = line_differential_null_test(*rows, sigma_pu=PMU_SIGMA)["differential_detection_floor_pu"]
    legacy_floor = line_differential_null_test(*rows, sigma_pu=1e-3)["differential_detection_floor_pu"]
    # The compact summary rounds its floats to four decimals.
    assert floor == pytest.approx(declared_floor, abs=1e-4)
    assert abs(floor - legacy_floor) > 10 * abs(floor - declared_floor)
    # No phasors, only a stored diagnosis and model dirs: fail closed instead of replaying truth.
    truth_only = _provider_snapshot(alarm=True)
    for key in ("three_phase_voltages", "three_phase_branch_currents", "hif_runtime", "hif_scan_window"):
        truth_only["metadata"].pop(key, None)
    truth_only["metadata"].update(nlm_diagnostic={"top_hif_groups": [{"branch_row0": 3}]},
                                  pristine_model_dir="C:/model", faulted_model_dir="C:/truth")
    result = provider.run_three_phase_nlm(truth_only, {"arguments": {}})
    assert result["execution_status"] == "failure"
    assert result["error_code"] == "nlm_runtime_missing"


def test_multiscan_passes_declared_scan_sigmas_and_requires_them(monkeypatch):
    captured = {}

    def fake_logic(**kwargs):
        captured.clear(); captured.update(kwargs)
        return {"success": False, "error": "captured"}

    monkeypatch.setattr("psse_env.providers.matpower._estimate_hif_location_magnitude_multiscan_logic", fake_logic)
    provider = MatpowerDeploymentProviders()
    action = {"arguments": {"candidate_branch_row0": 0}}
    result = provider.estimate_hif_multiscan(_provider_snapshot(alarm=True), action)
    assert result["error_code"] == "hif_multiscan_failure"
    assert captured["require_declared_sigmas"] is True
    assert all(scan["three_phase_sigma"] == PMU_SIGMA for scan in captured["scans"])
    captured.clear()
    undeclared = provider.estimate_hif_multiscan(_provider_snapshot(alarm=True, declare_sigma=False), action)
    assert undeclared["error_code"] == "three_phase_sigma_undeclared"
    assert captured == {}


def test_multiscan_estimator_rejects_undeclared_scan_sigma_when_required():
    from three_phase_nlm.hif_multiscan_estimator import _parse_scans
    scans = [{"scan_index": 0, "z_obs": [1.] * 4, "three_phase_voltages": [], "op_point": {"load_scale": 1.}}]
    with pytest.raises(ValueError, match="three_phase_sigma"):
        _parse_scans(scans=scans, scan_window_path=None, require_declared_sigmas=True)
    parsed, _ = _parse_scans(scans=[{**scans[0], "three_phase_sigma": PMU_SIGMA}], scan_window_path=None,
                             require_declared_sigmas=True)
    assert parsed[0].three_phase_sigma == PMU_SIGMA


def test_mcp_estimator_logic_has_no_silent_sigma_default():
    from mcp_server.matpower_server import _estimate_hif_location_magnitude_logic, _run_three_phase_nlm_logic
    result = _estimate_hif_location_magnitude_logic(case_path="case14", candidate_branch_row0=0, z_obs=[1.] * 122,
                                                    three_phase_voltages=_balanced_voltages([1.] * 14))
    assert result["success"] is False and "three_phase_sigma" in result["error"]
    result = _run_three_phase_nlm_logic(case_path="case14", three_phase_voltages=_balanced_voltages([1.] * 14),
                                        three_phase_branch_currents=_branch_currents())
    assert result["success"] is False and "branch_current_sigma_pu" in result["error"]


# ------------------------------------------------------------------ GNN screen


@pytest.mark.parametrize("profile", [WLS_GATED_PROFILE, SCADA_ONLY_PROFILE])
def test_learned_screen_is_refused_under_strict_profiles(profile):
    with pytest.raises(ValueError, match="screen_checkpoint"):
        MatpowerDeploymentProviders(evidence_profile=profile, screen_checkpoint="x.pt", screen_calibration="c.json")
    assert MatpowerDeploymentProviders(evidence_profile=AUXILIARY_EVIDENCE_PROFILE,
                                       screen_checkpoint="x.pt", screen_calibration="c.json").screen_checkpoint == "x.pt"
