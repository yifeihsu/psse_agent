"""SCADA-only execution cannot see or react to auxiliary waveform artifacts."""
from __future__ import annotations

from copy import deepcopy
import unittest

from psse_env.evidence_profile import (
    AUXILIARY_EVIDENCE_PROFILE, SCADA_ONLY_PROFILE, SCADA_DISABLED_TOOLS,
    SCADA_DISABLED_REQUESTS, sanitize_scada_execution, sanitize_scada_metadata,
    sanitize_scada_observation,
)
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import build_measurement_vector
from psse_env.transactional_env import TransactionalPSSEEnv
from mcp_server.matpower_server import _load_python_case
from three_phase_nlm.measurement_noise import generated_noise_contract


def _scenario():
    sigma = [.001] * 14 + [.01] * 108
    measurements = build_measurement_vector(_load_python_case("case14")).tolist()
    measurements[40] += .2
    return {
        "scenario_id": "same_observable_acquisition", "scenario_family": "measurement+hif",
        "case": "case14", "measurements": measurements,
        "unresolved_signatures": ["hif_suspected_zero_sequence"],
        "remaining_anomaly_score": 999., "no_material_anomaly_remaining": True,
        "semantic_field_provenance": {"unresolved_signatures": "deployment_sensor:waveform_capture",
            "remaining_anomaly_score": "deployment_sensor:waveform_capture",
            "no_material_anomaly_remaining": "deployment_sensor:waveform_capture"},
        "oracle_action_hints": [{"tool": "correct_parameters", "arguments": {"line_index": 3}}],
        "hidden_truth": {"true_hif_errors": [{"branch_row0": 3, "phase": "A"}]},
        "metadata": {
            "sigma_z": sigma,
            "noise_contract": generated_noise_contract(sigma, noise_scale=1., three_phase_sigma=.005, branch_current_sigma_pu=.001),
            "parameter_scans": {"z_scans": [measurements, measurements], "sigma_z": sigma,
                "initial_states": [[7.] * 28], "op_point": {"load_scale": 7.}, "labels": {"true_line": 3}},
            "three_phase_voltages": [{"bus": "b3", "vln_pu": [9., 2., 1.]}],
            "three_phase_branch_currents": [{"branch_row0": 3, "i_from_pu": [9., 2., 1.]}],
            "harmonic_measurements": {"source_bus": 3, "phasors": [9., 2., 1.]},
            "hif_runtime": {"z_obs": [777.] * 122, "op_point": {"load_scale": 8.}},
            "hif_scan_window": {"scans": [{"label": {"phase": "A"}}]},
            "nlm_diagnostic": {"top_hif_groups": [{"branch_row0": 3}]},
            "op_point": {"load_scale": 8.}, "label": {"family": "hif"},
            "reported_breaker_status": {"CB3": False}, "substation_telemetry": {"secret": 9},
        },
    }


class RecordingProvider:
    provider_kind = "deployment"

    def __init__(self, kind):
        self.kind, self.inputs = kind, []

    def __call__(self, state, action=None):
        self.inputs.append(deepcopy(state))
        binding = {"state_id": state["state_id"], "state_hash": state["state_hash"],
                   "evidence_source": "deployment_wls:scada_fixture"}
        if self.kind == "wls":
            return {**binding, "wls_objective": 10., "remaining_anomaly_score": 2.,
                "no_material_anomaly_remaining": False,
                "unresolved_signatures": ["wls_residual_outlier index=40 channel=Qinj"]}
        if self.kind == "correction":
            return {**binding, "execution_status": "failure", "error_code": "measurement_correction_failure"}
        return {**binding, "supported_corrections": [], "route_status": "complete_negative"}


class SCADAEvidenceBoundaryTests(unittest.TestCase):
    """Invariants of the literal scada_only profile (the research default is wls_gated_diagnostics)."""

    def test_scada_only_reset_removes_seeded_diagnoses_and_auxiliary_store_data(self):
        scenario = _scenario()
        original = deepcopy(scenario)
        env = TransactionalPSSEEnv(evidence_profile=SCADA_ONLY_PROFILE)
        state = env.reset(scenario)
        self.assertEqual(env.evidence_profile, SCADA_ONLY_PROFILE)
        self.assertEqual(state["unresolved_signatures"], [])
        self.assertIsNone(state["remaining_anomaly_score"])
        self.assertFalse(state["no_material_anomaly_remaining"])
        self.assertEqual(env.get_policy_observation().available_evidence, [])
        stored = env.store.get_state(state["active_state_id"])
        metadata = stored["metadata"]
        for key in ("hif_runtime", "hif_scan_window", "nlm_diagnostic", "three_phase_voltages",
                    "three_phase_branch_currents", "harmonic_measurements", "op_point", "label",
                    "reported_breaker_status", "substation_telemetry"):
            self.assertNotIn(key, metadata)
        self.assertEqual(set(metadata["noise_contract"]["channels"]), {"scada"})
        self.assertIn("applied_sigma_per_component", metadata["noise_contract"]["channels"]["scada"])
        self.assertEqual(set(metadata["parameter_scans"]), {"z_scans", "sigma_z"})
        self.assertEqual(stored["measurements"], scenario["measurements"])
        self.assertEqual(scenario, original)

    def test_all_auxiliary_tools_and_hif_handoffs_are_blocked_before_provider_calls(self):
        spy = RecordingProvider("context")
        env = TransactionalPSSEEnv(evidence_profile=SCADA_ONLY_PROFILE,
            evidence_providers={name: spy for name in [*SCADA_DISABLED_TOOLS, "ask_for_more_evidence"]})
        state = env.reset(_scenario())
        for tool in SCADA_DISABLED_TOOLS:
            with self.subTest(tool=tool):
                _, output = env.step({"tool": tool, "arguments": {"state_id": state["active_state_id"]}})
                self.assertEqual(output["execution_status"], "failure")
                self.assertTrue(all(action["tool"] not in SCADA_DISABLED_TOOLS for action in output["valid_next_actions"]))
                direct = env.dispatch_valid_action({"tool": tool, "arguments": {"state_id": state["active_state_id"]}})
                self.assertEqual(direct["error_code"], "evidence_profile_tool_unavailable")
        for request in SCADA_DISABLED_REQUESTS:
            _, output = env.step({"tool": "ask_for_more_evidence", "arguments": {"state_id": state["active_state_id"], "request": request}})
            self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(spy.inputs, [])
        self.assertFalse(env.is_terminal())

    def test_every_provider_dispatch_receives_the_sanitized_payload(self):
        wls, context, correction, evidence = [RecordingProvider(kind) for kind in ("wls", "context", "correction", "evidence")]
        env = TransactionalPSSEEnv(evidence_profile=SCADA_ONLY_PROFILE, wls_runner=wls,
            context_providers={"get_measurement_context": context},
            correction_executors={"correct_measurements": correction},
            evidence_providers={"ask_for_more_evidence": evidence})
        state = env.reset(_scenario())
        active = state["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active}})
        env._step_context({"tool": "get_measurement_context", "arguments": {"state_id": active}})
        env._step_correction({"tool": "correct_measurements", "arguments": {"state_id": active, "suspect_group": [40]}})
        env.dispatch_valid_action({"tool": "ask_for_more_evidence", "arguments": {"state_id": active, "request": "additional_balanced_residual_evidence"}})
        for provider in (wls, context, correction, evidence):
            self.assertEqual(len(provider.inputs), 1)
            seen = provider.inputs[0]
            self.assertEqual(seen["evidence_profile"], SCADA_ONLY_PROFILE)
            self.assertEqual(seen["policy_observation"]["evidence_profile"], SCADA_ONLY_PROFILE)
            self.assertEqual(seen["policy_observation"]["explained_anomalies"], [])
            self.assertNotIn("hidden_truth", seen)
            self.assertEqual(seen["metadata"], sanitize_scada_metadata(_scenario()["metadata"]))
        self.assertIn("wls_residual_outlier index=40 channel=Qinj", env.current_state()["unresolved_signatures"])
        self.assertIsInstance(env.current_state()["semantic_field_provenance"]["unresolved_signatures"], str)

    def test_scada_signature_filter_preserves_field_provenance_string_types(self):
        value = {"unresolved_signatures": ["hif_suspected", "wls_residual_outlier index=40"],
                 "semantic_field_provenance": {"unresolved_signatures": "deployment_wls:lagrangian_port",
                    "remaining_anomaly_score": "observable_candidate_verification"}}
        result = sanitize_scada_observation(value)
        self.assertEqual(result["unresolved_signatures"], ["wls_residual_outlier index=40"])
        self.assertEqual(result["semantic_field_provenance"], value["semantic_field_provenance"])

    def test_stale_auxiliary_explanations_and_conditioning_cannot_reappear(self):
        env = TransactionalPSSEEnv(evidence_profile=SCADA_ONLY_PROFILE)
        state = env.reset(_scenario())
        env.context_flags.update(explained_anomalies=[{"family": "hif", "detail": {"conditioning_fit": {"success": True}}}],
            unresolved_signatures=["hif_suspected", "wls_residual_outlier index=40"],
            fresh_context_evidence={"hif_conditioning": {"status": "ready"}, "three_phase": {"request_attempted": True}})
        current = env.current_state()
        self.assertEqual(current["explained_anomalies"], [])
        self.assertEqual(current["unresolved_signatures"], ["wls_residual_outlier index=40"])
        self.assertNotIn("hif_conditioning", current["fresh_context_evidence"])
        env._record_anomaly_explanation("ask_for_more_evidence", state["active_state_id"],
            {"anomaly_explanation": {"family": "hif", "detail": {"conditioning_fit": {"success": True}}}})
        self.assertEqual(env.current_state()["explained_anomalies"], [])
        self.assertFalse(env._validated_hif_meter_partial_acceptance({}, {}, {}))

    def test_real_wls_and_expert_actions_are_invariant_to_all_auxiliary_and_truth_labels(self):
        first = _scenario()
        second = deepcopy(first)
        second.update(scenario_family="harmonic", unresolved_signatures=["harmonic_distortion"],
            remaining_anomaly_score=-88., no_material_anomaly_remaining=False,
            oracle_action_hints=[{"tool": "correct_measurements", "arguments": {"suspect_group": [9]}}],
            hidden_truth={"true_hif_errors": [{"branch_row0": 15, "phase": "C"}]})
        for key in ("three_phase_voltages", "three_phase_branch_currents", "harmonic_measurements",
                    "hif_runtime", "hif_scan_window", "nlm_diagnostic", "op_point", "label",
                    "reported_breaker_status", "substation_telemetry"):
            second["metadata"][key] = {"changed_auxiliary_or_label": 999}
        second["metadata"]["parameter_scans"]["initial_states"] = [[999.]]
        states, outputs, public_actions, private_actions, hashes = [], [], [], [], []
        for scenario in (first, second):
            providers = MatpowerDeploymentProviders(evidence_profile=SCADA_ONLY_PROFILE,
                                                    chi2_alpha=.01, normalized_residual_threshold=4.)
            env = TransactionalPSSEEnv(**providers.env_kwargs(), production_dataset_mode=True)
            env.reset(scenario)
            active = env.store.active_state_id
            hashes.append(env.store.state_hash(active))
            state, output = env.step({"tool": "run_wls", "arguments": {"state_id": active}})
            self.assertEqual(output["execution_status"], "success", output)
            states.append(state)
            outputs.append(output["tool_metrics"])
            oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
            public_actions.append(oracle.next_actions(env.get_policy_observation()))
            private_actions.append(oracle.next_actions(env.get_oracle_state()))
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(public_actions[0], public_actions[1])
        self.assertEqual(private_actions, public_actions)
        self.assertTrue(all(action["tool"] not in SCADA_DISABLED_TOOLS for action in public_actions[0]))

    def test_auxiliary_compatibility_requires_explicit_configuration(self):
        providers = MatpowerDeploymentProviders(evidence_profile=AUXILIARY_EVIDENCE_PROFILE)
        env = TransactionalPSSEEnv(**providers.env_kwargs())
        state = env.reset(_scenario())
        self.assertEqual(env.get_policy_observation().evidence_profile, AUXILIARY_EVIDENCE_PROFILE)
        self.assertIn("hif_suspected_zero_sequence", state["unresolved_signatures"])
        self.assertIn("hif_runtime", env.store.get_state(state["active_state_id"])["metadata"])

    def test_execution_sanitizer_has_no_private_truth_or_seeded_semantics(self):
        output = sanitize_scada_execution(_scenario())
        for key in ("hidden_truth", "scenario_family", "oracle_action_hints", "unresolved_signatures",
                    "remaining_anomaly_score", "no_material_anomaly_remaining", "semantic_field_provenance"):
            self.assertNotIn(key, output)
        self.assertEqual(output["evidence_profile"], SCADA_ONLY_PROFILE)


if __name__ == "__main__":
    unittest.main()
