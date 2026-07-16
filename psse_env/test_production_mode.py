from __future__ import annotations

import unittest

from psse_env.actions import (
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    FINALIZE_DIAGNOSIS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    ROLLBACK_STATE,
    RUN_WLS,
)
from psse_env.dagger.rollout_collector import (
    DaggerRolloutCollector,
    audit_target_aware_state_classes,
    classify_state_example,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.state_store import OracleState, PolicyObservation
from psse_env.transactional_env import TransactionalPSSEEnv


REQUIRED_ADAPTERS = {
    RUN_WLS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
}


def _deterministic_adapter(function):
    function.provider_kind = "deterministic"
    return function


@_deterministic_adapter
def _wls_adapter(state):
    measurements = list(state.get("measurements") or [])
    remaining = sum(value != expected for value, expected in zip(measurements, [1.0]))
    return {
        "wls_objective": float(remaining),
        "remaining_anomaly_score": float(remaining),
        "anomaly_threshold": 0.5,
        "target_progress": 1.0 if state.get("parent_state_id") and remaining == 0 else 0.0,
        "global_progress": 1.0 if state.get("parent_state_id") and remaining == 0 else 0.0,
        "remaining_fault_count": remaining,
        "globally_resolved": remaining == 0,
        "unresolved_signatures": [] if remaining == 0 else ["measurement_residual_outlier"],
        "converged": True,
    }


@_deterministic_adapter
def _context_adapter(state):
    return {
        "context_rows": [{"state_hash": state["state_hash"]}],
        "unresolved_signatures": ["measurement_residual_outlier"],
        "supported_corrections": [
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": state["state_id"],
                    "measurement_updates": {0: 1.0},
                },
            },
        ],
    }


@_deterministic_adapter
def _correction_adapter(state, action):
    del state
    return {
        "modification": {
            key: value
            for key, value in action["arguments"].items()
            if key not in {"state_id", "candidate_state_id"}
        },
        "executor_receipt": "reviewed_deterministic_adapter",
    }


def _production_env(*, wls=_wls_adapter, contexts=None) -> TransactionalPSSEEnv:
    context_providers = contexts or {
        GET_MEASUREMENT_CONTEXT: _context_adapter,
        GET_PARAMETER_CONTEXT: _context_adapter,
        GET_TOPOLOGY_CONTEXT: _context_adapter,
    }
    return TransactionalPSSEEnv(
        production_dataset_mode=True,
        approved_deterministic_providers=REQUIRED_ADAPTERS,
        wls_runner=wls,
        context_providers=context_providers,
        correction_executors={
            CORRECT_MEASUREMENTS: _correction_adapter,
            CORRECT_PARAMETERS: _correction_adapter,
            CORRECT_TOPOLOGY: _correction_adapter,
        },
        max_steps=8,
    )


def _measurement_scenario():
    return {
        "scenario_id": "production-measurement",
        "case": {},
        "measurements": [9.0],
        "clean_measurements": [1.0],
        "true_measurement_errors": [{"index": 0, "clean": 1.0}],
        "oracle_action_hints": [
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": "s0", "measurement_updates": {0: 1.0}},
            }
        ],
    }


class ProductionConfigurationTests(unittest.TestCase):
    def test_missing_provider_error_lists_actionable_names(self):
        with self.assertRaisesRegex(ValueError, "run_wls") as raised:
            TransactionalPSSEEnv(production_dataset_mode=True)
        message = str(raised.exception)
        for name in REQUIRED_ADAPTERS:
            self.assertIn(name, message)

    def test_unclassified_callables_are_not_assumed_real(self):
        def unclassified(*args):
            del args
            return {"wls_objective": 1.0}

        with self.assertRaisesRegex(ValueError, "must declare provider_kind"):
            TransactionalPSSEEnv(
                production_dataset_mode=True,
                wls_runner=unclassified,
                context_providers={tool: unclassified for tool in (
                    GET_MEASUREMENT_CONTEXT,
                    GET_PARAMETER_CONTEXT,
                    GET_TOPOLOGY_CONTEXT,
                )},
                correction_executors={tool: unclassified for tool in (
                    CORRECT_MEASUREMENTS,
                    CORRECT_PARAMETERS,
                    CORRECT_TOPOLOGY,
                )},
            )

    def test_deterministic_adapters_require_explicit_approval(self):
        with self.assertRaisesRegex(ValueError, "without explicit approval"):
            TransactionalPSSEEnv(
                production_dataset_mode=True,
                wls_runner=_wls_adapter,
                context_providers={
                    GET_MEASUREMENT_CONTEXT: _context_adapter,
                    GET_PARAMETER_CONTEXT: _context_adapter,
                    GET_TOPOLOGY_CONTEXT: _context_adapter,
                },
                correction_executors={
                    CORRECT_MEASUREMENTS: _correction_adapter,
                    CORRECT_PARAMETERS: _correction_adapter,
                    CORRECT_TOPOLOGY: _correction_adapter,
                },
            )

    def test_approved_production_pilot_collects_evidence_backed_commit(self):
        class RunWLSPolicy:
            def act(self, observation):
                return {"tool": RUN_WLS, "arguments": {"state_id": observation["active_state_id"]}}

        class FirstChoiceRandom:
            @staticmethod
            def random():
                return 0.0

            @staticmethod
            def choice(items):
                return items[0]

        env = _production_env()
        rows = DaggerRolloutCollector(
            env=env,
            policy=RunWLSPolicy(),
            expert_oracle=ExpertPolicyOracle(),
            rng=FirstChoiceRandom(),
        ).collect_iteration(
            scenarios=[_measurement_scenario()], iteration=0, beta=1.0, max_steps=5
        )
        self.assertEqual(
            [row["preferred_action"]["tool"] for row in rows],
            [
                RUN_WLS,
                GET_MEASUREMENT_CONTEXT,
                CORRECT_MEASUREMENTS,
                RUN_WLS,
                COMMIT_STATE,
            ],
        )
        self.assertEqual(rows[-1]["state_class"], "accepted_final_commit")
        self.assertTrue(
            all(
                row["dataset_mode"] == "production"
                and row["labels"]["dataset_mode"] == "production"
                for row in rows
            )
        )
        self.assertTrue(audit_target_aware_state_classes(rows)["passed"])
        provenance = env.get_policy_observation().semantic_field_provenance
        self.assertTrue(provenance["remaining_anomaly_score"].startswith("configured_provider:"))
        self.assertTrue(provenance["no_material_anomaly_remaining"].startswith("configured_provider:"))


class ProductionEvidenceTests(unittest.TestCase):
    def test_measurement_context_requirement_cannot_encode_scenario_family(self):
        env = _production_env()
        for supplied in (False, True):
            scenario = _measurement_scenario()
            scenario["requires_measurement_context"] = supplied
            env.reset(scenario)
            self.assertTrue(env.get_policy_observation().requires_measurement_context)

    def test_domain_labels_require_observable_family_evidence_and_fresh_context(self):
        env = _production_env()
        state = env.reset(_measurement_scenario())
        with self.assertRaisesRegex(ValueError, "observable measurement evidence"):
            env.assert_training_decision_evidence(
                {
                    "tool": GET_MEASUREMENT_CONTEXT,
                    "arguments": {"state_id": state["active_state_id"]},
                }
            )
        with self.assertRaisesRegex(ValueError, "fresh observable measurement context"):
            env.assert_training_decision_evidence(
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": state["active_state_id"],
                        "measurement_updates": {0: 1.0},
                    },
                }
            )
        env.step({"tool": RUN_WLS, "arguments": {"state_id": state["active_state_id"]}})
        env.assert_training_decision_evidence(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": state["active_state_id"]},
            }
        )
        env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": state["active_state_id"]},
            }
        )
        supported = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {
                "state_id": state["active_state_id"],
                "measurement_updates": {0: 1.0},
            },
        }
        env.assert_training_decision_evidence(supported)
        unsupported = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {
                "state_id": state["active_state_id"],
                "measurement_updates": {0: 2.0},
            },
        }
        with self.assertRaisesRegex(ValueError, "not supported by the latest"):
            env.assert_training_decision_evidence(unsupported)

    def test_placeholder_or_empty_wls_evidence_fails_without_verifying_candidate(self):
        @_deterministic_adapter
        def placeholder_wls(state):
            del state
            return {"wls_objective": 0.0, "evidence_source": "placeholder_fallback"}

        env = _production_env(wls=placeholder_wls)
        root = env.reset(_measurement_scenario())
        env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": root["active_state_id"]},
            }
        )
        candidate, _ = env.step(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": root["active_state_id"],
                    "measurement_updates": {0: 1.0},
                },
            }
        )
        candidate_id = candidate["candidate_state_id"]
        state, output = env.step({"tool": RUN_WLS, "arguments": {"state_id": candidate_id}})
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "insufficient_observable_evidence")
        self.assertTrue(state["has_unverified_candidate"])
        self.assertFalse(state["has_verified_candidate"])

    def test_oracle_sourced_wls_evidence_is_rejected(self):
        @_deterministic_adapter
        def oracle_wls(state):
            del state
            return {"wls_objective": 0.0, "evidence_source": "oracle_hint"}

        env = _production_env(wls=oracle_wls)
        root = env.reset(_measurement_scenario())
        _, output = env.step(
            {"tool": RUN_WLS, "arguments": {"state_id": root["active_state_id"]}}
        )
        self.assertEqual(output["execution_status"], "failure")
        self.assertIn("non_observable_evidence_source", output["error_detail"])

    def test_empty_context_output_is_not_fresh_evidence(self):
        @_deterministic_adapter
        def empty_context(state):
            del state
            return {}

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: empty_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        root = env.reset(_measurement_scenario())
        state, output = env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": root["active_state_id"]},
            }
        )
        self.assertEqual(output["error_code"], "insufficient_observable_evidence")
        self.assertFalse(state["has_fresh_measurement_context"])

    def _manually_verified_candidate(self, disposition):
        env = _production_env()
        root = env.reset(_measurement_scenario())
        env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": root["active_state_id"]},
            }
        )
        state, _ = env.step(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": root["active_state_id"],
                    "measurement_updates": {0: 1.0},
                },
            }
        )
        candidate_id = state["candidate_state_id"]
        env.store.mark_verified(
            candidate_id,
            {
                "state_id": candidate_id,
                "state_hash": env.store.state_hash(candidate_id),
                "evidence_source": "configured_provider:test_adapter",
            },
            disposition,
        )
        return env, candidate_id

    def test_commit_without_decision_metrics_fails_closed(self):
        env, candidate_id = self._manually_verified_candidate("ACCEPT_FINAL")
        before = env.store.episode_hash()
        with self.assertRaisesRegex(ValueError, "decision_metrics_missing"):
            env.assert_training_decision_evidence(
                {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}}
            )
        state, output = env.step(
            {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}}
        )
        self.assertEqual(output["error_code"], "insufficient_observable_evidence")
        self.assertEqual(env.store.episode_hash(), before)
        self.assertEqual(state["candidate_state_id"], candidate_id)

    def test_rollback_without_decision_metrics_fails_closed(self):
        env, candidate_id = self._manually_verified_candidate("REJECT")
        before = env.store.episode_hash()
        with self.assertRaisesRegex(ValueError, "decision_metrics_missing"):
            env.assert_training_decision_evidence(
                {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}}
            )
        state, output = env.step(
            {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}}
        )
        self.assertEqual(output["error_code"], "insufficient_observable_evidence")
        self.assertEqual(env.store.episode_hash(), before)
        self.assertEqual(state["candidate_state_id"], candidate_id)

    def test_synthetic_pilot_mode_keeps_placeholder_compatibility(self):
        env = TransactionalPSSEEnv()
        root = env.reset(_measurement_scenario())
        _, output = env.step({"tool": RUN_WLS, "arguments": {"state_id": root["active_state_id"]}})
        self.assertEqual(output["execution_status"], "success")

    def test_finalize_label_requires_observable_terminal_evidence(self):
        env = _production_env()
        scenario = _measurement_scenario()
        scenario["oracle_terminal_eligible"] = True
        env.reset(scenario)
        with self.assertRaisesRegex(ValueError, "observable terminal evidence"):
            env.assert_training_decision_evidence(
                {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
            )

        clean = _measurement_scenario()
        clean["measurements"] = [1.0]
        clean["true_measurement_errors"] = []
        clean["oracle_action_hints"] = []
        state = env.reset(clean)
        env.step({"tool": RUN_WLS, "arguments": {"state_id": state["active_state_id"]}})
        env.assert_training_decision_evidence(
            {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
        )


class SemanticProvenanceTests(unittest.TestCase):
    def test_hidden_truth_semantic_initializer_is_rejected(self):
        env = _production_env()
        scenario = _measurement_scenario()
        scenario["hidden_truth"] = {"remaining_anomaly_score": 0.0}
        with self.assertRaisesRegex(ValueError, "hidden_truth"):
            env.reset(scenario)

    def test_production_semantic_initializer_requires_observable_provenance(self):
        env = _production_env()
        scenario = _measurement_scenario()
        scenario["unresolved_signatures"] = ["measurement_residual_outlier"]
        with self.assertRaisesRegex(ValueError, "requires semantic_field_provenance"):
            env.reset(scenario)

        scenario["semantic_field_provenance"] = {
            "unresolved_signatures": "oracle_hint"
        }
        with self.assertRaisesRegex(ValueError, "non-observable provenance"):
            env.reset(scenario)

    def test_observable_semantic_initializer_propagates_provenance(self):
        env = _production_env()
        scenario = _measurement_scenario()
        scenario["unresolved_signatures"] = ["measurement_residual_outlier"]
        scenario["semantic_field_provenance"] = {
            "unresolved_signatures": "observable_input:scada_residuals"
        }
        env.reset(scenario)
        observation = env.get_policy_observation()
        self.assertEqual(observation.unresolved_signatures, ["measurement_residual_outlier"])
        self.assertEqual(
            observation.semantic_field_provenance["unresolved_signatures"],
            "observable_input:scada_residuals",
        )


class TeacherRealizabilityAndReplayTests(unittest.TestCase):
    def test_hidden_fault_family_cannot_change_first_diagnostic_action(self):
        observation = PolicyObservation(active_state_id="active", remaining_budget=8)
        states = [
            OracleState(
                policy_observation=observation,
                true_measurement_errors=[{"index": 0}],
                oracle_action_hints=[
                    {
                        "tool": GET_MEASUREMENT_CONTEXT,
                        "arguments": {"state_id": "active"},
                    }
                ],
            ),
            OracleState(
                policy_observation=observation,
                true_parameter_errors=[{"branch_row0": 0}],
                oracle_action_hints=[
                    {
                        "tool": GET_PARAMETER_CONTEXT,
                        "arguments": {"state_id": "active"},
                    }
                ],
            ),
            OracleState(
                policy_observation=observation,
                true_topology_errors=[{"branch_row0": 0}],
                oracle_action_hints=[
                    {
                        "tool": GET_TOPOLOGY_CONTEXT,
                        "arguments": {"state_id": "active"},
                    }
                ],
            ),
        ]
        first_tools = [ExpertPolicyOracle().next_actions(state)[0]["tool"] for state in states]
        self.assertEqual(first_tools, [RUN_WLS, RUN_WLS, RUN_WLS])

    def test_ambiguous_observable_signatures_use_deterministic_context_tiebreak(self):
        observation = PolicyObservation(
            active_state_id="active",
            last_tool=RUN_WLS,
            last_tool_status="success",
            unresolved_signatures=[
                "measurement_residual:index=0",
                "parameter_reactance:branch=L1",
            ],
            remaining_anomaly_score=2.0,
            remaining_budget=7,
        )
        measurement = OracleState(
            policy_observation=observation,
            true_measurement_errors=[{"index": 0}],
            oracle_action_hints=[
                {
                    "tool": GET_MEASUREMENT_CONTEXT,
                    "arguments": {"state_id": "active"},
                }
            ],
        )
        parameter = OracleState(
            policy_observation=observation,
            true_parameter_errors=[{"branch_row0": 0}],
            oracle_action_hints=[
                {
                    "tool": GET_PARAMETER_CONTEXT,
                    "arguments": {"state_id": "active"},
                }
            ],
        )
        tools = [
            ExpertPolicyOracle().next_actions(state)[0]["tool"]
            for state in (measurement, parameter)
        ]
        self.assertEqual(tools, [GET_PARAMETER_CONTEXT, GET_PARAMETER_CONTEXT])

    def test_hidden_terminal_flag_cannot_choose_finalize_label(self):
        state = OracleState(
            policy_observation=PolicyObservation(
                active_state_id="active",
                remaining_budget=8,
            ),
            hidden_truth={"oracle_terminal_eligible": True},
        )
        self.assertEqual(ExpertPolicyOracle().next_actions(state)[0]["tool"], RUN_WLS)

    def test_observable_context_supplies_correction_without_oracle_hint(self):
        action = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {
                "state_id": "active",
                "measurement_updates": {0: 1.0},
            },
        }
        state = OracleState(
            policy_observation=PolicyObservation(
                active_state_id="active",
                last_tool=GET_MEASUREMENT_CONTEXT,
                last_tool_status="success",
                last_tool_output={
                    "execution_status": "success",
                    "tool_metrics": {"supported_corrections": [action]},
                },
                unresolved_signatures=["measurement_residual_outlier"],
                has_fresh_measurement_context=True,
                measurement_context_state_id="active",
                requires_measurement_context=True,
                remaining_budget=6,
            )
        )
        self.assertEqual(ExpertPolicyOracle().next_actions(state)[0], action)

    def test_target_aware_classes_distinguish_all_decisions_and_recovery(self):
        base = {"active_state_id": "active"}
        cases = (
            (
                {"tool": "finalize_diagnosis", "arguments": {}},
                "ACCEPT_FINAL",
                "terminal_decision",
            ),
            (
                {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": "candidate"}},
                "REJECT",
                "rejected_candidate_recovery",
            ),
            (
                {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": "candidate"}},
                "ACCEPT_FINAL",
                "accepted_final_commit",
            ),
            (
                {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": "candidate"}},
                "ACCEPT_PARTIAL",
                "accepted_partial_commit",
            ),
        )
        rows = []
        for index, (action, disposition, expected) in enumerate(cases):
            actual = classify_state_example(
                base,
                {"process_valid": False},
                preferred_action=action,
                target_candidate_disposition=disposition,
            )
            self.assertEqual(actual, expected)
            rows.append(
                {
                    "example_id": f"row-{index}",
                    "policy_observation": base,
                    "preferred_action": action,
                    "transition_label": {"process_valid": False},
                    "labels": {
                        "target_candidate_disposition": disposition,
                        "state_class": actual,
                    },
                    "state_class": actual,
                }
            )
        invalid = classify_state_example(
            {
                **base,
                "last_tool_status": "failure",
                "last_tool_output": {"error_code": "missing_precondition"},
            },
            {"process_valid": True},
            preferred_action={"tool": RUN_WLS, "arguments": {"state_id": "active"}},
        )
        self.assertEqual(invalid, "invalid_precondition_recovery")
        self.assertTrue(audit_target_aware_state_classes(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
