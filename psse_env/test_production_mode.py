from __future__ import annotations

import unittest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    FINALIZE_DIAGNOSIS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_WLS,
)
from psse_env.dagger.rollout_collector import (
    DaggerRolloutCollector,
    audit_target_aware_state_classes,
    classify_state_example,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.state_store import (
    OracleState,
    PolicyObservation,
    find_forbidden_policy_paths,
)
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
        "remaining_suspect_count": remaining,
        "globally_resolved": remaining == 0,
        "physical_constraints_ok": True,
        "new_constraint_violations": 0,
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
    @staticmethod
    def _parent_metric_env():
        env = _production_env()
        env.reset(
            {
                "scenario_id": "parent-metric",
                "case": {"branch": [{}]},
                "measurements": [1.0],
            }
        )
        state_id = env.current_state()["active_state_id"]
        action = {
            "tool": CORRECT_PARAMETERS,
            "arguments": {"state_id": state_id, "line_index": 1},
        }
        metrics = {
            "state_id": state_id,
            "state_hash": env.store.state_hash(state_id),
            "evidence_source": "deployment_context:test",
            "supported_corrections": [action],
            "parameter_findings": [{"line_row0": 0, "value": 5.0}],
        }
        env.history = [
            {
                "action": {
                    "tool": GET_PARAMETER_CONTEXT,
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": metrics,
                },
            }
        ]
        return env, state_id, action, metrics

    def test_parent_target_metric_rejects_malformed_supported_action(self):
        env, state_id, action, metrics = self._parent_metric_env()
        metrics["supported_corrections"].append(
            {
                "tool": CORRECT_PARAMETERS,
                "arguments": {
                    "state_id": state_id,
                    "line_index": 2,
                    "value": object(),
                },
            }
        )

        self.assertIsNone(env._parent_target_metric_value(state_id, action))

    def test_parent_target_metric_rejects_malformed_or_nonfinite_finding(self):
        for finding in (
            {"line_row0": "bad", "value": 5.0},
            {"line_row0": 0, "value": object()},
            {"line_row0": 0, "value": float("nan")},
        ):
            with self.subTest(finding=finding):
                env, state_id, action, metrics = self._parent_metric_env()
                metrics["parameter_findings"] = [finding]
                self.assertIsNone(
                    env._parent_target_metric_value(state_id, action)
                )

    def test_wls_convergence_is_not_physical_safety_evidence(self):
        metrics = {
            "converged": True,
            "state_estimation_converged": True,
            "new_constraint_violations": 0,
            "globally_resolved": True,
        }
        gaps = TransactionalPSSEEnv._target_decision_evidence_missing(
            metrics, "ACCEPT_FINAL"
        )
        self.assertIn("physical_constraint_evidence_missing", gaps)

        metrics["power_flow_converged"] = True
        gaps = TransactionalPSSEEnv._target_decision_evidence_missing(
            metrics, "ACCEPT_FINAL"
        )
        self.assertNotIn("physical_constraint_evidence_missing", gaps)

    def test_topology_structural_ambiguity_is_explicit_rejection_evidence(self):
        metrics = {
            "target_fixed": True,
            "target_progress": 1.0,
            "global_progress": 0.86,
            "globally_resolved": False,
            "physical_constraints_ok": True,
            "topology_target_status_matches_requested": True,
            "topology_target_branch_multiplier": 18.0,
            "topology_target_branch_multiplier_threshold": 3.0,
        }

        gaps = TransactionalPSSEEnv._target_decision_evidence_missing(
            metrics,
            "REJECT",
            min_partial_global_progress=0.30,
            min_topology_structural_global_progress=0.95,
            max_branch_target_threshold_ratio=1.25,
        )

        self.assertNotIn("rejection_evidence_missing", gaps)

        metrics["global_progress"] = 0.96
        gaps = TransactionalPSSEEnv._target_decision_evidence_missing(
            metrics,
            "REJECT",
            min_partial_global_progress=0.30,
            min_topology_structural_global_progress=0.95,
            max_branch_target_threshold_ratio=1.25,
        )
        self.assertIn("rejection_evidence_missing", gaps)

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
        confirmation_source = (
            "controller_default:post_correction_resolution_confirmation_required"
        )
        self.assertEqual(
            provenance["no_material_anomaly_remaining"], confirmation_source
        )
        self.assertEqual(provenance["unresolved_signatures"], confirmation_source)


class ProductionEvidenceTests(unittest.TestCase):
    def test_durable_recovery_evidence_recursively_strips_forbidden_fields(self):
        @_deterministic_adapter
        def leaky_wls(state):
            metrics = _wls_adapter(state)
            if state.get("parent_state_id"):
                metrics["physical_bound_violations"] = [
                    {
                        "constraint": "voltage",
                        "nested": {
                            "hidden_truth": {"fault": "private"},
                            "true_parameter_errors": [{"line": 3}],
                        },
                    }
                ]
            return metrics

        @_deterministic_adapter
        def leaky_context(state):
            return {
                "context_rows": [{"state_hash": state["state_hash"]}],
                "unresolved_signatures": ["measurement_residual_outlier"],
                "supported_corrections": [
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "measurement_updates": {0: 8.0},
                        },
                    }
                ],
                "measurement_findings": [
                    {
                        "index": 0,
                        "nested": {
                            "oracle_action_hints": [{"tool": "private"}],
                            "true_measurement_errors": [{"index": 0}],
                        },
                    }
                ],
            }

        env = _production_env(
            wls=leaky_wls,
            contexts={
                GET_MEASUREMENT_CONTEXT: leaky_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            },
        )
        state = env.reset(_measurement_scenario())
        active_id = state["active_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": active_id}})
        _, context_output = env.step(
            {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": active_id}}
        )
        observation = env.get_policy_observation()
        self.assertEqual(
            find_forbidden_policy_paths(observation.fresh_context_evidence), []
        )

        action = context_output["tool_metrics"]["supported_corrections"][0]
        _, correction_output = env.step(action)
        candidate_id = correction_output["candidate_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": candidate_id}})
        env.step(
            {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}}
        )
        verification_summary = env.get_policy_observation().rejected_hypotheses[-1][
            "verification_summary"
        ]
        self.assertEqual(find_forbidden_policy_paths(verification_summary), [])
        self.assertEqual(
            verification_summary["physical_bound_violations"][0]["constraint"],
            "voltage",
        )

    def test_second_same_family_context_replaces_durable_inventory(self):
        call_count = 0

        @_deterministic_adapter
        def changing_context(state):
            nonlocal call_count
            call_count += 1
            value = float(call_count)
            result = {
                "context_rows": [{"state_hash": state["state_hash"]}],
                "unresolved_signatures": ["measurement_residual_outlier"],
                "supported_corrections": [
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "measurement_updates": {0: value},
                        },
                    }
                ],
            }
            if call_count == 1:
                result["measurement_findings"] = [{"inventory": "first"}]
            return result

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: changing_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        active_id = env.reset(_measurement_scenario())["active_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": active_id}})
        action = {
            "tool": GET_MEASUREMENT_CONTEXT,
            "arguments": {"state_id": active_id},
        }
        env.step(action)
        _, second_output = env.step(action)

        durable = env.get_policy_observation().fresh_context_evidence[
            "measurement"
        ]
        self.assertEqual(
            durable["supported_corrections"],
            second_output["tool_metrics"]["supported_corrections"],
        )
        self.assertNotIn("measurement_findings", durable)

    def test_context_evidence_survives_same_state_rollback(self):
        @_deterministic_adapter
        def no_progress_context(state):
            return {
                "context_rows": [{"state_hash": state["state_hash"]}],
                "unresolved_signatures": ["measurement_residual_outlier"],
                "supported_corrections": [
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "measurement_updates": {0: 8.0},
                        },
                    }
                ],
            }

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: no_progress_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        state = env.reset(_measurement_scenario())
        active_id = state["active_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": active_id}})
        _, context_output = env.step(
            {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": active_id}}
        )
        action = context_output["tool_metrics"]["supported_corrections"][0]
        before = env.get_policy_observation().fresh_context_evidence

        _, correction_output = env.step(action)
        candidate_id = correction_output["candidate_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": candidate_id}})
        env.step(
            {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}}
        )
        observation = env.get_policy_observation()

        self.assertEqual(observation.fresh_context_evidence, before)
        self.assertTrue(observation.has_fresh_measurement_context)
        self.assertEqual(
            observation.rejected_hypotheses[-1]["verification_summary"][
                "global_progress"
            ],
            0.0,
        )

    def test_context_evidence_is_cleared_when_candidate_commits(self):
        env = _production_env()
        state = env.reset(_measurement_scenario())
        active_id = state["active_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": active_id}})
        _, context_output = env.step(
            {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": active_id}}
        )
        action = context_output["tool_metrics"]["supported_corrections"][0]
        self.assertTrue(env.get_policy_observation().fresh_context_evidence)
        env.context_flags["rejected_hypotheses"] = [
            {
                "candidate_parent_id": active_id,
                "candidate_state_id": f"{active_id}:rejected-before-commit",
                "source_action": action,
            }
        ]
        self.assertTrue(env.get_policy_observation().rejected_hypotheses)

        _, correction_output = env.step(action)
        candidate_id = correction_output["candidate_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": candidate_id}})
        env.step(
            {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}}
        )
        observation = env.get_policy_observation()

        self.assertEqual(observation.fresh_context_evidence, {})
        self.assertFalse(observation.has_fresh_measurement_context)
        self.assertEqual(observation.rejected_hypotheses, [])

    def test_malformed_direct_context_inventory_fails_atomically(self):
        @_deterministic_adapter
        def malformed_context(state):
            return {
                "context_rows": [{"state_hash": state["state_hash"]}],
                "supported_corrections": [
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "measurement_updates": {0: 1.0},
                        },
                    },
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": "wrong-state",
                            "measurement_updates": {0: 1.0},
                        },
                    },
                ],
            }

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: malformed_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        active_id = env.reset(_measurement_scenario())["active_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": active_id}})

        _, output = env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": active_id},
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        observation = env.get_policy_observation()
        self.assertFalse(observation.has_fresh_measurement_context)
        self.assertEqual(observation.fresh_context_evidence, {})

    def test_production_branch_route_contract_rejects_incoherent_status(self):
        missing = object()
        cases = (
            ("missing", missing, True),
            ("unknown", "unknown", True),
            ("actionable_empty", "actionable", False),
            ("complete_negative_nonempty", "complete_negative", True),
            (
                "unavailable_nonempty",
                "unavailable_or_inconclusive",
                True,
            ),
        )
        for name, route_status, with_action in cases:
            with self.subTest(name=name):
                @_deterministic_adapter
                def branch_context(state):
                    payload = {
                        "parameter_findings": [{"line_row0": 0}],
                        "supported_corrections": (
                            [
                                {
                                    "tool": CORRECT_PARAMETERS,
                                    "arguments": {
                                        "state_id": state["state_id"],
                                        "line_index": 1,
                                    },
                                }
                            ]
                            if with_action
                            else []
                        ),
                    }
                    if route_status is not missing:
                        payload["route_status"] = route_status
                    return payload

                env = _production_env(
                    contexts={
                        GET_MEASUREMENT_CONTEXT: _context_adapter,
                        GET_PARAMETER_CONTEXT: branch_context,
                        GET_TOPOLOGY_CONTEXT: _context_adapter,
                    }
                )
                active_id = env.reset(_measurement_scenario())["active_state_id"]
                _, output = env.step(
                    {
                        "tool": GET_PARAMETER_CONTEXT,
                        "arguments": {"state_id": active_id},
                    }
                )

                self.assertEqual(output["execution_status"], "failure")
                self.assertEqual(
                    output["error_detail"],
                    "get_parameter_context_route_contract_invalid",
                )
                observation = env.get_policy_observation()
                self.assertFalse(observation.has_fresh_parameter_context)
                self.assertNotIn(
                    "parameter", observation.fresh_context_evidence
                )

    def test_production_branch_route_contract_accepts_coherent_status(self):
        cases = (
            ("actionable", True),
            ("complete_negative", False),
            ("unavailable_or_inconclusive", False),
        )
        for route_status, with_action in cases:
            with self.subTest(route_status=route_status):
                @_deterministic_adapter
                def branch_context(state):
                    return {
                        "parameter_findings": [{"line_row0": 0}],
                        "supported_corrections": (
                            [
                                {
                                    "tool": CORRECT_PARAMETERS,
                                    "arguments": {
                                        "state_id": state["state_id"],
                                        "line_index": 1,
                                    },
                                }
                            ]
                            if with_action
                            else []
                        ),
                        "route_status": route_status,
                    }

                env = _production_env(
                    contexts={
                        GET_MEASUREMENT_CONTEXT: _context_adapter,
                        GET_PARAMETER_CONTEXT: branch_context,
                        GET_TOPOLOGY_CONTEXT: _context_adapter,
                    }
                )
                active_id = env.reset(_measurement_scenario())["active_state_id"]
                _, output = env.step(
                    {
                        "tool": GET_PARAMETER_CONTEXT,
                        "arguments": {"state_id": active_id},
                    }
                )

                self.assertEqual(output["execution_status"], "success")
                observation = env.get_policy_observation()
                self.assertTrue(observation.has_fresh_parameter_context)
                self.assertEqual(
                    observation.fresh_context_evidence["parameter"][
                        "route_status"
                    ],
                    route_status,
                )

    def test_parameter_ranking_contract_is_preserved_as_durable_evidence(self):
        ranking = {
            "parameter_ranking_contract": (
                "distinct_line_abs_lambda_dominance_v1"
            ),
            "parameter_ranking_distinct_lines": [
                {"line_index1": 1, "abs_lambda_score": 9.0},
                {"line_index1": 2, "abs_lambda_score": 3.0},
            ],
            "parameter_ranking_top_abs_lambda": 9.0,
            "parameter_ranking_runner_up_abs_lambda": 3.0,
            "parameter_ranking_dominance_ratio": 3.0,
            "parameter_ranking_dominance_threshold": 1.2,
            "parameter_ranking_singleton": False,
            "parameter_ranking_dominant": True,
        }

        @_deterministic_adapter
        def parameter_context(state):
            return {
                **ranking,
                "parameter_findings": [
                    {"line_row0": 0, "value": 9.0},
                    {"line_row0": 1, "value": 3.0},
                ],
                "supported_corrections": [
                    {
                        "tool": CORRECT_PARAMETERS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "line_index": 1,
                        },
                    },
                    {
                        "tool": CORRECT_PARAMETERS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "line_index": 2,
                        },
                    },
                ],
                "route_status": "actionable",
            }

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: _context_adapter,
                GET_PARAMETER_CONTEXT: parameter_context,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        active_id = env.reset(_measurement_scenario())["active_state_id"]
        _, output = env.step(
            {
                "tool": GET_PARAMETER_CONTEXT,
                "arguments": {"state_id": active_id},
            }
        )

        self.assertEqual(output["execution_status"], "success")
        durable = env.get_policy_observation().fresh_context_evidence[
            "parameter"
        ]
        for key, expected in ranking.items():
            self.assertEqual(durable[key], expected)

    def test_malformed_bundled_inventory_does_not_close_branch_route(self):
        @_deterministic_adapter
        def bundled_context(state):
            return {
                "context_rows": [{"state_hash": state["state_hash"]}],
                "supported_corrections": [
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "measurement_updates": {0: 1.0},
                        },
                    }
                ],
                "branch_route_screening": {
                    "parameter": {
                        "state_id": state["state_id"],
                        "state_hash": state["state_hash"],
                        "evidence_source": "deployment_context:test",
                        "context_tool": GET_PARAMETER_CONTEXT,
                        "route_status": "complete_negative",
                        "supported_corrections": [
                            {
                                "tool": CORRECT_PARAMETERS,
                                "arguments": {"state_id": "wrong-state"},
                            }
                        ],
                    }
                },
            }

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: bundled_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        active_id = env.reset(_measurement_scenario())["active_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": active_id}})
        _, output = env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": active_id},
            }
        )

        self.assertEqual(output["execution_status"], "success")
        observation = env.get_policy_observation()
        self.assertTrue(observation.has_fresh_measurement_context)
        self.assertFalse(observation.has_fresh_parameter_context)
        self.assertNotIn("parameter", observation.fresh_context_evidence)

    def test_incoherent_bundled_route_status_does_not_expose_correction(self):
        @_deterministic_adapter
        def bundled_context(state):
            return {
                "context_rows": [{"state_hash": state["state_hash"]}],
                "supported_corrections": [
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "measurement_updates": {0: 1.0},
                        },
                    }
                ],
                "branch_route_screening": {
                    "parameter": {
                        "state_id": state["state_id"],
                        "state_hash": state["state_hash"],
                        "evidence_source": "deployment_context:test",
                        "context_tool": GET_PARAMETER_CONTEXT,
                        "route_status": "unavailable_or_inconclusive",
                        "supported_corrections": [
                            {
                                "tool": CORRECT_PARAMETERS,
                                "arguments": {
                                    "state_id": state["state_id"],
                                    "line_index": 1,
                                },
                            }
                        ],
                    }
                },
            }

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: bundled_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        active_id = env.reset(_measurement_scenario())["active_state_id"]
        _, output = env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": active_id},
            }
        )

        self.assertEqual(output["execution_status"], "success")
        observation = env.get_policy_observation()
        self.assertTrue(observation.has_fresh_measurement_context)
        self.assertFalse(observation.has_fresh_parameter_context)
        self.assertNotIn("parameter", observation.fresh_context_evidence)

    def test_bundled_branch_inventory_supports_training_decision_audit(self):
        captured_action = None

        @_deterministic_adapter
        def bundled_context(state):
            nonlocal captured_action
            captured_action = {
                "tool": CORRECT_PARAMETERS,
                "arguments": {
                    "state_id": state["state_id"],
                    "line_index": 1,
                },
            }
            return {
                "context_rows": [{"state_hash": state["state_hash"]}],
                "supported_corrections": [
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": state["state_id"],
                            "measurement_updates": {0: 1.0},
                        },
                    }
                ],
                "branch_route_screening": {
                    "parameter": {
                        "state_id": state["state_id"],
                        "state_hash": state["state_hash"],
                        "evidence_source": "deployment_context:test",
                        "context_tool": GET_PARAMETER_CONTEXT,
                        "route_status": "actionable",
                        "supported_corrections": [captured_action],
                    },
                    "topology": {
                        "state_id": state["state_id"],
                        "state_hash": state["state_hash"],
                        "evidence_source": "deployment_context:test",
                        "context_tool": GET_TOPOLOGY_CONTEXT,
                        "route_status": "complete_negative",
                        "supported_corrections": [],
                    },
                },
            }

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: bundled_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        active_id = env.reset(_measurement_scenario())["active_state_id"]
        _, output = env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": active_id},
            }
        )

        self.assertEqual(output["execution_status"], "success")
        self.assertIsNotNone(captured_action)
        env.assert_training_decision_evidence(captured_action)

    def test_malformed_terminal_closure_context_fails_atomically(self):
        @_deterministic_adapter
        def malformed_closure_context(state):
            closure = {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": state["state_id"],
                    "suspect_group": [0, 1],
                },
            }
            exhausted = {
                family: {
                    "state_id": state["state_id"],
                    "state_hash": state["state_hash"],
                    "evidence_source": "deployment_context:test",
                    "context_tool": context_tool,
                    "route_status": "complete_negative",
                    "supported_corrections": [],
                }
                for family, context_tool in (
                    ("parameter", GET_PARAMETER_CONTEXT),
                    ("topology", GET_TOPOLOGY_CONTEXT),
                )
            }
            return {
                "context_rows": [{"state_hash": state["state_hash"]}],
                "supported_corrections": [closure],
                "verified_terminal_measurement_closure_targets": [0, 1],
                # The target marker must never authorize a closure without the
                # companion candidate-quality attestation.
                "branch_route_screening": exhausted,
            }

        env = _production_env(
            contexts={
                GET_MEASUREMENT_CONTEXT: malformed_closure_context,
                GET_PARAMETER_CONTEXT: _context_adapter,
                GET_TOPOLOGY_CONTEXT: _context_adapter,
            }
        )
        active_id = env.reset(_measurement_scenario())["active_state_id"]
        env.context_flags["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "production-measurement:s0",
                        "suspect_group": [0],
                    },
                }
            }
        ]

        _, output = env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": active_id},
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(
            output["error_detail"],
            "get_measurement_context_terminal_closure_contract_invalid",
        )
        observation = env.get_policy_observation()
        self.assertFalse(observation.has_fresh_measurement_context)
        self.assertFalse(observation.has_fresh_parameter_context)
        self.assertFalse(observation.has_fresh_topology_context)
        self.assertEqual(observation.fresh_context_evidence, {})

    def test_nonactionable_route_inventory_is_not_an_observable_expert_hint(self):
        active_id = "episode:s1"
        action = {
            "tool": CORRECT_PARAMETERS,
            "arguments": {"state_id": active_id, "line_index": 1},
        }
        for route_status in ("unavailable_or_inconclusive", None):
            with self.subTest(route_status=route_status):
                metrics = {
                    "state_id": active_id,
                    "route_status": route_status,
                    "supported_corrections": [action],
                }
                policy = {
                    "active_state_id": active_id,
                    "has_fresh_parameter_context": True,
                    "parameter_context_state_id": active_id,
                    "fresh_context_evidence": {"parameter": metrics},
                }
                history = [
                    {
                        "action": {
                            "tool": GET_PARAMETER_CONTEXT,
                            "arguments": {"state_id": active_id},
                        },
                        "tool_output": {
                            "execution_status": "success",
                            "tool_metrics": metrics,
                        },
                    }
                ]

                self.assertEqual(
                    ExpertPolicyOracle._observable_supported_corrections(
                        policy, history
                    ),
                    [],
                )

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
        _, output = env.step(
            {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
        )
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "terminal_condition_not_met")

        clean = _measurement_scenario()
        clean["measurements"] = [1.0]
        clean["true_measurement_errors"] = []
        clean["oracle_action_hints"] = []
        state = env.reset(clean)
        env.step({"tool": RUN_WLS, "arguments": {"state_id": state["active_state_id"]}})
        env.assert_training_decision_evidence(
            {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
        )

    def test_deployment_accept_final_does_not_mint_private_terminal_bypass(self):
        env = _production_env()
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
        env.step({"tool": RUN_WLS, "arguments": {"state_id": candidate_id}})
        state, output = env.step(
            {
                "tool": COMMIT_STATE,
                "arguments": {"candidate_state_id": candidate_id},
            }
        )

        self.assertEqual(output["execution_status"], "success")
        self.assertNotIn("oracle_terminal_eligible", env._oracle_payload)
        self.assertFalse(state["no_material_anomaly_remaining"])
        self.assertIn(
            "post_correction_resolution_confirmation_required:measurement_context",
            state["unresolved_signatures"],
        )
        confirmation_source = (
            "controller_default:post_correction_resolution_confirmation_required"
        )
        self.assertEqual(
            state["semantic_field_provenance"]["unresolved_signatures"],
            confirmation_source,
        )
        self.assertEqual(
            state["semantic_field_provenance"]["no_material_anomaly_remaining"],
            confirmation_source,
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
                {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
                "ACCEPT_FINAL",
                "terminal_resolved",
            ),
            (
                {
                    "tool": ASK_FOR_MORE_EVIDENCE,
                    "arguments": {
                        "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                    },
                },
                "INCONCLUSIVE",
                "terminal_operator_escalation",
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
        self.assertEqual(rows[0]["state_class"], "terminal_resolved")
        self.assertEqual(rows[1]["state_class"], "terminal_operator_escalation")
        self.assertNotEqual(rows[1]["state_class"], rows[0]["state_class"])
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

    def test_commit_class_uses_observable_evidence_when_disposition_absent(self):
        """Regression for the DAgger-1 round-2 159-row misclassification.

        Those rows carried a VERIFIED_CANDIDATE with ``commit_state`` as the
        expert's rank-one target and no candidate disposition from any source
        (absent from the transition label, empty candidate summary, ``None`` in
        the observation).  The previous catch-all filed all 159 as
        ``invalid_precondition_recovery`` even though nothing was invalid.
        """
        commit = {
            "tool": COMMIT_STATE,
            "arguments": {"candidate_state_id": "r0_fixture_episode1:s3"},
        }
        verified = {
            "active_state_id": "active",
            "candidate_state_id": "r0_fixture_episode1:s3",
            "candidate_lifecycle": "VERIFIED_CANDIDATE",
            "candidate_status": "verified",
            "has_verified_candidate": True,
            "has_open_candidate": True,
            "no_material_anomaly_remaining": False,
            "candidate_disposition": None,
        }
        # No disposition anywhere: the label carries none, as observed in the
        # failed collection.  Verified candidate with anomalies remaining is a
        # partial commit, not an invalid precondition.
        self.assertEqual(
            classify_state_example(
                verified,
                {"process_valid": True},
                preferred_action=commit,
            ),
            "accepted_partial_commit",
        )
        # Same evidence, nothing material left to explain -> final commit.
        self.assertEqual(
            classify_state_example(
                {**verified, "no_material_anomaly_remaining": True},
                {"process_valid": True},
                preferred_action=commit,
            ),
            "accepted_final_commit",
        )
        # An unverified or absent candidate is still an invalid precondition.
        for lifecycle, status, verified_flag in (
            ("OPEN_UNVERIFIED_CANDIDATE", "unverified", False),
            ("NO_CANDIDATE", None, False),
        ):
            self.assertEqual(
                classify_state_example(
                    {
                        **verified,
                        "candidate_lifecycle": lifecycle,
                        "candidate_status": status,
                        "has_verified_candidate": verified_flag,
                    },
                    {"process_valid": True},
                    preferred_action=commit,
                ),
                "invalid_precondition_recovery",
            )
        # An explicit disposition still wins over the observable fallback.
        for disposition, expected in (
            ("ACCEPT_FINAL", "accepted_final_commit"),
            ("ACCEPT_PARTIAL", "accepted_partial_commit"),
            ("REJECT", "rejected_candidate_recovery"),
        ):
            self.assertEqual(
                classify_state_example(
                    verified,
                    {"process_valid": True},
                    preferred_action=commit,
                    target_candidate_disposition=disposition,
                ),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
