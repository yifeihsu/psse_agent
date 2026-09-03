import json
import random
import tempfile
import unittest
from pathlib import Path

from psse_env import (
    FORBIDDEN_POLICY_KEYS,
    OracleState,
    PolicyObservation,
    PowerSystemStateStore,
    StateStoreError,
    TransactionalPSSEEnv,
)
from psse_env.actions import action_signature
from psse_env.dagger import (
    AggreVaTeLite,
    BalancedReplayBuffer,
    CounterfactualGenerator,
    DaggerRolloutCollector,
    examples_to_chat_sft,
    grouped_scenario_split,
    load_jsonl,
    run_dagger,
    to_pairwise_examples,
    write_jsonl,
)
from psse_env.dagger.error_injectors import InjectedAction
from psse_env.oracle import CandidateDisposition, CandidateQualityOracle, ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.candidate_quality import CandidateAssessment
from psse_env.state_store import find_forbidden_policy_paths, policy_safe_copy
from psse_env.verifier import RuleBasedVerifier, build_verifier_dataset, evaluate_predictions


STANDARD_OUTPUT_KEYS = {
    "execution_status",
    "error_code",
    "error_detail",
    "state_mutated",
    "active_state_id",
    "candidate_state_id",
    "tool_metrics",
    "valid_next_actions",
}


def synthetic_scenario(*, scenario_id="case", parameter_fault=False, final=False):
    scenario = {
        "scenario_id": scenario_id,
        "case": {"name": "case14", "branch": [{"x": 0.1}]},
        "measurements": [1.2, 2.0, 3.0],
        "true_measurement_errors": [{"index": 0, "observed": 1.2, "clean": 1.0}],
    }
    if parameter_fault:
        scenario["true_parameter_errors"] = [{"line_index": 2}]
    if final:
        scenario["metadata"] = {"remaining_anomaly_score": 0.0, "no_material_anomaly_remaining": True}
    return scenario


def correct_measurement(state_id, *, index=0, value=1.0):
    return {
        "tool": "correct_measurements",
        "arguments": {"state_id": state_id, "measurement_updates": {index: value}},
    }


class StateStoreTests(unittest.TestCase):
    def test_candidate_does_not_mutate_parent(self):
        store = PowerSystemStateStore()
        root = store.create_root({"branch": [{"x": 0.1}]}, [1.0, 2.0], {})
        candidate = store.clone_candidate(
            root,
            {"measurement_updates": {1: 9.0}},
            correct_measurement(root, index=1),
            created_at_step=3,
        )
        self.assertEqual(store.get_state(root)["measurements"], [1.0, 2.0])
        self.assertEqual(store.get_state(candidate)["measurements"], [1.0, 9.0])
        self.assertNotEqual(store.state_hash(root), store.state_hash(candidate))
        provenance = store.candidate_provenance(candidate)
        self.assertEqual(provenance["episode_id"], store.current_episode_id)
        self.assertEqual(provenance["depth"], 1)
        self.assertEqual(provenance["created_at_step"], 3)

    def test_previous_episode_state_is_not_addressable(self):
        store = PowerSystemStateStore()
        old = store.create_root({}, [1.0], {}, episode_id="old")
        store.create_root({}, [2.0], {}, episode_id="new")
        self.assertFalse(store.exists(old))
        with self.assertRaises(StateStoreError):
            store.get_state(old)
        self.assertEqual(store.get_state_for_audit(old)["episode_id"], "old")

    def test_candidate_parent_belongs_to_same_episode(self):
        store = PowerSystemStateStore()
        old = store.create_root({}, [1.0], {}, episode_id="old")
        store.create_root({}, [2.0], {}, episode_id="new")
        with self.assertRaises(StateStoreError):
            store.clone_candidate(old, {"measurement_updates": {0: 3.0}}, correct_measurement(old))

    def test_commit_preserves_lineage(self):
        store = PowerSystemStateStore()
        root = store.create_root({}, [1.0])
        candidate = store.clone_candidate(root, {"measurement_updates": {0: 2.0}}, correct_measurement(root))
        store.mark_verified(candidate, {"target_progress": 1.0}, CandidateDisposition.ACCEPT_PARTIAL)
        self.assertEqual(store.commit(candidate), candidate)
        self.assertEqual(store.lineage(candidate), [root, candidate])
        self.assertEqual(store.get_state(candidate)["episode_id"], store.get_state(root)["episode_id"])

    def test_rollback_restores_exact_parent_hash(self):
        store = PowerSystemStateStore()
        root = store.create_root({}, [1.0])
        root_hash = store.state_hash(root)
        candidate = store.clone_candidate(root, {"measurement_updates": {0: 8.0}}, correct_measurement(root))
        store.mark_verified(candidate, {"target_progress": -1.0}, CandidateDisposition.REJECT)
        self.assertEqual(store.rollback(candidate), root)
        self.assertEqual(store.state_hash(root), root_hash)
        self.assertEqual(store.active_state_id, root)


class ObservationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.env = TransactionalPSSEEnv()
        self.env.reset(
            {
                "scenario_id": "boundary",
                "case": {"secret": "corrupt"},
                "measurements": [2.0],
                "clean_case": {"secret": "clean"},
                "clean_measurements": [1.0],
                "true_measurement_errors": [{"index": 0}],
                "true_parameter_errors": [{"line_index": 2}],
                "suggested_actions": [{"tool": "run_wls", "arguments": {"state_id": "s0"}}],
            }
        )

    def test_policy_observation_excludes_suggested_actions(self):
        observation = self.env.get_policy_observation().as_dict()
        self.assertNotIn("suggested_actions", json.dumps(observation, sort_keys=True))

    def test_policy_observation_excludes_hidden_truth(self):
        serialized = json.dumps(self.env.get_policy_observation().as_dict(), sort_keys=True)
        for key in FORBIDDEN_POLICY_KEYS:
            self.assertNotIn(f'"{key}"', serialized)
        self.assertNotIn("clean_measurements", serialized)

    def test_oracle_state_contains_hidden_truth(self):
        oracle_state = self.env.get_oracle_state()
        self.assertIsInstance(oracle_state, OracleState)
        self.assertEqual(oracle_state.clean_measurements, [1.0])
        self.assertEqual(oracle_state.true_parameter_errors[0]["line_index"], 2)
        self.assertTrue(oracle_state.oracle_action_hints)

    def test_policy_observation_fails_closed_on_nested_hidden_truth(self):
        observation = PolicyObservation(
            active_state_id="e:s0",
            remaining_budget=1,
            last_tool_output={"clean_case": {"hidden": True}},
        )
        with self.assertRaises(ValueError):
            observation.as_dict()

    def test_policy_boundary_normalizes_privileged_key_aliases(self):
        aliases = (
            "HiddenTruth",
            "GROUND_TRUTH",
            "TrueMeasurementErrors",
            "true-parameter-errors",
            "TRUE TOPOLOGY ERRORS",
            "truth",
            "ExpectedFinalState",
            "recommended-action",
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                payload = {"nested": [{alias: ["private"]}]}
                self.assertEqual(
                    find_forbidden_policy_paths(payload),
                    [f"$.nested[0].{alias}"],
                )
                self.assertEqual(policy_safe_copy(payload), {"nested": [{}]})

    def test_policy_observation_rejects_case_variant_private_truth(self):
        for alias in (
            "HiddenTruth",
            "GROUND_TRUTH",
            "TrueMeasurementErrors",
        ):
            with self.subTest(alias=alias):
                observation = PolicyObservation(
                    active_state_id="e:s0",
                    remaining_budget=1,
                    last_tool_output={alias: []},
                )
                with self.assertRaisesRegex(ValueError, alias):
                    observation.as_dict()

    def test_hidden_truth_is_not_in_policy_history_after_verification(self):
        state = self.env.current_state()
        state, _ = self.env.step(correct_measurement(state["active_state_id"]))
        _, output = self.env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertNotIn("candidate_assessment", json.dumps(output, sort_keys=True))
        self.assertNotIn("remaining_fault_count", json.dumps(output, sort_keys=True))
        serialized = json.dumps(self.env.get_policy_observation().as_dict(), sort_keys=True)
        self.assertNotIn("remaining_fault_count", serialized)
        self.assertNotIn("candidate_assessment", serialized)
        self.assertNotIn("candidate_disposition", serialized)
        self.assertNotIn("progress_class", serialized)
        self.assertNotIn("target_fixed", serialized)

    def test_candidate_truth_label_does_not_change_policy_observation(self):
        def verified_observation(fault_index):
            env = TransactionalPSSEEnv()
            state = env.reset(
                {
                    "scenario_id": "same",
                    "case": {"name": "same"},
                    "measurements": [9.0, 2.0],
                    "clean_measurements": [1.0, 2.0],
                    "true_measurement_errors": [{"index": fault_index}],
                }
            )
            state, _ = env.step(correct_measurement(state["active_state_id"], value=1.0))
            env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
            return env.get_policy_observation().as_dict()

        self.assertEqual(verified_observation(0), verified_observation(1))

    def test_oracle_finality_does_not_set_policy_terminal_bit(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        state, _ = env.step(correct_measurement(state["active_state_id"]))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        env.step({"tool": "commit_state", "arguments": {"candidate_state_id": state["candidate_state_id"]}})
        self.assertFalse(env.get_policy_observation().no_material_anomaly_remaining)
        self.assertEqual(env.get_oracle_state().hidden_truth["oracle_terminal_eligible"], True)


class ProcessValidityTests(unittest.TestCase):
    @staticmethod
    def _context_bound_state(
        family, supported, *, route_status=None, route_status_reason=None
    ):
        state = {
            "active_state_id": "e:s0",
            f"has_fresh_{family}_context": True,
            f"{family}_context_state_id": "e:s0",
            "fresh_context_evidence": {
                family: {
                    "state_id": "e:s0",
                    "supported_corrections": supported,
                }
            },
        }
        if route_status is not None:
            state["fresh_context_evidence"][family]["route_status"] = (
                route_status
            )
        if route_status_reason is not None:
            state["fresh_context_evidence"][family]["route_status_reason"] = (
                route_status_reason
            )
        return state

    def test_missing_parameter_context_routes_to_parameter_context(self):
        state = {"active_state_id": "e:s0", "has_fresh_parameter_context": False}
        result = ProcessValidityOracle().check(
            state,
            {"tool": "correct_parameters", "arguments": {"state_id": "e:s0", "line_index": 2}},
        )
        self.assertFalse(result["process_valid"])
        self.assertEqual(result["error_code"], "missing_precondition")
        self.assertEqual(result["error_detail"], "parameter_context_missing")
        self.assertEqual(result["valid_next_actions"][0]["tool"], "get_parameter_context")

    def test_hydrated_corrections_require_exact_same_state_context_support(self):
        cases = (
            (
                "measurement",
                {
                    "tool": "correct_measurements",
                    "arguments": {"state_id": "e:s0", "suspect_group": [3]},
                },
                {
                    "tool": "correct_measurements",
                    "arguments": {"state_id": "e:s0", "suspect_group": [4]},
                },
                None,
            ),
            (
                "parameter",
                {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": "e:s0", "line_index1": 2},
                },
                {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": "e:s0", "line_index1": 3},
                },
                "actionable",
            ),
            (
                "topology",
                {
                    "tool": "correct_topology",
                    "arguments": {
                        "state_id": "e:s0",
                        "line_index1": 2,
                        "status": 1,
                    },
                },
                {
                    "tool": "correct_topology",
                    "arguments": {
                        "state_id": "e:s0",
                        "line_index1": 3,
                        "status": 1,
                    },
                },
                "actionable",
            ),
        )
        oracle = ProcessValidityOracle(executor_hydrated_corrections=True)
        for family, supported, unsupported, route_status in cases:
            with self.subTest(family=family):
                state = self._context_bound_state(
                    family, [supported], route_status=route_status
                )
                allowed = oracle.check(state, supported)
                rejected = oracle.check(state, unsupported)

                self.assertTrue(allowed["process_valid"])
                self.assertFalse(rejected["process_valid"])
                self.assertEqual(
                    rejected["error_code"],
                    "correction_not_supported_by_current_context",
                )
                self.assertLessEqual(len(rejected["valid_next_actions"]), 2)
                self.assertEqual(rejected["valid_next_actions"][0], supported)

    def test_nonactionable_branch_route_blocks_every_target(self):
        state = self._context_bound_state(
            "parameter",
            [],
            route_status="unavailable_or_inconclusive",
            route_status_reason="parameter_findings_require_repeated_scans",
        )
        result = ProcessValidityOracle(
            executor_hydrated_corrections=True
        ).check(
            state,
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": "e:s0", "line_index1": 2},
            },
        )

        self.assertFalse(result["process_valid"])
        self.assertEqual(result["error_code"], "correction_route_not_actionable")
        self.assertEqual(
            result["error_detail"],
            (
                "parameter_route_not_actionable:unavailable_or_inconclusive:"
                "parameter_findings_require_repeated_scans"
            ),
        )
        self.assertEqual(
            [action["tool"] for action in result["valid_next_actions"]],
            ["get_topology_context", "get_measurement_context"],
        )

    def test_context_inventory_must_be_bound_to_active_state(self):
        supported = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "old:s0", "suspect_group": [3]},
        }
        state = self._context_bound_state("measurement", [supported])
        state["fresh_context_evidence"]["measurement"]["state_id"] = "old:s0"
        result = ProcessValidityOracle(
            executor_hydrated_corrections=True
        ).check(
            state,
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": "e:s0", "suspect_group": [3]},
            },
        )

        self.assertFalse(result["process_valid"])
        self.assertEqual(result["error_code"], "missing_precondition")
        self.assertEqual(result["valid_next_actions"][0]["tool"], "get_measurement_context")

    def test_rejected_candidate_cannot_commit(self):
        state = {
            "active_state_id": "e:s0",
            "candidate_state_id": "e:s1",
            "has_verified_candidate": True,
            "candidate_disposition": "REJECT",
        }
        result = ProcessValidityOracle().check(
            state, {"tool": "commit_state", "arguments": {"candidate_state_id": "e:s1"}}
        )
        self.assertFalse(result["process_valid"])
        self.assertEqual(result["error_detail"], "commit_rejected_or_inconclusive_candidate")
        self.assertEqual(result["valid_next_actions"][0]["tool"], "rollback_state")

    def test_evidence_action_rejects_stale_state_reference(self):
        class CurrentOnlyStore:
            def exists(self, state_id):
                return state_id == "e:s1"

        state = {
            "active_state_id": "e:s0",
            "candidate_state_id": "e:s1",
            "has_open_candidate": True,
            "has_verified_candidate": True,
            "candidate_disposition": "INCONCLUSIVE",
        }
        result = ProcessValidityOracle().check(
            state,
            {"tool": "ask_for_more_evidence", "arguments": {"state_id": "old:s0"}},
            store=CurrentOnlyStore(),
        )
        self.assertFalse(result["process_valid"])
        self.assertEqual(result["error_code"], "unknown_state_id")


class TransactionTests(unittest.TestCase):
    def test_malformed_action_becomes_structured_failure(self):
        env = TransactionalPSSEEnv()
        env.reset(synthetic_scenario())
        _, output = env.step('{"tool":')
        self.assertEqual(set(output), STANDARD_OUTPUT_KEYS)
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "json_parse_error")
        self.assertFalse(output["state_mutated"])

    def test_invalid_action_does_not_mutate_state(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        before = (env.store.episode_hash(), state["active_state_id"], state["candidate_state_id"])
        state, output = env.step({"tool": "commit_state", "arguments": {"candidate_state_id": "missing"}})
        after = (env.store.episode_hash(), state["active_state_id"], state["candidate_state_id"])
        self.assertEqual(before, after)
        self.assertFalse(output["state_mutated"])
        self.assertFalse(env.is_terminal())

    def test_parameter_scan_executor_failure_offers_bounded_cross_family_recovery(self):
        def parameter_context(state):
            return {
                "route_status": "actionable",
                "supported_corrections": [
                    {
                        "tool": "correct_parameters",
                        "arguments": {
                            "state_id": state["state_id"],
                            "line_index1": 1,
                        },
                    }
                ],
            }

        def missing_scans(_state, _action):
            return {
                "execution_status": "failure",
                "error_code": "parameter_scans_missing",
                "error_detail": "observed scan bundle is absent",
            }

        env = TransactionalPSSEEnv(
            process_oracle=ProcessValidityOracle(
                executor_hydrated_corrections=True
            ),
            context_providers={"get_parameter_context": parameter_context},
            correction_executors={"correct_parameters": missing_scans},
        )
        state = env.reset(synthetic_scenario())
        state, _ = env.step(
            {
                "tool": "get_parameter_context",
                "arguments": {"state_id": state["active_state_id"]},
            }
        )
        _, output = env.step(
            {
                "tool": "correct_parameters",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "line_index1": 1,
                },
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "parameter_scans_missing")
        self.assertFalse(output["state_mutated"])
        self.assertLessEqual(len(output["valid_next_actions"]), 2)
        self.assertEqual(
            [action["tool"] for action in output["valid_next_actions"]],
            ["get_topology_context", "get_measurement_context"],
        )
        self.assertNotIn(
            "correct_parameters",
            [action["tool"] for action in output["valid_next_actions"]],
        )

    def test_unsupported_context_target_is_rejected_before_executor(self):
        executor_calls = []

        def parameter_context(state):
            return {
                "route_status": "actionable",
                "supported_corrections": [
                    {
                        "tool": "correct_parameters",
                        "arguments": {
                            "state_id": state["state_id"],
                            "line_index1": 1,
                        },
                    }
                ],
            }

        def correction_executor(_state, action):
            executor_calls.append(action)
            return {"modification": {"line_index1": 2, "value": 1.0}}

        env = TransactionalPSSEEnv(
            process_oracle=ProcessValidityOracle(
                executor_hydrated_corrections=True
            ),
            context_providers={"get_parameter_context": parameter_context},
            correction_executors={"correct_parameters": correction_executor},
        )
        state = env.reset(synthetic_scenario())
        state, _ = env.step(
            {
                "tool": "get_parameter_context",
                "arguments": {"state_id": state["active_state_id"]},
            }
        )
        active_id = state["active_state_id"]
        before_hash = env.store.episode_hash()
        next_state, output = env.step(
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": active_id, "line_index1": 2},
            }
        )

        self.assertEqual(executor_calls, [])
        self.assertEqual(
            output["error_code"],
            "correction_not_supported_by_current_context",
        )
        self.assertFalse(output["state_mutated"])
        self.assertEqual(next_state["active_state_id"], active_id)
        self.assertIsNone(next_state["candidate_state_id"])
        self.assertEqual(env.store.episode_hash(), before_hash)
        self.assertLessEqual(len(output["valid_next_actions"]), 2)

    def test_unverified_candidate_cannot_rollback(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        state, _ = env.step(correct_measurement(state["active_state_id"]))
        candidate_id = state["candidate_state_id"]
        state, output = env.step({"tool": "rollback_state", "arguments": {"candidate_state_id": candidate_id}})
        self.assertEqual(output["error_detail"], "rollback_without_verified_candidate")
        self.assertEqual(state["candidate_state_id"], candidate_id)

    def test_rejected_candidate_cannot_commit(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        state, _ = env.step(correct_measurement(state["active_state_id"], index=1))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "REJECT")

        env = TransactionalPSSEEnv()
        state = env.reset(
            {
                "scenario_id": "parameter-case-collateral",
                "case": {"branch": [{"x": 9.0}], "bus": [{"vm": 1.0}]},
                "measurements": [5.0],
                "clean_case": {"branch": [{"x": 1.0}], "bus": [{"vm": 1.0}]},
                "true_parameter_errors": [{"line_index": 0, "parameter": "x", "clean": 1.0}],
            }
        )
        env.step({"tool": "get_parameter_context", "arguments": {"state_id": state["active_state_id"]}})
        state, _ = env.step(
            {
                "tool": "correct_parameters",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "line_index": 0,
                    "parameter": "x",
                    "value": 1.0,
                    "case": {"branch": [{"x": 1.0}], "bus": [{"vm": 999.0}]},
                },
            }
        )
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "REJECT")
        _, output = env.step({"tool": "commit_state", "arguments": {"candidate_state_id": state["candidate_state_id"]}})
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_detail"], "commit_rejected_or_inconclusive_candidate")

    def test_false_finalization_is_blocked(self):
        env = TransactionalPSSEEnv()
        env.reset({"scenario_id": "unresolved", "case": {}, "measurements": [1.0]})
        _, output = env.step({"tool": "finalize_diagnosis", "arguments": {}})
        self.assertEqual(output["error_code"], "terminal_condition_not_met")
        self.assertFalse(env.is_terminal())

    def test_second_correction_is_blocked(self):
        env = TransactionalPSSEEnv()
        root = env.reset(synthetic_scenario())
        state, _ = env.step(correct_measurement(root["active_state_id"]))
        before_hash = env.store.episode_hash()
        state, output = env.step(correct_measurement(root["active_state_id"], index=1))
        self.assertEqual(output["error_detail"], "correction_with_open_candidate")
        self.assertEqual(env.store.episode_hash(), before_hash)
        self.assertTrue(state["has_unverified_candidate"])

    def test_wrong_candidate_id_is_blocked(self):
        env = TransactionalPSSEEnv()
        root = env.reset(synthetic_scenario())
        state, _ = env.step(correct_measurement(root["active_state_id"]))
        candidate_id = state["candidate_state_id"]
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": candidate_id}})
        _, output = env.step({"tool": "commit_state", "arguments": {"candidate_state_id": "other:s9"}})
        self.assertEqual(output["error_code"], "state_reference_mismatch")
        self.assertEqual(env.store.active_state_id, root["active_state_id"])

    def test_wls_must_target_active_or_current_candidate(self):
        env = TransactionalPSSEEnv()
        root = env.reset(synthetic_scenario())
        state, _ = env.step(correct_measurement(root["active_state_id"]))
        _, output = env.step({"tool": "run_wls", "arguments": {"state_id": root["active_state_id"]}})
        self.assertEqual(output["error_code"], "state_reference_mismatch")
        self.assertTrue(state["has_unverified_candidate"])

    def test_previous_episode_state_is_not_addressable_by_environment(self):
        env = TransactionalPSSEEnv()
        old = env.reset(synthetic_scenario(scenario_id="old"))["active_state_id"]
        env.reset(synthetic_scenario(scenario_id="new"))
        _, output = env.step(correct_measurement(old))
        self.assertEqual(output["error_code"], "unknown_state_id")

    def test_empty_and_no_effect_corrections_are_atomic_failures(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        before = env.store.episode_hash()
        _, empty = env.step(
            {"tool": "correct_measurements", "arguments": {"state_id": state["active_state_id"]}}
        )
        self.assertEqual(empty["error_detail"], "empty_correction_payload")
        self.assertEqual(env.store.episode_hash(), before)
        _, noop = env.step(correct_measurement(state["active_state_id"], value=1.2))
        self.assertEqual(noop["execution_status"], "failure")
        self.assertFalse(noop["state_mutated"])
        self.assertIsNone(env.current_candidate_id)
        self.assertEqual(env.store.episode_hash(), before)
        _, metadata_only = env.step(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "metadata_updates": {"note": "not a physical correction"},
                },
            }
        )
        self.assertEqual(metadata_only["error_detail"], "empty_correction_payload")
        self.assertEqual(env.store.episode_hash(), before)

    def test_whole_measurement_vector_is_not_a_bounded_correction(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        before = env.store.episode_hash()
        _, output = env.step(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "measurements": [99.0, 99.0, 99.0],
                },
            }
        )
        self.assertEqual(output["error_code"], "schema_error")
        self.assertEqual(output["error_detail"], "empty_correction_payload")
        self.assertEqual(env.store.episode_hash(), before)

    def test_nested_correction_payload_is_canonical_and_conflicts_fail_closed(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        state, output = env.step(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "modification": {
                        "measurement_updates": [{"index": 0, "value": 1.0}],
                    },
                },
            }
        )
        self.assertEqual(output["execution_status"], "success")
        source = env.store.get_state(state["candidate_state_id"])["source_action"]
        self.assertNotIn("modification", source["arguments"])
        self.assertEqual(source["arguments"]["measurement_updates"], {0: 1.0})

        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        before = env.store.episode_hash()
        _, output = env.step(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "measurement_updates": {0: 1.0},
                    "modification": {"measurement_updates": {1: 2.0}},
                },
            }
        )
        self.assertEqual(output["error_code"], "schema_error")
        self.assertEqual(env.store.episode_hash(), before)

        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        before = env.store.episode_hash()
        _, output = env.step(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "measurement_index": 0,
                    "modification": {"measurement_updates": {1: 2.0}},
                },
            }
        )
        self.assertEqual(output["error_code"], "schema_error")
        self.assertEqual(env.store.episode_hash(), before)

    def test_non_json_policy_values_are_collectable_failures(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        _, output = env.step(
            {
                "tool": "run_wls",
                "arguments": {"state_id": state["active_state_id"], "bad": {1, 2}},
            }
        )
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "schema_error")
        self.assertEqual(env.history[-1]["action"]["tool"], "__invalid_action__")

    def test_commit_precomputation_failure_is_atomic_and_history_uses_parent(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        parent_id = state["active_state_id"]
        state, _ = env.step(correct_measurement(parent_id))
        candidate_id = state["candidate_state_id"]
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": candidate_id}})
        before = env.store.episode_hash()

        def fail_truth_update(candidate):
            raise RuntimeError("private truth failure")

        env._truth_after_commit = fail_truth_update
        state, output = env.step(
            {"tool": "commit_state", "arguments": {"candidate_state_id": candidate_id}}
        )
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(env.store.episode_hash(), before)
        self.assertEqual(state["active_state_id"], parent_id)
        self.assertEqual(state["candidate_state_id"], candidate_id)

        del env._truth_after_commit
        state, output = env.step(
            {"tool": "commit_state", "arguments": {"candidate_state_id": candidate_id}}
        )
        self.assertEqual(output["execution_status"], "success")
        self.assertEqual(env.history[-1]["state_id"], parent_id)
        self.assertEqual(env.history[-1]["candidate_state_id"], candidate_id)

    def test_context_provider_receives_physical_state_payload(self):
        seen = []

        def provider(payload):
            seen.append(payload)
            return {"parameter_context_ready": True}

        env = TransactionalPSSEEnv(
            context_providers={"get_parameter_context": provider}
        )
        state = env.reset(synthetic_scenario())
        _, output = env.step(
            {"tool": "get_parameter_context", "arguments": {"state_id": state["active_state_id"]}}
        )
        self.assertEqual(output["execution_status"], "success")
        self.assertEqual(seen[0]["case"], synthetic_scenario()["case"])
        self.assertEqual(seen[0]["measurements"], synthetic_scenario()["measurements"])
        self.assertIn("policy_observation", seen[0])

    def test_parameter_and_topology_macros_change_physical_case(self):
        parameter_env = TransactionalPSSEEnv()
        state = parameter_env.reset(
            {
                "scenario_id": "parameter",
                "case": {"branch": [{"branch_id": "L1", "x": 0.1, "status": 1}]},
                "measurements": [1.0],
                "clean_case": {"branch": [{"branch_id": "L1", "x": 0.2, "status": 1}]},
                "true_parameter_errors": [{"line_index": 0, "clean": 0.2}],
            }
        )
        parameter_env.step(
            {"tool": "get_parameter_context", "arguments": {"state_id": state["active_state_id"]}}
        )
        state, output = parameter_env.step(
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": state["active_state_id"], "line_index": 0, "value": 0.2},
            }
        )
        self.assertEqual(output["execution_status"], "success")
        self.assertEqual(parameter_env.store.get_state(state["candidate_state_id"])["case"]["branch"][0]["x"], 0.2)

        topology_env = TransactionalPSSEEnv()
        state = topology_env.reset(
            {
                "scenario_id": "topology",
                "case": {"branch": [{"branch_id": "L1", "x": 0.1, "status": 1}]},
                "measurements": [1.0],
                "true_topology_errors": [{"branch_id": "L1", "expected_status": 0}],
            }
        )
        topology_env.step(
            {"tool": "get_topology_context", "arguments": {"state_id": state["active_state_id"]}}
        )
        state, output = topology_env.step(
            {
                "tool": "correct_topology",
                "arguments": {"state_id": state["active_state_id"], "branch_id": "L1", "status": 0},
            }
        )
        self.assertEqual(output["execution_status"], "success")
        self.assertEqual(topology_env.store.get_state(state["candidate_state_id"])["case"]["branch"][0]["status"], 0)

    def test_topology_index_conventions_all_dispatch(self):
        for reference in (
            {"branch_row0": 0},
            {"line_index": 0},
            {"line_index1": 1},
        ):
            env = TransactionalPSSEEnv()
            state = env.reset(
                {
                    "scenario_id": f"topology-{next(iter(reference))}",
                    "case": {"branch": [{"branch_id": "b0", "status": 0}]},
                    "measurements": [1.0],
                    "true_topology_errors": [{**reference, "expected_status": 1}],
                }
            )
            env.step(
                {"tool": "get_topology_context", "arguments": {"state_id": state["active_state_id"]}}
            )
            state, output = env.step(
                {
                    "tool": "correct_topology",
                    "arguments": {"state_id": state["active_state_id"], **reference, "status": 1},
                }
            )
            self.assertEqual(output["execution_status"], "success")
            candidate = env.store.get_state(state["candidate_state_id"])
            self.assertEqual(candidate["case"]["branch"][0]["status"], 1)

    def test_parameter_branch_id_dispatches(self):
        env = TransactionalPSSEEnv()
        state = env.reset(
            {
                "scenario_id": "parameter-branch-id",
                "case": {"branch": [{"branch_id": "b0", "x": 9.0}]},
                "measurements": [1.0],
                "clean_case": {"branch": [{"branch_id": "b0", "x": 1.0}]},
                "true_parameter_errors": [{"branch_id": "b0", "parameter": "x", "clean": 1.0}],
            }
        )
        env.step({"tool": "get_parameter_context", "arguments": {"state_id": state["active_state_id"]}})
        state, output = env.step(
            {
                "tool": "correct_parameters",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "branch_id": "b0",
                    "parameter": "x",
                    "value": 1.0,
                },
            }
        )
        self.assertEqual(output["execution_status"], "success")
        self.assertEqual(env.store.get_state(state["candidate_state_id"])["case"]["branch"][0]["x"], 1.0)

    def test_privileged_oracle_exception_text_is_redacted(self):
        class BrokenCandidateOracle:
            def label_candidate(self, **kwargs):
                raise RuntimeError("hidden_truth=true_measurement_errors")

        env = TransactionalPSSEEnv(candidate_quality_oracle=BrokenCandidateOracle())
        state = env.reset(synthetic_scenario())
        state, _ = env.step(correct_measurement(state["active_state_id"]))
        _, output = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        serialized = json.dumps({"output": output, "observation": env.get_policy_observation().as_dict()})
        self.assertEqual(output["error_detail"], "RuntimeError")
        self.assertNotIn("hidden_truth", serialized)
        self.assertNotIn("true_measurement_errors", serialized)

    def test_observable_signature_does_not_claim_complete_truth(self):
        env = TransactionalPSSEEnv()
        env.reset(
            {
                "scenario_id": "observable-signature-only",
                "case": {},
                "measurements": [1.0],
                "unresolved_signatures": ["measurement_residual_outlier"],
            }
        )

        self.assertFalse(env.get_oracle_state().hidden_truth["truth_complete"])


class CandidateQualityTests(unittest.TestCase):
    def setUp(self):
        self.oracle = CandidateQualityOracle()
        self.parent = {"state_id": "e:s0"}
        self.candidate = {"state_id": "e:s1", "parent_state_id": "e:s0"}
        self.action = {"tool": "correct_measurements", "arguments": {"measurement_updates": {0: 1.0}}}

    def assess(self, verification=None, truth=None, action=None):
        return self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=action or self.action,
            candidate_state=self.candidate,
            verification_output=verification or {},
            hidden_truth=truth,
        )

    def test_correct_final_candidate_is_accept_final(self):
        result = self.assess(
            {"globally_resolved": True},
            {"target_fixed": True, "remaining_fault_count": 0},
        )
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_FINAL)

    def test_correct_partial_candidate_is_accept_partial(self):
        result = self.assess({}, {"target_fixed": True, "remaining_fault_count": 1})
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)

    def test_parameter_repair_with_marginal_target_on_clean_solve_is_accept_final(self):
        # Frozen root r0_8c0755fce51c after the 2026-09-03 Jacobian repair:
        # the true line's multiplier settles at 3.13 against the 3.0 cutoff
        # while the solve is globally clean (79 against 130) and nothing else
        # remains suspect.  Rejecting it sent the expert to healthy lines.
        action = {"tool": "correct_parameters", "arguments": {"line_index": 7}}
        # In-memory case pair: only row 6's reactance changes, so the
        # fail-closed structural collateral check sees a clean single-row edit.
        parent_rows = [
            {"branch_id": f"L{index + 1}", "r": 0.01 * (index + 1), "x": 0.05 * (index + 1)}
            for index in range(20)
        ]
        candidate_rows = [dict(row) for row in parent_rows]
        candidate_rows[6]["x"] = 0.0129
        measurements = [1.0] * 122
        parent = {
            **self.parent,
            "case": {"baseMVA": 100.0, "branch": parent_rows},
            "measurements": list(measurements),
        }
        candidate = {
            **self.candidate,
            "case": {"baseMVA": 100.0, "branch": candidate_rows},
            "measurements": list(measurements),
        }
        verification = {
            "evidence_source": "deployment_wls:lagrangian_port",
            "target_fixed": False,
            "target_progress": 0.0,
            "target_metric_kind": "max_abs_branch_multiplier",
            "target_metric_value": 3.1286,
            "target_metric_threshold": 3.0,
            "remaining_suspect_count": 0,
            "post_action_resolved": True,
            "globally_resolved": False,
            "no_material_anomaly_remaining": True,
            "global_progress": 0.62,
            "physical_constraints_ok": True,
        }

        def assess(verification_output, source_action=action):
            return self.oracle.label_candidate(
                parent_state=parent,
                source_action=source_action,
                candidate_state=candidate,
                verification_output=verification_output,
                hidden_truth=None,
            )

        result = assess(verification)
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_FINAL)
        self.assertEqual(result.progress_class, "observable_resolved_marginal_target")

        # Outside the tolerance band, with another suspect left, or with the
        # anomaly remaining, no final acceptance.
        cases = (
            ({"target_metric_value": 4.0}, CandidateDisposition.REJECT),
            ({"remaining_suspect_count": 1}, CandidateDisposition.REJECT),
            (
                {"post_action_resolved": False, "no_material_anomaly_remaining": False},
                CandidateDisposition.REJECT,
            ),
        )
        for override, expected in cases:
            with self.subTest(override=override):
                self.assertEqual(assess({**verification, **override}).disposition, expected)
        # A topology edit never uses this basis (its target test is structural).
        topology = {"tool": "correct_topology", "arguments": {"line_index": 7, "status": 0}}
        self.assertNotEqual(
            assess(verification, topology).disposition, CandidateDisposition.ACCEPT_FINAL
        )

    def test_wrong_family_candidate_is_reject(self):
        result = self.assess(
            {},
            {
                "truth_complete": True,
                "true_measurement_errors": [{"index": 0}],
                "true_parameter_errors": [],
                "true_topology_errors": [],
            },
            {"tool": "correct_parameters", "arguments": {"line_index": 2}},
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)

    def test_healthy_component_corruption_is_reject(self):
        result = self.assess({}, {"healthy_component_modified": True})
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertTrue(result.collateral_damage)

    def test_unknown_remaining_faults_is_inconclusive(self):
        result = self.assess({}, {"target_fixed": True})
        self.assertEqual(result.disposition, CandidateDisposition.INCONCLUSIVE)
        self.assertIsNone(result.remaining_fault_count)

    def test_learner_controlled_candidate_metadata_cannot_claim_success(self):
        candidate = {
            **self.candidate,
            "metadata": {"target_fixed": True, "remaining_fault_count": 0},
        }
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=self.action,
            candidate_state=candidate,
            verification_output={},
            hidden_truth=None,
        )
        self.assertNotEqual(result.disposition, CandidateDisposition.ACCEPT_FINAL)

    def test_deployment_mode_ignores_hidden_truth(self):
        result = CandidateQualityOracle(mode="deployment").label_candidate(
            parent_state=self.parent,
            source_action=self.action,
            candidate_state=self.candidate,
            verification_output={},
            hidden_truth={"target_fixed": True, "remaining_fault_count": 0},
        )
        self.assertNotEqual(result.disposition, CandidateDisposition.ACCEPT_FINAL)

    def test_deployment_resolution_without_physical_evidence_is_inconclusive(self):
        result = CandidateQualityOracle(mode="deployment").label_candidate(
            parent_state=self.parent,
            source_action=self.action,
            candidate_state=self.candidate,
            verification_output={
                "target_progress": 1.0,
                "remaining_anomaly_score": 0.0,
                "anomaly_threshold": 1.0,
                "globally_resolved": True,
            },
        )
        self.assertEqual(result.disposition, CandidateDisposition.INCONCLUSIVE)
        self.assertIsNone(result.remaining_true_fault_count)

    def test_deployment_global_test_supports_final_partial_and_inconclusive(self):
        parent = {"state_id": "e:s0", "case": {}, "measurements": [9.0]}
        candidate = {
            "state_id": "e:s1",
            "parent_state_id": "e:s0",
            "case": {},
            "measurements": [1.0],
        }
        oracle = CandidateQualityOracle(mode="deployment")
        base = {
            "target_progress": 1.0,
            "global_progress": 0.5,
            "physical_constraints_ok": True,
            "new_constraint_violations": 0,
        }
        final = oracle.label_candidate(
            parent_state=parent,
            source_action=self.action,
            candidate_state=candidate,
            verification_output={**base, "globally_resolved": True, "remaining_suspect_count": 0},
        )
        partial = oracle.label_candidate(
            parent_state=parent,
            source_action=self.action,
            candidate_state=candidate,
            verification_output={**base, "globally_resolved": False, "remaining_suspect_count": 2},
        )
        unknown = oracle.label_candidate(
            parent_state=parent,
            source_action=self.action,
            candidate_state=candidate,
            verification_output=base,
        )
        self.assertEqual(final.disposition, CandidateDisposition.ACCEPT_FINAL)
        self.assertEqual(partial.disposition, CandidateDisposition.ACCEPT_PARTIAL)
        self.assertEqual(unknown.disposition, CandidateDisposition.INCONCLUSIVE)

    def test_deployment_rejects_broad_or_malformed_physical_edits(self):
        verification = {
            "target_fixed": True,
            "target_progress": 1.0,
            "remaining_fault_count": 0,
            "globally_resolved": True,
        }
        broad = CandidateQualityOracle(mode="deployment").label_candidate(
            parent_state={"state_id": "e:s0", "case": {}, "measurements": [9.0, 8.0]},
            source_action={
                "tool": "correct_measurements",
                "arguments": {"measurement_updates": {0: 1.0, 1: 2.0}},
            },
            candidate_state={
                "state_id": "e:s1",
                "parent_state_id": "e:s0",
                "case": {},
                "measurements": [1.0, 2.0],
            },
            verification_output=verification,
        )
        malformed = CandidateQualityOracle(mode="deployment").label_candidate(
            parent_state={
                "state_id": "e:s0",
                "case": {"branch": [{"x": 9.0}]},
                "measurements": [1.0],
            },
            source_action={
                "tool": "correct_parameters",
                "arguments": {"branch_row0": 0, "parameter": "x", "value": 1.0},
            },
            candidate_state={
                "state_id": "e:s1",
                "parent_state_id": "e:s0",
                "case": {"branch": "not-a-row-list"},
                "measurements": [1.0],
            },
            verification_output=verification,
        )
        for result in (broad, malformed):
            self.assertEqual(result.disposition, CandidateDisposition.REJECT)
            self.assertTrue(result.collateral_damage)

    def test_provenance_rejection_keeps_belief_update_consistent(self):
        result = self.oracle.label_candidate(
            parent_state={"state_id": "e:s0"},
            source_action=self.action,
            candidate_state={"state_id": "e:s1", "parent_state_id": "other:s0"},
            verification_output={"globally_resolved": True},
            hidden_truth={"target_fixed": True, "remaining_fault_count": 0},
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "invalid_provenance")
        self.assertEqual(result.belief_update, {"measurement": "invalid_provenance"})

    def test_deployment_mode_rejects_structural_cross_family_mutations(self):
        verification = {
            "target_fixed": True,
            "remaining_fault_count": 0,
            "globally_resolved": True,
        }
        cases = (
            (
                {
                    "tool": "correct_measurements",
                    "arguments": {"state_id": "e:s0", "case_updates": {"x": 999}},
                },
                {"state_id": "e:s0", "case": {"x": 1}, "measurements": [9.0]},
                {
                    "state_id": "e:s1",
                    "parent_state_id": "e:s0",
                    "case": {"x": 999},
                    "measurements": [9.0],
                },
            ),
            (
                {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": "e:s0", "line_index": 0, "value": 1.0},
                },
                {"state_id": "e:s0", "case": {"branch": [{"x": 9.0}]}, "measurements": [5.0]},
                {
                    "state_id": "e:s1",
                    "parent_state_id": "e:s0",
                    "case": {"branch": [{"x": 1.0}]},
                    "measurements": [99.0],
                },
            ),
        )
        for action, parent, candidate in cases:
            result = CandidateQualityOracle(mode="deployment").label_candidate(
                parent_state=parent,
                source_action=action,
                candidate_state=candidate,
                verification_output=verification,
            )
            self.assertEqual(result.disposition, CandidateDisposition.REJECT)
            self.assertTrue(result.collateral_damage)

    def _record_candidate_oracle_call(self):
        calls = []

        class RecordingOracle:
            def label_candidate(self, **kwargs):
                calls.append(kwargs)
                return CandidateAssessment(
                    disposition=CandidateDisposition.ACCEPT_PARTIAL,
                    progress_class="target_progress_remaining_true_faults",
                    remaining_true_fault_count=1,
                )

        env = TransactionalPSSEEnv(candidate_quality_oracle=RecordingOracle())
        state = env.reset(synthetic_scenario(parameter_fault=True))
        state, _ = env.step(correct_measurement(state["active_state_id"]))
        env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        return calls[0], state["candidate_state_id"]

    def test_candidate_oracle_receives_source_action(self):
        call, _ = self._record_candidate_oracle_call()
        self.assertEqual(call["source_action"]["tool"], "correct_measurements")

    def test_candidate_oracle_receives_candidate_state(self):
        call, candidate_id = self._record_candidate_oracle_call()
        self.assertEqual(call["candidate_state"]["state_id"], candidate_id)

    def test_candidate_oracle_receives_hidden_truth_only_in_oracle_view(self):
        call, _ = self._record_candidate_oracle_call()
        self.assertIn("true_parameter_errors", call["hidden_truth"])
        self.assertNotIn("true_parameter_errors", json.dumps(call["candidate_state"]))

    def test_partial_and_healthy_change_have_different_labels(self):
        env = TransactionalPSSEEnv()
        root = env.reset(synthetic_scenario(parameter_fault=True))
        partial, _ = env.step(correct_measurement(root["active_state_id"], index=0))
        partial, _ = env.step({"tool": "run_wls", "arguments": {"state_id": partial["candidate_state_id"]}})
        self.assertEqual(partial["candidate_disposition"], "ACCEPT_PARTIAL")

        env = TransactionalPSSEEnv()
        root = env.reset(synthetic_scenario(parameter_fault=True))
        rejected, _ = env.step(correct_measurement(root["active_state_id"], index=1))
        rejected, _ = env.step({"tool": "run_wls", "arguments": {"state_id": rejected["candidate_state_id"]}})
        self.assertEqual(rejected["candidate_disposition"], "REJECT")

    def test_separate_clean_measurements_reject_wrong_magnitude_and_noop(self):
        scenario = {
            "scenario_id": "separate-clean",
            "case": {},
            "measurements": [9.0, 9.0],
            "clean_measurements": [1.0, 1.0],
            "true_measurement_errors": [{"index": 0}],
        }
        env = TransactionalPSSEEnv()
        state = env.reset(scenario)
        state, _ = env.step(correct_measurement(state["active_state_id"], value=90.0))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "REJECT")

        result = self.oracle.label_candidate(
            parent_state={"state_id": "e:s0", "state_hash": "same"},
            source_action=correct_measurement("e:s0", value=9.0),
            candidate_state={"state_id": "e:s1", "parent_state_id": "e:s0", "state_hash": "same"},
            verification_output={},
            hidden_truth={
                "truth_complete": True,
                "clean_measurements": [1.0, 1.0],
                "true_measurement_errors": [{"index": 0}],
            },
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)

    def test_candidate_rejects_cross_family_and_healthy_target_mutations(self):
        base = {
            "scenario_id": "collateral",
            "case": {"x": 1},
            "measurements": [9.0, 2.0],
            "clean_case": {"x": 1},
            "clean_measurements": [1.0, 2.0],
            "true_measurement_errors": [{"index": 0}],
        }
        for arguments in (
            {"measurement_updates": {0: 1.0, 1: 99.0}},
            {"measurement_updates": {0: 1.0}, "case_updates": {"x": 999}},
        ):
            env = TransactionalPSSEEnv()
            state = env.reset(base)
            action = {
                "tool": "correct_measurements",
                "arguments": {"state_id": state["active_state_id"], **arguments},
            }
            state, _ = env.step(action)
            state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
            self.assertEqual(state["candidate_disposition"], "REJECT")

        env = TransactionalPSSEEnv()
        state = env.reset(
            {
                "scenario_id": "parameter-collateral",
                "case": {"branch": [{"x": 9.0}]},
                "measurements": [5.0],
                "clean_case": {"branch": [{"x": 1.0}]},
                "clean_measurements": [5.0],
                "true_parameter_errors": [{"line_index": 0, "clean": 1.0}],
            }
        )
        env.step({"tool": "get_parameter_context", "arguments": {"state_id": state["active_state_id"]}})
        state, _ = env.step(
            {
                "tool": "correct_parameters",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "line_index": 0,
                    "value": 1.0,
                    "measurement_updates": {0: 99.0},
                },
            }
        )
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "REJECT")

    def test_last_remaining_fault_is_accept_final(self):
        def observable_wls(state):
            score = sum(
                abs(float(observed) - expected)
                for observed, expected in zip(state.get("measurements") or [], (1.0, 2.0))
            )
            return {
                "wls_objective": score,
                "remaining_anomaly_score": score,
                "anomaly_threshold": 0.5,
                "post_action_resolved": score < 0.5,
                "unresolved_signatures": [] if score < 0.5 else ["measurement_residual"],
            }

        env = TransactionalPSSEEnv(wls_runner=observable_wls)
        state = env.reset(
            {
                "scenario_id": "two-faults",
                "case": {},
                "measurements": [9.0, 8.0],
                "clean_measurements": [1.0, 2.0],
                "true_measurement_errors": [{"index": 0}, {"index": 1}],
            }
        )
        state, _ = env.step(correct_measurement(state["active_state_id"], index=0, value=1.0))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_PARTIAL")
        state, _ = env.step({"tool": "commit_state", "arguments": {"candidate_state_id": state["candidate_state_id"]}})
        state, _ = env.step(correct_measurement(state["active_state_id"], index=1, value=2.0))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_FINAL")
        env.step({"tool": "commit_state", "arguments": {"candidate_state_id": state["candidate_state_id"]}})
        self.assertEqual(env.get_oracle_state().remaining_true_faults, [])
        self.assertEqual(env.get_oracle_state().hidden_truth["remaining_true_fault_count"], 0)
        self.assertEqual(ExpertPolicyOracle().next_actions(env.get_oracle_state())[0]["tool"], "finalize_diagnosis")

    def test_remaining_truth_is_initialized_and_explicit_subset_does_not_resurrect(self):
        fault0 = {"index": 0, "clean": 1.0}
        fault1 = {"index": 1, "clean": 2.0}
        env = TransactionalPSSEEnv()
        state = env.reset(
            {
                "scenario_id": "explicit-remaining-subset",
                "case": {},
                "measurements": [1.0, 9.0],
                "clean_measurements": [1.0, 2.0],
                "true_measurement_errors": [fault0, fault1],
                "remaining_true_faults": [fault1],
            }
        )
        oracle_state = env.get_oracle_state()
        self.assertEqual(oracle_state.true_measurement_errors, [fault1])
        self.assertEqual(oracle_state.remaining_true_faults, [fault1])
        self.assertEqual(oracle_state.hidden_truth["remaining_true_fault_count"], 1)

        state, _ = env.step(correct_measurement(state["active_state_id"], index=1, value=2.0))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_FINAL")
        state, _ = env.step(
            {"tool": "commit_state", "arguments": {"candidate_state_id": state["candidate_state_id"]}}
        )
        oracle_state = env.get_oracle_state()
        self.assertEqual(oracle_state.true_measurement_errors, [])
        self.assertEqual(oracle_state.remaining_true_faults, [])
        self.assertEqual(oracle_state.hidden_truth["remaining_true_fault_count"], 0)

        fresh = TransactionalPSSEEnv()
        fresh.reset(
            {
                "scenario_id": "initialize-all-remaining",
                "case": {},
                "measurements": [9.0],
                "true_measurement_errors": [fault0],
            }
        )
        self.assertEqual(fresh.get_oracle_state().remaining_true_faults, [fault0])
        self.assertEqual(fresh.get_oracle_state().hidden_truth["remaining_true_fault_count"], 1)

    def test_synthetic_truth_overrides_conflicting_verifier_count(self):
        def optimistic_wls(state):
            return {
                "execution_status": "success",
                "remaining_fault_count": 0,
                "globally_resolved": True,
            }

        env = TransactionalPSSEEnv(wls_runner=optimistic_wls)
        state = env.reset(
            {
                "scenario_id": "truth-authority",
                "case": {},
                "measurements": [9.0, 8.0],
                "clean_measurements": [1.0, 2.0],
                "true_measurement_errors": [
                    {"index": 0, "clean": 1.0},
                    {"index": 1, "clean": 2.0},
                ],
            }
        )
        state, _ = env.step(correct_measurement(state["active_state_id"], index=0, value=1.0))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_PARTIAL")
        self.assertEqual(
            env.get_oracle_state().candidate_assessment["remaining_true_fault_count"],
            1,
        )

    def test_measurement_update_list_form_matches_mapping_semantics(self):
        scenario = {
            "scenario_id": "list-updates",
            "case": {},
            "measurements": [9.0, 8.0],
            "clean_measurements": [1.0, 2.0],
            "true_measurement_errors": [
                {"index": 0, "clean": 1.0},
                {"index": 1, "clean": 2.0},
            ],
        }
        env = TransactionalPSSEEnv()
        state = env.reset(scenario)
        state, _ = env.step(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "measurement_updates": [{"index": 0, "value": 1.0}],
                },
            }
        )
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_PARTIAL")

        env = TransactionalPSSEEnv()
        state = env.reset(scenario)
        state, _ = env.step(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "measurement_updates": [
                        {"index": 0, "value": 1.0},
                        {"index": 1, "value": 2.0},
                    ],
                },
            }
        )
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_FINAL")

    def test_topology_status_field_must_match_the_injected_fault(self):
        scenario = {
            "scenario_id": "custom-status-field",
            "case": {"branch": [{"branch_id": "b0", "in_service": 0}]},
            "measurements": [1.0],
            "true_topology_errors": [
                {"branch_row0": 0, "status_field": "in_service", "expected_status": 1}
            ],
        }
        env = TransactionalPSSEEnv()
        state = env.reset(scenario)
        env.step({"tool": "get_topology_context", "arguments": {"state_id": state["active_state_id"]}})
        state, _ = env.step(
            {
                "tool": "correct_topology",
                "arguments": {"state_id": state["active_state_id"], "branch_row0": 0, "status": 1},
            }
        )
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "REJECT")

        env = TransactionalPSSEEnv()
        state = env.reset(scenario)
        env.step({"tool": "get_topology_context", "arguments": {"state_id": state["active_state_id"]}})
        state, _ = env.step(
            {
                "tool": "correct_topology",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "branch_row0": 0,
                    "status_field": "in_service",
                    "status": 1,
                },
            }
        )
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_FINAL")

    def test_partial_fix_rejects_regression_on_another_known_fault(self):
        env = TransactionalPSSEEnv()
        state = env.reset(
            {
                "scenario_id": "mixed-regression",
                "case": {},
                "measurements": [9.0, 8.0],
                "clean_measurements": [1.0, 2.0],
                "true_measurement_errors": [{"index": 0}, {"index": 1}],
            }
        )
        state, _ = env.step(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "measurement_updates": {0: 1.0, 1: 99.0},
                },
            }
        )
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "REJECT")
        self.assertTrue(env.get_oracle_state().candidate_assessment["collateral_damage"])

    def test_parameter_target_conventions_resolve_to_same_branch(self):
        pairs = (
            ({"line_index": 1}, {"line_index1": 1}),
            ({"branch_row0": 0}, {"branch_id": "b0"}),
            ({"branch_id": "b0"}, {"line_index": 1}),
        )
        for truth_target, action_target in pairs:
            env = TransactionalPSSEEnv()
            state = env.reset(
                {
                    "scenario_id": f"convention-{next(iter(truth_target))}-{next(iter(action_target))}",
                    "case": {"branch": [{"branch_id": "b0", "x": 9.0}]},
                    "measurements": [1.0],
                    "clean_case": {"branch": [{"branch_id": "b0", "x": 1.0}]},
                    "true_parameter_errors": [{**truth_target, "parameter": "x", "clean": 1.0}],
                }
            )
            env.step(
                {"tool": "get_parameter_context", "arguments": {"state_id": state["active_state_id"]}}
            )
            state, _ = env.step(
                {
                    "tool": "correct_parameters",
                    "arguments": {
                        "state_id": state["active_state_id"],
                        **action_target,
                        "parameter": "x",
                        "value": 1.0,
                    },
                }
            )
            state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
            self.assertEqual(state["candidate_disposition"], "ACCEPT_FINAL")

    def test_truth_advancement_preserves_second_fault_on_same_component(self):
        env = TransactionalPSSEEnv()
        state = env.reset(
            {
                "scenario_id": "same-component",
                "case": {"branch": [{"branch_id": "b0", "x": 9.0, "r": 8.0}]},
                "measurements": [1.0],
                "clean_case": {"branch": [{"branch_id": "b0", "x": 1.0, "r": 0.1}]},
                "true_parameter_errors": [
                    {"branch_row0": 0, "parameter": "x", "clean": 1.0},
                    {"branch_row0": 0, "parameter": "r", "clean": 0.1},
                ],
            }
        )
        env.step({"tool": "get_parameter_context", "arguments": {"state_id": state["active_state_id"]}})
        state, _ = env.step(
            {
                "tool": "correct_parameters",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "branch_row0": 0,
                    "parameter": "x",
                    "value": 1.0,
                },
            }
        )
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_PARTIAL")
        env.step({"tool": "commit_state", "arguments": {"candidate_state_id": state["candidate_state_id"]}})
        remaining = env.get_oracle_state().true_parameter_errors
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["parameter"], "r")
        self.assertEqual(env.get_oracle_state().hidden_truth["remaining_true_fault_count"], 1)

    def test_episode_static_candidate_labels_do_not_override_physical_truth(self):
        common = {
            "scenario_id": "static-labels",
            "case": {},
            "measurements": [9.0],
            "clean_measurements": [1.0],
            "true_measurement_errors": [{"index": 0}],
            "hidden_truth": {"target_fixed": True, "healthy_component_modified": True},
        }
        env = TransactionalPSSEEnv()
        state = env.reset(common)
        state, _ = env.step(correct_measurement(state["active_state_id"], value=8.0))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "REJECT")

        env = TransactionalPSSEEnv()
        state = env.reset(common)
        state, _ = env.step(correct_measurement(state["active_state_id"], value=1.0))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        self.assertEqual(state["candidate_disposition"], "ACCEPT_FINAL")


class ExpertPolicyTests(unittest.TestCase):
    def setUp(self):
        self.oracle = ExpertPolicyOracle()

    def test_missing_parameter_context_routes_to_parameter_context(self):
        state = {
            "active_state_id": "e:s0",
            "last_tool_status": "failure",
            "last_tool_output": {
                "execution_status": "failure",
                "error_code": "missing_precondition",
                "error_detail": "parameter_context_missing",
            },
        }
        self.assertEqual(self.oracle.next_actions(state)[0]["tool"], "get_parameter_context")

    def test_rejected_measurement_hypothesis_increases_parameter_priority(self):
        state = {
            "active_state_id": "e:s0",
            "unresolved_signatures": ["parameter_error_possible"],
            "rejected_hypotheses": [{"source_action": {"tool": "correct_measurements", "arguments": {}}}],
        }
        self.assertEqual(self.oracle.next_actions(state)[0]["tool"], "get_parameter_context")

    def test_accepted_partial_measurement_fix_refreshes_measurement_context(self):
        state = {
            "active_state_id": "e:s1",
            "unresolved_signatures": [
                "wls_branch_multiplier_dominant line_status_or_parameter line=7"
            ],
            "accepted_corrections": [
                {
                    "source_action": {"tool": "correct_measurements", "arguments": {}},
                    "candidate_disposition": "ACCEPT_PARTIAL",
                }
            ],
        }
        self.assertEqual(self.oracle.next_actions(state)[0]["tool"], "get_measurement_context")

    def test_expert_does_not_repeat_rejected_action_signature(self):
        rejected = correct_measurement("e:s0")
        state = OracleState(
            policy_observation=PolicyObservation(
                active_state_id="e:s0",
                rejected_hypotheses=[
                    {"source_action": rejected, "action_signature": action_signature(rejected)}
                ],
                remaining_budget=4,
            ),
            true_measurement_errors=[{"index": 0}],
            oracle_action_hints=[rejected],
        )
        actions = self.oracle.next_actions(state)
        self.assertNotIn(action_signature(rejected), {action_signature(action) for action in actions})

    def test_expert_does_not_use_hidden_truth_in_policy_features(self):
        env = TransactionalPSSEEnv()
        env.reset(synthetic_scenario(parameter_fault=True))
        observation = env.get_policy_observation().as_dict()
        self.oracle.next_actions(env.get_oracle_state())
        serialized = json.dumps(observation)
        self.assertNotIn("true_parameter_errors", serialized)

    def test_inconclusive_candidate_rolls_back_after_one_evidence_request(self):
        state = OracleState(
            policy_observation=PolicyObservation(
                active_state_id="e:s0",
                candidate_state_id="e:s1",
                candidate_status="verified",
                has_open_candidate=True,
                has_verified_candidate=True,
                last_tool="ask_for_more_evidence",
                last_tool_status="success",
                last_tool_output={"execution_status": "success"},
            ),
            candidate_disposition="INCONCLUSIVE",
        )
        self.assertEqual(self.oracle.next_actions(state)[0]["tool"], "rollback_state")


class _RunWLSOracle:
    def __init__(self):
        self.history_lengths = []

    def next_actions(self, state, history=None):
        self.history_lengths.append(len(history or []))
        return [{"tool": "run_wls", "arguments": {"state_id": state.get("active_state_id")}}]

    def label_transition(self, **kwargs):
        return {
            "process_valid": kwargs["tool_output"]["execution_status"] == "success",
            "error_code": kwargs["tool_output"].get("error_code"),
            "error_detail": kwargs["tool_output"].get("error_detail"),
            "candidate_disposition": None,
            "progress_class": None,
            "valid_next_actions": [],
        }


class _RunWLSPolicy:
    def __init__(self, version=0):
        self.version = version

    def act(self, observation):
        return {"tool": "run_wls", "arguments": {"state_id": observation["active_state_id"]}}


class DaggerCollectorTests(unittest.TestCase):
    def test_policy_exception_is_collected(self):
        class BrokenPolicy:
            def act(self, observation):
                raise RuntimeError("boom")

        collector = DaggerRolloutCollector(
            env=TransactionalPSSEEnv(), policy=BrokenPolicy(), expert_oracle=ExpertPolicyOracle(), rng=random.Random(0)
        )
        rows = collector.collect_iteration(scenarios=[synthetic_scenario()], iteration=0, beta=0.0, max_steps=1)
        self.assertEqual(rows[0]["model_action"]["tool"], "__invalid_action__")
        self.assertEqual(rows[0]["tool_output"]["error_code"], "policy_exception")

    def test_malformed_policy_output_is_collected(self):
        class BrokenJSONPolicy:
            def act(self, observation):
                return '{"tool":'

        rows = DaggerRolloutCollector(
            env=TransactionalPSSEEnv(), policy=BrokenJSONPolicy(), expert_oracle=ExpertPolicyOracle(), rng=random.Random(0)
        ).collect_iteration(scenarios=[synthetic_scenario()], iteration=0, beta=0.0, max_steps=1)
        self.assertEqual(rows[0]["model_action"]["arguments"]["error_code"], "json_parse_error")
        self.assertEqual(rows[0]["tool_output"]["error_code"], "json_parse_error")

    def test_non_json_argument_value_is_collected(self):
        class PythonObjectPolicy:
            def act(self, observation):
                return {
                    "tool": "run_wls",
                    "arguments": {"state_id": observation["active_state_id"], "bad": {1, 2}},
                }

        rows = DaggerRolloutCollector(
            env=TransactionalPSSEEnv(),
            policy=PythonObjectPolicy(),
            expert_oracle=ExpertPolicyOracle(),
            rng=random.Random(0),
        ).collect_iteration(
            scenarios=[synthetic_scenario()], iteration=0, beta=0.0, max_steps=1
        )
        self.assertEqual(rows[0]["model_action"]["tool"], "__invalid_action__")
        self.assertEqual(rows[0]["tool_output"]["error_code"], "schema_error")

    def test_next_state_oracle_receives_updated_history(self):
        oracle = _RunWLSOracle()
        collector = DaggerRolloutCollector(
            env=TransactionalPSSEEnv(), policy=_RunWLSPolicy(), expert_oracle=oracle, rng=random.Random(0)
        )
        rows = collector.collect_iteration(scenarios=[synthetic_scenario()], iteration=0, beta=0.0, max_steps=2)
        self.assertIn(1, oracle.history_lengths)
        self.assertIn(2, oracle.history_lengths)
        self.assertEqual(rows[0]["parent_state_summary"]["active_state_id"], rows[0]["next_state_summary"]["active_state_id"])

    def test_aggregate_dataset_contains_all_iterations(self):
        scenarios = (scenario for scenario in [synthetic_scenario()])
        _, rows = run_dagger(
            policy=_RunWLSPolicy(),
            expert_oracle=_RunWLSOracle(),
            env=TransactionalPSSEEnv(),
            scenarios_by_iteration=scenarios,
            num_iterations=3,
            beta_schedule=[0.0],
            max_steps=1,
            rng=random.Random(0),
        )
        self.assertEqual([row["iteration"] for row in rows], [0, 1, 2])

    def test_generator_scenarios_are_not_silently_exhausted(self):
        scenarios = (scenario for scenario in [synthetic_scenario()])
        _, rows = run_dagger(
            policy=_RunWLSPolicy(),
            expert_oracle=_RunWLSOracle(),
            env=TransactionalPSSEEnv(),
            scenarios_by_iteration=scenarios,
            num_iterations=2,
            beta_schedule=[0.0],
            max_steps=1,
            rng=random.Random(0),
        )
        self.assertEqual(len(rows), 2)

    def test_best_checkpoint_is_returned(self):
        scores = iter([1.0, 2.0, 4.0, 3.0])

        def train(policy, dataset):
            return _RunWLSPolicy(policy.version + 1)

        best, _ = run_dagger(
            policy=_RunWLSPolicy(),
            expert_oracle=_RunWLSOracle(),
            env=TransactionalPSSEEnv(),
            scenarios_by_iteration=[synthetic_scenario()],
            num_iterations=3,
            beta_schedule=[0.0],
            max_steps=1,
            train_policy_fn=train,
            evaluate_fn=lambda policy, env, oracle: next(scores),
            replay_require_late_iteration_model_quota=False,
            rng=random.Random(0),
        )
        self.assertEqual(best.version, 2)

    def test_initial_policy_can_remain_best_checkpoint(self):
        scores = iter([10.0, 2.0, 1.0])

        def train(policy, dataset):
            return _RunWLSPolicy(policy.version + 1)

        best, _ = run_dagger(
            policy=_RunWLSPolicy(),
            expert_oracle=_RunWLSOracle(),
            env=TransactionalPSSEEnv(),
            scenarios_by_iteration=[synthetic_scenario()],
            num_iterations=2,
            beta_schedule=[0.0],
            max_steps=1,
            train_policy_fn=train,
            evaluate_fn=lambda policy, env, oracle: next(scores),
            replay_require_late_iteration_model_quota=False,
            rng=random.Random(0),
        )
        self.assertEqual(best.version, 0)

    def test_fixed_seed_produces_identical_rollout_json(self):
        def collect():
            return DaggerRolloutCollector(
                env=TransactionalPSSEEnv(),
                policy=_RunWLSPolicy(),
                expert_oracle=_RunWLSOracle(),
                rng=random.Random(42),
            ).collect_iteration(scenarios=[synthetic_scenario()], iteration=0, beta=0.5, max_steps=2)

        self.assertEqual(json.dumps(collect(), sort_keys=True), json.dumps(collect(), sort_keys=True))


class DatasetConversionTests(unittest.TestCase):
    def example(self):
        observation = PolicyObservation(active_state_id="e:s0", remaining_budget=3).as_dict()
        return {
            "example_id": "example",
            "scenario_id": "case",
            "policy_observation": observation,
            "history_window": [],
            "preferred_action": {"tool": "run_wls", "arguments": {"state_id": "e:s0"}},
            "labels": {},
        }

    def test_chat_sft_contains_no_privileged_fields(self):
        rows = examples_to_chat_sft(
            [self.example()], available_tools=["run_wls"], protocol="controller"
        )
        user = json.loads(rows[0]["messages"][1]["content"])
        self.assertFalse(any(f'"{key}"' in json.dumps(user) for key in FORBIDDEN_POLICY_KEYS))
        tool_call = rows[0]["messages"][2]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "run_wls")

    def test_chat_generation_fails_on_privileged_key(self):
        example = self.example()
        example["policy_observation"]["clean_case"] = {"hidden": True}
        with self.assertRaises(ValueError):
            examples_to_chat_sft([example], protocol="controller")

    def test_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            write_jsonl(path, [self.example()])
            self.assertEqual(load_jsonl(path), [self.example()])


class ReplayAndEvaluationTests(unittest.TestCase):
    def test_balanced_sampler_respects_state_classes(self):
        classes = [
            "clean_successful",
            "rejected_candidate_recovery",
            "accepted_partial_continuation",
            "invalid_precondition_recovery",
            "terminal_resolved",
            "terminal_operator_escalation",
            "loop_repetition",
        ]
        rows = [{"id": f"{name}-{index}", "state_class": name} for name in classes for index in range(10)]
        sampled = BalancedReplayBuffer(rows).sample(20, rng=random.Random(0))
        counts = {name: sum(row["state_class"] == name for row in sampled) for name in classes}
        self.assertEqual(counts, dict(zip(classes, [6, 5, 4, 2, 1, 1, 1])))
        self.assertEqual(counts["terminal_resolved"], 1)
        self.assertEqual(counts["terminal_operator_escalation"], 1)

    def test_grouped_split_keeps_root_branches_together(self):
        rows = [
            {"root_scenario_id": root, "branch": branch}
            for root in ("a", "b", "c", "d")
            for branch in range(3)
        ]
        split = grouped_scenario_split(rows, seed=3)
        locations = {}
        for split_name, split_rows in split.items():
            for row in split_rows:
                locations.setdefault(row["root_scenario_id"], set()).add(split_name)
        self.assertTrue(all(len(value) == 1 for value in locations.values()))


class CounterfactualTests(unittest.TestCase):
    def test_truth_corrections_preserve_branch_identifier_convention(self):
        for family, target in (
            ("parameter", {"line_index1": 1}),
            ("topology", {"line_index1": 1}),
            ("topology", {"branch_row0": 0}),
        ):
            scenario = {
                "scenario_id": f"counterfactual-{family}-{next(iter(target))}",
                "case": {"branch": [{"x": 9.0, "status": 0}]},
                "measurements": [1.0],
            }
            if family == "parameter":
                scenario["true_parameter_errors"] = [{**target, "parameter": "x", "clean": 1.0}]
            else:
                scenario["true_topology_errors"] = [{**target, "expected_status": 1}]
            env = TransactionalPSSEEnv()
            env.reset(scenario)
            actions = CounterfactualGenerator._truth_correction_actions(env.get_oracle_state())
            tool = "correct_parameters" if family == "parameter" else "correct_topology"
            action = next(item for item in actions if item["tool"] == tool)
            self.assertIn(next(iter(target)), action["arguments"])
            self.assertNotIn(
                "line_index" if "line_index1" in target else "branch_id",
                action["arguments"],
            )

    def test_default_generator_uses_separate_clean_measurements(self):
        rows = CounterfactualGenerator(
            env=TransactionalPSSEEnv(), expert_oracle=ExpertPolicyOracle()
        ).generate(
            scenario={
                "scenario_id": "separate-clean-counterfactual",
                "case": {},
                "measurements": [9.0, 2.0],
                "clean_measurements": [1.0, 2.0],
                "true_measurement_errors": [{"index": 0}],
            }
        )
        families = {row["branch_family"] for row in rows}
        self.assertTrue(
            {
                "wrong_fault_family",
                "wrong_target_component",
                "wrong_correction_magnitude",
                "wrong_correction_sign",
                "skipped_verification",
            }.issubset(families)
        )

    def test_parameter_and_topology_wrong_variants_are_physical_rejections(self):
        scenarios = (
            {
                "scenario_id": "parameter-wrong-variants",
                "case": {"branch": [{"branch_id": "b0", "x": 9.0, "r": 5.0}]},
                "measurements": [1.0],
                "clean_case": {"branch": [{"branch_id": "b0", "x": 1.0, "r": 5.0}]},
                "true_parameter_errors": [
                    {"branch_row0": 0, "parameter": "x", "clean": 1.0}
                ],
            },
            {
                "scenario_id": "topology-wrong-variants",
                "case": {
                    "branch": [
                        {"branch_id": "b0", "status": 0},
                        {"branch_id": "healthy", "status": 1},
                    ]
                },
                "measurements": [1.0],
                "true_topology_errors": [{"branch_row0": 0, "expected_status": 1}],
            },
        )
        for scenario in scenarios:
            rows = CounterfactualGenerator(
                env=TransactionalPSSEEnv(), expert_oracle=ExpertPolicyOracle()
            ).generate(scenario=scenario)
            variants = {
                row["branch_family"]: row
                for row in rows
                if row["branch_family"]
                in {"wrong_target_component", "wrong_correction_magnitude", "wrong_correction_sign"}
            }
            self.assertEqual(set(variants), {
                "wrong_target_component",
                "wrong_correction_magnitude",
                "wrong_correction_sign",
            })
            self.assertTrue(
                all(row["labels"]["candidate_disposition"] == "REJECT" for row in variants.values())
            )
            self.assertTrue(
                all(row["injected_tool_output"]["state_mutated"] for row in variants.values())
            )

    def test_solver_failure_is_process_valid_verification_transition(self):
        def failing_wls(state):
            return {"execution_status": "failure", "error_code": "solver_nonconvergence"}

        env = TransactionalPSSEEnv(wls_runner=failing_wls)
        state = env.reset(synthetic_scenario())
        row = CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
            [InjectedAction("wrong_correction_magnitude", correct_measurement(state["active_state_id"], value=10.0))],
            root_scenario_id="solver-failure",
        )[0]
        self.assertEqual(row["verification_output"]["execution_status"], "failure")
        self.assertTrue(row["verification_transition"]["labels"]["process_valid"])

    def test_wrong_family_branch_generates_reject(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        root_hash = env.store.episode_hash()
        wrong = InjectedAction(
            "wrong_fault_family",
            {
                "tool": "correct_parameters",
                "arguments": {
                    "state_id": state["active_state_id"],
                    "case_updates": {"counterfactual_wrong_family": True},
                },
            },
        )
        rows = CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
            [wrong], root_scenario_id="case"
        )
        self.assertEqual(rows[0]["labels"]["candidate_disposition"], "REJECT")
        self.assertEqual(rows[0]["verification_transition"]["action"]["tool"], "run_wls")
        self.assertEqual(rows[0]["executed_action"]["tool"], "rollback_state")
        self.assertIsNone(rows[0]["next_state_summary"]["candidate_state_id"])
        self.assertEqual(env.store.episode_hash(), root_hash)

    def test_correct_partial_branch_generates_accept_partial(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario(parameter_fault=True))
        partial = InjectedAction("correct_partial", correct_measurement(state["active_state_id"]))
        rows = CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
            [partial], root_scenario_id="case"
        )
        self.assertEqual(rows[0]["labels"]["candidate_disposition"], "ACCEPT_PARTIAL")
        self.assertEqual(rows[0]["executed_action"]["tool"], "commit_state")
        self.assertTrue(rows[0]["continuation_actions"])

    def test_wrong_target_branch_preserves_parent(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        root_hash = env.store.state_hash(state["active_state_id"])
        wrong_target = InjectedAction("wrong_target_component", correct_measurement(state["active_state_id"], index=1))
        rows = CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
            [wrong_target], root_scenario_id="case"
        )
        self.assertEqual(rows[0]["labels"]["candidate_disposition"], "REJECT")
        self.assertEqual(env.store.state_hash(state["active_state_id"]), root_hash)

    def test_wrong_magnitude_and_sign_branches_reject(self):
        for family, value in (("wrong_correction_magnitude", 10.0), ("wrong_correction_sign", -1.0)):
            env = TransactionalPSSEEnv()
            state = env.reset(synthetic_scenario())
            row = CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
                [InjectedAction(family, correct_measurement(state["active_state_id"], value=value))],
                root_scenario_id="case",
            )[0]
            self.assertEqual(row["labels"]["candidate_disposition"], "REJECT")

    def test_counterfactual_branch_does_not_mutate_root(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario(parameter_fault=True))
        root_hash = env.store.episode_hash()
        CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
            [InjectedAction("correct_partial", correct_measurement(state["active_state_id"]))],
            root_scenario_id="case",
        )
        self.assertEqual(env.store.episode_hash(), root_hash)

    def test_skipped_verification_bootstraps_verify_recovery(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        injected = InjectedAction(
            "skipped_verification",
            {"tool": "commit_state", "arguments": {"candidate_state_id": "__candidate__"}},
            (correct_measurement(state["active_state_id"]),),
        )
        row = CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
            [injected], root_scenario_id="case"
        )[0]
        self.assertEqual(row["injected_tool_output"]["execution_status"], "failure")
        self.assertEqual(row["executed_action"]["tool"], "run_wls")

    def test_rollback_of_valid_partial_correction_bootstraps_commit(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario(parameter_fault=True))
        injected = InjectedAction(
            "rollback_of_valid_partial_correction",
            {"tool": "rollback_state", "arguments": {"candidate_state_id": "__candidate__"}},
            (
                correct_measurement(state["active_state_id"]),
                {"tool": "run_wls", "arguments": {"state_id": "__candidate__"}},
            ),
        )
        row = CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
            [injected], root_scenario_id="case"
        )[0]
        self.assertEqual(row["injected_tool_output"]["execution_status"], "failure")
        self.assertEqual(row["executed_action"]["tool"], "commit_state")


class ProcessVerifierTests(unittest.TestCase):
    def test_collector_rows_preserve_complete_transition_for_verifier(self):
        class CorrectionPolicy:
            def act(self, observation):
                return correct_measurement(observation["active_state_id"])

        rows = DaggerRolloutCollector(
            env=TransactionalPSSEEnv(),
            policy=CorrectionPolicy(),
            expert_oracle=ExpertPolicyOracle(),
            rng=random.Random(0),
        ).collect_iteration(scenarios=[synthetic_scenario()], iteration=0, beta=0.0, max_steps=1)
        verifier_row = build_verifier_dataset(rows)[0]
        self.assertEqual(verifier_row["action"]["tool"], "correct_measurements")
        self.assertIsNotNone(verifier_row["candidate_state_summary"]["candidate_state_id"])

    def test_counterfactual_nested_transitions_reach_verifier_dataset(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        rows = CounterfactualGenerator(env=env, expert_oracle=ExpertPolicyOracle()).generate_from_current(
            [InjectedAction("wrong_target_component", correct_measurement(state["active_state_id"], index=1))],
            root_scenario_id="case",
        )
        verifier_rows = build_verifier_dataset(rows)
        self.assertEqual(len(verifier_rows), 3)
        self.assertTrue(verifier_rows[1]["labels"]["process_valid"])
        self.assertEqual(verifier_rows[2]["labels"]["candidate_disposition"], "REJECT")

    def test_verifier_requires_context_state_identity(self):
        result = RuleBasedVerifier().verify(
            {
                "parent_state_summary": {
                    "active_state_id": "e:s0",
                    "has_fresh_parameter_context": True,
                    "parameter_context_state_id": None,
                },
                "action": {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": "e:s0", "line_index": 0, "value": 0.2},
                },
                "tool_output": {"execution_status": "success"},
                "candidate_state_summary": {},
                "verification_metrics": {},
                "history_summary": {},
            }
        )
        self.assertFalse(result["process_valid"])

    def test_successful_commit_without_privileged_parent_label_is_inconclusive(self):
        result = RuleBasedVerifier().verify(
            {
                "parent_state_summary": {
                    "active_state_id": "e:s0",
                    "candidate_state_id": "e:s1",
                    "has_open_candidate": True,
                    "has_verified_candidate": True,
                },
                "action": {"tool": "commit_state", "arguments": {"candidate_state_id": "e:s1"}},
                "tool_output": {"execution_status": "success", "active_state_id": "e:s1"},
                "candidate_state_summary": {"active_state_id": "e:s1"},
                "verification_metrics": {},
                "history_summary": {},
            }
        )
        self.assertTrue(result["process_valid"])
        self.assertEqual(result["candidate_disposition"], "INCONCLUSIVE")

    def test_solver_dispatch_failure_does_not_make_process_illegal(self):
        result = RuleBasedVerifier().verify(
            {
                "parent_state_summary": {"active_state_id": "e:s0"},
                "action": {"tool": "run_wls", "arguments": {"state_id": "e:s0"}},
                "tool_output": {
                    "execution_status": "failure",
                    "error_code": "dispatch_error",
                    "error_detail": "SolverRuntimeError",
                },
                "candidate_state_summary": {},
                "verification_metrics": {},
                "history_summary": {},
            }
        )
        self.assertTrue(result["process_valid"])

    def test_rule_verifier_rejects_false_finalization(self):
        verifier = RuleBasedVerifier()
        result = verifier.verify(
            {
                "parent_state_summary": {"active_state_id": "e:s0", "remaining_anomaly_score": 4.0},
                "action": {"tool": "finalize_diagnosis", "arguments": {}},
                "tool_output": {"execution_status": "success"},
                "candidate_state_summary": {},
                "verification_metrics": {"remaining_anomaly_score": 4.0, "anomaly_threshold": 1.0},
                "history_summary": {},
            }
        )
        self.assertFalse(result["process_valid"])
        self.assertNotEqual(result["candidate_disposition"], "ACCEPT_FINAL")

    def test_false_accept_final_rate_is_reported(self):
        metrics = evaluate_predictions(
            [{"candidate_disposition": "REJECT"}, {"candidate_disposition": "ACCEPT_FINAL"}],
            [{"candidate_disposition": "ACCEPT_FINAL"}, {"candidate_disposition": "ACCEPT_FINAL"}],
        )
        self.assertEqual(metrics["false_accept_final_rate"], 1.0)


class AggreVaTeLiteTests(unittest.TestCase):
    def test_valid_oracle_finalization_has_no_false_final_cost(self):
        env = TransactionalPSSEEnv()
        state = env.reset(synthetic_scenario())
        state, _ = env.step(correct_measurement(state["active_state_id"]))
        state, _ = env.step({"tool": "run_wls", "arguments": {"state_id": state["candidate_state_id"]}})
        state, _ = env.step({"tool": "commit_state", "arguments": {"candidate_state_id": state["candidate_state_id"]}})
        ranking = AggreVaTeLite(env=env, oracle=ExpertPolicyOracle(), top_l=2).rank_actions(
            state,
            candidate_actions=[
                {"tool": "finalize_diagnosis", "arguments": {}},
                {"tool": "run_wls", "arguments": {"state_id": state["active_state_id"]}},
            ],
        )
        finalize = next(
            item for item in ranking["action_costs"] if item["action"]["tool"] == "finalize_diagnosis"
        )
        self.assertEqual(finalize["raw_cost_components"]["false_finalization"], 0.0)
        self.assertEqual(ranking["action_costs"][0]["action"]["tool"], "finalize_diagnosis")

    def test_branch_ranking_does_not_mutate_stateful_root_runner(self):
        class StatefulRunner:
            def __init__(self):
                self.calls = 0

            def __call__(self, state):
                self.calls += 1
                return {"execution_status": "success", "wls_objective": 1.0}

        runner = StatefulRunner()
        env = TransactionalPSSEEnv(wls_runner=runner)
        state = env.reset({"scenario_id": "stateful", "case": {}, "measurements": [1.0]})
        AggreVaTeLite(env=env, oracle=ExpertPolicyOracle()).rank_actions(
            state,
            candidate_actions=[
                {"tool": "run_wls", "arguments": {"state_id": state["active_state_id"]}}
            ],
        )
        self.assertEqual(runner.calls, 0)

    def test_branch_ranking_rejects_shared_closure_state(self):
        calls = []

        def closure_runner(state):
            calls.append(state["state_id"])
            return {"execution_status": "success", "wls_objective": 1.0}

        env = TransactionalPSSEEnv(wls_runner=closure_runner)
        state = env.reset({"scenario_id": "closure", "case": {}, "measurements": [1.0]})
        ranking = AggreVaTeLite(env=env, oracle=ExpertPolicyOracle()).rank_actions(
            state,
            candidate_actions=[
                {"tool": "run_wls", "arguments": {"state_id": state["active_state_id"]}}
            ],
        )
        self.assertEqual(calls, [])
        self.assertIn("branch_error", ranking["action_costs"][0])

    def test_environment_clone_rejects_nondeepcopyable_runner(self):
        class NonCopyableRunner:
            def __deepcopy__(self, memo):
                raise TypeError("not cloneable")

            def __call__(self, state):
                return {"execution_status": "success"}

        env = TransactionalPSSEEnv(wls_runner=NonCopyableRunner())
        env.reset({"scenario_id": "noncopyable", "case": {}, "measurements": [1.0]})
        with self.assertRaises(StateStoreError):
            env.clone()

    def test_branch_cost_separates_correct_and_healthy_corrupting_corrections(self):
        env = TransactionalPSSEEnv()
        state = env.reset(
            {
                "scenario_id": "rank-corrections",
                "case": {},
                "measurements": [9.0, 2.0],
                "clean_measurements": [1.0, 2.0],
                "true_measurement_errors": [{"index": 0}],
            }
        )
        root_hash = env.store.episode_hash()
        ranking = AggreVaTeLite(env=env, oracle=ExpertPolicyOracle(), top_l=2).rank_actions(
            state,
            candidate_actions=[
                correct_measurement(state["active_state_id"], index=0, value=1.0),
                correct_measurement(state["active_state_id"], index=1, value=99.0),
            ],
        )
        costs = ranking["action_costs"]
        self.assertEqual(costs[0]["action"]["arguments"]["measurement_updates"], {0: 1.0})
        self.assertLess(costs[0]["q_cost"], costs[1]["q_cost"])
        self.assertGreater(costs[1]["raw_cost_components"]["healthy_corruption"], 0.0)
        self.assertEqual(env.store.episode_hash(), root_hash)

    def test_top_l_branch_ranking_is_isolated_and_penalizes_false_commit(self):
        env = TransactionalPSSEEnv()
        state = env.reset(
            {
                "scenario_id": "rank",
                "case": {},
                "measurements": [1.0],
                "metadata": {"remaining_anomaly_score": 0.0, "no_material_anomaly_remaining": True},
            }
        )
        root_hash = env.store.episode_hash()
        ranking = AggreVaTeLite(env=env, oracle=ExpertPolicyOracle(), top_l=2).rank_actions(
            state,
            candidate_actions=[
                {"tool": "finalize_diagnosis", "arguments": {}},
                {"tool": "commit_state", "arguments": {"candidate_state_id": "missing"}},
            ],
        )
        self.assertEqual(ranking["action_costs"][0]["action"]["tool"], "finalize_diagnosis")
        self.assertEqual(env.store.episode_hash(), root_hash)
        self.assertFalse(env.is_terminal())
        pairs = to_pairwise_examples(ranking)
        self.assertEqual(pairs[0]["chosen"]["tool"], "finalize_diagnosis")


if __name__ == "__main__":
    unittest.main()
