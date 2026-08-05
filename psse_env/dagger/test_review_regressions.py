from __future__ import annotations

import copy
import inspect
import json
import random
import unittest

from psse_env.actions import (
    FINALIZE_DIAGNOSIS,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_WLS,
)
from psse_env.dagger.aggrevate import AggreVaTeLite, to_pairwise_examples
from psse_env.dagger.counterfactual_generator import CounterfactualGenerator
from psse_env.dagger.error_injectors import InjectedAction
from psse_env.dagger.replay_buffer import BalancedReplayBuffer
from psse_env.dagger.rollout_collector import (
    BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    DaggerRolloutCollector,
    audit_dagger1_recovery_labels,
    classify_dagger1_recovery_stratum,
    observable_rank_one_target_proof,
    run_dagger,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.transactional_env import TransactionalPSSEEnv
from psse_env.state_store import OracleState, PolicyObservation


def _scenario(**updates):
    scenario = {
        "scenario_id": "review-regression",
        "case": {},
        "measurements": [1.0],
    }
    scenario.update(updates)
    return scenario


class _TwoActionOracle:
    def next_actions(self, state, history=None):
        del history
        return [
            {"tool": RUN_WLS, "arguments": {"state_id": state.get("active_state_id")}},
            {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
        ]

    def label_transition(self, **kwargs):
        return {
            "process_valid": kwargs["tool_output"].get("execution_status") == "success",
            "valid_next_actions": [],
        }


class _FinalizePolicy:
    def act(self, observation):
        del observation
        return {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}


class _NoChoiceRng:
    @staticmethod
    def random():
        return 0.0

    @staticmethod
    def choice(items):
        del items
        raise AssertionError("ordinary expert-controlled DAgger must not sample proposals")


class _LearnerRecoveryPolicy:
    def __init__(self):
        self.seen_observations = []

    def act(self, observation):
        self.seen_observations.append(copy.deepcopy(observation))
        candidate = observation.get("candidate_state_id")
        if candidate:
            return {
                "tool": "rollback_state",
                "arguments": {"candidate_state_id": candidate},
            }
        return {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": observation["active_state_id"],
                "measurement_updates": {"0": 9.0},
            },
        }


class _LearnerRecoveryOracle:
    def __init__(self):
        self.seen_truth = []

    def next_actions(self, state, history=None):
        del history
        self.seen_truth.append(copy.deepcopy(state.truth_dict()))
        disposition = getattr(state, "candidate_disposition", None)
        if disposition == "REJECT":
            candidate = state.policy_observation.candidate_state_id
            return [
                {
                    "tool": "rollback_state",
                    "arguments": {"candidate_state_id": candidate},
                }
            ]
        observation = state.policy_observation
        return [
            {
                "tool": RUN_WLS,
                "arguments": {"state_id": observation.active_state_id},
            }
        ]

    def label_transition(self, **kwargs):
        return {
            "process_valid": True,
            "execution_status": kwargs["tool_output"]["execution_status"],
            "valid_next_actions": [],
        }


class _LearnerRecoveryEnv:
    production_dataset_mode = True

    def __init__(self):
        self.stage = 0
        self.terminal = False
        self.last_reset_scenario = None

    def reset(self, scenario):
        self.last_reset_scenario = copy.deepcopy(scenario)
        self.stage = 0
        self.terminal = False
        return self.current_state()

    def current_state(self):
        return {
            "active_state_id": "active",
            "candidate_state_id": "candidate" if self.stage == 1 else None,
            "remaining_budget": 4 - self.stage,
        }

    def get_policy_observation(self, history):
        return PolicyObservation(
            active_state_id="active",
            candidate_state_id="candidate" if self.stage == 1 else None,
            candidate_lifecycle=("VERIFIED_REJECT" if self.stage == 1 else "NO_CANDIDATE"),
            has_open_candidate=self.stage == 1,
            has_verified_candidate=self.stage == 1,
            history_window=list(history),
            remaining_budget=4 - self.stage,
        )

    def get_oracle_state(self, history):
        observation = self.get_policy_observation(history)
        reset = self.last_reset_scenario or {}
        return OracleState(
            policy_observation=observation,
            clean_measurements=copy.deepcopy(reset.get("clean_measurements")),
            true_measurement_errors=copy.deepcopy(
                list(reset.get("true_measurement_errors") or [])
            ),
            candidate_disposition="REJECT" if self.stage == 1 else None,
            candidate_lifecycle=observation.candidate_lifecycle,
            candidate_assessment=(
                {"disposition": "REJECT", "rationale_codes": ["wrong_target"]}
                if self.stage == 1
                else {}
            ),
        )

    def assert_training_decision_evidence(self, action):
        if self.stage == 1 and action.get("tool") != "rollback_state":
            raise ValueError("rejected learner candidate must roll back")

    def step(self, action):
        del action
        if self.stage == 0:
            self.stage = 1
        else:
            self.stage = 2
            self.terminal = True
        return self.current_state(), {
            "execution_status": "success",
            "error_code": None,
            "state_mutated": True,
            "tool_metrics": {},
        }

    def is_terminal(self, state=None):
        del state
        return self.terminal


class DaggerExecutionRegressionTests(unittest.TestCase):
    def test_expert_execution_matches_stored_preferred_action(self):
        rows = DaggerRolloutCollector(
            env=TransactionalPSSEEnv(),
            policy=_FinalizePolicy(),
            expert_oracle=_TwoActionOracle(),
            rng=_NoChoiceRng(),
        ).collect_iteration(
            scenarios=[_scenario(physical_root_fingerprint="physical-root")],
            iteration=0,
            beta=1.0,
            max_steps=1,
        )
        self.assertEqual(rows[0]["executed_by"], "expert")
        self.assertEqual(rows[0]["executed_action"], rows[0]["preferred_action"])
        self.assertEqual(rows[0]["executed_action"]["tool"], RUN_WLS)
        self.assertEqual(rows[0]["physical_root_fingerprint"], "physical-root")

    def test_bc0_observable_sequential_policy_exposes_only_current_action(self):
        env = TransactionalPSSEEnv()
        env.production_dataset_mode = True
        rows = DaggerRolloutCollector(
            env=env,
            policy=_FinalizePolicy(),
            expert_oracle=_TwoActionOracle(),
            rng=_NoChoiceRng(),
            supervision_policy=BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
        ).collect_iteration(
            scenarios=[_scenario(physical_root_fingerprint="physical-root")],
            iteration=0,
            beta=1.0,
            max_steps=1,
        )

        self.assertEqual(
            rows[0]["valid_next_actions"],
            [rows[0]["preferred_action"]],
        )
        self.assertEqual(
            [action["tool"] for action in rows[0]["deferred_expert_actions"]],
            [FINALIZE_DIAGNOSIS],
        )
        self.assertEqual(
            rows[0]["labels"]["supervision_policy"],
            BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
        )
        self.assertEqual(rows[0]["labels"]["deferred_expert_action_count"], 1)

    def test_bc0_observable_sequential_policy_fails_closed_outside_round0(self):
        with self.assertRaisesRegex(ValueError, "production_dataset_mode"):
            DaggerRolloutCollector(
                env=TransactionalPSSEEnv(),
                policy=_FinalizePolicy(),
                expert_oracle=_TwoActionOracle(),
                supervision_policy=BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
            )

        env = TransactionalPSSEEnv()
        env.production_dataset_mode = True
        collector = DaggerRolloutCollector(
            env=env,
            policy=_FinalizePolicy(),
            expert_oracle=_TwoActionOracle(),
            supervision_policy=BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
        )
        for iteration, beta in ((1, 1.0), (0, 0.5)):
            with self.subTest(iteration=iteration, beta=beta):
                with self.assertRaisesRegex(ValueError, "iteration=0, beta=1.0"):
                    collector.collect_iteration(
                        scenarios=[_scenario()],
                        iteration=iteration,
                        beta=beta,
                        max_steps=1,
                    )

    def test_dagger1_marks_only_rank_one_learner_recovery_state_eligible(self):
        rows = DaggerRolloutCollector(
            env=_LearnerRecoveryEnv(),
            policy=_LearnerRecoveryPolicy(),
            expert_oracle=_LearnerRecoveryOracle(),
            rng=random.Random(0),
            supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
            forbidden_physical_roots={"held-out-root"},
        ).collect_iteration(
            scenarios=[
                _scenario(
                    physical_root_fingerprint="dagger-train-root",
                    dataset_split="dagger_train",
                )
            ],
            iteration=1,
            beta=0.25,
            max_steps=2,
            collection_role="training",
        )

        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["production_label_eligible"])
        self.assertEqual(
            rows[0]["production_label_ineligibility_reason"],
            "not_learner_visited_state",
        )
        recovery = rows[1]
        self.assertEqual(recovery["state_origin"], "learner_policy")
        self.assertEqual(recovery["state_class"], "rejected_candidate_recovery")
        self.assertEqual(recovery["preferred_action"]["tool"], "rollback_state")
        self.assertTrue(recovery["production_label_eligible"])
        self.assertEqual(recovery["dataset_source"], "dagger_rollout")
        self.assertEqual(
            recovery["recovery_label_contract"],
            "observable_rank_one_learner_state_v1",
        )
        self.assertEqual(
            recovery["recovery_stratum"], "rejected_candidate_rollback"
        )
        self.assertTrue(audit_dagger1_recovery_labels(rows)["passed"])

        diagnostic_rows = DaggerRolloutCollector(
            env=_LearnerRecoveryEnv(),
            policy=_LearnerRecoveryPolicy(),
            expert_oracle=_LearnerRecoveryOracle(),
            rng=random.Random(0),
            supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
            forbidden_physical_roots={"held-out-root"},
        ).collect_iteration(
            scenarios=[
                _scenario(
                    physical_root_fingerprint="dagger-diagnostic-root",
                    dataset_split="dagger_train",
                )
            ],
            iteration=1,
            beta=0.0,
            max_steps=2,
            collection_role="diagnostic",
        )
        self.assertFalse(
            any(row["production_label_eligible"] for row in diagnostic_rows)
        )
        self.assertEqual(
            {
                row["production_label_ineligibility_reason"]
                for row in diagnostic_rows
            },
            {"diagnostic_beta_zero_not_training_eligible"},
        )
        mutated = copy.deepcopy(diagnostic_rows)
        mutated[-1]["production_label_eligible"] = True
        mutated_audit = audit_dagger1_recovery_labels(mutated)
        self.assertFalse(mutated_audit["passed"])
        self.assertIn(
            "collection_role_not_training",
            mutated_audit["eligibility_violations"][0]["reasons"],
        )
        self.assertIn(
            "training_beta_contract_not_verified",
            mutated_audit["eligibility_violations"][0]["reasons"],
        )

    def test_dagger1_envelope_truth_is_oracle_private_not_policy_visible(self):
        env = _LearnerRecoveryEnv()
        policy = _LearnerRecoveryPolicy()
        oracle = _LearnerRecoveryOracle()
        envelope = {
            "scenario_schema_version": 1,
            "execution": {
                "scenario_id": "private-truth-envelope",
                "case": {},
                "measurements": [1.0],
            },
            "audit": {
                "truth": {
                    "truth_complete": True,
                    "clean_measurements": [1.0],
                    "true_measurement_errors": [{"index": 0}],
                },
                "release_audit": {"offline_only": True},
            },
            "grouping": {
                "root_scenario_id": "private-truth-envelope",
                "physical_root_fingerprint": "new-envelope-root",
                "scenario_family": "measurement",
                "error_cardinality": 1,
                "case_id": "case14",
                "split": "dagger_train",
                "source_tier": "generated",
            },
        }
        rows = DaggerRolloutCollector(
            env=env,
            policy=policy,
            expert_oracle=oracle,
            rng=random.Random(0),
            supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
            forbidden_physical_roots={"held-out-root", "d0-root"},
        ).collect_iteration(
            scenarios=[envelope],
            iteration=1,
            beta=0.25,
            max_steps=2,
            collection_role="training",
        )
        self.assertEqual(
            env.last_reset_scenario,
            {
                **envelope["execution"],
                "clean_measurements": [1.0],
                "true_measurement_errors": [{"index": 0}],
            },
        )
        self.assertNotIn("audit", env.last_reset_scenario)
        self.assertEqual(
            oracle.seen_truth[0]["true_measurement_errors"], [{"index": 0}]
        )
        policy_payload = json.dumps(policy.seen_observations, sort_keys=True)
        exported_payload = json.dumps(
            [row["policy_observation"] for row in rows], sort_keys=True
        )
        for private_key in ("true_measurement_errors", "clean_measurements"):
            self.assertNotIn(private_key, policy_payload)
            self.assertNotIn(private_key, exported_payload)

    def test_dagger1_recovery_strata_use_only_observable_state(self):
        target = {"tool": RUN_WLS, "arguments": {"state_id": "active"}}
        cases = [
            (
                {
                    "last_tool": "correct_parameters",
                    "last_tool_status": "failure",
                    "last_tool_output": {
                        "execution_status": "failure",
                        "error_code": "correction_route_not_actionable",
                    },
                },
                target,
                "parameter",
                1,
                "clean_successful",
                "unsupported_correction_recovery",
            ),
            (
                {
                    "last_tool": "run_hse_from_path",
                    "last_tool_status": "failure",
                    "last_tool_output": {
                        "execution_status": "failure",
                        "error_code": "solver_failure",
                    },
                    "has_open_candidate": False,
                },
                target,
                "multi_measurement",
                2,
                "invalid_precondition_recovery",
                "post_failure_no_candidate",
            ),
            (
                {
                    "last_tool": "commit_state",
                    "last_tool_status": "failure",
                    "last_tool_output": {
                        "execution_status": "failure",
                        "error_code": "candidate_lifecycle_violation",
                    },
                },
                target,
                "measurement+parameter",
                2,
                "invalid_precondition_recovery",
                "premature_commit_recovery",
            ),
            (
                {
                    "last_tool": "ask_for_more_evidence",
                    "last_tool_status": "failure",
                    "last_tool_output": {
                        "execution_status": "failure",
                        "error_code": "operator_escalation_precondition_not_met",
                    },
                },
                target,
                "multi_measurement",
                4,
                "invalid_precondition_recovery",
                "premature_escalation_recovery",
            ),
            (
                {},
                {
                    "tool": "ask_for_more_evidence",
                    "arguments": {
                        "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                    },
                },
                "multi_measurement",
                5,
                "terminal_operator_escalation",
                "multi_measurement_safe_handoff",
            ),
            (
                {
                    "history_window": [
                        {
                            "action": {
                                "tool": "get_measurement_context",
                                "arguments": {"state_id": "active"},
                            }
                        }
                    ]
                },
                {
                    "tool": "get_parameter_context",
                    "arguments": {"state_id": "active"},
                },
                "measurement+parameter",
                2,
                "clean_successful",
                "sequential_measurement_parameter_recovery",
            ),
        ]
        for observation, preferred, family, cardinality, state_class, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    classify_dagger1_recovery_stratum(
                        observation,
                        preferred_action=preferred,
                        state_class=state_class,
                        scenario_family=family,
                        error_cardinality=cardinality,
                    ),
                    expected,
                )
        for transition_derived_class in (
            "invalid_precondition_recovery",
            "rejected_candidate_recovery",
            "loop_repetition",
        ):
            with self.subTest(transition_derived_class=transition_derived_class):
                self.assertIsNone(
                    classify_dagger1_recovery_stratum(
                        {},
                        preferred_action=target,
                        state_class=transition_derived_class,
                        scenario_family="measurement",
                        error_cardinality=1,
                    )
                )

    def test_dagger1_parameter_rank_one_proof_accepts_strict_rank_not_ties(self):
        actions = [
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": "active", "line_index": 11},
            },
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": "active", "line_index": 18},
            },
        ]

        def observation(
            top: float, runner: float, *, bundled: bool = False
        ) -> dict:
            evidence = {
                "context_tool": "get_parameter_context",
                "context_binding": (
                    "branch_route_screening.parameter"
                    if bundled
                    else "direct_context"
                ),
                "evidence_source": "deployment_context:wls_lagrange",
                "route_status": "actionable",
                "state_id": "active",
                "state_hash": "state-hash",
                "parameter_ranking_contract": (
                    "distinct_line_abs_lambda_dominance_v1"
                ),
                "parameter_ranking_distinct_lines": [
                    {"line_index1": 11, "abs_lambda_score": top},
                    {"line_index1": 18, "abs_lambda_score": runner},
                ],
                "parameter_ranking_top_abs_lambda": top,
                "parameter_ranking_runner_up_abs_lambda": runner,
                "parameter_ranking_dominance_ratio": top / runner,
                "parameter_ranking_dominance_threshold": 1.0,
                "parameter_ranking_dominant": top > runner,
                "supported_corrections": copy.deepcopy(actions),
            }
            if bundled:
                evidence["bundled_by_context_tool"] = "get_measurement_context"
            return {
                "active_state_id": "active",
                "has_fresh_parameter_context": True,
                "parameter_context_state_id": "active",
                "fresh_context_evidence": {
                    "parameter": evidence
                },
            }

        for top, bundled in (
            (2.0, False),
            (1.000001, False),
            (1.000001, True),
        ):
            with self.subTest(top=top, bundled=bundled):
                proof = observable_rank_one_target_proof(
                    observation(top, 1.0, bundled=bundled),
                    preferred_action=actions[0],
                    expert_actions=actions,
                )
                self.assertTrue(proof["passed"])
                self.assertEqual(
                    proof["basis"], "strict_observable_parameter_ranking"
                )

        tied = observable_rank_one_target_proof(
            observation(1.0, 1.0),
            preferred_action=actions[0],
            expert_actions=actions,
        )
        self.assertFalse(tied["passed"])
        mismatched = observable_rank_one_target_proof(
            observation(2.0, 1.0),
            preferred_action=actions[1],
            expert_actions=[actions[1], actions[0]],
        )
        self.assertFalse(mismatched["passed"])
    def test_dagger1_rejects_nontraining_splits_and_round0_parameters(self):
        with self.assertRaisesRegex(ValueError, "forbidden_physical_roots"):
            DaggerRolloutCollector(
                env=_LearnerRecoveryEnv(),
                policy=_LearnerRecoveryPolicy(),
                expert_oracle=_LearnerRecoveryOracle(),
                supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
            )
        collector = DaggerRolloutCollector(
            env=_LearnerRecoveryEnv(),
            policy=_LearnerRecoveryPolicy(),
            expert_oracle=_LearnerRecoveryOracle(),
            supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
            forbidden_physical_roots={"frozen-root"},
        )
        with self.assertRaisesRegex(ValueError, "iteration>=1"):
            collector.collect_iteration(
                scenarios=[], iteration=0, beta=1.0, max_steps=1
            )
        with self.assertRaisesRegex(ValueError, "explicit collection_role"):
            collector.collect_iteration(
                scenarios=[], iteration=1, beta=0.25, max_steps=1
            )
        with self.assertRaisesRegex(ValueError, "train/dagger_train"):
            collector.collect_iteration(
                scenarios=[
                    _scenario(
                        physical_root_fingerprint="frozen-root",
                        dataset_split="release_eval",
                    )
                ],
                iteration=1,
                beta=0.0,
                max_steps=1,
                collection_role="diagnostic",
            )
        with self.assertRaisesRegex(ValueError, "protected D0/evaluation holdout"):
            collector.collect_iteration(
                scenarios=[
                    _scenario(
                        physical_root_fingerprint="frozen-root",
                        dataset_split="train",
                    )
                ],
                iteration=1,
                beta=0.0,
                max_steps=1,
                collection_role="diagnostic",
            )

    def test_default_multi_error_horizon_is_24(self):
        self.assertEqual(inspect.signature(run_dagger).parameters["max_steps"].default, 24)
        self.assertIs(
            inspect.signature(run_dagger).parameters[
                "replay_require_late_iteration_model_quota"
            ].default,
            True,
        )
        self.assertEqual(TransactionalPSSEEnv().max_steps, 24)


class CounterfactualSafetyRegressionTests(unittest.TestCase):
    def test_truth_helper_covers_every_fault_in_a_family(self):
        env = TransactionalPSSEEnv()
        env.reset(
            _scenario(
                measurements=[9.0, 8.0, 3.0],
                clean_measurements=[1.0, 2.0, 3.0],
                true_measurement_errors=[{"index": 0}, {"index": 1}],
            )
        )
        actions = CounterfactualGenerator._truth_correction_actions(env.get_oracle_state())
        measurement_actions = [
            action for action in actions if action["tool"] == "correct_measurements"
        ]
        self.assertEqual(len(measurement_actions), 2)
        self.assertEqual(
            {
                next(iter(action["arguments"]["measurement_updates"]))
                for action in measurement_actions
            },
            {0, 1},
        )

    def test_counterfactual_rows_are_explicitly_ineligible_auxiliary_data(self):
        env = TransactionalPSSEEnv()
        env.reset(_scenario())
        row = CounterfactualGenerator(
            env=env, expert_oracle=ExpertPolicyOracle()
        ).generate_from_current(
            [
                InjectedAction(
                    "premature_finalization",
                    {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
                )
            ],
            root_scenario_id="root",
            physical_root_fingerprint="fingerprint",
        )[0]
        self.assertEqual(row["dataset_mode"], "synthetic_counterfactual")
        self.assertEqual(row["dataset_source"], "synthetic_counterfactual")
        self.assertIs(row["production_label_eligible"], False)
        self.assertIs(row["labels"]["production_label_eligible"], False)
        self.assertEqual(row["physical_root_fingerprint"], "fingerprint")


class AggreVaTePreferenceRegressionTests(unittest.TestCase):
    @staticmethod
    def _action(name):
        return {"tool": name, "arguments": {}}

    def test_near_optimal_action_is_never_a_negative_pair(self):
        best = self._action("best")
        equivalent = self._action("equivalent")
        worse = self._action("worse")
        ranking = {
            "state": {"active_state_id": "active"},
            "action_costs": [
                {"action": best, "q_cost": 1.0, "near_optimal": True},
                {"action": equivalent, "q_cost": 1.05, "near_optimal": True},
                {"action": worse, "q_cost": 2.0, "near_optimal": False},
            ],
            "near_optimal_actions": [best, equivalent],
            "near_optimal_cost_tolerance": 0.1,
        }
        pairs = to_pairwise_examples(ranking)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"], best)
        self.assertEqual(pairs[0]["rejected"], worse)

    def test_all_near_optimal_state_emits_no_contradictory_pair(self):
        first = self._action("first")
        second = self._action("second")
        self.assertEqual(
            to_pairwise_examples(
                {
                    "state": {},
                    "action_costs": [
                        {"action": first, "q_cost": 1.0},
                        {"action": second, "q_cost": 1.0},
                    ],
                    "near_optimal_actions": [first, second],
                    "near_optimal_cost_tolerance": 0.0,
                }
            ),
            [],
        )

    def test_ranking_records_membership_margin_and_one_step_scope(self):
        env = TransactionalPSSEEnv()
        state = env.reset(_scenario())
        ranking = AggreVaTeLite(
            env=env,
            oracle=ExpertPolicyOracle(),
            top_l=2,
            near_optimal_tolerance=1_000.0,
        ).rank_actions(
            state,
            candidate_actions=[
                {"tool": RUN_WLS, "arguments": {"state_id": state["active_state_id"]}},
                {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            ],
        )
        self.assertTrue(all(item["near_optimal"] for item in ranking["action_costs"]))
        self.assertTrue(
            all(item["cost_margin_from_best"] >= 0 for item in ranking["action_costs"])
        )
        self.assertFalse(ranking["cost_evaluation"]["full_expert_rollout"])
        self.assertEqual(to_pairwise_examples(ranking), [])


class ReplaySafetyRegressionTests(unittest.TestCase):
    def test_unknown_state_class_fails_closed_or_gets_explicit_fallback_mass(self):
        rows = [
            {"id": "known", "state_class": "clean_successful"},
            {"id": "new", "state_class": "new_recovery_class"},
        ]
        with self.assertRaisesRegex(ValueError, "Unknown replay state classes"):
            BalancedReplayBuffer(rows).sample(1, rng=random.Random(0))
        sampled = BalancedReplayBuffer(
            rows,
            unknown_class_policy="fallback",
            unknown_class_weight=1.0,
        ).sample(2, rng=random.Random(0))
        self.assertEqual({row["id"] for row in sampled}, {"known", "new"})

    def test_sampler_caps_roots_avoids_duplicates_and_reports_before_after(self):
        rows = [
            {
                "id": f"a-{index}",
                "state_class": "clean_successful",
                "physical_root_fingerprint": "a",
            }
            for index in range(8)
        ] + [
            {
                "id": f"b-{index}",
                "state_class": "clean_successful",
                "physical_root_fingerprint": "b",
            }
            for index in range(2)
        ]
        buffer = BalancedReplayBuffer(rows)
        sampled = buffer.sample(4, rng=random.Random(0))
        self.assertEqual(len({row["id"] for row in sampled}), 4)
        self.assertEqual(
            {root: sum(row["physical_root_fingerprint"] == root for row in sampled)
             for root in ("a", "b")},
            {"a": 2, "b": 2},
        )
        report = buffer.sample_report()
        self.assertEqual(report["max_rows_per_root"], 2)
        self.assertEqual(report["duplicate_occurrences"], 0)
        self.assertEqual(report["before"]["rows"], 10)
        self.assertEqual(report["after"]["rows"], 4)

    def test_latest_iteration_model_quota_is_explicit(self):
        rows = []
        for index in range(4):
            rows.append(
                {
                    "id": f"early-{index}",
                    "state_class": "clean_successful",
                    "iteration": 0,
                    "executed_by": "model",
                    "state_visited_by": "model",
                    "root_scenario_id": f"early-{index}",
                }
            )
        for index, executed_by in enumerate(("model", "model", "expert", "expert")):
            rows.append(
                {
                    "id": f"late-{index}",
                    "state_class": "clean_successful",
                    "iteration": 3,
                    "executed_by": executed_by,
                    "state_visited_by": executed_by,
                    "root_scenario_id": f"late-{index}",
                }
            )
        buffer = BalancedReplayBuffer(rows, late_iteration_model_fraction=0.5)
        sampled = buffer.sample(4, rng=random.Random(0))
        late_model = [
            row
            for row in sampled
            if row["iteration"] == 3 and row["executed_by"] == "model"
        ]
        self.assertEqual(len(late_model), 2)
        self.assertEqual(buffer.sample_report()["late_iteration_model_selected"], 2)

    def test_duplicate_count_is_bounded_and_class_shortfalls_are_reported(self):
        duplicate_buffer = BalancedReplayBuffer(
            [
                {
                    "id": "one",
                    "state_class": "clean_successful",
                    "root_scenario_id": "only-root",
                },
                {
                    "id": "two",
                    "state_class": "clean_successful",
                    "root_scenario_id": "only-root",
                },
            ]
        )
        sampled = duplicate_buffer.sample(4, rng=random.Random(0))
        self.assertLessEqual(max(sampled.count(row) for row in sampled), 2)
        self.assertEqual(duplicate_buffer.sample_report()["duplicate_occurrences"], 2)

        imbalanced = [
            {
                "id": f"clean-{index}",
                "state_class": "clean_successful",
                "root_scenario_id": f"clean-root-{index}",
            }
            for index in range(5)
        ] + [
            {
                "id": "rare",
                "state_class": "rejected_candidate_recovery",
                "root_scenario_id": "rare-root",
            }
        ]
        report_buffer = BalancedReplayBuffer(imbalanced)
        report_buffer.sample(6, rng=random.Random(0))
        report = report_buffer.sample_report()
        self.assertGreater(
            report["state_class_target_shortfalls"]["rejected_candidate_recovery"],
            0,
        )

    def test_late_iteration_quota_can_fail_closed(self):
        buffer = BalancedReplayBuffer(
            [
                {
                    "id": "late-expert-only",
                    "state_class": "clean_successful",
                    "iteration": 2,
                    "executed_by": "expert",
                    "state_visited_by": "expert",
                    "root_scenario_id": "root",
                }
            ],
            require_late_iteration_model_quota=True,
        )
        with self.assertRaisesRegex(ValueError, "learner-state replay quota"):
            buffer.sample(1, rng=random.Random(0))
        self.assertFalse(buffer.sample_report()["late_iteration_model_quota_met"])


class SnapshotSafetyRegressionTests(unittest.TestCase):
    class _ShardedPolicy:
        hf_device_map = {"": 0}

    class _PolicyWrapper:
        def __init__(self, policy):
            self.policy = policy

    def test_sharded_policy_requires_explicit_snapshot_hook(self):
        with self.assertRaisesRegex(TypeError, "requires snapshot_policy_fn"):
            run_dagger(
                policy=self._ShardedPolicy(),
                expert_oracle=object(),
                env=object(),
                scenarios_by_iteration=[{}],
                num_iterations=0,
                evaluate_fn=lambda policy, env, oracle: 1.0,
            )

    def test_explicit_snapshot_hook_supports_sharded_policy(self):
        policy = self._ShardedPolicy()
        selected, rows = run_dagger(
            policy=policy,
            expert_oracle=object(),
            env=object(),
            scenarios_by_iteration=[{}],
            num_iterations=0,
            evaluate_fn=lambda current, env, oracle: 1.0,
            snapshot_policy_fn=lambda current: {"snapshot_of": current},
        )
        self.assertIs(selected["snapshot_of"], policy)
        self.assertEqual(rows, [])

    def test_wrapper_around_sharded_policy_also_fails_closed(self):
        with self.assertRaisesRegex(TypeError, "requires snapshot_policy_fn"):
            run_dagger(
                policy=self._PolicyWrapper(self._ShardedPolicy()),
                expert_oracle=object(),
                env=object(),
                scenarios_by_iteration=[{}],
                num_iterations=0,
                evaluate_fn=lambda policy, env, oracle: 1.0,
            )


if __name__ == "__main__":
    unittest.main()
