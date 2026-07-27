from __future__ import annotations

import inspect
import random
import unittest

from psse_env.actions import FINALIZE_DIAGNOSIS, RUN_WLS
from psse_env.dagger.aggrevate import AggreVaTeLite, to_pairwise_examples
from psse_env.dagger.counterfactual_generator import CounterfactualGenerator
from psse_env.dagger.error_injectors import InjectedAction
from psse_env.dagger.replay_buffer import BalancedReplayBuffer
from psse_env.dagger.rollout_collector import (
    BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
    DaggerRolloutCollector,
    run_dagger,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.transactional_env import TransactionalPSSEEnv


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
