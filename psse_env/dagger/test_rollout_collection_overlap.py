from __future__ import annotations

import copy
import json
import random
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from psse_env import PolicyObservation
from psse_env.dagger.release_factories import (
    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
)
from psse_env.dagger.rollout_collector import (
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    DaggerRolloutCollector,
)


class _OverlapTestEnvironment:
    production_dataset_mode = True

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self._state = {"active_state_id": "state-0", "remaining_budget": 1}

    def reset(self, scenario):
        self.events.append("env_reset")
        self._state = {"active_state_id": "state-0", "remaining_budget": 1}
        return copy.deepcopy(self._state)

    def get_policy_observation(self, history=None):
        return PolicyObservation(
            active_state_id=str(self._state["active_state_id"]),
            remaining_budget=int(self._state["remaining_budget"]),
            history_window=copy.deepcopy(list(history or [])),
            episode_id="overlap-episode",
        )

    def get_oracle_state(self, history=None):
        return {"active_state_id": self._state["active_state_id"]}

    def assert_training_decision_evidence(self, action):
        self.events.append("evidence_audit")

    def current_state(self):
        self.events.append("env_current_state")
        return copy.deepcopy(self._state)

    def step(self, action):
        self.events.append("env_step")
        next_state = {
            "active_state_id": "state-0",
            "remaining_budget": 0,
            "terminal": True,
        }
        self._state = copy.deepcopy(next_state)
        return next_state, {
            "execution_status": "success",
            "error_code": None,
            "error_detail": None,
            "state_mutated": False,
            "active_state_id": "state-0",
            "candidate_state_id": None,
            "tool_metrics": {},
            "valid_next_actions": [],
        }

    @staticmethod
    def is_terminal(state):
        return state.get("terminal") is True


class _OverlapTestExpert:
    @staticmethod
    def label_transition(**kwargs):
        return {
            "process_valid": True,
            "error_code": None,
            "error_detail": None,
            "candidate_disposition": None,
            "progress_class": None,
            "valid_next_actions": [],
        }


class _OverlapTestCollector(DaggerRolloutCollector):
    def __init__(self, *, events: list[str], **kwargs) -> None:
        self.events = events
        super().__init__(**kwargs)

    def _select_expert_actions(
        self, *, policy_observation, oracle_state, history
    ):
        self.events.append("expert_action_selection")
        if "worker_mutation" in policy_observation.as_dict():
            raise AssertionError("policy worker mutated the main-thread observation")
        return [
            {
                "tool": "run_wls",
                "arguments": {
                    "state_id": policy_observation.active_state_id,
                },
            }
        ]


class _OrderCheckingRng:
    def __init__(self, events: list[str], policy_done: threading.Event) -> None:
        self.events = events
        self.policy_done = policy_done

    def random(self) -> float:
        if not self.policy_done.is_set():
            raise AssertionError("beta RNG advanced before the policy join barrier")
        self.events.append("beta_rng")
        return 1.0


def _overlap_test_scenario() -> dict:
    return {
        "scenario_schema_version": 1,
        "execution": {"scenario_id": "overlap-test", "case": {}},
        "audit": {"truth": {"truth_complete": True}},
        "grouping": {
            "dataset_split": "dagger_train",
            "physical_root_fingerprint": "overlap-root",
            "root_scenario_id": "overlap-test",
            "scenario_family": "measurement+parameter",
            "error_cardinality": 2,
        },
    }


class Dagger1CollectionSafetyTests(unittest.TestCase):
    @staticmethod
    def _release_threshold_report():
        return {
            "source_partition": {"enabled": True, "selected": "train"},
            "parameter_ranking_admission": {
                "contract": "distinct_line_abs_lambda_dominance_v1",
                "enforced": True,
                "threshold": BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
            },
        }


    def test_policy_audit_overlap_barrier_preserves_rng_and_env_order(self) -> None:
        events: list[str] = []
        policy_started = threading.Event()
        audit_finished = threading.Event()
        policy_done = threading.Event()

        class CoordinatedPolicy:
            def act(self, observation):
                events.append("policy_start")
                observation["worker_mutation"] = "worker-only"
                policy_started.set()
                if not audit_finished.wait(timeout=5):
                    raise RuntimeError("private audit did not overlap policy work")
                events.append("policy_finish")
                policy_done.set()
                return {
                    "tool": "run_wls",
                    "arguments": {"state_id": observation["active_state_id"]},
                }

        class CoordinatedCollector(_OverlapTestCollector):
            def _select_expert_actions(self, **kwargs):
                if not policy_started.wait(timeout=5):
                    raise AssertionError("policy was not submitted before expert work")
                return super()._select_expert_actions(**kwargs)

        def rank_one(*args, **kwargs):
            events.append("observable_rank_one_target_proof")
            return {"contract": "test_rank_one", "passed": True}

        def private_audit(**kwargs):
            events.append("private_teacher_target_audit")
            audit_finished.set()
            return {"contract": "test_private_audit", "passed": True}

        env = _OverlapTestEnvironment(events)
        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            patch(
                "psse_env.dagger.rollout_collector.observable_rank_one_target_proof",
                side_effect=rank_one,
            ),
            patch(
                "psse_env.dagger.rollout_collector.offline_teacher_target_audit",
                side_effect=private_audit,
            ),
        ):
            rows = CoordinatedCollector(
                events=events,
                env=env,
                policy=CoordinatedPolicy(),
                expert_oracle=_OverlapTestExpert(),
                rng=_OrderCheckingRng(events, policy_done),
                supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
                forbidden_physical_roots={"held-out-root"},
                policy_executor=executor,
            ).collect_iteration(
                scenarios=[_overlap_test_scenario()],
                iteration=1,
                beta=0.25,
                max_steps=1,
                collection_role="training",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(events.count("policy_start"), 1)
        self.assertLess(
            events.index("policy_start"),
            events.index("expert_action_selection"),
        )
        self.assertLess(
            events.index("expert_action_selection"),
            events.index("observable_rank_one_target_proof"),
        )
        self.assertLess(
            events.index("observable_rank_one_target_proof"),
            events.index("private_teacher_target_audit"),
        )
        self.assertLess(
            events.index("private_teacher_target_audit"),
            events.index("policy_finish"),
        )
        self.assertLess(events.index("policy_finish"), events.index("beta_rng"))
        self.assertLess(events.index("beta_rng"), events.index("env_current_state"))
        self.assertLess(events.index("env_current_state"), events.index("env_step"))
        self.assertNotIn("worker_mutation", rows[0]["policy_observation"])

    def test_overlapped_collection_is_sequentially_equivalent(self) -> None:
        class Policy:
            @staticmethod
            def act(observation):
                return {
                    "tool": "run_wls",
                    "arguments": {"state_id": observation["active_state_id"]},
                }

        def collect(executor=None):
            events: list[str] = []
            with (
                patch(
                    "psse_env.dagger.rollout_collector.observable_rank_one_target_proof",
                    return_value={"contract": "test_rank_one", "passed": True},
                ),
                patch(
                    "psse_env.dagger.rollout_collector.offline_teacher_target_audit",
                    return_value={
                        "contract": "test_private_audit",
                        "passed": True,
                    },
                ),
            ):
                return _OverlapTestCollector(
                    events=events,
                    env=_OverlapTestEnvironment(events),
                    policy=Policy(),
                    expert_oracle=_OverlapTestExpert(),
                    rng=random.Random(913),
                    supervision_policy=(
                        DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                    ),
                    forbidden_physical_roots={"held-out-root"},
                    policy_executor=executor,
                ).collect_iteration(
                    scenarios=[_overlap_test_scenario()],
                    iteration=1,
                    beta=0.25,
                    max_steps=1,
                    collection_role="training",
                )

        sequential = collect()
        with ThreadPoolExecutor(max_workers=1) as executor:
            overlapped = collect(executor)
        self.assertEqual(
            json.dumps(overlapped, sort_keys=True),
            json.dumps(sequential, sort_keys=True),
        )

    def test_overlapped_policy_exception_matches_sequential_collection(self) -> None:
        class BrokenPolicy:
            @staticmethod
            def act(observation):
                raise RuntimeError("expected policy failure")

        def collect(executor=None):
            events: list[str] = []
            with (
                patch(
                    "psse_env.dagger.rollout_collector.observable_rank_one_target_proof",
                    return_value={"contract": "test_rank_one", "passed": True},
                ),
                patch(
                    "psse_env.dagger.rollout_collector.offline_teacher_target_audit",
                    return_value={
                        "contract": "test_private_audit",
                        "passed": True,
                    },
                ),
            ):
                return _OverlapTestCollector(
                    events=events,
                    env=_OverlapTestEnvironment(events),
                    policy=BrokenPolicy(),
                    expert_oracle=_OverlapTestExpert(),
                    rng=random.Random(47),
                    supervision_policy=(
                        DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
                    ),
                    forbidden_physical_roots={"held-out-root"},
                    policy_executor=executor,
                ).collect_iteration(
                    scenarios=[_overlap_test_scenario()],
                    iteration=1,
                    beta=0.25,
                    max_steps=1,
                    collection_role="training",
                )

        sequential = collect()
        with ThreadPoolExecutor(max_workers=1) as executor:
            overlapped = collect(executor)
        self.assertEqual(overlapped, sequential)
        self.assertEqual(
            overlapped[0]["model_action"]["arguments"]["error_code"],
            "policy_exception",
        )


    @staticmethod
    def _scheduled_scenarios():
        specifications = (
            ("multi-primary", "multi_measurement", "primary", 0),
            ("mixed-primary", "measurement+parameter", "primary", 0),
            ("multi-reserve", "multi_measurement", "reserve", 1),
            ("mixed-reserve", "measurement+parameter", "reserve", 2),
            ("parameter-reserve", "parameter", "reserve", 3),
        )
        return [
            {
                "execution": {"scenario_id": name},
                "audit": {"truth": {"truth_complete": True}},
                "grouping": {
                    "physical_root_fingerprint": name,
                    "scenario_family": family,
                    "collection_cohort": cohort,
                    "collection_subcohort": (
                        "primary" if cohort == "primary" else "base_reserve"
                    ),
                    "collection_priority": priority,
                    "collection_order": order,
                    "split": "dagger_train",
                },
            }
            for order, (name, family, cohort, priority) in enumerate(
                specifications
            )
        ]

    @staticmethod
    def _strict_coverage_rows(extra_rows: int = 0):
        rows = []

        def add(
            prefix,
            count,
            *,
            stratum,
            family="measurement+parameter",
            cardinality=2,
            observation=None,
            preferred_action=None,
            parameter_scans_available=None,
        ):
            for index in range(count):
                rows.append(
                    {
                        "example_id": f"{prefix}-{index}",
                        "physical_root_fingerprint": f"{prefix}-root-{index}",
                        "production_label_eligible": True,
                        "recovery_stratum": stratum,
                        "scenario_family": family,
                        "error_cardinality": cardinality,
                        "parameter_scans_available": parameter_scans_available,
                        "policy_observation": copy.deepcopy(observation or {}),
                        "preferred_action": copy.deepcopy(preferred_action),
                    }
                )

        for cardinality in (2, 4, 5):
            add(
                f"multi-{cardinality}",
                5,
                stratum="multi_measurement_safe_handoff",
                family="multi_measurement",
                cardinality=cardinality,
                parameter_scans_available=False,
            )
        add(
            "route-actionable",
            5,
            stratum="premature_commit_recovery",
            observation={
                "fresh_context_evidence": {
                    "parameter": {
                        "route_status": "actionable",
                        "parameter_ranking_dominance_ratio": 1.1,
                    }
                }
            },
        )
        add(
            "route-negative",
            5,
            stratum="premature_escalation_recovery",
            observation={
                "fresh_context_evidence": {
                    "parameter": {"route_status": "complete_negative"}
                }
            },
        )
        add(
            "route-unavailable",
            5,
            stratum="unsupported_correction_recovery",
            observation={
                "fresh_context_evidence": {
                    "parameter": {
                        "route_status": "unavailable_or_inconclusive"
                    }
                }
            },
        )
        for first, second in (
            ("measurement", "parameter"),
            ("parameter", "measurement"),
        ):
            add(
                f"sequence-{first}",
                5,
                stratum="sequential_measurement_parameter_recovery",
                observation={
                    "history_window": [
                        {"action": {"tool": f"correct_{first}"}}
                    ]
                },
                preferred_action={
                    "tool": f"correct_{second}",
                    "arguments": {},
                },
            )
        add(
            "partial",
            5,
            stratum="post_failure_no_candidate",
            observation={
                "accepted_corrections": [{"target": "measurement:1"}],
                "no_material_anomaly_remaining": False,
            },
        )
        add(
            "unsupported-extra",
            5,
            stratum="unsupported_correction_recovery",
        )
        add("post-extra", 5, stratum="post_failure_no_candidate")
        for index in range(extra_rows):
            rows.append(
                {
                    "example_id": f"extra-{index}",
                    "physical_root_fingerprint": f"extra-root-{index}",
                    "production_label_eligible": True,
                    "recovery_stratum": "multi_measurement_safe_handoff",
                    "scenario_family": "measurement+parameter",
                    "error_cardinality": 2,
                    "policy_observation": {},
                }
            )
        return rows


if __name__ == "__main__":
    unittest.main()
