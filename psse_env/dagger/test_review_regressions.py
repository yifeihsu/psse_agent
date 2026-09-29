from __future__ import annotations

import copy
import json
import random
import unittest
from collections.abc import Mapping

from psse_env.actions import (
    FINALIZE_DIAGNOSIS,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_WLS,
)
from psse_env.dagger.counterfactual_generator import CounterfactualGenerator
from psse_env.dagger.error_injectors import InjectedAction
from psse_env.dagger.rollout_collector import (
    ALL_ADMISSIBLE_SUPERVISION,
    BC0_OBSERVABLE_SEQUENTIAL_SUPERVISION,
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    DaggerRolloutCollector,
    classify_dagger1_recovery_stratum,
    observable_rank_one_target_proof,
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


def _observation_field(observation, name):
    if isinstance(observation, Mapping):
        return observation.get(name)
    return getattr(observation, name, None)


class _LearnerRecoveryOracle:
    def __init__(self):
        self.seen_truth = []
        self.teacher_state_types = []

    def next_actions(self, state, history=None):
        del history
        self.teacher_state_types.append(type(state))
        observation = (
            state.policy_observation
            if isinstance(state, OracleState)
            else state
        )
        if _observation_field(observation, "candidate_lifecycle") == "VERIFIED_REJECT":
            candidate = _observation_field(observation, "candidate_state_id")
            return [
                {
                    "tool": "rollback_state",
                    "arguments": {"candidate_state_id": candidate},
                }
            ]
        return [
            {
                "tool": RUN_WLS,
                "arguments": {"state_id": _observation_field(observation, "active_state_id")},
            }
        ]

    def label_transition(self, **kwargs):
        state = kwargs["state"]
        if isinstance(state, OracleState):
            self.seen_truth.append(copy.deepcopy(state.truth_dict()))
        return {
            "process_valid": True,
            "execution_status": kwargs["tool_output"]["execution_status"],
            "valid_next_actions": [],
        }


class _TruthSensitiveRecoveryOracle(_LearnerRecoveryOracle):
    """Adversarial fixture: private input would change the selected target."""

    def __init__(self):
        super().__init__()
        self.private_teacher_selection_calls = 0

    def next_actions(self, state, history=None):
        if isinstance(state, OracleState):
            self.private_teacher_selection_calls += 1
            if state.true_parameter_errors:
                return [
                    {
                        "tool": "get_parameter_context",
                        "arguments": {"state_id": state.policy_observation.active_state_id},
                    }
                ]
        return super().next_actions(state, history)


class _HistorySensitiveRecoveryOracle(_LearnerRecoveryOracle):
    """Adversarial fixture for post-target private-history leakage."""

    def __init__(self):
        super().__init__()
        self.teacher_histories = []

    def next_actions(self, state, history=None):
        visible_history = copy.deepcopy(list(history or []))
        self.teacher_histories.append(visible_history)
        if any(
            item.get("transition_label", {}).get("opaque_private_route")
            == "parameter"
            for item in visible_history
        ):
            observation = (
                state.policy_observation
                if isinstance(state, OracleState)
                else state
            )
            return [
                {
                    "tool": "get_parameter_context",
                    "arguments": {"state_id": _observation_field(observation, "active_state_id")},
                }
            ]
        return super().next_actions(state, history)

    def label_transition(self, **kwargs):
        result = super().label_transition(**kwargs)
        state = kwargs["state"]
        result["opaque_private_route"] = (
            "parameter"
            if isinstance(state, OracleState) and state.true_parameter_errors
            else "measurement"
        )
        return result


class _LearnerRecoveryStore:
    def __init__(self, scenario):
        self.active_state_id = "active"
        self.states = {
            "active": {
                "state_id": "active",
                "state_hash": "active-hash",
                "case": copy.deepcopy(scenario.get("case", {})),
                "measurements": copy.deepcopy(scenario.get("measurements", [])),
            }
        }

    def exists(self, state_id):
        return str(state_id) in self.states

    def get_state(self, state_id):
        return copy.deepcopy(self.states[str(state_id)])

    def create_rejected_candidate(self, action):
        candidate = copy.deepcopy(self.states[self.active_state_id])
        candidate.update(
            {
                "state_id": "candidate",
                "state_hash": "candidate-hash",
                "parent_state_id": self.active_state_id,
                "source_action": copy.deepcopy(action),
                "verification_output": {
                    "execution_status": "success",
                    "state_id": "candidate",
                    "state_hash": "candidate-hash",
                },
                "candidate_disposition": "REJECT",
            }
        )
        updates = action.get("arguments", {}).get("measurement_updates", {})
        for raw_index, value in updates.items():
            candidate["measurements"][int(raw_index)] = value
        self.states["candidate"] = candidate


class _LearnerRecoveryEnv:
    production_dataset_mode = True

    def __init__(self):
        self.stage = 0
        self.terminal = False
        self.last_reset_scenario = None

    def reset(self, scenario):
        self.last_reset_scenario = copy.deepcopy(scenario)
        self.store = _LearnerRecoveryStore(scenario)
        self.current_candidate_id = None
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
        verification = (
            {
                "execution_status": "success",
                "state_id": "candidate",
                "evidence_source": "observable:test_verifier",
                "physical_constraints_ok": False,
            }
            if self.stage == 1
            else {}
        )
        return PolicyObservation(
            active_state_id="active",
            candidate_state_id="candidate" if self.stage == 1 else None,
            candidate_lifecycle=("VERIFIED_REJECT" if self.stage == 1 else "NO_CANDIDATE"),
            has_open_candidate=self.stage == 1,
            has_verified_candidate=self.stage == 1,
            last_verification=verification,
            history_window=list(history),
            remaining_budget=4 - self.stage,
        )

    def get_oracle_state(self, history):
        observation = self.get_policy_observation(history)
        reset = self.last_reset_scenario or {}
        hidden_truth = {
            "truth_complete": reset.get("truth_complete") is True,
            "clean_case": copy.deepcopy(reset.get("clean_case")),
            "clean_measurements": copy.deepcopy(reset.get("clean_measurements")),
            "true_measurement_errors": copy.deepcopy(
                list(reset.get("true_measurement_errors") or [])
            ),
            "true_parameter_errors": copy.deepcopy(
                list(reset.get("true_parameter_errors") or [])
            ),
            "true_topology_errors": copy.deepcopy(
                list(reset.get("true_topology_errors") or [])
            ),
        }
        return OracleState(
            policy_observation=observation,
            clean_case=copy.deepcopy(reset.get("clean_case")),
            clean_measurements=copy.deepcopy(reset.get("clean_measurements")),
            true_measurement_errors=copy.deepcopy(
                list(reset.get("true_measurement_errors") or [])
            ),
            true_parameter_errors=copy.deepcopy(
                list(reset.get("true_parameter_errors") or [])
            ),
            true_topology_errors=copy.deepcopy(
                list(reset.get("true_topology_errors") or [])
            ),
            candidate_disposition="REJECT" if self.stage == 1 else None,
            candidate_lifecycle=observation.candidate_lifecycle,
            candidate_assessment=(
                {"disposition": "REJECT", "rationale_codes": ["wrong_target"]}
                if self.stage == 1
                else {}
            ),
            hidden_truth=hidden_truth,
        )

    def assert_training_decision_evidence(self, action):
        if self.stage == 1 and action.get("tool") != "rollback_state":
            raise ValueError("rejected learner candidate must roll back")

    def step(self, action):
        if self.stage == 0:
            self.store.create_rejected_candidate(action)
            self.current_candidate_id = "candidate"
            self.stage = 1
        else:
            self.current_candidate_id = None
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


class _HistoryBoundaryEnv(_LearnerRecoveryEnv):
    """Expose a bounded observation history while retaining private labels."""

    def current_state(self):
        return {
            "active_state_id": "active",
            "candidate_state_id": None,
            "remaining_budget": 2 - self.stage,
        }

    def get_policy_observation(self, history):
        del history
        return PolicyObservation(
            active_state_id="active",
            history_window=[],
            remaining_budget=2 - self.stage,
        )

    def assert_training_decision_evidence(self, action):
        del action

    def step(self, action):
        del action
        self.stage += 1
        self.terminal = self.stage >= 2
        return self.current_state(), {
            "execution_status": "success",
            "error_code": None,
            "state_mutated": False,
            "tool_metrics": {},
        }


class _RunWlsPolicy:
    @staticmethod
    def act(observation):
        return {
            "tool": RUN_WLS,
            "arguments": {"state_id": observation["active_state_id"]},
        }


def _collect_history_boundary(true_parameter_errors, supervision_policy):
    oracle = _HistorySensitiveRecoveryOracle()
    rows = DaggerRolloutCollector(
        env=_HistoryBoundaryEnv(),
        policy=_RunWlsPolicy(),
        expert_oracle=oracle,
        rng=random.Random(0),
        supervision_policy=supervision_policy,
        forbidden_physical_roots={"held-out-root"},
    ).collect_iteration(
        scenarios=[
            _scenario(
                scenario_id="history-boundary",
                case={},
                measurements=[1.0],
                root_scenario_id="history-boundary",
                physical_root_fingerprint="history-boundary-root",
                scenario_family="parameter",
                error_cardinality=1,
                case_id="case14",
                dataset_split="dagger_train",
                source_tier="generated",
                truth_complete=True,
                clean_measurements=[1.0],
                true_parameter_errors=copy.deepcopy(true_parameter_errors),
            )
        ],
        iteration=1,
        beta=0.25,
        max_steps=2,
        collection_role="training",
    )
    return oracle, rows


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
                "release_audit": {"offline_only": True},
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
        for private_key in (
            "true_measurement_errors",
            "clean_measurements",
            "release_audit",
            "offline_only",
        ):
            self.assertNotIn(private_key, policy_payload)
            self.assertNotIn(private_key, exported_payload)

    def test_dagger1_teacher_targets_are_invariant_to_hidden_truth(self):
        def collect(private_truth):
            oracle = _TruthSensitiveRecoveryOracle()
            scenario = _scenario(
                scenario_id="truth-boundary",
                case={},
                measurements=[1.0, 9.0],
                root_scenario_id="truth-boundary",
                physical_root_fingerprint="truth-boundary-root",
                scenario_family="measurement",
                error_cardinality=1,
                case_id="case14",
                dataset_split="dagger_train",
                source_tier="generated",
                **copy.deepcopy(private_truth),
            )
            rows = DaggerRolloutCollector(
                env=_LearnerRecoveryEnv(),
                policy=_LearnerRecoveryPolicy(),
                expert_oracle=oracle,
                rng=random.Random(0),
                supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
                forbidden_physical_roots={"held-out-root"},
            ).collect_iteration(
                scenarios=[scenario],
                iteration=1,
                beta=0.25,
                max_steps=2,
                collection_role="training",
            )
            return oracle, rows

        shared_truth = {
            "truth_complete": True,
            "clean_measurements": [1.0, 2.0],
            "true_measurement_errors": [{"index": 1, "clean": 2.0}],
        }
        plain_oracle, plain_rows = collect(shared_truth)
        changed_oracle, changed_rows = collect(
            {
                **shared_truth,
                # This private fault would deliberately change the adversarial
                # fixture's target if the collector passed OracleState into
                # teacher selection.
                "true_parameter_errors": [
                    {"line_index": 1, "field": "r", "clean": 0.1}
                ],
            }
        )

        self.assertEqual(
            [row["policy_observation"] for row in plain_rows],
            [row["policy_observation"] for row in changed_rows],
        )
        self.assertEqual(
            [row["preferred_action"] for row in plain_rows],
            [row["preferred_action"] for row in changed_rows],
        )
        self.assertEqual(
            [row["next_valid_actions"] for row in plain_rows],
            [row["next_valid_actions"] for row in changed_rows],
        )
        for oracle in (plain_oracle, changed_oracle):
            self.assertEqual(oracle.private_teacher_selection_calls, 0)
            self.assertTrue(oracle.teacher_state_types)
            # No OracleState may reach teacher selection.
            self.assertNotIn(OracleState, set(oracle.teacher_state_types))
            # And under D1 observable supervision the teacher must receive the
            # validated truth-free mapping produced by the shared selector.
            # Asserting the exact type stops a future caller from bypassing the
            # helper and quietly restoring the PolicyObservation path, which is
            # the architecture this regression exists to protect.
            self.assertEqual(set(oracle.teacher_state_types), {dict})
        self.assertEqual(
            [row["preferred_action"]["tool"] for row in plain_rows],
            [RUN_WLS, "rollback_state"],
        )
        self.assertTrue(plain_rows[1]["production_label_eligible"])
        self.assertTrue(changed_rows[1]["production_label_eligible"])

    def test_dagger1_teacher_cannot_read_private_transition_history(self):
        plain_oracle, plain_rows = _collect_history_boundary(
            [], DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
        )
        changed_oracle, changed_rows = _collect_history_boundary(
            [{"line_index": 1, "field": "r", "clean": 0.1}],
            DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
        )

        self.assertEqual(
            [row["policy_observation"] for row in plain_rows],
            [row["policy_observation"] for row in changed_rows],
        )
        self.assertEqual(
            [row["preferred_action"] for row in plain_rows],
            [row["preferred_action"] for row in changed_rows],
        )
        self.assertEqual(
            [row["next_valid_actions"] for row in plain_rows],
            [row["next_valid_actions"] for row in changed_rows],
        )
        self.assertEqual(
            [row["preferred_action"]["tool"] for row in plain_rows],
            [RUN_WLS, RUN_WLS],
        )
        for oracle in (plain_oracle, changed_oracle):
            self.assertTrue(oracle.teacher_histories)
            self.assertTrue(all(history == [] for history in oracle.teacher_histories))

    def test_history_fixture_retargets_a_teacher_that_reads_private_history(self):
        # Positive control for the D1 history boundary: the all-admissible
        # collector hands the teacher its private transition history, so the
        # adversarial fixture must see the parameter route and retarget.  An
        # inert fixture would make the D1 invariance above hold vacuously.
        parameter_context = {
            "tool": "get_parameter_context",
            "arguments": {"state_id": "active"},
        }
        _, plain_rows = _collect_history_boundary([], ALL_ADMISSIBLE_SUPERVISION)
        _, changed_rows = _collect_history_boundary(
            [{"line_index": 1, "field": "r", "clean": 0.1}],
            ALL_ADMISSIBLE_SUPERVISION,
        )
        self.assertEqual(
            [row["preferred_action"]["tool"] for row in plain_rows],
            [RUN_WLS, RUN_WLS],
        )
        self.assertEqual(
            [row["preferred_action"] for row in changed_rows],
            [{"tool": RUN_WLS, "arguments": {"state_id": "active"}}, parameter_context],
        )
        # The D1 selector hands the teacher a truth-free mapping rather than
        # an OracleState; a leaked route must retarget that input too.
        self.assertEqual(
            _HistorySensitiveRecoveryOracle().next_actions(
                {"active_state_id": "active"},
                [{"transition_label": {"opaque_private_route": "parameter"}}],
            ),
            [parameter_context],
        )

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
                    "last_tool": "correct_measurements",
                    "last_tool_status": "failure",
                    "last_tool_output": {
                        "execution_status": "failure",
                        "error_code": "post_correction_confirmation_required",
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


if __name__ == "__main__":
    unittest.main()
