from __future__ import annotations

import copy
import json
import unittest

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
)
from psse_env.dagger.release_audit import (
    REMAINING_FAULTS_CHECK,
    audit_episode_against_truth,
)
from psse_env.oracle import CandidateQualityOracle
from psse_env.private_target_matching import matched_private_fault_indices
from psse_env.transactional_env import TransactionalPSSEEnv


def _branch_row(r: float, x: float) -> list[float]:
    return [1.0, 2.0, r, x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class PrivateTruthRetirementTests(unittest.TestCase):
    def test_grouped_measurement_cardinalities_two_through_five(self) -> None:
        for cardinality in (2, 3, 4, 5):
            with self.subTest(cardinality=cardinality):
                targets = list(range(cardinality))
                clean = [float(index + 1) for index in targets]
                parent = [value + 5.0 for value in clean]
                candidate = [value + 0.05 for value in clean]
                truth = {
                    "truth_complete": True,
                    "clean_measurements": clean,
                    "true_measurement_errors": [
                        {"index": index, "clean": clean[index]}
                        for index in targets
                    ],
                    "release_audit": {
                        "tolerances": {"measurement_abs": 0.1}
                    },
                }
                action = {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {"suspect_group": targets},
                }

                self.assertEqual(
                    matched_private_fault_indices(
                        action,
                        truth,
                        parent_state={"measurements": parent},
                        candidate_state={"measurements": candidate},
                    ),
                    targets,
                )

    def test_wrong_and_outside_tolerance_measurements_do_not_retire(self) -> None:
        truth = {
            "truth_complete": True,
            "clean_measurements": [1.0, 2.0],
            "true_measurement_errors": [{"index": 1, "clean": 2.0}],
            "release_audit": {"tolerances": {"measurement_abs": 0.1}},
        }
        parent = {"measurements": [9.0, 8.0]}
        outside = {"measurements": [9.0, 2.11]}
        corrected_wrong_target = {"measurements": [1.0, 8.0]}

        self.assertEqual(
            matched_private_fault_indices(
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {"suspect_group": [0]},
                },
                truth,
                parent_state=parent,
                candidate_state=corrected_wrong_target,
            ),
            [],
        )
        self.assertEqual(
            matched_private_fault_indices(
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {"suspect_group": [1]},
                },
                truth,
                parent_state=parent,
                candidate_state=outside,
            ),
            [],
        )

    def test_compact_parameter_rx_uses_one_based_line_and_path_loader(self) -> None:
        cases = {
            "parent": {"branch": [_branch_row(0.01, 0.02), _branch_row(0.5, 0.2)]},
            "within": {"branch": [_branch_row(0.01, 0.02), _branch_row(0.11, 0.205)]},
            "outside": {"branch": [_branch_row(0.01, 0.02), _branch_row(0.121, 0.205)]},
        }
        truth = {
            "truth_complete": True,
            "true_parameter_errors": [
                {
                    "branch_row0": 1,
                    "line_index1": 2,
                    "parameter": "rx",
                    "clean_r": 0.1,
                    "clean_x": 0.2,
                }
            ],
            "release_audit": {"tolerances": {"final_case_abs": 0.02}},
        }
        loader = cases.__getitem__
        action = {
            "tool": CORRECT_PARAMETERS,
            "arguments": {"line_index": 2},
        }

        self.assertEqual(
            matched_private_fault_indices(
                action,
                truth,
                parent_state={"case": "parent"},
                candidate_state={"case": "within"},
                case_loader=loader,
            ),
            [0],
        )
        self.assertEqual(
            matched_private_fault_indices(
                action,
                truth,
                parent_state={"case": "parent"},
                candidate_state={"case": "outside"},
                case_loader=loader,
            ),
            [],
        )
        self.assertEqual(
            matched_private_fault_indices(
                {
                    "tool": CORRECT_PARAMETERS,
                    "arguments": {"line_index": 1},
                },
                truth,
                parent_state={"case": "parent"},
                candidate_state={"case": "within"},
                case_loader=loader,
            ),
            [],
        )

    def test_topology_status_field_must_match_before_truth_retires(self) -> None:
        truth = {
            "truth_complete": True,
            "true_topology_errors": [
                {
                    "branch_row0": 0,
                    "status_field": "in_service",
                    "expected_status": 1,
                }
            ],
        }
        parent = {
            "case": {"branch": [{"branch_id": "b0", "in_service": 0}]}
        }
        wrong_field = {
            "case": {
                "branch": [
                    {"branch_id": "b0", "in_service": 0, "status": 1}
                ]
            }
        }
        corrected = {
            "case": {"branch": [{"branch_id": "b0", "in_service": 1}]}
        }

        self.assertEqual(
            matched_private_fault_indices(
                {
                    "tool": CORRECT_TOPOLOGY,
                    "arguments": {"branch_row0": 0, "status": 1},
                },
                truth,
                parent_state=parent,
                candidate_state=wrong_field,
            ),
            [],
        )
        self.assertEqual(
            matched_private_fault_indices(
                {
                    "tool": CORRECT_TOPOLOGY,
                    "arguments": {
                        "branch_row0": 0,
                        "status_field": "in_service",
                        "status": 1,
                    },
                },
                truth,
                parent_state=parent,
                candidate_state=corrected,
            ),
            [0],
        )
    def _advance_private_commit(
        self,
        env: TransactionalPSSEEnv,
        *,
        action: dict,
        modification: dict,
    ) -> None:
        parent_id = str(env.store.active_state_id)
        candidate_id = env.store.clone_candidate(
            parent_id,
            modification,
            action,
            created_at_step=1,
        )
        candidate = env.store.get_state(candidate_id)
        next_truth = env._truth_after_commit(candidate)
        accepted = env._accepted_candidate_record(candidate)
        env.store.mark_verified(candidate_id, {}, "ACCEPT_PARTIAL")
        env.store.commit(candidate_id)
        env._oracle_payload = next_truth
        env.context_flags.setdefault("accepted_corrections", []).append(accepted)

    def test_environment_retirement_preserves_configured_path_case_loader(
        self,
    ) -> None:
        cases = {
            "parent": {"branch": [_branch_row(0.5, 0.2)]},
            "candidate": {"branch": [_branch_row(0.11, 0.205)]},
        }
        env = TransactionalPSSEEnv(
            candidate_quality_oracle=CandidateQualityOracle(
                case_loader=cases.__getitem__
            )
        )
        state = env.reset(
            {
                "scenario_id": "configured-path-loader-retirement",
                "case": "parent",
                "measurements": [1.0],
                "true_parameter_errors": [
                    {
                        "branch_row0": 0,
                        "line_index1": 1,
                        "parameter": "rx",
                        "clean_r": 0.1,
                        "clean_x": 0.2,
                    }
                ],
                "release_audit": {
                    "tolerances": {"final_case_abs": 0.02}
                },
            }
        )
        self._advance_private_commit(
            env,
            action={
                "tool": CORRECT_PARAMETERS,
                "arguments": {
                    "state_id": state["active_state_id"],
                    "line_index": 1,
                },
            },
            modification={"case": "candidate"},
        )

        remaining = env.get_oracle_state().truth_dict()
        self.assertEqual(remaining["remaining_true_fault_count"], 0)
        self.assertEqual(remaining["remaining_true_faults"], [])

    def test_mixed_family_both_orders_end_with_zero_supplied_and_derived(self) -> None:
        initial_case = {"branch": [_branch_row(0.5, 0.2)]}
        corrected_case = {"branch": [_branch_row(0.11, 0.205)]}
        scenario = {
            "scenario_id": "private-retirement-both-orders",
            "case": initial_case,
            "clean_case": {"branch": [_branch_row(0.1, 0.2)]},
            "measurements": [9.0],
            "clean_measurements": [1.0],
            "true_measurement_errors": [{"index": 0, "clean": 1.0}],
            "true_parameter_errors": [
                {
                    "branch_row0": 0,
                    "line_index1": 1,
                    "parameter": "rx",
                    "clean_r": 0.1,
                    "clean_x": 0.2,
                }
            ],
            "release_audit": {
                "tolerances": {
                    "measurement_abs": 0.1,
                    "final_case_abs": 0.02,
                }
            },
        }
        measurement = (
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"suspect_group": [0]},
            },
            {"measurement_updates": {0: 1.05}},
        )
        parameter = (
            {
                "tool": CORRECT_PARAMETERS,
                "arguments": {"line_index": 1},
            },
            {"case": corrected_case},
        )

        for ordered in ((measurement, parameter), (parameter, measurement)):
            with self.subTest(order=[item[0]["tool"] for item in ordered]):
                env = TransactionalPSSEEnv(
                    candidate_quality_oracle=CandidateQualityOracle()
                )
                env.reset(copy.deepcopy(scenario))
                for action, modification in ordered:
                    bound_action = copy.deepcopy(action)
                    bound_action["arguments"]["state_id"] = str(
                        env.store.active_state_id
                    )
                    self._advance_private_commit(
                        env,
                        action=bound_action,
                        modification=copy.deepcopy(modification),
                    )

                remaining = env.get_oracle_state().truth_dict()
                self.assertEqual(remaining["remaining_true_fault_count"], 0)
                self.assertEqual(remaining["remaining_true_faults"], [])

                audit = audit_episode_against_truth(
                    scenario,
                    env.current_state(),
                    terminal=True,
                    terminal_outcome="resolved",
                    active_physical_state=env.store.get_state(
                        str(env.store.active_state_id)
                    ),
                    remaining_truth=remaining,
                )
                remaining_check = audit["checks"][REMAINING_FAULTS_CHECK]
                self.assertEqual(remaining_check["status"], "passed", audit)
                self.assertEqual(
                    remaining_check["derived_remaining_fault_count"], 0
                )

    def test_private_release_tolerances_never_enter_policy_observation(self) -> None:
        env = TransactionalPSSEEnv()
        env.reset(
            {
                "scenario_id": "private-tolerance-boundary",
                "case": {},
                "measurements": [9.0],
                "clean_measurements": [1.0],
                "true_measurement_errors": [{"index": 0, "clean": 1.0}],
                "release_audit": {
                    "tolerances": {"measurement_abs": 0.25}
                },
            }
        )

        serialized = json.dumps(env.get_policy_observation().as_dict())
        self.assertNotIn("release_audit", serialized)
        self.assertNotIn("measurement_abs", serialized)
        self.assertEqual(
            env.get_oracle_state().hidden_truth["release_audit"]["tolerances"][
                "measurement_abs"
            ],
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
