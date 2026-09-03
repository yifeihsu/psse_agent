from __future__ import annotations

import unittest

from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.measurement_expert import MeasurementExpert
from psse_env.oracle.candidate_quality import (
    CandidateDisposition,
    CandidateQualityOracle,
)
from psse_env.oracle.parameter_expert import ParameterExpert
from psse_env.oracle.topology_expert import TopologyExpert


def _state(signatures: list[str]) -> dict:
    return {
        "active_state_id": "episode:s1",
        "candidate_state_id": None,
        "no_material_anomaly_remaining": False,
        "unresolved_signatures": signatures,
        "has_fresh_measurement_context": False,
        "measurement_context_state_id": None,
        "has_fresh_parameter_context": False,
        "parameter_context_state_id": None,
    }


class MultiMeasurementContinuationTests(unittest.TestCase):
    @staticmethod
    def _rejected_singleton_history(
        state_id: str,
        candidate_id: str,
        target: int,
        *,
        physical_ok: bool = True,
        violation_indices: tuple[int, ...] = (),
    ) -> list[dict]:
        action = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [target]},
        }
        return [
            {"action": action, "tool_output": {"execution_status": "success"}},
            {
                "action": {
                    "tool": "run_wls",
                    "arguments": {"state_id": candidate_id},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "target_metric_value": 0.01,
                        "target_metric_threshold": 3.0,
                        "target_progress": 0.99,
                        "global_progress": 0.15,
                        "globally_resolved": False,
                        "physical_constraints_ok": physical_ok,
                        "physical_bound_violations": [
                            {
                                "type": "bus_voltage_out_of_bounds",
                                "measurement_index0": index,
                            }
                            for index in violation_indices
                        ],
                    },
                },
            },
            {
                "action": {
                    "tool": "rollback_state",
                    "arguments": {"candidate_state_id": candidate_id},
                },
                "tool_output": {"execution_status": "success"},
            },
        ]

    def test_two_safe_locally_fixed_rejections_get_one_joint_retry(self) -> None:
        state_id = "episode:s0"
        state = _state(
            [
                "wls_residual_outlier index=67 channel=Qf",
                "wls_residual_outlier index=69 channel=Qf",
            ]
        )
        state.update(
            {
                "active_state_id": state_id,
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": state_id,
                "remaining_budget": 4,
                "rejected_hypotheses": [
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:c1",
                        "source_action": {
                            "tool": "correct_measurements",
                            "arguments": {
                                "state_id": state_id,
                                "suspect_group": [67],
                            },
                        },
                    },
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:c2",
                        "source_action": {
                            "tool": "correct_measurements",
                            "arguments": {
                                "state_id": state_id,
                                "suspect_group": [69],
                            },
                        },
                    },
                ],
            }
        )
        hints = [
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": state_id, "suspect_group": [target]},
            }
            for target in (67, 69, 74)
        ]
        hints.append(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state_id,
                    "suspect_group": [67, 69],
                },
            }
        )
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": state_id,
                        "supported_corrections": hints,
                    },
                },
            },
            *self._rejected_singleton_history(
                state_id, f"{state_id}:c1", 67
            ),
            *self._rejected_singleton_history(
                state_id, f"{state_id}:c2", 69
            ),
        ]

        proposals = MeasurementExpert().propose(
            state, history, oracle_hints=hints
        )
        joint = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [67, 69]},
        }

        self.assertIn(joint, [proposal.action for proposal in proposals])

    def test_joint_retry_survives_aged_out_context_and_first_verification(self) -> None:
        state_id = "episode:s0"
        singleton_67 = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [67]},
        }
        singleton_69 = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [69]},
        }
        joint = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [67, 69]},
        }
        verification = {
            "target_metric_value": 0.01,
            "target_metric_threshold": 3.0,
            "target_progress": 0.99,
            "global_progress": 0.15,
            "globally_resolved": False,
            "physical_constraints_ok": True,
            "physical_bound_violations": [],
        }
        state = _state(
            [
                "wls_residual_outlier index=67 channel=Qf",
                "wls_residual_outlier index=69 channel=Qf",
            ]
        )
        state.update(
            {
                "active_state_id": state_id,
                "last_tool": "rollback_state",
                "last_tool_status": "success",
                "last_tool_output": {"execution_status": "success"},
                "remaining_anomaly_score": 5.0,
                "remaining_budget": 4,
                "requires_measurement_context": True,
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": state_id,
                "fresh_context_evidence": {
                    "measurement": {
                        "state_id": state_id,
                        "supported_corrections": [
                            singleton_67,
                            singleton_69,
                            joint,
                        ],
                    }
                },
                "rejected_hypotheses": [
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:c1",
                        "source_action": singleton_67,
                        "verification_summary": verification,
                    },
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:c2",
                        "source_action": singleton_69,
                        "verification_summary": verification,
                    },
                ],
            }
        )
        # The four-event release window has lost both the context transition and
        # c1's WLS verification.  All inputs needed for the conditional retry are
        # therefore read from the active-state observable recovery ledger.
        history = [
            {
                "action": {
                    "tool": "rollback_state",
                    "arguments": {"candidate_state_id": f"{state_id}:c1"},
                },
                "tool_output": {"execution_status": "success"},
            },
            {
                "action": singleton_69,
                "tool_output": {
                    "execution_status": "success",
                    "candidate_state_id": f"{state_id}:c2",
                },
            },
            {
                "action": {
                    "tool": "run_wls",
                    "arguments": {"state_id": f"{state_id}:c2"},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": verification,
                },
            },
            {
                "action": {
                    "tool": "rollback_state",
                    "arguments": {"candidate_state_id": f"{state_id}:c2"},
                },
                "tool_output": {"execution_status": "success"},
            },
        ]
        oracle = ExpertPolicyOracle(
            process_oracle=ProcessValidityOracle(
                executor_hydrated_corrections=True
            )
        )

        actions = oracle.next_actions(state, history)

        self.assertTrue(actions)
        self.assertEqual(actions[0], joint)

    def test_joint_retry_requires_physical_safety_and_budget(self) -> None:
        state_id = "episode:s0"
        hints = [
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": state_id, "suspect_group": [target]},
            }
            for target in (67, 69)
        ]
        hints.append(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": state_id,
                    "suspect_group": [67, 69],
                },
            }
        )
        rejected = [
            {
                "candidate_parent_id": state_id,
                "candidate_state_id": f"{state_id}:c{offset}",
                "source_action": hints[offset - 1],
            }
            for offset in (1, 2)
        ]
        base = _state(
            [
                "wls_residual_outlier index=67 channel=Qf",
                "wls_residual_outlier index=69 channel=Qf",
            ]
        )
        base.update(
            {
                "active_state_id": state_id,
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": state_id,
                "rejected_hypotheses": rejected,
            }
        )
        history = [
            *self._rejected_singleton_history(
                state_id, f"{state_id}:c1", 67
            ),
            *self._rejected_singleton_history(
                state_id, f"{state_id}:c2", 69, physical_ok=False
            ),
        ]
        joint = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [67, 69]},
        }

        unsafe = dict(base, remaining_budget=4)
        low_budget = dict(base, remaining_budget=3)

        self.assertNotIn(
            joint,
            [
                proposal.action
                for proposal in MeasurementExpert().propose(
                    unsafe, history, oracle_hints=hints
                )
            ],
        )
        self.assertNotIn(
            joint,
            [
                proposal.action
                for proposal in MeasurementExpert().propose(
                    low_budget,
                    [
                        *self._rejected_singleton_history(
                            state_id, f"{state_id}:c1", 67
                        ),
                        *self._rejected_singleton_history(
                            state_id, f"{state_id}:c2", 69
                        ),
                    ],
                    oracle_hints=hints,
                )
            ],
        )

    def test_dependency_closure_joins_only_the_exact_vm_violation_target(
        self,
    ) -> None:
        state_id = "episode:s0"
        hints = [
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": state_id, "suspect_group": [target]},
            }
            for target in (0, 10)
        ]
        hints.append(
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": state_id, "suspect_group": [0, 10]},
            }
        )
        state = _state(
            [
                "wls_residual_outlier index=0 channel=Vm",
                "wls_residual_outlier index=10 channel=Vm",
            ]
        )
        state.update(
            {
                "active_state_id": state_id,
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": state_id,
                "remaining_budget": 4,
                "rejected_hypotheses": [
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:blocked",
                        "source_action": hints[1],
                    },
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:closure",
                        "source_action": hints[0],
                    },
                ],
            }
        )
        blocked_history = self._rejected_singleton_history(
            state_id,
            f"{state_id}:blocked",
            10,
            physical_ok=False,
            violation_indices=(0,),
        )
        # The blocked target made enough global progress that physical safety,
        # not the ordinary partial-progress floor, is the reason it rejected.
        blocked_history[1]["tool_output"]["tool_metrics"][
            "global_progress"
        ] = 0.30
        history = [
            *blocked_history,
            *self._rejected_singleton_history(
                state_id, f"{state_id}:closure", 0
            ),
        ]
        joint = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [0, 10]},
        }

        proposals = MeasurementExpert().propose(
            state, history, oracle_hints=hints
        )

        self.assertIn(joint, [proposal.action for proposal in proposals])

    def test_dependency_closure_rejects_uncovered_or_non_vm_violations(self) -> None:
        state_id = "episode:s0"
        hints = [
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": state_id, "suspect_group": [target]},
            }
            for target in (0, 10)
        ]
        hints.append(
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": state_id, "suspect_group": [0, 10]},
            }
        )
        state = _state(
            [
                "wls_residual_outlier index=0 channel=Vm",
                "wls_residual_outlier index=10 channel=Vm",
            ]
        )
        state.update(
            {
                "active_state_id": state_id,
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": state_id,
                "remaining_budget": 4,
                "rejected_hypotheses": [
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:blocked",
                        "source_action": hints[1],
                    },
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:closure",
                        "source_action": hints[0],
                    },
                ],
            }
        )
        blocked = self._rejected_singleton_history(
            state_id,
            f"{state_id}:blocked",
            10,
            physical_ok=False,
            violation_indices=(1,),
        )
        blocked[1]["tool_output"]["tool_metrics"]["global_progress"] = 0.30
        history = [
            *blocked,
            *self._rejected_singleton_history(
                state_id, f"{state_id}:closure", 0
            ),
        ]
        joint = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [0, 10]},
        }

        proposals = MeasurementExpert().propose(
            state, history, oracle_hints=hints
        )

        self.assertNotIn(joint, [proposal.action for proposal in proposals])

    def test_production_evidence_requires_the_exact_context_supported_joint(self) -> None:
        from psse_env.transactional_env import TransactionalPSSEEnv

        state_id = "episode:s0"
        hints = [
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": state_id, "suspect_group": [target]},
            }
            for target in (0, 10)
        ]
        hints.append(
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": state_id, "suspect_group": [0, 10]},
            }
        )
        state = _state(
            [
                "wls_residual_outlier index=0 channel=Vm",
                "wls_residual_outlier index=10 channel=Vm",
            ]
        )
        state.update(
            {
                "active_state_id": state_id,
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": state_id,
                "remaining_budget": 4,
                "rejected_hypotheses": [
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:blocked",
                        "source_action": hints[1],
                    },
                    {
                        "candidate_parent_id": state_id,
                        "candidate_state_id": f"{state_id}:closure",
                        "source_action": hints[0],
                    },
                ],
            }
        )
        blocked = self._rejected_singleton_history(
            state_id,
            f"{state_id}:blocked",
            10,
            physical_ok=False,
            violation_indices=(0,),
        )
        blocked[1]["tool_output"]["tool_metrics"]["global_progress"] = 0.30
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": state_id,
                        "supported_corrections": hints,
                    },
                },
            },
            *blocked,
            *self._rejected_singleton_history(
                state_id, f"{state_id}:closure", 0
            ),
        ]
        env = object.__new__(TransactionalPSSEEnv)
        env.production_dataset_mode = True
        env.history = history
        env.current_state = lambda: state
        exact = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [0, 10]},
        }
        unsupported_superset = {
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "suspect_group": [0, 10, 99]},
        }

        env.assert_training_decision_evidence(exact)
        with self.assertRaisesRegex(ValueError, "not supported by the latest"):
            env.assert_training_decision_evidence(unsupported_superset)

    def test_near_threshold_branch_cross_signal_still_refreshes_accepted_targets(
        self,
    ) -> None:
        state = _state(
            [
                "wls_residual_outlier index=120 channel=Qt",
                "wls_branch_multiplier line_status_or_parameter line=7",
            ]
        )
        state.update(
            {
                "remaining_anomaly_score": 1.024,
                "remaining_budget": 11,
                "accepted_corrections": [
                    {
                        "source_action": {
                            "tool": "correct_measurements",
                            "arguments": {
                                "state_id": "episode:s0",
                                "suspect_group": [target],
                            },
                        }
                    }
                    for target in (24, 25, 26)
                ],
            }
        )

        proposals = MeasurementExpert().propose(state, [])

        self.assertEqual(proposals[0].action["tool"], "get_measurement_context")
        self.assertIn(
            "near_threshold_accepted_target_refinement",
            proposals[0].evidence_codes,
        )

    def test_homogeneous_residual_channel_breaks_neutral_context_tie(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=67 channel=Qf",
                "wls_residual_outlier index=68 channel=Qf",
                "wls_residual_outlier index=69 channel=Qf",
                "wls_branch_multiplier line_status_or_parameter line=7",
            ]
        )

        proposal = MeasurementExpert().propose(state, [])[0]

        self.assertEqual(proposal.action["tool"], "get_measurement_context")
        self.assertGreater(proposal.confidence, 0.87)

    def test_homogeneous_residuals_do_not_hide_branch_routes_before_partial_commit(
        self,
    ) -> None:
        state = _state(
            [
                "wls_residual_outlier index=67 channel=Qf",
                "wls_residual_outlier index=68 channel=Qf",
                "wls_residual_outlier index=69 channel=Qf",
                "wls_branch_multiplier line_status_or_parameter line=7",
            ]
        )

        parameter = ParameterExpert().propose(state, [])
        topology = TopologyExpert().propose(state, [])

        self.assertEqual(parameter[0].action["tool"], "get_parameter_context")
        self.assertEqual(topology[0].action["tool"], "get_topology_context")

    def test_homogeneous_partial_measurement_keeps_weak_branch_routes_live(
        self,
    ) -> None:
        state = _state(
            [
                "wls_residual_outlier index=67 channel=Qf",
                "wls_residual_outlier index=68 channel=Qf",
                "wls_residual_outlier index=69 channel=Qf",
                "wls_branch_multiplier line_status_or_parameter line=7",
            ]
        )
        state["accepted_corrections"] = [
            {
                "candidate_state_id": "episode:s1",
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s0",
                        "suspect_group": [64],
                    },
                },
            }
        ]

        parameter = ParameterExpert().propose(state, [])
        topology = TopologyExpert().propose(state, [])

        self.assertEqual(parameter[0].action["tool"], "get_parameter_context")
        self.assertEqual(topology[0].action["tool"], "get_topology_context")

    def test_partial_joint_group_remains_conditional_without_singleton_closure(
        self,
    ) -> None:
        active_id = "episode:s1"
        singleton_68 = {
            "tool": "correct_measurements",
            "arguments": {"state_id": active_id, "suspect_group": [68]},
        }
        joint = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": active_id,
                "suspect_group": [67, 68, 69],
            },
        }
        state = _state(
            [
                "wls_residual_outlier index=67 channel=Qf",
                "wls_residual_outlier index=68 channel=Qf",
                "wls_residual_outlier index=69 channel=Qf",
            ]
        )
        state.update(
            {
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": active_id,
                "accepted_corrections": [
                    {
                        "candidate_state_id": active_id,
                        "source_action": {
                            "tool": "correct_measurements",
                            "arguments": {
                                "state_id": "episode:s0",
                                "suspect_group": [67],
                            },
                        },
                    }
                ],
            }
        )

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[singleton_68, joint]
        )

        self.assertNotIn(joint, [proposal.action for proposal in proposals])

    def test_observable_joint_refinement_of_accepted_targets_is_admissible(self) -> None:
        state = _state([])
        state["has_fresh_measurement_context"] = True
        state["measurement_context_state_id"] = "episode:s1"
        state["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s0",
                        "suspect_group": [index],
                    },
                }
            }
            for index in (104, 110, 113, 114)
        ]
        refinement = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": "episode:s1",
                "suspect_group": [104, 110, 113, 114],
            },
        }
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "episode:s1"},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": "episode:s1",
                        "accepted_target_refinement": True,
                        "supported_corrections": [refinement],
                    },
                },
            }
        ]

        proposals = MeasurementExpert().propose(
            state, history, oracle_hints=[refinement]
        )

        self.assertIn(refinement, [proposal.action for proposal in proposals])
        proposal = next(item for item in proposals if item.action == refinement)
        self.assertIn(
            "observable_accepted_target_refinement", proposal.evidence_codes
        )

    def test_observable_post_branch_singleton_refinement_is_admissible(self) -> None:
        state = _state([])
        state["has_fresh_measurement_context"] = True
        state["measurement_context_state_id"] = "episode:s1"
        state["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s0",
                        "suspect_group": [45],
                    },
                }
            },
            {
                "source_action": {
                    "tool": "correct_parameters",
                    "arguments": {
                        "state_id": "episode:s0",
                        "line_index": 7,
                    },
                }
            },
        ]
        refinement = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": "episode:s1",
                "suspect_group": [45],
            },
        }
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "episode:s1"},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": "episode:s1",
                        "accepted_target_refinement": True,
                        "accepted_target_refinement_kind": (
                            "post_branch_model_reestimate"
                        ),
                        "supported_corrections": [refinement],
                    },
                },
            }
        ]

        proposals = MeasurementExpert().propose(
            state, history, oracle_hints=[refinement]
        )

        self.assertIn(refinement, [proposal.action for proposal in proposals])
        proposal = next(item for item in proposals if item.action == refinement)
        self.assertIn(
            "observable_accepted_target_refinement", proposal.evidence_codes
        )

    def test_joint_refinement_cannot_add_an_unaccepted_target(self) -> None:
        state = _state([])
        state["has_fresh_measurement_context"] = True
        state["measurement_context_state_id"] = "episode:s1"
        state["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s0",
                        "suspect_group": [index],
                    },
                }
            }
            for index in (104, 110)
        ]
        unsafe_refinement = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": "episode:s1",
                "suspect_group": [104, 110, 999],
            },
        }
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "episode:s1"},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": "episode:s1",
                        "accepted_target_refinement": True,
                        "supported_corrections": [unsafe_refinement],
                    },
                },
            }
        ]

        proposals = MeasurementExpert().propose(
            state, history, oracle_hints=[unsafe_refinement]
        )

        self.assertNotIn(
            unsafe_refinement, [proposal.action for proposal in proposals]
        )

    def test_partial_measurement_commit_requires_fresh_context_on_active_state(self) -> None:
        state = _state([])
        state["accepted_corrections"] = [
            {
                "candidate_state_id": "episode:s1",
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s0",
                        "suspect_group": [12, 18],
                    },
                },
            }
        ]

        proposals = MeasurementExpert().propose(state, [])

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action["tool"], "get_measurement_context")
        self.assertEqual(
            proposals[0].action["arguments"]["state_id"], "episode:s1"
        )
        self.assertIn(
            "fresh_post_commit_context_required", proposals[0].evidence_codes
        )

    @staticmethod
    def _partial_measurement_branch_inventory_state(
        *, topology_rejected: bool
    ) -> tuple[dict, dict]:
        active_id = "episode:s1"
        measurement_action = {
            "tool": "correct_measurements",
            "arguments": {"state_id": active_id, "suspect_group": [13]},
        }
        topology_action = {
            "tool": "correct_topology",
            "arguments": {"state_id": active_id, "line_index": 5, "status": 0},
        }
        state = _state(
            [
                "wls_residual_outlier index=13 channel=Vm",
                "wls_branch_multiplier line_status_or_parameter line=5",
            ]
        )
        state.update(
            {
                "active_state_id": active_id,
                "remaining_anomaly_score": 1.05,
                "remaining_budget": 12,
                "accepted_corrections": [
                    {
                        "candidate_state_id": active_id,
                        "source_action": {
                            "tool": "correct_measurements",
                            "arguments": {
                                "state_id": "episode:s0",
                                "suspect_group": [8],
                            },
                        },
                    }
                ],
                "has_fresh_measurement_context": True,
                "measurement_context_state_id": active_id,
                "has_fresh_parameter_context": True,
                "parameter_context_state_id": active_id,
                "has_fresh_topology_context": True,
                "topology_context_state_id": active_id,
                "fresh_context_evidence": {
                    "measurement": {
                        "state_id": active_id,
                        "supported_corrections": [measurement_action],
                    },
                    # An explicit empty provider inventory is affirmative
                    # evidence that the parameter route is exhausted.
                    "parameter": {
                        "state_id": active_id,
                        "supported_corrections": [],
                        "route_status": "complete_negative",
                    },
                    "topology": {
                        "state_id": active_id,
                        "supported_corrections": [topology_action],
                        "route_status": "actionable",
                    },
                },
                "rejected_hypotheses": (
                    [
                        {
                            "candidate_parent_id": active_id,
                            "source_action": topology_action,
                        }
                    ]
                    if topology_rejected
                    else []
                ),
            }
        )
        return state, measurement_action

    @staticmethod
    def _install_terminal_closure_evidence(
        state: dict,
        closure: dict,
    ) -> None:
        active_id = state["active_state_id"]
        state_hash = "observable-state-hash-s1"
        targets = list(closure["arguments"]["suspect_group"])
        state["fresh_context_evidence"]["measurement"].update(
            {
                "state_hash": state_hash,
                "supported_corrections": [closure],
                "verified_terminal_measurement_closure_targets": targets,
                "verified_terminal_measurement_closure_evidence": {
                    "eligible": True,
                    "state_id": active_id,
                    "state_hash": state_hash,
                    "screening_method": (
                        "singleton_then_grouped_deployment_candidate_quality"
                    ),
                    "new_target": 13,
                    "closure_targets": targets,
                    "attempts": [
                        {
                            "stage": "new_target_singleton",
                            "targets": [13],
                            "disposition": "ACCEPT_PARTIAL",
                            "target_test_passed": True,
                            "physical_constraints_ok": True,
                        },
                        {
                            "stage": "accepted_targets_plus_singleton",
                            "targets": targets,
                            "disposition": "ACCEPT_FINAL",
                            "target_test_passed": True,
                            "globally_resolved": True,
                            "physical_constraints_ok": True,
                        },
                    ],
                },
            }
        )

    def test_partial_measurement_resumes_after_branch_inventories_exhausted(
        self,
    ) -> None:
        state, measurement_action = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )

        actions = ExpertPolicyOracle(
            process_oracle=ProcessValidityOracle(
                executor_hydrated_corrections=True
            )
        ).next_actions(state, [])

        self.assertTrue(actions)
        self.assertEqual(actions[0], measurement_action)

    def test_partial_measurement_remains_blocked_while_branch_candidate_untried(
        self,
    ) -> None:
        state, measurement_action = self._partial_measurement_branch_inventory_state(
            topology_rejected=False
        )

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[measurement_action]
        )

        self.assertNotIn(
            measurement_action, [proposal.action for proposal in proposals]
        )

    def test_provider_verified_terminal_closure_requires_exhausted_branches(
        self,
    ) -> None:
        state, _ = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        active_id = state["active_state_id"]
        closure = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": active_id,
                "suspect_group": [8, 13],
            },
        }
        self._install_terminal_closure_evidence(state, closure)

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[closure]
        )

        proposal = next(item for item in proposals if item.action == closure)
        self.assertIn(
            "provider_verified_terminal_measurement_closure",
            proposal.evidence_codes,
        )

    def test_terminal_closure_new_member_must_carry_residual_signature(self) -> None:
        # Held-out leak r0_680cc8de358a: a provider closure group folded in an
        # unflagged healthy meter (index 64) because editing it resolved the
        # global statistic -- a masking commit.  The single new closure member
        # must itself carry a current residual-outlier signature.
        state, _ = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        active_id = state["active_state_id"]
        closure = {
            "tool": "correct_measurements",
            "arguments": {"state_id": active_id, "suspect_group": [8, 64]},
        }
        self._install_terminal_closure_evidence(state, closure)
        evidence = state["fresh_context_evidence"]["measurement"][
            "verified_terminal_measurement_closure_evidence"
        ]
        evidence["new_target"] = 64
        evidence["attempts"][0]["targets"] = [64]

        proposals = MeasurementExpert().propose(state, [], oracle_hints=[closure])

        self.assertNotIn(closure, [proposal.action for proposal in proposals])

    def test_bare_terminal_closure_targets_do_not_authorize_hint(self) -> None:
        state, _ = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        active_id = state["active_state_id"]
        closure = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": active_id,
                "suspect_group": [8, 13],
            },
        }
        state["fresh_context_evidence"]["measurement"].update(
            {
                "supported_corrections": [closure],
                "verified_terminal_measurement_closure_targets": [8, 13],
            }
        )

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[closure]
        )

        self.assertNotIn(closure, [proposal.action for proposal in proposals])

    def test_terminal_closure_requires_exact_verified_action_signature(self) -> None:
        state, _ = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        active_id = state["active_state_id"]
        closure = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": active_id,
                "suspect_group": [8, 13],
            },
        }
        self._install_terminal_closure_evidence(state, closure)
        extra_payload = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": active_id,
                "suspect_group": [8, 13],
                "measurement_updates": {13: 1.0},
            },
        }

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[extra_payload]
        )

        self.assertNotIn(
            extra_payload, [proposal.action for proposal in proposals]
        )

    def test_terminal_closure_cannot_bypass_live_topology_inventory(self) -> None:
        state, _ = self._partial_measurement_branch_inventory_state(
            topology_rejected=False
        )
        active_id = state["active_state_id"]
        closure = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": active_id,
                "suspect_group": [8, 13],
            },
        }
        self._install_terminal_closure_evidence(state, closure)

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[closure]
        )

        self.assertNotIn(closure, [proposal.action for proposal in proposals])

    def test_inconclusive_empty_parameter_route_authorizes_measurement(self) -> None:
        # V2-A continuation contract: an explicit inconclusive screen that
        # advertises no executable parameter correction closes the route just
        # like a complete negative — there is nothing to try, and on
        # multi-measurement states the branch cross-signals it waits on are
        # caused by the remaining meter errors themselves.
        state, measurement_action = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        state["fresh_context_evidence"]["parameter"]["route_status"] = (
            "unavailable_or_inconclusive"
        )

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[measurement_action]
        )

        self.assertIn(
            measurement_action, [proposal.action for proposal in proposals]
        )

    def test_inconclusive_parameter_route_with_candidates_still_blocks(self) -> None:
        # The closure applies only to empty inventories: an inconclusive route
        # that still advertises an untried parameter candidate stays open and
        # keeps blocking further measurement corrections.
        state, measurement_action = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        active_id = state["active_state_id"]
        state["fresh_context_evidence"]["parameter"] = {
            "state_id": active_id,
            "supported_corrections": [
                {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": active_id, "line_index": 5},
                }
            ],
            "route_status": "unavailable_or_inconclusive",
        }

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[measurement_action]
        )

        self.assertNotIn(
            measurement_action, [proposal.action for proposal in proposals]
        )

    def test_legacy_empty_parameter_contract_still_blocks(self) -> None:
        # An empty inventory without any route_status is a legacy contract that
        # never affirmed the screen completed; it must remain open.
        state, measurement_action = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        del state["fresh_context_evidence"]["parameter"]["route_status"]

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[measurement_action]
        )

        self.assertNotIn(
            measurement_action, [proposal.action for proposal in proposals]
        )

    def test_explicit_null_branch_status_cannot_authorize_measurement(self) -> None:
        state, measurement_action = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        state["fresh_context_evidence"]["topology"]["route_status"] = None

        proposals = MeasurementExpert().propose(
            state, [], oracle_hints=[measurement_action]
        )

        self.assertNotIn(
            measurement_action, [proposal.action for proposal in proposals]
        )

    def test_homogeneous_partial_joint_never_bypasses_branch_inventory(
        self,
    ) -> None:
        state, measurement_action = self._partial_measurement_branch_inventory_state(
            topology_rejected=False
        )
        active_id = state["active_state_id"]
        second_measurement = {
            "tool": "correct_measurements",
            "arguments": {"state_id": active_id, "suspect_group": [14]},
        }
        joint = {
            "tool": "correct_measurements",
            "arguments": {"state_id": active_id, "suspect_group": [8, 13, 14]},
        }
        state["unresolved_signatures"] = [
            "wls_residual_outlier index=8 channel=Vm",
            "wls_residual_outlier index=13 channel=Vm",
            "wls_residual_outlier index=14 channel=Vm",
            "wls_branch_multiplier line_status_or_parameter line=5",
        ]
        state["fresh_context_evidence"]["measurement"][
            "supported_corrections"
        ] = [measurement_action, second_measurement, joint]

        actions = ExpertPolicyOracle(
            process_oracle=ProcessValidityOracle(
                executor_hydrated_corrections=True
            )
        ).next_actions(state, [])

        self.assertTrue(actions)
        self.assertEqual(actions[0]["tool"], "correct_topology")
        self.assertNotIn(
            "correct_measurements", [action["tool"] for action in actions]
        )

        rejected_state, _ = self._partial_measurement_branch_inventory_state(
            topology_rejected=True
        )
        rejected_state["unresolved_signatures"] = list(
            state["unresolved_signatures"]
        )
        rejected_state["fresh_context_evidence"]["measurement"][
            "supported_corrections"
        ] = [measurement_action, second_measurement, joint]

        resumed = ExpertPolicyOracle(
            process_oracle=ProcessValidityOracle(
                executor_hydrated_corrections=True
            )
        ).next_actions(rejected_state, [])

        self.assertTrue(resumed)
        self.assertEqual(resumed[0], measurement_action)
        self.assertNotIn(joint, resumed)

    def test_partial_measurement_commit_does_not_manufacture_parameter_route(self) -> None:
        state = _state([])
        state["accepted_corrections"] = [
            {
                "candidate_state_id": "episode:s1",
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s0",
                        "suspect_group": [12],
                    },
                },
            }
        ]

        self.assertFalse(ParameterExpert().propose(state, []))

    def test_measurement_dominance_suppresses_weak_parameter_cross_signal(self) -> None:
        state = _state(
            [
                "wls_residual_outlier_dominant index=12 channel=Vm",
                "wls_branch_multiplier line_status_or_parameter line=7",
            ]
        )
        state["rejected_hypotheses"] = [
            {
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s1",
                        "suspect_group": [12],
                    },
                }
            }
        ]

        self.assertFalse(ParameterExpert().propose(state, []))

    def test_measurement_dominance_suppresses_weak_topology_cross_signal(self) -> None:
        state = _state(
            [
                "wls_residual_outlier_dominant index=12 channel=Vm",
                "wls_branch_multiplier line_status_or_parameter line=7",
            ]
        )

        self.assertFalse(TopologyExpert().propose(state, []))

    def test_branch_dominance_retains_parameter_route_after_measurement_rejection(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=12 channel=Vm",
                "wls_branch_multiplier_dominant line_status_or_parameter line=7",
            ]
        )
        state["rejected_hypotheses"] = [
            {
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "episode:s1",
                        "suspect_group": [12],
                    },
                }
            }
        ]

        proposals = ParameterExpert().propose(state, [])

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action["tool"], "get_parameter_context")
        self.assertIn("measurement_correction_rejected", proposals[0].evidence_codes)

    def test_old_state_rejections_do_not_reroute_new_branch_dominant_state(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=12 channel=Vm",
                "wls_branch_multiplier_dominant line_status_or_parameter line=7",
            ]
        )
        old_state_id = "episode:s0"
        active_id = state["active_state_id"]
        state["rejected_hypotheses"] = [
            {
                "candidate_parent_id": old_state_id,
                "source_action": {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": active_id, "line_index": 7},
                },
            },
            {
                "candidate_parent_id": active_id,
                "source_action": {
                    "tool": "correct_topology",
                    "arguments": {"state_id": old_state_id, "line_index": 7},
                },
            },
        ]
        state["rejected_hypotheses"].append(
            {
                "candidate_parent_id": old_state_id,
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": active_id,
                        "suspect_group": [12],
                    },
                },
            }
        )

        measurement_proposals = MeasurementExpert().propose(state, [])
        parameter_proposals = ParameterExpert().propose(state, [])

        self.assertNotIn(
            "get_measurement_context",
            [proposal.action["tool"] for proposal in measurement_proposals],
        )
        self.assertTrue(parameter_proposals)
        self.assertNotIn(
            "measurement_correction_rejected",
            parameter_proposals[0].evidence_codes,
        )

    def test_branch_dominance_retains_topology_route(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=12 channel=Vm",
                "wls_branch_multiplier_dominant line_status_or_parameter line=7",
            ]
        )

        proposals = TopologyExpert().propose(state, [])

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].action["tool"], "get_topology_context")

    def test_partial_measurement_does_not_reuse_branch_induced_residual_target(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=64 channel=Qf",
                "wls_branch_multiplier line_status_or_parameter line=6",
            ]
        )
        state["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": "correct_measurements",
                    "arguments": {"state_id": "episode:s0", "suspect_group": [102]},
                }
            }
        ]
        state["has_fresh_measurement_context"] = True
        state["measurement_context_state_id"] = "episode:s1"
        hints = [
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": "episode:s1", "suspect_group": [64]},
            }
        ]

        proposals = MeasurementExpert().propose(state, [], oracle_hints=hints)

        self.assertNotIn(
            "correct_measurements", {proposal.action["tool"] for proposal in proposals}
        )

    def test_partial_branch_does_not_recast_colocated_residual_as_meter_fault(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=42 channel=Pf",
                "wls_branch_multiplier line_status_or_parameter line=1",
            ]
        )
        state["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": "episode:s0", "line_index": 1},
                }
            }
        ]
        state["has_fresh_measurement_context"] = True
        state["measurement_context_state_id"] = "episode:s1"
        hints = [
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": "episode:s1", "suspect_group": [42]},
            }
        ]
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "episode:s1"},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": "episode:s1",
                        "measurement_findings": [
                            {"index0": 42, "channel": "Pf", "channel_offset": 0}
                        ],
                    },
                },
            }
        ]

        proposals = MeasurementExpert().propose(state, history, oracle_hints=hints)
        tools = {proposal.action["tool"] for proposal in proposals}

        self.assertNotIn("correct_measurements", tools)
        self.assertNotIn("get_measurement_context", tools)

    def test_partial_branch_keeps_independent_flow_residual_route(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=47 channel=Pf",
                "wls_branch_multiplier line_status_or_parameter line=1",
            ]
        )
        state["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": "episode:s0", "line_index": 1},
                }
            }
        ]
        state["has_fresh_measurement_context"] = True
        state["measurement_context_state_id"] = "episode:s1"
        hint = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s1", "suspect_group": [47]},
        }
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "episode:s1"},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": "episode:s1",
                        "measurement_findings": [
                            {"index0": 47, "channel": "Pf", "channel_offset": 5}
                        ],
                    },
                },
            }
        ]

        proposals = MeasurementExpert().propose(state, history, oracle_hints=[hint])

        self.assertIn(hint, [proposal.action for proposal in proposals])

    def test_partial_branch_filters_only_colocated_measurement_targets(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=42 channel=Pf",
                "wls_residual_outlier index=47 channel=Pf",
            ]
        )
        state["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": "correct_topology",
                    "arguments": {"state_id": "episode:s0", "line_index": 1},
                }
            }
        ]
        state["has_fresh_measurement_context"] = True
        state["measurement_context_state_id"] = "episode:s1"
        colocated = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s1", "suspect_group": [42]},
        }
        independent = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s1", "suspect_group": [47]},
        }
        grouped = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": "episode:s1",
                "suspect_group": [42, 47],
            },
        }
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "episode:s1"},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": "episode:s1",
                        "measurement_findings": [
                            {"index0": 42, "channel": "Pf", "channel_offset": 0},
                            {"index0": 47, "channel": "Pf", "channel_offset": 5},
                        ],
                    },
                },
            }
        ]

        proposals = MeasurementExpert().propose(
            state,
            history,
            oracle_hints=[colocated, independent, grouped],
        )
        actions = [proposal.action for proposal in proposals]

        self.assertNotIn(colocated, actions)
        self.assertNotIn(grouped, actions)
        self.assertIn(independent, actions)

    def test_malformed_flow_offset_does_not_establish_colocation(self) -> None:
        state = _state(
            [
                "wls_residual_outlier index=42 channel=Pf",
                "wls_branch_multiplier line_status_or_parameter line=1",
            ]
        )
        state["accepted_corrections"] = [
            {
                "source_action": {
                    "tool": "correct_parameters",
                    "arguments": {"state_id": "episode:s0", "line_index": 1},
                }
            }
        ]
        state["has_fresh_measurement_context"] = True
        state["measurement_context_state_id"] = "episode:s1"
        hint = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s1", "suspect_group": [42]},
        }
        history = [
            {
                "action": {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "episode:s1"},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": "episode:s1",
                        "measurement_findings": [
                            {
                                "index0": 42,
                                "channel": "Pf",
                                "channel_offset": "bad",
                            }
                        ],
                    },
                },
            }
        ]

        proposals = MeasurementExpert().propose(
            state, history, oracle_hints=[hint]
        )

        self.assertIn(hint, [proposal.action for proposal in proposals])


class ObservableGlobalProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = {
            "state_id": "episode:s0",
            "case": {"branch": [{"r": 1.0, "x": 2.0}]},
            "measurements": [1.0],
        }
        self.candidate = {
            "state_id": "episode:s1",
            "parent_state_id": "episode:s0",
            "case": {"branch": [{"r": 0.5, "x": 2.0}]},
            "measurements": [1.0],
        }
        self.action = {
            "tool": "correct_parameters",
            "arguments": {"state_id": "episode:s0", "branch_row0": 0, "parameter": "r"},
        }
        self.oracle = CandidateQualityOracle(mode="deployment")

    @staticmethod
    def _topology_states() -> tuple[dict, dict]:
        parent = {
            "state_id": "episode:s0",
            "case": {"branch": [{"r": 1.0, "x": 2.0, "status": 1}]},
            "measurements": [1.0],
        }
        candidate = {
            "state_id": "episode:s1",
            "parent_state_id": "episode:s0",
            "case": {"branch": [{"r": 1.0, "x": 2.0, "status": 0}]},
            "measurements": [1.0],
        }
        return parent, candidate

    def test_strong_target_local_branch_progress_can_be_retained_as_partial(self) -> None:
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=self.action,
            candidate_state=self.candidate,
            verification_output={
                "target_fixed": False,
                "target_progress": 0.92,
                "target_metric_value": 3.59,
                "target_metric_threshold": 3.0,
                "global_progress": 0.90,
                "globally_resolved": False,
                "physical_constraints_ok": True,
            },
        )
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)
        self.assertIn(
            "target_local_progress", result.rationale_codes
        )

    def test_structural_topology_target_rejects_uncleared_moderate_progress(self) -> None:
        parent, candidate = self._topology_states()
        action = {
            "tool": "correct_topology",
            "arguments": {"state_id": "episode:s0", "branch_row0": 0, "status": 0},
        }
        result = self.oracle.label_candidate(
            parent_state=parent,
            source_action=action,
            candidate_state=candidate,
            verification_output={
                "target_fixed": True,
                "target_progress": 1.0,
                "target_metric_kind": "branch_status_mismatch",
                "target_metric_value": 0.0,
                "target_metric_threshold": 0.5,
                "topology_target_status_matches_requested": True,
                "topology_target_branch_multiplier": 18.0,
                "topology_target_branch_multiplier_threshold": 3.0,
                "global_progress": 0.86,
                "globally_resolved": False,
                "physical_constraints_ok": True,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertIn(
            "topology_global_progress_below_structural_threshold",
            result.rationale_codes,
        )

    def test_structural_topology_target_allows_exceptional_global_progress(self) -> None:
        parent, candidate = self._topology_states()
        action = {
            "tool": "correct_topology",
            "arguments": {"state_id": "episode:s0", "branch_row0": 0, "status": 0},
        }
        result = self.oracle.label_candidate(
            parent_state=parent,
            source_action=action,
            candidate_state=candidate,
            verification_output={
                "target_fixed": True,
                "target_progress": 1.0,
                "target_metric_kind": "branch_status_mismatch",
                "target_metric_value": 0.0,
                "target_metric_threshold": 0.5,
                "topology_target_status_matches_requested": True,
                "topology_target_branch_multiplier": 11.6,
                "topology_target_branch_multiplier_threshold": 3.0,
                "global_progress": 0.996,
                "globally_resolved": False,
                "physical_constraints_ok": True,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)

    def test_structural_topology_target_keeps_marginal_multiplier_route(self) -> None:
        parent, candidate = self._topology_states()
        action = {
            "tool": "correct_topology",
            "arguments": {"state_id": "episode:s0", "branch_row0": 0, "status": 0},
        }
        result = self.oracle.label_candidate(
            parent_state=parent,
            source_action=action,
            candidate_state=candidate,
            verification_output={
                "target_fixed": True,
                "target_progress": 1.0,
                "target_metric_kind": "branch_status_mismatch",
                "target_metric_value": 0.0,
                "target_metric_threshold": 0.5,
                "topology_target_status_matches_requested": True,
                "topology_target_branch_multiplier": 3.5,
                "topology_target_branch_multiplier_threshold": 3.0,
                "global_progress": 0.60,
                "globally_resolved": False,
                "physical_constraints_ok": True,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)

    def test_global_progress_alone_cannot_override_explicit_target_failure(self) -> None:
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=self.action,
            candidate_state=self.candidate,
            verification_output={
                "target_fixed": False,
                "target_progress": 0.0,
                "global_progress": 0.99,
                "globally_resolved": False,
                "physical_constraints_ok": True,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertNotIn("target_local_progress", result.rationale_codes)

    def test_weak_partial_global_progress_rejects_wrong_line_target(self) -> None:
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=self.action,
            candidate_state=self.candidate,
            verification_output={
                "target_fixed": True,
                "target_progress": 1.0,
                "global_progress": 0.15,
                "globally_resolved": False,
                "physical_constraints_ok": True,
            },
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertIn(
            "partial_global_progress_below_threshold", result.rationale_codes
        )

    def test_branch_partial_uses_stricter_global_progress_floor(self) -> None:
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=self.action,
            candidate_state=self.candidate,
            verification_output={
                "target_fixed": True,
                "target_progress": 0.79,
                "global_progress": 0.25,
                "globally_resolved": False,
                "physical_constraints_ok": True,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertIn(
            "partial_global_progress_below_threshold", result.rationale_codes
        )

    def test_measurement_partial_keeps_general_global_progress_floor(self) -> None:
        action = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s0", "suspect_group": [0]},
        }
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=action,
            candidate_state={
                "state_id": "episode:s1",
                "parent_state_id": "episode:s0",
                "case": self.parent["case"],
                "measurements": [0.9],
            },
            verification_output={
                "target_fixed": True,
                "target_progress": 0.79,
                "global_progress": 0.25,
                "globally_resolved": False,
                "physical_constraints_ok": True,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)

    def test_nondominant_measurement_cleanup_after_branch_commit_is_rejected(self) -> None:
        action = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s0", "suspect_group": [0]},
        }
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=action,
            candidate_state={
                "state_id": "episode:s1",
                "parent_state_id": "episode:s0",
                "case": self.parent["case"],
                "measurements": [0.9],
            },
            verification_output={
                "target_fixed": True,
                "target_progress": 0.9,
                "global_progress": 0.6,
                "globally_resolved": True,
                "physical_constraints_ok": True,
                "sequential_cross_family_measurement": True,
                "measurement_evidence_dominant": False,
                "measurement_target_branch_colocated": True,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertIn(
            "independent_measurement_evidence_missing", result.rationale_codes
        )

    def test_independent_measurement_cleanup_after_branch_commit_can_progress(self) -> None:
        action = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s0", "suspect_group": [0]},
        }
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=action,
            candidate_state={
                "state_id": "episode:s1",
                "parent_state_id": "episode:s0",
                "case": self.parent["case"],
                "measurements": [0.9],
            },
            verification_output={
                "target_fixed": True,
                "target_progress": 0.9,
                "global_progress": 0.6,
                "globally_resolved": True,
                "physical_constraints_ok": True,
                "sequential_cross_family_measurement": True,
                "measurement_evidence_dominant": False,
                "measurement_target_branch_colocated": False,
                "independent_measurement_target": True,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_FINAL)

    def test_missing_measurement_target_locality_is_inconclusive(self) -> None:
        action = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s0", "suspect_group": [0]},
        }
        result = self.oracle.label_candidate(
            parent_state=self.parent,
            source_action=action,
            candidate_state={
                "state_id": "episode:s1",
                "parent_state_id": "episode:s0",
                "case": self.parent["case"],
                "measurements": [0.9],
            },
            verification_output={
                "target_fixed": True,
                "target_progress": 0.9,
                "global_progress": 0.6,
                "globally_resolved": True,
                "physical_constraints_ok": True,
                "sequential_cross_family_measurement": True,
                "measurement_evidence_dominant": False,
            },
        )

        self.assertEqual(result.disposition, CandidateDisposition.INCONCLUSIVE)
        self.assertIn("measurement_target_locality_missing", result.rationale_codes)

    def _coupled_cluster_case(self, **overrides):
        action = {
            "tool": "correct_measurements",
            "arguments": {"state_id": "episode:s0", "suspect_group": [0]},
        }
        verification = {
            "target_fixed": True,
            "target_progress": 0.9998,
            "global_progress": 0.13,
            "globally_resolved": False,
            "physical_constraints_ok": True,
            "measurement_target_cluster_size": 2,
            "measurement_target_channel": "Pinj",
        }
        verification.update(overrides.pop("verification", {}))
        action["arguments"].update(overrides.pop("arguments", {}))
        oracle = overrides.pop("oracle", self.oracle)
        assert not overrides
        return oracle.label_candidate(
            parent_state=self.parent,
            source_action=action,
            candidate_state={
                "state_id": "episode:s1",
                "parent_state_id": "episode:s0",
                "case": self.parent["case"],
                "measurements": [0.9],
            },
            verification_output=verification,
        )

    def test_coupled_cluster_singleton_passes_halved_floor(self) -> None:
        # Mirrors the measured rejection of true target 16 in
        # r0_81e17a28abbd: local fix 0.9998, global progress 0.13 with two
        # same-channel residual outliers remaining.
        result = self._coupled_cluster_case()
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)
        self.assertEqual(result.progress_class, "coupled_measurement_partial")
        self.assertIn(
            "coupled_same_channel_residual_cluster", result.rationale_codes
        )

    def test_off_cluster_singleton_still_rejected(self) -> None:
        # Mirrors false distractor 84 (channel Pt, cluster of one): the
        # halved floor never applies outside a coherent cluster.
        result = self._coupled_cluster_case(
            verification={"measurement_target_cluster_size": 1}
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "insufficient_global_progress")

    def test_coupled_cluster_below_halved_floor_still_rejected(self) -> None:
        result = self._coupled_cluster_case(verification={"global_progress": 0.08})
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "insufficient_global_progress")

    def test_coupled_relaxation_is_singleton_only(self) -> None:
        result = self._coupled_cluster_case(arguments={"suspect_group": [0, 1]})
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "insufficient_global_progress")

    def test_coupled_relaxation_can_be_disabled(self) -> None:
        oracle = CandidateQualityOracle(
            mode="deployment", coupled_measurement_partial=False
        )
        result = self._coupled_cluster_case(oracle=oracle)
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)

    def test_physical_failure_preempts_coupled_relaxation(self) -> None:
        # Mirrors r0_beddcf84bee5: infeasible candidates stay rejected no
        # matter how coherent the residual cluster looks.
        result = self._coupled_cluster_case(
            verification={"physical_constraints_ok": False}
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "physical_regression")

    _ACCEPTED_CHANNEL_FIELDS = {
        # Mirrors the measured rejection of true target 10 in
        # r0_1f572de0a5e1: the last unmasked error of a chain has no
        # same-channel companion left (cluster of one), but four accepted
        # corrections share its channel and it is the rank-1 residual.
        "measurement_target_cluster_size": 1,
        "measurement_target_channel": "Vm",
        "accepted_measurement_target_count": 4,
        "accepted_measurement_shared_channel": "Vm",
        "measurement_target_rank_one": True,
        "measurement_branch_routes_closed": True,
        "global_progress": 0.197,
    }

    def test_accepted_channel_singleton_passes_halved_floor(self) -> None:
        result = self._coupled_cluster_case(
            verification=dict(self._ACCEPTED_CHANNEL_FIELDS)
        )
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)
        self.assertEqual(
            result.progress_class, "accepted_channel_measurement_partial"
        )
        self.assertIn("accepted_channel_coherent", result.rationale_codes)

    def test_accepted_channel_requires_matching_channel(self) -> None:
        # Mirrors false distractor 38 (Qinj) in the same state: off the
        # accepted channel, so the route never applies.
        result = self._coupled_cluster_case(
            verification={
                **self._ACCEPTED_CHANNEL_FIELDS,
                "measurement_target_channel": "Qinj",
                "global_progress": 0.114,
            }
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "insufficient_global_progress")

    def test_accepted_channel_requires_rank_one(self) -> None:
        result = self._coupled_cluster_case(
            verification={
                **self._ACCEPTED_CHANNEL_FIELDS,
                "measurement_target_rank_one": False,
            }
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)

    def test_accepted_channel_requires_two_accepted_targets(self) -> None:
        result = self._coupled_cluster_case(
            verification={
                **self._ACCEPTED_CHANNEL_FIELDS,
                "accepted_measurement_target_count": 1,
                "accepted_measurement_shared_channel": "Vm",
            }
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)

    def test_accepted_channel_requires_closed_branch_routes(self) -> None:
        result = self._coupled_cluster_case(
            verification={
                **self._ACCEPTED_CHANNEL_FIELDS,
                "measurement_branch_routes_closed": False,
            }
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)

    def test_accepted_channel_route_can_be_disabled(self) -> None:
        oracle = CandidateQualityOracle(
            mode="deployment", accepted_channel_measurement_partial=False
        )
        result = self._coupled_cluster_case(
            oracle=oracle, verification=dict(self._ACCEPTED_CHANNEL_FIELDS)
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)


class MultiMeasurementEndToEndRoutingTests(unittest.TestCase):
    @staticmethod
    def _truth_free(scenario: dict) -> dict:
        scenario = dict(scenario)
        for key in list(scenario):
            if key.startswith("true_") or key.startswith("clean_") or key in {
                "hidden_truth",
                "oracle_action_hints",
            }:
                scenario.pop(key, None)
        return scenario

    @staticmethod
    def _run(seed: int, *, max_steps: int = 24) -> tuple[object, list[dict]]:
        from psse_env.oracle import ExpertPolicyOracle
        from psse_env.providers import MatpowerDeploymentProviders
        from psse_env.providers.scenario_generator import Round0ScenarioGenerator
        from psse_env.transactional_env import TransactionalPSSEEnv

        source = Round0ScenarioGenerator(seed=seed).build(
            {"multi_measurement": 1}
        )[0]
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(),
            production_dataset_mode=True,
            max_steps=int(max_steps),
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(MultiMeasurementEndToEndRoutingTests._truth_free(source))
        actions: list[dict] = []
        for _ in range(int(max_steps)):
            if env.is_terminal():
                break
            proposals = oracle.next_actions(
                env.get_oracle_state(env.history), env.history
            )
            if not proposals:
                break
            action = proposals[0]
            actions.append(action)
            env.step(action)
        return env, actions

    def test_measurement_dominant_episode_never_flips_branch_hypotheses(self) -> None:
        # V2-B episode budget: this seed's five meter errors are repaired one
        # accepted commit at a time, which does not fit the pre-V2 24 steps.
        env, actions = self._run(31, max_steps=40)
        tools = [action["tool"] for action in actions]

        self.assertIn("correct_measurements", tools)
        self.assertNotIn("correct_parameters", tools)
        self.assertNotIn("correct_topology", tools)
        # V2-A continuation contract: after every accepted commit the teacher
        # re-screens the parameter and topology routes on the committed state.
        # Reading those contexts is not a flipped hypothesis; acting on one
        # would be, and that never happens in a measurement-dominant episode.
        for index, tool in enumerate(tools):
            if tool in {"get_parameter_context", "get_topology_context"}:
                self.assertIn(
                    "commit_state",
                    tools[max(0, index - 3) : index],
                    f"branch context screened outside a post-commit re-screen at step {index}",
                )
        self.assertGreaterEqual(tools.count("commit_state"), 1)
        self.assertLess(len(actions), 40)
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        self.assertEqual(tools[-1], "ask_for_more_evidence")
        self.assertEqual(
            actions[-1]["arguments"]["request"],
            "operator_escalation:recovery_options_exhausted",
        )

    def test_partial_commit_continues_recovery_before_safe_handoff(self) -> None:
        env, actions = self._run(20260719)
        tools = [action["tool"] for action in actions]

        # V2-A continuation contract: the first accepted correction no longer
        # ends autonomous recovery.  The teacher re-screens both branch routes
        # on the committed state, takes the next supported measurement
        # correction, and repeats until a fresh same-state context exposes no
        # further supported route — only then does it hand off.
        self.assertGreaterEqual(tools.count("commit_state"), 2)
        first_commit = tools.index("commit_state")
        committed_state = actions[first_commit]["arguments"]["candidate_state_id"]
        window = tools[first_commit + 1 : first_commit + 4]
        self.assertIn("get_parameter_context", window)
        self.assertIn("get_topology_context", window)
        self.assertIn("get_measurement_context", window)
        for offset in range(first_commit + 1, first_commit + 4):
            self.assertEqual(
                actions[offset]["arguments"]["state_id"], committed_state
            )
        self.assertEqual(tools[first_commit + 4], "correct_measurements")
        self.assertEqual(
            actions[first_commit + 4]["arguments"]["state_id"], committed_state
        )
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        self.assertEqual(tools[-1], "ask_for_more_evidence")
        self.assertEqual(
            actions[-1]["arguments"]["request"],
            "operator_escalation:recovery_options_exhausted",
        )
        # The handoff decision is made against fresh evidence, not a stale
        # pre-commit context.
        self.assertEqual(tools[-2], "get_measurement_context")


if __name__ == "__main__":
    unittest.main()
