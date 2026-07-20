from __future__ import annotations

import unittest

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
    def _run(seed: int) -> tuple[object, list[dict]]:
        from psse_env.oracle import ExpertPolicyOracle
        from psse_env.providers import MatpowerDeploymentProviders
        from psse_env.providers.scenario_generator import Round0ScenarioGenerator
        from psse_env.transactional_env import TransactionalPSSEEnv

        source = Round0ScenarioGenerator(seed=seed).build(
            {"multi_measurement": 1}
        )[0]
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(), production_dataset_mode=True, max_steps=24
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(MultiMeasurementEndToEndRoutingTests._truth_free(source))
        actions: list[dict] = []
        for _ in range(24):
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
        env, actions = self._run(31)
        tools = [action["tool"] for action in actions]

        self.assertIn("correct_measurements", tools)
        self.assertNotIn("correct_parameters", tools)
        self.assertNotIn("correct_topology", tools)
        self.assertNotIn("get_parameter_context", tools)
        self.assertNotIn("get_topology_context", tools)
        self.assertLess(len(actions), 24)
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "operator_escalation")
        self.assertEqual(tools[-1], "ask_for_more_evidence")
        self.assertEqual(
            actions[-1]["arguments"]["request"],
            "operator_escalation:recovery_options_exhausted",
        )

    def test_partial_commit_refreshes_measurement_context_before_resolution(self) -> None:
        env, actions = self._run(20260719)
        tools = [action["tool"] for action in actions]

        commit_index = tools.index("commit_state")
        self.assertEqual(tools[commit_index + 1], "get_measurement_context")
        self.assertEqual(
            actions[commit_index + 1]["arguments"]["state_id"],
            actions[commit_index]["arguments"]["candidate_state_id"],
        )
        self.assertNotIn("get_parameter_context", tools)
        self.assertNotIn("get_topology_context", tools)
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "resolved")
        self.assertEqual(tools[-1], "finalize_diagnosis")


if __name__ == "__main__":
    unittest.main()
