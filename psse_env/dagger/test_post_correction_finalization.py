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
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_WLS,
)
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.termination_expert import TerminationExpert
from psse_env.transactional_env import TransactionalPSSEEnv


FINALIZE = {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
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
    remaining = sum(
        value != expected
        for value, expected in zip(measurements, [1.0], strict=False)
    )
    resolved = remaining == 0
    return {
        "wls_objective": float(remaining),
        "remaining_anomaly_score": float(remaining),
        "anomaly_threshold": 1.0,
        "target_progress": (
            1.0 if state.get("parent_state_id") and resolved else 0.0
        ),
        "global_progress": (
            1.0 if state.get("parent_state_id") and resolved else 0.0
        ),
        "remaining_suspect_count": remaining,
        "globally_resolved": resolved,
        "physical_constraints_ok": True,
        "new_constraint_violations": 0,
        "unresolved_signatures": (
            [] if resolved else ["measurement_residual_outlier"]
        ),
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
            }
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
        "executor_receipt": "post_correction_finalization_regression_v1",
    }


def _production_env() -> TransactionalPSSEEnv:
    return TransactionalPSSEEnv(
        production_dataset_mode=True,
        approved_deterministic_providers=REQUIRED_ADAPTERS,
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
        max_steps=8,
    )


def _uncertified_post_correction_observation() -> dict:
    return {
        "active_state_id": "episode:s1",
        "candidate_state_id": None,
        "has_open_candidate": False,
        "has_unverified_candidate": False,
        "has_verified_candidate": False,
        "accepted_corrections": [
            {
                "candidate_parent_id": "episode:s0",
                "candidate_state_id": "episode:s1",
                "source_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "episode:s0",
                        "measurement_updates": {0: 1.0},
                    },
                },
            }
        ],
        "last_tool": COMMIT_STATE,
        "last_tool_status": "success",
        "last_tool_output": {
            "execution_status": "success",
            "error_code": None,
            "error_detail": None,
        },
        "remaining_anomaly_score": 0.5,
        "no_material_anomaly_remaining": True,
        "unresolved_signatures": [],
        "explained_anomalies": [],
        "semantic_field_provenance": {
            "remaining_anomaly_score": "observable_candidate_verification:wls",
            "no_material_anomaly_remaining": (
                "observable_candidate_verification:wls"
            ),
            "unresolved_signatures": "observable_candidate_verification:wls",
        },
        "history_window": [],
        "remaining_budget": 8,
        "tried_action_signatures": [],
        "rejected_hypotheses": [],
    }


def _clean_wls_observation() -> dict:
    return {
        **_uncertified_post_correction_observation(),
        "active_state_id": "clean:s0",
        "accepted_corrections": [],
        "last_tool": RUN_WLS,
        "remaining_anomaly_score": 0.5,
        "semantic_field_provenance": {
            "remaining_anomaly_score": "observable_wls:clean_control",
            "no_material_anomaly_remaining": "observable_wls:clean_control",
            "unresolved_signatures": "observable_wls:clean_control",
        },
    }


def _all_context_masking_observation() -> dict:
    """Return the public shape shared by a repair and a strong residual mask.

    The numeric values come from the frozen measurement-plus-parameter masking
    probe.  Its mutable measurement vector absorbs the branch mismatch until
    WLS and all three context providers are quiet.  The private remaining fault
    is intentionally absent here: this fixture exercises only evidence the
    production policy is allowed to consume.
    """

    state_id = "masked:active"
    state_hash = "masked-state-sha256"
    return {
        **_uncertified_post_correction_observation(),
        "active_state_id": state_id,
        "accepted_corrections": [
            {
                "candidate_parent_id": "masked:parent",
                "candidate_state_id": state_id,
                "source_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "masked:parent",
                        "suspect_group": [12, 44, 47, 104, 106],
                    },
                },
            }
        ],
        "last_tool": GET_TOPOLOGY_CONTEXT,
        "last_tool_output": {
            "execution_status": "success",
            "error_code": None,
            "error_detail": None,
            "tool_metrics": {
                "state_id": state_id,
                "state_hash": state_hash,
                "evidence_source": (
                    "deployment_context:wls_lagrange_candidate_screened"
                ),
                "context_tool": GET_TOPOLOGY_CONTEXT,
                "finding_count": 0,
                "supported_corrections": [],
                "route_status": "complete_negative",
            },
        },
        "remaining_anomaly_score": 0.7482941461812352,
        "no_material_anomaly_remaining": True,
        "unresolved_signatures": [],
        "last_verification": {
            "state_id": state_id,
            "state_hash": state_hash,
            "evidence_source": "deployment_wls:lagrangian_port",
            "remaining_anomaly_score": 0.7482941461812352,
            "globally_resolved": True,
            "max_normalized_residual": 2.2158524367108154,
            "physical_constraints_ok": True,
            "physical_evidence_complete": True,
            "physical_evidence_scope": "observed_snapshot_topology_vm_rate_a",
            "steady_state_physical_evidence": {
                "scope": "observed_snapshot_topology_vm_rate_a",
                "method": "matpower_case_limits_with_observed_wls_telemetry",
                "complete": True,
                "violation_count": 0,
            },
        },
        "has_fresh_measurement_context": True,
        "has_fresh_parameter_context": True,
        "has_fresh_topology_context": True,
        "measurement_context_state_id": state_id,
        "parameter_context_state_id": state_id,
        "topology_context_state_id": state_id,
        "fresh_context_evidence": {
            "measurement": {
                "context_tool": GET_MEASUREMENT_CONTEXT,
                "context_binding": "direct_context",
                "state_id": state_id,
                "state_hash": state_hash,
                "evidence_source": "deployment_context:wls_residuals",
                "finding_count": 0,
                "measurement_findings": [],
                "supported_corrections": [],
            },
            "parameter": {
                "context_tool": GET_PARAMETER_CONTEXT,
                "context_binding": "direct_context",
                "state_id": state_id,
                "state_hash": state_hash,
                "evidence_source": "deployment_context:wls_lagrange",
                "finding_count": 0,
                "parameter_findings": [],
                "supported_corrections": [],
                "route_status": "complete_negative",
                "route_status_reason": "no_parameter_findings",
            },
            "topology": {
                "context_tool": GET_TOPOLOGY_CONTEXT,
                "context_binding": "direct_context",
                "state_id": state_id,
                "state_hash": state_hash,
                "evidence_source": (
                    "deployment_context:wls_lagrange_candidate_screened"
                ),
                "finding_count": 0,
                "topology_findings": [],
                "supported_corrections": [],
                "route_status": "complete_negative",
                "route_status_reason": "no_topology_findings",
            },
        },
        "semantic_field_provenance": {
            "remaining_anomaly_score": "deployment_wls:lagrangian_port",
            "no_material_anomaly_remaining": "deployment_wls:lagrangian_port",
            "unresolved_signatures": "deployment_wls:lagrangian_port",
        },
    }


class PostCorrectionFinalizationRegressionTests(unittest.TestCase):
    """Statistical quiescence after correction is not a release certificate."""

    def test_uncertified_post_correction_pattern_is_not_terminal(self) -> None:
        # Some physically correct corrected roots share this exact public
        # pattern.  Without an additional independent observable certificate,
        # the production policy must therefore fail closed for both rather
        # than use the score as a hidden-truth proxy.
        state = _uncertified_post_correction_observation()

        proposals = TerminationExpert().propose(state)
        process = ProcessValidityOracle().check(state, FINALIZE)
        expert_actions = ExpertPolicyOracle().next_actions(state, [])

        self.assertNotIn(
            FINALIZE_DIAGNOSIS,
            {proposal.action["tool"] for proposal in proposals},
        )
        self.assertFalse(process["process_valid"], process)
        self.assertEqual(process["error_code"], "terminal_condition_not_met")
        self.assertNotIn(
            FINALIZE_DIAGNOSIS,
            {action["tool"] for action in expert_actions},
        )

    def test_confirmation_marker_blocks_off_policy_autonomous_correction(
        self,
    ) -> None:
        state_id = "episode:s1"
        correction = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {
                "state_id": state_id,
                "suspect_group": [65],
            },
        }
        state = {
            **_uncertified_post_correction_observation(),
            "active_state_id": state_id,
            "no_material_anomaly_remaining": False,
            "unresolved_signatures": [
                POST_CORRECTION_CONFIRMATION_SIGNATURE
            ],
            "has_fresh_measurement_context": True,
            "measurement_context_state_id": state_id,
            "fresh_context_evidence": {
                "measurement": {
                    "state_id": state_id,
                    "supported_corrections": [correction],
                }
            },
        }
        process = ProcessValidityOracle(
            executor_hydrated_corrections=True
        )

        blocked = process.check(state, correction)

        self.assertFalse(blocked["process_valid"], blocked)
        self.assertEqual(
            blocked["error_code"],
            "post_correction_confirmation_required",
        )
        self.assertEqual(
            blocked["valid_next_actions"],
            [
                {
                    "tool": ASK_FOR_MORE_EVIDENCE,
                    "arguments": {
                        "state_id": state_id,
                        "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                    },
                }
            ],
        )

        no_confirmation_context = {
            **state,
            "has_fresh_measurement_context": False,
            "measurement_context_state_id": None,
        }
        off_policy_parameter = {
            "tool": CORRECT_PARAMETERS,
            "arguments": {"state_id": state_id, "line_index": 5},
        }
        canonical_repair = process.check(
            no_confirmation_context,
            off_policy_parameter,
        )
        self.assertFalse(canonical_repair["process_valid"], canonical_repair)
        self.assertEqual(
            canonical_repair["valid_next_actions"],
            [
                {
                    "tool": GET_MEASUREMENT_CONTEXT,
                    "arguments": {"state_id": state_id},
                }
            ],
        )

        # A substantive anomaly signature still permits the exact
        # context-supported sequential correction even when the controller
        # marker remains appended.  The marker is a handoff obligation only
        # when it is the sole unresolved signature.
        active_recovery = {
            **state,
            "unresolved_signatures": [
                POST_CORRECTION_CONFIRMATION_SIGNATURE,
                "measurement_residual_outlier",
            ],
        }
        allowed = process.check(active_recovery, correction)
        self.assertTrue(allowed["process_valid"], allowed)

        # The marker is a reserved controller obligation, but its literal
        # string alone cannot activate the post-commit guard without an
        # accepted correction ledger.
        no_accepted_correction = {
            **state,
            "accepted_corrections": [],
        }
        unarmed = process.check(no_accepted_correction, correction)
        self.assertTrue(unarmed["process_valid"], unarmed)

    def test_confirmation_guard_failure_is_not_counted_as_exhausted_target(
        self,
    ) -> None:
        env = _production_env()
        root = env.reset(
            {
                "scenario_id": "confirmation-process-failure-audit",
                "case": {},
                "measurements": [1.0],
                "clean_measurements": [1.0],
                "true_measurement_errors": [],
            }
        )
        state_id = root["active_state_id"]
        state_hash = env.store.get_state(state_id)["state_hash"]
        correction = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {"state_id": state_id, "suspect_group": [0]},
        }
        env.context_flags.update(
            {
                "accepted_corrections": [
                    {
                        "candidate_parent_id": "parent:s0",
                        "candidate_state_id": state_id,
                        "source_action": {
                            "tool": CORRECT_MEASUREMENTS,
                            "arguments": {
                                "state_id": "parent:s0",
                                "suspect_group": [0],
                            },
                        },
                    }
                ],
                "unresolved_signatures": [
                    POST_CORRECTION_CONFIRMATION_SIGNATURE
                ],
                "remaining_anomaly_score": 0.5,
                "no_material_anomaly_remaining": False,
            }
        )
        env.context_flags["semantic_field_provenance"][
            "unresolved_signatures"
        ] = "controller_default:post_correction_resolution_confirmation_required"
        env.history = [
            {
                "action": {
                    "tool": RUN_WLS,
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": state_id,
                        "state_hash": state_hash,
                        "evidence_source": "deployment_wls:test",
                    },
                },
            },
            {
                "action": {
                    "tool": GET_MEASUREMENT_CONTEXT,
                    "arguments": {"state_id": state_id},
                },
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "state_id": state_id,
                        "state_hash": state_hash,
                        "evidence_source": "deployment_context:test",
                        "supported_corrections": [correction],
                    },
                },
            },
            {
                "action": correction,
                "tool_output": {
                    "execution_status": "failure",
                    "error_code": "post_correction_confirmation_required",
                    "error_detail": (
                        "measurement_autonomous_correction_blocked_for_operator_review"
                    ),
                },
            },
        ]

        audit = env._operator_escalation_audit(
            {
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {
                    "state_id": state_id,
                    "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                },
            }
        )

        # The fixture intentionally omits the external escalation provider;
        # this assertion targets the independent environment history ledger.
        self.assertIn("operator_escalation_provider_missing", audit["missing"])
        ledger = audit["ledger"]
        self.assertEqual(ledger["supported_recovery_target_count"], 1)
        self.assertEqual(ledger["exhausted_recovery_target_count"], 0)
        self.assertEqual(len(ledger["safety_blocked_recovery_targets"]), 1)
        self.assertEqual(ledger["outstanding_recovery_targets"], [])

    def test_all_current_contexts_do_not_certify_a_masked_correction(self) -> None:
        state = _all_context_masking_observation()

        # This is the strongest closure pattern the current providers can
        # produce: quiescent WLS, a complete passing snapshot check, no meter
        # finding, and complete-negative parameter and topology inventories.
        # All are nevertheless derived from the same mutable case/measurement
        # snapshot, so state/hash freshness does not make them independent.
        self.assertTrue(state["last_verification"]["globally_resolved"])
        self.assertTrue(state["last_verification"]["physical_constraints_ok"])
        self.assertTrue(state["last_verification"]["physical_evidence_complete"])
        self.assertEqual(
            state["fresh_context_evidence"]["measurement"]["finding_count"],
            0,
        )
        for family in ("parameter", "topology"):
            evidence = state["fresh_context_evidence"][family]
            self.assertEqual(evidence["route_status"], "complete_negative")
            self.assertEqual(evidence["supported_corrections"], [])

        proposals = TerminationExpert().propose(state)
        process = ProcessValidityOracle().check(state, FINALIZE)
        expert_actions = ExpertPolicyOracle().next_actions(state, [])

        self.assertNotIn(
            FINALIZE_DIAGNOSIS,
            {proposal.action["tool"] for proposal in proposals},
        )
        self.assertFalse(process["process_valid"], process)
        self.assertEqual(process["error_code"], "terminal_condition_not_met")
        self.assertNotIn(
            FINALIZE_DIAGNOSIS,
            {action["tool"] for action in expert_actions},
        )

    def test_public_transition_flow_blocks_uncertified_post_correction_finalization(
        self,
    ) -> None:
        env = _production_env()
        root = env.reset(
            {
                "scenario_id": "post-correction-finalization-regression",
                "case": {},
                "measurements": [9.0],
                "clean_measurements": [1.0],
                "true_measurement_errors": [{"index": 0, "clean": 1.0}],
            }
        )
        active_id = root["active_state_id"]
        env.step({"tool": RUN_WLS, "arguments": {"state_id": active_id}})
        env.step(
            {
                "tool": GET_MEASUREMENT_CONTEXT,
                "arguments": {"state_id": active_id},
            }
        )
        candidate_state, correction_output = env.step(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": active_id,
                    "measurement_updates": {0: 1.0},
                },
            }
        )
        self.assertEqual(correction_output["execution_status"], "success")
        candidate_id = candidate_state["candidate_state_id"]
        env.step(
            {"tool": RUN_WLS, "arguments": {"state_id": candidate_id}}
        )
        committed_state, commit_output = env.step(
            {
                "tool": COMMIT_STATE,
                "arguments": {"candidate_state_id": candidate_id},
            }
        )

        self.assertEqual(commit_output["execution_status"], "success")
        self.assertTrue(committed_state["accepted_corrections"])
        self.assertEqual(committed_state["last_tool"], COMMIT_STATE)
        self.assertEqual(committed_state["last_tool_status"], "success")
        # Candidate WLS quiescence is current evidence, but after the
        # correction is committed it is not by itself a strict release
        # certificate.  The public controller persists an explicit recovery
        # obligation instead of turning private truth into a policy feature.
        self.assertFalse(committed_state["no_material_anomaly_remaining"])
        self.assertLess(committed_state["remaining_anomaly_score"], 1.0)
        self.assertIn(
            "post_correction_resolution_confirmation_required:measurement_context",
            committed_state["unresolved_signatures"],
        )
        self.assertNotIn(
            "oracle_terminal_eligible",
            env.get_policy_observation().as_dict(),
        )

        process = env.process_oracle.check(
            committed_state,
            FINALIZE,
            store=env.store,
        )
        expert_actions = ExpertPolicyOracle(
            process_oracle=env.process_oracle
        ).next_actions(env.get_policy_observation(), env.history)

        self.assertFalse(process["process_valid"], process)
        self.assertNotIn(
            FINALIZE_DIAGNOSIS,
            {action["tool"] for action in expert_actions},
        )
        with self.assertRaises(ValueError):
            env.assert_training_decision_evidence(FINALIZE)

    def test_clean_no_correction_wls_remains_terminal(self) -> None:
        state = _clean_wls_observation()

        proposals = TerminationExpert().propose(state)
        process = ProcessValidityOracle().check(state, FINALIZE)
        expert_actions = ExpertPolicyOracle().next_actions(state, [])

        self.assertEqual(proposals[0].action["tool"], FINALIZE_DIAGNOSIS)
        self.assertTrue(process["process_valid"], process)
        self.assertEqual(expert_actions[0]["tool"], FINALIZE_DIAGNOSIS)

        env = _production_env()
        root = env.reset(
            {
                "scenario_id": "clean-finalization-control",
                "case": {},
                "measurements": [1.0],
                "clean_measurements": [1.0],
                "true_measurement_errors": [],
            }
        )
        env.step(
            {
                "tool": RUN_WLS,
                "arguments": {"state_id": root["active_state_id"]},
            }
        )
        env.assert_training_decision_evidence(FINALIZE)

    def test_explanation_only_no_correction_control_remains_terminal(self) -> None:
        state = {
            **_clean_wls_observation(),
            "remaining_anomaly_score": 3.5,
            "no_material_anomaly_remaining": False,
            "unresolved_signatures": ["harmonic_distortion_detected"],
            "explained_anomalies": [
                {
                    "tool": "run_hse_from_path",
                    "family": "harmonic",
                    "kind": "harmonic_source_localized",
                    "evidence_source": (
                        "deployment_diagnostic:harmonic_state_estimation"
                    ),
                    "explained_signatures": [
                        "harmonic_distortion_detected"
                    ],
                }
            ],
            "last_tool": "run_hse_from_path",
        }

        proposals = TerminationExpert().propose(state)
        process = ProcessValidityOracle().check(state, FINALIZE)
        expert_actions = ExpertPolicyOracle().next_actions(state, [])

        self.assertEqual(proposals[0].action["tool"], FINALIZE_DIAGNOSIS)
        self.assertTrue(process["process_valid"], process)
        self.assertEqual(expert_actions[0]["tool"], FINALIZE_DIAGNOSIS)

    def test_diagnostic_explanation_does_not_certify_prior_corrections(self) -> None:
        state = {
            **_uncertified_post_correction_observation(),
            "remaining_anomaly_score": 3.5,
            "no_material_anomaly_remaining": False,
            "unresolved_signatures": ["harmonic_distortion_detected"],
            "explained_anomalies": [
                {
                    "tool": "run_hse_from_path",
                    "family": "harmonic",
                    "kind": "harmonic_source_localized",
                    "evidence_source": (
                        "deployment_diagnostic:harmonic_state_estimation"
                    ),
                    "explained_signatures": [
                        "harmonic_distortion_detected"
                    ],
                }
            ],
            "last_tool": "run_hse_from_path",
        }

        proposals = TerminationExpert().propose(state)
        process = ProcessValidityOracle().check(state, FINALIZE)
        expert_actions = ExpertPolicyOracle().next_actions(state, [])

        self.assertNotIn(
            FINALIZE_DIAGNOSIS,
            {proposal.action["tool"] for proposal in proposals},
        )
        self.assertFalse(process["process_valid"], process)
        self.assertNotIn(
            FINALIZE_DIAGNOSIS,
            {action["tool"] for action in expert_actions},
        )


if __name__ == "__main__":
    unittest.main()
