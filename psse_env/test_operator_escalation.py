from __future__ import annotations

import unittest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    ESTIMATE_HIF_FROM_PATH,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_THREE_PHASE_NLM_FROM_PATH,
)
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.transactional_env import (
    TransactionalPSSEEnv,
    _semantic_correction_signature,
)


def _truth_free(scenario: dict) -> dict:
    scenario = dict(scenario)
    for key in list(scenario):
        if key.startswith(("true_", "clean_")) or key in {
            "hidden_truth",
            "oracle_action_hints",
        }:
            scenario.pop(key, None)
    return scenario


class _UnboundEvidenceProvider:
    provider_kind = "deployment"

    def __call__(self, state):
        return {
            "state_id": "wrong-state",
            "state_hash": "wrong-hash",
            "evidence_source": "deployment_diagnostic:test_unbound",
            "request": "operator_escalation:recovery_options_exhausted",
            "additional_evidence_available": False,
            "operator_review_required": True,
        }


class OperatorEscalationContractTests(unittest.TestCase):
    @staticmethod
    def _environment(seed: int = 20260719):
        scenario = _truth_free(
            Round0ScenarioGenerator(seed=seed).build({"multi_measurement": 1})[0]
        )
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(), production_dataset_mode=True, max_steps=24
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(scenario)
        return env, oracle

    def _joint_fallback_audit_fixture(
        self,
        *,
        second_physical_ok: bool,
        second_violation_index: int | None = None,
    ):
        env, _ = self._environment(seed=31)
        active_id = env.current_state()["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active_id}})
        env.step(
            {"tool": "get_measurement_context", "arguments": {"state_id": active_id}}
        )
        singleton_actions = [
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": active_id, "suspect_group": [target]},
            }
            for target in (67, 69)
        ]
        joint_action = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {"state_id": active_id, "suspect_group": [67, 69]},
        }
        context_metrics = env.history[-1]["tool_output"]["tool_metrics"]
        context_metrics["supported_corrections"] = [
            *singleton_actions,
            joint_action,
        ]
        context_metrics["physical_vm_joint_targets"] = []
        context_metrics["physical_vm_closure_targets"] = []
        context_metrics["coupled_measurement_fallback_targets"] = [67, 69]
        context_metrics["accepted_target_refinement"] = False

        candidate_ids = [f"{active_id}:joint-proof-{offset}" for offset in (1, 2)]
        env.context_flags["rejected_hypotheses"] = [
            {
                "candidate_parent_id": active_id,
                "candidate_state_id": candidate_id,
                "source_action": source_action,
            }
            for candidate_id, source_action in zip(
                candidate_ids, singleton_actions, strict=True
            )
        ]
        for offset, candidate_id in enumerate(candidate_ids):
            physical_ok = offset == 0 or second_physical_ok
            violations = (
                []
                if physical_ok or second_violation_index is None
                else [
                    {
                        "type": "bus_voltage_out_of_bounds",
                        "measurement_index0": second_violation_index,
                    }
                ]
            )
            env.history.append(
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
                            "physical_bound_violations": violations,
                        },
                    },
                }
            )
        handoff = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": active_id,
                "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            },
        }
        return env, handoff, joint_action

    def test_premature_handoff_fails_closed_and_does_not_terminalize(self) -> None:
        env, _ = self._environment()
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": env.current_state()["active_state_id"],
                "request": "operator_escalation:recovery_options_exhausted",
            },
        }

        _, output = env.step(action)

        self.assertEqual(output["execution_status"], "failure")
        self.assertFalse(env.is_terminal())
        self.assertIsNone(env.terminal_outcome)

    def test_failed_hif_estimator_cannot_satisfy_handoff_audit(self) -> None:
        env, _ = self._environment()
        active_id = env.current_state()["active_state_id"]
        active_hash = env.store.state_hash(active_id)
        env.context_flags["unresolved_signatures"] = [
            "hif_suspected_zero_sequence"
        ]
        env.context_flags.setdefault("semantic_field_provenance", {})[
            "unresolved_signatures"
        ] = "deployment_sensor:waveform_capture"
        env.history.extend(
            [
                {
                    "action": {
                        "tool": RUN_THREE_PHASE_NLM_FROM_PATH,
                        "arguments": {"state_id": active_id},
                    },
                    "tool_output": {
                        "execution_status": "success",
                        "tool_metrics": {
                            "state_id": active_id,
                            "state_hash": active_hash,
                            "evidence_source": "deployment_diagnostic:three_phase_nlm",
                            "nlm_summary": {
                                "top_hif_groups": [{"branch_row0": 4}]
                            },
                        },
                    },
                },
                {
                    "action": {
                        "tool": ESTIMATE_HIF_FROM_PATH,
                        "arguments": {
                            "state_id": active_id,
                            "candidate_branch_row0": 4,
                        },
                    },
                    "tool_output": {
                        "execution_status": "failure",
                        "error_code": "hif_estimation_failure",
                        "tool_metrics": {},
                    },
                },
            ]
        )
        handoff = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": active_id,
                "request": HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
            },
        }

        audit = env._operator_escalation_audit(handoff)

        self.assertFalse(audit["sufficient"])
        self.assertIn(
            f"{ESTIMATE_HIF_FROM_PATH}_successful_evidence_missing",
            audit["missing"],
        )
        self.assertEqual(audit["ledger"]["rejected_estimators"], [])
        self.assertFalse(env.is_terminal())

    def test_handoff_rejects_untried_same_state_supported_corrections(self) -> None:
        env, _ = self._environment()
        active_id = env.current_state()["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active_id}})
        _, context_output = env.step(
            {"tool": "get_measurement_context", "arguments": {"state_id": active_id}}
        )
        self.assertTrue(context_output["tool_metrics"]["supported_corrections"])

        _, output = env.step(
            {
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {
                    "state_id": active_id,
                    "request": "operator_escalation:recovery_options_exhausted",
                },
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(
            output["error_code"], "operator_escalation_precondition_not_met"
        )
        self.assertIn(
            "same_state_supported_corrections_unexhausted",
            output["error_detail"],
        )
        self.assertFalse(env.is_terminal())

    def test_safe_low_progress_joint_remains_outstanding_until_tried(self) -> None:
        env, handoff, joint_action = self._joint_fallback_audit_fixture(
            second_physical_ok=True
        )

        before = env._operator_escalation_audit(handoff)

        joint_signature = _semantic_correction_signature(joint_action)
        self.assertFalse(before["sufficient"])
        self.assertIn(
            "same_state_supported_corrections_unexhausted", before["missing"]
        )
        self.assertEqual(
            before["ledger"]["outstanding_recovery_targets"], [joint_signature]
        )
        self.assertNotIn(
            joint_signature,
            before["ledger"]["safety_blocked_recovery_targets"],
        )

        active_id = env.current_state()["active_state_id"]
        env.context_flags["rejected_hypotheses"].append(
            {
                "candidate_parent_id": active_id,
                "candidate_state_id": f"{active_id}:joint-tried",
                "source_action": joint_action,
            }
        )
        after = env._operator_escalation_audit(handoff)

        self.assertTrue(after["sufficient"], after["missing"])
        self.assertEqual(after["ledger"]["outstanding_recovery_targets"], [])

    def test_uncovered_physical_violation_safety_blocks_conditional_joint(self) -> None:
        env, handoff, joint_action = self._joint_fallback_audit_fixture(
            second_physical_ok=False,
            second_violation_index=999,
        )

        audit = env._operator_escalation_audit(handoff)

        joint_signature = _semantic_correction_signature(joint_action)
        self.assertTrue(audit["sufficient"], audit["missing"])
        self.assertEqual(audit["ledger"]["outstanding_recovery_targets"], [])
        self.assertIn(
            joint_signature,
            audit["ledger"]["safety_blocked_recovery_targets"],
        )

    def test_handoff_rejects_unbound_supported_correction_inventory(self) -> None:
        env, _ = self._environment()
        active_id = env.current_state()["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active_id}})
        env.step(
            {"tool": "get_measurement_context", "arguments": {"state_id": active_id}}
        )
        supported = env.history[-1]["tool_output"]["tool_metrics"][
            "supported_corrections"
        ]
        supported[0]["arguments"].pop("state_id")
        env.context_flags["rejected_hypotheses"] = [
            {
                "candidate_parent_id": active_id,
                "source_action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": active_id,
                        "suspect_group": supported[0]["arguments"]["suspect_group"],
                    },
                },
            }
        ]

        _, output = env.step(
            {
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {
                    "state_id": active_id,
                    "request": "operator_escalation:recovery_options_exhausted",
                },
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        self.assertIn("supported_correction_state_unbound", output["error_detail"])
        self.assertFalse(env.is_terminal())

    def test_handoff_rejects_empty_supported_correction_contract(self) -> None:
        env, _ = self._environment()
        active_id = env.current_state()["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active_id}})
        env.step(
            {"tool": "get_measurement_context", "arguments": {"state_id": active_id}}
        )
        env.history[-1]["tool_output"]["tool_metrics"]["supported_corrections"] = [
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": active_id, "suspect_group": []},
            }
        ]

        _, output = env.step(
            {
                "tool": ASK_FOR_MORE_EVIDENCE,
                "arguments": {
                    "state_id": active_id,
                    "request": "operator_escalation:recovery_options_exhausted",
                },
            }
        )

        self.assertEqual(output["execution_status"], "failure")
        self.assertIn("supported_correction_malformed", output["error_detail"])
        self.assertFalse(env.is_terminal())

    def test_measurement_target_signature_canonicalizes_group_order(self) -> None:
        left = _semantic_correction_signature(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": "s0", "suspect_group": [7, 2]},
            }
        )
        right = _semantic_correction_signature(
            {
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {"state_id": "s1", "suspect_group": [2, 7]},
            }
        )

        self.assertEqual(left, right)

    def test_handoff_is_illegal_with_an_open_candidate(self) -> None:
        env, oracle = self._environment(seed=29)
        for _ in range(3):
            action = oracle.next_actions(env.get_oracle_state(env.history), env.history)[0]
            env.step(action)
        self.assertTrue(env.current_state()["has_open_candidate"])
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": env.current_state()["candidate_state_id"],
                "request": "operator_escalation:recovery_options_exhausted",
            },
        }

        _, output = env.step(action)

        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "candidate_lifecycle_violation")
        self.assertFalse(env.is_terminal())

    def test_budget_handoff_closes_before_an_unfinishable_candidate_lifecycle(self) -> None:
        scenario = _truth_free(
            Round0ScenarioGenerator(seed=20260719).build(
                {"multi_measurement": 1}
            )[0]
        )
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01)
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(), production_dataset_mode=True, max_steps=3
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(scenario)

        first = oracle.next_actions(env.get_oracle_state(env.history), env.history)[0]
        self.assertEqual(first["tool"], "run_wls")
        env.step(first)
        second = oracle.next_actions(env.get_oracle_state(env.history), env.history)[0]
        self.assertEqual(second["tool"], "get_measurement_context")
        env.step(second)
        handoff = oracle.next_actions(env.get_oracle_state(env.history), env.history)[0]

        self.assertEqual(handoff["tool"], ASK_FOR_MORE_EVIDENCE)
        self.assertEqual(
            handoff["arguments"]["request"],
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        )
        _, output = env.step(handoff)
        self.assertEqual(output["execution_status"], "success")
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.terminal_outcome, "operator_escalation")

    def test_unbound_inventory_response_cannot_terminalize(self) -> None:
        # Seed 31 is the deterministic safe-handoff episode.  The former
        # default fixture now resolves autonomously, so it cannot exercise an
        # unbound escalation-provider response.
        env, oracle = self._environment(seed=31)
        escalation = None
        for _ in range(24):
            actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
            self.assertTrue(actions)
            if actions[0]["tool"] == ASK_FOR_MORE_EVIDENCE:
                escalation = actions[0]
                break
            env.step(actions[0])
        self.assertIsNotNone(escalation)
        env.evidence_providers[ASK_FOR_MORE_EVIDENCE] = _UnboundEvidenceProvider()

        _, output = env.step(escalation)

        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "insufficient_observable_evidence")
        self.assertFalse(env.is_terminal())
        self.assertIsNone(env.terminal_outcome)


if __name__ == "__main__":
    unittest.main()
