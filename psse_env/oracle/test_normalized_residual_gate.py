from __future__ import annotations

import unittest

from psse_env.actions import POST_CORRECTION_CONFIRMATION_SIGNATURE
from psse_env.dagger.test_post_correction_finalization import _production_env
from psse_env.oracle.anomaly_evidence import normalized_residual_alarm
from psse_env.oracle.candidate_quality import CandidateDisposition, CandidateQualityOracle
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.oracle.process_validity import ProcessValidityOracle, post_correction_confirmation_required
from psse_env.oracle.termination_expert import TerminationExpert
from psse_env.transactional_env import TransactionalPSSEEnv


def _state(*, alarm: bool = True) -> dict:
    return {
        "active_state_id": "episode:s1",
        "remaining_anomaly_score": 0.98,
        "no_material_anomaly_remaining": True,
        "fresh_context_evidence": {
            "wls": {
                "state_id": "episode:s1",
                "state_hash": "fixture-current-state",
                "evidence_source": "deployment_wls:dual_alarm_fixture",
                "successful": True,
                "normalized_residual_alarm": alarm,
                "normalized_residual_threshold": 4.0,
                "max_normalized_residual": 9.3776 if alarm else 2.0,
            }
        },
    }


def _dual_wls(state):
    candidate = bool(state.get("parent_state_id"))
    return {
        "evidence_source": "deployment_wls:dual_alarm_fixture",
        "chi_square_statistic": 400.0 if candidate else 800.0,
        "chi_square_threshold": 444.889,
        "remaining_anomaly_score": 2.25 if candidate else 2.5,
        "anomaly_threshold": 1.0,
        "normalized_residual_alarm": True,
        "normalized_residual_threshold": 4.0,
        "max_normalized_residual": 9.0 if candidate else 10.0,
        "chi_square_alarm": not candidate,
        "post_action_resolved": False,
        "no_material_anomaly_remaining": False,
        "globally_resolved": False,
        "target_fixed": candidate,
        "target_progress": 0.9 if candidate else 0.0,
        # Simulate a provider using the composite score for progress. The
        # controller must instead preserve the established J-based metric.
        "global_progress": 0.1 if candidate else 0.0,
        "physical_constraints_ok": True,
        "new_constraint_violations": 0,
        "converged": True,
        "unresolved_signatures": ["wls_residual_outlier index=1"],
    }


_dual_wls.provider_kind = "deterministic"


class NormalizedResidualGateTests(unittest.TestCase):
    def test_local_alarm_blocks_initial_clean_score_finalization(self):
        state = _state()
        self.assertFalse(ProcessValidityOracle()._terminal_condition_met(state))
        self.assertEqual(TerminationExpert().propose(state), [])
        state["fresh_context_evidence"]["wls"].update(
            normalized_residual_alarm=False, max_normalized_residual=2.0,
        )
        self.assertTrue(ProcessValidityOracle()._terminal_condition_met(state))

    def test_only_explicit_current_successful_alarm_enables_new_gate(self):
        state = _state()
        self.assertTrue(normalized_residual_alarm(state))
        state["fresh_context_evidence"]["wls"]["state_id"] = "old:s0"
        self.assertFalse(normalized_residual_alarm(state))
        state["fresh_context_evidence"]["wls"]["state_id"] = state["active_state_id"]
        state["fresh_context_evidence"]["wls"]["successful"] = False
        self.assertFalse(normalized_residual_alarm(state))
        self.assertFalse(normalized_residual_alarm({
            "max_normalized_residual": 9.4,
            "unresolved_signatures": ["wls_residual_outlier index=97"],
        }))

    def test_local_alarm_prevents_confirmation_handoff_priority(self):
        state = _state()
        state.update({
            "accepted_corrections": [{"candidate_state_id": state["active_state_id"]}],
            "unresolved_signatures": [POST_CORRECTION_CONFIRMATION_SIGNATURE],
            "has_fresh_measurement_context": True,
            "measurement_context_state_id": state["active_state_id"],
            "semantic_field_provenance": {
                "remaining_anomaly_score": "deployment_wls:test",
            },
        })
        self.assertFalse(post_correction_confirmation_required(state))
        self.assertEqual(
            ExpertPolicyOracle()._post_correction_confirmation_handoff_proposals(state, []),
            [],
        )
        state["fresh_context_evidence"]["wls"].update(
            normalized_residual_alarm=False, max_normalized_residual=2.0,
        )
        self.assertTrue(post_correction_confirmation_required(state))
        self.assertTrue(
            ExpertPolicyOracle()._post_correction_confirmation_handoff_proposals(state, [])
        )
        # Score provenance alone cannot certify an incomplete durable WLS
        # record after the original event has left the bounded history.
        state["fresh_context_evidence"]["wls"].pop("state_hash")
        self.assertEqual(
            ExpertPolicyOracle()._post_correction_confirmation_handoff_proposals(state, []),
            [],
        )

    def test_runtime_wls_ledger_supplies_the_binding_used_by_handoff(self):
        env = _production_env()
        env.wls_runner = _dual_wls
        state = env.reset({"scenario_id": "bound-dual-alarm", "case": {}, "measurements": [9.0, 5.0]})
        active = state["active_state_id"]
        _, output = env.step({"tool": "run_wls", "arguments": {"state_id": active}})
        self.assertEqual(output["execution_status"], "success", output)
        evidence = env.get_policy_observation().fresh_context_evidence["wls"]
        self.assertTrue(evidence["successful"])
        self.assertEqual(evidence["state_id"], active)
        self.assertEqual(evidence["state_hash"], env.store.state_hash(active))
        self.assertEqual(evidence["evidence_source"], "deployment_wls:dual_alarm_fixture")
        self.assertTrue(normalized_residual_alarm(env.get_policy_observation()))

    def test_residual_alarm_vetoes_contradictory_final_candidate_flags(self):
        metrics = {
            "normalized_residual_alarm": True,
            "post_action_resolved": True,
            "globally_resolved": True,
            "remaining_anomaly_score": 0.98,
            "anomaly_threshold": 1.0,
            "target_fixed": True,
            "physical_constraints_ok": True,
        }
        oracle = CandidateQualityOracle(mode="deployment")
        self.assertFalse(oracle._solve_resolved(metrics))
        self.assertFalse(oracle._global_resolved(metrics))
        self.assertEqual(
            TransactionalPSSEEnv._target_decision_evidence_missing(metrics, "ACCEPT_FINAL"),
            ["final_resolution_evidence_missing"],
        )
        self.assertEqual(
            TransactionalPSSEEnv._target_decision_evidence_missing(metrics, "ACCEPT_PARTIAL"),
            [],
        )

    def test_partial_repair_keeps_j_progress_and_does_not_enter_confirmation(self):
        env = _production_env()
        env.wls_runner = _dual_wls
        root = env.reset({"scenario_id": "dual-alarm", "case": {}, "measurements": [9.0, 5.0]})
        state_id = root["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": state_id}})
        env.step({"tool": "get_measurement_context", "arguments": {"state_id": state_id}})
        candidate, output = env.step({
            "tool": "correct_measurements",
            "arguments": {"state_id": state_id, "measurement_updates": {0: 1.0}},
        })
        self.assertEqual(output["execution_status"], "success")
        candidate_id = candidate["candidate_state_id"]
        _, verification = env.step({"tool": "run_wls", "arguments": {"state_id": candidate_id}})
        self.assertEqual(verification["execution_status"], "success", verification)
        metrics = verification["tool_metrics"]
        self.assertAlmostEqual(metrics["global_progress"], 0.5)
        self.assertEqual(
            env.store.get_state(candidate_id)["candidate_disposition"],
            CandidateDisposition.ACCEPT_PARTIAL.value,
        )
        committed, output = env.step({
            "tool": "commit_state", "arguments": {"candidate_state_id": candidate_id},
        })
        self.assertEqual(output["execution_status"], "success", output)
        self.assertFalse(committed["no_material_anomaly_remaining"])
        self.assertNotIn(POST_CORRECTION_CONFIRMATION_SIGNATURE, committed["unresolved_signatures"])
        self.assertTrue(normalized_residual_alarm(env.get_policy_observation()))


if __name__ == "__main__":
    unittest.main()
