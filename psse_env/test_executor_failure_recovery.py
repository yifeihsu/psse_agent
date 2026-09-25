"""Numerical correction failures survive bounded history without faking verification."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from psse_env.actions import action_signature, ambiguous_branch_candidate_lines
from psse_env.dagger.release_factories import select_observable_expert_actions
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.measurement_expert import MeasurementExpert
from psse_env.oracle.measurement_recovery_evidence import eligible_joint_measurement_targets
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.transactional_env import TransactionalPSSEEnv


ACTIVE = "bounded:s1"
STATE_HASH = "bounded-state-hash"


def _parameter(line, state_id=ACTIVE):
    return {"tool": "correct_parameters", "arguments": {"state_id": state_id, "line_index": line}}


def _failure_receipt(action, state_hash=STATE_HASH):
    return {
        "candidate_parent_id": action["arguments"]["state_id"],
        "source_action": deepcopy(action),
        "action_signature": action_signature(action),
        "rejection_kind": "executor_failure",
        "execution_status": "failure",
        "error_code": {
            "correct_parameters": "parameter_correction_failure",
            "correct_measurements": "measurement_correction_failure",
            "correct_topology": "topology_correction_failure",
        }[action["tool"]],
        "state_hash": state_hash,
        "evidence_source": "controller_observed:correction_executor_failure",
    }


def _minimal_observation():
    meter = {"tool": "correct_measurements", "arguments": {"state_id": ACTIVE, "suspect_group": [40]}}
    parameter_actions = [_parameter(line) for line in (12, 19, 20, 13)]
    fresh = {}
    for family, actions, status in (
        ("measurement", [meter], "actionable"),
        ("parameter", parameter_actions, "actionable"),
        ("topology", [], "complete_negative"),
    ):
        fresh[family] = {
            "state_id": ACTIVE, "state_hash": STATE_HASH,
            "evidence_source": "deployment_context:test",
            "supported_corrections": actions, "route_status": status,
        }
    for family in ("harmonic", "three_phase"):
        fresh[family] = {
            "state_id": ACTIVE, "state_hash": STATE_HASH,
            "evidence_source": "deployment_context:test",
            "request_attempted": True, "available_evidence_channels": [],
            f"{family}_context_status": "unavailable",
            "carried_from_state_id": "bounded:s0",
        }
    fresh["wls"] = {
        "state_id": ACTIVE, "state_hash": STATE_HASH,
        "successful": True, "anomalous": True,
        "evidence_source": "deployment_wls:test",
        "normalized_residual_alarm": True,
        "max_normalized_residual": 19.0, "normalized_residual_threshold": 4.0,
    }
    state = {
        "active_state_id": ACTIVE, "candidate_state_id": None,
        "has_open_candidate": False, "no_material_anomaly_remaining": False,
        "last_tool": "rollback_state", "last_tool_status": "success",
        "last_tool_output": {"execution_status": "success"},
        "remaining_anomaly_score": 4.9, "remaining_budget": 15,
        "unresolved_signatures": [
            "wls_residual_outlier_dominant index=40 channel=Qinj",
            "wls_branch_multiplier line_status_or_parameter line=12",
        ],
        "semantic_field_provenance": {
            "remaining_anomaly_score": "deployment_wls:test",
            "unresolved_signatures": "deployment_wls:test",
        },
        "accepted_corrections": [{"source_action": {
            "tool": "correct_measurements",
            "arguments": {"state_id": "bounded:s0", "suspect_group": [39]},
        }}],
        "rejected_hypotheses": [{
            "candidate_parent_id": ACTIVE,
            "candidate_state_id": f"bounded:rejected-{line}",
            "source_action": _parameter(line),
        } for line in (12, 13, 20)],
        "fresh_context_evidence": fresh,
        "requires_measurement_context": True,
        "tried_action_signatures": [
            *[action_signature(action) for action in parameter_actions],
            action_signature({"tool": "run_wls", "arguments": {"state_id": ACTIVE}}),
        ],
        # The earlier line19 failure is intentionally outside this window.
        "history_window": [],
    }
    for family in ("measurement", "parameter", "topology"):
        state[f"has_fresh_{family}_context"] = True
        state[f"{family}_context_state_id"] = ACTIVE
    return state, meter


def _select(state):
    oracle = ExpertPolicyOracle(
        process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True)
    )
    return select_observable_expert_actions(policy_observation=state, expert_oracle=oracle).actions


class ExecutorFailureRecoveryTests(unittest.TestCase):
    def test_bounded_observation_resumes_meter_only_with_structured_failure_receipt(self):
        state, meter = _minimal_observation()
        # A ledger entry alone is not a test: without a durable same-state
        # receipt the line-19 attempt (outside the window) may have been a
        # process-gate refusal, and the escalation audit still counts line 19
        # as an outstanding supported correction, so the expert retries it
        # instead of labelling a handoff the audit would reject.
        self.assertEqual(_select(state)[0], _parameter(19))
        state["rejected_hypotheses"].append(_failure_receipt(_parameter(19)))
        self.assertEqual(_select(state)[0], meter)

    def test_old_state_failure_does_not_close_current_parameter_inventory(self):
        state, _ = _minimal_observation()
        state["rejected_hypotheses"].append(_failure_receipt(_parameter(19, "bounded:s0")))
        self.assertFalse(MeasurementExpert._branch_recovery_routes_exhausted(
            state, [], active_id=ACTIVE
        ))

    def test_executor_failure_is_not_verified_ambiguous_branch_evidence(self):
        state, _ = _minimal_observation()
        state["fresh_context_evidence"]["parameter"].update(
            parameter_ranking_ambiguous=True, parameter_ranking_candidate_lines=[12, 19]
        )
        state["rejected_hypotheses"].append(_failure_receipt(_parameter(19)))
        self.assertIsNone(ambiguous_branch_candidate_lines(state))
        state["rejected_hypotheses"][-1] = {
            "candidate_parent_id": ACTIVE, "candidate_state_id": "bounded:verified19",
            "source_action": _parameter(19),
        }
        self.assertEqual(ambiguous_branch_candidate_lines(state), [12, 19])

    def test_executor_failure_cannot_open_branch_dominant_meter_route(self):
        state, _ = _minimal_observation()
        state["unresolved_signatures"] = ["wls_branch_multiplier_dominant line=12"]
        topology = {"tool": "correct_topology", "arguments": {"state_id": ACTIVE, "line_index": 12}}
        state["rejected_hypotheses"] = [
            _failure_receipt(_parameter(12)),
            {"candidate_parent_id": ACTIVE, "candidate_state_id": "bounded:verified-topology", "source_action": topology},
        ]
        self.assertTrue(MatpowerDeploymentProviders._branch_dominance_block({"policy_observation": state}))
        state["rejected_hypotheses"][0] = {
            "candidate_parent_id": ACTIVE, "candidate_state_id": "bounded:verified-parameter",
            "source_action": _parameter(12),
        }
        self.assertFalse(MatpowerDeploymentProviders._branch_dominance_block({"policy_observation": state}))

    def test_failure_only_records_cannot_supply_verified_joint_meter_proof(self):
        actions = [{"tool": "correct_measurements", "arguments": {"state_id": ACTIVE, "suspect_group": group}}
                   for group in ([67], [69], [67, 69])]
        state = {"remaining_budget": 15, "rejected_hypotheses": [_failure_receipt(a) for a in actions[:2]]}
        self.assertEqual(eligible_joint_measurement_targets(
            state, [], active_id=ACTIVE, supported_actions=actions, accepted_indices=set()
        ), [])

    @staticmethod
    def _environment():
        env = TransactionalPSSEEnv(history_window=4)
        env.reset({"scenario_id": "executor_failure", "case": {"branch": [{"r": 1.0, "x": 2.0}]},
                   "measurements": [1.0]})
        return env

    @staticmethod
    def _record_failure(env, action=None, **overrides):
        active = env.store.active_state_id
        action = action or _parameter(19, active)
        output = {"execution_status": "failure", "error_code": "parameter_correction_failure",
                  "error_detail": None, "state_mutated": False, "active_state_id": active,
                  "candidate_state_id": None, "tool_metrics": {}, "valid_next_actions": []}
        output.update(overrides)
        env._record_transition(action, output, source_state_id=active)
        return action, output

    def test_real_transition_records_truthful_deduplicated_durable_receipt(self):
        env = self._environment()
        active = env.store.active_state_id
        state_hash = env.store.get_state(active)["state_hash"]
        action, _ = self._record_failure(env)
        self._record_failure(env)
        for _ in range(5):
            env._record_transition(
                {"tool": "run_wls", "arguments": {"state_id": active}},
                {"execution_status": "success", "active_state_id": active, "tool_metrics": {}},
                source_state_id=active,
            )
        observation = env.get_policy_observation().as_dict()
        receipts = observation["rejected_hypotheses"]
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0], _failure_receipt(action, state_hash))
        self.assertNotIn("candidate_state_id", receipts[0])
        self.assertNotIn("verification_summary", receipts[0])
        self.assertEqual(len(observation["history_window"]), 4)
        self.assertTrue(all(item["action"]["tool"] == "run_wls" for item in observation["history_window"]))
        self.assertEqual(env.store.get_state(active)["state_hash"], state_hash)
        self.assertFalse(env.is_terminal())
        from psse_env.dagger.dataset_builder import prepare_model_policy_observation
        model_observation, _ = prepare_model_policy_observation(observation)
        visible = model_observation["rejected_hypotheses"][0]
        self.assertEqual(visible["rejection_kind"], "executor_failure")
        self.assertEqual(visible["error_code"], "parameter_correction_failure")
        self.assertEqual(visible["source_action"]["arguments"]["line_index"], 19)

    def test_process_or_prerequisite_failure_never_becomes_executor_receipt(self):
        for error in ("missing_precondition", "schema_error", "correction_route_not_actionable",
                      "candidate_lifecycle_violation", "unknown_state_id", "state_reference_mismatch",
                      "measurement_correction_failure"):
            with self.subTest(error=error):
                env = self._environment()
                self._record_failure(env, error_code=error, error_detail="measurement_harmonic_evidence_request_pending")
                self.assertEqual(env.current_state()["rejected_hypotheses"], [])

    def test_mutation_candidate_or_mismatched_state_prevents_failure_receipt(self):
        for overrides in ({"state_mutated": True}, {"candidate_state_id": "unexpected:candidate"},
                          {"active_state_id": "other:state"}, {"execution_status": "success"}):
            with self.subTest(overrides=overrides):
                env = self._environment()
                self._record_failure(env, **overrides)
                self.assertEqual(env.current_state()["rejected_hypotheses"], [])

    def test_captured_step25_resumes_rank_one_meter_with_only_visible_receipt_added(self):
        artifact = Path(__file__).resolve().parents[1]/"output/r1_recovery_20260922/failed_state_before.json"
        if not artifact.exists():
            self.skipTest("Captured historical mixed-policy diagnostic is a local audit artifact")
        captured = json.loads(artifact.read_text())
        state = deepcopy(captured["policy_observation"])
        self.assertEqual(_select(state)[0], captured["actions"][0])
        failed_event = captured["environment_history"][16]
        self.assertEqual(failed_event["tool_output"]["error_code"], "parameter_correction_failure")
        state["rejected_hypotheses"].append(_failure_receipt(
            failed_event["action"], captured["audit"]["ledger"]["active_state_hash"]
        ))
        self.assertEqual(_select(state)[0], {
            "tool": "correct_measurements", "arguments": {
                "state_id": state["active_state_id"], "suspect_group": [40],
            },
        })
        self.assertEqual(state["history_window"], captured["policy_observation"]["history_window"])


if __name__ == "__main__":
    unittest.main()
