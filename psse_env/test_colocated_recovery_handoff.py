"""Observable meter-safety exclusions must agree with the handoff inventory.

These tests exercise the environment's real audit against minimal, state-bound
provider records. They do not run power-flow generation or claim that a handoff
physically repairs a fault.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE
from psse_env.oracle.measurement_expert import MeasurementExpert
from psse_env.transactional_env import (
    TransactionalPSSEEnv,
    _semantic_correction_signature,
)


ACTIVE = "episode:s1"
ACTIVE_HASH = "hash-s1"


def _action(tool, **arguments):
    return {"tool": tool, "arguments": {"state_id": ACTIVE, **arguments}}


def _event(tool, **metrics):
    return {
        "action": _action(tool),
        "tool_output": {
            "execution_status": "success",
            "tool_metrics": {
                "state_id": ACTIVE,
                "state_hash": ACTIVE_HASH,
                "evidence_source": f"deployment_diagnostic:{tool}",
                **metrics,
            },
        },
    }


def _fixture(*, independent_meter=False, grouped=False):
    colocated = _action("correct_measurements", suspect_group=[42])
    independent = _action("correct_measurements", suspect_group=[47])
    joint = _action("correct_measurements", suspect_group=[42, 47])
    supported = [colocated]
    if independent_meter:
        supported.append(independent)
    if grouped:
        supported.append(joint)
    context = _event(
        "get_measurement_context",
        supported_corrections=supported,
        measurement_findings=[
            {"index0": 42, "channel": "Pf", "channel_offset": 0},
            {"index0": 47, "channel": "Pf", "channel_offset": 5},
        ],
    )
    history = [_event("run_wls"), context]
    state = {
        "active_state_id": ACTIVE,
        "candidate_state_id": None,
        "has_open_candidate": False,
        "no_material_anomaly_remaining": False,
        "remaining_anomaly_score": 2.0,
        "remaining_budget": 15,
        "unresolved_signatures": ["wls_residual_outlier index=42 channel=Pf"],
        "semantic_field_provenance": {
            "unresolved_signatures": "deployment_diagnostic:wls",
            "remaining_anomaly_score": "deployment_diagnostic:wls",
        },
        "has_fresh_measurement_context": True,
        "measurement_context_state_id": ACTIVE,
        "fresh_context_evidence": {
            "measurement": deepcopy(context["tool_output"]["tool_metrics"])
        },
        "accepted_corrections": [{
            "source_action": {
                "tool": "correct_parameters",
                "arguments": {"state_id": "episode:s0", "line_index": 1},
            }
        }],
        "rejected_hypotheses": [],
    }
    env = SimpleNamespace(
        # The audit reads the environment's declared profile; this stub
        # exercises the historical (ungated) handoff inventory.
        evidence_profile=AUXILIARY_EVIDENCE_PROFILE,
        current_state=lambda: state,
        history=history,
        store=SimpleNamespace(get_state=lambda _: {"state_hash": ACTIVE_HASH}),
        process_oracle=SimpleNamespace(anomaly_threshold=1.0),
        evidence_providers={"ask_for_more_evidence": lambda _: None},
        _observable_evidence_channels=lambda: [],
    )
    return env, state, context, colocated, independent, joint


def _audit(env):
    return TransactionalPSSEEnv._operator_escalation_audit(
        env,
        _action(
            "ask_for_more_evidence",
            request="operator_escalation:recovery_options_exhausted",
        ),
    )


def _exhaust(state, action):
    state["rejected_hypotheses"].append({
        "candidate_parent_id": ACTIVE,
        "candidate_state_id": ACTIVE + ":rejected",
        "source_action": deepcopy(action),
    })


class ColocatedRecoveryHandoffTests(unittest.TestCase):
    def test_colocated_flow_is_safety_blocked_without_a_fake_correction_attempt(self):
        env, state, _, colocated, _, _ = _fixture()
        proposals = MeasurementExpert().propose(
            state, env.history, oracle_hints=[colocated]
        )
        self.assertNotIn(colocated, [proposal.action for proposal in proposals])

        audit = _audit(env)

        self.assertTrue(audit["sufficient"], audit["missing"])
        self.assertEqual(audit["ledger"]["outstanding_recovery_targets"], [])
        self.assertIn(
            _semantic_correction_signature(colocated),
            audit["ledger"]["safety_blocked_recovery_targets"],
        )
        self.assertEqual(audit["ledger"]["exhausted_recovery_target_count"], 0)
        self.assertEqual(state["rejected_hypotheses"], [])

    def test_independent_meter_stays_outstanding_until_actually_exhausted(self):
        env, state, _, colocated, independent, joint = _fixture(
            independent_meter=True, grouped=True
        )
        audit = _audit(env)
        self.assertFalse(audit["sufficient"])
        self.assertIn("same_state_supported_corrections_unexhausted", audit["missing"])
        self.assertEqual(
            audit["ledger"]["outstanding_recovery_targets"],
            [_semantic_correction_signature(independent)],
        )
        self.assertEqual(
            set(audit["ledger"]["safety_blocked_recovery_targets"]),
            {_semantic_correction_signature(colocated), _semantic_correction_signature(joint)},
        )
        _exhaust(state, independent)
        self.assertTrue(_audit(env)["sufficient"])

    def test_independent_branch_route_stays_outstanding(self):
        env, state, _, _, _, _ = _fixture()
        branch = _action("correct_parameters", line_index=2)
        env.history.append(_event("get_parameter_context", supported_corrections=[branch]))
        audit = _audit(env)
        self.assertFalse(audit["sufficient"])
        self.assertEqual(
            audit["ledger"]["outstanding_recovery_targets"],
            [_semantic_correction_signature(branch)],
        )
        _exhaust(state, branch)
        self.assertTrue(_audit(env)["sufficient"])

    def test_rejected_branch_candidate_is_not_an_accepted_branch_repair(self):
        env, state, _, colocated, _, _ = _fixture()
        branch = state["accepted_corrections"].pop()["source_action"]
        _exhaust(state, branch)
        audit = _audit(env)
        self.assertFalse(audit["sufficient"])
        self.assertIn(_semantic_correction_signature(colocated), audit["ledger"]["outstanding_recovery_targets"])
        self.assertEqual(audit["ledger"]["safety_blocked_recovery_targets"], [])

    def test_stale_or_unbound_context_cannot_establish_a_safety_exclusion(self):
        for field, value in (
            ("state_id", "episode:s0"),
            ("state_hash", "stale-hash"),
            ("evidence_source", "scenario_truth"),
        ):
            with self.subTest(field=field):
                env, _, context, _, _, _ = _fixture()
                context["tool_output"]["tool_metrics"][field] = value
                audit = _audit(env)
                self.assertFalse(audit["sufficient"])
                self.assertIn("required_recovery_contexts_missing", audit["missing"])
                self.assertEqual(audit["ledger"]["safety_blocked_recovery_targets"], [])

    def test_stale_policy_findings_cannot_override_current_bound_provider_findings(self):
        env, _, context, colocated, _, _ = _fixture()
        # The durable snapshot still says branch0, but the authoritative bound
        # provider event places this meter on independent branch5.
        context["tool_output"]["tool_metrics"]["measurement_findings"][0]["channel_offset"] = 5
        audit = _audit(env)
        self.assertFalse(audit["sufficient"])
        self.assertIn(_semantic_correction_signature(colocated), audit["ledger"]["outstanding_recovery_targets"])
        self.assertEqual(audit["ledger"]["safety_blocked_recovery_targets"], [])

    def test_old_state_history_cannot_override_current_independent_findings(self):
        env, state, context, colocated, _, _ = _fixture()
        old_context = deepcopy(context)
        old_context["action"]["arguments"]["state_id"] = "episode:s0"
        old_context["tool_output"]["tool_metrics"].update(state_id="episode:s0", state_hash="hash-s0")
        context["tool_output"]["tool_metrics"]["measurement_findings"][0]["channel_offset"] = 5
        state["fresh_context_evidence"]["measurement"] = deepcopy(context["tool_output"]["tool_metrics"])
        env.history.append(old_context)
        audit = _audit(env)
        self.assertFalse(audit["sufficient"])
        self.assertIn(_semantic_correction_signature(colocated), audit["ledger"]["outstanding_recovery_targets"])
        self.assertEqual(audit["ledger"]["safety_blocked_recovery_targets"], [])

    def test_malformed_or_nonflow_findings_do_not_block_a_supported_meter(self):
        invalid_findings = [
            {"index0": 42, "channel": "Pf", "channel_offset": "0"},
            {"index0": 42, "channel": "Pf", "channel_offset": False},
            {"index0": 42, "channel": "Pf", "channel_offset": -1},
            {"index0": "42", "channel": "Pf", "channel_offset": 0},
            {"index0": True, "channel": "Pf", "channel_offset": 0},
            {"index0": 42, "channel": "Pinj", "channel_offset": 0},
        ]
        for finding in invalid_findings:
            with self.subTest(finding=finding):
                env, state, context, colocated, _, _ = _fixture()
                context["tool_output"]["tool_metrics"]["measurement_findings"] = [finding]
                state["fresh_context_evidence"]["measurement"] = deepcopy(context["tool_output"]["tool_metrics"])
                audit = _audit(env)
                self.assertFalse(audit["sufficient"])
                self.assertIn(_semantic_correction_signature(colocated), audit["ledger"]["outstanding_recovery_targets"])
                self.assertEqual(audit["ledger"]["safety_blocked_recovery_targets"], [])


if __name__ == "__main__":
    unittest.main()
