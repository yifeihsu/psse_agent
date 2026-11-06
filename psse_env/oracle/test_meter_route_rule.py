"""The meter-route rule probe (compound errors on IEEE 118, 2026-10-29): order, cue and the lifted guard."""
from __future__ import annotations

import unittest

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RUN_WLS,
)
from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.expert_policy import METER_ROUTE_EVIDENCE, _measurement_first_order, _structural_first_order
from psse_env.oracle.expert_types import ExpertActionProposal
from psse_env.oracle.expert_variants import (
    DEFAULT_EXPERT_VARIANT,
    EXPERT_VARIANT_ENV,
    EXPERT_VARIANTS,
    METER_ROUTE_RESIDUAL,
    current_expert_variant,
    expert_variant_options,
)
from psse_env.oracle.measurement_expert import MeasurementExpert

ACTIVE = "episode:s1"


def _proposal(tool: str, **arguments) -> ExpertActionProposal:
    return ExpertActionProposal(
        action={"tool": tool, "arguments": {"state_id": ACTIVE, **arguments}},
        source_expert="test", confidence=0.9, evidence_codes=["x"], admissible=True, estimated_immediate_risk=0.1,
    )


def _tools(proposals) -> list[str]:
    return [proposal.action["tool"] for proposal in proposals]


def _committed(tool: str, **arguments) -> dict:
    return {"candidate_state_id": ACTIVE, "source_action": {"tool": tool, "arguments": {"state_id": "episode:s0", **arguments}}}


def _cue_state(*, residual: float | None = 15.7, accepted: list | None = None, successful: bool = True,
               bound_to: str = ACTIVE) -> dict:
    wls = {"state_id": bound_to, "successful": successful}
    if residual is not None:
        wls["max_normalized_residual"] = residual
    return {
        "active_state_id": ACTIVE,
        "accepted_corrections": [_committed(CORRECT_MEASUREMENTS, suspect_group=[8])] if accepted is None else accepted,
        "fresh_context_evidence": {"wls": wls},
    }


class MeasurementFirstOrderTests(unittest.TestCase):
    def test_meters_lead_and_carry_the_rule_code(self) -> None:
        ranked = [
            _proposal(CORRECT_PARAMETERS, line_index=3),
            _proposal(CORRECT_MEASUREMENTS, suspect_group=[4]),
            _proposal(CORRECT_TOPOLOGY, line_index=5, status=0),
            _proposal(GET_PARAMETER_CONTEXT),
            _proposal(GET_TOPOLOGY_CONTEXT),
            _proposal(GET_MEASUREMENT_CONTEXT),
        ]
        ordered = _measurement_first_order(ranked)
        self.assertEqual(_tools(ordered), [CORRECT_MEASUREMENTS, GET_MEASUREMENT_CONTEXT, CORRECT_PARAMETERS,
                                           CORRECT_TOPOLOGY, GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT])
        self.assertEqual([METER_ROUTE_EVIDENCE in p.evidence_codes for p in ordered], [True, True, False, False, False, False])
        self.assertEqual(ordered[0].evidence_codes, ["x", METER_ROUTE_EVIDENCE])
        # The structural tier keeps the breaker-first option when it is on.
        self.assertEqual(_tools(_measurement_first_order(ranked, topology_first=True))[2:4], [CORRECT_TOPOLOGY, GET_TOPOLOGY_CONTEXT])

    def test_without_a_meter_proposal_the_structural_order_stands(self) -> None:
        ranked = [_proposal(GET_PARAMETER_CONTEXT), _proposal(RUN_WLS), _proposal(CORRECT_TOPOLOGY, line_index=5, status=0)]
        self.assertEqual(_measurement_first_order(ranked), _structural_first_order(ranked))


class MeterRouteCueTests(unittest.TestCase):
    def test_the_rule_is_off_unless_named(self) -> None:
        self.assertIsNone(ExpertPolicyOracle().meter_route_residual)
        self.assertIsNone(ExpertPolicyOracle(**expert_variant_options("baseline_compare")).meter_route_residual)
        self.assertFalse(ExpertPolicyOracle()._meter_route_for(_cue_state()))
        self.assertFalse(ExpertPolicyOracle(**expert_variant_options("baseline_compare"))._meter_route_for(_cue_state()))

    def test_the_cue_needs_a_latest_meter_commit_and_a_large_bound_residual(self) -> None:
        expert = ExpertPolicyOracle(candidate_comparison=True, meter_route_residual=11)
        self.assertEqual(expert.meter_route_residual, 11.0)
        self.assertTrue(expert._meter_route_for(_cue_state()))
        self.assertTrue(expert._meter_route_for(_cue_state(residual=11.0)))
        self.assertFalse(expert._meter_route_for(_cue_state(residual=10.9)))
        self.assertFalse(expert._meter_route_for(_cue_state(residual=None)))
        self.assertFalse(expert._meter_route_for(_cue_state(accepted=[])))
        # The latest commit decides: a line committed after the meter turns it off.
        self.assertFalse(expert._meter_route_for(_cue_state(accepted=[
            _committed(CORRECT_MEASUREMENTS, suspect_group=[8]), _committed(CORRECT_PARAMETERS, line_index=3)])))
        self.assertTrue(expert._meter_route_for(_cue_state(accepted=[
            _committed(CORRECT_PARAMETERS, line_index=3), _committed(CORRECT_MEASUREMENTS, suspect_group=[8])])))
        self.assertFalse(expert._meter_route_for(_cue_state(successful=False)))
        self.assertFalse(expert._meter_route_for(_cue_state(bound_to="episode:s0")))


class MeterRouteVariantTests(unittest.TestCase):
    def test_the_named_variant_is_baseline_compare_with_the_rule(self) -> None:
        self.assertIn("baseline_compare_meter_route", EXPERT_VARIANTS)
        self.assertEqual(METER_ROUTE_RESIDUAL, 11.0)
        self.assertEqual(expert_variant_options("baseline_compare_meter_route"),
                         {**expert_variant_options("baseline_compare"), "meter_route_residual": METER_ROUTE_RESIDUAL})
        expert = ExpertPolicyOracle(**expert_variant_options("baseline_compare_meter_route"))
        self.assertTrue(expert.candidate_comparison)
        self.assertEqual(expert.meter_route_residual, 11.0)
        self.assertTrue(expert._meter_route_for(_cue_state()))
        self.assertFalse(expert._meter_route_for(_cue_state(residual=10.9)))

    def test_no_other_variant_carries_the_rule_and_the_default_is_untouched(self) -> None:
        for name in EXPERT_VARIANTS:
            if name != "baseline_compare_meter_route":
                self.assertNotIn("meter_route_residual", expert_variant_options(name))
        self.assertEqual(expert_variant_options("baseline_compare"), {"candidate_comparison": True})
        # The unnamed default is baseline_compare since the user's decision of 2026-10-28 (the IEEE 14
        # study's branch); this test was written on the IEEE 118 branch, which split off before it.
        self.assertEqual(DEFAULT_EXPERT_VARIANT, "baseline_compare")
        self.assertNotIn("meter_route_residual", expert_variant_options(DEFAULT_EXPERT_VARIANT))

    def test_the_environment_selects_it_by_name(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {EXPERT_VARIANT_ENV: "baseline_compare_meter_route"}):
            self.assertEqual(current_expert_variant(), "baseline_compare_meter_route")
            self.assertEqual(expert_variant_options()["meter_route_residual"], METER_ROUTE_RESIDUAL)


def _partial_meter_state(signatures: list[str], *, residual: float = 15.7) -> tuple[dict, dict]:
    """A partial meter commit on s1 with a fresh meter context and an untried breaker candidate."""
    meter = {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": ACTIVE, "suspect_group": [13]}}
    breaker = {"tool": CORRECT_TOPOLOGY, "arguments": {"state_id": ACTIVE, "line_index": 5, "status": 0}}
    state = {
        "active_state_id": ACTIVE,
        "evidence_profile": AUXILIARY_EVIDENCE_PROFILE,
        "candidate_state_id": None,
        "no_material_anomaly_remaining": False,
        "unresolved_signatures": signatures,
        "remaining_anomaly_score": 1.05,
        "remaining_budget": 12,
        "accepted_corrections": [_committed(CORRECT_MEASUREMENTS, suspect_group=[8])],
        "rejected_hypotheses": [],
        "has_fresh_measurement_context": True,
        "measurement_context_state_id": ACTIVE,
        "has_fresh_parameter_context": True,
        "parameter_context_state_id": ACTIVE,
        "has_fresh_topology_context": True,
        "topology_context_state_id": ACTIVE,
        "fresh_context_evidence": {
            "measurement": {"state_id": ACTIVE, "supported_corrections": [meter]},
            "parameter": {"state_id": ACTIVE, "supported_corrections": [], "route_status": "complete_negative"},
            "topology": {"state_id": ACTIVE, "supported_corrections": [breaker], "route_status": "actionable"},
            "wls": {"state_id": ACTIVE, "evidence_source": "deployment_wls:diagnosis", "successful": True, "anomalous": True,
                    "max_normalized_residual": residual},
            "three_phase": {"state_id": ACTIVE, "evidence_source": "deployment_context:three_phase_measurements",
                            "request_attempted": True, "three_phase_context_status": "unavailable",
                            "available_evidence_channels": []},
            "harmonic": {"state_id": ACTIVE, "evidence_source": "deployment_context:harmonic_measurements",
                         "request_attempted": True, "harmonic_context_status": "unavailable",
                         "available_evidence_channels": [], "harmonic_distortion_detected": False},
        },
    }
    return state, meter


_METER_LED = ["wls_residual_outlier_dominant index=13 channel=Vm", "wls_branch_multiplier line_status_or_parameter line=5"]
_BRANCH_DOMINANT = ["wls_residual_outlier index=13 channel=Vm", "wls_branch_multiplier_dominant line_status_or_parameter line=5"]


class MeterRouteGuardTests(unittest.TestCase):
    def test_the_rule_lifts_the_post_commit_branch_guard(self) -> None:
        state, meter = _partial_meter_state(_METER_LED)
        blocked = MeasurementExpert().propose(state, [], oracle_hints=[meter])
        self.assertNotIn(meter, [p.action for p in blocked])
        lifted = MeasurementExpert().propose(state, [], oracle_hints=[meter], meter_route=True)
        self.assertIn(meter, [p.action for p in lifted])

    def test_branch_dominance_still_holds_the_meter_route(self) -> None:
        state, meter = _partial_meter_state(_BRANCH_DOMINANT)
        proposals = MeasurementExpert().propose(state, [], oracle_hints=[meter], meter_route=True)
        self.assertNotIn(meter, [p.action for p in proposals])

    def test_the_oracle_takes_the_meter_route_only_where_the_cue_fires(self) -> None:
        process = ProcessValidityOracle(executor_hydrated_corrections=True)
        state, meter = _partial_meter_state(_METER_LED)
        baseline = ExpertPolicyOracle(process_oracle=process, candidate_comparison=True).next_actions(state, [])
        self.assertTrue(baseline)
        self.assertNotEqual(baseline[0], meter)
        rule = ExpertPolicyOracle(process_oracle=process, candidate_comparison=True, meter_route_residual=11)
        self.assertEqual(rule.next_actions(state, [])[0], meter)
        quiet, _ = _partial_meter_state(_METER_LED, residual=9.0)
        self.assertEqual(rule.next_actions(quiet, []), ExpertPolicyOracle(
            process_oracle=process, candidate_comparison=True).next_actions(quiet, []))


if __name__ == "__main__":
    unittest.main()
