"""The topology-first structural order (compound-error study, 2026-10-27) and its variant."""
from __future__ import annotations

import unittest

from psse_env.actions import (
    COMPARE_PARAMETER_CANDIDATES,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RUN_WLS,
)
from psse_env.oracle.expert_policy import FLOW_AT_NOISE_GUARD, ExpertPolicyOracle, _structural_first_order
from psse_env.oracle.expert_types import ExpertActionProposal
from psse_env.oracle.expert_variants import EXPERT_VARIANTS, expert_variant_options


def _proposal(tool: str, **arguments) -> ExpertActionProposal:
    return ExpertActionProposal(
        action={"tool": tool, "arguments": {"state_id": "s0", **arguments}},
        source_expert="test", confidence=0.9, evidence_codes=[], admissible=True, estimated_immediate_risk=0.1,
    )


def _tools(proposals) -> list[str]:
    return [proposal.action["tool"] for proposal in proposals]


class TopologyFirstOrderTests(unittest.TestCase):
    def test_default_order_is_unchanged(self) -> None:
        ranked = [
            _proposal(CORRECT_PARAMETERS, line_index=3),
            _proposal(CORRECT_MEASUREMENTS, suspect_group=[4]),
            _proposal(CORRECT_TOPOLOGY, cb_name="CB_4_N1_N2", status=0),
            _proposal(GET_PARAMETER_CONTEXT),
            _proposal(GET_TOPOLOGY_CONTEXT),
            _proposal(GET_MEASUREMENT_CONTEXT),
        ]
        self.assertEqual(
            _tools(_structural_first_order(ranked)),
            [CORRECT_PARAMETERS, CORRECT_TOPOLOGY, GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT,
             CORRECT_MEASUREMENTS, GET_MEASUREMENT_CONTEXT],
        )
        self.assertEqual(_tools(_structural_first_order(ranked, topology_first=False)), _tools(_structural_first_order(ranked)))

    def test_topology_first_leads_the_structural_tier_only(self) -> None:
        ranked = [
            _proposal(COMPARE_PARAMETER_CANDIDATES, line_indices=[3, 6]),
            _proposal(CORRECT_PARAMETERS, line_index=3),
            _proposal(CORRECT_MEASUREMENTS, suspect_group=[4]),
            _proposal(CORRECT_TOPOLOGY, cb_name="CB_4_N1_N2", status=0),
            _proposal(GET_PARAMETER_CONTEXT),
            _proposal(GET_TOPOLOGY_CONTEXT),
            _proposal(GET_MEASUREMENT_CONTEXT),
        ]
        self.assertEqual(
            _tools(_structural_first_order(ranked, topology_first=True)),
            [CORRECT_TOPOLOGY, GET_TOPOLOGY_CONTEXT, COMPARE_PARAMETER_CANDIDATES, CORRECT_PARAMETERS,
             GET_PARAMETER_CONTEXT, CORRECT_MEASUREMENTS, GET_MEASUREMENT_CONTEXT],
        )
        # Without a structural proposal nothing moves, whatever the flag.
        meters = [_proposal(CORRECT_MEASUREMENTS, suspect_group=[4]), _proposal(RUN_WLS)]
        self.assertEqual(_tools(_structural_first_order(meters, topology_first=True)), _tools(meters))

    def test_variant_selects_the_flag_and_the_conventional_teacher_does_not(self) -> None:
        self.assertIn("baseline_compare_topology_first", EXPERT_VARIANTS)
        self.assertEqual(expert_variant_options("baseline_compare"), {"candidate_comparison": True})
        self.assertEqual(expert_variant_options("baseline_compare_topology_first"),
                         {"candidate_comparison": True, "topology_first": True})
        self.assertFalse(ExpertPolicyOracle().topology_first)
        self.assertFalse(ExpertPolicyOracle(**expert_variant_options("baseline_compare")).topology_first)
        self.assertTrue(ExpertPolicyOracle(**expert_variant_options("baseline_compare_topology_first")).topology_first)


def _policy(**fields) -> dict:
    return {"active_state_id": "s0", **fields}


def _fired(lines=(7,), successful=True) -> dict:
    return _policy(fresh_context_evidence={"wls": {"successful": successful, "flow_at_noise_lines": list(lines)}})


class GuardedTopologyFirstTests(unittest.TestCase):
    """Stage 2 (2026-10-28): the breaker route leads only where the solve shows the open-terminal cue."""

    def test_guard_follows_the_bound_wls_cue(self) -> None:
        expert = ExpertPolicyOracle(**expert_variant_options("baseline_compare_topology_guarded"))
        self.assertEqual(expert.topology_first, FLOW_AT_NOISE_GUARD)
        self.assertTrue(expert._topology_first_for(_fired()))
        self.assertFalse(expert._topology_first_for(_fired(lines=())))
        self.assertFalse(expert._topology_first_for(_fired(successful=False)))
        self.assertFalse(expert._topology_first_for(_policy()))
        self.assertFalse(expert._topology_first_for(_policy(fresh_context_evidence={"wls": {"successful": True}})))
        # The solve's own output serves before the durable ledger exists, and
        # only the solve's: another tool's metrics are not read.
        direct = _policy(last_tool=RUN_WLS, last_tool_output={"tool_metrics": {"flow_at_noise_lines": [3]}})
        self.assertTrue(expert._topology_first_for(direct))
        other = _policy(last_tool=GET_PARAMETER_CONTEXT, last_tool_output={"tool_metrics": {"flow_at_noise_lines": [3]}})
        self.assertFalse(expert._topology_first_for(other))

    def test_the_cue_reorders_the_structural_tier_only_when_it_fires(self) -> None:
        expert = ExpertPolicyOracle(**expert_variant_options("baseline_compare_topology_guarded"))
        ranked = [
            _proposal(CORRECT_PARAMETERS, line_index=3),
            _proposal(CORRECT_TOPOLOGY, cb_name="CB_4_N1_N2", status=0),
            _proposal(GET_PARAMETER_CONTEXT),
            _proposal(GET_TOPOLOGY_CONTEXT),
            _proposal(CORRECT_MEASUREMENTS, suspect_group=[4]),
        ]
        self.assertEqual(
            _tools(_structural_first_order(ranked, topology_first=expert._topology_first_for(_fired()))),
            [CORRECT_TOPOLOGY, GET_TOPOLOGY_CONTEXT, CORRECT_PARAMETERS, GET_PARAMETER_CONTEXT, CORRECT_MEASUREMENTS],
        )
        self.assertEqual(
            _tools(_structural_first_order(ranked, topology_first=expert._topology_first_for(_fired(lines=())))),
            _tools(_structural_first_order(ranked)),
        )

    def test_other_modes_ignore_the_cue(self) -> None:
        self.assertFalse(ExpertPolicyOracle(**expert_variant_options("baseline_compare"))._topology_first_for(_fired()))
        self.assertFalse(ExpertPolicyOracle()._topology_first_for(_fired()))
        self.assertTrue(ExpertPolicyOracle(topology_first=True)._topology_first_for(_policy()))
        with self.assertRaises(ValueError):
            ExpertPolicyOracle(topology_first="sometimes")

    def test_naming_the_guarded_teacher_turns_its_cue_on(self) -> None:
        """The guarded teacher reads the open-terminal cue, which no other teacher's environment reports (review of 2026-10-28)."""
        from psse_env.oracle.expert_variants import variant_provider_options

        self.assertEqual(variant_provider_options("baseline_compare_topology_guarded"), {"flow_at_noise_cue": True})
        for name in ("baseline", "baseline_compare", "baseline_compare_topology_first"):
            self.assertEqual(variant_provider_options(name), {}, name)

    def test_the_default_teacher_for_new_work_is_baseline_compare(self) -> None:
        """The user's decision of 2026-10-28: no topology-first order by default; recorded cells name their own teacher."""
        import os
        from unittest import mock

        from psse_env.oracle.expert_variants import DEFAULT_EXPERT_VARIANT, current_expert_variant, variant_provider_options

        self.assertEqual(DEFAULT_EXPERT_VARIANT, "baseline_compare")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PSSE_EXPERT_VARIANT", None)
            self.assertEqual(current_expert_variant(), "baseline_compare")
            self.assertEqual(expert_variant_options(), {"candidate_comparison": True})
            self.assertFalse(ExpertPolicyOracle(**expert_variant_options()).topology_first)
            # The default teacher's environment reports no stage-2 signal.
            self.assertEqual(variant_provider_options(), {})
            os.environ["PSSE_EXPERT_VARIANT"] = "baseline_compare_topology_guarded"
            self.assertEqual(current_expert_variant(), "baseline_compare_topology_guarded")
            self.assertEqual(variant_provider_options(), {"flow_at_noise_cue": True})

    def test_variant_registry(self) -> None:
        self.assertIn("baseline_compare_topology_guarded", EXPERT_VARIANTS)
        # Removed on 2026-10-28 as brute-force screening (study section 7.8).
        self.assertNotIn("baseline_compare_topology_clean_flip", EXPERT_VARIANTS)
        self.assertEqual(expert_variant_options("baseline_compare_topology_guarded"),
                         {"candidate_comparison": True, "topology_first": "flow_at_noise"})
        self.assertEqual(expert_variant_options("baseline_compare"), {"candidate_comparison": True})


if __name__ == "__main__":
    unittest.main()
