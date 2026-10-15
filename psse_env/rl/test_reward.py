from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.actions import ASK_FOR_MORE_EVIDENCE, GET_HARMONIC_CONTEXT, GET_THREE_PHASE_CONTEXT, RUN_WLS
from psse_env.rl.reward import (
    BUDGET_EXHAUSTED_CREDIT,
    BUDGET_EXHAUSTED_REQUEST,
    RewardWeights,
    acquisition_counts,
    episode_basis,
    episode_reward,
    escalation_request,
    load_expert_reference,
    unneeded_acquisitions,
)


def _step(tool: str, status: str = "success", request: str | None = None) -> dict:
    arguments = {"state_id": "active"}
    if request is not None:
        arguments["request"] = request
    return {"action": {"tool": tool, "arguments": arguments}, "execution_status": status,
            "policy_tool_output": {"execution_status": status}}


def _episode(*, success: bool, basis: str | None, trace: list[dict], steps: int | None = None, **extra) -> dict:
    episode = {
        "scenario_id": "r0_a", "family": "parameter", "truth_audited_task_success": success,
        "audit": {"truth_audited_task_assessment": {"basis": basis, "eligible": success}},
        "trace": trace, "steps": steps if steps is not None else len(trace), "policy_steps": steps if steps is not None else len(trace),
        "false_commit_count": 0, "false_finalization_count": 0, "healthy_components_preserved": True,
        "healthy_preservation_known": True, "loop_detected": False, "invalid_action_count": 0,
    }
    episode.update(extra)
    return episode


class RewardTests(unittest.TestCase):
    def test_decisive_success_outranks_every_handoff_basis(self) -> None:
        trace = [_step(RUN_WLS)] * 6
        decisive = episode_reward(_episode(success=True, basis="counterfactual_resolution", trace=trace))
        bounded = episode_reward(_episode(success=True, basis="bounded_localization_handoff", trace=trace))
        unexplained = episode_reward(_episode(success=True, basis="unexplained_discrepancy_handoff", trace=trace))
        failure = episode_reward(_episode(success=False, basis=None, trace=trace))
        self.assertGreater(decisive.total, bounded.total)
        self.assertGreater(bounded.total, unexplained.total)
        self.assertGreater(unexplained.total, failure.total)
        self.assertEqual(failure.credit, 0.0)
        self.assertEqual(episode_basis(_episode(success=False, basis=None, trace=trace)), "none")

    def test_budget_exhaustion_is_capped_below_a_bounded_handoff(self) -> None:
        trace = [_step(RUN_WLS)] * 5 + [_step(ASK_FOR_MORE_EVIDENCE, request=BUDGET_EXHAUSTED_REQUEST)]
        episode = _episode(success=True, basis="counterfactual_resolution", trace=trace)
        reward = episode_reward(episode)
        self.assertEqual(escalation_request(episode), BUDGET_EXHAUSTED_REQUEST)
        self.assertEqual(reward.credit, BUDGET_EXHAUSTED_CREDIT)
        bounded = episode_reward(_episode(success=True, basis="bounded_localization_handoff", trace=[_step(RUN_WLS)] * 6))
        self.assertLess(reward.credit, bounded.credit)

    def test_calls_and_unneeded_acquisitions_cost_and_are_capped(self) -> None:
        weights = RewardWeights(call_cost=0.1, unneeded_acquisition_cost=1.0, max_charged_calls=4)
        trace = [_step(RUN_WLS), _step(GET_THREE_PHASE_CONTEXT), _step(GET_HARMONIC_CONTEXT, status="failure"),
                 _step(GET_HARMONIC_CONTEXT), _step(RUN_WLS), _step(RUN_WLS)]
        episode = _episode(success=True, basis="counterfactual_resolution", trace=trace)
        reward = episode_reward(episode, family="parameter", weights=weights)
        self.assertEqual(reward.calls, 6)
        self.assertEqual(reward.charged_calls, 4)
        self.assertEqual(reward.acquisitions, {GET_HARMONIC_CONTEXT: 1, GET_THREE_PHASE_CONTEXT: 1})
        self.assertEqual(reward.unneeded_acquisitions, 2)
        self.assertAlmostEqual(reward.costs, 0.4 + 2.0)
        self.assertAlmostEqual(reward.total, 1.0 - 2.4)
        hif = episode_reward(episode, family="hif", weights=weights)
        self.assertEqual(hif.unneeded_acquisitions, 1)
        self.assertEqual(unneeded_acquisitions(acquisition_counts(episode), "harmonic"), 1)

    def test_penalties_apply_once_per_event_and_per_count(self) -> None:
        weights = RewardWeights(call_cost=0.0, false_commit=0.5, false_finalization=0.25,
                                healthy_component_corruption=0.75, loop=0.2, invalid_action=0.05)
        episode = _episode(success=False, basis=None, trace=[_step(RUN_WLS)] * 3,
                           false_commit_count=2, false_finalization_count=1,
                           healthy_components_preserved=False, loop_detected=True, invalid_action_count=4)
        reward = episode_reward(episode, weights=weights)
        self.assertTrue(reward.healthy_component_corrupted)
        self.assertAlmostEqual(reward.penalties, 1.0 + 0.25 + 0.75 + 0.2 + 0.2)
        self.assertAlmostEqual(reward.total, -reward.penalties)
        unknown = _episode(success=False, basis=None, trace=[_step(RUN_WLS)],
                           healthy_components_preserved=False, healthy_preservation_known=False)
        self.assertFalse(episode_reward(unknown, weights=weights).healthy_component_corrupted)

    def test_tool_regret_charges_calls_beyond_the_expert_reference(self) -> None:
        weights = RewardWeights(call_cost=0.0, tool_regret_cost=0.1)
        episode = _episode(success=True, basis="counterfactual_resolution", trace=[_step(RUN_WLS)] * 9)
        with_reference = episode_reward(episode, expert_reference_calls=6, weights=weights)
        self.assertEqual(with_reference.tool_regret, 3)
        self.assertAlmostEqual(with_reference.total, 1.0 - 0.3)
        shorter = episode_reward(episode, expert_reference_calls=12, weights=weights)
        self.assertEqual(shorter.tool_regret, 0)
        self.assertIsNone(episode_reward(episode, weights=weights).tool_regret)

    def test_weights_reject_negative_values_and_load_from_mapping(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RewardWeights(call_cost=value).validate()
        with self.assertRaises(ValueError):
            RewardWeights(call_cost=-1.0).validate()
        with self.assertRaises(ValueError):
            RewardWeights.from_mapping({"basis_credit": {"counterfactual_resolution": 1.5}})
        weights = RewardWeights.from_mapping({"call_cost": 0.03, "basis_credit": {"counterfactual_resolution": 1.0, "bounded_localization_handoff": 0.5}})
        self.assertEqual(weights.call_cost, 0.03)
        self.assertEqual(weights.basis_credit["bounded_localization_handoff"], 0.5)
        self.assertEqual(RewardWeights.from_mapping(None), RewardWeights())

    def test_expert_reference_loads_steps_per_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_text(json.dumps({"scenario_id": "r0_a", "steps": 6}) + "\n\n" + json.dumps({"scenario_id": "r0_b", "steps": 8}) + "\n", encoding="utf-8")
            self.assertEqual(load_expert_reference(path), {"r0_a": 6, "r0_b": 8})
        self.assertEqual(load_expert_reference(None), {})


if __name__ == "__main__":
    unittest.main()
