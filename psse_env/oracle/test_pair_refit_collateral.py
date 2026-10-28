"""The candidate-quality oracle accepts the declared partner row of an alternating pair refit (compound study, stage 2 item 3)."""
from __future__ import annotations

import unittest

from psse_env.oracle.candidate_quality import CandidateDisposition, CandidateQualityOracle
from psse_env.oracle.test_branch_first_partial import ACTION, PARENT, _verification

PAIR = {
    "state_id": "episode:s1",
    "parent_state_id": "episode:s0",
    # Both the target row (1) and the declared partner row (0) re-estimated.
    "case": {"branch": [{"r": 0.9, "x": 1.8, "status": 1}, {"r": 0.03, "x": 0.067, "status": 1}]},
    "measurements": [1.0],
    "metadata": {"last_parameter_correction": {"line_index": 2, "pair_refit_line_index": 1}},
}


def _label(candidate, **overrides):
    return CandidateQualityOracle(mode="deployment").label_candidate(
        parent_state=PARENT, source_action=ACTION, candidate_state=candidate,
        verification_output=_verification(**overrides),
    )


class PairRefitCollateralTests(unittest.TestCase):
    def test_a_declared_partner_row_is_the_corrections_own_work(self) -> None:
        result = _label(PAIR, global_progress=0.6, globally_resolved=False)
        self.assertNotEqual(result.progress_class, "healthy_component_corruption")
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)

    def test_an_undeclared_second_row_is_still_collateral(self) -> None:
        undeclared = dict(PAIR, metadata={"last_parameter_correction": {"line_index": 2}})
        result = _label(undeclared, global_progress=0.6, globally_resolved=False)
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "healthy_component_corruption")
        wrong_partner = dict(PAIR, metadata={"last_parameter_correction": {"line_index": 2, "pair_refit_line_index": 3}})
        self.assertEqual(
            _label(wrong_partner, global_progress=0.6, globally_resolved=False).progress_class,
            "healthy_component_corruption",
        )

    def test_the_partner_may_change_only_its_impedance_and_the_target_must_change(self) -> None:
        status_flip = dict(PAIR, case={"branch": [{"r": 0.9, "x": 1.8, "status": 0}, {"r": 0.03, "x": 0.067, "status": 1}]})
        self.assertEqual(
            _label(status_flip, global_progress=0.6, globally_resolved=False).progress_class,
            "healthy_component_corruption",
        )
        untouched_target = dict(PAIR, case={"branch": [{"r": 0.9, "x": 1.8, "status": 1}, {"r": 0.03, "x": 0.09, "status": 1}]})
        self.assertEqual(
            _label(untouched_target, global_progress=0.6, globally_resolved=False).progress_class,
            "healthy_component_corruption",
        )


if __name__ == "__main__":
    unittest.main()
