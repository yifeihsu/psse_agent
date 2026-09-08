"""The branch-first partial waiver of the candidate-quality oracle.

A correct line repair on a measurement+parameter root removes only a small
share of the chi-square while the gross meter error remains, so the 30%
branch floor refuses it.  With the waiver on, the repair is kept as partial
progress when the line's own multiplier is resolved, global progress is not
negative, and a residual outlier is still flagged for the meter route.
"""

from __future__ import annotations

import unittest

from psse_env.oracle.candidate_quality import CandidateDisposition, CandidateQualityOracle

# In-memory cases so the structural collateral check can see that only the
# declared field of the corrected branch row changed (a path-valued case
# would fail closed without a case differ).
PARENT = {
    "state_id": "episode:s0",
    "case": {"branch": [{"r": 1.0, "x": 2.0, "status": 1}, {"r": 0.03, "x": 0.09, "status": 1}]},
    "measurements": [1.0],
}
CANDIDATE = {
    "state_id": "episode:s1",
    "parent_state_id": "episode:s0",
    "case": {"branch": [{"r": 1.0, "x": 2.0, "status": 1}, {"r": 0.03, "x": 0.067, "status": 1}]},
    "measurements": [1.0],
}
ACTION = {
    "tool": "correct_parameters",
    "arguments": {"state_id": "episode:s0", "branch_row0": 1, "parameter": "x"},
}


def _verification(**overrides):
    # The measured numbers of the rejected true-line repair on root
    # physical_v3_093c (line 13 with meter 102 still bad).
    verification = {
        "target_fixed": True,
        "target_progress": 0.8995,
        "target_metric_kind": "max_abs_branch_multiplier",
        "target_metric_value": 1.0565,
        "target_metric_threshold": 3.0,
        "parent_target_metric_value": 10.5123,
        "global_progress": 0.1514,
        "parent_anomaly_score": 7.2748,
        "remaining_anomaly_score": 6.1736,
        "globally_resolved": False,
        "post_action_resolved": False,
        "physical_constraints_ok": True,
        "unresolved_signatures": [
            "wls_residual_outlier_dominant index=102 channel=Qt",
            "wls_residual_outlier_dominant index=28 channel=Qinj",
        ],
    }
    verification.update(overrides)
    return verification


def _label(oracle, **overrides):
    return oracle.label_candidate(
        parent_state=PARENT,
        source_action=ACTION,
        candidate_state=CANDIDATE,
        verification_output=_verification(**overrides),
    )


class BranchFirstPartialTests(unittest.TestCase):
    def test_default_gate_refuses_the_true_line_below_the_branch_floor(self) -> None:
        result = _label(CandidateQualityOracle(mode="deployment"))
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "insufficient_global_progress")

    def test_waiver_keeps_a_resolved_branch_target_as_partial(self) -> None:
        oracle = CandidateQualityOracle(mode="deployment", branch_first_partial=True)
        result = _label(oracle)
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)
        self.assertEqual(result.progress_class, "branch_first_partial")
        self.assertIn("branch_partial_floor_waived", result.rationale_codes)
        self.assertIn("residual_outlier_remains", result.rationale_codes)

    def test_waiver_still_needs_local_resolution_progress_and_a_meter_remainder(self) -> None:
        oracle = CandidateQualityOracle(mode="deployment", branch_first_partial=True)
        # Root physical_v3_a89b: the meter next to line 20 keeps the line's
        # multiplier at 6.55, so the local test is not met and the repair is
        # still refused (target_fixed False on that solve).
        unresolved_target = _label(
            oracle,
            target_fixed=False,
            target_progress=0.5085,
            target_metric_value=6.5476,
            global_progress=0.1792,
        )
        self.assertEqual(unresolved_target.disposition, CandidateDisposition.REJECT)
        # A multiplier only marginally above threshold still passes the local
        # test; global regression or a missing meter outlier does not.
        marginal = _label(oracle, target_metric_value=3.6)
        self.assertEqual(marginal.disposition, CandidateDisposition.ACCEPT_PARTIAL)
        regressed = _label(oracle, global_progress=-0.02)
        self.assertEqual(regressed.disposition, CandidateDisposition.REJECT)
        no_meter = _label(
            oracle,
            unresolved_signatures=["wls_branch_multiplier line_status_or_parameter line=14"],
        )
        self.assertEqual(no_meter.disposition, CandidateDisposition.REJECT)
        self.assertEqual(no_meter.progress_class, "insufficient_global_progress")

    def test_waiver_never_applies_to_meter_corrections(self) -> None:
        oracle = CandidateQualityOracle(mode="deployment", branch_first_partial=True)
        result = oracle.label_candidate(
            parent_state=PARENT,
            source_action={
                "tool": "correct_measurements",
                "arguments": {"state_id": "episode:s0", "suspect_group": [102]},
            },
            candidate_state=CANDIDATE,
            verification_output=_verification(
                target_metric_kind="max_abs_normalized_residual",
                global_progress=0.05,
                measurement_target_cluster_size=1,
            ),
        )
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertNotIn("branch_partial_floor_waived", result.rationale_codes)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
