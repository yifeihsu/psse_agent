"""The target-first partial acceptance of the candidate-quality oracle (compound-error study, stage 2 item 4)."""
from __future__ import annotations

import unittest

from psse_env.oracle.candidate_quality import CandidateDisposition, CandidateQualityOracle
from psse_env.oracle.test_branch_first_partial import ACTION, CANDIDATE, PARENT, _verification


def _label(oracle, **overrides):
    return oracle.label_candidate(
        parent_state=PARENT, source_action=ACTION, candidate_state=CANDIDATE,
        verification_output=_verification(**overrides),
    )


# The true line repaired beside an open fault that is not a residual outlier
# (a second line, say): the branch-first waiver does not apply and the floor
# refuses it although its own target test clears.
QUIET = {"unresolved_signatures": ["wls_branch_multiplier_dominant line_status_or_parameter line=4"]}


class TargetFirstPartialTests(unittest.TestCase):
    def test_default_and_branch_first_refuse_below_the_floor_without_an_outlier(self) -> None:
        for oracle in (CandidateQualityOracle(mode="deployment"),
                       CandidateQualityOracle(mode="deployment", branch_first_partial=True)):
            result = _label(oracle, **QUIET)
            self.assertEqual(result.disposition, CandidateDisposition.REJECT)
            self.assertEqual(result.progress_class, "insufficient_global_progress")

    def test_target_first_keeps_a_fixed_target_with_nonnegative_progress(self) -> None:
        oracle = CandidateQualityOracle(mode="deployment", target_first_partial=True)
        result = _label(oracle, **QUIET)
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)
        self.assertEqual(result.progress_class, "target_first_partial")
        self.assertIn("global_partial_floor_waived", result.rationale_codes)
        zero = _label(oracle, global_progress=0.0, **QUIET)
        self.assertEqual(zero.disposition, CandidateDisposition.ACCEPT_PARTIAL)

    def test_target_first_still_refuses_regression_and_an_unfixed_target(self) -> None:
        oracle = CandidateQualityOracle(mode="deployment", target_first_partial=True)
        regressed = _label(oracle, global_progress=-0.05, **QUIET)
        self.assertEqual(regressed.disposition, CandidateDisposition.REJECT)
        self.assertEqual(regressed.progress_class, "insufficient_global_progress")
        unfixed = _label(oracle, target_fixed=False, target_progress=0.1, target_metric_value=9.5, **QUIET)
        self.assertEqual(unfixed.disposition, CandidateDisposition.REJECT)
        self.assertEqual(unfixed.progress_class, "no_target_progress")

    def test_branch_first_keeps_precedence_when_both_are_on(self) -> None:
        oracle = CandidateQualityOracle(mode="deployment", branch_first_partial=True, target_first_partial=True)
        self.assertEqual(_label(oracle).progress_class, "branch_first_partial")
        self.assertEqual(_label(oracle, **QUIET).progress_class, "target_first_partial")

    def test_an_open_terminal_cue_in_the_verified_solve_closes_the_route(self) -> None:
        oracle = CandidateQualityOracle(mode="deployment", target_first_partial=True)
        cued = _label(oracle, flow_at_noise_lines=[7], **QUIET)
        self.assertEqual(cued.disposition, CandidateDisposition.REJECT)
        self.assertEqual(cued.progress_class, "insufficient_global_progress")
        self.assertEqual(_label(oracle, flow_at_noise_lines=[], **QUIET).progress_class, "target_first_partial")

    def test_only_a_parameter_target_takes_the_route(self) -> None:
        """A breaker edit's status and a meter edit's residual are not evidential target tests: the floor stays."""
        oracle = CandidateQualityOracle(mode="deployment", target_first_partial=True)
        parent = {"state_id": "episode:s0", "case": {"branch": [{"r": 1.0, "x": 2.0, "status": 1}, {"r": 0.03, "x": 0.09, "status": 1}]}, "measurements": [1.0, 2.0]}
        flipped = {"state_id": "episode:s1", "parent_state_id": "episode:s0",
                   "case": {"branch": [{"r": 1.0, "x": 2.0, "status": 1}, {"r": 0.03, "x": 0.09, "status": 0}]}, "measurements": [1.0, 2.0]}
        topology = {"tool": "correct_topology", "arguments": {"state_id": "episode:s0", "branch_row0": 1, "status": 0}}
        result = oracle.label_candidate(parent_state=parent, source_action=topology, candidate_state=flipped,
                                        verification_output=_verification(topology_target_status_matches_requested=True, **QUIET))
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "insufficient_global_progress")
        meter = {"tool": "correct_measurements", "arguments": {"state_id": "episode:s0", "suspect_group": [1]}}
        edited = {"state_id": "episode:s1", "parent_state_id": "episode:s0", "case": parent["case"], "measurements": [1.0, 2.5]}
        result = oracle.label_candidate(parent_state=parent, source_action=meter, candidate_state=edited,
                                        verification_output=_verification(global_progress=0.1, **QUIET))
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertEqual(result.progress_class, "insufficient_global_progress")

    def test_the_option_is_off_in_the_deployment_provider_by_default(self) -> None:
        from psse_env.providers.matpower import MatpowerDeploymentProviders

        providers = MatpowerDeploymentProviders(chi2_alpha=0.01, evidence_profile="scada_only")
        self.assertFalse(providers._deployment_candidate_quality_oracle().target_first_partial)
        enabled = MatpowerDeploymentProviders(chi2_alpha=0.01, evidence_profile="scada_only", target_first_partial=True)
        self.assertTrue(enabled._deployment_candidate_quality_oracle().target_first_partial)


if __name__ == "__main__":
    unittest.main()
