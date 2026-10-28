"""The parameter-versus-open-terminal confidence the provider reports beside the cue (compound study, item 7 step 2)."""
from __future__ import annotations

import unittest

from psse_env.providers.matpower import MatpowerDeploymentProviders, measurement_index_map

NB, NL = 14, 20
SIGMA = 0.01


def _solved(lines: dict[int, dict[str, float]], multipliers: dict[int, float]) -> dict:
    """A solve bundle: each line row0 in ``lines`` reads p_meas/q (sigmas) and p_resid; ``multipliers`` gives |lambda| per row0."""
    index_map = measurement_index_map(NB, NL)
    nz = max(block.stop for block in index_map.values())
    z = [0.0] * nz
    residual = [0.0] * nz
    for row0, spec in lines.items():
        z[index_map["Pf"].start + row0] = spec.get("p_meas", 0.0) * SIGMA
        z[index_map["Pt"].start + row0] = -spec.get("p_meas", 0.0) * SIGMA
        z[index_map["Qf"].start + row0] = spec.get("q_max", 5.0) * SIGMA
        z[index_map["Qt"].start + row0] = spec.get("q_min", 1.0) * SIGMA
        residual[index_map["Pf"].start + row0] = spec.get("p_resid", 0.0)
    lambda_values = [0.0] * (2 * NL)
    for row0, value in multipliers.items():
        lambda_values[2 * row0 + 1] = value
    return {"payload": {"r": residual, "lambdaN": lambda_values}, "z": z, "index_map": index_map,
            "nb": NB, "nl": NL, "noise_options": {"measurement_sigma": [SIGMA] * nz}}


class RouteConfidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.providers = MatpowerDeploymentProviders(chi2_alpha=0.01, evidence_profile="scada_only", flow_at_noise_cue=True)

    def test_an_open_terminal_on_the_strongest_candidate_is_the_likely_hypothesis(self) -> None:
        confidence = self.providers._route_confidence(_solved({4: {"p_meas": 1.0, "p_resid": 20.0}}, {4: 30.0, 7: 10.0}))
        self.assertEqual((confidence["line_index"], confidence["topology"], confidence["basis"]), (5, 1.0, "open_terminal_cue"))
        self.assertEqual(confidence["parameter"], 0.0)

    def test_a_cued_line_behind_a_stronger_candidate_keeps_the_parameter_route(self) -> None:
        # The hard-draw pattern of section 7.3: a starved healthy line shows the cue
        # while another line carries the strongest multiplier.
        confidence = self.providers._route_confidence(_solved(
            {5: {"p_meas": 1.2, "p_resid": 22.7}, 2: {"p_meas": 40.0, "p_resid": 30.0}}, {2: 47.9, 5: 17.0}))
        self.assertEqual(confidence["line_index"], 6)
        self.assertAlmostEqual(confidence["topology"], round(17.0 / 47.9, 3))
        self.assertLess(confidence["topology"], 0.5)

    def test_a_carrying_line_cannot_be_open_and_a_quiet_one_is_ambiguous(self) -> None:
        carrying = self.providers._route_confidence(_solved({4: {"p_meas": 12.0, "p_resid": 20.0}}, {4: 30.0}))
        self.assertEqual((carrying["topology"], carrying["basis"]), (0.0, "measurable_flow"))
        quiet = self.providers._route_confidence(_solved({4: {"p_meas": 1.0, "p_resid": 4.0}}, {4: 30.0}))
        self.assertEqual((quiet["topology"], quiet["basis"]), (0.5, "flow_at_noise_without_cue"))

    def test_no_multiplier_reports_nothing(self) -> None:
        self.assertIsNone(self.providers._route_confidence(_solved({4: {"p_meas": 1.0, "p_resid": 20.0}}, {})))

    def test_breaker_multipliers_raise_support_only_when_the_flip_lowers_the_objective(self) -> None:
        solved = _solved({4: {"p_meas": 1.0, "p_resid": 5.0}}, {4: 30.0, 7: 25.0})
        base = self.providers._route_confidence(solved)
        self.assertEqual(base["topology"], 0.5)
        findings = [
            {"cb_name": "CB_A", "line_index": 5, "score": 8.0, "gse_progress_after_flip": 0.4},
            {"cb_name": "CB_B", "line_index": None, "score": 10.0, "gse_progress_after_flip": 0.9},
        ]
        refined = self.providers._breaker_refined_confidence(solved, findings)
        self.assertEqual(refined["topology"], 0.8)
        self.assertEqual(refined["breaker_evidence"], {"cb_name": "CB_A", "score_ratio": 0.8, "flip_lowers_objective": True})
        findings[0]["gse_progress_after_flip"] = -0.1
        unchanged = self.providers._breaker_refined_confidence(solved, findings)
        self.assertEqual(unchanged["topology"], 0.5)
        self.assertFalse(unchanged["breaker_evidence"]["flip_lowers_objective"])
        self.assertIsNone(self.providers._breaker_refined_confidence(solved, [])["breaker_evidence"])


if __name__ == "__main__":
    unittest.main()
