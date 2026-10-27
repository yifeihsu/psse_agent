from __future__ import annotations

import unittest

from research.rl.hard_draw_report import build_report, markdown, root_stratum


def _root(root: str, family: str, stratum: str | None = None) -> dict:
    audit = {"parameter_ranking": {"stratum": stratum}} if stratum else {}
    return {"execution": {"scenario_id": root}, "grouping": {"scenario_family": family}, "audit": audit}


class HardDrawReportTests(unittest.TestCase):
    def test_strata_tags_and_expert_ceilings(self) -> None:
        suite = [_root("a", "parameter", "misranked"), _root("b", "parameter", "ambiguous"),
                 _root("c", "parameter", "ambiguous"), _root("d", "measurement"), _root("e", "harmonic")]
        self.assertEqual(root_stratum(suite[0]), "parameter:misranked")
        self.assertEqual(root_stratum(suite[3]), "measurement")
        compound = {"execution": {"scenario_id": "z"}, "grouping": {"scenario_family": "compound"},
                    "audit": {"compound": {"class": "rx2_adjacent"}, "parameter_ranking": {"stratum": "ambiguous"}}}
        self.assertEqual(root_stratum(compound), "compound:rx2_adjacent")
        tags = [
            {"scenario_id": "a", "tag": "not_separable", "distinguishable_by_final_verdict": True},
            {"scenario_id": "b", "tag": "separable_by_verification"},
            {"scenario_id": "c", "tag": "not_separable", "distinguishable_by_final_verdict": False},
            {"scenario_id": "d", "tag": "separable_by_verification"},
            {"scenario_id": "e", "tag": "no_balanced_target"},
        ]
        experts = {
            "baseline": [{"scenario_id": "a", "truth_audited_task_success": False, "steps": 38},
                         {"scenario_id": "b", "truth_audited_task_success": True, "steps": 8},
                         {"scenario_id": "c", "truth_audited_task_success": False, "steps": 12},
                         {"scenario_id": "d", "truth_audited_task_success": True, "steps": 6},
                         {"scenario_id": "e", "truth_audited_task_success": True, "steps": 4}],
        }
        report = build_report(suite, tags=tags, experts=experts, with_effect=False)
        self.assertEqual(report["roots"], 5)
        self.assertEqual(report["strata"]["parameter:ambiguous"]["roots"], 2)
        self.assertEqual(report["strata"]["parameter:ambiguous"]["experts"]["baseline"]["success"], 1)
        self.assertEqual(report["strata"]["parameter:misranked"]["not_separable_but_distinguishable_by_final_verdict"], 1)
        self.assertEqual(report["experts"]["baseline"]["success"], 3)
        self.assertAlmostEqual(report["experts"]["baseline"]["success_rate"], 0.6)
        self.assertAlmostEqual(report["separable_fraction_of_balanced_roots"], 0.5)
        self.assertEqual(report["decision"], "hard enough (best expert 60-95%)")
        text = markdown(report)
        self.assertIn("| parameter:misranked | 1 |", text)
        self.assertIn("| all | 5 |", text)

    def test_decision_thresholds(self) -> None:
        suite = [_root("a", "measurement")]
        easy = build_report(suite, experts={"x": [{"scenario_id": "a", "truth_audited_task_success": True, "steps": 6}]}, with_effect=False)
        self.assertEqual(easy["decision"], "not hard enough (best expert above 95%)")
        hard = build_report(suite, experts={"x": [{"scenario_id": "a", "truth_audited_task_success": False, "steps": 6}]}, with_effect=False)
        self.assertTrue(hard["decision"].startswith("very hard"))
        self.assertIsNone(build_report(suite, with_effect=False)["decision"])


if __name__ == "__main__":
    unittest.main()
