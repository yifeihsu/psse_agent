from __future__ import annotations

import unittest

from psse_env.examples.generate_round0_aggregate import _terminal_scenario_matrix


class TerminalScenarioMatrixTests(unittest.TestCase):
    def test_matrix_requires_every_root_terminal_and_unquarantined(self) -> None:
        matrix = _terminal_scenario_matrix(
            [
                {
                    "scenario_id": "m1",
                    "scenario_family": "measurement",
                    "terminal": True,
                    "terminal_outcome": "resolved",
                    "quarantined": False,
                },
                {
                    "scenario_id": "mt1",
                    "scenario_family": "measurement+topology",
                    "terminal": False,
                    "quarantined": False,
                },
                {
                    "scenario_id": "mt2",
                    "scenario_family": "measurement+topology",
                    "terminal": True,
                    "terminal_outcome": "operator_escalation",
                    "quarantined": True,
                },
            ]
        )

        self.assertTrue(matrix["measurement"]["release_terminal_coverage"])
        self.assertTrue(matrix["measurement"]["release_resolution_coverage"])
        self.assertEqual(
            matrix["measurement"]["terminal_outcome_counts"], {"resolved": 1}
        )
        mixed = matrix["measurement+topology"]
        self.assertEqual(mixed["episodes"], 2)
        self.assertEqual(mixed["terminal_episodes"], 1)
        self.assertEqual(mixed["nonterminal_episode_ids"], ["mt1"])
        self.assertEqual(mixed["quarantined_episode_ids"], ["mt2"])
        self.assertEqual(mixed["operator_escalation_episode_ids"], ["mt2"])
        self.assertEqual(
            mixed["terminal_outcome_counts"], {"operator_escalation": 1}
        )
        self.assertFalse(mixed["release_terminal_coverage"])
        self.assertFalse(mixed["release_resolution_coverage"])

    def test_escalation_is_terminal_but_not_resolution_coverage(self) -> None:
        matrix = _terminal_scenario_matrix(
            [
                {
                    "scenario_id": "mh1",
                    "scenario_family": "measurement+hif",
                    "terminal": True,
                    "terminal_outcome": "operator_escalation",
                    "quarantined": False,
                }
            ]
        )

        hif = matrix["measurement+hif"]
        self.assertTrue(hif["release_terminal_coverage"])
        self.assertFalse(hif["release_resolution_coverage"])
        self.assertEqual(hif["resolution_rate"], 0.0)

    def test_unknown_terminal_outcome_is_not_release_terminal_coverage(self) -> None:
        matrix = _terminal_scenario_matrix(
            [
                {
                    "scenario_id": "legacy",
                    "scenario_family": "measurement",
                    "terminal": True,
                    "quarantined": False,
                }
            ]
        )
        entry = matrix["measurement"]
        self.assertEqual(entry["unknown_terminal_outcome_episode_ids"], ["legacy"])
        self.assertFalse(entry["release_terminal_coverage"])


if __name__ == "__main__":
    unittest.main()
