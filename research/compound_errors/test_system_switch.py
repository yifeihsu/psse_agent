"""The system switch of the compound-error scripts (2026-10-28): class shards, the suite's system, the progress basis."""
from __future__ import annotations

import unittest

from research.compound_errors.build_suite import BUS_BRANCH_PLAN, DEFAULT_PLAN, class_seed, class_shards, resolve_physics
from research.compound_errors.rollout import suite_system, system_progress_basis


class SystemSwitchTests(unittest.TestCase):
    def test_one_shard_keeps_the_class_seed_of_the_original_suites(self) -> None:
        self.assertEqual(class_shards(20261027, "rx2_adjacent", 150, 1), [(0, class_seed(20261027, "rx2_adjacent"), 150)])
        shards = class_shards(20261027, "meter_on_rx", 75, 10)
        self.assertEqual(sum(count for _, _, count in shards), 75)
        self.assertEqual(len({seed for _, seed, _ in shards}), 10)
        self.assertEqual(len(class_shards(20261027, "control_rx", 3, 10)), 3)

    def test_physics_and_plans_follow_the_system(self) -> None:
        self.assertEqual(resolve_physics("case14", "auto"), "node_breaker")
        self.assertEqual(resolve_physics("ieee118", "auto"), "bus_branch")
        self.assertEqual(resolve_physics("case14", "bus_branch"), "bus_branch")
        self.assertLessEqual(set(BUS_BRANCH_PLAN), set(DEFAULT_PLAN))
        self.assertEqual(sum(BUS_BRANCH_PLAN.values()), 625)

    def test_the_suite_names_its_system_and_a_larger_system_measures_progress_on_the_excess(self) -> None:
        roots = [{"grouping": {"case_id": "case118"}}] * 2
        self.assertEqual(suite_system(roots), "case118")
        self.assertEqual(suite_system(roots, "118"), "case118")
        with self.assertRaises(SystemExit):
            suite_system(roots, "case14")
        with self.assertRaises(SystemExit):
            suite_system([*roots, {"grouping": {"case_id": "case14"}}])
        self.assertEqual(suite_system([{"grouping": {}}]), "case14")
        self.assertIsNone(system_progress_basis("case14", None))
        self.assertEqual(system_progress_basis("case118", None), "chi_square_excess")
        self.assertEqual(system_progress_basis("case118", "chi_square_statistic"), "chi_square_statistic")


if __name__ == "__main__":
    unittest.main()
