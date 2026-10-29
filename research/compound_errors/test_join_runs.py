"""Joining a rollout that ran in parts (2026-10-29)."""
from __future__ import annotations

import unittest

from research.compound_errors.join_runs import join, joined_summary


def _episode(sid: str, success: bool) -> dict:
    return {"scenario_id": sid, "class": "control_meter", "success": success, "steps": 9, "corrections": [], "trace": []}


class JoinRunsTests(unittest.TestCase):
    def test_the_parts_are_joined_in_suite_order(self) -> None:
        records = join(["r0", "r1", "r2"], [[_episode("r0", True)], [_episode("r2", False), _episode("r1", True)]])
        self.assertEqual([r["scenario_id"] for r in records], ["r0", "r1", "r2"])
        summary = joined_summary(records, {"expert": "baseline_compare", "max_steps": 16, "workers": 6, "total_seconds": 12.0}, [1, 2])
        self.assertEqual((summary["totals"]["n"], summary["totals"]["success"]), (3, 2))
        self.assertEqual((summary["expert"], summary["max_steps"], summary["parts"]), ("baseline_compare", 16, [1, 2]))
        self.assertIsNone(summary["workers"])  # a joined run has no single worker count or run time
        self.assertIsNone(summary["total_seconds"])

    def test_the_parts_must_cover_the_suite_once(self) -> None:
        with self.assertRaises(ValueError):
            join(["r0", "r1"], [[_episode("r0", True)]])
        with self.assertRaises(ValueError):
            join(["r0"], [[_episode("r0", True)], [_episode("r0", False)]])
        with self.assertRaises(ValueError):
            join(["r0"], [[_episode("r0", True), _episode("r9", True)]])


if __name__ == "__main__":
    unittest.main()
