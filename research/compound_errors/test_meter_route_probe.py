"""The meter-route probe's tabulation (2026-10-29)."""
from __future__ import annotations

import unittest

from research.compound_errors.meter_route_probe import compare, cue_outcomes, meter_commits, wrong_commits


def _trace() -> list[dict]:
    tools = ["correct_measurements", "run_wls", "commit_state", "get_parameter_context", "correct_parameters", "run_wls",
             "rollback_state", "get_measurement_context", "correct_measurements", "run_wls", "commit_state",
             "correct_parameters", "run_wls", "commit_state"]
    trace = [{"step": i, "tool": tool, "status": "success"} for i, tool in enumerate(tools)]
    trace[1]["metrics"] = {"max_normalized_residual": 15.7, "globally_resolved": False}
    trace[5]["metrics"] = {"max_normalized_residual": 15.6, "globally_resolved": False}
    trace[9]["metrics"] = {"max_normalized_residual": 9.0, "globally_resolved": False}
    trace[12]["metrics"] = {"max_normalized_residual": 2.0, "globally_resolved": True}
    return trace


def _episode(sid: str, success: bool, *, steps: int = 14, wrong_meter: bool = False) -> dict:
    corrections = [
        {"tool": "correct_measurements", "step": 0, "lifecycle": "commit_state", "target_is_true": not wrong_meter},
        {"tool": "correct_parameters", "step": 4, "lifecycle": "rollback_state", "target_is_true": False},
        {"tool": "correct_measurements", "step": 8, "lifecycle": "commit_state", "target_is_true": True},
        {"tool": "correct_parameters", "step": 11, "lifecycle": "commit_state", "target_is_true": True},
    ]
    return {"scenario_id": sid, "class": "meter_rx2", "success": success, "steps": steps, "trace": _trace(),
            "corrections": corrections, "escalation_request": None}


class MeterRouteProbeTests(unittest.TestCase):
    def test_each_alarm_on_meter_commit_is_read_with_the_next_true_correction(self) -> None:
        commits = meter_commits(_episode("r0", True), 11.0)
        self.assertEqual(commits, [
            {"step": 2, "max_normalized_residual": 15.7, "fires": True, "next_true": "correct_measurements"},
            {"step": 10, "max_normalized_residual": 9.0, "fires": False, "next_true": "correct_parameters"},
        ])
        self.assertEqual(cue_outcomes([_episode("r0", True)], 11.0), {"fires": {"meter": 1}, "quiet": {"line": 1}})
        # A rolled-back line is not a wrong commit; a committed healthy meter is.
        self.assertEqual(wrong_commits(_episode("r0", True)), {})
        self.assertEqual(wrong_commits(_episode("r0", True, wrong_meter=True)), {"correct_measurements": 1})

    def test_roots_are_paired_by_id(self) -> None:
        baseline = {"r0": _episode("r0", False), "r1": _episode("r1", True), "r2": _episode("r2", True, steps=30)}
        rule = {"r0": _episode("r0", True), "r1": _episode("r1", False, wrong_meter=True), "r2": _episode("r2", True, steps=14)}
        cell = compare(baseline, rule, 11.0)["meter_rx2"]
        self.assertEqual((cell["n"], cell["baseline_solved"], cell["rule_solved"], cell["gained"], cell["lost"]), (3, 2, 2, 1, 1))
        self.assertEqual(cell["mean_actions_both_solved"], {"baseline": 30, "rule": 14, "roots": 1})
        self.assertEqual(cell["episodes_with_wrong_meter_commit"], {"baseline": 0, "rule": 1})
        self.assertEqual(cell["rule_fired_episodes"], 3)
        self.assertEqual(sorted(cell["lost_roots"]), ["r1"])
        self.assertEqual(cell["lost_roots"]["r1"]["rule"]["wrong_commits"], {"correct_measurements": 1})


if __name__ == "__main__":
    unittest.main()
