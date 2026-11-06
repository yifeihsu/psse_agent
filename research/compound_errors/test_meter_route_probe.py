"""The meter-route probe's tabulation (2026-10-29)."""
from __future__ import annotations

import unittest

from research.compound_errors.meter_route_probe import (
    against_reference,
    compare,
    cue_outcomes,
    meter_commits,
    separation,
    wrong_commits,
)


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
            {"step": 2, "max_normalized_residual": 15.7, "fires": True, "next_true": "correct_measurements", "true": True},
            {"step": 10, "max_normalized_residual": 9.0, "fires": False, "next_true": "correct_parameters", "true": True},
        ])
        self.assertEqual(cue_outcomes([_episode("r0", True)], 11.0), {"fires": {"meter": 1}, "quiet": {"line": 1}})
        # Split by the committed meter itself: the first commit of this episode is on a healthy meter.
        wrong_first = [_episode("r0", True, wrong_meter=True)]
        self.assertEqual(meter_commits(wrong_first[0], 11.0)[0]["true"], False)
        self.assertEqual(cue_outcomes(wrong_first, 11.0, committed_true=False), {"fires": {"meter": 1}, "quiet": {}})
        self.assertEqual(cue_outcomes(wrong_first, 11.0, committed_true=True), {"fires": {}, "quiet": {"line": 1}})
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

    def test_changed_trajectories_and_the_quiet_root_check(self) -> None:
        def quiet(episode: dict) -> dict:
            episode["trace"][1]["metrics"] = {"max_normalized_residual": 9.5, "globally_resolved": False}
            return episode

        def other_meter(episode: dict) -> dict:  # what the agent did differs, not only the truth of it
            episode["corrections"][2]["arguments"] = {"suspect_group": [9]}
            return episode

        baseline = {"r0": _episode("r0", True), "r1": quiet(_episode("r1", True)), "r2": quiet(_episode("r2", True))}
        rule = {"r0": other_meter(_episode("r0", True)), "r1": quiet(_episode("r1", True, wrong_meter=True)),
                "r2": other_meter(quiet(_episode("r2", True, steps=12)))}
        cell = compare(baseline, rule, 11.0)["meter_rx2"]
        self.assertEqual(cell["trajectories_changed"], {"roots": 2, "action_change": -2})
        # r2 changed although baseline_compare never reached a firing state there: the check names it.
        self.assertEqual(cell["baseline_never_fires"], {"roots": 2, "changed": ["r2"]})

    def test_separation_reads_the_first_commit_with_the_alarm_on(self) -> None:
        episode = _episode("r0", True)
        episode["trace"][1]["metrics"]["chi_square_ratio"] = 3.2
        line_first = _episode("r1", True)
        line_first["trace"][1]["metrics"].update(chi_square_ratio=2.0, max_normalized_residual=8.0)
        line_first["trace"][0]["tool"] = "correct_parameters"
        self.assertEqual(separation([episode, line_first], 11.0), {
            "line -> meter": {"n": 1, "median": 8.0, "at_least_threshold": 0},
            "meter -> meter": {"n": 1, "median": 15.7, "at_least_threshold": 1},
        })

    def test_a_fresh_draw_is_set_beside_the_reference_probe(self) -> None:
        here = {"meter_rx2": {"n": 3, "baseline_solved": 2, "rule_solved": 2, "gained": 1, "lost": 1}}
        there = {"meter_rx2": {"n": 4, "baseline_solved": 1, "rule_solved": 3, "gained": 2, "lost": 0}}
        lines = against_reference({"ieee118": {"16": {"classes": here}}}, {"ieee118": {"16": {"classes": there}, "40": {"classes": there}}})
        self.assertIn("| class | 16 actions, reference | 16 actions, this draw |", lines)
        self.assertIn("| meter_rx2 (3) | 1 / 3 of 4 (+2, -0) | 2 / 2 (+1, -1) |", lines)
        self.assertEqual(against_reference({"ieee14_control": {"16": {"classes": here}}}, {"ieee118": {"16": {"classes": there}}}),
                         ["## Against the reference probe", ""])


if __name__ == "__main__":
    unittest.main()
