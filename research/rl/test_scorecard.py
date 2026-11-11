"""The paired scorecard and the pilot rules, on hand-written evaluation records."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.rl.scorecard import (
    build_scorecard,
    exact_sign_test,
    load_policy,
    markdown,
    paired,
    promotion_check,
    stop_rule,
    summarize_policy,
    tool_calls_on_shared_solves,
)

COMMIT, ROLLBACK = "commit_state", "rollback_state"


def _decision(*, true: bool, declined: bool) -> dict:
    return {"target_is_true": true, "declined": declined, "tool": ROLLBACK if declined else COMMIT}


def _episode(root: str, name: str, *, success: bool, steps: int, decisions=(), invalid: int = 0, false_commits: int = 0,
             loop: bool = False, corrupted: bool = False, tokens: int | None = None) -> dict:
    cost = {"internal_wls_calls": 2, "internal_correction_calls": 1, "pair_refits": 0, "policy_seconds": 1.5, "tool_seconds": 3.0}
    if tokens is not None:
        cost.update(input_tokens=tokens, output_tokens=20 * steps)
    return {"scenario_id": root, "compound_class": name, "truth_audited_task_success": success, "policy_steps": steps, "steps": steps,
            "invalid_action_count": invalid, "false_commit_count": false_commits, "false_finalization_count": 0, "loop_detected": loop,
            "reward": {"healthy_component_corrupted": corrupted}, "lifecycle": list(decisions), "cost": cost,
            "prompt_limit_refusals": 0}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _files(directory: Path) -> dict[str, Path]:
    baseline = [
        _episode("r1", "cb2_split", success=False, steps=14, decisions=[_decision(true=False, declined=False)], corrupted=True),
        _episode("r2", "rx_cb_meter", success=True, steps=12, decisions=[_decision(true=True, declined=False)] * 2),
        _episode("r3", "rx2_distant", success=True, steps=10, decisions=[_decision(true=True, declined=False)]),
        _episode("r4", "rx2_distant", success=False, steps=30, invalid=4, loop=True),
        _episode("r5", "control_rx", success=True, steps=8, decisions=[_decision(true=True, declined=False)], invalid=1),
    ]
    learner = [
        # The wrong line declined, the true one kept, the case solved: the recovery the series is after.
        _episode("r1", "cb2_split", success=True, steps=18, tokens=120000,
                 decisions=[_decision(true=False, declined=True), _decision(true=True, declined=False)]),
        # A true second candidate declined in a class with a breaker and a line fault: the case is lost.
        _episode("r2", "rx_cb_meter", success=False, steps=20, tokens=150000,
                 decisions=[_decision(true=True, declined=False), _decision(true=True, declined=True)]),
        _episode("r3", "rx2_distant", success=True, steps=9, tokens=60000, decisions=[_decision(true=True, declined=False)]),
        _episode("r4", "rx2_distant", success=True, steps=16, tokens=110000, decisions=[_decision(true=True, declined=False)]),
        _episode("r5", "control_rx", success=True, steps=8, tokens=50000, decisions=[_decision(true=True, declined=False)]),
    ]
    expert = {"r1": {"success": False, "clean": False, "steps": 13, "wrong_kept": 1, "class": "cb2_split"},
              "r2": {"success": True, "clean": True, "steps": 12, "wrong_kept": 0, "class": "rx_cb_meter"},
              "r3": {"success": True, "clean": True, "steps": 11, "wrong_kept": 0, "class": "rx2_distant"},
              "r4": {"success": False, "clean": False, "steps": 25, "wrong_kept": 0, "class": "rx2_distant"},
              "r5": {"success": True, "clean": True, "steps": 8, "wrong_kept": 0, "class": "control_rx"}}
    teacher_run = [{"scenario_id": "r1", "class": "cb2_split", "success": False, "steps": 13, "invalid_action_count": 0,
                    "false_commit_count": 0, "loop_detected": False, "healthy_components_preserved": False,
                    "corrections": [{"lifecycle": COMMIT, "target_is_true": False}, {"lifecycle": ROLLBACK, "target_is_true": True}]}]
    expert_path = directory / "expert.json"
    expert_path.write_text(json.dumps(expert), encoding="utf-8")
    return {"baseline": _write(directory / "baseline.jsonl", baseline), "learner": _write(directory / "learner.jsonl", learner),
            "expert": expert_path, "teacher_run": _write(directory / "teacher.jsonl", teacher_run)}


class ScorecardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.files = _files(Path(self.directory.name))
        self.learner = load_policy(self.files["learner"])
        self.baseline = load_policy(self.files["baseline"])
        self.expert = load_policy(self.files["expert"])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_the_three_kinds_of_file_are_read_into_one_record(self) -> None:
        self.assertEqual(sorted(self.learner), ["r1", "r2", "r3", "r4", "r5"])
        self.assertEqual((self.baseline["r5"]["success"], self.baseline["r5"]["clean"]), (True, False))  # a refused action
        self.assertEqual((self.baseline["r1"]["wrong_kept"], self.baseline["r1"]["healthy_corrupted"]), (1, True))
        self.assertEqual((self.expert["r2"]["steps"], self.expert["r2"]["decisions"], self.expert["r2"]["invalid_actions"]), (12, None, None))
        teacher = load_policy(self.files["teacher_run"])["r1"]
        self.assertEqual((teacher["wrong_kept"], teacher["healthy_corrupted"], teacher["clean"], teacher["decisions"]), (1, True, False, None))

    def test_a_policy_is_summarized_with_its_decisions_and_what_they_led_to(self) -> None:
        summary = summarize_policy(self.learner)
        self.assertEqual((summary["roots"], summary["solved"], summary["clean"]), (5, 4, 4))
        decisions = summary["decisions"]
        self.assertEqual(decisions["true_target"], {"candidates": 6, "declined": 1, "rate": 1 / 6})
        self.assertEqual((decisions["true_first_candidate"]["declined"], decisions["true_later_candidate"]["declined"]), (0, 1))
        self.assertEqual(decisions["true_in_mixed_classes"], {"candidates": 2, "declined": 1, "rate": 0.5})
        self.assertEqual(decisions["wrong_target"], {"candidates": 1, "declined": 1, "rate": 1.0})
        self.assertEqual(decisions["recovery"], {"wrong_target_declined": 1, "followed_by_a_kept_true_correction": 1, "and_solved": 1,
                                                 "episodes": 1, "episodes_solved": 1})
        self.assertEqual((summary["wrong_targets_kept"], summary["healthy_component_corrupted_episodes"]), (0, 0))
        self.assertEqual(summary["by_class"]["rx2_distant"], {"roots": 2, "solved": 2, "clean": 2})
        self.assertEqual((summary["cost"]["input_tokens"], summary["cost"]["internal_wls_calls"]), (490000, 10))
        self.assertAlmostEqual(summary["cost"]["mean_tool_calls"], 71 / 5)
        baseline = summarize_policy(self.baseline)
        self.assertEqual((baseline["solved"], baseline["clean"], baseline["wrong_targets_kept"], baseline["process"]["loop_episodes"]),
                         (3, 2, 1, 1))
        self.assertIsNone(baseline["cost"]["input_tokens"])  # that evaluation recorded no token counts
        expert = summarize_policy(self.expert)
        self.assertEqual((expert["solved"], expert["clean"], expert["decisions"], expert["process"]["invalid_actions"]), (3, 3, None, None))

    def test_the_paired_counts_show_both_sides_and_the_exact_test(self) -> None:
        result = paired(self.learner, self.baseline)
        self.assertEqual((result["both"], result["learner_only"], result["reference_only"], result["neither"], result["net"]),
                         (2, 2, 1, 0, 1))
        self.assertEqual((result["learner_only_roots"], result["reference_only_roots"]), (["r1", "r4"], ["r2"]))
        self.assertAlmostEqual(result["p_exact_two_sided"], 1.0)
        self.assertEqual(paired(self.learner, self.baseline, "clean")["net"], 2)
        self.assertEqual((exact_sign_test(0, 0), exact_sign_test(6, 0), exact_sign_test(8, 1)), (1.0, 0.03125, 0.0390625))
        self.assertAlmostEqual(exact_sign_test(10, 2), 0.03857421875)
        calls = tool_calls_on_shared_solves(self.learner, self.baseline)
        self.assertEqual((calls["roots"], calls["learner"], calls["reference"], calls["fewer"], calls["same"], calls["more"]),
                         (2, 8.5, 9.0, 1, 1, 0))
        self.assertEqual(paired(self.learner, {"r1": self.baseline["r1"]})["roots_missing_in_one"], 4)

    def test_the_scorecard_and_its_promotion_rule(self) -> None:
        card = build_scorecard({"S1": self.learner}, {"S0": self.baseline, "expert": self.expert}, expert="expert", matched="S0")
        entry = card["learners"]["S1"]
        self.assertEqual(entry["expert_failures_recovered"], {"expert_failed_roots": 2, "recovered": 2, "roots": ["r1", "r4"],
                                                              "learner_roots_without_an_expert_record": 0})
        self.assertEqual(entry["against"]["expert"]["solved"]["net"], 1)
        promotion = entry["promotion"]
        self.assertEqual(promotion["conditions"], {"evaluated_on_the_same_roots": True, "positive_paired_net_solves": True,
                                                   "expert_failed_roots_recovered": True, "nonnegative_paired_clean_change": True,
                                                   "no_more_healthy_component_corruption": True})
        self.assertTrue(promotion["promote"])
        self.assertEqual(promotion["coverage"], {"learner_roots": 5, "matched_reference_roots": 5, "learner_roots_missing": 0,
                                                 "reference_roots_missing": 0, "learner_roots_without_an_expert_record": 0})
        self.assertEqual(promotion["inspect"]["true_candidates_declined_in_mixed_classes"]["declined"], 1)
        self.assertEqual(promotion["counts"]["healthy_component_corrupted_episodes"], [0, 1])
        # No gain, or a gain bought with a clean-success loss, is not promoted.
        same = build_scorecard({"S1": self.baseline}, {"S0": self.baseline, "expert": self.expert}, expert="expert", matched="S0")
        self.assertFalse(same["learners"]["S1"]["promotion"]["promote"])
        # Against a reference that records no corruption the rule cannot be decided and says so.
        against_expert = promotion_check(entry, card["references"]["expert"], matched="expert")
        self.assertEqual((against_expert["promote"], against_expert["undetermined"]), (False, ["no_more_healthy_component_corruption"]))
        with self.assertRaisesRegex(ValueError, "not among the references"):
            build_scorecard({"S1": self.learner}, {"S0": self.baseline}, expert="expert")
        seeds = build_scorecard({"seed1": self.learner, "seed2": self.baseline}, {"S0": self.baseline})
        self.assertEqual(seeds["across_learners"], {"labels": ["seed1", "seed2"], "solved": [4, 3], "min": 3, "max": 4, "mean": 3.5})
        text = markdown(card, title="Arm B, iteration 1")
        for needle in ("# Arm B, iteration 1", "| solved | 3 | 3 | 4 |", "| true candidates declined | 0 of 4 |  | 1 of 6 |",
                       "| S0 | 5 | 2 | 2 | 1 | +1 | 1.000 | +2 |", "Roots the expert fails that the learner solves: 2 of 2 (r1, r4).",
                       "Promotion rule against S0: promote.", "| rx2_distant | 2 | 1 | 1 | 2 |"):
            self.assertIn(needle, text)

    def test_an_evaluation_that_did_not_finish_is_not_promoted(self) -> None:
        references = {"S0": self.baseline, "expert": self.expert}
        # Only r1 was evaluated: the learner solves it, the reference and the expert do not, and nothing it has is worse.
        partial = {"r1": self.learner["r1"]}
        entry = build_scorecard({"S1": partial}, references, expert="expert", matched="S0")["learners"]["S1"]
        solved = entry["against"]["S0"]["solved"]
        self.assertEqual((solved["roots"], solved["net"], solved["learner_roots_missing"], solved["reference_roots_missing"]), (1, 1, 4, 0))
        promotion = entry["promotion"]
        self.assertEqual(promotion["conditions"], {"evaluated_on_the_same_roots": False, "positive_paired_net_solves": True,
                                                   "expert_failed_roots_recovered": True, "nonnegative_paired_clean_change": True,
                                                   "no_more_healthy_component_corruption": True})
        self.assertFalse(promotion["promote"])
        self.assertEqual((promotion["coverage"]["learner_roots"], promotion["coverage"]["learner_roots_missing"]), (1, 4))
        text = markdown(build_scorecard({"S1": partial}, references, expert="expert", matched="S0"))
        self.assertIn("Promotion rule against S0: do not promote. evaluated on the same roots: no;", text)
        self.assertIn("Not run on the same roots as S0: 4 of its roots have no learner record and 0 of the learner's roots are not in it. "
                      "The paired counts above are of the 1 roots they share.", text)
        # A learner run on a root the reference was not run on, or one the expert's record does not hold, is not either.
        extra = {**self.learner, "r9": self.learner["r1"]}
        wider = build_scorecard({"S1": extra}, references, expert="expert", matched="S0")["learners"]["S1"]["promotion"]
        self.assertEqual((wider["promote"], wider["conditions"]["evaluated_on_the_same_roots"]), (False, False))
        self.assertEqual((wider["coverage"]["reference_roots_missing"], wider["coverage"]["learner_roots_without_an_expert_record"]), (1, 1))
        narrow_expert = {root: record for root, record in self.expert.items() if root != "r5"}
        unknown = build_scorecard({"S1": self.learner}, {"S0": self.baseline, "expert": narrow_expert}, expert="expert",
                                  matched="S0")["learners"]["S1"]["promotion"]
        self.assertEqual((unknown["promote"], unknown["coverage"]["learner_roots_without_an_expert_record"]), (False, 1))
        # An expert's record that is wider than the evaluation is no gap: every evaluated root is in it.
        wide_expert = {**self.expert, "r9": self.expert["r1"]}
        self.assertTrue(build_scorecard({"S1": self.learner}, {"S0": self.baseline, "expert": wide_expert}, expert="expert",
                                        matched="S0")["learners"]["S1"]["promotion"]["promote"])

    def test_the_stop_rule_counts_completed_iterations_without_a_paired_gain(self) -> None:
        def nets(*values):
            return [{"net": value} for value in values]

        self.assertEqual(stop_rule(nets(-2, 1))["comparisons_on_unequal_roots"], 0)
        self.assertEqual(stop_rule([{"net": 1, "roots_missing_in_one": 4}, {"net": 0, "roots_missing_in_one": 0}])
                         ["comparisons_on_unequal_roots"], 1)

        first = stop_rule(nets(-2))
        self.assertEqual((first["stop"], first["screen_complete"], first["further_iterations_allowed"]), (False, False, 2))
        self.assertEqual(stop_rule(nets(-2, -1))["stop"], False)  # the screen is done; one more iteration is allowed
        self.assertEqual(stop_rule(nets(-2, -1))["further_iterations_allowed"], 1)
        third = stop_rule(nets(-2, -1, 0))
        self.assertEqual((third["stop"], third["iterations_without_gain_in_a_row"], third["further_iterations_allowed"]), (True, 3, 0))
        gained = stop_rule(nets(-2, 3, 0))
        self.assertEqual((gained["stop"], gained["iterations_without_gain_in_a_row"]), (False, 1))
        self.assertEqual(stop_rule(nets(1, 0, 0, 0))["stop"], True)
        self.assertEqual(stop_rule([])["completed_valid_iterations"], 0)


if __name__ == "__main__":
    unittest.main()
