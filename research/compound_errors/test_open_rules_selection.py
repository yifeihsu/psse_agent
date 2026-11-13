"""The open-rules training selection on small made-up episodes: one source per case, the rows left out, the views."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.compound_errors import open_rules_selection as sel

RUN_WLS, CONTEXT, CORRECT, HANDOFF = "run_wls", "get_parameter_context", "correct_parameters", "ask_for_more_evidence"
MEASUREMENT_CONTEXT = "get_measurement_context"


def _step(index: int, tool: str, *, status: str = "success", arguments=None, **extra) -> dict:
    return dict({"step": index, "tool": tool, "status": status, "arguments": arguments or {}}, **extra)


def _truth_episode(case: str, *, success: bool = True, rule_expert_solves: bool | None = False, absent: bool = False,
                   verdict: str = "ACCEPT_FINAL") -> dict:
    """First solve, context, one correction, verification, commit, closing context, handoff."""
    kept = verdict in sel.REJECTED
    trace = [
        _step(0, RUN_WLS, kind="first_solve"),
        _step(1, CONTEXT, kind="context"),
        _step(2, CORRECT, kind="correction", arguments={"line_index": 3}, target_in_evidence="absent" if absent else "listed",
              **({"open_rules_used": ["correction_outside_context"]} if absent else {})),
        _step(3, RUN_WLS, kind="verify"),
        _step(4, sel.COMMIT, kind="commit", **({"open_rules_used": ["commit_rejected_candidate"]} if kept else {})),
        _step(5, MEASUREMENT_CONTEXT, kind="closing_context"),
        _step(6, HANDOFF, kind="handoff"),
    ]
    return {"scenario_id": case, "class": "meter_on_rx", "success": success, "rule_expert_solves": rule_expert_solves, "steps": len(trace),
            "trace": trace, "corrections": [{"step": 2, "tool": CORRECT, "execution_status": "success", "verdict": verdict,
                                             "lifecycle": sel.COMMIT, "target_is_true": True}]}


def _demonstration_episode(case: str, *, success: bool, decline: bool = False, rejected_true: bool = False, repeat: bool = False) -> dict:
    """A wrong target tried and rolled back (declined, or rejected), then the true target kept; optional extras after it."""
    first_verdict = "ACCEPT_PARTIAL" if decline else "REJECT"
    trace = [
        _step(0, RUN_WLS),
        _step(1, CONTEXT),
        _step(2, CORRECT, arguments={"line_index": 9}),
        _step(3, RUN_WLS),
        _step(4, sel.ROLLBACK, disposition=first_verdict),
        _step(5, CORRECT, arguments={"line_index": 3}),
        _step(6, RUN_WLS),
        _step(7, sel.COMMIT, disposition="ACCEPT_FINAL"),
    ]
    corrections = [
        {"step": 2, "tool": CORRECT, "execution_status": "success", "verdict": first_verdict, "lifecycle": sel.ROLLBACK, "target_is_true": False},
        {"step": 5, "tool": CORRECT, "execution_status": "success", "verdict": "ACCEPT_FINAL", "lifecycle": sel.COMMIT, "target_is_true": True},
    ]
    if rejected_true:
        base = len(trace)
        trace += [_step(base, CORRECT, arguments={"line_index": 4}), _step(base + 1, RUN_WLS), _step(base + 2, sel.ROLLBACK, disposition="REJECT")]
        corrections.append({"step": base, "tool": CORRECT, "execution_status": "success", "verdict": "REJECT", "lifecycle": sel.ROLLBACK,
                            "target_is_true": True})
    if repeat:
        base = len(trace)
        trace += [_step(base, CORRECT, status="failure", arguments={"line_index": 5}),
                  _step(base + 1, CORRECT, status="failure", arguments={"line_index": 5})]
        corrections += [{"step": base, "tool": CORRECT, "execution_status": "failure", "verdict": None, "lifecycle": None, "target_is_true": True},
                        {"step": base + 1, "tool": CORRECT, "execution_status": "failure", "verdict": None, "lifecycle": None, "target_is_true": True}]
    trace.append(_step(len(trace), HANDOFF))
    return {"scenario_id": case, "class": "rx2_adjacent", "success": success, "steps": len(trace), "trace": trace, "corrections": corrections,
            "gate_decisions": [{"step": 4, "declined": True, "target_is_true": False}] if decline else [],
            "declines": int(decline)}


def _rows(episodes: list[dict], tag: str) -> list[dict]:
    return [{"example_id": f"{episode['scenario_id']}:{tag}:{entry['step']:02d}", "scenario_id": episode["scenario_id"],
             "cloning": {"step": entry["step"], "tool": entry["tool"]}, "messages": []}
            for episode in episodes for entry in episode["trace"]]


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


class _Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.folder = Path(self._folder.name)
        fresh = [f"n{index:02d}" for index in range(20)]
        self.truth = ([_truth_episode("a", verdict="REJECT", absent=True),            # the demonstration fails it
                       _truth_episode("b"),                                           # the demonstration solves it by declining
                       _truth_episode("c", success=False),                            # nobody solves it
                       _truth_episode("d", rule_expert_solves=True),                  # the rule expert solves it
                       _truth_episode("e")]                                           # the demonstration declines, then fails
                      + [_truth_episode(case) for case in fresh])
        self.demonstration = ([_demonstration_episode("a", success=False),
                               _demonstration_episode("b", success=True, decline=True),
                               _demonstration_episode("c", success=False, rejected_true=True, repeat=True),
                               _demonstration_episode("d", success=True),
                               _demonstration_episode("e", success=False, decline=True)]
                              + [_demonstration_episode(case, success=False) for case in fresh])
        self.paths = {
            "truth_episodes": _write(self.folder / "truth_episodes.jsonl", self.truth),
            "truth_rows": _write(self.folder / "truth_rows.jsonl", _rows(self.truth, "truth")),
            "demonstration_episodes": _write(self.folder / "demonstration_episodes.jsonl", self.demonstration),
            "demonstration_rows": _write(self.folder / "demonstration_rows.jsonl", _rows(self.demonstration, "clone")),
        }
        self.cases = self.folder / "cases.json"
        self.cases.write_text(json.dumps({"train": ["a", "b", "d", "e"], "validation": ["c"]}), encoding="utf-8")
        self.fresh = fresh

    def run_selection(self, *extra: str) -> tuple[dict, list[dict], list[dict], dict]:
        # One output folder per call (an argument may be a path, so it cannot name the folder).
        self._runs = getattr(self, "_runs", 0) + 1
        out = self.folder / f"out{self._runs:02d}"
        code = sel.main(["--truth-rows", str(self.paths["truth_rows"]), "--truth-episodes", str(self.paths["truth_episodes"]),
                         "--demonstration-rows", str(self.paths["demonstration_rows"]),
                         "--demonstration-episodes", str(self.paths["demonstration_episodes"]),
                         "--demonstration-cases", str(self.cases), "--output-dir", str(out), *extra])
        self.assertEqual(code, 0)
        report = json.loads((out / "selection_report.json").read_text(encoding="utf-8"))
        chosen = json.loads((out / "selection_cases.json").read_text(encoding="utf-8"))
        train = list(sel.stream(out / "selection.train_view.jsonl"))
        validation = list(sel.stream(out / "selection.validation.jsonl"))
        return report, train, validation, chosen


class AssignmentTests(_Fixture):
    def test_one_source_per_case(self) -> None:
        _report, train, validation, chosen = self.run_selection()
        # Solved by the sequence and failed by the demonstration: the truth-built rows, in the split the case had.
        self.assertEqual(chosen["a"], {"source": sel.SOURCE_TRUTH, "split": "train", "class": "meter_on_rx",
                                       "demonstration_contrast": True})
        # Solved by the demonstration (through a decline): its own rows, with the decline.
        self.assertEqual(chosen["b"]["source"], sel.SOURCE_DEMONSTRATION)
        # Nobody solves it: the demonstration's rows, as the cloning pass had them, in its split.
        self.assertEqual(chosen["c"], {"source": sel.SOURCE_DEMONSTRATION, "split": "validation", "class": "rx2_adjacent"})
        self.assertEqual(chosen["d"]["source"], sel.SOURCE_DEMONSTRATION)
        # Every other case the sequence solves and the demonstration fails is taken, a seeded share held out whole.
        self.assertTrue(all(chosen[case]["source"] == sel.SOURCE_TRUTH for case in self.fresh))
        held = [case for case in self.fresh if chosen[case]["split"] == "validation"]
        self.assertEqual(len(held), 1)
        # Every row knows its case's source; a row of the other source is a rollback of a wrong candidate, nothing else.
        for row in train + validation:
            selection = row["cloning_selection"]
            self.assertEqual(selection["case_source"], chosen[row["scenario_id"]]["source"])
            self.assertEqual(selection["contrast_only"], selection["source"] != selection["case_source"])
            if selection["contrast_only"]:
                self.assertEqual((selection["source"], row["cloning"]["tool"]), (sel.SOURCE_DEMONSTRATION, sel.ROLLBACK))
        self.assertFalse(any("demonstration_contrast" in chosen[case] for case in self.fresh))
        self.assertEqual(chosen["e"]["source"], sel.SOURCE_TRUTH)
        self.assertEqual({row["scenario_id"] for row in validation}, {"c", *held})
        self.assertFalse({row["scenario_id"] for row in train} & {"c", *held})

    def test_the_other_rule_gives_the_sequence_every_case_the_rule_expert_fails(self) -> None:
        _report, _train, _validation, chosen = self.run_selection("--truth-roots", "rule-expert-fails")
        self.assertEqual(chosen["b"]["source"], sel.SOURCE_TRUTH)
        self.assertEqual(chosen["d"]["source"], sel.SOURCE_DEMONSTRATION)

    def test_the_split_is_seeded(self) -> None:
        first = self.run_selection()[3]
        again = self.run_selection("--seed", "20261106")[3]
        other = self.run_selection("--seed", "7")[3]
        self.assertEqual(first, again)
        self.assertEqual(sum(entry["split"] == "validation" for entry in other.values()), sum(entry["split"] == "validation" for entry in first.values()))

    def test_a_named_case_needs_its_episode(self) -> None:
        self.cases.write_text(json.dumps({"train": ["a", "zz"], "validation": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_selection()


class RowTests(_Fixture):
    def test_a_correction_the_prompt_does_not_name_is_left_out(self) -> None:
        report, train, _validation, _chosen = self.run_selection()
        mine = [row for row in train if row["scenario_id"] == "a" and row["cloning_selection"]["source"] == sel.SOURCE_TRUTH]
        self.assertEqual([row["cloning"]["step"] for row in mine], [0, 1, 3, 4, 5, 6])
        self.assertEqual(report["rows_left_out"][f"{sel.SOURCE_TRUTH} | train | {sel.LEFT_ABSENT}"], 1)
        kept = self.run_selection("--keep-absent-targets")[1]
        self.assertEqual([row["cloning"]["step"] for row in kept
                          if row["scenario_id"] == "a" and row["cloning_selection"]["source"] == sel.SOURCE_TRUTH], list(range(7)))

    def test_the_kept_rejected_commit_is_a_contrast_row(self) -> None:
        _report, train, _validation, _chosen = self.run_selection()
        commit = next(row for row in train if row["scenario_id"] == "a" and row["cloning"]["tool"] == sel.COMMIT)
        self.assertEqual(commit["cloning_selection"], {
            "contract": sel.SELECT_CONTRACT, "system": "ieee14", "source": sel.SOURCE_TRUTH, "case_source": sel.SOURCE_TRUTH,
            "contrast_only": False,
            "class": "meter_on_rx", "split": "train", "case_solved_by_source": True, "decline_row": False,
            "contrast": "rejected candidate on a faulted component, kept", "verdict": "REJECT",
            "open_rules_used": ["commit_rejected_candidate"]})

    def test_a_truth_built_case_keeps_the_demonstrations_rollbacks_of_wrong_candidates(self) -> None:
        _report, train, _validation, chosen = self.run_selection()

        def other(case):
            return [row for row in train if row["scenario_id"] == case and row["cloning_selection"]["source"] == sel.SOURCE_DEMONSTRATION]

        # Case a: the demonstration rolled back a rejected candidate on a healthy line, then failed the case.
        [rollback] = other("a")
        self.assertEqual(rollback["example_id"], "a:clone:04")
        self.assertEqual(rollback["cloning_selection"]["contrast"], "rejected candidate on a healthy component, rolled back")
        self.assertTrue(rollback["cloning_selection"]["contrast_only"])
        self.assertIs(rollback["cloning_selection"]["case_solved_by_source"], False)
        # Case e: it declined an accepted candidate on a healthy line.
        [decline] = other("e")
        self.assertTrue(decline["cloning_selection"]["decline_row"])
        self.assertEqual(decline["cloning_selection"]["contrast"], "accepted candidate on a healthy component, rolled back")
        # A truth-built case the cloning pass never used keeps nothing of the demonstration.
        self.assertTrue(all(not other(case) for case in self.fresh))
        # Switched off, no truth-built case does.
        _report, train, _validation, chosen = self.run_selection("--no-demonstration-contrast")
        self.assertNotIn("demonstration_contrast", chosen["a"])
        self.assertFalse([row for row in train if row["scenario_id"] in ("a", "e")
                          and row["cloning_selection"]["source"] == sel.SOURCE_DEMONSTRATION])

    def test_the_demonstration_keeps_its_declines_and_its_rollbacks_of_wrong_targets(self) -> None:
        _report, train, _validation, _chosen = self.run_selection()
        decline = next(row for row in train if row["scenario_id"] == "b" and row["cloning"]["tool"] == sel.ROLLBACK)
        self.assertTrue(decline["cloning_selection"]["decline_row"])
        self.assertEqual(decline["cloning_selection"]["contrast"], "accepted candidate on a healthy component, rolled back")
        rollback = next(row for row in train if row["scenario_id"] == "d" and row["cloning"]["tool"] == sel.ROLLBACK)
        self.assertFalse(rollback["cloning_selection"]["decline_row"])
        self.assertEqual(rollback["cloning_selection"]["contrast"], "rejected candidate on a healthy component, rolled back")
        self.assertEqual(rollback["cloning_selection"]["verdict"], "REJECT")

    def test_a_rollback_of_a_rejected_true_target_and_a_repeated_failure_are_left_out(self) -> None:
        report, _train, validation, _chosen = self.run_selection("--validation-rows", "1000")
        mine = {row["cloning"]["step"]: row for row in validation if row["scenario_id"] == "c"}
        # Steps 8 to 10 correct, verify and roll back a true target the check rejected; 11 and 12 fail alike.
        self.assertNotIn(10, mine)
        self.assertIn(4, mine)
        self.assertIn(11, mine)
        self.assertNotIn(12, mine)
        self.assertEqual(report["rows_left_out"][f"{sel.SOURCE_DEMONSTRATION} | validation | {sel.LEFT_TRUE_ROLLBACK}"], 1)
        self.assertEqual(report["rows_left_out"][f"{sel.SOURCE_DEMONSTRATION} | validation | {sel.LEFT_REPEAT}"], 1)
        kept = {row["cloning"]["step"]: row for row in self.run_selection("--validation-rows", "1000", "--keep-true-rejected-rollbacks")[2]
                if row["scenario_id"] == "c"}
        self.assertEqual(kept[10]["cloning_selection"]["contrast"], "rejected candidate on a faulted component, rolled back")

    def test_thinning_spares_lifecycle_and_correction_rows(self) -> None:
        report, train, _validation, _chosen = self.run_selection("--thin", "run_wls=0")
        self.assertFalse(any(row["cloning"]["tool"] == RUN_WLS for row in train))
        self.assertTrue(any(row["cloning"]["tool"] == sel.COMMIT for row in train))
        self.assertTrue(any(row["cloning"]["tool"] == CORRECT for row in train))
        self.assertGreater(report["rows_left_out"][f"{sel.SOURCE_TRUTH} | train | {sel.LEFT_THINNED}"], 0)
        with self.assertRaises(SystemExit):
            self.run_selection("--thin", "commit_state=0.5")
        # The same rows every time, and the demonstration by the cloning pass's own ids.
        half = [row["example_id"] for row in self.run_selection("--thin", "run_wls=0.5")[1]]
        self.assertEqual(half, [row["example_id"] for row in self.run_selection("--thin", "run_wls=0.5")[1]])
        self.assertEqual(sel.row_id(sel.SOURCE_DEMONSTRATION, "x", 3), "x:3")
        self.assertEqual(sel.row_id(sel.SOURCE_TRUTH, "x", 3), "x:truth:3")
        self.assertEqual(sel.keep_probability("x:3", 20261106), sel.keep_probability("x:3", 20261106))


class ViewTests(_Fixture):
    def test_the_validation_view_holds_every_contrast_row_of_the_held_out_cases(self) -> None:
        report, _train, validation, chosen = self.run_selection("--validation-rows", "8")
        self.assertEqual(len(validation), 8)
        self.assertEqual(report["rows_validation_view"], 8)
        contrast = [row for row in validation if row["cloning_selection"]["contrast"] is not None]
        # Case c: the rollback of the rejected wrong target (the rejected true one is left out).
        self.assertEqual([(row["scenario_id"], row["cloning"]["step"]) for row in contrast], [("c", 4)])
        self.assertTrue(all(chosen[row["scenario_id"]]["split"] == "validation" for row in validation))
        # The view is a sample: the held-out cases hold more rows than the budget.
        self.assertGreater(report["rows_held_out"], 8)
        # Lifecycle rows fill half the budget before the other rows fill the rest.
        lifecycle = [row for row in validation if row["cloning"]["tool"] in sel.LIFECYCLE_TOOLS]
        self.assertGreaterEqual(len(lifecycle), 3)

    def test_the_report_counts_what_was_written(self) -> None:
        report, train, _validation, chosen = self.run_selection()
        self.assertEqual(report["rows_train"], len(train))
        self.assertEqual(report["contract"], sel.SELECT_CONTRACT)
        self.assertEqual(report["truth_roots"], "demonstration-fails")
        counts = report["counts"]
        self.assertEqual(counts[f"rows | {sel.SOURCE_TRUTH} | train"] + counts[f"rows | {sel.SOURCE_DEMONSTRATION} | train"], len(train))
        # The decline of case b, and the one case e keeps beside its truth-built rows.
        self.assertEqual(counts["action | demonstration | train | decline"], 2)
        self.assertEqual(counts["contrast | train | accepted candidate on a healthy component, rolled back"], 2)
        self.assertEqual(counts["demonstration rollbacks kept in truth-built cases | train"], 2)
        self.assertEqual(counts["contrast | train | rejected candidate on a faulted component, kept"], 1)
        self.assertEqual(counts[f"lifted rule | {sel.SOURCE_TRUTH} | train | commit_rejected_candidate"], 1)
        cases = report["cases"]
        self.assertEqual(cases[f"{sel.SOURCE_DEMONSTRATION} | train"], {"cases": 2, "solved by their source": 2, "with a decline": 1,
                                                                        "keeping the demonstration's contrast rows": 0})
        self.assertEqual(cases[f"{sel.SOURCE_TRUTH} | train"]["keeping the demonstration's contrast rows"], 2)
        self.assertEqual(cases[f"{sel.SOURCE_TRUTH} | train"]["cases"], sum(1 for entry in chosen.values()
                                                                             if entry["source"] == sel.SOURCE_TRUTH and entry["split"] == "train"))
        self.assertEqual(sum(sum(entry.values()) for entry in report["cases_by_class"].values()), len(chosen))


class ReservedTests(_Fixture):
    def test_a_reserved_case_is_taken_only_where_the_cloning_pass_had_it(self) -> None:
        reserved = self.folder / "reserved.json"
        # Case a is a case of the cloning pass, n00 is not.
        reserved.write_text(json.dumps({"roots": ["a", "n00"], "purpose": "kept apart"}), encoding="utf-8")
        report, train, validation, chosen = self.run_selection("--reserved-roots", str(reserved))
        self.assertIn("a", chosen)
        self.assertNotIn("n00", chosen)
        self.assertFalse([row for row in train + validation if row["scenario_id"] == "n00"])
        self.assertEqual(report["reserved_roots"], {"file": str(reserved), "roots": 2, "in_the_selection": 1,
                                                    "in_the_selection_and_the_cloning_pass": 1, "left_out": ["n00"]})
        # Without the file nothing is left out, and every row names its system.
        report, train, _validation, chosen = self.run_selection()
        self.assertIn("n00", chosen)
        self.assertEqual(report["reserved_roots"]["roots"], 0)
        self.assertEqual({row["cloning_selection"]["system"] for row in train}, {"ieee14"})
        self.assertEqual(sel.read_reserved(None), frozenset())
        listed = self.folder / "listed.json"
        listed.write_text(json.dumps(["x", "y"]), encoding="utf-8")
        self.assertEqual(sel.read_reserved(listed), frozenset({"x", "y"}))


class HelperTests(unittest.TestCase):
    def test_the_lifecycle_step_of_a_correction(self) -> None:
        episode = _demonstration_episode("x", success=False, rejected_true=True)
        disposed = sel.lifecycle_of_corrections(episode)
        self.assertEqual(sorted(disposed), [4, 7, 10])
        self.assertEqual((disposed[4]["target_is_true"], disposed[10]["target_is_true"]), (False, True))

    def test_repeated_failures_are_per_active_state(self) -> None:
        trace = [_step(0, CORRECT, status="failure", arguments={"line_index": 5}),
                 _step(1, CORRECT, status="failure", arguments={"line_index": 5}),
                 _step(2, CORRECT, status="failure", arguments={"line_index": 6}),
                 _step(3, sel.COMMIT),
                 _step(4, CORRECT, status="failure", arguments={"line_index": 5})]
        self.assertEqual(sel.repeated_failures(trace), {1})


if __name__ == "__main__":
    unittest.main()
