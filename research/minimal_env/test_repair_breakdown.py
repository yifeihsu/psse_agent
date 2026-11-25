"""The split of an evaluation's episodes by how far they got, on hand-made episode records."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from research.minimal_env import evaluate, repair_breakdown
from research.minimal_env.repair_breakdown import (
    DAMAGE_ONLY,
    HONEST_PARTIAL,
    NOTHING,
    OUT_OF_ACTIONS,
    PARTIAL,
    RESOLVED,
    SOLVED,
    UNRESOLVED,
)


def _error(kind: str, repaired: bool) -> dict:
    return {"kind": kind, "repaired": repaired, **({"breaker": "CB_1"} if kind == "breaker" else {})}


def _episode(case: str, errors: list, *, declared: str | None, damage: list | None = None, name: str = "x",
             passes: bool = False, outcome: str | None = None) -> dict:
    damage = damage or []
    repaired = sum(1 for item in errors if item["repaired"])
    solved = repaired == len(errors) and not damage
    return {"scenario_id": case, "compound_class": name, "outcome": outcome or declared, "steps": [], "reward": {"total": 0.0},
            "score": {"solved": solved, "success": solved and declared == "resolved", "declared": declared,
                      "estimator_passes": passes, "strict": {"solved": solved}, "actions": 3, "tool_errors": 0,
                      "physical": {"errors": errors, "true_errors": len(errors), "repaired": repaired, "damage": damage,
                                   "damaged": len(damage), "solved": solved}}}


def _records() -> list[dict]:
    return [
        _episode("a", [_error("line", True), _error("breaker", True)], declared="resolved", passes=True),
        # Every error repaired, but a healthy line was re-fitted: unsolved, and a false claim.
        _episode("b", [_error("line", True)], declared="resolved", damage=[{"kind": "line", "line": 4}], passes=True),
        # Two of three repaired, the model says so: the honest partial repair.
        _episode("c", [_error("line", True), _error("meter", True), _error("breaker", False)], declared="unresolved", name="y"),
        # One of two repaired, claimed resolved with the estimator passing.
        _episode("d", [_error("line", True), _error("line", False)], declared="resolved", passes=True, name="y"),
        # Partial with damage, out of actions.
        _episode("e", [_error("meter", True), _error("line", False)], declared=None, outcome="budget_exhausted",
                 damage=[{"kind": "meter", "meter": 7}, {"kind": "meter", "meter": 8}], name="y"),
        _episode("f", [_error("breaker", False)], declared="unresolved", name="y"),
    ]


REFERENCE = {"a": {"expert": True, "truth": True}, "b": {"expert": True, "truth": True}, "c": {"expert": False, "truth": True},
             "d": {"expert": False, "truth": True}, "e": {"expert": False, "truth": False}, "f": {"expert": False, "truth": False}}


class BreakdownTests(unittest.TestCase):
    def test_every_episode_falls_in_one_group(self) -> None:
        found = repair_breakdown.breakdown(_records())
        self.assertEqual(found["cases"], 6)
        self.assertEqual(found["episodes"], {SOLVED: 1, DAMAGE_ONLY: 1, PARTIAL: 3, NOTHING: 1})
        self.assertEqual(sum(found["episodes"].values()), found["cases"])

    def test_errors_are_counted_by_kind(self) -> None:
        errors = repair_breakdown.breakdown(_records())["errors"]
        self.assertEqual((errors["injected"], errors["repaired"]), (11, 7))
        self.assertEqual(errors["by_kind"], {"breaker": {"injected": 3, "repaired": 1}, "line": {"injected": 6, "repaired": 4},
                                             "meter": {"injected": 2, "repaired": 2}})
        self.assertEqual(list(errors["by_kind"]), ["breaker", "line", "meter"])

    def test_the_share_is_given_over_the_errors_and_as_a_mean_over_the_cases(self) -> None:
        found = repair_breakdown.breakdown(_records())
        self.assertAlmostEqual(found["share_of_errors_repaired"], round(7 / 11, 4))
        self.assertAlmostEqual(found["mean_share_repaired_per_case"], round((1 + 1 + 2 / 3 + 0.5 + 0.5 + 0) / 6, 4))

    def test_damage_is_counted_in_objects_and_in_episodes(self) -> None:
        damage = repair_breakdown.breakdown(_records())["damage"]
        self.assertEqual(damage, {"episodes": 2, "objects": 3, "by_kind": {"line": 1, "meter": 2}})

    def test_the_unsolved_episodes_are_split_by_how_they_ended(self) -> None:
        unsolved = repair_breakdown.breakdown(_records())["unsolved"]
        self.assertEqual(unsolved["episodes"], 5)
        self.assertEqual(unsolved["by_ending"][RESOLVED], {
            "episodes": 2, DAMAGE_ONLY: 1, PARTIAL: 1, NOTHING: 0, "with a damaged healthy object": 1, "estimator passes at the end": 2})
        self.assertEqual(unsolved["by_ending"][UNRESOLVED]["episodes"], 2)
        self.assertEqual(unsolved["by_ending"][UNRESOLVED][NOTHING], 1)
        self.assertEqual(unsolved["by_ending"][OUT_OF_ACTIONS], {
            "episodes": 1, DAMAGE_ONLY: 0, PARTIAL: 1, NOTHING: 0, "with a damaged healthy object": 1, "estimator passes at the end": 0})
        self.assertEqual(unsolved["errors_left"], {"0": 1, "1": 4})
        self.assertEqual(unsolved["errors_left_by_kind"], {"breaker": 2, "line": 2})

    def test_only_a_partial_repair_that_says_so_and_damages_nothing_is_honest(self) -> None:
        # c is; d claims resolved; e damaged two meters; f repaired nothing.
        self.assertEqual(repair_breakdown.breakdown(_records())["unsolved"][HONEST_PARTIAL], 1)

    def test_classes_and_reference_groups_get_the_same_split(self) -> None:
        found = repair_breakdown.breakdown(_records(), REFERENCE)
        self.assertEqual(found["by_class"]["x"]["episodes"], {SOLVED: 1, DAMAGE_ONLY: 1, PARTIAL: 0, NOTHING: 0})
        self.assertEqual(found["by_class"]["y"]["errors"]["repaired"], 4)
        groups = found["by_reference"]
        self.assertEqual([groups[name]["cases"] for name in repair_breakdown.REFERENCE_GROUPS], [2, 2, 2])
        self.assertEqual(groups["only the truth-built sequence repairs"]["episodes"], {SOLVED: 0, DAMAGE_ONLY: 0, PARTIAL: 2, NOTHING: 0})
        self.assertEqual(groups["neither repairs"]["unsolved"]["episodes"], 2)
        self.assertNotIn("by_reference", repair_breakdown.breakdown(_records()))

    def test_a_record_without_the_error_list_is_counted_from_its_totals(self) -> None:
        record = _episode("g", [_error("line", True), _error("line", False)], declared="unresolved")
        del record["score"]["physical"]["errors"], record["score"]["physical"]["damage"]
        found = repair_breakdown.breakdown([record])
        self.assertEqual(found["episodes"][PARTIAL], 1)
        self.assertEqual(found["errors"], {"injected": 2, "repaired": 1, "by_kind": {}})

    def test_the_evaluation_summary_carries_the_split_and_each_case_its_counts(self) -> None:
        records = _records()
        summary = evaluate.summarize(records, REFERENCE)
        self.assertEqual(summary["repairs"], repair_breakdown.breakdown(records, REFERENCE))
        self.assertEqual(summary["totals"]["solved"], summary["repairs"]["episodes"][SOLVED])
        self.assertEqual(summary["totals"]["true_errors_repaired"], summary["repairs"]["errors"]["repaired"])
        payload = evaluate.evaluation_payload(records, suite="s", adapter="a", settings={}, reference=REFERENCE)
        line = next(item for item in payload["episodes"] if item["scenario_id"] == "e")
        self.assertEqual((line["errors"], line["repaired"], line["damaged"]), (2, 1, 2))
        self.assertNotIn("group", line)
        json.dumps(payload, allow_nan=False)

    def test_the_reference_groups_are_the_evaluations_own(self) -> None:
        self.assertIs(evaluate.REFERENCE_GROUPS, repair_breakdown.REFERENCE_GROUPS)


class PairedTests(unittest.TestCase):
    def test_solved_and_partial_progress_are_compared_case_by_case(self) -> None:
        first = _records()
        other = [
            _episode("a", [_error("line", True), _error("breaker", False)], declared="resolved"),           # lost
            _episode("b", [_error("line", True)], declared="resolved", passes=True),                        # gained, damage gone
            _episode("c", [_error("line", True), _error("meter", True), _error("breaker", True)], declared="resolved"),  # gained
            _episode("d", [_error("line", False), _error("line", False)], declared="unresolved"),          # one error fewer
            _episode("e", [_error("meter", True), _error("line", False)], declared=None, outcome="budget_exhausted"),
            _episode("z", [_error("line", True)], declared="resolved"),                                    # not in the first
        ]
        found = repair_breakdown.paired(first, other)
        self.assertEqual((found["cases"], found["solved gained"], found["solved lost"]), (5, 2, 1))
        self.assertEqual((found["more errors repaired"], found["fewer errors repaired"]), (1, 2))
        self.assertEqual(found["errors repaired, change"], -1)
        self.assertEqual(found["healthy objects damaged, change"], -3)
        self.assertEqual(repair_breakdown.paired(first, other, ["a", "b"])["cases"], 2)

    def test_the_exact_two_sided_test_on_the_discordant_pairs(self) -> None:
        self.assertEqual(repair_breakdown.mcnemar(0, 0), 1.0)
        self.assertEqual(repair_breakdown.mcnemar(5, 5), 1.0)
        self.assertAlmostEqual(repair_breakdown.mcnemar(20, 13), 0.2962, places=3)
        self.assertAlmostEqual(repair_breakdown.mcnemar(15, 26), 0.1173, places=3)
        self.assertAlmostEqual(repair_breakdown.mcnemar(0, 6), 2 / 64)


class ReadingTests(unittest.TestCase):
    def test_the_three_stored_forms_are_read(self) -> None:
        records = _records()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "evaluation" / "episodes"
            folder.mkdir(parents=True)
            for record in records:
                (folder / f"{record['scenario_id']}.json").write_text(json.dumps(record), encoding="utf-8")
            expert = root / "expert"
            expert.mkdir()
            (expert / "episodes.jsonl").write_text("".join(
                json.dumps({"compound_class": record["compound_class"], "episode": {key: value for key, value in record.items()
                                                                                   if key != "compound_class"}}) + "\n"
                for record in records), encoding="utf-8")
            truth = root / "truth.jsonl"
            lines = []
            for record in records:
                lines.append({**record, "demonstration": "success"})
                # A recovery path is another record of the same case and is not its result.
                lines.append({**record, "demonstration": "recovery"})
            truth.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
            wanted = repair_breakdown.breakdown(records)
            for path in (root / "evaluation", expert, truth):
                self.assertEqual(repair_breakdown.breakdown(repair_breakdown.read_records(path)), wanted, str(path))

    def test_a_case_stored_twice_is_refused(self) -> None:
        record = _records()[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episodes.jsonl"
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "appears twice"):
                repair_breakdown.read_records(path)

    def test_the_command_prints_the_split_and_the_comparison(self) -> None:
        records = _records()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("one", "two"):
                (root / name).mkdir()
                (root / name / "episodes.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            reference = root / "reference.json"
            reference.write_text(json.dumps(REFERENCE), encoding="utf-8")
            output = root / "out" / "breakdown.json"
            printed = io.StringIO()
            with redirect_stdout(printed):
                code = repair_breakdown.main([str(root / "one"), str(root / "two"), "--names", "First,Second",
                                              "--reference", str(reference), "--output", str(output)])
            self.assertEqual(code, 0)
            text = printed.getvalue()
            self.assertIn("| | First | Second |", text)
            self.assertIn("| Some errors repaired | 3 | 3 |", text)
            self.assertIn("| Errors repaired | 7 of 11 | 7 of 11 |", text)
            self.assertIn("Second, all cases (6): solved gained 0, lost 0 (p = 1.00)", text)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload["evaluations"]), ["First", "Second"])
            self.assertEqual(payload["against_the_first"]["Second"]["all cases"]["cases"], 6)
            self.assertNotIn(b"\r", output.read_bytes())


if __name__ == "__main__":
    unittest.main()
