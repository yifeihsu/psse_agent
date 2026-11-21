"""The minimal environment's data tools on made-up records: the row selection, the trainer's two replaced checks, the summaries."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from psse_env.minimal.actions import tool_schemas
from psse_env.minimal.rows import MINIMAL_PROTOCOL
from psse_env.sft.gates import GateError
from research.minimal_env import build_demonstrations, expert_baseline, select_rows, train_sft


def _episode(scenario: str, name: str, *, tag: str = "success", solved: bool = True, strict: bool | None = None) -> dict:
    return {"scenario_id": scenario, "compound_class": name, "tag": tag,
            "demonstration": "recovery" if tag.startswith("recovery") else "success", "solved": solved,
            "score": {"strict": {"solved": solved if strict is None else strict}}}


def _row(scenario: str, tag: str, step: int, kind: str, tool: str, *, solved: bool = True, passes: bool = True) -> dict:
    return {
        "example_id": f"{scenario}:{tag}:{step:02d}", "scenario_id": scenario, "physical_root_fingerprint": f"root_{scenario}",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u" * 10},
                     {"role": "assistant", "content": "", "tool_calls": [
                         {"id": "call_0", "type": "function", "function": {"name": "finish", "arguments": {"outcome": "resolved"}}}]}],
        "tools": tool_schemas(breakers=True), "metadata": {"protocol": MINIMAL_PROTOCOL},
        "minimal": {"demonstration": "recovery" if tag.startswith("recovery") else "success", "kind": kind, "tool": tool,
                    "sequence_solved": solved, "estimator_passes_at_end": passes},
    }


class CaseSelectionTests(unittest.TestCase):
    def test_a_case_is_taken_when_both_audits_call_its_success_path_repaired(self) -> None:
        episodes = [
            _episode("a", "x"), _episode("a", "x", tag="recovery0"),
            _episode("b", "x", solved=False), _episode("b", "x", tag="recovery0"),
            _episode("c", "x", strict=False),
            _episode("d", "y"), _episode("d", "y", tag="recovery0", solved=False),
        ]
        cases = select_rows.choose_cases(episodes, keep_unsolved=False)
        self.assertEqual({name: entry["tags"] for name, entry in cases.items()}, {"a": ["success", "recovery0"], "d": ["success"]})
        kept = select_rows.choose_cases(episodes, excluded={"a"})
        self.assertEqual(sorted(kept), ["b", "c", "d"])
        self.assertFalse(kept["b"]["solved"])
        self.assertEqual(kept["c"]["tags"], ["success"])

    def test_the_split_holds_out_whole_cases_of_every_class(self) -> None:
        cases = {f"{name}{index}": {"class": name} for name in ("x", "y") for index in range(50)}
        cases["z0"] = {"class": "z"}
        held = select_rows.split_cases(cases, share=0.1, seed=3)
        self.assertEqual(sum(1 for case in held if case.startswith("x")), 5)
        self.assertEqual(sum(1 for case in held if case.startswith("y")), 5)
        self.assertNotIn("z0", held)  # a class of one case is never emptied
        self.assertEqual(held, select_rows.split_cases(cases, share=0.1, seed=3))
        self.assertNotEqual(held, select_rows.split_cases(cases, share=0.1, seed=4))

    def test_the_views_share_no_case_and_keep_the_rare_kinds_in_validation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder) / "demo", Path(folder) / "selection"
            source.mkdir()
            episodes, rows = [], []
            for index in range(40):
                scenario = f"case{index:02d}"
                episodes += [_episode(scenario, "x"), _episode(scenario, "x", tag="recovery0")]
                rows += [_row(scenario, "success", 1, "correction", "correct_parameters"), _row(scenario, "success", 2, "finish", "finish"),
                         _row(scenario, "recovery0", 2, "undo", "undo"), _row(scenario, "recovery0", 3, "finish", "finish")]
            # Two cases beyond the tools: one whose final estimate still alarms, one whose final estimate passes.
            episodes += [_episode("alarm", "x", solved=False), _episode("quiet", "x", solved=False)]
            rows += [_row("alarm", "success", 1, "correction", "correct_parameters", solved=False, passes=False),
                     _row("alarm", "success", 2, "finish", "finish", solved=False, passes=False),
                     _row("quiet", "success", 1, "correction", "correct_parameters", solved=False, passes=True),
                     _row("quiet", "success", 2, "finish", "finish", solved=False, passes=True)]
            (source / "episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in episodes), encoding="utf-8")
            (source / "rows.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            self.assertEqual(select_rows.main(["--demonstrations", str(source), "--output-dir", str(out), "--validation-share", "0.25",
                                               "--validation-rows", "8"]), 0)
            train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            validation = [json.loads(line) for line in (out / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
            report = json.loads((out / "selection_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["cases"]["taken"], 42)
        self.assertEqual((report["cases"]["repaired"], report["cases"]["beyond_the_tools"], report["cases"]["with_a_recovery_path"]),
                         (40, 2, 40))
        self.assertEqual(report["cases"]["train"] + report["cases"]["validation"], 42)
        self.assertEqual(len(validation), 8)
        self.assertFalse({row["scenario_id"] for row in train} & {row["scenario_id"] for row in validation})
        # The case whose final estimate passes loses its last word; the one that still alarms keeps it.
        self.assertEqual(report["rows_dropped"], {"unresolved_finish_on_a_passing_estimate": 1})
        self.assertEqual(report["train_rows"] + report["validation_pool_rows"], 40 * 4 + 3)
        self.assertEqual(report["validation_rows_by_kind"], {"recovery:finish": 2, "recovery:undo": 2, "success:correction": 2, "success:finish": 2})
        self.assertEqual({row["selection"]["split"] for row in train}, {"train"})
        train_sft.validate_minimal_splits(train, validation)

    def test_paths_of_a_kind_can_be_left_out_and_one_breakers_rows_repeated(self) -> None:
        def breaker_row(scenario: str, tag: str, step: int, breaker: str) -> dict:
            row = _row(scenario, tag, step, "correction", "correct_topology")
            row["messages"][-1]["tool_calls"][0]["function"] = {"name": "correct_topology",
                                                                "arguments": {"breaker": breaker, "status": "open"}}
            return row

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "demo"
            source.mkdir()
            episodes, rows = [], []
            for index in range(20):
                scenario = f"case{index:02d}"
                refusal = {**_episode(scenario, "x", tag="refusal0"), "demonstration": "refusal"}
                episodes += [_episode(scenario, "x"), refusal]
                rows += [breaker_row(scenario, "success", 2, "CB_2R4_2R5" if index % 2 else "CB_6_B1_B2"),
                         _row(scenario, "success", 3, "finish", "finish"), _row(scenario, "refusal0", 4, "inspect", "inspect")]
                rows[-1]["minimal"]["demonstration"] = "refusal"
            (source / "episodes.jsonl").write_text("".join(json.dumps(item) + "\n" for item in episodes), encoding="utf-8")
            (source / "rows.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            reports = {}
            for name, extra in (("plain", []), ("chosen", ["--skip-demonstrations", "refusal", "--repeat-breaker", "CB_2R4_2R5=3"])):
                out = Path(folder) / name
                self.assertEqual(select_rows.main(["--demonstrations", str(source), "--output-dir", str(out), "--validation-share",
                                                   "0.2", "--validation-rows", "4", *extra]), 0)
                train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
                validation = [json.loads(line) for line in (out / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
                reports[name] = (json.loads((out / "selection_report.json").read_text(encoding="utf-8")), train, validation)
        plain, plain_train, _ = reports["plain"]
        chosen, chosen_train, chosen_validation = reports["chosen"]
        self.assertEqual(plain["cases"]["paths_by_kind"], {"refusal": 20, "success": 20})
        self.assertEqual(chosen["cases"]["paths_by_kind"], {"success": 20})
        self.assertNotIn("refusal", {row["minimal"]["demonstration"] for row in chosen_train + chosen_validation})
        named = [row for row in plain_train if select_rows.target_breaker(row) == "CB_2R4_2R5"]
        repeated = [row for row in chosen_train if select_rows.target_breaker(row) == "CB_2R4_2R5"]
        self.assertEqual(len(repeated), 3 * len(named))
        self.assertEqual(chosen["repeated_rows"], {"CB_2R4_2R5": 2 * len(named)})
        self.assertEqual(len({row["example_id"] for row in repeated}), len(repeated))  # every copy has its own id
        self.assertEqual(chosen["settings"]["repeat_breaker"], {"CB_2R4_2R5": 3})
        self.assertFalse(any(":repeat" in row["example_id"] for row in chosen_validation))
        with self.assertRaises(ValueError):
            select_rows.parse_repeats(["CB_2R4_2R5"])


class TrainerChecksTests(unittest.TestCase):
    def test_minimal_rows_keep_their_own_tools(self) -> None:
        rows = [_row("a", "success", 1, "finish", "finish")]
        kept, report = train_sft.keep_row_tools(rows, source_label="train")
        self.assertEqual([item["function"]["name"] for item in kept[0]["tools"]],
                         ["inspect", "correct_topology", "correct_parameters", "correct_measurements", "undo", "finish"])
        self.assertEqual((report["source_registry_replaced"], report["rows_changed"], report["rows_by_tool_count"]), (False, 0, {"6": 1}))
        self.assertIsNot(kept[0], rows[0])

    def test_a_row_of_another_protocol_or_a_shared_case_is_refused(self) -> None:
        train, validation = [_row("a", "success", 1, "finish", "finish")], [_row("b", "success", 1, "finish", "finish")]
        self.assertEqual(train_sft.validate_minimal_splits(train, validation)["overlap"], [])
        old = copy.deepcopy(train)
        old[0]["metadata"]["protocol"] = "canonical"
        with self.assertRaises(GateError):
            train_sft.validate_minimal_splits(old, validation)
        with self.assertRaises(GateError):
            train_sft.validate_minimal_splits(train, train)
        with self.assertRaises(GateError):
            train_sft.validate_minimal_splits(train, [])
        broken = copy.deepcopy(train)
        broken[0]["tools"][0]["type"] = "tool"
        with self.assertRaises(GateError):
            train_sft.keep_row_tools(broken, source_label="train")


class SummaryTests(unittest.TestCase):
    def _score(self, *, solved: bool, strict: bool, passes: bool, declared: str = "resolved", damaged: int = 0, actions: int = 4) -> dict:
        return {"solved": solved, "success": solved and declared == "resolved", "declared": declared, "estimator_passes": passes,
                "strict": {"solved": strict, "reasons": [] if strict else ["final_case_outside_clean_tolerance"]},
                "physical": {"damaged": damaged}, "actions": actions,
                "cost": {"wls_solves": 3, "substation_solves": 9, "parameter_fits": 1, "measurement_repairs": 0, "tool_seconds": 2.5}}

    def test_the_demonstration_summary_separates_repaired_from_passing(self) -> None:
        episodes = [
            {"demonstration": "success", "solved": True, "compound_class": "x", "steps": [None] * 4, "orders_tried": 1,
             "score": self._score(solved=True, strict=True, passes=True)},
            {"demonstration": "success", "solved": False, "compound_class": "x", "steps": [None] * 6, "orders_tried": 2,
             "score": self._score(solved=False, strict=False, passes=True)},
            {"demonstration": "recovery", "solved": True, "compound_class": "x", "steps": [None] * 7,
             "score": self._score(solved=True, strict=True, passes=True)},
        ]
        summary = build_demonstrations.summarize(episodes)
        totals = summary["totals"]
        self.assertEqual((totals["success_sequences"], totals["success_solved"], totals["recovery_solved"]), (2, 1, 1))
        self.assertEqual((totals["estimator_passes_at_end"], totals["unsolved_with_estimator_passing"]), (2, 1))
        self.assertEqual(totals["solved_and_strict_agree"], 2)
        self.assertEqual(totals["substation_solves"], 18)
        self.assertEqual(summary["mean_actions_success_path"], 5.0)

    def test_the_expert_replay_summary_lists_every_disagreement(self) -> None:
        def record(scenario: str, recorded: bool, strict: bool) -> dict:
            return {"scenario_id": scenario, "compound_class": "x",
                    "recorded": {"success": recorded, "actions": 12, "wls_calls": 4, "audit_reasons": []},
                    "episode": {"outcome": "resolved", "score": self._score(solved=strict, strict=strict, passes=True, actions=6)}}

        summary = expert_baseline.summarize([record("a", True, True), record("b", True, False),
                                             {"scenario_id": "c", "compound_class": "x", "untranslatable": "explicit values",
                                              "recorded": {"success": False, "actions": 9, "wls_calls": 3, "audit_reasons": []}}])
        self.assertEqual(summary["totals"]["untranslatable"], 1)
        self.assertEqual((summary["outcome_agreement"]["replayed"], summary["outcome_agreement"]["agree"]), (2, 1))
        self.assertEqual(summary["outcome_agreement"]["disagree"][0]["scenario_id"], "b")
        self.assertEqual(summary["mean_actions"], {"old_interface": 12.0, "minimal_interface": 6.0})
        self.assertEqual(summary["totals"]["replay_false_resolved"], 1)


if __name__ == "__main__":
    unittest.main()
