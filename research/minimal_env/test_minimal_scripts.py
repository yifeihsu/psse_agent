"""The scripts a cluster cell runs for the minimal environment, without a model or a tokenizer.

The model's wire form of the six actions, the smoke cuts of the row gate, the row gate's cache in the trainer
entry, and the evaluation's summary.  What needs the tokenizer (the gate itself) or the model (the policy's
generation) is first exercised by a cell's smoke.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from psse_env.dagger.release_factories import _validated_generated_action
from psse_env.minimal import observation as view
from psse_env.minimal.actions import validate
from psse_env.sft import research_cli
from psse_env.sft.gates import GateError, ParsedToolCall, parse_tool_call
from research.minimal_env import evaluate, row_gate, train_sft, truth_sequences

#: The six actions as Gemma 4's chat template writes a tool call: bare keys, ``<|"|>`` around strings.
WIRE_CALLS = (
    ('<|tool_call>call:inspect{view:<|"|>breakers<|"|>}<tool_call|>', "inspect", {"view": "breakers"}),
    ('<|tool_call>call:inspect{flip:[<|"|>CB_9_B1_N3<|"|>,<|"|>CB_Y1014_14N1_14N2<|"|>],view:<|"|>breakers<|"|>}<tool_call|>',
     "inspect", {"flip": ["CB_9_B1_N3", "CB_Y1014_14N1_14N2"], "view": "breakers"}),
    ('<|tool_call>call:inspect{version:0,view:<|"|>estimate<|"|>}<tool_call|>', "inspect", {"version": 0, "view": "estimate"}),
    ('<|tool_call>call:correct_topology{breaker:<|"|>CB_9_B1_N3<|"|>,status:<|"|>open<|"|>}<tool_call|>',
     "correct_topology", {"breaker": "CB_9_B1_N3", "status": "open"}),
    ('<|tool_call>call:correct_parameters{line:6}<tool_call|>', "correct_parameters", {"line": 6}),
    ('<|tool_call>call:correct_measurements{meters:[19,66,104]}<tool_call|>', "correct_measurements", {"meters": [19, 66, 104]}),
    ('<|tool_call>call:undo{}<tool_call|>', "undo", {}),
    ('<|tool_call>call:undo{version:2}<tool_call|>', "undo", {"version": 2}),
    ('<|tool_call>call:finish{outcome:<|"|>resolved<|"|>}<tool_call|>', "finish", {"outcome": "resolved"}),
)


class WireFormatTests(unittest.TestCase):
    """The parser the row gate and the policy share reads every action of the minimal interface.

    This covers the reading of the wire form, not the template that writes it:
    the template runs only with the model's processor.
    """

    def setUp(self) -> None:
        tools = view.tools({"breaker_model": True})
        self.parameters = {item["function"]["name"]: item["function"]["parameters"] for item in tools}

    def test_every_action_reads_back_from_the_wire_form(self) -> None:
        for text, name, arguments in WIRE_CALLS:
            self.assertEqual(parse_tool_call(text), ParsedToolCall(name=name, arguments=arguments), text)

    def test_a_generated_call_passes_the_policys_check_and_the_environments(self) -> None:
        for text, name, arguments in WIRE_CALLS:
            action = _validated_generated_action(text, self.parameters)
            self.assertEqual(action, {"tool": name, "arguments": arguments}, text)
            self.assertEqual(validate(action, breakers=True)["tool"], name)

    def test_a_call_outside_the_six_tools_or_their_arguments_is_refused(self) -> None:
        for text in ('<|tool_call>call:commit_state{}<tool_call|>',
                     '<|tool_call>call:correct_parameters{line:6,value:0.1}<tool_call|>',
                     '<|tool_call>call:finish{outcome:<|"|>done<|"|>}<tool_call|>',
                     '<|tool_call>call:correct_measurements{meters:<|"|>19<|"|>}<tool_call|>'):
            with self.assertRaises(GateError, msg=text):
                _validated_generated_action(text, self.parameters)


def _row(index: int, demonstration: str, kind: str, tool: str, length: int) -> dict:
    return {"example_id": f"case{index}:{demonstration}:{index:02d}", "messages": [{"role": "user", "content": "x" * length}],
            "minimal": {"demonstration": demonstration, "kind": kind, "tool": tool}}


class SmokeCutTests(unittest.TestCase):
    def test_the_longest_row_comes_first_and_the_groups_take_turns(self) -> None:
        rows = [_row(index, "success", "correction", "correct_parameters", 10) for index in range(6)]
        rows += [_row(10 + index, "recovery", "undo", "undo", 20) for index in range(2)]
        rows.append(_row(99, "success", "finish", "finish", 500))
        lines = [json.dumps(row) for row in rows]
        chosen = row_gate.choose_rows(lines, 5, seed=3)
        self.assertEqual(chosen[0], 8)  # the 500-character row
        groups = [row_gate._row_group(rows[index]) for index in chosen[1:]]
        self.assertIn(("recovery", "undo", "undo"), groups)  # a rare group is not crowded out
        self.assertEqual(len(set(chosen)), 5)
        self.assertEqual(chosen, row_gate.choose_rows(lines, 5, seed=3))
        self.assertEqual(len(row_gate.choose_rows(lines, 50, seed=3)), len(rows))

    def test_the_smoke_files_are_written_whole(self) -> None:
        rows = [_row(index, "success", "correction", "correct_parameters", 10 + index) for index in range(8)]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "view.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = row_gate.disjoint_smoke_rows(source, source, Path(folder) / "out", train_rows=3, validation_rows=2, seed=1)
            train = (Path(folder) / "out" / "train.jsonl").read_bytes()
        self.assertEqual((result["train"]["rows"], result["train"]["of"], result["validation"]["rows"]), (3, 8, 2))
        self.assertEqual(train.count(b"\n"), 3)
        self.assertNotIn(b"\r", train)

    def test_the_smoke_suite_takes_the_classes_in_turn(self) -> None:
        roots = [{"audit": {"compound": {"class": name}}, "id": f"{name}{index}"} for name in ("a", "b", "c") for index in range(4)]
        chosen = row_gate.choose_roots(roots, 5)
        self.assertEqual([root["id"] for root in chosen], ["a0", "b0", "c0", "a1", "b1"])
        self.assertEqual(len(row_gate.choose_roots(roots, 99)), 12)


class GateCacheTests(unittest.TestCase):
    """The row gate's result is reused only for the same rows, settings and code."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        base = Path(self.folder.name)
        self.train, self.validation, self.cache = base / "train.jsonl", base / "validation.jsonl", base / "cache.pkl"
        self.train.write_text('{"row": 1}\n', encoding="utf-8")
        self.validation.write_text('{"row": 2}\n', encoding="utf-8")
        self.settings = SimpleNamespace(model_name="m", revision="r", max_length=32768, allow_prompt_truncation=False,
                                        local_files_only=True, trust_remote_code=False, validate=lambda: None)
        self.calls = 0

        def stock(*, train_file, validation_file, settings):
            self.calls += 1
            return "processor", ["train example"], ["validation example"], {"train": {"length_audit": {}}}

        patches = [
            mock.patch.object(train_sft, "_STOCK_PREPARE", stock),
            mock.patch.object(research_cli, "load_exact_processor", lambda *a, **k: (SimpleNamespace(apply_chat_template=lambda: None), "AutoProcessor")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _prepare(self):
        return train_sft.prepare_with_gate_cache(train_file=self.train, validation_file=self.validation, settings=self.settings)

    def test_without_the_variable_the_stock_gate_runs_every_time(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(train_sft.GATE_CACHE_ENV, None)
            self._prepare()
            self._prepare()
        self.assertEqual(self.calls, 2)
        self.assertFalse(self.cache.exists())

    def test_the_second_start_loads_what_the_first_gated(self) -> None:
        with mock.patch.dict(os.environ, {train_sft.GATE_CACHE_ENV: str(self.cache)}):
            _, train, validation, first = self._prepare()
            _, again_train, again_validation, second = self._prepare()
        self.assertEqual(self.calls, 1)
        self.assertEqual((train, validation), (again_train, again_validation))
        self.assertEqual((first["row_gate_cache"]["hit"], second["row_gate_cache"]["hit"]), (False, True))
        self.assertEqual(first["row_gate_cache"]["key"], second["row_gate_cache"]["key"])

    def test_other_rows_or_another_limit_are_gated_again(self) -> None:
        with mock.patch.dict(os.environ, {train_sft.GATE_CACHE_ENV: str(self.cache)}):
            self._prepare()
            self.train.write_text('{"row": 3}\n', encoding="utf-8")
            self._prepare()
            self.settings.max_length = 16384
            self._prepare()
        self.assertEqual(self.calls, 3)

    def test_an_unreadable_cache_is_a_miss(self) -> None:
        self.cache.write_bytes(b"not a pickle")
        with mock.patch.dict(os.environ, {train_sft.GATE_CACHE_ENV: str(self.cache)}):
            self._prepare()
        self.assertEqual(self.calls, 1)

    def test_install_puts_the_three_replacements_into_the_trainer(self) -> None:
        saved = (research_cli.validate_research_splits, research_cli.normalize_research_rows, research_cli.prepare_research_examples)
        try:
            train_sft.install()
            self.assertIs(research_cli.validate_research_splits, train_sft.validate_minimal_splits)
            self.assertIs(research_cli.normalize_research_rows, train_sft.keep_row_tools)
            self.assertIs(research_cli.prepare_research_examples, train_sft.prepare_with_gate_cache)
        finally:
            (research_cli.validate_research_splits, research_cli.normalize_research_rows, research_cli.prepare_research_examples) = saved


def _episode(case: str, name: str, *, solved: bool, declared: str | None, strict: bool | None = None, outcome: str | None = None,
             damaged: int = 0, steps: list | None = None) -> dict:
    steps = steps if steps is not None else [
        {"action": {"tool": "correct_parameters", "arguments": {"line": 4}}, "result": {"status": "ok"},
         "policy_metrics": {"original_prompt_tokens": 9000}},
        {"action": {"tool": "finish", "arguments": {"outcome": declared}}, "result": {"status": "ok"},
         "policy_metrics": {"original_prompt_tokens": 12000}},
    ]
    return {"scenario_id": case, "compound_class": name, "outcome": outcome or declared, "steps": steps, "reward": {"total": 0.5},
            "score": {"solved": solved, "success": solved and declared == "resolved", "declared": declared, "estimator_passes": True,
                      "strict": {"solved": solved if strict is None else strict}, "actions": len(steps), "tool_errors": 0,
                      "corrections_in_effect": 1, "physical": {"damaged": damaged, "true_errors": 2, "repaired": 2 if solved else 1},
                      "cost": {"wls_solves": 2, "substation_solves": 0, "parameter_fits": 1, "measurement_repairs": 0}}}


class EvaluationSummaryTests(unittest.TestCase):
    def test_declared_passing_and_repaired_are_counted_apart(self) -> None:
        refused = [{"action": {"tool": None, "arguments": {}}, "result": {"status": "error", "error_code": "unparseable_action"},
                    "policy_metrics": {"original_prompt_tokens": 5000, "hit_max_new_tokens": True}}]
        records = [
            _episode("a", "x", solved=True, declared="resolved"),
            _episode("b", "x", solved=True, declared="unresolved"),
            _episode("c", "y", solved=False, declared="resolved", damaged=2),
            _episode("d", "y", solved=False, declared="unresolved"),
            _episode("e", "y", solved=False, declared=None, outcome="budget_exhausted", steps=refused),
            _episode("f", "y", solved=True, declared="resolved", strict=False),
        ]
        reference = {"a": {"expert": True, "truth": True}, "b": {"expert": True, "truth": True}, "c": {"expert": False, "truth": True},
                     "d": {"expert": False, "truth": False}, "f": {"expert": False, "truth": True}}
        summary = evaluate.summarize(records, reference)
        totals = summary["totals"]
        self.assertEqual((totals["cases"], totals["solved"], totals["strict_solved"]), (6, 3, 2))
        self.assertEqual((totals["solved_and_declared_resolved"], totals["solved_but_not_declared"], totals["false_resolved"]), (2, 1, 1))
        self.assertEqual((totals["declared_resolved"], totals["declared_unresolved"], totals["unresolved_declared_on_an_unsolved_case"]), (3, 2, 1))
        self.assertEqual((totals["budget_exhausted"], totals["episodes_with_a_damaged_healthy_object"], totals["damaged_healthy_objects"]), (1, 1, 2))
        self.assertEqual((totals["error:unparseable_action"], totals["steps_cut_at_max_new_tokens"]), (1, 1))
        self.assertEqual(summary["by_class"], {"x": {"cases": 2, "solved": 2, "strict_solved": 2}, "y": {"cases": 4, "solved": 1, "strict_solved": 0}})
        self.assertEqual(summary["by_reference"]["rule expert solves"], {"cases": 2, "solved": 2, "strict_solved": 2})
        self.assertEqual(summary["by_reference"]["only the truth-built sequence repairs"], {"cases": 2, "solved": 1, "strict_solved": 0})
        self.assertEqual(summary["by_reference"]["neither repairs"], {"cases": 1, "solved": 0, "strict_solved": 0})
        self.assertEqual(summary["prompt_tokens"]["max"], 12000.0)
        self.assertEqual(summary["actions_by_tool"]["finish"], 5)
        self.assertNotIn("by_reference", evaluate.summarize(records))

    def test_a_shard_names_every_nth_case(self) -> None:
        self.assertEqual(evaluate.parse_shard("1/4"), (1, 4))
        self.assertIsNone(evaluate.parse_shard(None))
        for bad in ("4/4", "a/2", "1", "-1/2"):
            with self.assertRaises(ValueError):
                evaluate.parse_shard(bad)

    def test_a_case_file_keeps_the_cases_it_names_in_the_suites_order(self) -> None:
        roots = [{"execution": {"scenario_id": name}} for name in ("a", "b", "c", "d")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            for payload in (["c", "a"], {"c": 1, "a": 2}, {"cases": ["a", "c"]}, {"cases": {"a": {}, "c": {}}}):
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual([evaluate.case_id(root) for root in evaluate.keep_cases(roots, path)], ["a", "c"], payload)
            path.write_text(json.dumps(["a", "z"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "1 cases of .* are not in the suite"):
                evaluate.keep_cases(roots, path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "names no case"):
                evaluate.keep_cases(roots, path)


class TruthSequenceTests(unittest.TestCase):
    """The file of truth-built sequences: the repaired success paths of a demonstration build, and nothing else."""

    @staticmethod
    def _record(case: str, *, solved: bool = True, kind: str = "success", actions=None, status: str = "ok") -> dict:
        actions = actions if actions is not None else [("correct_parameters", {"line": 6}), ("finish", {"outcome": "resolved"})]
        return {"scenario_id": case, "demonstration": kind, "solved": solved, "compound_class": "rx2_adjacent",
                "steps": [{"step": number, "action": {"tool": tool, "arguments": arguments}, "result": {"status": status}}
                          for number, (tool, arguments) in enumerate(actions, start=1)]}

    def test_the_file_holds_the_repaired_success_paths_and_reads_back(self) -> None:
        records = [self._record("b"), self._record("a", solved=False), self._record("b", kind="recovery"),
                   self._record("c", actions=[("inspect", {"view": "breakers"}), ("correct_topology", {"breaker": "CB_1", "status": "open"}),
                                              ("finish", {"outcome": "resolved"})])]
        payload = truth_sequences.build(records)
        self.assertEqual(payload["counts"], {"kept": 2, "not_repaired": 1, "success_paths": 3})
        self.assertEqual((list(payload["cases"]), payload["by_class"]), (["b", "c"], {"rx2_adjacent": 2}))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sequences.json"
            truth_sequences.dump(payload, path)
            text = path.read_bytes()
            self.assertNotIn(b"\r", text)
            self.assertEqual(text.count(b"\n"), 4 + 2 + 2 + 1)  # the head, a case a line, the closing lines
            loaded = truth_sequences.load(path)
            self.assertEqual(loaded["c"], [{"tool": "inspect", "arguments": {"view": "breakers"}},
                                           {"tool": "correct_topology", "arguments": {"breaker": "CB_1", "status": "open"}},
                                           {"tool": "finish", "arguments": {"outcome": "resolved"}}])
            self.assertEqual(truth_sequences.case_ids(path), {"b", "c"})
            path.write_text(json.dumps({"cases": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a file of truth-built sequences"):
                truth_sequences.load(path)

    def test_a_path_that_is_not_a_clean_repair_is_refused(self) -> None:
        for label, records in (
                ("two success paths", [self._record("a"), self._record("a")]),
                ("a refused call", [self._record("a", status="error")]),
                ("no declaration", [self._record("a", actions=[("correct_parameters", {"line": 6})])]),
                ("declared unresolved", [self._record("a", actions=[("finish", {"outcome": "unresolved"})])])):
            with self.assertRaises(ValueError, msg=label):
                truth_sequences.build(records)

    def test_the_cells_file_is_what_the_training_demonstrations_repair(self) -> None:
        path = Path(evaluate.REPO_ROOT, "research", "hpc", "minimal_rl_20261116", "train_truth_sequences.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["counts"], {"kept": 2093, "not_repaired": 457, "success_paths": 2550})
        sequences = truth_sequences.load(path)
        self.assertEqual(len(sequences), 2093)
        tools = view.tools({"breaker_model": True})
        names = {item["function"]["name"] for item in tools}
        for case, actions in sequences.items():
            self.assertEqual(actions[-1], {"tool": "finish", "arguments": {"outcome": "resolved"}}, case)
            self.assertTrue({action["tool"] for action in actions} <= names, case)
            for action in actions:
                validate(action, breakers=True)


if __name__ == "__main__":
    unittest.main()
