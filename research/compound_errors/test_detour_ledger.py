"""The detour ledger on synthetic episodes in both record formats (2026-11-05)."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("detour_ledger", _HERE / "detour_ledger.py")
ledger = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["detour_ledger"] = ledger
_SPEC.loader.exec_module(ledger)


def _slim(tool: str, status: str = "success", error_code: str | None = None, disposition: str | None = None,
          arguments: dict | None = None, comparison: dict | None = None) -> dict:
    record = {"tool": tool, "status": status, "error_code": error_code, "disposition": disposition,
              "arguments": arguments or {}}
    if comparison is not None:
        record["comparison"] = comparison
    return record


def _compound_episode(trace: list[dict], corrections: list[dict], *, success: bool, klass: str = "control_meter",
                      escalation: str = "operator_escalation:recovery_options_exhausted") -> dict:
    for index, step in enumerate(trace):
        step["step"] = index
    return {"scenario_id": "r0_test", "class": klass, "success": success, "steps": len(trace),
            "tools": [s["tool"] for s in trace], "trace": trace, "corrections": corrections, "terminal": True,
            "escalation_request": escalation, "wls_calls": sum(1 for s in trace if s["tool"] == "run_wls"),
            "invalid_action_count": sum(1 for s in trace if s["status"] != "success"), "loop_detected": False}


class CompoundRecordTests(unittest.TestCase):
    def test_floor_episode_has_no_waste(self) -> None:
        # A single meter fixed at the protocol floor: WLS, context, correction, verify, commit, handoff.
        trace = [_slim("run_wls"), _slim("get_measurement_context"), _slim("correct_measurements", arguments={"suspect_group": [3]}),
                 _slim("run_wls"), _slim("commit_state", disposition="ACCEPT_FINAL"), _slim("ask_for_more_evidence")]
        corrections = [{"step": 2, "tool": "correct_measurements", "target_is_true": True}]
        row = ledger.episode_ledger(_compound_episode(trace, corrections, success=True), atoms=1)
        self.assertEqual(row["actions"], 6)
        self.assertEqual(row["floor"], 6)
        self.assertEqual(row["slack"], 0)
        self.assertEqual(row["wasted"], 0)
        self.assertEqual(row["hypotheses"], 1)
        self.assertEqual(row["accepted_true"], 1)
        self.assertEqual(row["routes"], 1)
        self.assertEqual(row["fruitless_routes"], 0)

    def test_meter_control_detour_is_attributed(self) -> None:
        # The IEEE 118 meter-control detour: after a true meter commit, the parameter route is taken, two lines are
        # tried and rolled back and one is refused by the corrector, before the measurement route fixes the second meter.
        trace = [
            _slim("run_wls"), _slim("get_measurement_context"),
            _slim("correct_measurements", arguments={"suspect_group": [3]}), _slim("run_wls"),
            _slim("commit_state", disposition="ACCEPT_PARTIAL"), _slim("run_wls"),
            _slim("get_parameter_context"), _slim("compare_parameter_candidates", comparison={"best_line": 5}),
            _slim("correct_parameters", arguments={"line_index": 5}), _slim("run_wls"),
            _slim("rollback_state", disposition="REJECT"),
            _slim("correct_parameters", arguments={"line_index": 7}), _slim("run_wls"),
            _slim("rollback_state", disposition="REJECT"),
            _slim("correct_parameters", "failure", "parameter_correction_failure", arguments={"line_index": 9}),
            _slim("get_topology_context"),
            _slim("get_measurement_context"),
            _slim("correct_measurements", arguments={"suspect_group": [8]}), _slim("run_wls"),
            _slim("commit_state", disposition="ACCEPT_FINAL"), _slim("ask_for_more_evidence"),
        ]
        corrections = [
            {"step": 2, "tool": "correct_measurements", "target_is_true": True},
            {"step": 8, "tool": "correct_parameters", "target_is_true": False},
            {"step": 11, "tool": "correct_parameters", "target_is_true": False},
            {"step": 14, "tool": "correct_parameters", "target_is_true": False},
            {"step": 17, "tool": "correct_measurements", "target_is_true": True},
        ]
        row = ledger.episode_ledger(_compound_episode(trace, corrections, success=True), atoms=2)
        self.assertEqual(row["actions"], 21)
        self.assertEqual(row["floor"], 10)
        self.assertEqual(row["slack"], 11)
        self.assertEqual(row["hypotheses"], 5)
        self.assertEqual(row["accepted_true"], 2)
        self.assertEqual(row["rejected_wrong"], 2)
        self.assertEqual(row["refused_wrong"], 1)
        self.assertEqual(row["refused_tool_refused"], 1)
        self.assertEqual(row["refused_policy_invalid"], 0)
        self.assertEqual(row["rollbacks"], 2)
        # Two rejected cycles of three actions, one refused correction, and the parameter and topology fetches that led
        # to no commit of their family: 6 + 1 + 2 = 9 wasted actions.
        self.assertEqual(row["wasted_rejected_cycles"], 6)
        self.assertEqual(row["wasted_fruitless_fetches"], 2)
        self.assertEqual(row["fruitless_routes_parameter"], 1)
        self.assertEqual(row["fruitless_routes_topology"], 1)
        self.assertEqual(row["wasted"], 9)
        self.assertEqual(row["comparisons"], 1)
        self.assertEqual(row["comparisons_productive"], 0)
        self.assertAlmostEqual(row["hypotheses_per_accepted"], 2.5)
        self.assertEqual(row["refused_codes"], {"parameter_correction_failure": 1})

    def test_gate_refusal_counts_as_policy_invalid_and_open_candidate_as_undisposed(self) -> None:
        trace = [
            _slim("run_wls"), _slim("get_parameter_context"),
            _slim("correct_parameters", "failure", "correction_not_supported_by_current_context", arguments={"line_index": 2}),
            _slim("correct_parameters", arguments={"line_index": 3}), _slim("run_wls"),
        ]
        corrections = [{"step": 2, "tool": "correct_parameters", "target_is_true": True},
                       {"step": 3, "tool": "correct_parameters", "target_is_true": False}]
        episode = _compound_episode(trace, corrections, success=False, klass="control_rx",
                                    escalation="operator_escalation:recovery_budget_exhausted")
        episode["terminal"] = False
        row = ledger.episode_ledger(episode, atoms=1)
        self.assertEqual(row["refused_policy_invalid"], 1)
        self.assertEqual(row["refused_true"], 1)
        self.assertEqual(row["undisposed"], 1)
        self.assertEqual(row["wasted_undisposed"], 2)
        self.assertTrue(row["budget_exhausted"])
        # The parameter fetch produced no commit, so it is fruitless as well.
        self.assertEqual(row["wasted"], 1 + 2 + 1)

    def test_productive_comparison_is_exploration_not_waste(self) -> None:
        trace = [_slim("run_wls"), _slim("get_parameter_context"),
                 _slim("compare_parameter_candidates", comparison={"best_line": 4}),
                 _slim("correct_parameters", arguments={"line_index": 4}), _slim("run_wls"),
                 _slim("commit_state", disposition="ACCEPT_FINAL"), _slim("ask_for_more_evidence")]
        corrections = [{"step": 3, "tool": "correct_parameters", "target_is_true": True}]
        row = ledger.episode_ledger(_compound_episode(trace, corrections, success=True, klass="control_rx"), atoms=1)
        self.assertEqual(row["slack"], 1)
        self.assertEqual(row["wasted"], 0)
        self.assertEqual(row["comparisons_productive"], 1)


class EvaluatorRecordTests(unittest.TestCase):
    def test_evaluator_trace_is_read_and_interventions_are_skipped(self) -> None:
        def step(tool: str, status: str = "success", error: str | None = None, disposition: str | None = None,
                 intervention: bool = False, arguments: dict | None = None) -> dict:
            return {"intervention": intervention, "action": {"tool": tool, "arguments": {"state_id": "s", **(arguments or {})}},
                    "execution_status": status, "error_code": error, "candidate_disposition_offline": disposition}
        trace = [step("run_wls", intervention=True), step("run_wls"), step("get_topology_context"),
                 step("correct_topology", arguments={"cb_name": "CB_1"}), step("run_wls"),
                 step("rollback_state", disposition="REJECT"), step("get_parameter_context"),
                 step("correct_parameters", arguments={"line_index": 1}), step("run_wls"),
                 step("commit_state", disposition="ACCEPT_FINAL"), step("ask_for_more_evidence")]
        episode = {"scenario_id": "r0_eval", "family": "parameter", "cardinality": 1, "steps": 10, "policy_steps": 10,
                   "truth_audited_task_success": True, "trace": trace, "wls_calls": 3, "invalid_action_count": 0,
                   "loop_detected": False, "terminal": True}
        row = ledger.episode_ledger(episode)
        self.assertEqual(row["group"], "parameter")
        self.assertEqual(row["floor"], 6)
        self.assertEqual(row["actions"], 10)
        self.assertEqual(row["rejected"], 1)
        self.assertEqual(row["accepted"], 1)
        # Truth flags are unknown in this format.
        self.assertEqual(row["rejected_true"] + row["rejected_wrong"], 0)
        self.assertEqual(row["fruitless_routes_topology"], 1)
        self.assertEqual(row["wasted"], 3 + 1)

    def test_record_without_trace_keeps_cost_columns_only(self) -> None:
        row = ledger.episode_ledger({"scenario_id": "x", "family": "measurement", "success": True, "steps": 7,
                                     "wls_calls": 2, "invalid_action_count": 0, "loop_detected": False})
        self.assertFalse(row["trace_available"])
        self.assertEqual(row["actions"], 7)
        self.assertNotIn("wasted", row)


class AggregateAndSummaryTests(unittest.TestCase):
    def test_aggregate_splits_by_group_and_outcome(self) -> None:
        trace_ok = [_slim("run_wls"), _slim("get_measurement_context"), _slim("correct_measurements"), _slim("run_wls"),
                    _slim("commit_state", disposition="ACCEPT_FINAL"), _slim("ask_for_more_evidence")]
        trace_bad = [_slim("run_wls"), _slim("get_parameter_context"), _slim("correct_parameters"), _slim("run_wls"),
                     _slim("rollback_state", disposition="REJECT"), _slim("ask_for_more_evidence")]
        rows = [ledger.episode_ledger(_compound_episode(list(map(dict, trace_ok)), [], success=True), atoms=1),
                ledger.episode_ledger(_compound_episode(list(map(dict, trace_bad)), [], success=False), atoms=1)]
        agg = ledger.aggregate(rows)
        self.assertEqual(agg["all"]["all"]["n"], 2)
        self.assertEqual(agg["all"]["solved"]["total_wasted"], 0)
        self.assertEqual(agg["all"]["failed"]["total_wasted"], 4)
        self.assertEqual(agg["groups"]["control_meter"]["failed"]["mean_rollbacks"], 1.0)
        text = ledger.markdown(agg, "test")
        self.assertIn("| control_meter | failed |", text)

    def test_summary_mode_reads_the_verdict_ledger(self) -> None:
        summary = {"expert": "baseline_compare", "max_steps": 40, "system": "case118", "by_class": {
            "control_meter": {"n": 50, "success": 45, "mean_steps": 21.6, "mean_wls_calls": 5.78, "mean_comparisons": 0.62,
                              "verdicts": {"correct_measurements:true:ACCEPT_FINAL": 45,
                                           "correct_measurements:true:ACCEPT_PARTIAL": 51,
                                           "correct_parameters:wrong:REJECT": 143},
                              "invalid_action_episodes": 0, "loop_episodes": 0}}}
        manifest = {"classes": [{"name": "control_meter", "parameters": 0, "meters": [1, 3], "breakers": 0}]}
        atoms = ledger.atoms_by_class(manifest)
        self.assertEqual(atoms, {"control_meter": 2.0})
        out = ledger.summary_ledger(summary, atoms=atoms)
        row = out["classes"]["control_meter"]
        self.assertEqual(row["floor"], 10.0)
        self.assertAlmostEqual(row["slack"], 11.6)
        self.assertEqual(row["hypotheses"], 239)
        self.assertEqual(row["accepted_true"], 96)
        self.assertEqual(row["rejected_wrong"], 143)
        self.assertAlmostEqual(row["est_wasted_actions"], 429.0)
        self.assertAlmostEqual(row["est_wasted_per_root"], 8.58)
        self.assertAlmostEqual(row["est_wasted_share_of_slack"], 429.0 / (11.6 * 50))
        text = ledger.summary_markdown({"deployed": out})
        self.assertIn("| deployed | control_meter | 50 | 45 |", text)

    def test_cli_episodes_mode_writes_outputs(self) -> None:
        trace = [_slim("run_wls"), _slim("get_measurement_context"), _slim("correct_measurements"), _slim("run_wls"),
                 _slim("commit_state", disposition="ACCEPT_FINAL"), _slim("ask_for_more_evidence")]
        episode = _compound_episode(trace, [{"step": 2, "tool": "correct_measurements", "target_is_true": True}], success=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"classes": [{"name": "control_meter", "parameters": 0, "meters": 1, "breakers": 0}]}))
            out = Path(tmp) / "ledger"
            code = ledger.main(["episodes", "--episodes", str(path), "--manifest", str(manifest), "--output-dir", str(out)])
            self.assertEqual(code, 0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["aggregate"]["all"]["all"]["mean_floor"], 6.0)
            self.assertTrue((out / "ledger.jsonl").exists())
            self.assertTrue((out / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
