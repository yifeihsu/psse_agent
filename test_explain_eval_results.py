import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import explain_eval_results as explain  # noqa: E402

TEST_SPLIT = SCRIPT_DIR / "artifacts" / "traces" / "out_traces_balanced" / "sft_traces.test.jsonl"


def synthetic_row(sample_index: int = 0, *, family: str = "topology_error", correct: bool = True) -> dict:
    """A compact eval row shaped like build_result_from_state output."""
    user_payload = {
        "case_path": "case14",
        "z_obs": [1.0] * 10,
        "measurement_order": ["Vm", "Pinj"],
        "index_map": {"Vm": [0, 5], "Pinj": [5, 10]},
        "branch_info": [
            {"i": 0, "from_bus": 1, "to_bus": 2, "is_line": True},
            {"i": 1, "from_bus": 1, "to_bus": 5, "is_line": True},
        ],
    }
    verdict = {
        "verdict": {"has_error": True, "error_family": family, "confidence": 0.99},
        "evidence": {"global_metrics": {"global_residual_ratio": 300.7}},
        "suspect_location": {
            "domain": "topology",
            "details": {"substation": 1, "cb_name": "CB_1_B1_N3", "observed_status": "open"},
        },
        "action": {"applied_tool": "correct_topology_from_path"},
        "summary": "Residuals are widespread and consistent with a model/topology mismatch.",
    }
    wls_raw = {
        "success": True,
        "r": [0.1, 90.2, 85.7, 62.3, 57.4, 39.7, 0.2, 0.1, 0.3, 0.0],
        "lambdaN": [20.3, -108.0, 80.4, 135.7],
        "lambda_layout": "per_branch_R_X_interleaved",
        "global_residual_sum": 35713.9,
        "dof": 5,
        "measurement_jacobian": [[0.0] * 10 for _ in range(10)],
        "measurement_sigma": [0.01] * 10,
    }
    wls_compact = {
        "success": True,
        "top_residuals": [{"index0": 1, "channel": "Vm", "channel_offset": 1, "value": 90.2}],
        "top_lagrange": [{"lambda_index0": 3, "line_row0": 1, "from_bus": 1, "to_bus": 5, "value": 135.7}],
        "global_metrics": {
            "global_residual_sum": 35713.9,
            "global_residual_threshold": 118.75,
            "global_residual_ratio": 300.7,
        },
    }
    gt_family = family if correct else "parameter_error"
    return {
        "sample_index": sample_index,
        "gt_error_family": gt_family,
        "pred_error_family": family,
        "gt_has_error": True,
        "pred_has_error": True,
        "family_correct": correct,
        "detection_correct": True,
        "gt_error_families": [gt_family],
        "pred_error_families": [family],
        "family_set_correct": correct,
        "required_tools": ["correct_topology_from_path"],
        "required_tools_called": ["correct_topology_from_path"],
        "required_tools_missing": [],
        "required_tool_coverage": 1.0,
        "predicted_applied_tools": ["correct_topology_from_path"],
        "tool_calls": ["wls_from_path", "correct_topology_from_path"],
        "num_turns": 3,
        "predicted_verdict": verdict,
        "error": None,
        "input_truncated": False,
        "parse_notes": [],
        "last_raw_generation": json.dumps(verdict),
        "required_wls_satisfied": True,
        "runtime": {"generated_tokens_total": 500},
        "turn_trace": [
            {
                "turn": 1,
                "raw_generation": "<tool_call>",
                "parse_type": "tool_call",
                "tool_name": "wls_from_path",
                "tool_arguments": {"case_path": "case14"},
                "tool_arguments_for_prompt": {"case_path": "case14"},
                "tool_result": wls_raw,
                "tool_result_for_prompt": wls_compact,
            },
            {
                "turn": 2,
                "parse_type": "tool_call",
                "tool_name": "correct_topology_from_path",
                "tool_arguments_for_prompt": {"case_path": "case14", "cb_name": "CB_1_B1_N3", "desired_status": True},
                "tool_result": {"success": True, "cb_name": "CB_1_B1_N3", "applied_status": True},
                "tool_result_for_prompt": {"success": True, "cb_name": "CB_1_B1_N3", "applied_status": True},
            },
            {"turn": 3, "parse_type": "verdict", "verdict": verdict},
        ],
        "final_conversation": [
            {"role": "system", "content": "You are a power-system state-estimation diagnostic agent."},
            {"role": "user", "content": json.dumps(user_payload)},
            {"role": "assistant", "content": json.dumps(verdict)},
        ],
    }


def good_analysis(family: str = "topology_error", *, agrees: bool = True) -> dict:
    return {
        "verdict_restatement": {
            "verdict_present": True,
            "has_error": True,
            "error_family": "topology_error",
            "applied_tool": "correct_topology_from_path",
            "location": "substation 1, breaker CB_1_B1_N3 observed open",
        },
        "agrees_with_verdict": agrees,
        "analyst_assessment": {
            "has_error": True,
            "error_family": family,
            "confidence": 0.9,
            "location": "CB_1_B1_N3",
        },
        "explanation": "The initial WLS fails the global test with a ratio of 300.7.",
        "evidence_chain": ["global_residual_ratio 300.7 before correction"],
        "concerns": [] if agrees else ["Multipliers concentrate on line 2 (buses 1-5); parameter error fits better."],
        "process_flags": [],
        "cited_entities": {
            "buses": [1, 5],
            "line_index1": [2],
            "breakers": ["CB_1_B1_N3"],
            "measurement_index0": [1],
            "channels": ["Vm"],
        },
    }


def message_dict(analysis: dict, *, stop_reason: str = "end_turn") -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(analysis)}],
        "stop_reason": stop_reason,
        "model": "claude-opus-5-5",
        "usage": {"input_tokens": 1000, "output_tokens": 300},
    }


class AnalysisInputTests(unittest.TestCase):
    def test_labels_never_reach_the_model(self) -> None:
        row = synthetic_row()
        analysis_input = explain.build_analysis_input(row)
        serialized = json.dumps(analysis_input)
        for key in explain.FORBIDDEN_ROW_KEYS:
            self.assertNotIn(f'"{key}"', serialized)
        self.assertNotIn("gt_", serialized)
        self.assertEqual(analysis_input["case"]["case_path"], "case14")
        self.assertEqual(analysis_input["case"]["measurement_count"], 10)
        self.assertNotIn("z_obs", analysis_input["case"])
        self.assertEqual([turn["tool_name"] for turn in analysis_input["turns"] if "tool_name" in turn],
                         ["wls_from_path", "correct_topology_from_path"])
        wls_turn = analysis_input["turns"][0]
        self.assertIn("result_compact", wls_turn)
        self.assertIn("r", wls_turn["result_raw_extras"])
        self.assertIn("lambdaN", wls_turn["result_raw_extras"])
        self.assertNotIn("measurement_jacobian", wls_turn["result_raw_extras"])
        self.assertNotIn("success", wls_turn["result_raw_extras"])
        self.assertEqual(analysis_input["final_verdict"]["verdict"]["error_family"], "topology_error")
        self.assertEqual(analysis_input["turns"][-1]["verdict"], "see final_verdict")

    def test_rejected_verdict_turn_is_kept_in_full(self) -> None:
        row = synthetic_row()
        rejected = {"verdict": {"has_error": False, "error_family": "no_error", "confidence": 0.5}}
        row["turn_trace"].insert(
            0, {"turn": 0, "parse_type": "verdict", "verdict": rejected, "error": "Verdict before required wls_from_path at turn 1"}
        )
        analysis_input = explain.build_analysis_input(row)
        self.assertEqual(analysis_input["turns"][0]["rejected_verdict"], rejected)
        self.assertEqual(analysis_input["turns"][0]["error"], "Verdict before required wls_from_path at turn 1")

    def test_leak_guard_rejects_label_fields(self) -> None:
        with self.assertRaises(RuntimeError):
            explain.assert_no_label_leak({"turns": [{"gt_error_family": "x"}]})
        with self.assertRaises(RuntimeError):
            explain.assert_no_label_leak({"nested": {"required_tools": []}})

    def test_include_measurements_and_array_cap(self) -> None:
        row = synthetic_row()
        analysis_input = explain.build_analysis_input(row, include_measurements=True, max_array_items=4)
        z_obs = analysis_input["case"]["z_obs"]
        self.assertEqual(len(z_obs), 5)
        self.assertTrue(str(z_obs[-1]).endswith("more items omitted)"))

    def test_request_params_shape(self) -> None:
        requests, custom_ids = explain.build_batch_requests(
            [synthetic_row(7)],
            model="claude-opus-5-5",
            max_tokens=8192,
            effort="high",
            system_prompt=explain.ANALYST_SYSTEM_PROMPT,
            structured_output=True,
            include_measurements=False,
            max_array_items=explain.DEFAULT_MAX_ARRAY_ITEMS,
        )
        self.assertEqual(custom_ids, {"sample-7": 7})
        params = requests[0]["params"]
        self.assertEqual(params["model"], "claude-opus-5-5")
        self.assertEqual(params["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(params["output_config"]["effort"], "high")
        self.assertEqual(params["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertIn("CB_1_B1_N3", params["messages"][0]["content"])
        self.assertNotIn("gt_error_family", params["messages"][0]["content"])
        estimate = explain.estimate_request_cost(requests, price_input_per_mtok=2.0, price_output_per_mtok=10.0)
        self.assertEqual(estimate["requests"], 1)
        self.assertGreater(estimate["estimated_input_tokens"], 500)


class ParsingTests(unittest.TestCase):
    def test_parse_ok(self) -> None:
        record = explain.message_to_record(message_dict(good_analysis()))
        analysis, status, error = explain.parse_analysis_record(record)
        self.assertEqual(status, "ok")
        self.assertIsNone(error)
        self.assertTrue(analysis["agrees_with_verdict"])
        self.assertEqual(record["usage"]["input_tokens"], 1000)

    def test_parse_wrapped_json(self) -> None:
        record = {"text": "Here you go:\n```json\n" + json.dumps(good_analysis()) + "\n```", "stop_reason": "end_turn"}
        analysis, status, _ = explain.parse_analysis_record(record)
        self.assertEqual(status, "ok")
        self.assertEqual(analysis["analyst_assessment"]["error_family"], "topology_error")

    def test_parse_truncated_refused_and_garbage(self) -> None:
        truncated = {"text": '{"verdict_restatement": {"verdict_present": true', "stop_reason": "max_tokens"}
        self.assertEqual(explain.parse_analysis_record(truncated)[1], "truncated")
        refused = {"text": "", "stop_reason": "refusal", "stop_details": {"category": "x", "explanation": "y"}}
        self.assertEqual(explain.parse_analysis_record(refused)[1], "refused")
        garbage = {"text": "no json here", "stop_reason": "end_turn"}
        self.assertEqual(explain.parse_analysis_record(garbage)[1], "parse_error")
        partial = {"text": json.dumps({"explanation": "only this"}), "stop_reason": "end_turn"}
        analysis, status, error = explain.parse_analysis_record(partial)
        self.assertEqual(status, "schema_error")
        self.assertIn("missing keys", error)


class ChecksAndScoringTests(unittest.TestCase):
    def test_checks_pass_for_consistent_analysis(self) -> None:
        analysis_input = explain.build_analysis_input(synthetic_row())
        checks = explain.run_local_checks(good_analysis(), analysis_input)
        self.assertTrue(checks["passed"], checks)

    def test_checks_flag_unknown_entities_and_mismatches(self) -> None:
        analysis_input = explain.build_analysis_input(synthetic_row())
        analysis = good_analysis()
        analysis["cited_entities"]["buses"].append(99)
        analysis["cited_entities"]["line_index1"].append(3)
        analysis["cited_entities"]["breakers"].append("CB_9_B9_N9")
        analysis["cited_entities"]["measurement_index0"].append(10)
        analysis["cited_entities"]["channels"].append("Qt")
        analysis["verdict_restatement"]["error_family"] = "parameter_error"
        analysis["analyst_assessment"]["confidence"] = 1.5
        checks = explain.run_local_checks(analysis, analysis_input)
        self.assertFalse(checks["passed"])
        self.assertIn("unknown_entities_cited", checks["problems"])
        self.assertEqual(checks["unknown_entities"]["buses"], [99])
        self.assertEqual(checks["unknown_entities"]["line_index1"], [3])
        self.assertEqual(checks["unknown_entities"]["breakers"], ["CB_9_B9_N9"])
        self.assertEqual(checks["unknown_entities"]["measurement_index0"], [10])
        self.assertEqual(checks["unknown_entities"]["channels"], ["Qt"])
        self.assertIn("restatement_family_mismatch", checks["problems"])
        self.assertIn("analyst_confidence_out_of_range", checks["problems"])

    def test_checks_flag_inconsistent_agreement(self) -> None:
        analysis_input = explain.build_analysis_input(synthetic_row())
        analysis = good_analysis(family="parameter_error", agrees=True)
        checks = explain.run_local_checks(analysis, analysis_input)
        self.assertIn("agreement_inconsistent_with_assessment", checks["problems"])

    def test_checks_without_verdict(self) -> None:
        row = synthetic_row()
        row["predicted_verdict"] = None
        row["turn_trace"] = row["turn_trace"][:2]
        analysis_input = explain.build_analysis_input(row)
        analysis = good_analysis()
        analysis["verdict_restatement"]["verdict_present"] = False
        checks = explain.run_local_checks(analysis, analysis_input)
        self.assertNotIn("restatement_presence_mismatch", checks["problems"])
        self.assertNotIn("restatement_family_mismatch", checks["problems"])

    def test_score_row_outcomes(self) -> None:
        confirmed = explain.score_row(synthetic_row(correct=True), good_analysis())
        self.assertEqual(confirmed["review_outcome"], "confirmed_correct")
        self.assertTrue(confirmed["analyst_family_correct"])
        caught = explain.score_row(synthetic_row(correct=False), good_analysis("parameter_error", agrees=False))
        self.assertEqual(caught["review_outcome"], "caught_error")
        self.assertTrue(caught["analyst_family_correct"])
        missed = explain.score_row(synthetic_row(correct=False), good_analysis(agrees=True))
        self.assertEqual(missed["review_outcome"], "missed_error")
        self.assertFalse(missed["analyst_family_correct"])
        false_alarm = explain.score_row(synthetic_row(correct=True), good_analysis("parameter_error", agrees=False))
        self.assertEqual(false_alarm["review_outcome"], "false_alarm")
        self.assertIsNone(explain.score_row(synthetic_row(), None)["review_outcome"])

    def test_merge_and_summary(self) -> None:
        rows = [synthetic_row(0, correct=True), synthetic_row(1, correct=False), synthetic_row(2)]
        results = {
            0: {"status": "succeeded", "record": explain.message_to_record(message_dict(good_analysis())), "error": None},
            1: {
                "status": "succeeded",
                "record": explain.message_to_record(message_dict(good_analysis("parameter_error", agrees=False))),
                "error": None,
            },
            2: {"status": "errored", "record": None, "error": "server_error: boom"},
        }
        merged = explain.merge_results(
            rows,
            results_by_index=results,
            existing_by_index={},
            base_meta={"model": "claude-opus-5-5", "batch_id": "msgbatch_1"},
            include_measurements=False,
            max_array_items=explain.DEFAULT_MAX_ARRAY_ITEMS,
        )
        self.assertEqual([row["analysis_meta"]["status"] for row in merged], ["ok", "ok", "errored"])
        self.assertEqual(merged[0]["analysis_meta"]["custom_id"], "sample-0")
        self.assertEqual(merged[2]["analysis_meta"]["error"], "server_error: boom")
        self.assertIsNone(merged[2]["analysis"])
        summary = explain.summarize_rows(merged)
        self.assertEqual(summary["analyzed"], 2)
        self.assertEqual(summary["statuses"], {"ok": 2, "errored": 1})
        self.assertEqual(summary["reviewer_recall_on_agent_errors"], 1.0)
        self.assertEqual(summary["reviewer_false_alarm_rate"], 0.0)
        self.assertEqual(summary["analyst_family_accuracy"], 1.0)
        self.assertEqual(summary["agent_family_accuracy"], 0.5)
        self.assertEqual(summary["usage_totals"]["input_tokens"], 2000)

        # A second merge keeps the earlier good analyses and only overlays new results.
        remerged = explain.merge_results(
            rows,
            results_by_index={2: results[0]},
            existing_by_index={row["sample_index"]: row for row in merged},
            base_meta={"model": "claude-opus-5-5", "batch_id": "msgbatch_2"},
            include_measurements=False,
            max_array_items=explain.DEFAULT_MAX_ARRAY_ITEMS,
        )
        self.assertEqual([row["analysis_meta"]["status"] for row in remerged], ["ok", "ok", "ok"])
        self.assertEqual(remerged[0]["analysis_meta"]["batch_id"], "msgbatch_1")
        self.assertEqual(remerged[2]["analysis_meta"]["batch_id"], "msgbatch_2")


class CliTests(unittest.TestCase):
    def test_dry_run_writes_requests_and_skips_rows_without_trace(self) -> None:
        failed_row = {
            "sample_index": 1,
            "gt_error_family": "no_error",
            "pred_error_family": None,
            "family_correct": False,
            "predicted_verdict": None,
            "error": "CRITICAL ERROR: boom",
            "tool_calls": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "eval_1.jsonl"
            explain.write_jsonl(input_path, [synthetic_row(0), failed_row])
            code = explain.main(["--input", str(input_path), "--mode", "dry-run"])
            self.assertEqual(code, 0)
            requests = explain.load_jsonl(Path(tmp) / "eval_1.explained.requests.jsonl")
            self.assertEqual([request["custom_id"] for request in requests], ["sample-0"])

            code = explain.main(["--input", str(input_path), "--mode", "dry-run", "--include-failed"])
            self.assertEqual(code, 0)
            requests = explain.load_jsonl(Path(tmp) / "eval_1.explained.requests.jsonl")
            self.assertEqual([request["custom_id"] for request in requests], ["sample-0", "sample-1"])

    def test_only_missing_uses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "eval_2.jsonl"
            rows = [synthetic_row(0), synthetic_row(1)]
            explain.write_jsonl(input_path, rows)
            done = dict(rows[0])
            done["analysis_meta"] = {"status": "ok"}
            explain.write_jsonl(Path(tmp) / "eval_2.explained.jsonl", [done])
            code = explain.main(["--input", str(input_path), "--mode", "dry-run", "--only-missing"])
            self.assertEqual(code, 0)
            requests = explain.load_jsonl(Path(tmp) / "eval_2.explained.requests.jsonl")
            self.assertEqual([request["custom_id"] for request in requests], ["sample-1"])

    def test_sample_index_filter_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "eval_3.jsonl"
            explain.write_jsonl(input_path, [synthetic_row(i) for i in range(4)])
            explain.main(["--input", str(input_path), "--mode", "dry-run", "--sample-index", "2", "--sample-index", "3", "--limit", "1"])
            requests = explain.load_jsonl(Path(tmp) / "eval_3.explained.requests.jsonl")
            self.assertEqual([request["custom_id"] for request in requests], ["sample-2"])


@unittest.skipUnless(TEST_SPLIT.exists(), "balanced test split not present")
class RealTraceTests(unittest.TestCase):
    """Build eval-shaped rows from the real test split and check the request builder on them."""

    @staticmethod
    def row_from_trace(sample_index: int, record: dict) -> dict:
        messages = record["messages"]
        turn_trace = []
        tool_calls = []
        pending_call = None
        turn = 0
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                function = message["tool_calls"][0]["function"]
                pending_call = {
                    "tool_name": function["name"],
                    "tool_arguments_for_prompt": json.loads(function["arguments"]),
                }
                tool_calls.append(function["name"])
            elif message["role"] == "tool" and pending_call is not None:
                turn += 1
                payload = json.loads(message["content"])
                turn_trace.append(
                    {"turn": turn, "parse_type": "tool_call", **pending_call,
                     "tool_result": payload, "tool_result_for_prompt": payload}
                )
                pending_call = None
        final = json.loads([m for m in messages if m["role"] == "assistant" and m.get("content")][-1]["content"])
        turn_trace.append({"turn": turn + 1, "parse_type": "verdict", "verdict": final})
        family = final["verdict"]["error_family"]
        return {
            "sample_index": sample_index,
            "gt_error_family": family,
            "pred_error_family": family,
            "gt_has_error": final["verdict"]["has_error"],
            "pred_has_error": final["verdict"]["has_error"],
            "family_correct": True,
            "detection_correct": True,
            "tool_calls": tool_calls,
            "num_turns": len(turn_trace),
            "predicted_verdict": final,
            "error": None,
            "input_truncated": False,
            "parse_notes": [],
            "required_wls_satisfied": True,
            "turn_trace": turn_trace,
            "final_conversation": messages,
        }

    def test_real_rows_build_requests(self) -> None:
        rows = []
        seen = set()
        with TEST_SPLIT.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                row = self.row_from_trace(len(rows), record)
                if row["gt_error_family"] in seen:
                    continue
                seen.add(row["gt_error_family"])
                rows.append(row)
        self.assertGreaterEqual(len(rows), 5)
        requests, _ = explain.build_batch_requests(
            rows,
            model="claude-opus-5-5",
            max_tokens=8192,
            effort="high",
            system_prompt=explain.ANALYST_SYSTEM_PROMPT,
            structured_output=True,
            include_measurements=False,
            max_array_items=explain.DEFAULT_MAX_ARRAY_ITEMS,
        )
        for row, request in zip(rows, requests):
            text = request["params"]["messages"][0]["content"]
            self.assertNotIn("gt_error_family", text)
            analysis_input = explain.build_analysis_input(row)
            universe = explain.entity_universe(analysis_input)
            self.assertTrue(universe["known"])
            self.assertEqual(universe["n_branches"], 20)
            self.assertEqual(universe["measurement_count"], 122)
            checks = explain.run_local_checks(
                {
                    **good_analysis(row["gt_error_family"]),
                    "verdict_restatement": {
                        "verdict_present": True,
                        "has_error": row["gt_has_error"],
                        "error_family": row["gt_error_family"],
                        "applied_tool": "",
                        "location": "",
                    },
                    "analyst_assessment": {
                        "has_error": row["gt_has_error"],
                        "error_family": row["gt_error_family"],
                        "confidence": 0.9,
                        "location": "",
                    },
                    "cited_entities": {"buses": [1, 2], "line_index1": [1], "breakers": [], "measurement_index0": [0], "channels": ["Vm"]},
                },
                analysis_input,
            )
            self.assertTrue(checks["passed"], (row["gt_error_family"], checks))
        estimate = explain.estimate_request_cost(requests, price_input_per_mtok=2.0, price_output_per_mtok=10.0)
        self.assertLess(estimate["estimated_input_tokens"] / len(requests), 12000)


if __name__ == "__main__":
    unittest.main()
