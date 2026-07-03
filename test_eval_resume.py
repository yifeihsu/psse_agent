import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from eval_sft_agent_gemma_v4 import (
    CONTROLLER_STATE_KEY,
    TOOL_MAP,
    apply_tool_scope_to_messages,
    execute_tool,
    load_tools,
    load_completed_results_for_resume,
    parse_gemma_generation,
    pending_verification_error,
    validate_completed_results_for_resume,
)
from trace_protocol import SCADA_HARMONIC_SYSTEM_PROMPT, SYSTEM_PROMPT


class EvalResumeTests(unittest.TestCase):
    def test_canonical_eval_runtime_registers_hif_nlm_tool(self) -> None:
        self.assertIn("run_three_phase_nlm_from_path", TOOL_MAP)

    def test_final_json_parser_strips_trailing_gemma_turn_marker(self) -> None:
        class DummyTokenizer:
            def parse_response(self, raw: str):
                return {"content": raw}

        parsed = parse_gemma_generation(
            '{"verdict":{"has_error":false,"error_family":"no_error"}}<turn|>',
            DummyTokenizer(),
        )

        self.assertEqual(parsed["type"], "verdict")
        self.assertFalse(parsed["content"]["verdict"]["has_error"])

    def test_truncates_malformed_trailing_jsonl_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "eval.jsonl"
            output_path.write_text(
                json.dumps({"sample_index": 0, "family_correct": True}) + "\n{partial",
                encoding="utf-8",
            )

            rows = load_completed_results_for_resume(
                str(output_path),
                truncate_partial_output=True,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_index"], 0)
            self.assertEqual(output_path.read_text(encoding="utf-8").count("\n"), 1)

    def test_rejects_malformed_trailing_jsonl_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "eval.jsonl"
            output_path.write_text('{"sample_index": 0}\n{partial', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "malformed trailing JSONL"):
                load_completed_results_for_resume(
                    str(output_path),
                    truncate_partial_output=False,
                )

    def test_rejects_non_prefix_ordered_sample_indices(self) -> None:
        completed = [{"sample_index": 0}, {"sample_index": 2}]

        with self.assertRaisesRegex(RuntimeError, "not prefix ordered"):
            validate_completed_results_for_resume(
                completed,
                total_samples=3,
                output_path="eval.jsonl",
            )

    def test_rejects_more_completed_rows_than_requested_samples(self) -> None:
        completed = [{"sample_index": 0}, {"sample_index": 1}]

        with self.assertRaisesRegex(RuntimeError, "already has 2 results"):
            validate_completed_results_for_resume(
                completed,
                total_samples=1,
                output_path="eval.jsonl",
            )

    def test_verification_snapshot_can_be_requested_by_stage_only(self) -> None:
        runtime_context = {
            "tool_context": {
                "verification_snapshots": {
                    "post_measurement_correction": {
                        "case_path": "case14::post_measurement_correction::abc123",
                        "z_obs": [1.0, 2.0],
                        "stage": "post_measurement_correction",
                    }
                }
            }
        }
        hidden_context = {
            CONTROLLER_STATE_KEY: {
                "wls_completed": True,
                "pending_verification_stage": "post_measurement_correction",
            }
        }

        result = execute_tool(
            "get_verification_snapshot",
            {"stage": "post_measurement_correction"},
            runtime_context=runtime_context,
            hidden_context=hidden_context,
        )

        self.assertEqual(result["case_path"], "case14::post_measurement_correction::abc123")
        self.assertEqual(hidden_context["snapshot_context"]["z_obs"], [1.0, 2.0])
        self.assertTrue(hidden_context[CONTROLLER_STATE_KEY]["awaiting_verification_wls"])

    def test_verification_snapshot_empty_args_use_active_stage(self) -> None:
        runtime_context = {
            "tool_context": {
                "verification_snapshots": {
                    "post_topology_correction": {
                        "case_path": "case14::post_topology_correction::abc123",
                        "z_obs": [3.0],
                        "stage": "post_topology_correction",
                    }
                }
            }
        }
        hidden_context = {
            CONTROLLER_STATE_KEY: {
                "wls_completed": True,
                "pending_verification_stage": "post_topology_correction",
            }
        }

        result = execute_tool(
            "get_verification_snapshot",
            {},
            runtime_context=runtime_context,
            hidden_context=hidden_context,
        )

        self.assertEqual(result["stage"], "post_topology_correction")
        self.assertEqual(result["case_path"], "case14::post_topology_correction::abc123")

    def test_controller_blocks_correction_without_required_helper_context(self) -> None:
        runtime_context = {"tool_context": {"parameter_context": {"case_path": "case14"}}}
        hidden_context = {CONTROLLER_STATE_KEY: {"wls_completed": True}}

        result = execute_tool(
            "correct_parameters_from_path",
            {"case_path": "case14", "line_index": 1},
            runtime_context=runtime_context,
            hidden_context=hidden_context,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["tool_error_type"], "controller_precondition")
        self.assertIn("get_parameter_context", result["allowed_tools"])

    def test_controller_blocks_repeated_verification_snapshot_before_wls(self) -> None:
        runtime_context = {
            "tool_context": {
                "verification_snapshots": {
                    "post_measurement_correction": {
                        "case_path": "case14::post_measurement_correction::abc123",
                        "z_obs": [1.0],
                        "stage": "post_measurement_correction",
                    }
                }
            }
        }
        hidden_context = {
            CONTROLLER_STATE_KEY: {
                "wls_completed": True,
                "pending_verification_stage": "post_measurement_correction",
            }
        }

        execute_tool(
            "get_verification_snapshot",
            {"stage": "post_measurement_correction"},
            runtime_context=runtime_context,
            hidden_context=hidden_context,
        )
        repeated = execute_tool(
            "get_verification_snapshot",
            {"stage": "post_measurement_correction"},
            runtime_context=runtime_context,
            hidden_context=hidden_context,
        )

        self.assertFalse(repeated["success"])
        self.assertIn("wls_from_path", repeated["allowed_tools"])

    def test_scada_harmonic_eval_scope_excludes_hif_tools_and_prompt_text(self) -> None:
        args = Namespace(include_tool_schemas=True, tools_file="", tool_scope="scada_harmonic")
        tool_names = {tool["function"]["name"] for tool in load_tools(args)}
        scoped_messages = apply_tool_scope_to_messages(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "{}"}],
            "scada_harmonic",
        )

        self.assertIn("run_hse_from_path", tool_names)
        self.assertNotIn("run_three_phase_nlm_from_path", tool_names)
        self.assertEqual(scoped_messages[0]["content"], SCADA_HARMONIC_SYSTEM_PROMPT)
        self.assertNotIn("high_impedance_fault", scoped_messages[0]["content"])
        self.assertNotIn("three_phase_imbalance", scoped_messages[0]["content"])
        self.assertNotIn("top_hif_groups", scoped_messages[0]["content"])

    def test_sequence_only_parameter_topology_final_skips_post_parameter_verification(self) -> None:
        hidden_context = {
            CONTROLLER_STATE_KEY: {
                "wls_completed": True,
                "pending_verification_stage": "post_parameter_correction",
            }
        }
        gt_verdict = {
            "verdict": {
                "has_error": True,
                "error_family": "parameter_error",
                "error_families": ["parameter_error", "topology_error"],
            }
        }
        predicted = {
            "verdict": {
                "has_error": True,
                "error_family": "parameter_error",
                "error_families": ["parameter_error", "topology_error"],
            },
            "action": {
                "diagnosis_status": "curriculum_only",
                "tool_steps": [
                    {"family": "topology_error", "verification_policy": "verified_wls"},
                    {"family": "parameter_error", "verification_policy": "sequence_only"},
                ],
            },
        }

        self.assertIsNone(
            pending_verification_error(
                hidden_context,
                gt_verdict=gt_verdict,
                candidate_verdict=predicted,
            )
        )

    def test_sequence_only_exception_does_not_apply_to_normal_parameter_case(self) -> None:
        hidden_context = {
            CONTROLLER_STATE_KEY: {
                "wls_completed": True,
                "pending_verification_stage": "post_parameter_correction",
            }
        }
        gt_verdict = {
            "verdict": {
                "has_error": True,
                "error_family": "parameter_error",
                "error_families": ["parameter_error"],
            }
        }
        predicted = {
            "verdict": {
                "has_error": True,
                "error_family": "parameter_error",
                "error_families": ["parameter_error"],
            },
            "action": {"diagnosis_status": "sequence_only"},
        }

        self.assertEqual(
            pending_verification_error(
                hidden_context,
                gt_verdict=gt_verdict,
                candidate_verdict=predicted,
            ),
            "Verdict before required get_verification_snapshot for post_parameter_correction.",
        )

    def test_missing_context_error_includes_allowed_next_tools(self) -> None:
        runtime_context = {
            "tool_context": {
                "parameter_context": {"case_path": "case14::parameter_channel::abc"},
            }
        }
        hidden_context = {
            CONTROLLER_STATE_KEY: {
                "wls_completed": True,
                "successful_tools": ["wls_from_path"],
            }
        }

        result = execute_tool(
            "get_harmonic_context",
            {"case_path": "case14"},
            runtime_context=runtime_context,
            hidden_context=hidden_context,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["available_context_tools"], ["get_parameter_context"])
        self.assertIn("get_parameter_context", result["allowed_next_tools"])
        self.assertIn("correct_measurements_from_path", result["allowed_next_tools"])
        self.assertIn("final_json", result["allowed_next_tools"])
        self.assertNotIn("get_harmonic_context", result["allowed_next_tools"])


if __name__ == "__main__":
    unittest.main()
