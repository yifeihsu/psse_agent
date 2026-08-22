import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gpt_oss_power_sft_revised_v3 as sft_script

from gpt_oss_power_sft_revised_v3 import (
    assistant_tool_name,
    assistant_turn_trainable,
    audit_hardening_masks,
    audit_hardening_recovery_targets,
    explode_conversation,
    is_false_marker,
    make_sft_config_kwargs,
    normalize_messages,
    parse_args,
    records_to_dataset,
    validate_resume_policy,
)


def tool_call_message(name: str, args: dict, *, trainable=True, call_id: str | None = None) -> dict:
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id or f"call_{name}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
        ],
    }
    if trainable is not True:
        message["trainable"] = trainable
        message["train_on_assistant"] = trainable
        message["loss_mask"] = trainable
    return message


def tool_response_message(name: str, payload: dict, *, call_id: str | None = None) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id or f"call_{name}",
        "name": name,
        "content": json.dumps(payload),
    }


def tool_args_for_name(name: str) -> dict:
    if name == "get_verification_snapshot":
        return {"stage": "post_measurement_correction"}
    return {"case_path": "case14::hardening::001", "line_index": 3}


def hardening_row(
    false_marker="false",
    *,
    invalid_tool_name: str = "correct_parameters_from_path",
    recovery_tool_name: str = "correct_measurements_from_path",
) -> dict:
    return {
        "id": f"hardening_{invalid_tool_name}_recovery_001",
        "trace_type": "hardening_recovery",
        "trace_metadata": {"trace_kind": "tool_precondition_hardening"},
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": {"case_path": "case14::hardening::001"}},
            tool_call_message("wls_from_path", {"case_path": "case14::hardening::001"}),
            tool_response_message(
                "wls_from_path",
                {"success": True, "global_residual_ratio": 12.0},
            ),
            tool_call_message(
                invalid_tool_name,
                tool_args_for_name(invalid_tool_name),
                trainable=false_marker,
            ),
            tool_response_message(
                invalid_tool_name,
                {
                    "success": False,
                    "tool_error_type": "controller_precondition",
                    "message": "Missing runtime context; call get_parameter_context first.",
                    "allowed_tools": ["get_parameter_context", "correct_measurements_from_path"],
                },
            ),
            tool_call_message(
                recovery_tool_name,
                {"case_path": "case14::hardening::001", "suspect_indices": [48]},
            ),
        ],
    }


def final_message() -> dict:
    return {
        "role": "assistant",
        "content": json.dumps(
            {
                "verdict": {"has_error": True, "error_family": "measurement_error"},
                "evidence": {},
                "suspect_location": {"domain": "measurement", "details": {}},
                "action": {"applied_tool": "correct_measurements_from_path"},
                "summary": "Recovered after a helper precondition failure.",
            }
        ),
    }


class FakeDataset:
    def __init__(self, columns: dict):
        self.columns = columns
        self.column_names = list(columns.keys())

    @classmethod
    def from_dict(cls, columns: dict):
        return cls(columns)

    def __len__(self):
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def __getitem__(self, idx: int):
        return {key: value[idx] for key, value in self.columns.items()}


class SFTHardeningMaskTests(unittest.TestCase):
    def test_loss_only_evaluation_option_is_forwarded_when_supported(self) -> None:
        original = sft_script.SFTConfig

        class FakeSFTConfig:
            def __init__(self, prediction_loss_only=False):
                del prediction_loss_only

        try:
            sft_script.SFTConfig = FakeSFTConfig
            self.assertEqual(
                make_sft_config_kwargs(prediction_loss_only=True),
                {"prediction_loss_only": True},
            )
        finally:
            sft_script.SFTConfig = original

    def test_resume_default_is_fresh_and_init_adapter_auto_resume_is_guarded(self) -> None:
        self.assertEqual(parse_args([]).resume_from_checkpoint, "")
        self.assertFalse(parse_args([]).fail_on_prompt_truncation)
        self.assertTrue(
            parse_args(["--fail-on-prompt-truncation"]).fail_on_prompt_truncation
        )

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "checkpoint-25").mkdir()
            args = parse_args(
                [
                    "--init-adapter",
                    "/tmp/example-adapter",
                    "--resume-from-checkpoint",
                    "auto",
                    "--output-dir",
                    tmp,
                ]
            )
            with self.assertRaisesRegex(ValueError, "existing Trainer checkpoint"):
                validate_resume_policy(args)

    def test_false_marker_parser_handles_string_and_numeric_values(self) -> None:
        for value in (False, 0, 0.0, "false", "False", " FALSE ", "0", " no ", "off"):
            with self.subTest(value=value):
                self.assertTrue(is_false_marker(value))

        for value in (None, True, 1, 1.0, "", "true", "yes", "mask"):
            with self.subTest(value=value):
                self.assertFalse(is_false_marker(value))

        self.assertFalse(assistant_turn_trainable({"role": "assistant", "trainable": "false"}))
        self.assertFalse(assistant_turn_trainable({"role": "assistant", "loss_mask": 0}))
        self.assertFalse(assistant_turn_trainable({"role": "assistant", "train_on_assistant": "0"}))
        self.assertTrue(assistant_turn_trainable({"role": "assistant"}))

    def test_hardening_audit_requires_a_masked_assistant_turn(self) -> None:
        audit_hardening_masks([hardening_row(false_marker="0")], "train")

        unmasked = hardening_row(false_marker=True)
        for key in ("trainable", "train_on_assistant", "loss_mask"):
            unmasked["messages"][4].pop(key, None)
        with self.assertRaisesRegex(ValueError, "zero masked assistant turns"):
            audit_hardening_masks([unmasked], "train")

    def test_hardening_row_requires_recovery_tool_target(self) -> None:
        row = hardening_row(false_marker="false")
        row["messages"] = row["messages"][:-1]

        with self.assertRaisesRegex(ValueError, "no trainable recovery tool target"):
            audit_hardening_recovery_targets([row], "train")

    def test_hardening_row_final_answer_is_not_recovery_tool_target(self) -> None:
        row = hardening_row(false_marker="false")
        row["messages"][-1] = final_message()

        with self.assertRaisesRegex(ValueError, "no trainable recovery tool target"):
            audit_hardening_recovery_targets([row], "train")

    def test_helper_style_hardening_failures_pass_with_recovery_target(self) -> None:
        for helper_name in ("get_harmonic_context", "get_verification_snapshot", "get_parameter_context"):
            with self.subTest(helper_name=helper_name):
                audit_hardening_masks([hardening_row(invalid_tool_name=helper_name)], "train")
                audit_hardening_recovery_targets([hardening_row(invalid_tool_name=helper_name)], "train")

    def test_masked_invalid_helper_call_is_prefix_but_not_target(self) -> None:
        row = hardening_row(false_marker="false")
        normalized = normalize_messages(row["messages"], preserve_system_text=True)
        expanded = explode_conversation(normalized)

        target_tool_names = [assistant_tool_name(sample[-1]) for sample in expanded]
        self.assertIn("wls_from_path", target_tool_names)
        self.assertIn("correct_measurements_from_path", target_tool_names)
        self.assertNotIn("correct_parameters_from_path", target_tool_names)

        recovery_sample = next(
            sample
            for sample in expanded
            if assistant_tool_name(sample[-1]) == "correct_measurements_from_path"
        )
        prefix_tool_names = [
            assistant_tool_name(message)
            for message in recovery_sample[:-1]
            if message.get("role") == "assistant"
        ]
        self.assertIn("correct_parameters_from_path", prefix_tool_names)

        failed_tool_outputs = [
            message
            for message in recovery_sample[:-1]
            if message.get("role") == "tool" and message.get("name") == "correct_parameters_from_path"
        ]
        self.assertEqual(len(failed_tool_outputs), 1)
        self.assertIn("controller_precondition", json.dumps(failed_tool_outputs[0].get("content")))
        self.assertIn("Missing runtime context", json.dumps(failed_tool_outputs[0].get("content")))

    def test_records_to_dataset_drops_debug_text_by_default(self) -> None:
        old_module = sys.modules.get("datasets")
        sys.modules["datasets"] = types.SimpleNamespace(Dataset=FakeDataset)
        try:
            records = [
                {
                    "input_ids": [1, 2],
                    "attention_mask": [1, 1],
                    "labels": [-100, 2],
                    "completion_mask": [0, 1],
                    "text": "rendered debug text",
                    "_tools_for_sanity": [],
                }
            ]

            compact = records_to_dataset(records)
            self.assertNotIn("text", compact.column_names)
            self.assertNotIn("_tools_for_sanity", compact.column_names)

            debug = records_to_dataset(records, keep_debug_text=True)
            self.assertIn("text", debug.column_names)
        finally:
            if old_module is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = old_module


if __name__ == "__main__":
    unittest.main()
