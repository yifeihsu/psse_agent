from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from psse_env.sft.cli import main as cli_main
from psse_env.sft.gates import GateError, prepare_example
from psse_env.sft.smoke import (
    TARGETED_RECOVERY_CASES,
    TARGETED_TINY_OVERFIT_LEARNING_RATES,
    TargetedRecoveryExample,
    run_targeted_recovery_smoke,
    select_targeted_recovery_slice,
)
from psse_env.sft.training import (
    TrainerSettings,
    run_targeted_lora_smoke_sweep,
)
import psse_env.sft.training as training_module


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "submit_dagger_sft_round0.sh"


class _Processor:
    pad_token_id = 0

    def apply_chat_template(
        self, messages, *, tools, tokenize, add_generation_prompt
    ):
        assert tokenize is False
        pieces = ["<tools>", json.dumps(tools, sort_keys=True), "</tools>"]
        for message in messages:
            role = message["role"]
            calls = message.get("tool_calls")
            if role == "assistant" and calls:
                function = calls[0]["function"]
                pieces.extend(
                    [
                        "<assistant>",
                        f"<|tool_call|>call:{function['name']}",
                        json.dumps(function["arguments"], sort_keys=True),
                        "<|end_tool_call|></assistant>",
                    ]
                )
            else:
                pieces.extend(
                    [f"<{role}>", str(message.get("content", "")), f"</{role}>"]
                )
        if add_generation_prompt:
            pieces.append("<assistant>")
        return "".join(pieces)

    def __call__(self, text=None, **_kwargs):
        if text is None:
            raise TypeError("text is required")
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
        }

    def decode(self, ids, **_kwargs):
        return "".join(chr(int(value)) for value in ids)


def _tool_schema(name: str, arguments: dict[str, str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test schema for {name}.",
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "string"} for key in arguments
                },
                "required": list(arguments),
                "additionalProperties": False,
            },
        },
    }


def _case_row(case: str, *, example_id: str | None = None) -> dict:
    case_index = TARGETED_RECOVERY_CASES.index(case)
    state_id = f"s{case_index}"
    state: dict = {
        "active_state_id": state_id,
        "case_marker": case,
        "has_open_candidate": False,
        "candidate_state_id": None,
    }
    scenario_family = "measurement"
    target_tool = "run_wls"
    target_arguments = {"state_id": state_id}
    if case == "parameter_route_without_scans":
        stratum = "unsupported_correction_recovery"
        state.update(
            {
                "last_tool": "correct_parameters_from_path",
                "last_tool_output": {
                    "execution_status": "failure",
                    "error_code": "correction_route_not_actionable",
                },
            }
        )
        target_tool = "get_measurement_context"
    elif case == "measurement_parameter_sequence":
        stratum = "sequential_measurement_parameter_recovery"
        scenario_family = "measurement+parameter"
        target_tool = "get_parameter_context"
    elif case == "failed_correction_recovery":
        stratum = "post_failure_no_candidate"
        state.update(
            {
                "last_tool": "correct_measurements_from_path",
                "last_tool_output": {
                    "execution_status": "failure",
                    "error_code": "dispatch_error",
                },
            }
        )
        target_tool = "get_measurement_context"
    elif case == "premature_commit_recovery":
        stratum = "premature_commit_recovery"
        state.update(
            {
                "last_tool": "commit_state",
                "last_tool_output": {
                    "execution_status": "failure",
                    "error_code": "candidate_lifecycle_violation",
                },
            }
        )
    elif case == "valid_safe_escalation":
        stratum = "multi_measurement_safe_handoff"
        scenario_family = "multi_measurement"
        target_tool = "ask_for_more_evidence"
        target_arguments = {
            "state_id": state_id,
            "request": "operator_escalation:recovery_options_exhausted",
        }
    else:  # pragma: no cover - protects the fixture contract.
        raise AssertionError(case)
    return {
        "dataset_mode": "production",
        "example_id": example_id or f"targeted-{case}",
        "root_scenario_id": f"root-{case}",
        "physical_root_fingerprint": f"physical_v3_{case}",
        "production_label_eligible": True,
        "scenario_family": scenario_family,
        "tools": [_tool_schema(target_tool, target_arguments)],
        "messages": [
            {"role": "system", "content": "Use one canonical tool."},
            {"role": "user", "content": json.dumps({"state": state})},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": target_tool,
                            "arguments": target_arguments,
                        },
                    }
                ],
            },
        ],
        "metadata": {
            "dataset_mode": "production",
            "protocol": "canonical",
            "scenario_family": scenario_family,
            "labels": {"recovery_stratum": stratum},
        },
    }


def _prepared_slice():
    processor = _Processor()
    rows = [_case_row(case) for case in TARGETED_RECOVERY_CASES]
    examples = [
        prepare_example(row, processor, max_length=20_000)
        for row in rows
    ]
    return processor, rows, examples, select_targeted_recovery_slice(rows, examples)


class TargetedTinyOverfitTests(unittest.TestCase):
    def test_slice_selects_five_distinct_cases_deterministically(self) -> None:
        processor = _Processor()
        rows = [_case_row(case) for case in reversed(TARGETED_RECOVERY_CASES)]
        rows.append(
            _case_row(
                "parameter_route_without_scans",
                example_id="zz-duplicate-parameter-case",
            )
        )
        examples = [
            prepare_example(row, processor, max_length=20_000) for row in rows
        ]

        selected = select_targeted_recovery_slice(rows, examples)

        self.assertEqual(
            tuple(item.case for item in selected), TARGETED_RECOVERY_CASES
        )
        self.assertEqual(len({item.example_id for item in selected}), 5)
        self.assertEqual(
            selected[0].example_id,
            "targeted-parameter_route_without_scans",
        )

    def test_slice_fails_closed_when_one_required_case_is_missing(self) -> None:
        processor = _Processor()
        rows = [
            _case_row(case)
            for case in TARGETED_RECOVERY_CASES
            if case != "valid_safe_escalation"
        ]
        examples = [
            prepare_example(row, processor, max_length=20_000) for row in rows
        ]
        with self.assertRaisesRegex(GateError, "valid_safe_escalation"):
            select_targeted_recovery_slice(rows, examples)

    def test_targeted_gate_checks_masks_gradients_loss_and_all_generations(self) -> None:
        import torch

        processor, rows, _examples, selected = _prepared_slice()
        generated_suffixes = {}
        for row in rows:
            state = json.loads(row["messages"][1]["content"])["state"]
            function = row["messages"][-1]["tool_calls"][0]["function"]
            generated_suffixes[state["case_marker"]] = (
                f"<|tool_call|>call:{function['name']}"
                + json.dumps(function["arguments"], sort_keys=True)
                + "<|end_tool_call|>"
            )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 24)
                self.projection = torch.nn.Linear(24, 256)

            def forward(self, input_ids, attention_mask=None, labels=None, **_kwargs):
                logits = self.projection(self.embedding(input_ids))
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                return SimpleNamespace(loss=loss, logits=logits)

            def generate(self, input_ids, **_kwargs):
                prompt = "".join(chr(int(value)) for value in input_ids[0])
                matches = [
                    suffix
                    for marker, suffix in generated_suffixes.items()
                    if marker in prompt
                ]
                if len(matches) != 1:
                    raise AssertionError(f"unexpected prompt marker count: {len(matches)}")
                suffix = torch.tensor(
                    [[ord(character) for character in matches[0]]],
                    device=input_ids.device,
                )
                return torch.cat([input_ids, suffix], dim=1)

        torch.manual_seed(0)
        result = run_targeted_recovery_smoke(
            Model(),
            processor,
            selected,
            steps=20,
            learning_rate=0.05,
            minimum_relative_loss_reduction=0.01,
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.assistant_only_masks)
        self.assertTrue(result.gradients_finite)
        self.assertTrue(result.gradients_nonzero)
        self.assertGreater(result.minimum_gradient_norm, 0.0)
        self.assertGreaterEqual(result.relative_loss_reduction, 0.01)
        self.assertEqual(len(result.case_loss_checks), 5)
        self.assertEqual(len(result.generation_checks), 5)
        self.assertTrue(all(row["exact_match"] for row in result.generation_checks))
        self.assertEqual(
            {row["case"] for row in result.generation_checks},
            set(TARGETED_RECOVERY_CASES),
        )

    def test_targeted_gate_rejects_non_assistant_only_mask(self) -> None:
        processor, _rows, _examples, selected = _prepared_slice()
        broken = list(selected)
        prepared = broken[0].prepared
        broken[0] = TargetedRecoveryExample(
            case=broken[0].case,
            example_id=broken[0].example_id,
            prepared=replace(
                prepared,
                completion_mask=[1] * len(prepared.completion_mask),
            ),
        )
        with self.assertRaisesRegex(GateError, "Completion mask"):
            run_targeted_recovery_smoke(
                SimpleNamespace(parameters=lambda: iter(())),
                processor,
                broken,
            )

    def test_lr_sweep_runs_exact_rates_from_identical_settings(self) -> None:
        observed_rates: list[float] = []

        def fake_smoke(**kwargs):
            learning_rate = kwargs["settings"].learning_rate
            observed_rates.append(learning_rate)
            if learning_rate == 3e-4:
                raise GateError("diagnostic failure")
            return SimpleNamespace(
                to_dict=lambda: {
                    "passed": True,
                    "learning_rate": learning_rate,
                    "relative_loss_reduction": learning_rate * 100,
                }
            )

        settings = TrainerSettings(revision="a" * 40, learning_rate=3e-5)
        with (
            mock.patch.object(
                training_module,
                "run_lora_smoke",
                side_effect=fake_smoke,
            ),
            mock.patch.object(
                training_module,
                "load_jsonl",
                return_value=[_case_row(case) for case in TARGETED_RECOVERY_CASES],
            ),
        ):
            result = run_targeted_lora_smoke_sweep(
                train_file="train.jsonl",
                validation_file="validation.jsonl",
                settings=settings,
            )

        self.assertEqual(tuple(observed_rates), TARGETED_TINY_OVERFIT_LEARNING_RATES)
        self.assertEqual(settings.learning_rate, 3e-5)
        self.assertTrue(result.passed)
        self.assertEqual(result.successful_learning_rates, (1e-4, 1e-3))
        self.assertEqual(result.best_diagnostic_learning_rate, 1e-3)
        self.assertEqual(len(result.runs), 3)

    def test_cli_writes_targeted_sweep_report(self) -> None:
        sweep = SimpleNamespace(
            passed=True,
            to_dict=lambda: {
                "passed": True,
                "learning_rates": list(TARGETED_TINY_OVERFIT_LEARNING_RATES),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "targeted.json"
            with mock.patch(
                "psse_env.sft.cli.run_targeted_lora_smoke_sweep",
                return_value=sweep,
            ) as run_sweep:
                status = cli_main(
                    [
                        "smoke",
                        "--revision",
                        "a" * 40,
                        "--train",
                        "train.jsonl",
                        "--validation",
                        "validation.jsonl",
                        "--mode",
                        "tiny-overfit",
                        "--targeted-recovery-sweep",
                        "--report-output",
                        str(report),
                    ]
                )
            payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["mode"], "targeted-tiny-overfit-sweep")
        run_sweep.assert_called_once()

    def test_launcher_exposes_targeted_gate_without_changing_round1_lr(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        for contract in (
            "targeted-tiny-overfit",
            "--targeted-recovery-sweep",
            "TARGETED_TINY_OVERFIT_MIN_RELATIVE_LOSS_REDUCTION",
            "targeted_tiny_overfit_sweep.json",
            "ROUND1_LR=${ROUND1_LR:-0.00003}",
        ):
            self.assertIn(contract, launcher)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
