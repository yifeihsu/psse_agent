from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from psse_env.sft.cli import main as cli_main
from psse_env.sft.collator import AssistantOnlyCollator
from psse_env.sft.gates import (
    GateError,
    ParsedToolCall,
    audit_dataset,
    load_exact_processor,
    parse_tool_call,
    prepare_example,
    validate_grouped_pilot,
)
from psse_env.sft.smoke import run_generation_tool_call_smoke, run_training_smoke
from psse_env.sft.training import (
    LoraSettings,
    TrainerSettings,
    ensure_required_side_inputs,
    infer_required_side_input_names,
    resolve_language_lora_targets,
    trl_config_kwargs,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_wls",
            "description": "Run WLS.",
            "parameters": {
                "type": "object",
                "properties": {"state_id": {"type": "string"}},
                "required": ["state_id"],
            },
        },
    }
]


def row(group: str = "g0", state: str = "active") -> dict:
    return {
        "dataset_mode": "production",
        "example_id": f"{group}-{state}",
        "root_scenario_id": group,
        "tools": copy.deepcopy(TOOLS),
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": json.dumps({"state_id": state})},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "run_wls", "arguments": {"state_id": state}},
                    }
                ],
            },
        ],
        "metadata": {"dataset_mode": "production", "state_class": "diagnostic"},
    }


class FakeProcessor:
    pad_token_id = 0
    eos_token_id = 3

    def __init__(self) -> None:
        self.template_tools = []

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        assert tokenize is False
        self.template_tools.append(copy.deepcopy(tools))
        pieces = ["<tools>", json.dumps(tools, sort_keys=True), "</tools>"]
        for message in messages:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                function = message["tool_calls"][0]["function"]
                pieces.extend(
                    [
                        "<assistant>",
                        f"<|tool_call|>call:{function['name']}",
                        json.dumps(function["arguments"], sort_keys=True),
                        "<|end_tool_call|></assistant>",
                    ]
                )
            else:
                pieces.extend([f"<{role}>", str(message.get("content", "")), f"</{role}>"])
        if add_generation_prompt:
            pieces.append("<assistant>")
        return "".join(pieces)

    def __call__(self, text=None, **_kwargs):
        if text is None:
            raise TypeError("text required")
        return {"input_ids": [ord(char) for char in text], "attention_mask": [1] * len(text)}

    def decode(self, ids, **_kwargs):
        return "".join(chr(int(value)) for value in ids)


class FakeThinkingProcessor(FakeProcessor):
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        rendered = super().apply_chat_template(
            messages,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if add_generation_prompt:
            rendered = rendered[: -len("<assistant>")] + "<|turn>model\n<|channel>thought\n<channel|>"
        else:
            rendered = rendered.replace("<assistant>", "<|turn>model\n", 1)
        return rendered


class FakeGemma4Processor(FakeProcessor):
    model_input_names = ["input_ids", "attention_mask", "mm_token_type_ids"]

    def __init__(self) -> None:
        super().__init__()
        self.tokenize_kwargs = []

    def __call__(self, text=None, **kwargs):
        encoded = super().__call__(text=text, **kwargs)
        self.tokenize_kwargs.append(dict(kwargs))
        if kwargs.get("return_mm_token_type_ids"):
            encoded["mm_token_type_ids"] = [index % 3 for index in range(len(encoded["input_ids"]))]
        return encoded


class TestSchemaTemplateAndMasks(unittest.TestCase):
    def test_row_tools_dict_arguments_mask_and_round_trip(self) -> None:
        processor = FakeProcessor()
        example = prepare_example(row(), processor, max_length=10000)
        self.assertEqual(processor.template_tools, [TOOLS, TOOLS])
        first_supervised = example.labels.index(next(label for label in example.labels if label != -100))
        self.assertTrue(all(label == -100 for label in example.labels[:first_supervised]))
        self.assertEqual(example.labels[first_supervised:], example.input_ids[first_supervised:])
        self.assertEqual(example.expected_tool_call, ParsedToolCall("run_wls", {"state_id": "active"}))
        self.assertFalse(example.prompt_truncated)
        self.assertFalse(example.target_truncated)

    def test_processor_mm_token_type_ids_are_requested_preserved_and_sliced(self) -> None:
        source = row()
        source["messages"][1]["content"] = "x" * 500
        processor = FakeGemma4Processor()
        full = prepare_example(source, processor, max_length=10000)
        limit = full.supervised_tokens + 25
        truncated = prepare_example(source, processor, max_length=limit)

        self.assertTrue(
            all(kwargs.get("return_mm_token_type_ids") is True for kwargs in processor.tokenize_kwargs)
        )
        self.assertTrue(truncated.prompt_truncated)
        self.assertEqual(
            truncated.side_inputs["mm_token_type_ids"],
            full.side_inputs["mm_token_type_ids"][-truncated.used_length :],
        )

    def test_string_arguments_are_a_hard_failure(self) -> None:
        bad = row()
        bad["messages"][-1]["tool_calls"][0]["function"]["arguments"] = '{"state_id":"active"}'
        report = audit_dataset([bad], FakeProcessor(), max_length=10000)
        self.assertFalse(report.passed)
        self.assertIn("must be a dictionary", report.failures[0])

    def test_missing_row_tools_is_a_hard_failure(self) -> None:
        bad = row()
        del bad["tools"]
        report = audit_dataset([bad], FakeProcessor(), max_length=10000)
        self.assertFalse(report.passed)
        self.assertIn("row-level tools", report.failures[0])

    def test_target_arguments_must_conform_to_row_schema(self) -> None:
        bad = row()
        bad["messages"][-1]["tool_calls"][0]["function"]["arguments"] = {}
        report = audit_dataset([bad], FakeProcessor(), max_length=10000)
        self.assertFalse(report.passed)
        self.assertIn("missing required arguments", report.failures[0])

    def test_empty_assistant_target_counts_as_zero_supervision(self) -> None:
        bad = row()
        del bad["messages"][-1]["tool_calls"]
        report = audit_dataset([bad], FakeProcessor(), max_length=10000)
        self.assertFalse(report.passed)
        self.assertEqual(report.length_audit.zero_supervision_rows, 1)

    def test_prompt_truncation_preserves_target_but_requires_approval(self) -> None:
        source = row()
        source["messages"][1]["content"] = "x" * 500
        unrestricted = prepare_example(source, FakeProcessor(), max_length=10000)
        limit = unrestricted.supervised_tokens + 25
        example = prepare_example(source, FakeProcessor(), max_length=limit)
        self.assertTrue(example.prompt_truncated)
        self.assertEqual(example.supervised_tokens, unrestricted.supervised_tokens)
        self.assertEqual(example.input_ids[-example.supervised_tokens :], unrestricted.input_ids[-unrestricted.supervised_tokens :])
        rejected = audit_dataset([source], FakeProcessor(), max_length=limit)
        approved = audit_dataset([source], FakeProcessor(), max_length=limit, allow_prompt_truncation=True)
        self.assertFalse(rejected.passed)
        self.assertTrue(approved.passed)
        self.assertEqual(approved.length_audit.prompt_truncated_rows, 1)

    def test_target_truncation_and_length_percentiles_are_reported(self) -> None:
        source = row()
        full = prepare_example(source, FakeProcessor(), max_length=10000)
        report = audit_dataset([source], FakeProcessor(), max_length=full.supervised_tokens - 1)
        self.assertFalse(report.passed)
        self.assertEqual(report.length_audit.target_truncated_rows, 1)
        self.assertEqual(report.length_audit.p50, full.original_length)
        self.assertEqual(report.length_audit.p95, full.original_length)
        self.assertEqual(report.length_audit.p99, full.original_length)
        self.assertEqual(report.length_audit.maximum, full.original_length)

    def test_parse_supported_native_formats(self) -> None:
        expected = ParsedToolCall("run_wls", {"state_id": "active"})
        self.assertEqual(parse_tool_call('<|tool_call|>call:run_wls{"state_id":"active"}'), expected)
        self.assertEqual(parse_tool_call('<tool_call>{"name":"run_wls","arguments":{"state_id":"active"}}</tool_call>'), expected)
        self.assertEqual(
            parse_tool_call('<|tool_call>call:run_wls{state_id:<|"|>active<|"|>}<tool_call|>'),
            expected,
        )
        self.assertEqual(
            parse_tool_call(
                '<|tool_call>call:correct_measurements{measurement_updates:{0:1.0},'
                'state_id:<|"|>active<|"|>}<tool_call|>'
            ),
            ParsedToolCall(
                "correct_measurements",
                {"measurement_updates": {"0": 1.0}, "state_id": "active"},
            ),
        )
        self.assertEqual(
            parse_tool_call(
                '<|tool_call>call:ask_for_more_evidence{state_id:<|"|>active<|"|>,'
                'request:<|"|>{foo:bar}<|"|>}<tool_call|>'
            ),
            ParsedToolCall(
                "ask_for_more_evidence",
                {"state_id": "active", "request": "{foo:bar}"},
            ),
        )
        with self.assertRaises(GateError):
            parse_tool_call("not a tool call")

    def test_empty_thought_channel_is_aligned_for_gemma4(self) -> None:
        example = prepare_example(row(), FakeThinkingProcessor(), max_length=10000)
        self.assertTrue(example.empty_thought_injected)
        self.assertTrue(example.rendered_text.startswith(example.rendered_prompt))


class TestGroupedPilot(unittest.TestCase):
    def test_disjoint_groups_and_distributions(self) -> None:
        report = validate_grouped_pilot(
            {"train": [row("train0"), row("train1")], "validation": [row("valid0")]},
            minimum_rows=3,
            maximum_rows=3,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.action_distribution["train"], {"run_wls": 2})
        self.assertEqual(report.class_distribution["validation"], {"diagnostic": 1})

    def test_group_overlap_fails(self) -> None:
        report = validate_grouped_pilot(
            {"train": [row("same")], "validation": [row("same")]},
            minimum_rows=2,
            maximum_rows=2,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.overlapping_groups, {"same": ("train", "validation")})

    def test_nonproduction_row_fails_closed(self) -> None:
        untagged = row("train0")
        del untagged["dataset_mode"]
        report = validate_grouped_pilot(
            {"train": [untagged], "validation": [row("valid0")]},
            minimum_rows=2,
            maximum_rows=2,
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any("not tagged as a production dataset row" in item for item in report.failures)
        )


class TestExactLoader(unittest.TestCase):
    def test_requires_gemma4_and_revision(self) -> None:
        with self.assertRaisesRegex(GateError, "Gemma 4"):
            load_exact_processor("other/model", "abc")
        with self.assertRaisesRegex(GateError, "pinned"):
            load_exact_processor("unsloth/gemma-4-31B-it", "")

    def test_auto_processor_then_tokenizer_fallback(self) -> None:
        class Broken:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise OSError("not cached")

        class Working:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return FakeProcessor()

        processor, loader = load_exact_processor(
            "unsloth/gemma-4-31B-it",
            "a" * 40,
            auto_processor_cls=Broken,
            auto_tokenizer_cls=Working,
        )
        self.assertIsInstance(processor, FakeProcessor)
        self.assertEqual(loader, "AutoTokenizer")

    def test_unavailable_is_no_go_not_skip(self) -> None:
        class Broken:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise OSError("unavailable")

        with self.assertRaisesRegex(GateError, "NO-GO"):
            load_exact_processor(
                "unsloth/gemma-4-31B-it",
                "a" * 40,
                auto_processor_cls=Broken,
                auto_tokenizer_cls=Broken,
            )

    def test_cli_returns_nonzero_when_live_processor_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train = Path(temp_dir) / "train.jsonl"
            valid = Path(temp_dir) / "valid.jsonl"
            payload = json.dumps(row()) + "\n"
            train.write_text(payload, encoding="utf-8")
            valid.write_text(payload, encoding="utf-8")
            with mock.patch("psse_env.sft.cli.load_exact_processor", side_effect=GateError("NO-GO unavailable")):
                result = cli_main(
                    [
                        "gate", "--model", "unsloth/gemma-4-31B-it", "--revision", "a" * 40,
                        "--train", str(train), "--validation", str(valid),
                        "--pilot-min-rows", "2", "--pilot-max-rows", "2",
                    ]
                )
            self.assertEqual(result, 2)


class TinyLM(unittest.TestCase):
    pass


class TestTrainingSmoke(unittest.TestCase):
    def test_forward_backward_and_tiny_overfit(self) -> None:
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 12)
                self.projection = torch.nn.Linear(12, 256)

            def forward(self, input_ids, attention_mask=None, labels=None, **_kwargs):
                logits = self.projection(self.embedding(input_ids))
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                )
                return SimpleNamespace(loss=loss, logits=logits)

        torch.manual_seed(0)
        processor = FakeProcessor()
        example = prepare_example(row(), processor, max_length=10000)
        one_batch = run_training_smoke(Model(), processor, [example], steps=1, learning_rate=0.01)
        self.assertTrue(one_batch.passed)
        torch.manual_seed(0)
        overfit = run_training_smoke(Model(), processor, [example], steps=12, learning_rate=0.05)
        self.assertTrue(overfit.loss_decreased)
        self.assertLess(overfit.final_loss, overfit.initial_loss)

    def test_gemma4_missing_side_inputs_are_filled_and_forwarded(self) -> None:
        import torch

        class StrictGemma4Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 12)
                self.projection = torch.nn.Linear(12, 256)
                self.seen_mm_token_type_ids = None

            def forward(
                self,
                input_ids,
                attention_mask=None,
                labels=None,
                mm_token_type_ids=None,
            ):
                if mm_token_type_ids is None:
                    raise ValueError("`mm_token_type_ids` is required as a model input when training")
                self.seen_mm_token_type_ids = mm_token_type_ids.detach().clone()
                logits = self.projection(self.embedding(input_ids))
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                )
                return SimpleNamespace(loss=loss, logits=logits)

        torch.manual_seed(0)
        processor = FakeProcessor()
        model = StrictGemma4Model()
        original = prepare_example(row(), processor, max_length=10000)
        required = infer_required_side_input_names(model, processor, "unsloth/gemma-4-31B-it")
        prepared = ensure_required_side_inputs([original], required)

        self.assertNotIn("mm_token_type_ids", original.side_inputs)
        self.assertEqual(prepared[0].side_inputs["mm_token_type_ids"], [0] * len(original.input_ids))
        batch = AssistantOnlyCollator(processor)(prepared)
        self.assertEqual(batch["mm_token_type_ids"].shape, batch["input_ids"].shape)

        one_batch = run_training_smoke(model, processor, prepared, steps=1, learning_rate=0.01)
        self.assertTrue(one_batch.passed)
        self.assertIsNotNone(model.seen_mm_token_type_ids)
        self.assertTrue(
            bool(torch.equal(model.seen_mm_token_type_ids, torch.zeros_like(model.seen_mm_token_type_ids)))
        )

    def test_collator_pads_mm_token_type_ids_with_zero(self) -> None:
        import torch

        collator = AssistantOnlyCollator(SimpleNamespace(pad_token_id=99))
        batch = collator(
            [
                {
                    "input_ids": [11, 12, 13],
                    "attention_mask": [1, 1, 1],
                    "labels": [-100, 12, 13],
                    "mm_token_type_ids": [1, 2, 3],
                },
                {
                    "input_ids": [21],
                    "attention_mask": [1],
                    "labels": [21],
                    "mm_token_type_ids": [7],
                },
            ]
        )
        self.assertEqual(batch["input_ids"].tolist(), [[11, 12, 13], [21, 99, 99]])
        self.assertEqual(batch["mm_token_type_ids"].tolist(), [[1, 2, 3], [7, 0, 0]])
        self.assertEqual(batch["mm_token_type_ids"].dtype, torch.long)

    def test_pure_lora_and_trl_settings(self) -> None:
        lora = LoraSettings()
        self.assertEqual(lora.kwargs()["task_type"], "CAUSAL_LM")
        self.assertIn("q_proj", lora.kwargs()["target_modules"])
        settings = TrainerSettings(revision="a" * 40)
        settings.validate()
        kwargs = trl_config_kwargs(settings, has_validation=True)
        self.assertFalse(kwargs["completion_only_loss"])
        self.assertEqual(kwargs["dataset_kwargs"], {"skip_prepare_dataset": True})

    def test_generated_tool_call_round_trip(self) -> None:
        import torch

        processor = FakeProcessor()

        class GeneratingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.seen_mm_token_type_ids = None

            def generate(self, input_ids, mm_token_type_ids=None, **_kwargs):
                if mm_token_type_ids is None:
                    raise AssertionError("generation is missing mm_token_type_ids")
                self.seen_mm_token_type_ids = mm_token_type_ids.detach().clone()
                target = '<|tool_call|>call:run_wls{"state_id":"active"}'
                suffix = torch.tensor([[ord(char) for char in target]], device=input_ids.device)
                return torch.cat([input_ids, suffix], dim=1)

        model = GeneratingModel()
        original = prepare_example(row(), processor, max_length=10000)
        required = infer_required_side_input_names(model, processor, "unsloth/gemma-4-31B-it")
        example = ensure_required_side_inputs([original], required)[0]
        parsed = run_generation_tool_call_smoke(model, processor, example)
        self.assertEqual(parsed, ParsedToolCall("run_wls", {"state_id": "active"}))
        self.assertIsNotNone(model.seen_mm_token_type_ids)

    def test_lora_targets_are_language_tower_only(self) -> None:
        class Model:
            def named_modules(self):
                return iter(
                    [
                        ("model.language_model.layers.0.self_attn.q_proj", object()),
                        ("model.vision_tower.layers.0.self_attn.q_proj", object()),
                        ("model.audio_tower.layers.0.mlp.down_proj", object()),
                        ("model.language_model.layers.0.mlp.down_proj", object()),
                    ]
                )

        self.assertEqual(
            resolve_language_lora_targets(Model()),
            (
                "model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.mlp.down_proj",
            ),
        )


if __name__ == "__main__":
    unittest.main()
