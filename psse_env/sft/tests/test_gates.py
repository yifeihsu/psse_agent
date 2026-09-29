from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from psse_env.dagger.dataset_builder import TOOL_JSON_SCHEMAS
from psse_env.sft.collator import AssistantOnlyCollator
from psse_env.sft.gates import (
    GateError,
    ParsedToolCall,
    _check_schema_node,
    _validate_json_instance,
    audit_dataset,
    load_exact_processor,
    load_jsonl,
    parse_tool_call,
    prepare_example,
)
from psse_env.sft.smoke import (
    generate_single_tool_call,
    run_training_smoke,
)
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
        "physical_root_fingerprint": f"physical_v1_{group}",
        "production_label_eligible": True,
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
        "metadata": {
            "dataset_mode": "production",
            "state_class": "diagnostic",
            "protocol": "controller",
        },
    }


def recovery_probe_row(
    group: str = "probe0", *, provenance_id: str = "f" * 64
) -> dict:
    candidate = row(group)
    identity = {
        "dataset_mode": "production",
        "dataset_source": "observable_recovery_probe",
        "collector_contract": "dagger1_observable_recovery_probe_v1",
        "state_origin": "observable_recovery_probe",
        "collection_role": "auxiliary_training",
        "state_visited_by": "observable_recovery_probe",
        "replay_source": "observable_recovery_probe",
        "auxiliary_training_eligible": True,
        "production_label_eligible": False,
        "natural_on_policy_support_eligible": False,
        "training_decision_evidence_verified": True,
        "generation_provenance_id": provenance_id,
        "recovery_stratum": "post_failure_no_candidate",
    }
    candidate.update(identity)
    candidate["metadata"].update(identity)
    return candidate


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
    def test_omitted_additional_properties_rejects_extra_assistant_argument(
        self,
    ) -> None:
        candidate = row()
        parameters = candidate["tools"][0]["function"]["parameters"]
        self.assertNotIn("additionalProperties", parameters)
        candidate["messages"][-1]["tool_calls"][0]["function"]["arguments"][
            "line_index"
        ] = 3

        with self.assertRaisesRegex(
            GateError,
            "contains unsupported argument 'line_index'",
        ):
            prepare_example(candidate, FakeProcessor(), max_length=10000)

    def test_explicit_additional_properties_true_allows_open_arguments(
        self,
    ) -> None:
        schema = copy.deepcopy(TOOLS[0]["function"]["parameters"])
        schema["additionalProperties"] = True
        _validate_json_instance(
            {"state_id": "active", "extension": {"value": 1}},
            schema,
            path="arguments",
        )

    def test_json_schema_numeric_bounds_are_enforced(self) -> None:
        schema = {"type": "integer", "minimum": 2, "maximum": 5}
        _check_schema_node(schema, path="value")
        _validate_json_instance(2, schema, path="value")
        _validate_json_instance(5, schema, path="value")
        with self.assertRaisesRegex(GateError, "value must be >= 2"):
            _validate_json_instance(1, schema, path="value")
        with self.assertRaisesRegex(GateError, "value must be <= 5"):
            _validate_json_instance(6, schema, path="value")

    def test_json_schema_rejects_malformed_numeric_bounds(self) -> None:
        invalid = (
            (
                {"type": "integer", "minimum": "2"},
                "minimum must be a finite JSON number",
            ),
            (
                {"type": "number", "maximum": float("nan")},
                "maximum must be a finite JSON number",
            ),
            (
                {"type": "integer", "minimum": 6, "maximum": 5},
                "minimum must not exceed value.maximum",
            ),
            (
                {"type": "string", "minimum": 2},
                "minimum requires an integer or number schema type",
            ),
        )
        for schema, message in invalid:
            with self.subTest(schema=schema):
                with self.assertRaisesRegex(GateError, message):
                    _check_schema_node(schema, path="value")

    def test_release_gate_rejects_stale_partial_tool_registry(self) -> None:
        stale = row()
        stale["metadata"]["protocol"] = "controller"
        stale["tools"] = copy.deepcopy(TOOL_JSON_SCHEMAS[:-1])
        report = audit_dataset(
            [stale],
            FakeProcessor(),
            max_length=100000,
            require_current_registry=True,
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any("does not match current controller registry" in item for item in report.failures)
        )

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
        with self.assertRaisesRegex(GateError, "multiple found"):
            parse_tool_call(
                '<|tool_call|>call:run_wls{"state_id":"active"}'
                '<|tool_call|>call:run_wls{"state_id":"candidate"}'
            )
        self.assertEqual(
            parse_tool_call(
                '<|tool_call|>call:ask_for_more_evidence{'
                '"case_path":"active",'
                '"request":"inspect call:operator details"}'
            ),
            ParsedToolCall(
                "ask_for_more_evidence",
                {
                    "case_path": "active",
                    "request": "inspect call:operator details",
                },
            ),
        )
        repeated_json = (
            '{"name":"run_wls","arguments":{"state_id":"active"}}'
            '{"name":"run_wls","arguments":{"state_id":"active"}}'
        )
        with self.assertRaisesRegex(GateError, "multiple found"):
            parse_tool_call(repeated_json)

    def test_empty_thought_channel_is_aligned_for_gemma4(self) -> None:
        example = prepare_example(row(), FakeThinkingProcessor(), max_length=10000)
        self.assertTrue(example.empty_thought_injected)
        self.assertTrue(example.rendered_text.startswith(example.rendered_prompt))


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


class TestLoadJsonl(unittest.TestCase):
    def test_rows_load_in_order_and_blank_lines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
            self.assertEqual(load_jsonl(path), [{"a": 1}, {"b": 2}])

    def test_missing_invalid_non_object_and_empty_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            with self.assertRaisesRegex(GateError, "Unable to open"):
                load_jsonl(path)
            for content, message in (
                ("{not json}\n", "Invalid JSON"),
                ("[1, 2]\n", "Expected a JSON object"),
                ("\n\n", "empty"),
            ):
                with self.subTest(message=message):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(GateError, message):
                        load_jsonl(path)


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
        self.assertEqual(kwargs["eval_strategy"], "epoch")
        self.assertEqual(kwargs["save_strategy"], "epoch")
        self.assertIsNone(kwargs["eval_steps"])

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
        parsed = generate_single_tool_call(model, processor, example)
        self.assertEqual(parsed, ParsedToolCall("run_wls", {"state_id": "active"}))
        self.assertEqual(parsed, example.expected_tool_call)
        self.assertIsNotNone(model.seen_mm_token_type_ids)

        class AlternateArgumentsModel(GeneratingModel):
            def generate(self, input_ids, mm_token_type_ids=None, **_kwargs):
                target = '<|tool_call|>call:run_wls{"state_id":"candidate"}'
                suffix = torch.tensor(
                    [[ord(char) for char in target]], device=input_ids.device
                )
                return torch.cat([input_ids, suffix], dim=1)

        alternate = AlternateArgumentsModel()
        parsed = generate_single_tool_call(
            alternate,
            processor,
            example,
        )
        self.assertEqual(parsed, ParsedToolCall("run_wls", {"state_id": "candidate"}))
        self.assertNotEqual(parsed, example.expected_tool_call)

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
