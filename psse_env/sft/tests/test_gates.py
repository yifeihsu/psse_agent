from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from psse_env.dagger.dataset_builder import TOOL_JSON_SCHEMAS
from psse_env.sft.cli import _baseline_evaluation_gate, main as cli_main
from psse_env.sft.collator import AssistantOnlyCollator
from psse_env.sft.gates import (
    GateError,
    ParsedToolCall,
    _check_schema_node,
    _validate_json_instance,
    audit_dataset,
    load_exact_processor,
    parse_tool_call,
    prepare_example,
    validate_grouped_pilot,
)
from psse_env.sft.smoke import (
    generate_single_tool_call,
    run_generation_tool_call_smoke,
    run_training_smoke,
)
from psse_env.sft.provenance import (
    ROUND1_AGGREGATE_BUILDER_CONTRACT,
    file_sha256,
    stable_json_sha256,
    validate_generation_provenance,
    validate_release_gate_report,
)
import psse_env.sft.training as training_module
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

    def test_explicitly_ineligible_auxiliary_row_fails_closed(self) -> None:
        auxiliary = row("aux0")
        auxiliary["production_label_eligible"] = False
        report = validate_grouped_pilot(
            {"train": [auxiliary], "validation": [row("valid0")]},
            minimum_rows=2,
            maximum_rows=2,
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any("not explicitly production-label eligible" in item for item in report.failures)
        )

    def test_exact_probe_shape_requires_and_accepts_validated_round1_id(self) -> None:
        provenance_id = "f" * 64
        probe = recovery_probe_row(provenance_id=provenance_id)

        without_binding = validate_grouped_pilot(
            {"train": [probe], "validation": [row("valid0")]},
            minimum_rows=2,
            maximum_rows=2,
        )
        with_binding = validate_grouped_pilot(
            {"train": [probe], "validation": [row("valid0")]},
            minimum_rows=2,
            maximum_rows=2,
            validated_round1_generation_provenance_id=provenance_id,
        )

        self.assertFalse(without_binding.passed)
        self.assertTrue(with_binding.passed, with_binding.failures)

    def test_validated_id_does_not_admit_generic_or_tampered_auxiliary(self) -> None:
        provenance_id = "f" * 64
        generic = row("generic")
        generic["production_label_eligible"] = False
        generic["metadata"]["production_label_eligible"] = False
        tampered = recovery_probe_row("tampered", provenance_id=provenance_id)
        tampered["metadata"]["replay_source"] = "natural_d1"
        wrong_id = recovery_probe_row("wrong-id", provenance_id="e" * 64)
        laundered = recovery_probe_row("laundered", provenance_id=provenance_id)
        laundered["production_label_eligible"] = True
        laundered["metadata"]["production_label_eligible"] = True

        for candidate, expected in (
            (generic, "collector_contract"),
            (tampered, "metadata.replay_source"),
            (wrong_id, "generation_provenance_id"),
            (laundered, "production_label_eligible"),
        ):
            with self.subTest(expected=expected):
                report = validate_grouped_pilot(
                    {"train": [candidate], "validation": [row("valid0")]},
                    minimum_rows=2,
                    maximum_rows=2,
                    validated_round1_generation_provenance_id=provenance_id,
                )
                self.assertFalse(report.passed)
                self.assertTrue(
                    any(expected in item for item in report.failures),
                    report.failures,
                )


class TestGenerationProvenance(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, dict[str, Path], dict]:
        repo_root = Path(__file__).resolve().parents[3]
        current_source = {
            "source_commit": "a" * 40,
            "source_worktree_dirty": False,
            "tracked_diff_hash": "0" * 64,
            "untracked_source_files": [],
            "release_eligible_source": True,
        }
        descriptor = {
            "generation_provenance_version": 1,
            "source_state": current_source,
            "protocol": "canonical",
            "schema_registry_hash": stable_json_sha256(TOOLS),
            "generator_hashes": {
                "psse_env/sft/provenance.py": file_sha256(
                    repo_root / "psse_env/sft/provenance.py"
                )
            },
            "generation_config": {"seed": 7},
        }
        provenance_id = stable_json_sha256(descriptor)
        source_row = row()
        source_row["generation_provenance_id"] = provenance_id
        datasets = {
            "train": root / "aggregate.train.jsonl",
            "validation": root / "aggregate.validation.jsonl",
        }
        for path in datasets.values():
            path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
        aggregate_manifest = root / "aggregate.manifest.json"
        aggregate_manifest.write_text(
            json.dumps({"episode_audits": []}) + "\n",
            encoding="utf-8",
        )
        manifest = {
            **descriptor,
            "generation_descriptor": descriptor,
            "generation_provenance_id": provenance_id,
            "dataset_hashes": {
                path.name: file_sha256(path)
                for path in (*datasets.values(), aggregate_manifest)
            },
            "release_eligible": True,
            "release_failures": [],
        }
        (root / "aggregate.generation_provenance.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return source_row, datasets, current_source

    def test_clean_matching_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_row, datasets, source_state = self._fixture(Path(temp_dir))
            with mock.patch(
                "psse_env.sft.provenance.git_source_state",
                return_value=source_state,
            ):
                result = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                )
        self.assertTrue(result["passed"], result["failures"])

    def test_missing_manifest_and_row_or_dataset_mismatch_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_row, datasets, source_state = self._fixture(root)
            manifest_path = root / "aggregate.generation_provenance.json"
            manifest_path.unlink()
            with mock.patch(
                "psse_env.sft.provenance.git_source_state",
                return_value=source_state,
            ):
                missing = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                )
            self.assertFalse(missing["passed"])
            self.assertTrue(any("missing" in item for item in missing["failures"]))

            source_row, datasets, source_state = self._fixture(root)
            bad_row = {**source_row, "generation_provenance_id": "wrong"}
            with mock.patch(
                "psse_env.sft.provenance.git_source_state",
                return_value=source_state,
            ):
                wrong_id = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[bad_row, source_row],
                )
            self.assertFalse(wrong_id["passed"])
            self.assertTrue(any("identifier" in item for item in wrong_id["failures"]))

            datasets["train"].write_text("{}\n", encoding="utf-8")
            with mock.patch(
                "psse_env.sft.provenance.git_source_state",
                return_value=source_state,
            ):
                wrong_hash = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                )
            self.assertFalse(wrong_hash["passed"])
            self.assertTrue(any("hash mismatch" in item for item in wrong_hash["failures"]))

    def test_generation_commit_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_row, datasets, source_state = self._fixture(Path(temp_dir))
            current = {**source_state, "source_commit": "b" * 40}
            with mock.patch(
                "psse_env.sft.provenance.git_source_state", return_value=current
            ):
                result = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                )
        self.assertFalse(result["passed"])
        self.assertTrue(any("commit" in item for item in result["failures"]))

    def test_round0_private_manifest_is_mandatory_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_row, datasets, source_state = self._fixture(root)
            aggregate_manifest = root / "aggregate.manifest.json"
            aggregate_manifest.unlink()
            with mock.patch(
                "psse_env.sft.provenance.git_source_state",
                return_value=source_state,
            ):
                missing = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                )
            self.assertFalse(missing["passed"])
            self.assertTrue(
                any("aggregate.manifest.json is missing" in item for item in missing["failures"])
            )

            source_row, datasets, source_state = self._fixture(root)
            aggregate_manifest.write_text("tampered\n", encoding="utf-8")
            with mock.patch(
                "psse_env.sft.provenance.git_source_state",
                return_value=source_state,
            ):
                tampered = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                )
            self.assertFalse(tampered["passed"])
            self.assertTrue(
                any("manifest.json hash" in item for item in tampered["failures"])
            )

    def test_round1_union_provenance_does_not_require_a_d0_style_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_row, datasets, source_state = self._fixture(root)
            provenance_path = root / "aggregate.generation_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            descriptor = provenance["generation_descriptor"]
            descriptor["builder_contract"] = ROUND1_AGGREGATE_BUILDER_CONTRACT
            provenance_id = stable_json_sha256(descriptor)
            provenance["generation_provenance_id"] = provenance_id
            provenance["dataset_hashes"].pop("aggregate.manifest.json")
            source_row["generation_provenance_id"] = provenance_id
            for path in datasets.values():
                path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
                provenance["dataset_hashes"][path.name] = file_sha256(path)
            provenance_path.write_text(
                json.dumps(provenance),
                encoding="utf-8",
            )
            (root / "aggregate.manifest.json").unlink()
            with mock.patch(
                "psse_env.sft.provenance.git_source_state",
                return_value=source_state,
            ):
                result = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                )
        self.assertTrue(result["passed"], result["failures"])

    def test_caller_held_snapshot_hashes_are_authoritative_over_live_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_row, datasets, source_state = self._fixture(Path(temp_dir))
            snapshot_hashes = {
                split: file_sha256(path) for split, path in datasets.items()
            }
            with mock.patch(
                "psse_env.sft.provenance.git_source_state",
                return_value=source_state,
            ):
                matching = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                    dataset_sha256=snapshot_hashes,
                )
                transient_swap = validate_generation_provenance(
                    repo_root=Path(__file__).resolve().parents[3],
                    datasets=datasets,
                    rows=[source_row, source_row],
                    dataset_sha256={
                        **snapshot_hashes,
                        "train": "f" * 64,
                    },
                )
        self.assertTrue(matching["passed"], matching["failures"])
        self.assertFalse(transient_swap["passed"])
        self.assertTrue(
            any("hash mismatch" in item for item in transient_swap["failures"]),
            transient_swap["failures"],
        )

    def test_training_prepare_rejects_auxiliary_without_round1_source_gate(
        self,
    ) -> None:
        source_rows = [recovery_probe_row()]
        validation_rows = [row("validation")]
        with mock.patch(
            "psse_env.sft.training.load_jsonl",
            side_effect=[source_rows, validation_rows],
        ):
            with self.assertRaisesRegex(GateError, "Round-1 source gate"):
                training_module._prepare_pilot(
                    train_file="train.jsonl",
                    validation_file="validation.jsonl",
                    settings=TrainerSettings(revision="a" * 40),
                    pilot_minimum_rows=2,
                    pilot_maximum_rows=2,
                )

    def test_training_prepare_passes_only_source_gate_returned_id_to_grouped_gate(
        self,
    ) -> None:
        provenance_id = "f" * 64
        source_rows = [recovery_probe_row(provenance_id=provenance_id)]
        validation_rows = [row("validation")]
        grouped = SimpleNamespace(passed=True, failures=[])
        gate = SimpleNamespace(failures=[], prepared=[])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = TrainerSettings(
                revision="a" * 40,
                output_dir=str(root / "output"),
                initial_adapter_path=str(root / "adapter"),
                initial_adapter_revision="b" * 64,
                round1_provenance_path=str(root / "aggregate.generation_provenance.json"),
                round1_preflight_path=str(root / "aggregate.preflight.json"),
                reviewed_source_commit="c" * 40,
                round1_view="full",
            )
            with (
                mock.patch(
                    "psse_env.sft.training.load_jsonl",
                    side_effect=[source_rows, validation_rows],
                ),
                mock.patch(
                    "psse_env.sft.round1_source_gate.validate_round1_source_mix_gate",
                    return_value={
                        "passed": True,
                        "generation_provenance_id": provenance_id,
                        "canonical_dataset_content_sha256": {
                            "train": stable_json_sha256(source_rows),
                            "validation": stable_json_sha256(validation_rows),
                            "test": stable_json_sha256([]),
                        },
                    },
                ) as source_gate,
                mock.patch(
                    "psse_env.sft.training.validate_grouped_pilot",
                    return_value=grouped,
                ) as grouped_gate,
                mock.patch(
                    "psse_env.sft.training.validate_generation_provenance",
                    return_value={"passed": True, "failures": []},
                ),
                mock.patch(
                    "psse_env.sft.training.load_exact_processor",
                    return_value=(FakeProcessor(), "AutoProcessor"),
                ),
                mock.patch(
                    "psse_env.sft.training.audit_dataset", return_value=gate
                ),
            ):
                training_module._prepare_pilot(
                    train_file="train.jsonl",
                    validation_file="validation.jsonl",
                    settings=settings,
                    pilot_minimum_rows=2,
                    pilot_maximum_rows=2,
                )

        source_gate.assert_called_once_with(
            settings.round1_provenance_path,
            settings.round1_preflight_path,
            reviewed_source_commit="c" * 40,
            initial_adapter_revision="b" * 64,
            round1_view="full",
            train_path="train.jsonl",
            validation_path="validation.jsonl",
        )
        self.assertEqual(
            grouped_gate.call_args.kwargs[
                "validated_round1_generation_provenance_id"
            ],
            provenance_id,
        )

    def test_round1_sibling_provenance_forces_gate_after_probe_relabeling(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "aggregate.train_view.jsonl"
            validation_path = root / "aggregate.validation.jsonl"
            train_path.write_text(
                json.dumps(row("laundered-train")) + "\n",
                encoding="utf-8",
            )
            validation_path.write_text(
                json.dumps(row("laundered-validation")) + "\n",
                encoding="utf-8",
            )
            (root / "aggregate.generation_provenance.json").write_text(
                json.dumps(
                    {
                        "generation_descriptor": {
                            "builder_contract": ROUND1_AGGREGATE_BUILDER_CONTRACT
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GateError, "Round-1 source gate"):
                training_module._prepare_pilot(
                    train_file=train_path,
                    validation_file=validation_path,
                    settings=TrainerSettings(revision="a" * 40),
                    pilot_minimum_rows=2,
                    pilot_maximum_rows=2,
                )

    def test_training_rejects_gate_content_different_from_loaded_rows(self) -> None:
        provenance_id = "f" * 64
        source_rows = [recovery_probe_row(provenance_id=provenance_id)]
        validation_rows = [row("validation")]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = TrainerSettings(
                revision="a" * 40,
                output_dir=str(root / "output"),
                initial_adapter_path=str(root / "adapter"),
                initial_adapter_revision="b" * 64,
                round1_provenance_path=str(
                    root / "aggregate.generation_provenance.json"
                ),
                round1_preflight_path=str(root / "aggregate.preflight.json"),
                reviewed_source_commit="c" * 40,
                round1_view="full",
            )
            with (
                mock.patch(
                    "psse_env.sft.training.load_jsonl",
                    side_effect=[source_rows, validation_rows],
                ),
                mock.patch(
                    "psse_env.sft.round1_source_gate.validate_round1_source_mix_gate",
                    return_value={
                        "passed": True,
                        "generation_provenance_id": provenance_id,
                        "canonical_dataset_content_sha256": {
                            "train": "0" * 64,
                            "validation": stable_json_sha256(validation_rows),
                            "test": stable_json_sha256([]),
                        },
                    },
                ),
                self.assertRaisesRegex(GateError, "rows already loaded"),
            ):
                training_module._prepare_pilot(
                    train_file="train.jsonl",
                    validation_file="validation.jsonl",
                    settings=settings,
                    pilot_minimum_rows=2,
                    pilot_maximum_rows=2,
                )

    def test_training_prepare_requires_explicit_nonrelease_override(self) -> None:
        source_rows = [row("g0", "a"), row("g1", "b")]
        grouped = SimpleNamespace(passed=True, failures=[])
        gate = SimpleNamespace(failures=[], prepared=[])
        base = TrainerSettings(revision="a" * 40)
        patches = (
            mock.patch(
                "psse_env.sft.training.load_jsonl", side_effect=[source_rows, source_rows]
            ),
            mock.patch(
                "psse_env.sft.training.validate_grouped_pilot", return_value=grouped
            ),
            mock.patch(
                "psse_env.sft.training.validate_generation_provenance",
                return_value={"passed": False, "failures": ["nonrelease"]},
            ),
            mock.patch(
                "psse_env.sft.training.load_exact_processor",
                return_value=(FakeProcessor(), "fake"),
            ),
            mock.patch("psse_env.sft.training.audit_dataset", return_value=gate),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.assertRaisesRegex(GateError, "Generation provenance"):
                training_module._prepare_pilot(
                    train_file="train.jsonl",
                    validation_file="validation.jsonl",
                    settings=base,
                    pilot_minimum_rows=2,
                    pilot_maximum_rows=4,
                )

        override = replace(base, allow_nonrelease_artifacts=True)
        patches = (
            mock.patch(
                "psse_env.sft.training.load_jsonl", side_effect=[source_rows, source_rows]
            ),
            mock.patch(
                "psse_env.sft.training.validate_grouped_pilot", return_value=grouped
            ),
            mock.patch(
                "psse_env.sft.training.validate_generation_provenance",
                return_value={"passed": False, "failures": ["nonrelease"]},
            ),
            mock.patch(
                "psse_env.sft.training.load_exact_processor",
                return_value=(FakeProcessor(), "fake"),
            ),
            mock.patch("psse_env.sft.training.audit_dataset", return_value=gate),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            training_module._prepare_pilot(
                train_file="train.jsonl",
                validation_file="validation.jsonl",
                settings=override,
                pilot_minimum_rows=2,
                pilot_maximum_rows=4,
            )

    def test_release_training_requires_auto_processor(self) -> None:
        source_rows = [row("g0", "a"), row("g1", "b")]
        grouped = SimpleNamespace(passed=True, failures=[])
        gate = SimpleNamespace(failures=[], prepared=[])
        settings = TrainerSettings(
            revision="a" * 40,
            required_processor_loader="AutoProcessor",
        )

        def prepare_with(loader: str) -> None:
            with (
                mock.patch(
                    "psse_env.sft.training.load_jsonl",
                    side_effect=[source_rows, source_rows],
                ),
                mock.patch(
                    "psse_env.sft.training.validate_grouped_pilot",
                    return_value=grouped,
                ),
                mock.patch(
                    "psse_env.sft.training.validate_generation_provenance",
                    return_value={"passed": True, "failures": []},
                ),
                mock.patch(
                    "psse_env.sft.training.load_exact_processor",
                    return_value=(FakeProcessor(), loader),
                ),
                mock.patch(
                    "psse_env.sft.training.audit_dataset", return_value=gate
                ),
            ):
                training_module._prepare_pilot(
                    train_file="train.jsonl",
                    validation_file="validation.jsonl",
                    settings=settings,
                    pilot_minimum_rows=2,
                    pilot_maximum_rows=4,
                )

        with self.assertRaisesRegex(GateError, "requires AutoProcessor"):
            prepare_with("AutoTokenizer")
        prepare_with("AutoProcessor")

    def test_full_training_forces_auto_processor_for_programmatic_callers(
        self,
    ) -> None:
        captured: dict[str, TrainerSettings] = {}

        def stop_after_settings(**kwargs):
            captured["settings"] = kwargs["settings"]
            raise GateError("stop after processor contract")

        input_snapshot = SimpleNamespace(assert_current=lambda: None)
        with (
            mock.patch(
                "psse_env.sft.training._snapshot_training_inputs",
                return_value=input_snapshot,
            ),
            mock.patch(
                "psse_env.sft.training._prepare_checkpoint_receipt_binding",
                return_value={"variant_id": "bc0"},
            ),
            mock.patch(
                "psse_env.sft.training._prepare_pilot",
                side_effect=stop_after_settings,
            ),
            self.assertRaisesRegex(GateError, "stop after processor contract"),
        ):
            training_module.run_lora_training(
                train_file="train.jsonl",
                validation_file="validation.jsonl",
                settings=TrainerSettings(revision="a" * 40),
            )

        self.assertEqual(
            captured["settings"].required_processor_loader, "AutoProcessor"
        )


class TestReleaseGateReport(unittest.TestCase):
    @staticmethod
    def _report(root: Path) -> tuple[Path, dict[str, Path], dict]:
        datasets = {
            split: root / f"aggregate.{split}.jsonl"
            for split in ("train", "validation", "test")
        }
        for split, path in datasets.items():
            path.write_text(json.dumps({"split": split}) + "\n", encoding="utf-8")
        source_commit = "a" * 40
        revision = "b" * 40
        split_gate = {
            "passed": True,
            "length_audit": {
                "prompt_truncated_rows": 0,
                "target_truncated_rows": 0,
            },
        }
        payload = {
            "passed": True,
            "release_eligible": True,
            "processor_loader": "AutoProcessor",
            "processor_loader_requirement": "AutoProcessor",
            "processor_loader_passed": True,
            "model": "unsloth/gemma-4-31B-it",
            "revision": revision,
            "max_length": 8192,
            "provenance": {
                "release_eligible_source": True,
                "source_commit": source_commit,
                "processor_revision": revision,
                "dataset_hashes": {
                    split: file_sha256(path) for split, path in datasets.items()
                },
            },
            "generation_provenance": {
                "passed": True,
                "release_eligible": True,
                "source_commit": source_commit,
            },
            **{split: copy.deepcopy(split_gate) for split in datasets},
        }
        report_path = root / "gate_report.json"
        report_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return report_path, datasets, payload

    @staticmethod
    def _validate(report_path: Path, datasets: dict[str, Path]) -> dict:
        return validate_release_gate_report(
            report_path,
            model="unsloth/gemma-4-31B-it",
            revision="b" * 40,
            source_commit="a" * 40,
            datasets=datasets,
            max_length=8192,
        )

    def test_exact_auto_processor_report_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path, datasets, _ = self._report(Path(temp_dir))
            result = self._validate(report_path, datasets)
        self.assertTrue(result["passed"], result["failures"])

    def test_stale_or_tokenizer_report_fails_closed(self) -> None:
        mutations = {
            "tokenizer fallback": lambda payload: payload.update(
                processor_loader="AutoTokenizer"
            ),
            "stale source": lambda payload: payload["provenance"].update(
                source_commit="c" * 40
            ),
            "wrong model": lambda payload: payload.update(model="other/model"),
            "wrong revision": lambda payload: payload.update(revision="c" * 40),
            "wrong max length": lambda payload: payload.update(max_length=4096),
            "stale aggregate source": lambda payload: payload[
                "generation_provenance"
            ].update(source_commit="c" * 40),
            "wrong dataset": lambda payload: payload["provenance"][
                "dataset_hashes"
            ].update(train="0" * 64),
            "prompt truncation": lambda payload: payload["train"][
                "length_audit"
            ].update(prompt_truncated_rows=1),
        }
        for label, mutate in mutations.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temp_dir:
                report_path, datasets, payload = self._report(Path(temp_dir))
                mutate(payload)
                report_path.write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                )
                result = self._validate(report_path, datasets)
                self.assertFalse(result["passed"])
                self.assertTrue(result["failures"])

    def test_missing_or_empty_report_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path, datasets, _ = self._report(root)
            report_path.unlink()
            missing = self._validate(report_path, datasets)
            report_path.write_text("{}\n", encoding="utf-8")
            empty = self._validate(report_path, datasets)
        self.assertFalse(missing["passed"])
        self.assertFalse(empty["passed"])

    def test_round0_launcher_uses_one_strict_processor_gate_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        launcher = (repo_root / "submit_dagger_sft_round0.sh").read_text(
            encoding="utf-8"
        )
        for contract in (
            "#SBATCH --gres=gpu:1",
            '#SBATCH --constraint="h200|h100|rtx6000"',
            '#SBATCH --comment="preemption=yes;requeue=true"',
            "--query-gpu=name,memory.total,driver_version",
            "validate_torch_release_accelerator",
            '--expert-baseline-gate-receipt "$EXPERT_BASELINE_GATE_RECEIPT"',
            "release training OUTPUT_DIR must be outside the source checkout",
        ):
            self.assertIn(contract, launcher)
        self.assertIn("--require-auto-processor", launcher)
        self.assertIn(
            '--report-output "$PROCESSOR_GATE_REPORT"', launcher
        )
        self.assertNotIn(
            '--report-output "$OUTPUT_DIR/gate_report.json"', launcher
        )
        self.assertIn("validate_release_gate_report", launcher)
        for value in (
            '"$REVIEWED_SOURCE_COMMIT"',
            '"$TRAIN_FILE"',
            '"$VALIDATION_FILE"',
            '"$TEST_FILE"',
            '"$MAX_LENGTH"',
        ):
            self.assertIn(value, launcher)


class TestBaselineEvaluationGate(unittest.TestCase):
    @staticmethod
    def _result(
        *,
        role: str,
        passed: bool,
        evidence_passed: bool,
        performance_passed: bool,
        evidence_failures: tuple[str, ...] = (),
        performance_failures: tuple[str, ...] = (),
        source_commit: str = "a" * 40,
    ) -> SimpleNamespace:
        failures = (
            evidence_failures + performance_failures
            if role != "base-baseline"
            else evidence_failures
        )
        payload = {
            "passed": passed,
            "failures": failures,
            "validation_role": role,
            "evidence_passed": evidence_passed,
            "performance_passed": performance_passed,
            "performance_enforced": role != "base-baseline",
            "evidence_failures": evidence_failures,
            "performance_failures": performance_failures,
            "source_commit": source_commit,
        }
        return SimpleNamespace(
            **payload,
            as_dict=lambda: copy.deepcopy(payload),
        )

    @staticmethod
    def _args(root: Path) -> argparse.Namespace:
        paths = {
            name: root / f"{name}.json"
            for name in ("expert", "base", "suite", "policy")
        }
        for path in paths.values():
            path.write_text("{}\n", encoding="utf-8")
        return argparse.Namespace(
            expert_baseline_evaluation=paths["expert"],
            base_baseline_evaluation=paths["base"],
            evaluation_suite=paths["suite"],
            evaluation_policy=paths["policy"],
            expert_policy_identity="observable-expert-v1",
            model="unsloth/gemma-4-31B-it",
            revision="b" * 40,
            baseline_evaluation_report_output=root / "baseline-report.json",
        )

    @staticmethod
    def _source_state() -> dict[str, object]:
        return {
            "source_commit": "a" * 40,
            "release_eligible_source": True,
        }

    def test_weak_base_performance_is_recorded_but_does_not_block(self) -> None:
        expert = self._result(
            role="expert-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=True,
        )
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
            performance_failures=("minimum terminal rate was not met",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[expert, base, expert, base],
                ) as validate,
            ):
                payload = _baseline_evaluation_gate(args)
            report = json.loads(
                args.baseline_evaluation_report_output.read_text(encoding="utf-8")
            )

        self.assertTrue(payload["passed"])
        self.assertTrue(payload["pretraining_gate_passed"])
        self.assertFalse(payload["release_eligible"])
        self.assertFalse(payload["all_baselines_performance_qualified"])
        self.assertEqual(payload["failures"], [])
        self.assertEqual(
            payload["base_performance_findings"],
            ["base performance: minimum terminal rate was not met"],
        )
        self.assertEqual(
            report["base"]["performance_failures"],
            ["minimum terminal rate was not met"],
        )
        expert_call, base_call, fresh_expert_call, fresh_base_call = (
            validate.call_args_list
        )
        self.assertEqual(expert_call.kwargs["role"], "expert-baseline")
        self.assertEqual(base_call.kwargs["role"], "base-baseline")
        self.assertEqual(
            expert_call.kwargs["expected_suite_path"], args.evaluation_suite
        )
        self.assertEqual(
            base_call.kwargs["expected_suite_path"], args.evaluation_suite
        )
        self.assertNotIn("expected_suite_sha256", expert_call.kwargs)
        self.assertEqual(fresh_expert_call.kwargs, expert_call.kwargs)
        self.assertEqual(fresh_base_call.kwargs, base_call.kwargs)

    def test_historical_expert_closure_is_distinct_from_current_base_gate(self) -> None:
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
            performance_failures=("minimum terminal rate was not met",),
        )
        closure = {
            "passed": True,
            "failures": [],
            "contract": "historical_expert_closure_validation_v1",
            "artifact": {"source_commit": "e" * 40},
            "expert": {
                "performance_passed": True,
                "artifact_source_commit": "e" * 40,
                "validator_source_commit": "f" * 40,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            args.expert_baseline_gate_receipt = Path(temp_dir) / "receipt.json"
            args.expert_baseline_gate_receipt.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_historical_expert_closure",
                    return_value=closure,
                ) as validate_closure,
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    return_value=base,
                ) as validate_current,
            ):
                payload = _baseline_evaluation_gate(args)

        self.assertTrue(payload["passed"])
        self.assertEqual(
            payload["expert_validation_mode"], "historical-closure-reuse"
        )
        self.assertEqual(payload["expert_artifact_source_commit"], "e" * 40)
        self.assertEqual(payload["expert_validator_source_commit"], "f" * 40)
        self.assertEqual(payload["consumer_source_commit"], "a" * 40)
        self.assertEqual(payload["base_artifact_source_commit"], "a" * 40)
        expected_closure_call = mock.call(
            args.expert_baseline_gate_receipt,
            expert_artifact_path=args.expert_baseline_evaluation,
            repo_root=Path(__file__).resolve().parents[3],
            expected_suite_path=args.evaluation_suite,
            expected_policy_path=args.evaluation_policy,
            expected_policy_identity=args.expert_policy_identity,
            expected_protocol="canonical",
            expected_registry_sha256="c" * 64,
        )
        self.assertEqual(
            validate_closure.call_args_list,
            [expected_closure_call, expected_closure_call],
        )
        self.assertEqual(validate_current.call_count, 2)
        self.assertEqual(
            validate_current.call_args_list[0].kwargs["role"], "base-baseline"
        )

    def test_failed_historical_artifact_does_not_claim_observed_source(self) -> None:
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
        )
        closure = {
            "passed": False,
            "failures": ["historical expert artifact is not valid JSON"],
            "contract": "historical_expert_closure_validation_v1",
            "artifact": {},
            "expert": {
                "performance_passed": True,
                "artifact_source_commit": "e" * 40,
                "validator_source_commit": "f" * 40,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            args.expert_baseline_gate_receipt = Path(temp_dir) / "receipt.json"
            args.expert_baseline_gate_receipt.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_historical_expert_closure",
                    return_value=closure,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    return_value=base,
                ),
                self.assertRaisesRegex(GateError, "artifact is not valid JSON"),
            ):
                _baseline_evaluation_gate(args)
            report = json.loads(
                args.baseline_evaluation_report_output.read_text(encoding="utf-8")
            )

        self.assertEqual(report["expert_artifact_source_commit"], "")
        self.assertEqual(report["expert_validator_source_commit"], "f" * 40)

    def test_postpublication_evidence_change_removes_new_report(self) -> None:
        expert = self._result(
            role="expert-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=True,
        )
        changed_expert = self._result(
            role="expert-baseline",
            passed=False,
            evidence_passed=False,
            performance_passed=False,
            evidence_failures=("artifact identity changed",),
        )
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[expert, base, changed_expert, base],
                ),
                self.assertRaisesRegex(
                    GateError,
                    "expert evidence changed during baseline report publication",
                ),
            ):
                _baseline_evaluation_gate(args)
            report_exists = args.baseline_evaluation_report_output.exists()
        self.assertFalse(report_exists)

    def test_postpublication_historical_closure_change_removes_report(self) -> None:
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
        )
        closure = {
            "passed": True,
            "failures": [],
            "contract": "historical_expert_closure_validation_v1",
            "artifact": {"source_commit": "e" * 40},
            "expert": {
                "performance_passed": True,
                "artifact_source_commit": "e" * 40,
                "validator_source_commit": "f" * 40,
            },
            "consumer_source_attestation": {"unchanged": True},
        }
        changed = copy.deepcopy(closure)
        changed["passed"] = False
        changed["failures"] = ["consumer repository contains Git replacement refs"]
        changed["consumer_source_attestation"] = {"unchanged": False}
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            args.expert_baseline_gate_receipt = Path(temp_dir) / "receipt.json"
            args.expert_baseline_gate_receipt.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_historical_expert_closure",
                    side_effect=[closure, changed],
                ) as validate_closure,
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[base, base],
                ),
                self.assertRaisesRegex(
                    GateError,
                    "historical expert closure.*changed during baseline report publication",
                ),
            ):
                _baseline_evaluation_gate(args)
            report_exists = args.baseline_evaluation_report_output.exists()
        self.assertFalse(report_exists)
        self.assertEqual(validate_closure.call_count, 2)
        self.assertEqual(
            validate_closure.call_args_list[0], validate_closure.call_args_list[1]
        )

    def test_postpublication_base_change_removes_report(self) -> None:
        closure = {
            "passed": True,
            "failures": [],
            "contract": "historical_expert_closure_validation_v1",
            "artifact": {"source_commit": "e" * 40},
            "expert": {
                "performance_passed": True,
                "artifact_source_commit": "e" * 40,
                "validator_source_commit": "f" * 40,
            },
        }
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
        )
        changed_base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
            source_commit="d" * 40,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            args.expert_baseline_gate_receipt = Path(temp_dir) / "receipt.json"
            args.expert_baseline_gate_receipt.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_historical_expert_closure",
                    return_value=closure,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[base, changed_base],
                ) as validate_base,
                self.assertRaisesRegex(
                    GateError,
                    "Base evidence changed during baseline report publication",
                ),
            ):
                _baseline_evaluation_gate(args)
            report_exists = args.baseline_evaluation_report_output.exists()
        self.assertFalse(report_exists)
        self.assertEqual(validate_base.call_count, 2)
        self.assertEqual(
            validate_base.call_args_list[0].kwargs,
            validate_base.call_args_list[1].kwargs,
        )

    def test_postpublication_failed_safe_removal_is_explicit(self) -> None:
        expert = self._result(
            role="expert-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=True,
        )
        changed_expert = self._result(
            role="expert-baseline",
            passed=False,
            evidence_passed=False,
            performance_passed=False,
            evidence_failures=("artifact identity changed",),
        )
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[expert, base, changed_expert, base],
                ),
                mock.patch(
                    "psse_env.sft.cli.evaluation_gate._unlink_created_report",
                    return_value=False,
                ),
                self.assertRaisesRegex(
                    GateError,
                    "newly created report could not be safely removed",
                ),
            ):
                _baseline_evaluation_gate(args)

    def test_failed_foreign_base_preserves_observed_source_label(self) -> None:
        expert = self._result(
            role="expert-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=True,
        )
        base = self._result(
            role="base-baseline",
            passed=False,
            evidence_passed=False,
            performance_passed=False,
            evidence_failures=("source commit differs",),
            source_commit="d" * 40,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[expert, base, expert, base],
                ),
                self.assertRaisesRegex(GateError, "source commit differs"),
            ):
                _baseline_evaluation_gate(args)
            report = json.loads(
                args.baseline_evaluation_report_output.read_text(encoding="utf-8")
            )
        self.assertEqual(report["base_artifact_source_commit"], "d" * 40)
        self.assertEqual(report["base_expected_source_commit"], "a" * 40)

    def test_baseline_report_is_no_clobber(self) -> None:
        expert = self._result(
            role="expert-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=True,
        )
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            args.baseline_evaluation_report_output.write_text(
                "do not replace\n", encoding="utf-8"
            )
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[expert, base, expert, base],
                ),
                self.assertRaisesRegex(ValueError, "already exists"),
            ):
                _baseline_evaluation_gate(args)
            preserved = args.baseline_evaluation_report_output.read_text(
                encoding="utf-8"
            )
        self.assertEqual(preserved, "do not replace\n")

    def test_base_evidence_failure_blocks_and_is_reported(self) -> None:
        expert = self._result(
            role="expert-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=True,
        )
        base = self._result(
            role="base-baseline",
            passed=False,
            evidence_passed=False,
            performance_passed=False,
            evidence_failures=("strict audit evidence is incomplete",),
            performance_failures=("minimum terminal rate was not met",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[expert, base, expert, base],
                ),
                self.assertRaisesRegex(GateError, "strict audit evidence is incomplete"),
            ):
                _baseline_evaluation_gate(args)
            report = json.loads(
                args.baseline_evaluation_report_output.read_text(encoding="utf-8")
            )

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["failures"],
            ["base evidence: strict audit evidence is incomplete"],
        )
        self.assertEqual(
            report["base_performance_findings"],
            ["base performance: minimum terminal rate was not met"],
        )

    def test_expert_performance_failure_still_blocks(self) -> None:
        expert = self._result(
            role="expert-baseline",
            passed=False,
            evidence_passed=True,
            performance_passed=False,
            performance_failures=("minimum terminal rate was not met",),
        )
        base = self._result(
            role="base-baseline",
            passed=True,
            evidence_passed=True,
            performance_passed=False,
            performance_failures=("minimum terminal rate was not met",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            with (
                mock.patch(
                    "psse_env.sft.cli.git_source_state",
                    return_value=self._source_state(),
                ),
                mock.patch(
                    "psse_env.sft.cli.current_registry_sha256",
                    return_value="c" * 64,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_evaluation_artifact",
                    side_effect=[expert, base, expert, base],
                ),
                self.assertRaisesRegex(GateError, "expert: minimum terminal rate"),
            ):
                _baseline_evaluation_gate(args)


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

    def test_cli_strict_gate_rejects_tokenizer_fallback_and_writes_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                split: root / f"{split}.jsonl"
                for split in ("train", "validation", "test")
            }
            for split, path in paths.items():
                source = row(split)
                source["metadata"]["protocol"] = "canonical"
                path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            report_path = root / "gate_report.json"
            gate = SimpleNamespace(
                passed=True,
                to_dict=lambda: {
                    "passed": True,
                    "length_audit": {
                        "prompt_truncated_rows": 0,
                        "target_truncated_rows": 0,
                    },
                },
            )
            provenance = {
                "release_eligible_source": True,
                "source_commit": "a" * 40,
            }
            with (
                mock.patch(
                    "psse_env.sft.cli.load_exact_processor",
                    return_value=(FakeProcessor(), "AutoTokenizer"),
                ),
                mock.patch(
                    "psse_env.sft.cli.audit_dataset", return_value=gate
                ),
                mock.patch(
                    "psse_env.sft.cli.build_gate_provenance",
                    return_value=provenance,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_generation_provenance",
                    return_value={"passed": True},
                ),
            ):
                result = cli_main(
                    [
                        "gate",
                        "--model",
                        "unsloth/gemma-4-31B-it",
                        "--revision",
                        "a" * 40,
                        "--train",
                        str(paths["train"]),
                        "--validation",
                        str(paths["validation"]),
                        "--test",
                        str(paths["test"]),
                        "--pilot-min-rows",
                        "3",
                        "--pilot-max-rows",
                        "3",
                        "--require-auto-processor",
                        "--report-output",
                        str(report_path),
                    ]
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 2)
        self.assertFalse(report["passed"])
        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["processor_loader"], "AutoTokenizer")
        self.assertEqual(
            report["processor_loader_requirement"], "AutoProcessor"
        )
        self.assertFalse(report["processor_loader_passed"])

    def test_cli_data_gate_authenticates_round1_probe_before_grouped_gate(
        self,
    ) -> None:
        provenance_id = "f" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            test = root / "test.jsonl"
            probe = recovery_probe_row(provenance_id=provenance_id)
            probe["metadata"]["protocol"] = "canonical"
            valid = row("validation")
            valid["metadata"]["protocol"] = "canonical"
            test_row = row("test")
            test_row["metadata"]["protocol"] = "canonical"
            train.write_text(json.dumps(probe) + "\n", encoding="utf-8")
            validation.write_text(json.dumps(valid) + "\n", encoding="utf-8")
            test.write_text(json.dumps(test_row) + "\n", encoding="utf-8")
            common = [
                "gate",
                "--revision",
                "a" * 40,
                "--train",
                str(train),
                "--validation",
                str(validation),
                "--test",
                str(test),
                "--pilot-min-rows",
                "3",
                "--pilot-max-rows",
                "3",
            ]

            without_source = cli_main(common)
            self.assertEqual(without_source, 2)

            gate = SimpleNamespace(
                passed=True,
                to_dict=lambda: {
                    "passed": True,
                    "length_audit": {
                        "prompt_truncated_rows": 0,
                        "target_truncated_rows": 0,
                    },
                },
            )
            provenance = {
                "release_eligible_source": True,
                "source_commit": "a" * 40,
            }
            with (
                mock.patch(
                    "psse_env.sft.round1_source_gate.validate_round1_source_mix_gate",
                    return_value={
                        "passed": True,
                        "generation_provenance_id": provenance_id,
                        "canonical_dataset_content_sha256": {
                            "train": stable_json_sha256([probe]),
                            "validation": stable_json_sha256([valid]),
                            "test": stable_json_sha256([test_row]),
                        },
                    },
                ) as source_gate,
                mock.patch(
                    "psse_env.sft.cli.load_exact_processor",
                    return_value=(FakeProcessor(), "AutoProcessor"),
                ),
                mock.patch("psse_env.sft.cli.audit_dataset", return_value=gate),
                mock.patch(
                    "psse_env.sft.cli.build_gate_provenance",
                    return_value=provenance,
                ),
                mock.patch(
                    "psse_env.sft.cli.validate_generation_provenance",
                    return_value={"passed": True},
                ),
            ):
                with_source = cli_main(
                    [
                        *common,
                        "--round1-provenance",
                        str(root / "aggregate.generation_provenance.json"),
                        "--round1-preflight",
                        str(root / "aggregate.preflight.json"),
                        "--reviewed-source-commit",
                        "c" * 40,
                        "--initial-adapter-revision",
                        "b" * 64,
                        "--round1-view",
                        "full",
                    ]
                )

        self.assertEqual(with_source, 0)
        source_gate.assert_called_once_with(
            root / "aggregate.generation_provenance.json",
            root / "aggregate.preflight.json",
            reviewed_source_commit="c" * 40,
            initial_adapter_revision="b" * 64,
            round1_view="full",
            train_path=root / "train.jsonl",
            validation_path=root / "validation.jsonl",
            test_path=root / "test.jsonl",
        )

    def test_cli_round1_provenance_forces_source_args_after_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "aggregate.train_view.jsonl"
            validation = root / "aggregate.validation.jsonl"
            train.write_text(json.dumps(row("train")) + "\n", encoding="utf-8")
            validation.write_text(
                json.dumps(row("validation")) + "\n", encoding="utf-8"
            )
            (root / "aggregate.generation_provenance.json").write_text(
                json.dumps(
                    {
                        "generation_descriptor": {
                            "builder_contract": ROUND1_AGGREGATE_BUILDER_CONTRACT
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = cli_main(
                [
                    "gate",
                    "--revision",
                    "a" * 40,
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--pilot-min-rows",
                    "2",
                    "--pilot-max-rows",
                    "2",
                ]
            )
        self.assertEqual(result, 2)

    def test_cli_train_wires_auto_processor_requirement(self) -> None:
        result_object = SimpleNamespace(metrics={"train_loss": 1.0})
        with (
            mock.patch(
                "psse_env.sft.cli._baseline_evaluation_gate", return_value={}
            ),
            mock.patch(
                "psse_env.sft.cli.run_lora_training", return_value=result_object
            ) as run_training,
        ):
            result = cli_main(
                [
                    "train",
                    "--model",
                    "unsloth/gemma-4-31B-it",
                    "--revision",
                    "a" * 40,
                    "--train",
                    "train.jsonl",
                    "--validation",
                    "validation.jsonl",
                    "--seed",
                    "3407",
                ]
            )

        self.assertEqual(result, 0)
        settings = run_training.call_args.kwargs["settings"]
        self.assertEqual(settings.required_processor_loader, "AutoProcessor")


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
        parsed = run_generation_tool_call_smoke(model, processor, example)
        self.assertEqual(parsed, ParsedToolCall("run_wls", {"state_id": "active"}))
        self.assertIsNotNone(model.seen_mm_token_type_ids)

        class AlternateArgumentsModel(GeneratingModel):
            def generate(self, input_ids, mm_token_type_ids=None, **_kwargs):
                target = '<|tool_call|>call:run_wls{"state_id":"candidate"}'
                suffix = torch.tensor(
                    [[ord(char) for char in target]], device=input_ids.device
                )
                return torch.cat([input_ids, suffix], dim=1)

        alternate = AlternateArgumentsModel()
        with self.assertRaisesRegex(GateError, "does not round-trip"):
            run_generation_tool_call_smoke(alternate, processor, example)
        parsed = generate_single_tool_call(
            alternate,
            processor,
            example,
        )
        self.assertEqual(parsed, ParsedToolCall("run_wls", {"state_id": "candidate"}))

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
