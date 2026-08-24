from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from psse_env.research_models import GEMMA4_12B, GEMMA4_E2B_LEGACY, GEMMA4_E4B
from psse_env.sft import research_cli
from psse_env.sft.gates import GateError, ParsedToolCall
from psse_env.sft.research_cli import (
    ResearchTrainerSettings,
    _reload_saved_adapter,
    run_research_training,
    validate_research_splits,
)
from psse_env.sft.training import LoraSettings, resolve_language_lora_targets


REVISION = GEMMA4_E2B_LEGACY.revision


def _chat_row(root: str) -> dict:
    return {
        "example_id": root,
        "physical_root_fingerprint": root,
        "metadata": {"protocol": "canonical"},
    }


class ResearchSettingsTests(unittest.TestCase):
    def test_cli_selects_12b_atomically_by_default(self) -> None:
        args = research_cli.parser().parse_args(
            [
                "--train",
                "train.jsonl",
                "--validation",
                "validation.jsonl",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.model_choice, "12b")
        self.assertIsNone(args.model)
        self.assertIsNone(args.revision)

    def test_warm_start_requires_no_tree_digest_or_release_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "bc0" / "lora"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}), encoding="utf-8"
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            settings = ResearchTrainerSettings(
                model_name="unsloth/gemma-4-E2B-it",
                revision=REVISION,
                output_dir=str(root / "round1"),
                initial_adapter_path=str(adapter),
            )
            settings.validate()
            self.assertFalse(hasattr(settings, "initial_adapter_revision"))
            self.assertFalse(hasattr(settings, "round1_provenance_path"))

    def test_train_and_validation_roots_must_be_disjoint(self) -> None:
        report = validate_research_splits(
            [_chat_row("train_root")], [_chat_row("validation_root")]
        )
        self.assertEqual(report["overlap"], [])
        with self.assertRaisesRegex(GateError, "overlap"):
            validate_research_splits(
                [_chat_row("shared_root")], [_chat_row("shared_root")]
            )

    def test_production_label_eligibility_is_not_a_research_split_gate(self) -> None:
        train = _chat_row("train_root")
        validation = _chat_row("validation_root")
        train["production_label_eligible"] = False
        validation["production_label_eligible"] = False
        validate_research_splits([train], [validation])

    def test_cross_model_warm_start_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "e2b"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "base_model_name_or_path": "unsloth/gemma-4-E2B-it",
                    }
                ),
                encoding="utf-8",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            settings = ResearchTrainerSettings(
                model_name=GEMMA4_12B.model_id,
                revision=GEMMA4_12B.revision,
                architecture=GEMMA4_12B.architecture,
                output_dir=str(root / "output"),
                initial_adapter_path=str(adapter),
            )
            with self.assertRaisesRegex(GateError, "adapter/base mismatch"):
                settings.validate()


class ResearchModelLoaderTests(unittest.TestCase):
    def _modules(self, model_type: str):
        observed = {}

        class Model:
            config = types.SimpleNamespace(model_type=model_type)

            def generate(self, *_args, **_kwargs):
                return None

            def get_input_embeddings(self):
                return object()

        class AutoModel:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                observed.update(source=source, kwargs=dict(kwargs))
                return Model()

        transformers = types.ModuleType("transformers")
        transformers.AutoModelForMultimodalLM = AutoModel
        torch = types.ModuleType("torch")
        torch.bfloat16 = object()
        torch.float16 = object()
        return observed, {"transformers": transformers, "torch": torch}

    def test_multimodal_loader_accepts_e4b_and_unified_12b(self) -> None:
        for spec in (GEMMA4_E4B, GEMMA4_12B):
            with self.subTest(spec=spec.key):
                observed, modules = self._modules(spec.architecture)
                settings = ResearchTrainerSettings(
                    model_name=spec.model_id,
                    revision=spec.revision,
                    architecture=spec.architecture,
                    output_dir="output",
                    load_in_4bit=False,
                )
                with patch.dict(sys.modules, modules):
                    model = research_cli._load_research_base(settings)
                self.assertEqual(model.config.model_type, spec.architecture)
                self.assertEqual(observed["source"], spec.model_id)
                self.assertEqual(observed["kwargs"]["revision"], spec.revision)

    def test_architecture_mismatch_is_a_clear_failure(self) -> None:
        _, modules = self._modules("gemma4")
        settings = ResearchTrainerSettings(
            model_name=GEMMA4_12B.model_id,
            revision=GEMMA4_12B.revision,
            architecture=GEMMA4_12B.architecture,
            output_dir="output",
            load_in_4bit=False,
        )
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(
            GateError, "architecture mismatch"
        ):
            research_cli._load_research_base(settings)

    def test_unified_language_targets_do_not_expand_to_multimodal_towers(self) -> None:
        class UnifiedGraph:
            def named_modules(self):
                return iter(
                    (
                        ("", self),
                        ("model.language_model.layers.0.self_attn.q_proj", object()),
                        ("model.language_model.layers.0.mlp.down_proj", object()),
                        ("model.embed_vision.patch_dense.q_proj", object()),
                        ("model.embed_audio.projection.down_proj", object()),
                    )
                )

        self.assertEqual(
            resolve_language_lora_targets(UnifiedGraph()),
            (
                "model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.mlp.down_proj",
            ),
        )


class ResearchReloadTests(unittest.TestCase):
    def test_saved_adapter_is_actually_loaded_without_a_digest(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.peft_config = {"default": object()}
                self.evaluated = False

            def eval(self):
                self.evaluated = True

        base = object()
        observed = {}

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, model, path, **kwargs):
                observed.update(model=model, path=path, kwargs=kwargs)
                return FakeModel()

        peft = types.ModuleType("peft")
        peft.PeftModel = FakePeftModel
        with tempfile.TemporaryDirectory() as directory:
            settings = ResearchTrainerSettings(
                model_name="unsloth/gemma-4-E2B-it",
                revision=REVISION,
                output_dir=directory,
            )
            with patch.object(
                research_cli, "_load_research_base", return_value=base
            ), patch.dict(sys.modules, {"peft": peft}):
                report = _reload_saved_adapter(
                    settings,
                    object(),
                    Path(directory),
                    [],
                    canary_count=0,
                )
        self.assertTrue(report["adapter_reloaded"])
        self.assertFalse(report["generation_canary_pass"])
        self.assertTrue(report["fresh_base_reconstructed"])
        self.assertIs(observed["model"], base)
        self.assertFalse(observed["kwargs"]["is_trainable"])

    def test_reload_failure_is_reported_without_deleting_adapter(self) -> None:
        class BrokenPeftModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                raise TypeError("cannot reload")

        peft = types.ModuleType("peft")
        peft.PeftModel = BrokenPeftModel

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "adapter_model.safetensors"
            marker.write_bytes(b"preserve")
            settings = ResearchTrainerSettings(
                model_name="unsloth/gemma-4-E2B-it",
                revision=REVISION,
                output_dir=directory,
            )
            with patch.object(
                research_cli, "_load_research_base", return_value=object()
            ), patch.dict(sys.modules, {"peft": peft}):
                report = _reload_saved_adapter(
                    settings, object(), output, [], canary_count=0
                )
            self.assertFalse(report["adapter_reloaded"])
            self.assertIn("TypeError", report["reload_error"])
            self.assertTrue(marker.is_file())

    def test_reload_canary_requires_parseable_generation_not_exact_free_text(
        self,
    ) -> None:
        class FakeModel:
            peft_config = {"default": object()}

            def eval(self):
                return None

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return FakeModel()

        peft = types.ModuleType("peft")
        peft.PeftModel = FakePeftModel
        expected = ParsedToolCall(
            "ask_for_more_evidence",
            {
                "case_path": "active",
                "request": "operator_escalation:recovery_options_exhausted",
            },
        )
        generated = ParsedToolCall(
            "ask_for_more_evidence",
            {
                "case_path": "active",
                "request": "Please provide additional diagnostic evidence.",
            },
        )
        example = types.SimpleNamespace(expected_tool_call=expected)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            research_cli, "_load_research_base", return_value=object()
        ), patch.object(
            research_cli,
            "generate_single_tool_call",
            return_value=generated,
        ) as generation, patch.dict(sys.modules, {"peft": peft}):
            report = _reload_saved_adapter(
                ResearchTrainerSettings(
                    model_name="unsloth/gemma-4-E2B-it",
                    revision=REVISION,
                    output_dir=directory,
                ),
                object(),
                Path(directory),
                [example],
                canary_count=1,
            )
        self.assertTrue(report["generation_canary_pass"])
        self.assertEqual(
            report["canary_mode"], "parseable_single_tool_call_after_reload"
        )
        self.assertTrue(report["canaries"][0]["target_tool_match"])
        self.assertFalse(report["canaries"][0]["exact_action_match"])
        generation.assert_called_once()

        with tempfile.TemporaryDirectory() as directory, patch.object(
            research_cli, "_load_research_base", return_value=object()
        ), patch.object(
            research_cli,
            "generate_single_tool_call",
            side_effect=GateError("no parseable tool call"),
        ), patch.dict(sys.modules, {"peft": peft}):
            malformed = _reload_saved_adapter(
                ResearchTrainerSettings(
                    model_name="unsloth/gemma-4-E2B-it",
                    revision=REVISION,
                    output_dir=directory,
                ),
                object(),
                Path(directory),
                [example],
                canary_count=1,
            )
        self.assertFalse(malformed["generation_canary_pass"])
        self.assertFalse(malformed["canaries"][0]["passed"])
        self.assertIn("no parseable tool call", malformed["canaries"][0]["error"])

    def test_complete_saved_adapter_can_finalize_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            settings = ResearchTrainerSettings(
                model_name="unsloth/gemma-4-E2B-it",
                revision=REVISION,
                output_dir=directory,
            )
            data_report = {"splits": "checked"}
            stage_settings = asdict(settings)
            stage_settings["resume_from_checkpoint"] = True
            adapter = output / "lora"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}), encoding="utf-8"
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (output / "training_stage.json").write_text(
                json.dumps(
                    {
                        "training_metrics": {"train_loss": 0.5},
                        "adapter_delta": {"changed_tensors": 1},
                        "settings": stage_settings,
                        "lora": asdict(LoraSettings()),
                        "inputs": {
                            "train": str(Path("train.jsonl").resolve()),
                            "validation": str(Path("validation.jsonl").resolve()),
                        },
                        "data": data_report,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                research_cli,
                "prepare_research_examples",
                return_value=(object(), [], [], data_report),
            ), patch.object(
                research_cli,
                "_reload_saved_adapter",
                return_value={
                    "adapter_reloaded": True,
                    "generation_canary_pass": True,
                },
            ) as reload_adapter, patch.object(
                research_cli, "_load_research_base"
            ) as load_base:
                report = run_research_training(
                    train_file="train.jsonl",
                    validation_file="validation.jsonl",
                    settings=settings,
                    reload_canaries=1,
                )
        self.assertTrue(report["passed"])
        self.assertTrue(report["resumed_saved_adapter_finalization"])
        self.assertTrue(
            report["preserved_training_stage"]["settings"][
                "resume_from_checkpoint"
            ]
        )
        self.assertIsNone(report["settings"]["resume_from_checkpoint"])
        reload_adapter.assert_called_once()
        load_base.assert_not_called()


if __name__ == "__main__":
    unittest.main()
