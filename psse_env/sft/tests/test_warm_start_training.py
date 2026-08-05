"""Focused contracts for immutable Round-1 LoRA warm-start training."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from psse_env.sft.cli import main as cli_main
from psse_env.sft.gates import GateError
from psse_env.sft.training import (
    TrainerSettings,
    _attach_trainable_adapter,
    _inspect_initial_adapter,
    _load_trainable_initial_adapter,
    _restore_trainable_parameters,
    _snapshot_trainable_parameters,
    _write_initial_adapter_attestation,
)


PINNED_MODEL_REVISION = "a" * 40
PINNED_ADAPTER_REVISION = "b" * 64
REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "submit_dagger_sft_round0.sh"


class TestWarmStartSettings(unittest.TestCase):
    def test_initial_adapter_identity_is_both_or_neither(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid = TrainerSettings(
                revision=PINNED_MODEL_REVISION,
                output_dir=str(root / "output"),
                initial_adapter_path=str(root / "input"),
                initial_adapter_revision=PINNED_ADAPTER_REVISION,
            )
            valid.validate()

            invalid = (
                TrainerSettings(
                    revision=PINNED_MODEL_REVISION,
                    output_dir=str(root / "output"),
                    initial_adapter_path=str(root / "input"),
                ),
                TrainerSettings(
                    revision=PINNED_MODEL_REVISION,
                    output_dir=str(root / "output"),
                    initial_adapter_revision=PINNED_ADAPTER_REVISION,
                ),
                TrainerSettings(
                    revision=PINNED_MODEL_REVISION,
                    output_dir=str(root / "output"),
                    initial_adapter_path="relative/adapter",
                    initial_adapter_revision=PINNED_ADAPTER_REVISION,
                ),
                TrainerSettings(
                    revision=PINNED_MODEL_REVISION,
                    output_dir=str(root / "output"),
                    initial_adapter_path=str(root / "input"),
                    initial_adapter_revision="not-a-tree-hash",
                ),
            )
            for settings in invalid:
                with self.subTest(settings=settings):
                    with self.assertRaises(GateError):
                        settings.validate()

    def test_output_and_initial_adapter_must_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pairs = (
                (root / "adapter", root / "adapter"),
                (root / "adapter" / "new-output", root / "adapter"),
                (root / "output", root / "output" / "adapter"),
            )
            for output, initial in pairs:
                with self.subTest(output=output, initial=initial):
                    settings = TrainerSettings(
                        revision=PINNED_MODEL_REVISION,
                        output_dir=str(output),
                        initial_adapter_path=str(initial),
                        initial_adapter_revision=PINNED_ADAPTER_REVISION,
                    )
                    with self.assertRaisesRegex(GateError, "must not overlap"):
                        settings.validate()

    def test_tree_digest_mismatch_fails_preflight(self) -> None:
        settings = TrainerSettings(
            revision=PINNED_MODEL_REVISION,
            output_dir="/scratch/test/output",
            initial_adapter_path="/scratch/test/input",
            initial_adapter_revision=PINNED_ADAPTER_REVISION,
        )
        with mock.patch(
            "psse_env.dagger.release_factories.inspect_release_checkpoint",
            return_value={"tree_sha256": "c" * 64},
        ):
            with self.assertRaisesRegex(GateError, "tree digest mismatch"):
                _inspect_initial_adapter(settings)


class TestWarmStartLoadAndRestore(unittest.TestCase):
    def test_cold_start_still_attaches_a_new_lora(self) -> None:
        model = object()
        settings = TrainerSettings(revision=PINNED_MODEL_REVISION)
        attached = object()
        with mock.patch(
            "psse_env.sft.training._attach_lora",
            return_value=attached,
        ) as attach:
            result, attestation = _attach_trainable_adapter(
                model,
                settings,
                SimpleNamespace(),
            )

        self.assertIs(result, attached)
        self.assertIsNone(attestation)
        attach.assert_called_once_with(model, settings, mock.ANY)

    def test_initial_adapter_is_loaded_trainable_from_verified_private_copy(
        self,
    ) -> None:
        import torch

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adapter = torch.nn.Parameter(torch.ones(2))
                self.config = SimpleNamespace(use_cache=True)

        captured: dict[str, object] = {}

        class FakePeftModel:
            @staticmethod
            def from_pretrained(model, path, **kwargs):
                captured["path"] = path
                captured["kwargs"] = kwargs
                return model

        owner = SimpleNamespace(cleanup=mock.Mock())
        inspection = {
            "path": "/scratch/input/lora",
            "tree_sha256": PINNED_ADAPTER_REVISION,
            "file_count": 2,
            "total_bytes": 1234,
        }
        settings = TrainerSettings(
            revision=PINNED_MODEL_REVISION,
            output_dir="/scratch/output",
            initial_adapter_path="/scratch/input/lora",
            initial_adapter_revision=PINNED_ADAPTER_REVISION,
        )
        fake_peft = SimpleNamespace(
            PeftModel=FakePeftModel,
            prepare_model_for_kbit_training=lambda model: model,
        )
        with (
            mock.patch.dict(sys.modules, {"peft": fake_peft}),
            mock.patch(
                "psse_env.dagger.release_factories._copy_verified_checkpoint_tree",
                return_value=(Path("/private/adapter"), owner),
            ),
            mock.patch(
                "psse_env.dagger.release_factories.checkpoint_tree_sha256",
                return_value=PINNED_ADAPTER_REVISION,
            ),
        ):
            model, attestation = _load_trainable_initial_adapter(
                Model(),
                settings,
                inspection,
            )

        self.assertEqual(captured["path"], "/private/adapter")
        self.assertEqual(
            captured["kwargs"],
            {"is_trainable": True, "local_files_only": True},
        )
        self.assertFalse(model.config.use_cache)
        self.assertEqual(attestation["tree_sha256"], PINNED_ADAPTER_REVISION)
        self.assertTrue(attestation["peft_load"]["is_trainable"])
        owner.cleanup.assert_called_once_with()

    def test_smoke_mutation_is_restored_exactly(self) -> None:
        import torch

        model = torch.nn.Linear(2, 1, bias=False)
        original = _snapshot_trainable_parameters(model)
        with torch.no_grad():
            model.weight.add_(10)
        model.weight.grad = torch.ones_like(model.weight)

        report = _restore_trainable_parameters(model, original)

        self.assertTrue(torch.equal(model.weight.detach().cpu(), original["weight"]))
        self.assertIsNone(model.weight.grad)
        self.assertEqual(report["restored_parameter_tensors"], 1)
        self.assertEqual(report["restored_parameter_elements"], 2)

    def test_attestation_records_identity_source_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output"
            settings = TrainerSettings(
                revision=PINNED_MODEL_REVISION,
                output_dir=str(output),
            )
            with mock.patch(
                "psse_env.sft.training.git_source_state",
                return_value={
                    "source_commit": "d" * 40,
                    "release_eligible_source": True,
                },
            ):
                path = _write_initial_adapter_attestation(
                    settings=settings,
                    adapter_attestation={
                        "attestation_schema_version": 1,
                        "initial_adapter_path": "/scratch/input/lora",
                        "initial_adapter_revision": PINNED_ADAPTER_REVISION,
                        "tree_sha256": PINNED_ADAPTER_REVISION,
                    },
                    smoke_restore={"performed": True},
                    base_snapshot_attestation={
                        "model_id": "unsloth/gemma-4-31B-it",
                        "model_revision": PINNED_MODEL_REVISION,
                    },
                )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "initial_adapter_attestation.json")
        self.assertEqual(payload["tree_sha256"], PINNED_ADAPTER_REVISION)
        self.assertTrue(payload["smoke_restore"]["performed"])
        self.assertEqual(payload["training_source"]["source_commit"], "d" * 40)
        self.assertFalse(payload["output_input_overlap"])
        self.assertEqual(payload["training_configuration"]["epochs"], 1.0)


class TestWarmStartCliAndLauncher(unittest.TestCase):
    def test_train_cli_forwards_initial_adapter_identity_and_wandb(self) -> None:
        result_object = SimpleNamespace(metrics={"train_loss": 1.0})
        with (
            mock.patch(
                "psse_env.sft.cli._baseline_evaluation_gate",
                return_value={},
            ),
            mock.patch(
                "psse_env.sft.cli.run_lora_training",
                return_value=result_object,
            ) as run_training,
        ):
            result = cli_main(
                [
                    "train",
                    "--revision",
                    PINNED_MODEL_REVISION,
                    "--train",
                    "train.jsonl",
                    "--validation",
                    "validation.jsonl",
                    "--initial-adapter-path",
                    "/scratch/input/lora",
                    "--initial-adapter-revision",
                    PINNED_ADAPTER_REVISION,
                    "--report-to",
                    "wandb",
                    "--run-name",
                    "bc0-round1",
                ]
            )

        self.assertEqual(result, 0)
        settings = run_training.call_args.kwargs["settings"]
        self.assertEqual(settings.initial_adapter_path, "/scratch/input/lora")
        self.assertEqual(
            settings.initial_adapter_revision,
            PINNED_ADAPTER_REVISION,
        )
        self.assertEqual(settings.report_to, "wandb")
        self.assertEqual(settings.run_name, "bc0-round1")

    def test_smoke_cli_optionally_forwards_initial_adapter_identity(self) -> None:
        smoke_result = SimpleNamespace(to_dict=lambda: {"passed": True})
        with mock.patch(
            "psse_env.sft.cli.run_lora_smoke",
            return_value=smoke_result,
        ) as run_smoke:
            result = cli_main(
                [
                    "smoke",
                    "--revision",
                    PINNED_MODEL_REVISION,
                    "--train",
                    "train.jsonl",
                    "--validation",
                    "validation.jsonl",
                    "--mode",
                    "one-batch",
                    "--initial-adapter-path",
                    "/scratch/input/lora",
                    "--initial-adapter-revision",
                    PINNED_ADAPTER_REVISION,
                ]
            )

        self.assertEqual(result, 0)
        settings = run_smoke.call_args.kwargs["settings"]
        self.assertEqual(settings.initial_adapter_path, "/scratch/input/lora")
        self.assertEqual(
            settings.initial_adapter_revision,
            PINNED_ADAPTER_REVISION,
        )

    def test_launcher_exposes_bounded_round1_defaults_and_identity(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        for contract in (
            "gate|one-batch|targeted-tiny-overfit|tiny-overfit|round0|round1|checkpoint-gate",
            "ROUND1_LR=${ROUND1_LR:-0.00003}",
            "ROUND1_EPOCHS=${ROUND1_EPOCHS:-1}",
            "TRAIN_LR=${TRAIN_LR:-0.0001}",
            "TRAIN_EPOCHS=${TRAIN_EPOCHS:-2}",
            "INITIAL_ADAPTER_PATH=${INITIAL_ADAPTER_PATH:-}",
            "INITIAL_ADAPTER_REVISION=${INITIAL_ADAPTER_REVISION:-}",
            '--initial-adapter-path "$INITIAL_ADAPTER_PATH"',
            '--initial-adapter-revision "$INITIAL_ADAPTER_REVISION"',
            '--learning-rate "$ROUND1_LR"',
            '--epochs "$ROUND1_EPOCHS"',
            "OUTPUT_DIR and INITIAL_ADAPTER_PATH must not overlap",
        ):
            self.assertIn(contract, launcher)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
