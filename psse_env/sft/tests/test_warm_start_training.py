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
                round1_provenance_path=str(
                    root / "aggregate.generation_provenance.json"
                ),
                round1_preflight_path=str(root / "aggregate.preflight.json"),
                reviewed_source_commit="c" * 40,
                round1_view="full",
            )
            valid.validate()

            warm_start_without_source = TrainerSettings(
                revision=PINNED_MODEL_REVISION,
                output_dir=str(root / "output-without-source"),
                initial_adapter_path=str(root / "input"),
                initial_adapter_revision=PINNED_ADAPTER_REVISION,
            )
            with self.assertRaisesRegex(GateError, "complete Round-1 source"):
                warm_start_without_source.validate()

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

    def test_round1_source_binding_is_complete_and_requires_warm_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "revision": PINNED_MODEL_REVISION,
                "output_dir": str(root / "output"),
                "round1_provenance_path": str(root / "provenance.json"),
                "round1_preflight_path": str(root / "preflight.json"),
                "reviewed_source_commit": "c" * 40,
                "round1_view": "full",
            }
            valid = TrainerSettings(
                **common,
                initial_adapter_path=str(root / "adapter"),
                initial_adapter_revision=PINNED_ADAPTER_REVISION,
            )
            valid.validate()

            missing_adapter = TrainerSettings(**common)
            with self.assertRaisesRegex(GateError, "requires an immutable initial adapter"):
                missing_adapter.validate()

            partial = TrainerSettings(
                revision=PINNED_MODEL_REVISION,
                round1_provenance_path=str(root / "provenance.json"),
            )
            with self.assertRaisesRegex(GateError, "must be supplied together"):
                partial.validate()

    def test_study_variant_and_round1_view_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "revision": PINNED_MODEL_REVISION,
                "output_dir": str(root / "output"),
                "initial_adapter_path": str(root / "adapter"),
                "initial_adapter_revision": PINNED_ADAPTER_REVISION,
                "parent_checkpoint_receipt_path": str(
                    root / "checkpoint_receipt.json"
                ),
                "round1_provenance_path": str(root / "provenance.json"),
                "round1_preflight_path": str(root / "preflight.json"),
                "reviewed_source_commit": "c" * 40,
            }
            for variant, approved, mismatched in (
                ("natural_dagger", "natural-only", "full"),
                ("natural_dagger_probes", "full", "natural-only"),
            ):
                with self.subTest(variant=variant, view=approved):
                    TrainerSettings(
                        **common,
                        study_variant=variant,
                        round1_view=approved,
                    ).validate()
                with self.subTest(variant=variant, view=mismatched):
                    with self.assertRaisesRegex(
                        GateError,
                        f"requires round1_view={approved}",
                    ):
                        TrainerSettings(
                            **common,
                            study_variant=variant,
                            round1_view=mismatched,
                        ).validate()

    def test_round1_study_variant_requires_parent_receipt_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = TrainerSettings(
                revision=PINNED_MODEL_REVISION,
                output_dir=str(root / "output"),
                initial_adapter_path=str(root / "adapter"),
                initial_adapter_revision=PINNED_ADAPTER_REVISION,
                round1_provenance_path=str(root / "provenance.json"),
                round1_preflight_path=str(root / "preflight.json"),
                reviewed_source_commit="c" * 40,
                round1_view="natural-only",
                study_variant="natural_dagger",
            )
            with self.assertRaisesRegex(
                GateError,
                "same-seed BC0 parent checkpoint receipt",
            ):
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

        self.assertEqual(captured["path"], str(Path("/private/adapter")))
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
                    "--seed",
                    "3407",
                    "--study-variant",
                    "natural_dagger_probes",
                    "--initial-adapter-path",
                    "/scratch/input/lora",
                    "--initial-adapter-revision",
                    PINNED_ADAPTER_REVISION,
                    "--parent-checkpoint-receipt",
                    "/scratch/input/checkpoint_receipt.json",
                    "--round1-provenance",
                    "aggregate.generation_provenance.json",
                    "--round1-preflight",
                    "aggregate.preflight.json",
                    "--reviewed-source-commit",
                    "c" * 40,
                    "--round1-view",
                    "full",
                    "--report-to",
                    "wandb",
                    "--run-name",
                    "bc0-round1",
                ]
            )

        self.assertEqual(result, 0)
        settings = run_training.call_args.kwargs["settings"]
        self.assertEqual(
            settings.initial_adapter_path, str(Path("/scratch/input/lora"))
        )
        self.assertEqual(
            settings.initial_adapter_revision,
            PINNED_ADAPTER_REVISION,
        )
        self.assertEqual(
            settings.parent_checkpoint_receipt_path,
            str(Path("/scratch/input/checkpoint_receipt.json")),
        )
        self.assertEqual(settings.report_to, "wandb")
        self.assertEqual(settings.run_name, "bc0-round1")
        self.assertEqual(
            settings.round1_provenance_path,
            "aggregate.generation_provenance.json",
        )
        self.assertEqual(settings.round1_preflight_path, "aggregate.preflight.json")
        self.assertEqual(settings.reviewed_source_commit, "c" * 40)
        self.assertEqual(settings.round1_view, "full")

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
        self.assertEqual(
            settings.initial_adapter_path, str(Path("/scratch/input/lora"))
        )
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
            "PARENT_CHECKPOINT_RECEIPT=${PARENT_CHECKPOINT_RECEIPT:-}",
            '--initial-adapter-path "$INITIAL_ADAPTER_PATH"',
            '--initial-adapter-revision "$INITIAL_ADAPTER_REVISION"',
            '--parent-checkpoint-receipt "$PARENT_CHECKPOINT_RECEIPT"',
            '--round1-provenance "$ROUND1_PROVENANCE"',
            '--round1-preflight "$ROUND1_PREFLIGHT"',
            '--round1-view "$ROUND1_VIEW"',
            '--learning-rate "$ROUND1_LR"',
            '--epochs "$ROUND1_EPOCHS"',
            "OUTPUT_DIR and INITIAL_ADAPTER_PATH must not overlap",
            "psse_env.sft.round1_source_gate",
            '--reviewed-source-commit "$REVIEWED_SOURCE_COMMIT"',
            '--initial-adapter-revision "$INITIAL_ADAPTER_REVISION"',
            "ROUND1_SEED_COUPLING_REQUIRED=0",
            'if [[ "$ROUND1_SEED_COUPLING_REQUIRED" == "1" ]]',
            "gate|one-batch|targeted-tiny-overfit|tiny-overfit)",
            "one-batch|targeted-tiny-overfit|tiny-overfit",
            "ROUND1_GATE_SOURCE_ARGS=()",
            '"${ROUND1_GATE_SOURCE_ARGS[@]}" --test "$TEST_FILE"',
            "ROUND1_VIEW=${ROUND1_VIEW:-}",
            "aggregate.natural_only.train_view.jsonl",
            "STUDY_VARIANT=natural_dagger requires ROUND1_VIEW=natural-only",
            "STUDY_VARIANT=natural_dagger_probes requires ROUND1_VIEW=full",
            "requires canonical TRAIN_FILE=$EXPECTED_ROUND1_TRAIN_FILE",
            "requires PARENT_CHECKPOINT_RECEIPT for the same-seed BC0 adapter",
            "PARENT_CHECKPOINT_RECEIPT must be the canonical sibling",
            "PINNED_DEPENDENCY_LOCK_SHA256=4118e4bb6c7b7e4fa806afb33aa0689a594ff276fcb203c8aba015bb70246fea",
            "study training requires MAX_LENGTH=6144 and GRADIENT_ACCUMULATION_STEPS=4",
            "BC0 protocol requires TRAIN_LR=0.0001 and TRAIN_EPOCHS=2",
            "Round-1 protocol requires ROUND1_LR=0.00003 and ROUND1_EPOCHS=1",
            "--optimizer adamw_torch",
            "--lr-scheduler-type linear",
            "--lora-rank 16",
            "--lora-alpha 16",
            "--lora-dropout 0.0",
            "--max-steps -1",
        ):
            self.assertIn(contract, launcher)

    def test_launcher_delegates_round1_provenance_gate(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        for contract in (
            "release aggregate checksum manifest is missing",
            "sha256sum --check --quiet SHA256SUMS",
            '"$PYTHON" -m psse_env.sft.round1_source_gate',
            '--provenance "$ROUND1_PROVENANCE"',
            '--preflight "$ROUND1_PREFLIGHT"',
            '--reviewed-source-commit "$REVIEWED_SOURCE_COMMIT"',
            '--initial-adapter-revision "$INITIAL_ADAPTER_REVISION"',
            '--round1-view "$ROUND1_VIEW"',
            '--round1-provenance "$ROUND1_PROVENANCE"',
            '--round1-preflight "$ROUND1_PREFLIGHT"',
        ):
            self.assertIn(contract, launcher)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
