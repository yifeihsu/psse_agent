"""Focused contracts for immutable Round-1 LoRA warm-start training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from psse_env.sft.gates import GateError
from psse_env.sft.training import (
    TrainerSettings,
    _restore_trainable_parameters,
    _snapshot_trainable_parameters,
)


PINNED_MODEL_REVISION = "a" * 40
PINNED_ADAPTER_REVISION = "b" * 64
REPO_ROOT = Path(__file__).resolve().parents[3]


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


class TestWarmStartLoadAndRestore(unittest.TestCase):


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
