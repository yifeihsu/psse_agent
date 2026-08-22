"""Static regression tests for the fail-closed HPC release launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from psse_env.dagger.release_launcher import validate_release_evaluation_paths


class ReleaseEvaluationLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = (
            Path(__file__).resolve().parents[2] / "submit_dagger_release_eval.sh"
        ).read_text(encoding="utf-8")

    def test_targets_reviewed_offline_hardware_environment(self) -> None:
        for contract in (
            "#SBATCH --gres=gpu:1",
            '#SBATCH --constraint="h200|h100|rtx6000"',
            '#SBATCH --comment="preemption=yes;requeue=true"',
            "--query-gpu=name,memory.total,driver_version",
            "psse_env.sft.release_hardware",
            "EXPECTED_ACCELERATOR_CLASS=${EXPECTED_ACCELERATOR_CLASS:-auto}",
            '--require-class "$EXPECTED_ACCELERATOR_CLASS"',
            "sbatch --constraint=rtx6000 --cpus-per-task=4 --mem=128G",
            'echo "gpu:       $GPU_INVENTORY"',
            "sys.version_info[:2] != (3, 12)",
            'torch.__version__ != "2.10.0+cu128"',
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export HF_DATASETS_OFFLINE=1",
            "export PIP_NO_INDEX=1",
        ):
            self.assertIn(contract, self.launcher)
        self.assertNotIn("--allow-dirty-source", self.launcher)
        self.assertNotIn("--allow-download", self.launcher)
        self.assertNotIn("$GPU_NAMES", self.launcher)

    def test_invokes_exact_release_factories_and_suite_contract(self) -> None:
        for contract in (
            "production_environment_factory",
            "gemma_release_policy_factory",
            "deterministic_case_loader",
            "--seed 20260719",
            "--max-steps 24",
            "--minimum-suites 5",
            "--minimum-roots-per-suite 20",
            "--expected-source-commit \"$SOURCE_COMMIT\"",
            "EVALUATION_SCOPE=${EVALUATION_SCOPE:-frozen_suite}",
            "--seed 20260721",
            "--required-suite dagger1_development",
            "--minimum-roots-per-suite 30",
            "--diagnostic-only",
        ):
            self.assertIn(contract, self.launcher)

    def test_modes_gate_immutable_model_identities(self) -> None:
        for contract in (
            "ROLE=expert-baseline",
            "ROLE=base-baseline",
            "ROLE=checkpoint-promotion",
            "EXPERT_POLICY_IDENTITY=bc0-observable-handoff-expert-v2",
            'EVALUATE+=(--policy-identity "$EXPERT_POLICY_IDENTITY")',
            'GATE+=(--expected-policy-identity "$EXPERT_POLICY_IDENTITY")',
            "inspect_release_checkpoint",
            "CHECKPOINT_PATH\" != /*",
            "^[0-9a-fA-F]{64}$",
            "REVIEWED_SOURCE_COMMIT",
            'SOURCE_COMMIT\" != \"$REVIEWED_SOURCE_COMMIT',
            "--reference-artifact \"$REFERENCE_ARTIFACT\"",
            "--reference-model-id \"$BASE_MODEL_ID\"",
            "--reference-model-revision \"$BASE_MODEL_REVISION\"",
            "--study-manifest \"$STUDY_MANIFEST\"",
            "--study-variant \"$STUDY_VARIANT\"",
            "--reviewed-source-commit \"$SOURCE_COMMIT\"",
            "--training-seed \"$TRAIN_SEED\"",
            "--checkpoint-receipt \"$CHECKPOINT_RECEIPT\"",
            "extract_artifact_metrics",
        ):
            self.assertIn(contract, self.launcher)
        self.assertIn(
            "BASE_EVALUATION_ARTIFACT=artifacts/evaluations/base_gemma_evaluation.json",
            self.launcher,
        )
        self.assertNotIn("BASE_EVALUATION_ARTIFACT=${", self.launcher)
        self.assertNotIn("EVALUATION_ARTIFACT=${", self.launcher)

    def test_study_identity_is_fail_closed_before_hpc_setup(self) -> None:
        launcher_path = Path(__file__).resolve().parents[2] / "submit_dagger_release_eval.sh"
        base_with_seed = subprocess.run(
            ["bash", "-s"],
            cwd=launcher_path.parent,
            env=os.environ,
            input=(
                "EVALUATION_MODE=base\nTRAIN_SEED=3407\n"
                f"REVIEWED_SOURCE_COMMIT={'a' * 40}\n"
                + self.launcher
            ).encode("utf-8"),
            check=False,
            capture_output=True,
        )
        self.assertEqual(base_with_seed.returncode, 2)
        self.assertIn("canonical null seed", base_with_seed.stderr.decode("utf-8"))

        missing_receipt = subprocess.run(
            ["bash", "-s"],
            cwd=launcher_path.parent,
            env=os.environ,
            input=(
                "EVALUATION_MODE=checkpoint\n"
                "CHECKPOINT_PATH=/tmp/checkpoint/lora\n"
                "STUDY_VARIANT=bc0\nTRAIN_SEED=3407\n"
                "CHECKPOINT_RECEIPT=\n"
                f"REVIEWED_SOURCE_COMMIT={'a' * 40}\n"
                + self.launcher
            ).encode("utf-8"),
            check=False,
            capture_output=True,
        )
        self.assertEqual(missing_receipt.returncode, 2)
        self.assertIn(
            "absolute CHECKPOINT_RECEIPT", missing_receipt.stderr.decode("utf-8")
        )

        missing_development_inputs = subprocess.run(
            ["bash", "-s"],
            cwd=launcher_path.parent,
            env=os.environ,
            input=(
                "EVALUATION_MODE=base\n"
                "EVALUATION_SCOPE=development_holdout\n"
                f"REVIEWED_SOURCE_COMMIT={'a' * 40}\n"
                + self.launcher
            ).encode("utf-8"),
            check=False,
            capture_output=True,
        )
        self.assertEqual(missing_development_inputs.returncode, 2)
        self.assertIn(
            "development scope requires",
            missing_development_inputs.stderr.decode("utf-8"),
        )

    def test_invalid_mode_and_missing_reviewed_commit_fail_before_hpc_setup(self) -> None:
        launcher_path = Path(__file__).resolve().parents[2] / "submit_dagger_release_eval.sh"
        base_environment = {**os.environ, "EVALUATION_MODE": "invalid"}
        invalid = subprocess.run(
            ["bash", "-s"],
            cwd=launcher_path.parent,
            env=base_environment,
            input=("EVALUATION_MODE=invalid\n" + self.launcher).encode("utf-8"),
            check=False,
            capture_output=True,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn(
            "must be expert, base, or checkpoint",
            invalid.stderr.decode("utf-8"),
        )

        missing_review = subprocess.run(
            ["bash", "-s"],
            cwd=launcher_path.parent,
            env={
                **os.environ,
                "EVALUATION_MODE": "base",
                "REVIEWED_SOURCE_COMMIT": "",
            },
            input=(
                "EVALUATION_MODE=base\nREVIEWED_SOURCE_COMMIT=\n"
                + self.launcher
            ).encode("utf-8"),
            check=False,
            capture_output=True,
        )
        self.assertEqual(missing_review.returncode, 2)
        self.assertIn(
            "externally reviewed",
            missing_review.stderr.decode("utf-8"),
        )


class ReleaseEvaluationPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = tempfile.TemporaryDirectory()
        self.root = Path(self.owner.name).resolve()
        (self.root / "artifacts" / "evaluations").mkdir(parents=True)
        (self.root / "psse_env" / "dagger").mkdir(parents=True)
        self.protected = []
        for relative in (
            "psse_env/dagger/suite.json",
            "psse_env/dagger/policy.json",
            "psse_env/requirements-sft.txt",
            "psse_env/dagger/release_factories.py",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
            self.protected.append(relative)

    def tearDown(self) -> None:
        self.owner.cleanup()

    def test_base_outputs_are_confined_to_evidence_directory(self) -> None:
        result = validate_release_evaluation_paths(
            repo_root=self.root,
            mode="base",
            artifact="artifacts/evaluations/base.json",
            report="artifacts/evaluations/base.json.gate.json",
            protected_inputs=self.protected,
        )
        self.assertEqual(
            Path(result["artifact"]),
            self.root / "artifacts" / "evaluations" / "base.json",
        )
        with self.assertRaisesRegex(ValueError, "fixed evidence directory"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="base",
                artifact="psse_env/dagger/suite.json",
                report="artifacts/evaluations/base.json.gate.json",
                protected_inputs=self.protected,
            )

    def test_checkpoint_outputs_cannot_overwrite_base_or_checkpoint(self) -> None:
        base = self.root / "artifacts" / "evaluations" / "base.json"
        base.write_text("{}", encoding="utf-8")
        checkpoint = self.root / "outputs" / "round0" / "lora"
        checkpoint.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "protected evidence"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="checkpoint",
                artifact="artifacts/evaluations/checkpoint.json",
                report=base,
                protected_inputs=self.protected,
                reference_artifact=base,
                checkpoint_path=checkpoint,
            )

        nested_output = checkpoint / "gate.json"
        with self.assertRaisesRegex(ValueError, "fixed evidence directory"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="checkpoint",
                artifact="artifacts/evaluations/checkpoint.json",
                report=nested_output,
                protected_inputs=self.protected,
                reference_artifact=base,
                checkpoint_path=checkpoint,
            )

    def test_checkpoint_cannot_live_inside_evidence_directory(self) -> None:
        base = self.root / "artifacts" / "evaluations" / "base.json"
        base.write_text("{}", encoding="utf-8")
        checkpoint = self.root / "artifacts" / "evaluations" / "adapter"
        checkpoint.mkdir()
        with self.assertRaisesRegex(ValueError, "must not live inside"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="checkpoint",
                artifact="artifacts/evaluations/checkpoint.json",
                report="artifacts/evaluations/checkpoint.json.gate.json",
                protected_inputs=self.protected,
                reference_artifact=base,
                checkpoint_path=checkpoint,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
