"""Static regression tests for the fail-closed HPC release launcher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from psse_env.dagger.release_launcher import (
    publish_external_release_report,
    validate_release_evaluation_paths,
)


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
            'EVALUATION_REPORT_DIR="${REPO_ROOT%/}_evaluation_reports/$SOURCE_COMMIT"',
            'GATE_REPORT="$EVALUATION_REPORT_DIR/${EVALUATION_BASENAME}.${REPORT_RUN_ID}.gate.json"',
            "EVALUATION_ARTIFACT=${VALIDATED_RELEASE_PATHS[0]}",
            "GATE_REPORT=${VALIDATED_RELEASE_PATHS[1]}",
            "publish_external_release_report",
            "postpublication_revalidate",
            "EVALUATION_SCOPE=${EVALUATION_SCOPE:-frozen_suite}",
            "--seed 20260721",
            "--required-suite dagger1_development",
            "--minimum-roots-per-suite 30",
            "--diagnostic-only",
            "RECOVERY_STRESS_SUITE=${RECOVERY_STRESS_SUITE:-}",
            "RECOVERY_STRESS_MANIFEST=${RECOVERY_STRESS_MANIFEST:-}",
            "--seed 20260723",
            "--required-suite recovery_post_failure_no_candidate",
            "--required-suite recovery_unsupported_correction",
            "--minimum-suites 7",
            "--minimum-episodes-per-suite 10",
            "--minimum-roots-per-suite 10",
            '--recovery-stress-manifest "$RECOVERY_STRESS_MANIFEST"',
            '"recovery_stress": "recovery_stress_evaluation"',
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
        self.assertNotIn(
            "EVALUATION_ARTIFACT=${EVALUATION_ARTIFACT:-", self.launcher
        )

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

        missing_recovery_inputs = subprocess.run(
            ["bash", "-s"],
            cwd=launcher_path.parent,
            env=os.environ,
            input=(
                "EVALUATION_MODE=base\n"
                "EVALUATION_SCOPE=recovery_stress\n"
                f"REVIEWED_SOURCE_COMMIT={'a' * 40}\n"
                + self.launcher
            ).encode("utf-8"),
            check=False,
            capture_output=True,
        )
        self.assertEqual(missing_recovery_inputs.returncode, 2)
        self.assertIn(
            "recovery-stress scope requires",
            missing_recovery_inputs.stderr.decode("utf-8"),
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
        self.report_owner = tempfile.TemporaryDirectory()
        self.root = Path(self.owner.name).resolve()
        self.report_root = Path(self.report_owner.name).resolve()
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
        self.report_owner.cleanup()

    def test_base_outputs_are_confined_to_evidence_directory(self) -> None:
        report_dir = self.report_root
        result = validate_release_evaluation_paths(
            repo_root=self.root,
            mode="base",
            artifact="artifacts/evaluations/base.json",
            report=report_dir / "base.json.gate.json",
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
                artifact="psse_env/dagger/new.json",
                report=report_dir / "other.json.gate.json",
                protected_inputs=self.protected,
            )

    def test_report_must_be_new_absolute_and_outside_repository(self) -> None:
        report_dir = self.report_root
        with self.assertRaisesRegex(ValueError, "absolute path"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="base",
                artifact="artifacts/evaluations/base.json",
                report="relative.gate.json",
                protected_inputs=self.protected,
            )
        with self.assertRaisesRegex(ValueError, "outside the source repository"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="base",
                artifact="artifacts/evaluations/base.json",
                report=self.root / "inside.gate.json",
                protected_inputs=self.protected,
            )
        existing = report_dir / "existing.gate.json"
        existing.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="base",
                artifact="artifacts/evaluations/base.json",
                report=existing,
                protected_inputs=self.protected,
            )

    def test_returned_report_path_is_canonical_through_symlink_parent(self) -> None:
        real_parent = self.report_root / "real"
        real_parent.mkdir()
        alias = self.report_root / "alias"
        try:
            alias.symlink_to(real_parent, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        result = validate_release_evaluation_paths(
            repo_root=self.root,
            mode="base",
            artifact="artifacts/evaluations/../evaluations/base.json",
            report=alias / "study.json",
            protected_inputs=self.protected,
        )
        self.assertEqual(Path(result["report"]), real_parent / "study.json")
        self.assertEqual(
            Path(result["artifact"]),
            self.root / "artifacts" / "evaluations" / "base.json",
        )

    def test_checkpoint_outputs_cannot_overwrite_base_or_checkpoint(self) -> None:
        base = self.root / "artifacts" / "evaluations" / "base.json"
        base.write_text("{}", encoding="utf-8")
        checkpoint = self.root / "outputs" / "round0" / "lora"
        checkpoint.mkdir(parents=True)
        report_dir = self.report_root
        with self.assertRaisesRegex(ValueError, "artifact already exists"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="checkpoint",
                artifact=base,
                report=report_dir / "checkpoint.gate.json",
                protected_inputs=self.protected,
                reference_artifact=base,
                checkpoint_path=checkpoint,
            )

        nested_output = checkpoint / "gate.json"
        with self.assertRaisesRegex(ValueError, "outside the source repository"):
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
        report_dir = self.report_root
        with self.assertRaisesRegex(ValueError, "must not live inside"):
            validate_release_evaluation_paths(
                repo_root=self.root,
                mode="checkpoint",
                artifact="artifacts/evaluations/checkpoint.json",
                report=report_dir / "checkpoint.json.gate.json",
                protected_inputs=self.protected,
                reference_artifact=base,
                checkpoint_path=checkpoint,
            )


class ExternalReleaseReportPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_owner = tempfile.TemporaryDirectory()
        self.report_owner = tempfile.TemporaryDirectory()
        self.repo = Path(self.repo_owner.name).resolve()
        self.report_root = Path(self.report_owner.name).resolve()
        self.protected = self.repo / "protected.json"
        self.protected.write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.repo_owner.cleanup()
        self.report_owner.cleanup()

    def _publish(self, target: Path, callback: object = None) -> dict[str, str | int]:
        revalidate = callback if callable(callback) else lambda: None
        return publish_external_release_report(
            repo_root=self.repo,
            report=target,
            rendered='{"passed": true}\n',
            protected_inputs=(self.protected,),
            postpublication_revalidate=revalidate,
        )

    def test_publishes_exact_verified_read_only_bytes(self) -> None:
        target = self.report_root / "study.json"
        calls: list[str] = []
        result = self._publish(target, lambda: calls.append("checked"))
        expected = b'{"passed": true}\n'
        self.assertEqual(calls, ["checked"])
        self.assertEqual(target.read_bytes(), expected)
        self.assertEqual(
            result["sha256"], hashlib.sha256(expected).hexdigest()
        )
        self.assertEqual(result["path"], str(target))
        if os.name == "posix":
            self.assertEqual(target.stat().st_mode & 0o777, 0o400)

    def test_existing_target_is_preserved(self) -> None:
        target = self.report_root / "study.json"
        target.write_text("preserve\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self._publish(target)
        self.assertEqual(target.read_text(encoding="utf-8"), "preserve\n")

    def test_revalidation_failure_removes_exact_new_report(self) -> None:
        target = self.report_root / "study.json"

        def fail() -> None:
            raise ValueError("source changed")

        with self.assertRaisesRegex(ValueError, "source changed"):
            self._publish(target, fail)
        self.assertFalse(target.exists())

    def test_changed_report_after_callback_is_removed(self) -> None:
        target = self.report_root / "study.json"

        def mutate() -> None:
            if os.name == "posix":
                target.chmod(0o600)
            target.write_text("changed\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "report changed"):
            self._publish(target, mutate)
        self.assertFalse(target.exists())

    def test_failed_safe_removal_is_explicit(self) -> None:
        target = self.report_root / "study.json"

        def fail() -> None:
            raise ValueError("source changed")

        with (
            mock.patch(
                "psse_env.dagger.release_launcher.evaluation_gate._unlink_created_report",
                return_value=False,
            ),
            self.assertRaisesRegex(
                ValueError,
                "newly created report could not be safely removed",
            ),
        ):
            self._publish(target, fail)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
