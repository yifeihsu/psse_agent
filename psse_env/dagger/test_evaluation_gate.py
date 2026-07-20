from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from psse_env.dagger.evaluation_gate import (
    DEFAULT_POLICY_ID,
    EvaluationGateResult,
    current_registry_sha256,
    main as gate_main,
    validate_evaluation_artifact,
)
from psse_env.sft.cli import main as sft_main
from psse_env.sft.provenance import file_sha256, stable_json_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
SUITE_SHA = "b" * 64
MODEL_REVISION = "c" * 40


def _policy() -> dict:
    source_hash = file_sha256(Path(__file__))
    return {
        "policy_schema_version": 1,
        "policy_id": "test-hard-gate-v1",
        "approved_factories": {
            "environment": [
                {
                    "import_spec": "tests:environment",
                    "source_sha256": source_hash,
                }
            ],
            "expert_policy": [
                {
                    "import_spec": "tests:policy",
                    "source_sha256": source_hash,
                }
            ],
            "model_policy": [
                {
                    "import_spec": "tests:policy",
                    "source_sha256": source_hash,
                }
            ],
            "case_loader": [],
        },
        "hard_constraints": {
            "maximum_false_commit_count": 0,
            "maximum_false_finalization_count": 0,
            "maximum_false_rollback_count": 0,
            "maximum_healthy_component_corruption_episodes": 0,
            "maximum_invalid_action_rate": 0.2,
            "maximum_invalid_actions_per_episode": 1,
            "maximum_loop_episode_rate": 0.05,
            "maximum_steps_per_episode": 8,
            "minimum_terminal_rate": 1.0,
        },
        "family_policy": {
            "no_error": {
                "minimum_physical_roots": 1,
                "minimum_resolution_rate": 1.0,
                "maximum_operator_escalation_rate": 0.0,
            }
        },
    }


def _source_descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "location": "repository",
        "sha256": file_sha256(path),
    }


def _artifact(
    *,
    count: int = 1,
    explicit_identity: str | None = "expert-v1",
    model_id: str | None = None,
    model_revision: str | None = None,
    invalid_episode: int | None = None,
    false_commit_episode: int | None = None,
    duplicate_root: bool = False,
) -> dict:
    source_file = Path(__file__)
    evaluator_file = REPO_ROOT / "psse_env/dagger/evaluator.py"
    episodes = []
    policy_identity = {
        "explicit_policy_identity": explicit_identity,
        "model_id": model_id,
        "model_revision": model_revision,
    }
    for index in range(count):
        invalid = 1 if invalid_episode == index else 0
        false_commit = 1 if false_commit_episode == index else 0
        root = "root-0" if duplicate_root else f"root-{index}"
        episodes.append(
            {
                "episode_key": f"standard_success:case-{index}:0",
                "family": "no_error",
                "physical_root": root,
                "steps": 2,
                "terminal": True,
                "terminal_outcome": "resolved",
                "final_physical_success": True,
                "physical_correctness_known": True,
                "final_physical_correct": True,
                "healthy_preservation_known": True,
                "healthy_components_preserved": True,
                "false_commit_count": false_commit,
                "false_rollback_count": 0,
                "false_finalization_count": 0,
                "invalid_action_count": invalid,
                "loop_detected": False,
                "evaluator_error": None,
                "release_environment_attestation": {
                    "passed": True,
                    "production_dataset_mode": True,
                    "candidate_quality_oracle_mode": "deployment",
                    "failures": [],
                },
                "policy_identity_attestation": {
                    "passed": True,
                    "required": copy.deepcopy(policy_identity),
                    "actual": copy.deepcopy(policy_identity),
                    "failures": [],
                },
                "audit": {
                    "audit_mode": "strict_release_audit",
                    "quarantined": False,
                    "strict_release_audit": {
                        "audit_version": "strict_offline_episode_truth_v3",
                        "terminal": True,
                        "terminal_outcome": "resolved",
                        "scenario_family": "no_error",
                        "physical_root_fingerprint": root,
                        "problems": [],
                        "quarantined": False,
                        "checks": {
                            "accepted_correction_targets": {"status": "passed"},
                            "healthy_measurements_preserved": {"status": "passed"},
                            "healthy_case_components_preserved": {"status": "passed"},
                            "accepted_target_nonregression": {"status": "passed"},
                            "remaining_true_faults": {"status": "passed"},
                            "final_measurements_match_clean": {"status": "passed"},
                            "final_case_matches_clean": {"status": "passed"},
                        },
                    },
                },
            }
        )
    invalid_count = sum(row["invalid_action_count"] for row in episodes)
    false_commit_count = sum(row["false_commit_count"] for row in episodes)
    overall = {
        "episodes": count,
        "terminal_episodes": count,
        "terminal_rate": 1.0,
        "resolved_episodes": count,
        "resolution_rate": 1.0,
        "operator_escalation_episodes": 0,
        "operator_escalation_rate": 0.0,
        "false_commit_count": false_commit_count,
        "false_rollback_count": 0,
        "false_finalization_count": 0,
        "healthy_component_corruption_episodes": 0,
        "invalid_action_count": invalid_count,
        "loop_episodes": 0,
        "evaluator_error_episodes": 0,
    }
    suite_hashes = {"standard_success": "d" * 64}
    environment_rows = [
        {
            "production_dataset_mode": True,
            "candidate_quality_oracle_mode": "deployment",
        }
    ]
    core = {
        "provenance_schema_version": 1,
        "source_state": {
            "source_commit": COMMIT,
            "source_worktree_dirty": False,
            "tracked_diff_hash": "0" * 64,
            "untracked_source_files": [],
            "release_eligible_source": True,
        },
        "input_suite": {
            "provided_path": "suite.json",
            "resolved_path": "/frozen/suite.json",
            "sha256": SUITE_SHA,
            "size_bytes": 100,
        },
        "factories": {
            "environment": {
                "import_spec": "tests:environment",
                "source": _source_descriptor(source_file),
            },
            "policy": {
                "import_spec": "tests:policy",
                "source": _source_descriptor(source_file),
            },
            "case_loader": None,
        },
        "policy_identity": {
            "explicit_policy_identity": explicit_identity,
            "model_id": model_id,
            "model_revision": model_revision,
        },
        "protocol_registry": {
            "protocol": "canonical",
            "registry_sha256": current_registry_sha256("canonical"),
        },
        "evaluator_source": _source_descriptor(evaluator_file),
    }
    core["identity_sha256"] = stable_json_sha256(core)
    provenance = {**core, "release_eligible": True, "release_failures": []}
    artifact = {
        "artifact_schema_version": 2,
        "artifact_type": "closed_loop_release_evaluation",
        "release_eligible": True,
        "release_failures": [],
        "provenance": provenance,
        "evaluation": {
            # A deliberately terrible scalar demonstrates that it is not a gate.
            "score": -1.0e12,
            "metrics": {},
            "suite_metrics": {
                "configuration": {
                    "suite_coverage_validation": {"passed": True},
                    "suite_content_hashes": suite_hashes,
                    "suite_content_sha256": stable_json_sha256(suite_hashes),
                    "release_environment_validation": {
                        "passed": True,
                        "episodes_checked": count,
                        "required": {
                            "production_dataset_mode": True,
                            "candidate_quality_oracle_mode": "deployment",
                        },
                        "observed": environment_rows,
                        "failures": [],
                    },
                    "policy_identity_validation": {
                        "passed": True,
                        "episodes_checked": count,
                        "required": copy.deepcopy(policy_identity),
                        "observed": [copy.deepcopy(policy_identity)],
                        "failures": [],
                    },
                },
                "overall": overall,
                "episodes": episodes,
            },
        },
    }
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(
            artifact, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()
    return artifact


def _validate(artifact: dict, **kwargs) -> EvaluationGateResult:
    return validate_evaluation_artifact(
        artifact,
        policy=_policy(),
        expected_source_commit=COMMIT,
        expected_suite_sha256=SUITE_SHA,
        expected_protocol="canonical",
        expected_registry_sha256=current_registry_sha256("canonical"),
        expected_policy_identity="expert-v1",
        required_gate_policy_id="test-hard-gate-v1",
        repo_root=REPO_ROOT,
        require_current_clean_source=False,
        **kwargs,
    )


def _rehash(artifact: dict) -> None:
    unsigned = dict(artifact)
    unsigned.pop("content_sha256", None)
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


class EvaluationGateTests(unittest.TestCase):
    def test_valid_artifact_passes_without_using_scalar_score(self) -> None:
        result = _validate(_artifact())
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(result.observed["terminal_rate"], 1.0)

    def test_hard_safety_failure_cannot_be_offset_by_score(self) -> None:
        artifact = _artifact(false_commit_episode=0)
        artifact["evaluation"]["score"] = 1.0e100
        unsigned = dict(artifact)
        unsigned.pop("content_sha256")
        artifact["content_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()
        result = _validate(artifact)
        self.assertFalse(result.passed)
        self.assertTrue(any("maximum_false_commit_count" in row for row in result.failures))

    def test_artifact_and_provenance_hashes_are_recomputed(self) -> None:
        artifact = _artifact()
        artifact["evaluation"]["score"] = 99.0
        result = _validate(artifact)
        self.assertFalse(result.passed)
        self.assertTrue(any("content_sha256" in row for row in result.failures))

        artifact = _artifact()
        artifact["provenance"]["source_state"]["source_commit"] = "e" * 40
        unsigned = dict(artifact)
        unsigned.pop("content_sha256")
        artifact["content_sha256"] = hashlib.sha256(
            json.dumps(
                unsigned, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()
        result = _validate(artifact)
        self.assertFalse(result.passed)
        self.assertTrue(any("identity_sha256" in row for row in result.failures))

    def test_invalid_call_bound_is_explicit_and_fail_closed(self) -> None:
        within_bound = _validate(_artifact(count=5, invalid_episode=0))
        self.assertTrue(within_bound.passed, within_bound.failures)
        self.assertEqual(
            len(
                _artifact(count=5)["evaluation"]["suite_metrics"]["configuration"][
                    "release_environment_validation"
                ]["observed"]
            ),
            1,
        )
        over_bound = _validate(_artifact(count=4, invalid_episode=0))
        self.assertFalse(over_bound.passed)
        self.assertTrue(
            any("maximum_invalid_action_rate" in row for row in over_bound.failures)
        )

    def test_duplicate_family_root_cannot_inflate_coverage(self) -> None:
        result = _validate(_artifact(count=2, duplicate_root=True))
        self.assertFalse(result.passed)
        self.assertTrue(any("globally duplicate" in row for row in result.failures))

    def test_one_root_cannot_be_relabelled_across_families(self) -> None:
        artifact = _artifact(count=2)
        second = artifact["evaluation"]["suite_metrics"]["episodes"][1]
        second["physical_root"] = "root-0"
        second["family"] = "measurement"
        strict = second["audit"]["strict_release_audit"]
        strict["physical_root_fingerprint"] = "root-0"
        strict["scenario_family"] = "measurement"
        policy = _policy()
        policy["family_policy"]["measurement"] = {
            "minimum_physical_roots": 1,
            "minimum_resolution_rate": 1.0,
            "maximum_operator_escalation_rate": 0.0,
        }
        _rehash(artifact)
        result = validate_evaluation_artifact(
            artifact,
            policy=policy,
            expected_source_commit=COMMIT,
            expected_suite_sha256=SUITE_SHA,
            expected_protocol="canonical",
            expected_registry_sha256=current_registry_sha256("canonical"),
            expected_policy_identity="expert-v1",
            required_gate_policy_id="test-hard-gate-v1",
            repo_root=REPO_ROOT,
            require_current_clean_source=False,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("globally duplicate" in row for row in result.failures))

    def test_self_declared_mock_environment_needs_pinned_factory(self) -> None:
        policy = _policy()
        policy["approved_factories"]["environment"] = []
        result = validate_evaluation_artifact(
            _artifact(),
            policy=policy,
            expected_source_commit=COMMIT,
            expected_suite_sha256=SUITE_SHA,
            expected_protocol="canonical",
            expected_registry_sha256=current_registry_sha256("canonical"),
            expected_policy_identity="expert-v1",
            required_gate_policy_id="test-hard-gate-v1",
            repo_root=REPO_ROOT,
            require_current_clean_source=False,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("environment factory" in row for row in result.failures))

    def test_scripted_expert_factory_cannot_be_relabelled_as_model(self) -> None:
        artifact = _artifact(
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        policy = _policy()
        policy["approved_factories"]["model_policy"] = []
        result = validate_evaluation_artifact(
            artifact,
            policy=policy,
            expected_source_commit=COMMIT,
            expected_suite_sha256=SUITE_SHA,
            expected_protocol="canonical",
            expected_registry_sha256=current_registry_sha256("canonical"),
            expected_model_id="base/gemma",
            expected_model_revision=MODEL_REVISION,
            required_gate_policy_id="test-hard-gate-v1",
            repo_root=REPO_ROOT,
            require_current_clean_source=False,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("model_policy factory" in row for row in result.failures))

    def test_environment_terminal_and_required_evidence_fail_closed(self) -> None:
        artifact = _artifact()
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        episode["release_environment_attestation"]["passed"] = False
        episode["release_environment_attestation"]["production_dataset_mode"] = False
        _rehash(artifact)
        result = _validate(artifact)
        self.assertFalse(result.passed)
        self.assertTrue(any("deployment-environment" in row for row in result.failures))

        artifact = _artifact()
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        episode["terminal"] = False
        episode["terminal_outcome"] = None
        episode["final_physical_success"] = False
        overall = artifact["evaluation"]["suite_metrics"]["overall"]
        overall["terminal_episodes"] = 0
        overall["terminal_rate"] = 0.0
        overall["resolved_episodes"] = 0
        overall["resolution_rate"] = 0.0
        _rehash(artifact)
        result = _validate(artifact)
        self.assertFalse(result.passed)
        self.assertTrue(any("minimum_terminal_rate" in row for row in result.failures))
        self.assertTrue(any("family 'no_error' resolution" in row for row in result.failures))

        artifact = _artifact()
        del artifact["evaluation"]["suite_metrics"]["episodes"][0][
            "false_commit_count"
        ]
        _rehash(artifact)
        result = _validate(artifact)
        self.assertFalse(result.passed)
        self.assertTrue(any("missing required evidence" in row for row in result.failures))

    def test_outer_outcome_cannot_relabel_a_strict_escalation_as_resolution(self) -> None:
        artifact = _artifact()
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        strict = episode["audit"]["strict_release_audit"]
        strict["terminal_outcome"] = "operator_escalation"
        strict["checks"]["remaining_true_faults"] = {"status": "not_required"}
        strict["checks"]["final_measurements_match_clean"] = {
            "status": "not_required"
        }
        strict["checks"]["final_case_matches_clean"] = {"status": "not_required"}
        _rehash(artifact)
        result = _validate(artifact)
        self.assertFalse(result.passed)
        self.assertTrue(any("strict release audit" in row for row in result.failures))

    def test_current_dirty_source_is_rejected(self) -> None:
        dirty = {
            "source_commit": COMMIT,
            "release_eligible_source": False,
            "source_worktree_dirty": True,
        }
        with mock.patch(
            "psse_env.dagger.evaluation_gate.git_source_state", return_value=dirty
        ):
            result = validate_evaluation_artifact(
                _artifact(),
                policy=_policy(),
                expected_source_commit=COMMIT,
                expected_suite_sha256=SUITE_SHA,
                expected_protocol="canonical",
                expected_registry_sha256=current_registry_sha256("canonical"),
                expected_policy_identity="expert-v1",
                required_gate_policy_id="test-hard-gate-v1",
                repo_root=REPO_ROOT,
            )
        self.assertFalse(result.passed)
        self.assertTrue(any("current evaluation-gate source" in row for row in result.failures))

    def test_default_policy_identity_is_bound_to_packaged_content(self) -> None:
        weaker = _policy()
        weaker["policy_id"] = DEFAULT_POLICY_ID
        weaker["family_policy"]["no_error"]["minimum_resolution_rate"] = 0.0
        result = validate_evaluation_artifact(
            _artifact(),
            policy=weaker,
            expected_source_commit=COMMIT,
            expected_suite_sha256=SUITE_SHA,
            expected_protocol="canonical",
            expected_registry_sha256=current_registry_sha256("canonical"),
            expected_policy_identity="expert-v1",
            required_gate_policy_id=DEFAULT_POLICY_ID,
            repo_root=REPO_ROOT,
            require_current_clean_source=False,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("packaged policy" in row for row in result.failures))

    def test_model_identity_checkpoint_cli_path(self) -> None:
        artifact = _artifact(
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "checkpoint.json"
            suite_path = root / "suite.json"
            policy_path = root / "policy.json"
            report_path = root / "report.json"
            suite_path.write_bytes(b"frozen suite")
            artifact["provenance"]["input_suite"]["sha256"] = file_sha256(suite_path)
            identity_core = dict(artifact["provenance"])
            for field in ("identity_sha256", "release_eligible", "release_failures"):
                identity_core.pop(field)
            artifact["provenance"]["identity_sha256"] = stable_json_sha256(identity_core)
            unsigned = dict(artifact)
            unsigned.pop("content_sha256")
            artifact["content_sha256"] = hashlib.sha256(
                json.dumps(
                    unsigned, sort_keys=True, separators=(",", ":"), default=str
                ).encode("utf-8")
            ).hexdigest()
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
            clean = {
                "source_commit": COMMIT,
                "release_eligible_source": True,
                "source_worktree_dirty": False,
            }
            arguments = [
                "--artifact", str(artifact_path),
                "--role", "checkpoint-promotion",
                "--policy", str(policy_path),
                "--required-gate-policy-id", "test-hard-gate-v1",
                "--expected-source-commit", COMMIT,
                "--expected-suite", str(suite_path),
                "--expected-model-id", "checkpoint/bc0",
                "--expected-model-revision", MODEL_REVISION,
                "--report-output", str(report_path),
            ]
            with mock.patch(
                "psse_env.dagger.evaluation_gate.git_source_state",
                return_value=clean,
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(gate_main(arguments), 0)
            self.assertTrue(json.loads(report_path.read_text())["passed"])

    def test_sft_train_refuses_missing_baselines_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text("{}\n", encoding="utf-8")
            validation.write_text("{}\n", encoding="utf-8")
            with mock.patch("psse_env.sft.cli.run_lora_training") as run_training:
                with redirect_stderr(io.StringIO()):
                    status = sft_main(
                        [
                            "train",
                            "--revision", "f" * 40,
                            "--train", str(train),
                            "--validation", str(validation),
                        ]
                    )
            self.assertEqual(status, 2)
            run_training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
