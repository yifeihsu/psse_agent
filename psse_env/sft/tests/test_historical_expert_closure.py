"""Tests for manifest-authorized historical expert closure reuse."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from psse_env.sft import historical_expert_closure as closure
from psse_env.sft.provenance import stable_json_sha256


def _serialized(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


class _ClosureFixture:
    registry_sha256 = "8" * 64
    policy_id = "bc0_closed_loop_hard_gate_v3"
    policy_sha256 = "7" * 64
    policy_identity = "observable-expert-test-v1"

    def __init__(self) -> None:
        self.owner = tempfile.TemporaryDirectory()
        self.root = Path(self.owner.name).resolve()
        self.repo = self.root / "repo"
        self.evidence = self.root / "evidence"
        self.repo.mkdir()
        self.evidence.mkdir()
        self._git("init", "-b", "main")
        self._git("config", "user.email", "closure@example.invalid")
        self._git("config", "user.name", "Closure Test")
        self._git("config", "core.autocrlf", "false")

        self.protected_paths = (
            "psse_env/dagger/bc0_evaluation_policy.json",
            "psse_env/dagger/evaluator.py",
            "psse_env/dagger/release_factories.py",
            "psse_env/dagger/suites/bc0_eval_suite_v1.json",
        )
        self.gate_paths = (
            "psse_env/dagger/evaluation_gate.py",
            "psse_env/dagger/test_evaluation_gate.py",
        )
        self._write(self.protected_paths[0], b'{"policy_id":"test"}\n')
        self._write(self.protected_paths[1], b"EVALUATOR = 1\n")
        self._write(self.protected_paths[2], b"FACTORY = 1\n")
        self._write(self.protected_paths[3], b"[]\n")
        self._write(self.gate_paths[0], b"GATE = 1\n")
        self._write(self.gate_paths[1], b"TEST_GATE = 1\n")
        self.artifact_commit = self._commit("artifact producer")

        self._write(self.gate_paths[0], b"GATE = 2\n")
        self.intermediate_commit = self._commit("normalize JSON comparison")
        self._write(self.gate_paths[1], b"TEST_GATE = 2\n")
        self.validator_commit = self._commit("harden historical validator")

        self.protected_descriptors = {
            path: self._descriptor(self.validator_commit, path)
            for path in self.protected_paths
        }
        self.validator_descriptors = {
            path: self._descriptor(self.validator_commit, path)
            for path in self.gate_paths
        }
        self.artifact_descriptors = {
            path: self._descriptor(self.artifact_commit, path)
            for path in self.gate_paths
        }

        self.artifact_path = self.evidence / "expert_baseline_evaluation.json"
        artifact = {
            "artifact_schema_version": 1,
            "artifact_type": "closed_loop_evaluation",
            "evaluation": {"score": 1.0},
            "provenance": {
                "input_suite": {
                    "sha256": self.protected_descriptors[self.protected_paths[3]][
                        "sha256"
                    ]
                },
                "policy_identity": {
                    "explicit_policy_identity": self.policy_identity,
                    "model_id": None,
                    "model_revision": None,
                },
                "protocol_registry": {
                    "protocol": "canonical",
                    "registry_sha256": self.registry_sha256,
                },
                "source_state": {
                    "release_eligible_source": True,
                    "source_commit": self.artifact_commit,
                    "source_worktree_dirty": False,
                },
            },
            "release_eligible": True,
            "release_failures": [],
        }
        artifact["content_sha256"] = stable_json_sha256(artifact)
        self.artifact_path.write_bytes(_serialized(artifact))
        self.artifact_sha256 = hashlib.sha256(self.artifact_path.read_bytes()).hexdigest()

        self.receipt_path = self.evidence / "expert_baseline_gate_receipt.json"
        self.receipt = self._receipt(artifact)
        self.receipt_path.write_bytes(_serialized(self.receipt))
        self._set_receipt_mode()
        self.receipt_sha256 = hashlib.sha256(self.receipt_path.read_bytes()).hexdigest()

        self.manifest = self._manifest(artifact)
        self.manifest_path = self.repo / closure.HISTORICAL_EXPERT_CLOSURE_MANIFEST
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_bytes(_serialized(self.manifest))
        self._write("psse_env/sft/consumer_freeze.py", b"CONSUMER = 1\n")
        self.consumer_commit = self._commit("add historical closure consumer")

    def close(self) -> None:
        for path in (self.receipt_path, self.artifact_path):
            try:
                path.chmod(0o600)
            except OSError:
                pass
        self.owner.cleanup()

    def _git(self, *args: str, check: bool = True) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            check=check,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def _write(self, relative: str, payload: bytes) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def _commit(self, message: str) -> str:
        self._git("add", "--all")
        self._git("commit", "-m", message)
        return self._git("rev-parse", "HEAD")

    def _descriptor(self, commit: str, path: str) -> dict[str, str]:
        line = self._git("ls-tree", commit, "--", path)
        metadata, observed_path = line.split("\t", 1)
        mode, object_type, blob_id = metadata.split(" ")
        self.assert_equal(observed_path, path)
        self.assert_equal(object_type, "blob")
        blob = subprocess.run(
            ["git", "cat-file", "blob", blob_id],
            cwd=self.repo,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "path": path,
            "mode": mode,
            "git_blob_id": blob_id,
            "sha256": hashlib.sha256(blob).hexdigest(),
        }

    @staticmethod
    def assert_equal(left: object, right: object) -> None:
        if left != right:
            raise AssertionError(f"{left!r} != {right!r}")

    def _tree_delta(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for path in self.gate_paths:
            old = self.artifact_descriptors[path]
            new = self.validator_descriptors[path]
            rows.append(
                {
                    "path": path,
                    "path_hex": path.encode("utf-8").hex(),
                    "status": "M",
                    "old_mode": "100644",
                    "new_mode": "100644",
                    "old_blob_id": old["git_blob_id"],
                    "new_blob_id": new["git_blob_id"],
                }
            )
        return rows

    def _receipt(self, artifact: dict[str, object]) -> dict[str, object]:
        tree_delta = self._tree_delta()
        gate_sources = {
            path: {
                "artifact": copy.deepcopy(self.artifact_descriptors[path]),
                "validator": copy.deepcopy(self.validator_descriptors[path]),
            }
            for path in self.gate_paths
        }
        protected_sources = {
            path: {
                "artifact": copy.deepcopy(self.protected_descriptors[path]),
                "validator": copy.deepcopy(self.protected_descriptors[path]),
            }
            for path in self.protected_paths
        }
        dual = {
            "artifact_is_ancestor": True,
            "artifact_source_commit": self.artifact_commit,
            "contract": "gate_only_json_domain_revalidation_v1",
            "gate_tree_sources": gate_sources,
            "git_replacements_disabled": True,
            "grafts_present": False,
            "history_commits": [self.intermediate_commit, self.validator_commit],
            "history_path_hex": [
                path.encode("utf-8").hex() for path in self.gate_paths
            ],
            "history_paths": list(self.gate_paths),
            "merge_commits": [],
            "protected_blob_ids": {
                path: descriptor["git_blob_id"]
                for path, descriptor in self.protected_descriptors.items()
            },
            "protected_sources": protected_sources,
            "replace_refs": [],
            "tracked_tree_matches_validator": True,
            "tree_delta": tree_delta,
            "validator_source_commit": self.validator_commit,
        }
        source_state = {
            "release_eligible_source": True,
            "source_commit": self.validator_commit,
            "source_worktree_dirty": False,
            "tracked_diff_hash": hashlib.sha256(b"").hexdigest(),
            "untracked_source_files": [],
        }
        return {
            "artifact_content_sha256": artifact["content_sha256"],
            "artifact_path": "/historical/producer/expert_baseline_evaluation.json",
            "artifact_sha256": self.artifact_sha256,
            "evaluated_policy_identity": copy.deepcopy(
                artifact["provenance"]["policy_identity"]  # type: ignore[index]
            ),
            "evidence_passed": True,
            "failures": [],
            "frozen_suite_sha256": self.protected_descriptors[
                self.protected_paths[3]
            ]["sha256"],
            "gate_policy_id": self.policy_id,
            "gate_policy_sha256": self.policy_sha256,
            "observed": {
                "episodes": 115,
                "evaluator_error_episodes": 0,
                "false_commit_count": 0,
                "false_finalization_count": 0,
                "false_rollback_count": 0,
                "healthy_component_corruption_episodes": 0,
            },
            "passed": True,
            "performance_enforced": True,
            "performance_passed": True,
            "protocol": "canonical",
            "registry_sha256": self.registry_sha256,
            "source_commit": self.artifact_commit,
            "validation_role": "expert-baseline",
            "validator_source_attestation": {
                "schema_version": 1,
                **copy.deepcopy(dual),
                "current_source_enforced": True,
                "current_source_state": copy.deepcopy(source_state),
                "failures": [],
                "final_dual_source_attestation": copy.deepcopy(dual),
                "final_dual_source_failures": [],
                "final_source_state": copy.deepcopy(source_state),
                "passed": True,
                "validator_gate_source": copy.deepcopy(
                    self.validator_descriptors[self.gate_paths[0]]
                ),
            },
        }

    def _manifest(self, artifact: dict[str, object]) -> dict[str, object]:
        return {
            "closures": {
                closure.DEFAULT_HISTORICAL_EXPERT_CLOSURE_ID: {
                    "artifact": {
                        "content_sha256": artifact["content_sha256"],
                        "sha256": self.artifact_sha256,
                        "source_commit": self.artifact_commit,
                    },
                    "evaluation": {
                        "episodes": 115,
                        "gate_policy_id": self.policy_id,
                        "gate_policy_sha256": self.policy_sha256,
                        "policy_identity": copy.deepcopy(
                            artifact["provenance"]["policy_identity"]  # type: ignore[index]
                        ),
                        "protocol": "canonical",
                        "registry_sha256": self.registry_sha256,
                        "role": "expert-baseline",
                        "safety_counts": {
                            "evaluator_error_episodes": 0,
                            "false_commit_count": 0,
                            "false_finalization_count": 0,
                            "false_rollback_count": 0,
                            "healthy_component_corruption_episodes": 0,
                        },
                        "suite_sha256": self.protected_descriptors[
                            self.protected_paths[3]
                        ]["sha256"],
                    },
                    "protected_sources": {
                        path: {
                            key: descriptor[key]
                            for key in ("git_blob_id", "mode", "sha256")
                        }
                        for path, descriptor in self.protected_descriptors.items()
                    },
                    "receipt": {"sha256": self.receipt_sha256},
                    "validator": {
                        "contract": "gate_only_json_domain_revalidation_v1",
                        "history_commits": [
                            self.intermediate_commit,
                            self.validator_commit,
                        ],
                        "history_paths": list(self.gate_paths),
                        "source_commit": self.validator_commit,
                        "validator_sources": {
                            path: {
                                key: descriptor[key]
                                for key in ("git_blob_id", "mode", "sha256")
                            }
                            for path, descriptor in self.validator_descriptors.items()
                        },
                    },
                }
            },
            "contract": "bc0_historical_expert_closure_registry_v1",
            "schema_version": 1,
        }

    def _set_receipt_mode(self) -> None:
        if os.name == "posix":
            self.receipt_path.chmod(0o400)

    def validate(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "expert_artifact_path": self.artifact_path,
            "repo_root": self.repo,
            "expected_suite_path": self.repo / self.protected_paths[3],
            "expected_policy_path": self.repo / self.protected_paths[0],
            "expected_policy_identity": self.policy_identity,
            "expected_protocol": "canonical",
            "expected_registry_sha256": self.registry_sha256,
        }
        arguments.update(overrides)
        with mock.patch.object(
            closure,
            "current_registry_sha256",
            return_value=self.registry_sha256,
        ):
            return closure.validate_historical_expert_closure(
                self.receipt_path,
                **arguments,
            )

    def republish_receipt_and_manifest(self) -> None:
        if os.name == "posix":
            self.receipt_path.chmod(0o600)
        self.receipt_path.write_bytes(_serialized(self.receipt))
        self._set_receipt_mode()
        receipt_sha = hashlib.sha256(self.receipt_path.read_bytes()).hexdigest()
        entry = self.manifest["closures"][  # type: ignore[index]
            closure.DEFAULT_HISTORICAL_EXPERT_CLOSURE_ID
        ]
        entry["receipt"]["sha256"] = receipt_sha  # type: ignore[index]
        self.manifest_path.write_bytes(_serialized(self.manifest))
        self.consumer_commit = self._commit("repin synthetic receipt")


class HistoricalExpertClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _ClosureFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_exact_closure_passes_and_preserves_three_source_identities(self) -> None:
        result = self.fixture.validate()

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(
            result["contract"],
            closure.HISTORICAL_EXPERT_CLOSURE_VALIDATION_CONTRACT,
        )
        self.assertEqual(
            result["expert"]["artifact_source_commit"],  # type: ignore[index]
            self.fixture.artifact_commit,
        )
        self.assertEqual(
            result["expert"]["validator_source_commit"],  # type: ignore[index]
            self.fixture.validator_commit,
        )
        self.assertEqual(
            result["consumer_source_attestation"]["initial"]["source_commit"],  # type: ignore[index]
            self.fixture.consumer_commit,
        )
        self.assertTrue(
            result["consumer_source_attestation"]["unchanged"]  # type: ignore[index]
        )
        self.assertEqual(
            result["artifact"]["path"],  # type: ignore[index]
            str(self.fixture.artifact_path),
        )
        self.assertEqual(
            result["expert"]["producer_artifact_path"],  # type: ignore[index]
            "/historical/producer/expert_baseline_evaluation.json",
        )

    def test_receipt_and_artifact_are_each_read_once(self) -> None:
        original = closure._read_regular_snapshot
        counts: dict[Path, int] = {}

        def counted(value: str | Path, **kwargs: object) -> object:
            path = Path(value).resolve()
            counts[path] = counts.get(path, 0) + 1
            return original(value, **kwargs)  # type: ignore[arg-type]

        with mock.patch.object(closure, "_read_regular_snapshot", side_effect=counted):
            result = self.fixture.validate()

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(counts[self.fixture.receipt_path], 1)
        self.assertEqual(counts[self.fixture.artifact_path], 1)

    def test_receipt_digest_tamper_fails_closed(self) -> None:
        if os.name == "posix":
            self.fixture.receipt_path.chmod(0o600)
        self.fixture.receipt_path.write_bytes(
            self.fixture.receipt_path.read_bytes() + b"\n"
        )
        self.fixture._set_receipt_mode()

        result = self.fixture.validate()

        self.assertFalse(result["passed"])
        self.assertTrue(
            any("receipt SHA-256" in failure for failure in result["failures"])
        )

    def test_semantic_failure_is_rejected_even_when_manifest_repins_bytes(self) -> None:
        self.fixture.receipt["passed"] = False
        self.fixture.republish_receipt_and_manifest()

        result = self.fixture.validate()

        self.assertFalse(result["passed"])
        self.assertIn(
            "historical expert receipt field changed: passed",
            result["failures"],
        )

    def test_receipt_must_be_absolute_and_outside_repository(self) -> None:
        result = self.fixture.validate()
        self.assertTrue(result["passed"], result["failures"])

        inside = self.fixture.repo / "artifacts" / "receipt.json"
        inside.parent.mkdir(parents=True)
        inside.write_bytes(self.fixture.receipt_path.read_bytes())
        if os.name == "posix":
            inside.chmod(0o400)
        result = self.fixture.validate()
        with mock.patch.object(
            closure,
            "current_registry_sha256",
            return_value=self.fixture.registry_sha256,
        ):
            result = closure.validate_historical_expert_closure(
                inside,
                expert_artifact_path=self.fixture.artifact_path,
                repo_root=self.fixture.repo,
                expected_suite_path=self.fixture.repo / self.fixture.protected_paths[3],
                expected_policy_path=self.fixture.repo / self.fixture.protected_paths[0],
                expected_policy_identity=self.fixture.policy_identity,
                expected_registry_sha256=self.fixture.registry_sha256,
            )
        self.assertFalse(result["passed"])
        self.assertTrue(any("outside" in failure for failure in result["failures"]))

        relative = self.fixture.receipt_path.relative_to(self.fixture.root)
        with mock.patch.object(
            closure,
            "current_registry_sha256",
            return_value=self.fixture.registry_sha256,
        ):
            result = closure.validate_historical_expert_closure(
                relative,
                expert_artifact_path=self.fixture.artifact_path,
                repo_root=self.fixture.repo,
                expected_suite_path=self.fixture.repo / self.fixture.protected_paths[3],
                expected_policy_path=self.fixture.repo / self.fixture.protected_paths[0],
                expected_policy_identity=self.fixture.policy_identity,
            )
        self.assertFalse(result["passed"])
        self.assertTrue(any("absolute" in failure for failure in result["failures"]))

    def test_artifact_must_be_absolute_and_outside_repository(self) -> None:
        inside = self.fixture.repo / "artifacts" / "expert.json"
        inside.parent.mkdir(parents=True)
        inside.write_bytes(self.fixture.artifact_path.read_bytes())

        result = self.fixture.validate(expert_artifact_path=inside)
        self.assertFalse(result["passed"])
        self.assertTrue(any("outside" in failure for failure in result["failures"]))

        result = self.fixture.validate(
            expert_artifact_path=Path("artifacts/expert.json")
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("absolute" in failure for failure in result["failures"]))

    @unittest.skipUnless(os.name == "posix", "POSIX mode contract")
    def test_receipt_requires_exact_mode_0400(self) -> None:
        self.fixture.receipt_path.chmod(0o600)

        result = self.fixture.validate()

        self.assertFalse(result["passed"])
        self.assertTrue(any("mode must be 0400" in failure for failure in result["failures"]))

    def test_symlink_receipt_and_artifact_are_rejected(self) -> None:
        receipt_link = self.fixture.evidence / "receipt-link.json"
        artifact_link = self.fixture.evidence / "artifact-link.json"
        try:
            receipt_link.symlink_to(self.fixture.receipt_path)
            artifact_link.symlink_to(self.fixture.artifact_path)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        with mock.patch.object(
            closure,
            "current_registry_sha256",
            return_value=self.fixture.registry_sha256,
        ):
            receipt_result = closure.validate_historical_expert_closure(
                receipt_link,
                expert_artifact_path=self.fixture.artifact_path,
                repo_root=self.fixture.repo,
                expected_suite_path=self.fixture.repo / self.fixture.protected_paths[3],
                expected_policy_path=self.fixture.repo / self.fixture.protected_paths[0],
                expected_policy_identity=self.fixture.policy_identity,
            )
            artifact_result = closure.validate_historical_expert_closure(
                self.fixture.receipt_path,
                expert_artifact_path=artifact_link,
                repo_root=self.fixture.repo,
                expected_suite_path=self.fixture.repo / self.fixture.protected_paths[3],
                expected_policy_path=self.fixture.repo / self.fixture.protected_paths[0],
                expected_policy_identity=self.fixture.policy_identity,
            )
        self.assertFalse(receipt_result["passed"])
        self.assertFalse(artifact_result["passed"])
        self.assertTrue(any("symlink" in failure for failure in receipt_result["failures"]))
        self.assertTrue(any("symlink" in failure for failure in artifact_result["failures"]))

    def test_dirty_consumer_source_fails_closed(self) -> None:
        self.fixture._write("psse_env/sft/consumer_freeze.py", b"CONSUMER = 2\n")

        result = self.fixture.validate()

        self.assertFalse(result["passed"])
        self.assertTrue(any("clean" in failure for failure in result["failures"]))

    def test_consumer_merge_after_validator_fails_closed(self) -> None:
        self.fixture._git("checkout", "-b", "side")
        self.fixture._write("side.txt", b"side\n")
        self.fixture._commit("side commit")
        self.fixture._git("checkout", "main")
        self.fixture._write("main.txt", b"main\n")
        self.fixture._commit("main commit")
        self.fixture._git("merge", "--no-ff", "side", "-m", "merge side")

        result = self.fixture.validate()

        self.assertFalse(result["passed"])
        self.assertTrue(any("contains a merge" in failure for failure in result["failures"]))

    def test_committed_protected_source_change_fails_closed(self) -> None:
        self.fixture._write(self.fixture.protected_paths[1], b"EVALUATOR = 2\n")
        self.fixture._commit("change evaluator")

        result = self.fixture.validate()

        self.assertFalse(result["passed"])
        self.assertTrue(any("pinned source" in failure for failure in result["failures"]))

    def test_git_replace_ref_fails_closed(self) -> None:
        self.fixture._git(
            "replace",
            self.fixture.artifact_commit,
            self.fixture.validator_commit,
        )

        result = self.fixture.validate()

        self.assertFalse(result["passed"])
        self.assertTrue(any("replacement refs" in failure for failure in result["failures"]))

    def test_final_source_re_attestation_detects_mutation(self) -> None:
        original = closure._source_attestation
        calls = 0

        def mutate_after_initial(**kwargs: object) -> object:
            nonlocal calls
            calls += 1
            result = original(**kwargs)  # type: ignore[arg-type]
            if calls == 1:
                self.fixture._write(
                    "psse_env/sft/consumer_freeze.py",
                    b"CONSUMER = 2\n",
                )
            return result

        with mock.patch.object(
            closure,
            "_source_attestation",
            side_effect=mutate_after_initial,
        ):
            result = self.fixture.validate()

        self.assertFalse(result["passed"])
        self.assertFalse(
            result["consumer_source_attestation"]["unchanged"]  # type: ignore[index]
        )
        self.assertIn(
            "consumer source changed during historical closure validation",
            result["failures"],
        )


class ProductionClosureManifestTests(unittest.TestCase):
    def test_manifest_pins_the_completed_hpc_receipt(self) -> None:
        root = Path(__file__).resolve().parents[3]
        payload = json.loads(
            (root / closure.HISTORICAL_EXPERT_CLOSURE_MANIFEST).read_text(
                encoding="utf-8"
            )
        )
        entry = payload["closures"][closure.DEFAULT_HISTORICAL_EXPERT_CLOSURE_ID]

        self.assertEqual(
            entry["receipt"]["sha256"],
            "3c7d65f0f5f5821a779ff4dc5765b9a4774e0965cf079a594fe38bd655277475",
        )
        self.assertEqual(
            entry["artifact"],
            {
                "content_sha256": "2bbe9495c447df4da2da74e5f7e112f4e3c38f0467d792f90c5ba0fa25fab73e",
                "sha256": "193f745da59c0b527e2ca64423053508e8708cbb3b86c879799e8abdcb4027ce",
                "source_commit": "ef30899682ef84f54c0237df9a3eb5871e1d0d7d",
            },
        )
        self.assertEqual(
            entry["validator"]["source_commit"],
            "99b5c59a70063b89806f9f207b60b5a33c6cfe03",
        )
        self.assertEqual(
            entry["evaluation"]["suite_sha256"],
            "195cc7acfcffafbcbf8fc6a52a5eed5111f42eec586187ab051f62b0a7892081",
        )
        self.assertEqual(
            entry["evaluation"]["registry_sha256"],
            "8b3815ddecf2d4f5b9b4d1d0760ff32e8cee7872b6ea04d4d5285db80c93b709",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
