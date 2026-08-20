"""Probe-suite publication must be atomic, excluded-root aware, and separate."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import psse_env.dagger.build_recovery_probe_suite as probe_builder_module
from psse_env.dagger.build_recovery_probe_suite import (
    PROBE_GENERATOR_IDENTITY,
    PROBE_SUITE_ARTIFACT_TYPE,
    aggregate_roots,
    build_recovery_probe_suite,
    envelope_roots,
    main,
    validate_recovery_probe_suite_binding,
)
from psse_env.dagger.recovery_probes import (
    RECOVERY_PROBE_CONTRACT,
    RECOVERY_PROBE_ROOT_QUOTAS,
    audit_recovery_probe_support,
    verify_probe_stratum,
)
from psse_env.dagger.offline_teacher_target_audit import (
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
)
from psse_env.dagger.rollout_collector import (
    classify_state_example,
    observable_rank_one_target_proof,
)
from psse_env.sft.provenance import file_sha256, stable_json_sha256


def _envelope(root: str, index: int = 0) -> dict:
    if not root.startswith("physical_v"):
        import hashlib
        root = "physical_v3_" + hashlib.sha256(root.encode()).hexdigest()
    return {
        "grouping": {
            "physical_root_fingerprint": root,
            "scenario_family": "measurement",
            "error_cardinality": 1,
            "scenario_id": f"scenario-{index}",
        },
        "execution": {"scenario_id": f"scenario-{index}"},
    }


class ProbeSuiteRootReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_envelope_roots_reads_grouping_fingerprints(self):
        path = self.root / "scenarios.json"
        path.write_text(
            json.dumps([_envelope("root-a", 0), _envelope("root-b", 1)]),
            encoding="utf-8",
        )
        self.assertEqual(
            envelope_roots(path),
            {_envelope("root-a")["grouping"]["physical_root_fingerprint"],
             _envelope("root-b")["grouping"]["physical_root_fingerprint"]},
        )

    def test_envelope_roots_rejects_a_structure_that_is_neither_shape(self):
        path = self.root / "scenarios.json"
        path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            envelope_roots(path)

    def test_aggregate_roots_reads_the_raw_row_file(self):
        directory = self.root / "d0"
        directory.mkdir()
        (directory / "aggregate.raw.jsonl").write_text(
            "\n".join(
                json.dumps({"physical_root_fingerprint": "physical_v3_" + hashlib.sha256(f"d0-root-{i}".encode()).hexdigest()})
                for i in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            aggregate_roots(directory),
            {"physical_v3_" + hashlib.sha256(f"d0-root-{i}".encode()).hexdigest() for i in range(3)},
        )


class ProbeSuitePublicationTests(unittest.TestCase):
    """These guards fire before the environment is constructed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.scenarios = self.root / "scenarios.json"
        self.scenarios.write_text(
            json.dumps([_envelope("root-a", 0)]), encoding="utf-8"
        )

    def test_release_cli_requires_provenance_inputs_and_fixed_quotas(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(output):
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        for required in (
            "--scenario-manifest",
            "--scenario-generator-report",
            "--development-holdout-manifest",
            "--development-holdout-generator-report",
            "--reviewed-source-commit",
        ):
            self.assertIn(required, help_text)
        self.assertNotIn("--generator-identity", help_text)
        self.assertNotIn("--quota-post", help_text)

    def _build(self, **kwargs):
        scenarios = json.loads(self.scenarios.read_text(encoding="utf-8"))
        candidate_roots = {
            row["grouping"]["physical_root_fingerprint"] for row in scenarios
        }
        development = kwargs.pop("development_holdout", None)
        development_roots = envelope_roots(development) if development else set()
        facts = {
            "repo_root": Path(probe_builder_module.__file__).resolve().parents[2],
            "source_state": {
                "source_commit": "a" * 40,
                "release_eligible_source": True,
            },
            "scenarios": scenarios,
            "candidate_roots": candidate_roots,
            "d0_roots": set(),
            "development_roots": development_roots,
            "frozen_roots": set(),
            "d0_generation_provenance_id": "b" * 64,
            "input_artifacts": {},
        }
        with patch.object(
            probe_builder_module,
            "_validated_probe_inputs",
            return_value=facts,
        ):
            return build_recovery_probe_suite(
                scenarios_path=self.scenarios,
                scenario_manifest_path=self.root / "scenarios.manifest.json",
                scenario_generator_report_path=self.root / "scenarios.report.json",
                output=self.root / "probes.jsonl",
                manifest_path=self.root / "probes.manifest.json",
                reviewed_source_commit="a" * 40,
                forbidden_suite=self.root / "frozen.json",
                evaluation_policy=self.root / "policy.json",
                development_holdout=(development or self.root / "holdout.json"),
                development_holdout_manifest=self.root / "holdout.manifest.json",
                development_holdout_generator_report=self.root / "holdout.report.json",
                d0_aggregate_dir=self.root / "d0",
                **kwargs,
            )

    def test_existing_output_is_never_overwritten(self):
        (self.root / "probes.jsonl").write_text("stale\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._build()
        # The stale file is left exactly as it was.
        self.assertEqual(
            (self.root / "probes.jsonl").read_text(encoding="utf-8"), "stale\n"
        )

    def test_existing_manifest_is_never_overwritten(self):
        (self.root / "probes.manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._build()

    def test_empty_scenario_list_is_refused(self):
        self.scenarios.write_text(json.dumps([]), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._build()

    def test_fully_excluded_root_set_is_refused(self):
        """Every candidate root sits in the development holdout."""
        holdout = self.root / "holdout.json"
        holdout.write_text(json.dumps([_envelope("root-a", 0)]), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self._build(development_holdout=holdout)
        self.assertIn("excluded", str(caught.exception))


class ProbeSuiteBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.rows_path = self.root / "probes.jsonl"
        self.manifest_path = self.root / "probes.manifest.json"
        self.roots = {
            f"physical_v3_{hashlib.sha256(f'probe-{index}'.encode()).hexdigest()}"
            for index in range(24)
        }
        scenarios = [
            _envelope(root, index) for index, root in enumerate(sorted(self.roots))
        ]
        self.facts = {
            "repo_root": Path(probe_builder_module.__file__).resolve().parents[2],
            "source_state": {
                "source_commit": "a" * 40,
                "release_eligible_source": True,
            },
            "scenarios": scenarios,
            "candidate_roots": set(self.roots),
            "d0_roots": set(),
            "development_roots": set(),
            "frozen_roots": set(),
            "d0_generation_provenance_id": "b" * 64,
            "input_artifacts": {},
        }

    @staticmethod
    def _row(root: str, index: int, stratum: str) -> dict:
        state_id = f"probe-episode-{index}:s1"
        if stratum == "unsupported_correction_recovery":
            last_tool = "correct_measurements"
            error_code = "post_correction_confirmation_required"
        else:
            last_tool = "get_measurement_context"
            error_code = "unknown_state_id"
        observation = {
            "active_state_id": state_id,
            "candidate_state_id": None,
            "has_open_candidate": False,
            "last_tool": last_tool,
            "last_tool_status": "failure",
            "last_tool_output": {
                "execution_status": "failure",
                "error_code": error_code,
            },
        }
        preferred = {"tool": "run_wls", "arguments": {"state_id": state_id}}
        state_class = classify_state_example(
            observation,
            preferred_action=preferred,
        )
        verification = verify_probe_stratum(
            observation,
            preferred_action=preferred,
            state_class=state_class,
            scenario_family="measurement",
            error_cardinality=1,
            expected_stratum=stratum,
        )
        rank_one = observable_rank_one_target_proof(
            observation,
            preferred_action=preferred,
            expert_actions=[preferred],
        )
        return {
            "example_id": f"probe-{index}",
            "physical_root_fingerprint": root,
            "scenario_family": "measurement",
            "error_cardinality": 1,
            "scenario_id": f"scenario-{index}",
            "policy_observation": observation,
            "probe_setup_actions": [],
            "preferred_action": preferred,
            "state_class": state_class,
            "collector_contract": RECOVERY_PROBE_CONTRACT,
            "state_origin": "observable_recovery_probe",
            "dataset_source": "observable_recovery_probe",
            "replay_source": "observable_recovery_probe",
            "collection_role": "auxiliary_training",
            "state_visited_by": "observable_recovery_probe",
            "dataset_mode": "production",
            "recovery_stratum": stratum,
            "auxiliary_training_eligible": True,
            "production_label_eligible": False,
            "natural_on_policy_support_eligible": False,
            "training_decision_evidence_verified": True,
            "probe_intervention": {"tool": last_tool, "arguments": {}},
            "probe_stratum_verification": verification,
            "observable_rank_one_target_proof": rank_one,
            "offline_teacher_target_audit": {
                "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
                "passed": True,
                "action_class": "read_only",
                "checks": {"observable_evidence_gate_passed": True},
                "reason_codes": [],
            },
        }

    def _fixture_rows(self, roots_per_stratum: int = 12) -> list[dict]:
        ordered = sorted(self.roots)
        rows = [
            self._row(root, index, "post_failure_no_candidate")
            for index, root in enumerate(ordered[:roots_per_stratum])
        ]
        rows.extend(
            self._row(root, index, "unsupported_correction_recovery")
            for index, root in enumerate(
                ordered[12 : 12 + roots_per_stratum],
                start=12,
            )
        )
        return rows

    def _generation_report(self, rows: list[dict]) -> dict:
        support = audit_recovery_probe_support(rows)
        roots_admitted = {
            stratum: len(
                {
                    row["physical_root_fingerprint"]
                    for row in rows
                    if row["recovery_stratum"] == stratum
                }
            )
            for stratum in sorted(RECOVERY_PROBE_ROOT_QUOTAS)
        }
        quota_met = {
            stratum: roots_admitted[stratum] >= quota
            for stratum, quota in sorted(RECOVERY_PROBE_ROOT_QUOTAS.items())
        }
        return {
            "contract": RECOVERY_PROBE_CONTRACT,
            "scenarios_considered": len(self.facts["scenarios"]),
            "root_quotas": dict(sorted(RECOVERY_PROBE_ROOT_QUOTAS.items())),
            "roots_admitted": roots_admitted,
            "quota_met": quota_met,
            "skipped": {},
            "attempts": [
                {
                    "physical_root_fingerprint": row[
                        "physical_root_fingerprint"
                    ],
                    "expected_stratum": row["recovery_stratum"],
                    "actual_stratum": row["recovery_stratum"],
                    "admitted": True,
                }
                for row in rows
            ],
            "probe_support": support,
            "passed": bool(support["passed"] and all(quota_met.values())),
        }

    def _publish_valid_fixture(self, roots_per_stratum: int = 12) -> None:
        rows = self._fixture_rows(roots_per_stratum)
        probe_roots = {row["physical_root_fingerprint"] for row in rows}
        descriptor = probe_builder_module._probe_generation_descriptor(
            facts=self.facts,
            probe_roots=probe_roots,
            probe_rows=rows,
        )
        provenance_id = stable_json_sha256(descriptor)
        for row in rows:
            row["generation_provenance_id"] = provenance_id
        self.rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = probe_builder_module._expected_probe_manifest(
            rows=rows,
            facts=self.facts,
            descriptor=descriptor,
            provenance_id=provenance_id,
            generation_report=self._generation_report(rows),
            rows_name=self.rows_path.name,
            rows_sha256=file_sha256(self.rows_path),
        )
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _read_rows(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.rows_path.read_text(encoding="utf-8").splitlines()
        ]

    def _rebind_rows(self, rows: list[dict]) -> None:
        roots = {row["physical_root_fingerprint"] for row in rows}
        descriptor = probe_builder_module._probe_generation_descriptor(
            facts=self.facts,
            probe_roots=roots,
            probe_rows=rows,
        )
        provenance_id = stable_json_sha256(descriptor)
        for row in rows:
            row["generation_provenance_id"] = provenance_id
        self.rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = probe_builder_module._expected_probe_manifest(
            rows=rows,
            facts=self.facts,
            descriptor=descriptor,
            provenance_id=provenance_id,
            generation_report=self._generation_report(rows),
            rows_name=self.rows_path.name,
            rows_sha256=file_sha256(self.rows_path),
        )
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _validate(self) -> dict:
        with patch.object(
            probe_builder_module,
            "_validated_probe_inputs",
            return_value=self.facts,
        ):
            return validate_recovery_probe_suite_binding(
                rows_path=self.rows_path,
                manifest_path=self.manifest_path,
                scenarios_path=self.root / "scenarios.json",
                scenario_manifest_path=self.root / "scenarios.manifest.json",
                scenario_generator_report_path=self.root / "scenarios.report.json",
                development_holdout=self.root / "holdout.json",
                development_holdout_manifest=self.root / "holdout.manifest.json",
                development_holdout_generator_report=self.root / "holdout.report.json",
                d0_aggregate_dir=self.root / "d0",
                forbidden_suite=self.root / "suite.json",
                evaluation_policy=self.root / "policy.json",
                reviewed_source_commit="a" * 40,
            )

    def test_binding_recomputes_valid_rows_and_manifest(self) -> None:
        self._publish_valid_fixture()
        report = self._validate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["probe_rows"], 24)

    def test_rehashed_probe_laundering_still_fails(self) -> None:
        self._publish_valid_fixture()
        rows = self._read_rows()
        rows[0]["production_label_eligible"] = True
        self.rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["probe_rows"]["sha256"] = file_sha256(self.rows_path)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "row_markers"):
            self._validate()

    def test_normalized_row_digest_rejects_rehashed_action_mutation(self) -> None:
        self._publish_valid_fixture()
        rows = self._read_rows()
        rows[0]["preferred_action"] = {
            "tool": "ask_for_more_evidence",
            "arguments": {"request": "forged"},
        }
        self.rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["probe_rows"]["sha256"] = file_sha256(self.rows_path)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "generation_descriptor"):
            self._validate()

    def test_fully_rebound_observation_laundering_fails_semantic_recompute(self) -> None:
        self._publish_valid_fixture()
        rows = self._read_rows()
        rows[0]["policy_observation"]["last_tool_status"] = "success"
        rows[0]["policy_observation"]["last_tool_output"] = {
            "execution_status": "success"
        }
        self._rebind_rows(rows)
        with self.assertRaisesRegex(ValueError, "row_evidence"):
            self._validate()

    def test_fully_rebound_scenario_identity_spoof_fails_source_binding(self) -> None:
        self._publish_valid_fixture()
        rows = self._read_rows()
        rows[0]["scenario_id"] = "forged-scenario"
        self._rebind_rows(rows)
        with self.assertRaisesRegex(ValueError, "row_evidence"):
            self._validate()

    def test_exact_auxiliary_identity_is_recomputed_after_full_rebind(self) -> None:
        self._publish_valid_fixture()
        for field in ("replay_source", "state_visited_by"):
            with self.subTest(field=field):
                rows = self._read_rows()
                rows[0][field] = "natural_dagger1"
                self._rebind_rows(rows)
                with self.assertRaisesRegex(ValueError, "row_markers"):
                    self._validate()
                self._publish_valid_fixture()

    def test_offline_audit_pass_flag_alone_is_not_evidence(self) -> None:
        self._publish_valid_fixture()
        rows = self._read_rows()
        rows[0]["offline_teacher_target_audit"] = {"passed": True}
        self._rebind_rows(rows)
        with self.assertRaisesRegex(ValueError, "row_evidence"):
            self._validate()

    def test_fixed_twelve_root_quota_is_not_the_ten_root_floor(self) -> None:
        self._publish_valid_fixture(roots_per_stratum=10)
        with self.assertRaisesRegex(ValueError, "root_quotas"):
            self._validate()

    def test_full_manifest_schema_rejects_contradictory_claims(self) -> None:
        self._publish_valid_fixture()
        base = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        contradictions = {
            "contract": "forged_contract",
            "schema_version": 999,
            "source_commit": "f" * 40,
            "state_origin": "natural_dagger1",
            "dataset_source": "natural_dagger1",
            "collection_role": "natural_on_policy",
            "root_quotas": {},
            "distinct_physical_roots": 999,
            "probe_support": {"passed": True},
            "root_disjointness": {"passed": True},
        }
        for field, forged in contradictions.items():
            with self.subTest(field=field):
                manifest = copy.deepcopy(base)
                manifest[field] = forged
                self.manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "manifest_claims"):
                    self._validate()

    def test_generation_report_claims_recompute_exactly(self) -> None:
        self._publish_valid_fixture()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["generation_report"]["root_quotas"][
            "post_failure_no_candidate"
        ] = 10
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "generation_report"):
            self._validate()

    def test_prepublication_input_drift_writes_no_artifact(self) -> None:
        original = copy.deepcopy(self.facts)
        drifted = copy.deepcopy(self.facts)
        drifted["scenarios"][0]["execution"]["drifted"] = True

        class _Environment:
            process_oracle = object()

        with (
            patch.object(
                probe_builder_module,
                "_validated_probe_inputs",
                side_effect=[original, drifted],
            ),
            patch.object(
                probe_builder_module,
                "production_environment_factory",
                return_value=_Environment(),
            ),
            patch.object(
                probe_builder_module,
                "ExpertPolicyOracle",
                return_value=object(),
            ),
            patch.object(
                probe_builder_module,
                "generate_recovery_probes",
                return_value=([], {"passed": False}),
            ),
            patch.object(
                probe_builder_module,
                "_factory_identities",
                return_value={},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                build_recovery_probe_suite(
                    scenarios_path=self.root / "scenarios.json",
                    scenario_manifest_path=self.root / "scenarios.manifest.json",
                    scenario_generator_report_path=self.root / "scenarios.report.json",
                    output=self.rows_path,
                    manifest_path=self.manifest_path,
                    reviewed_source_commit="a" * 40,
                    forbidden_suite=self.root / "suite.json",
                    evaluation_policy=self.root / "policy.json",
                    development_holdout=self.root / "holdout.json",
                    development_holdout_manifest=self.root
                    / "holdout.manifest.json",
                    development_holdout_generator_report=self.root
                    / "holdout.report.json",
                    d0_aggregate_dir=self.root / "d0",
                )
        self.assertFalse(self.rows_path.exists())
        self.assertFalse(self.manifest_path.exists())

    def test_builder_prevalidates_and_publishes_exact_bytes(self) -> None:
        rows = self._fixture_rows()
        generation_report = self._generation_report(rows)

        class _Environment:
            process_oracle = object()

        with (
            patch.object(
                probe_builder_module,
                "_validated_probe_inputs",
                return_value=self.facts,
            ),
            patch.object(
                probe_builder_module,
                "production_environment_factory",
                return_value=_Environment(),
            ),
            patch.object(
                probe_builder_module,
                "ExpertPolicyOracle",
                return_value=object(),
            ),
            patch.object(
                probe_builder_module,
                "generate_recovery_probes",
                return_value=(rows, generation_report),
            ),
            patch.object(
                probe_builder_module,
                "_factory_identities",
                return_value={},
            ),
        ):
            manifest = build_recovery_probe_suite(
                scenarios_path=self.root / "scenarios.json",
                scenario_manifest_path=self.root / "scenarios.manifest.json",
                scenario_generator_report_path=self.root / "scenarios.report.json",
                output=self.rows_path,
                manifest_path=self.manifest_path,
                reviewed_source_commit="a" * 40,
                forbidden_suite=self.root / "suite.json",
                evaluation_policy=self.root / "policy.json",
                development_holdout=self.root / "holdout.json",
                development_holdout_manifest=self.root / "holdout.manifest.json",
                development_holdout_generator_report=self.root
                / "holdout.report.json",
                d0_aggregate_dir=self.root / "d0",
            )
            self.assertTrue(manifest["passed"])
            self.assertEqual(
                manifest["probe_rows"]["sha256"],
                file_sha256(self.rows_path),
            )
            self.assertTrue(self._validate()["passed"])

    def test_protected_root_overlap_fails_even_with_rehashed_manifest(self) -> None:
        self._publish_valid_fixture()
        self.facts["d0_roots"] = {next(iter(self.roots))}
        with self.assertRaisesRegex(ValueError, "protected_disjointness"):
            self._validate()


if __name__ == "__main__":
    unittest.main()
