from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from psse_env.dagger import release_factories
from psse_env.dagger.evaluator import (
    EVALUATION_SUITES,
    _load_import_spec,
    fingerprint_evaluation_suites,
    validate_release_scenario_suites,
)
from psse_env.dagger.splits import physical_root_fingerprint
from psse_env.dagger.suite_builder import (
    BC0_BUILDER_PACKAGE_VERSIONS,
    BC0_BUILDER_PYTHON_MAJOR_MINOR,
    BC0_SUITE_FAMILY_QUOTAS,
    BC0_SUITE_GENERATION_SEED,
    BC0_SUITE_SEED,
    BC0_SUITE_SOURCE_PARTITION,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_POLICY_PATH,
    REPO_ROOT,
    _TrackedArtifactScenarioGenerator,
    _builder_environment_descriptor,
    _opendss_runtime_version,
    allocate_suite_roots,
    build_bc0_suite,
    canonical_json_bytes,
    family_plan_from_policy,
    partition_release_scenario_v1,
    _tracked_release_inputs,
    validate_quota_matrix,
    validate_builder_environment,
    write_frozen_suite,
)
from psse_env.providers.scenario_generator import ScenarioRejected
from psse_env.sft.provenance import file_sha256


def _scenario(family: str, index: int) -> dict:
    scenario_id = f"scenario-{family}-{index:03d}"
    family_code = sum(
        position * ord(character)
        for position, character in enumerate(family, start=1)
    )
    observed_value = float(family_code * 100 + index)
    truth = {
        "true_measurement_errors": (
            []
            if family == "no_error"
            else [
                {
                    "index": index % 8,
                    "observed": observed_value,
                    "clean": 0.0,
                }
            ]
        )
    }
    return {
        "scenario_id": scenario_id,
        "root_scenario_id": scenario_id,
        "scenario_family": family,
        "network_case": "case14",
        "case": "case14",
        "measurements": [observed_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "metadata": {},
        "semantic_field_provenance": {"measurements": "deployment_sensor:test"},
        "clean_case": "case14",
        "clean_measurements": [0.0] * 8,
        "error_cardinality": len(truth["true_measurement_errors"]),
        "source_tier": "test_fixture",
        **truth,
    }


class BC0SuiteBuilderTests(unittest.TestCase):
    def test_builder_environment_descriptor_is_version_only(self) -> None:
        package_versions = dict(BC0_BUILDER_PACKAGE_VERSIONS)
        with mock.patch(
            "psse_env.dagger.suite_builder.platform.python_version",
            return_value="3.13.13",
        ), mock.patch(
            "psse_env.dagger.suite_builder.importlib.metadata.version",
            side_effect=package_versions.__getitem__,
        ):
            descriptor = _builder_environment_descriptor()

        self.assertEqual(
            descriptor,
            {
                "python_version": "3.13.13",
                "packages": package_versions,
            },
        )

    def test_builder_environment_contract_accepts_documented_stack(self) -> None:
        descriptor = {
            "python_version": "3.12.11",
            "packages": dict(BC0_BUILDER_PACKAGE_VERSIONS),
        }
        with mock.patch(
            "psse_env.dagger.suite_builder._builder_environment_descriptor",
            return_value=descriptor,
        ), mock.patch(
            "psse_env.dagger.suite_builder._opendss_runtime_version",
            return_value="approved native runtime",
        ):
            self.assertEqual(
                validate_builder_environment(),
                {
                    **descriptor,
                    "local_diagnostic_build": False,
                    "release_reproducible": True,
                },
            )

        self.assertEqual(BC0_BUILDER_PYTHON_MAJOR_MINOR, (3, 12))

    def test_builder_environment_contract_rejects_runtime_drift(self) -> None:
        approved = {
            "python_version": "3.12.11",
            "packages": dict(BC0_BUILDER_PACKAGE_VERSIONS),
        }
        drifted_environments = {
            "python_minor": {**approved, "python_version": "3.13.0"},
            **{
                distribution: {
                    **approved,
                    "packages": {
                        **approved["packages"],
                        distribution: "unexpected-version",
                    },
                }
                for distribution in BC0_BUILDER_PACKAGE_VERSIONS
            },
        }
        for name, descriptor in drifted_environments.items():
            with self.subTest(name=name), mock.patch(
                "psse_env.dagger.suite_builder._builder_environment_descriptor",
                return_value=descriptor,
            ), mock.patch(
                "psse_env.dagger.suite_builder._opendss_runtime_version",
                return_value="approved native runtime",
            ), self.assertRaisesRegex(RuntimeError, "environment is not approved"):
                validate_builder_environment()

    def test_builder_environment_contract_rejects_unloadable_native_runtime(self) -> None:
        descriptor = {
            "python_version": "3.12.11",
            "packages": dict(BC0_BUILDER_PACKAGE_VERSIONS),
        }
        with mock.patch(
            "psse_env.dagger.suite_builder._builder_environment_descriptor",
            return_value=descriptor,
        ), mock.patch(
            "psse_env.dagger.suite_builder._opendss_runtime_version",
            side_effect=RuntimeError("OpenDSSDirect native runtime could not be loaded"),
        ), self.assertRaisesRegex(RuntimeError, "native runtime could not be loaded"):
            validate_builder_environment()

    def test_native_runtime_smoke_checks_loaded_engine_versions(self) -> None:
        banner = (
            "DSS C-API Library version 0.14.5; "
            "DSS-Python version: 0.15.7; "
            "OpenDSSDirect.py version: 0.9.4"
        )
        runtime = mock.Mock()
        runtime.Basic.Version.return_value = banner
        with mock.patch.dict("sys.modules", {"opendssdirect": runtime}):
            self.assertEqual(_opendss_runtime_version(), banner)

    def test_build_rejects_unapproved_runtime_before_reading_inputs(self) -> None:
        descriptor = {
            "python_version": "3.13.13",
            "packages": dict(BC0_BUILDER_PACKAGE_VERSIONS),
        }
        with mock.patch(
            "psse_env.dagger.suite_builder._builder_environment_descriptor",
            return_value=descriptor,
        ), mock.patch(
            "psse_env.dagger.suite_builder._opendss_runtime_version",
            return_value="approved native runtime",
        ), mock.patch(
            "psse_env.dagger.suite_builder.family_plan_from_policy"
        ) as family_plan, self.assertRaisesRegex(
            RuntimeError, "Python '3.13.13'"
        ):
            build_bc0_suite(validate_physics=False)

        family_plan.assert_not_called()

    def test_check_ignores_report_environment_when_suite_bytes_match(self) -> None:
        suites = {"standard_success": []}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite.json"
            output.write_bytes(canonical_json_bytes(suites))
            with mock.patch(
                "psse_env.dagger.suite_builder.build_bc0_suite",
                return_value=(
                    suites,
                    {
                        "builder_environment": {
                            "python_version": "different-but-recorded",
                            "packages": {},
                        }
                    },
                ),
            ):
                report = write_frozen_suite(
                    output,
                    check=True,
                    validate_physics=False,
                )

        self.assertEqual(
            report["builder_environment"]["python_version"],
            "different-but-recorded",
        )

    def test_generation_and_evaluator_seeds_are_explicitly_separated(self) -> None:
        self.assertEqual(BC0_SUITE_SEED, 20260719)
        self.assertEqual(BC0_SUITE_GENERATION_SEED, 20260734)
        self.assertNotEqual(BC0_SUITE_GENERATION_SEED, BC0_SUITE_SEED)
        self.assertEqual(BC0_SUITE_SOURCE_PARTITION, "evaluation")

    def test_policy_plan_and_quota_matrix_cover_ten_release_families(self) -> None:
        plan = family_plan_from_policy()
        validate_quota_matrix(plan)
        self.assertEqual(len(plan), 10)
        self.assertEqual(sum(plan.values()), 115)
        self.assertEqual(set(BC0_SUITE_FAMILY_QUOTAS), set(EVALUATION_SUITES))
        expected_suite_sizes = {suite: 21 for suite in EVALUATION_SUITES}
        expected_suite_sizes["efficiency"] = 31
        self.assertEqual(
            {suite: sum(rows.values()) for suite, rows in BC0_SUITE_FAMILY_QUOTAS.items()},
            expected_suite_sizes,
        )

        family_policy = json.loads(
            DEFAULT_POLICY_PATH.read_text(encoding="utf-8")
        )["family_policy"]
        self.assertEqual(
            family_policy["multi_measurement"],
            {
                "minimum_physical_roots": 20,
                "minimum_audited_completion_rate": 0.0,
                "maximum_unqualified_operator_escalation_rate": 1.0,
            },
        )
        for family in ("measurement+parameter", "measurement+topology"):
            with self.subTest(family=family):
                self.assertEqual(
                    family_policy[family],
                    {
                        "minimum_physical_roots": 22,
                        "minimum_audited_completion_rate": 0.95,
                        "maximum_unqualified_operator_escalation_rate": 0.05,
                    },
                )
        generic_handoff_families = {
            family
            for family, contract in family_policy.items()
            if contract["minimum_audited_completion_rate"] == 0.0
            and contract["maximum_unqualified_operator_escalation_rate"] == 1.0
        }
        for suite in ("forced_error_recovery", "invalid_action_recovery"):
            with self.subTest(suite=suite):
                self.assertTrue(
                    generic_handoff_families.isdisjoint(
                        BC0_SUITE_FAMILY_QUOTAS[suite]
                    )
                )
        self.assertEqual(
            BC0_SUITE_FAMILY_QUOTAS["partial_success_retention"][
                "multi_measurement"
            ],
            16,
        )

    def test_packaged_suite_policy_and_factories_are_content_coherent(self) -> None:
        policy = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
        suites = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
        suite_policy = policy["suite_policy"]
        self.assertEqual(suite_policy["status"], "pinned")
        self.assertEqual(
            suite_policy["approved_suite_sha256"], file_sha256(DEFAULT_OUTPUT_PATH)
        )

        contract = fingerprint_evaluation_suites(
            suites,
            seed=suite_policy["evaluator_seed"],
            required_suites=tuple(suite_policy["required_suites"]),
            minimum_suites=len(suite_policy["required_suites"]),
            minimum_episodes_per_suite=1,
            minimum_roots_per_suite=suite_policy[
                "minimum_physical_roots_per_suite"
            ],
        )
        manifest_fields = (
            "suite_manifest",
            "suite_content_hashes",
            "suite_root_set_hashes",
            "suite_content_sha256",
            "root_set_sha256",
        )
        self.assertEqual(
            suite_policy["approved_suite_manifest"],
            {field: contract[field] for field in manifest_fields},
        )
        roots = [
            row["grouping"]["physical_root_fingerprint"]
            for suite_rows in suites.values()
            for row in suite_rows
        ]
        self.assertEqual(len(roots), 115)
        self.assertEqual(len(set(roots)), 115)
        self.assertTrue(all(root.startswith("physical_v3_") for root in roots))

        module_sha256 = file_sha256(Path(release_factories.__file__))
        expected_specs = {
            "environment": "production_environment_factory",
            "expert_policy": "observable_expert_policy_factory",
            "model_policy": "gemma_release_policy_factory",
            "case_loader": "deterministic_case_loader",
        }
        for role, callable_name in expected_specs.items():
            with self.subTest(factory_role=role):
                import_spec = (
                    "psse_env.dagger.release_factories:" + callable_name
                )
                self.assertEqual(
                    policy["approved_factories"][role],
                    [
                        {
                            "import_spec": import_spec,
                            "source_sha256": module_sha256,
                        }
                    ],
                )
                self.assertIs(
                    _load_import_spec(import_spec, field=f"{role} factory"),
                    getattr(release_factories, callable_name),
                )

    def test_partition_is_schema_v1_and_keeps_truth_out_of_execution(self) -> None:
        flat = _scenario("measurement", 3)
        envelope = partition_release_scenario_v1(flat)
        validate_release_scenario_suites({"standard_success": [envelope]})
        self.assertEqual(
            set(envelope),
            {"scenario_schema_version", "execution", "audit", "grouping"},
        )
        self.assertNotIn("true_measurement_errors", envelope["execution"])
        self.assertEqual(
            envelope["audit"]["truth"]["true_measurement_errors"],
            flat["true_measurement_errors"],
        )
        self.assertTrue(
            envelope["grouping"]["physical_root_fingerprint"].startswith("physical_v3_")
        )
        self.assertEqual(
            envelope["grouping"]["physical_root_fingerprint"],
            physical_root_fingerprint(flat),
        )

    def test_partition_removes_hif_corpus_only_evaluation_labels(self) -> None:
        flat = _scenario("hif", 2)
        flat["metadata"] = {
            "hif_scan_window": {
                "scans": [{"z_obs": [1.0], "z_clean": [0.0]}]
            },
            "nlm_diagnostic": {
                "success": True,
                "top_hif_groups": [{"branch_row0": 3}],
                "detected": True,
                "detected_top1": True,
                "detected_top3": True,
            },
        }
        envelope = partition_release_scenario_v1(flat)
        scan = envelope["execution"]["metadata"]["hif_scan_window"]["scans"][0]
        diagnostic = envelope["execution"]["metadata"]["nlm_diagnostic"]
        self.assertNotIn("z_clean", scan)
        self.assertFalse(
            {"detected", "detected_top1", "detected_top3"} & set(diagnostic)
        )
        self.assertEqual(diagnostic["top_hif_groups"], [{"branch_row0": 3}])

    def test_allocation_is_deterministic_unique_and_semantically_marked(self) -> None:
        plan = family_plan_from_policy()
        rows = [
            _scenario(family, index)
            for family, count in plan.items()
            for index in range(count)
        ]
        first = allocate_suite_roots(rows, plan=plan)
        second = allocate_suite_roots(list(reversed(copy.deepcopy(rows))), plan=plan)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        roots = [
            row["grouping"]["physical_root_fingerprint"]
            for suite_rows in first.values()
            for row in suite_rows
        ]
        self.assertEqual(len(roots), 115)
        self.assertEqual(len(set(roots)), 115)
        expected_kinds = {
            "standard_success": "none",
            "forced_error_recovery": "pre_policy_failure",
            "partial_success_retention": "committed_partial_correction",
            "invalid_action_recovery": "pre_policy_failure",
            "efficiency": "efficiency_budget",
        }
        for suite, suite_rows in first.items():
            self.assertEqual(
                len(suite_rows), 31 if suite == "efficiency" else 21
            )
            self.assertEqual(
                {
                    row["audit"]["evaluation_intervention"]["kind"]
                    for row in suite_rows
                },
                {expected_kinds[suite]},
            )
        partial = first["partial_success_retention"][0]["audit"][
            "evaluation_intervention"
        ]
        self.assertEqual(len(partial["setup_actions"]), 4)
        self.assertTrue(partial["retention_required"])

    def test_quota_matrix_fails_closed_on_drift(self) -> None:
        plan = family_plan_from_policy()
        changed = copy.deepcopy(BC0_SUITE_FAMILY_QUOTAS)
        changed["standard_success"]["no_error"] -= 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_quota_matrix(plan, changed)

    def test_partial_roots_are_reserved_by_admissibility(self) -> None:
        plan = family_plan_from_policy()
        rows = [
            _scenario(family, index)
            for family, count in plan.items()
            for index in range(count)
        ]

        suites = allocate_suite_roots(
            rows,
            plan=plan,
            partial_setup_validator=lambda row: int(
                row["execution"]["scenario_id"].rsplit("-", 1)[1]
            )
            >= 2,
        )

        self.assertTrue(
            all(
                int(row["execution"]["scenario_id"].rsplit("-", 1)[1]) >= 2
                for row in suites["partial_success_retention"]
            )
        )

    def test_source_allowlist_is_git_tracked_and_content_addressable(self) -> None:
        inputs = _tracked_release_inputs()
        relative = {path.relative_to(REPO_ROOT).as_posix() for path in inputs}
        self.assertIn("data/measurements_5class_merged.jsonl", relative)
        self.assertIn("mcp_server/case14.m", relative)
        self.assertTrue(
            any("cases_parameter_error" in name for name in relative)
        )

    def test_artifact_resolver_rejects_untracked_and_symlink_shadows(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as directory:
            root = Path(directory)
            cases = root / "cases_parameter_error"
            cases.mkdir()
            allowed = cases / "allowed.m"
            allowed.write_text("tracked", encoding="utf-8")
            shadow = cases / "shadow.m"
            shadow.write_text("ignored", encoding="utf-8")
            link = cases / "link.m"
            link.symlink_to(allowed)
            generator = _TrackedArtifactScenarioGenerator(
                balanced_artifact_dir=root,
                hif_sample_paths=[],
                tracked_inputs={allowed.absolute()},
                validate=False,
            )
            self.assertEqual(
                generator._local_artifact("allowed.m", "cases_parameter_error"),
                allowed.absolute(),
            )
            for name in ("shadow.m", "link.m"):
                with self.subTest(name=name), self.assertRaises(ScenarioRejected):
                    generator._local_artifact(name, "cases_parameter_error")


if __name__ == "__main__":
    unittest.main()
