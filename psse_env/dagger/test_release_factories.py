from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    FINALIZE_DIAGNOSIS,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_WLS,
)
from psse_env.dagger import release_factories as factories
from psse_env.dagger.evaluator import (
    ClosedLoopRolloutEvaluator,
    _call_factory,
    _load_import_spec,
)
from psse_env.sft.gates import GateError


class ReleaseEnvironmentFactoryTests(unittest.TestCase):
    def test_factory_enforces_production_deployment_contract(self) -> None:
        class Providers:
            def __init__(
                self,
                *,
                chi2_alpha: float,
                parameter_ranking_dominance_threshold: float,
                hif_alpha_grid_size: int,
                hif_r_grid_size: int,
                hif_max_scans: int,
            ) -> None:
                self.chi2_alpha = chi2_alpha
                self.parameter_ranking_dominance_threshold = (
                    parameter_ranking_dominance_threshold
                )
                self.hif_alpha_grid_size = hif_alpha_grid_size
                self.hif_r_grid_size = hif_r_grid_size
                self.hif_max_scans = hif_max_scans

            def env_kwargs(self) -> dict[str, Any]:
                return {
                    "provider_marker": "deployment",
                    "chi2_alpha": self.chi2_alpha,
                    "parameter_ranking_dominance_threshold": (
                        self.parameter_ranking_dominance_threshold
                    ),
                    "hif_alpha_grid_size": self.hif_alpha_grid_size,
                    "hif_r_grid_size": self.hif_r_grid_size,
                    "hif_max_scans": self.hif_max_scans,
                }

        class CandidateOracle:
            mode = "deployment"

        class Environment:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self.production_dataset_mode = kwargs["production_dataset_mode"]
                self.candidate_quality_oracle = CandidateOracle()
                self.validations = 0

            def validate_production_configuration(self) -> None:
                self.validations += 1

        with (
            mock.patch.object(factories, "MatpowerDeploymentProviders", Providers),
            mock.patch.object(factories, "TransactionalPSSEEnv", Environment),
        ):
            env = factories.production_environment_factory(seed=7, rng=object())

        self.assertIs(env.production_dataset_mode, True)
        self.assertEqual(env.candidate_quality_oracle.mode, "deployment")
        self.assertEqual(env.kwargs["provider_marker"], "deployment")
        self.assertEqual(env.kwargs["chi2_alpha"], factories.BC0_CHI2_ALPHA)
        self.assertEqual(
            env.kwargs["parameter_ranking_dominance_threshold"],
            factories.BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
        )
        self.assertEqual(
            factories.BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
            1.0,
        )
        self.assertEqual(
            env.kwargs["hif_alpha_grid_size"],
            factories.BC0_HIF_ALPHA_GRID_SIZE,
        )
        self.assertEqual(
            env.kwargs["hif_r_grid_size"],
            factories.BC0_HIF_R_GRID_SIZE,
        )
        self.assertEqual(
            env.kwargs["hif_max_scans"],
            factories.BC0_HIF_MAX_SCANS,
        )
        self.assertEqual(env.kwargs["max_steps"], 24)
        self.assertEqual(env.kwargs["history_window"], 4)
        self.assertEqual(env.validations, 1)

    def test_factory_rejects_non_deployment_candidate_oracle(self) -> None:
        class Providers:
            def __init__(
                self,
                *,
                chi2_alpha: float,
                parameter_ranking_dominance_threshold: float,
                hif_alpha_grid_size: int,
                hif_r_grid_size: int,
                hif_max_scans: int,
            ) -> None:
                self.chi2_alpha = chi2_alpha
                self.parameter_ranking_dominance_threshold = (
                    parameter_ranking_dominance_threshold
                )
                self.hif_alpha_grid_size = hif_alpha_grid_size
                self.hif_r_grid_size = hif_r_grid_size
                self.hif_max_scans = hif_max_scans

            def env_kwargs(self) -> dict[str, Any]:
                return {}

        class Environment:
            production_dataset_mode = True
            candidate_quality_oracle = types.SimpleNamespace(mode="synthetic")

            def __init__(self, **_kwargs: Any) -> None:
                pass

        with (
            mock.patch.object(factories, "MatpowerDeploymentProviders", Providers),
            mock.patch.object(factories, "TransactionalPSSEEnv", Environment),
            self.assertRaisesRegex(RuntimeError, "deployment mode"),
        ):
            factories.production_environment_factory()


class FactoryImportSpecTests(unittest.TestCase):
    MODULE = "psse_env.dagger.release_factories"

    def test_exact_release_import_specs_resolve_and_match_evaluator_calls(self) -> None:
        environment_factory = _load_import_spec(
            f"{self.MODULE}:production_environment_factory", field="environment"
        )
        expert_factory = _load_import_spec(
            f"{self.MODULE}:observable_expert_policy_factory", field="expert"
        )
        model_factory = _load_import_spec(
            f"{self.MODULE}:gemma_release_policy_factory", field="model"
        )
        case_loader = _load_import_spec(
            f"{self.MODULE}:deterministic_case_loader", field="case_loader"
        )
        self.assertIs(environment_factory, factories.production_environment_factory)
        self.assertIs(expert_factory, factories.observable_expert_policy_factory)
        self.assertIs(model_factory, factories.gemma_release_policy_factory)
        self.assertIs(case_loader, factories.deterministic_case_loader)

        environment = _call_factory(environment_factory, 101)
        self.assertIs(environment.production_dataset_mode, True)
        self.assertEqual(environment.candidate_quality_oracle.mode, "deployment")

        expert = _call_factory(
            expert_factory,
            102,
            policy_identity={
                "explicit_policy_identity": factories.EXPERT_POLICY_IDENTITY,
                "model_id": None,
                "model_revision": None,
            },
        )
        self.assertEqual(
            expert.release_policy_identity["explicit_policy_identity"],
            factories.EXPERT_POLICY_IDENTITY,
        )

        bundle = factories._ModelBundle(
            model=object(),
            processor=object(),
            model_id=factories.BASE_MODEL_ID,
            model_revision=factories.BASE_MODEL_REVISION,
        )
        with mock.patch.object(
            factories, "_cached_model_bundle", return_value=bundle
        ) as cache:
            model_policy = _call_factory(
                model_factory,
                103,
                policy_identity={
                    "explicit_policy_identity": None,
                    "model_id": factories.BASE_MODEL_ID,
                    "model_revision": factories.BASE_MODEL_REVISION,
                },
            )
        cache.assert_called_once_with(
            factories.BASE_MODEL_ID, factories.BASE_MODEL_REVISION
        )
        self.assertEqual(
            model_policy.release_policy_identity["model_revision"],
            factories.BASE_MODEL_REVISION,
        )

        parsed = {
            "baseMVA": 100.0,
            "bus": [],
            "gen": [],
            "branch": [],
        }
        with mock.patch(
            "mcp_server.matpower_server._load_python_case", return_value=parsed
        ) as parser:
            self.assertEqual(case_loader({"case_path": "case14"}), parsed)
        parser.assert_called_once_with(
            str((factories._REPO_ROOT / "mcp_server" / "case14.m").resolve())
        )


class ObservableExpertFactoryTests(unittest.TestCase):
    def test_factory_requires_and_exposes_exact_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "bc0-observable-expert-v1"):
            factories.observable_expert_policy_factory(policy_identity="other")

        policy = factories.observable_expert_policy_factory(
            policy_identity=factories.EXPERT_POLICY_IDENTITY
        )
        self.assertEqual(
            policy.release_policy_identity,
            {
                "explicit_policy_identity": "bc0-observable-expert-v1",
                "model_id": None,
                "model_revision": None,
            },
        )
        mutable_view = policy.release_policy_identity
        mutable_view["explicit_policy_identity"] = "tampered"
        self.assertEqual(
            policy.release_policy_identity["explicit_policy_identity"],
            factories.EXPERT_POLICY_IDENTITY,
        )

    def test_policy_uses_only_a_copied_observation_and_rejects_truth(self) -> None:
        class RecordingExpert:
            def __init__(self) -> None:
                self.state: dict[str, Any] | None = None
                self.history: list[Mapping[str, Any]] | None = None

            def next_actions(
                self,
                state: Mapping[str, Any],
                history: list[Mapping[str, Any]],
            ) -> list[dict[str, Any]]:
                self.state = dict(state)
                self.history = history
                self.state["active_state_id"] = "mutated"
                return [{"tool": RUN_WLS, "arguments": {"state_id": "active-real"}}]

        expert = RecordingExpert()
        policy = factories.ObservableExpertPolicy(expert)  # type: ignore[arg-type]
        observation = {
            "active_state_id": "active-real",
            "history_window": [
                {"action": {"tool": RUN_WLS, "arguments": {"state_id": "old"}}}
            ],
        }
        original = copy.deepcopy(observation)
        action = policy.act(observation)
        self.assertEqual(action["tool"], RUN_WLS)
        self.assertEqual(observation, original)
        self.assertIsNot(expert.history, observation["history_window"])

        with self.assertRaisesRegex(ValueError, "Privileged fields"):
            policy.act({**observation, "hidden_truth": {"fault": "measurement"}})

    def test_verified_partial_candidate_is_committed_from_observable_metrics(self) -> None:
        candidate = "episode:candidate-1"
        history = [
            {
                "action": {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "episode:active-1",
                        "measurement_updates": {"3": 1.0},
                    },
                },
                "tool_output": {
                    "execution_status": "success",
                    "candidate_state_id": candidate,
                },
            }
        ]
        observation = {
            "candidate_state_id": candidate,
            "has_verified_candidate": True,
            "last_tool": RUN_WLS,
            "last_verification": {
                "execution_status": "success",
                "state_id": candidate,
                "evidence_source": "configured_provider:matpower_wls",
                "physical_constraints_ok": True,
                "target_metric_value": 0.002,
                "target_metric_threshold": 3.0,
                "global_progress": 0.545,
                "globally_resolved": False,
            },
        }
        self.assertEqual(
            factories._observable_candidate_disposition_action(observation, history),
            {
                "tool": COMMIT_STATE,
                "arguments": {"candidate_state_id": candidate},
            },
        )

        unsafe = copy.deepcopy(observation)
        unsafe["last_verification"]["physical_constraints_ok"] = False
        self.assertEqual(
            factories._observable_candidate_disposition_action(unsafe, history)["tool"],
            ROLLBACK_STATE,
        )

    def test_inconclusive_candidate_gets_one_evidence_attempt_then_rollback(self) -> None:
        candidate = "episode:candidate-1"
        observation = {
            "candidate_state_id": candidate,
            "has_verified_candidate": True,
            "last_tool": RUN_WLS,
            "last_verification": {},
        }
        self.assertEqual(
            factories._observable_candidate_disposition_action(observation, [])["tool"],
            ASK_FOR_MORE_EVIDENCE,
        )
        observation["last_tool"] = ASK_FOR_MORE_EVIDENCE
        self.assertEqual(
            factories._observable_candidate_disposition_action(observation, [])["tool"],
            ROLLBACK_STATE,
        )


class RealProductionExpertRecoveryTests(unittest.TestCase):
    @staticmethod
    def _evaluator(*, required_suite: str) -> ClosedLoopRolloutEvaluator:
        return ClosedLoopRolloutEvaluator(
            env_factory=factories.production_environment_factory,
            policy_factory=factories.observable_expert_policy_factory,
            case_loader=factories.deterministic_case_loader,
            max_steps=24,
            seed=20260719,
            required_suites=(required_suite,),
            minimum_suites=1,
            minimum_episodes_per_suite=1,
            minimum_roots_per_suite=1,
            require_release_environment=True,
            expected_policy_identity={
                "explicit_policy_identity": factories.EXPERT_POLICY_IDENTITY,
                "model_id": None,
                "model_revision": None,
            },
            require_policy_identity=True,
        )

    def test_release_alpha_preserves_every_frozen_no_error_root(self) -> None:
        suite_path = (
            Path(factories.__file__).with_name("suites") / "bc0_eval_suite_v1.json"
        )
        suites = json.loads(suite_path.read_text(encoding="utf-8"))
        scenarios = [
            row
            for row in suites["standard_success"]
            if row["grouping"]["scenario_family"] == "no_error"
        ]
        result = self._evaluator(required_suite="standard_success").evaluate(
            {"standard_success": scenarios}
        )
        self.assertEqual(len(result.suite_metrics["episodes"]), 4)
        for episode in result.suite_metrics["episodes"]:
            with self.subTest(scenario_id=episode["scenario_id"]):
                self.assertIs(episode["final_physical_success"], True)
                self.assertEqual(episode["invalid_action_count"], 0)
                self.assertEqual(
                    [row["action"]["tool"] for row in episode["trace"]],
                    [RUN_WLS, FINALIZE_DIAGNOSIS],
                )

    def test_every_frozen_parameter_route_resolves_with_release_expert(self) -> None:
        suite_path = (
            Path(factories.__file__).with_name("suites") / "bc0_eval_suite_v1.json"
        )
        suites = json.loads(suite_path.read_text(encoding="utf-8"))
        target_families = {"parameter", "measurement+parameter"}
        selected = {
            suite_name: [
                row
                for row in rows
                if row["grouping"]["scenario_family"] in target_families
            ]
            for suite_name, rows in suites.items()
        }
        selected = {
            suite_name: rows for suite_name, rows in selected.items() if rows
        }
        self.assertEqual(sum(len(rows) for rows in selected.values()), 30)

        observed_regressions: set[str] = set()
        prior_failures = {
            "r0_a80744a6c25e",
            "r0_b8173b30f6a6",
            "r0_8e0647a30ae4",
            "r0_b91e784871bd",
            "r0_026579c5ac67",
            "r0_33817d90f478",
            "r0_14bf9b268327",
        }
        for suite_name, scenarios in selected.items():
            result = self._evaluator(required_suite=suite_name).evaluate(
                {suite_name: scenarios}
            )
            self.assertEqual(
                len(result.suite_metrics["episodes"]), len(scenarios)
            )
            for episode in result.suite_metrics["episodes"]:
                scenario_id = episode["scenario_id"]
                family = episode["family"]
                observed_regressions.add(scenario_id)
                tools = [row["action"]["tool"] for row in episode["trace"]]
                with self.subTest(
                    suite=suite_name,
                    scenario_id=scenario_id,
                    family=family,
                ):
                    self.assertIs(episode["terminal"], True)
                    self.assertEqual(episode["terminal_outcome"], "resolved")
                    self.assertIs(episode["final_physical_success"], True)
                    self.assertIn(CORRECT_PARAMETERS, tools)
                    if family == "measurement+parameter":
                        self.assertIn(CORRECT_MEASUREMENTS, tools)
                    self.assertEqual(episode["false_commit_count"], 0)
                    self.assertEqual(episode["false_finalization_count"], 0)
                    self.assertEqual(episode["false_rollback_count"], 0)
                    self.assertIs(episode["loop_detected"], False)

        self.assertTrue(prior_failures <= observed_regressions)

    def test_partial_multi_measurement_root_hands_off_without_invalid_actions(self) -> None:
        suite_path = (
            Path(factories.__file__).with_name("suites") / "bc0_eval_suite_v1.json"
        )
        suites = json.loads(suite_path.read_text(encoding="utf-8"))
        scenario = next(
            row
            for row in suites["partial_success_retention"]
            if row["grouping"]["scenario_family"] == "multi_measurement"
        )
        evaluator = self._evaluator(required_suite="partial_success_retention")
        result = evaluator.evaluate(
            {"partial_success_retention": [scenario]}
        )
        episode = result.suite_metrics["episodes"][0]
        self.assertIs(episode["terminal"], True)
        self.assertEqual(episode["terminal_outcome"], "operator_escalation")
        self.assertEqual(episode["invalid_action_count"], 0)
        self.assertIs(episode["healthy_preservation_known"], True)
        self.assertIs(episode["healthy_components_preserved"], True)
        self.assertEqual(episode["false_commit_count"], 0)
        self.assertEqual(episode["false_finalization_count"], 0)
        self.assertIn(COMMIT_STATE, [row["action"]["tool"] for row in episode["trace"]])

    def test_standard_multi_measurement_roots_handoff_without_invalid_actions(self) -> None:
        suite_path = (
            Path(factories.__file__).with_name("suites") / "bc0_eval_suite_v1.json"
        )
        suites = json.loads(suite_path.read_text(encoding="utf-8"))
        scenarios = [
            row
            for row in suites["standard_success"]
            if row["grouping"]["scenario_family"] == "multi_measurement"
        ]
        self.assertEqual(len(scenarios), 4)

        result = self._evaluator(required_suite="standard_success").evaluate(
            {"standard_success": scenarios}
        )

        for episode in result.suite_metrics["episodes"]:
            with self.subTest(scenario_id=episode["scenario_id"]):
                self.assertIs(episode["terminal"], True)
                self.assertEqual(
                    episode["terminal_outcome"], "operator_escalation"
                )
                self.assertEqual(episode["invalid_action_count"], 0)
                self.assertIs(episode["loop_detected"], False)
                self.assertEqual(episode["false_commit_count"], 0)
                self.assertEqual(episode["false_finalization_count"], 0)
                self.assertEqual(
                    [
                        row["error_code"]
                        for row in episode["trace"]
                        if row["execution_status"] == "failure"
                    ],
                    [],
                )
                self.assertEqual(
                    episode["trace"][-1]["action"]["tool"],
                    ASK_FOR_MORE_EVIDENCE,
                )
                # Which exhausted-handoff path fires depends on the root's
                # remaining step budget; both are designed safe escalations.
                self.assertIn(
                    episode["trace"][-1]["action"]["arguments"]["request"],
                    {
                        RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                        RECOVERY_BUDGET_EXHAUSTED_REQUEST,
                    },
                )

    def test_every_forced_five_meter_root_terminates_without_invalid_actions(self) -> None:
        suite_path = (
            Path(factories.__file__).with_name("suites") / "bc0_eval_suite_v1.json"
        )
        suites = json.loads(suite_path.read_text(encoding="utf-8"))
        scenarios = [
            row
            for row in suites["forced_error_recovery"]
            if row["grouping"]["scenario_family"] == "multi_measurement"
            and int(row["grouping"]["error_cardinality"]) >= 5
        ]
        self.assertGreaterEqual(len(scenarios), 2)
        result = self._evaluator(required_suite="forced_error_recovery").evaluate(
            {"forced_error_recovery": scenarios}
        )
        self.assertEqual(len(result.suite_metrics["episodes"]), len(scenarios))
        for episode in result.suite_metrics["episodes"]:
            with self.subTest(scenario_id=episode["scenario_id"]):
                policy_trace = [
                    row for row in episode["trace"] if row["intervention"] is False
                ]

                # The injected failed WLS is history, not state evidence.  The
                # policy must retry the observable baseline immediately rather
                # than emit an invalid no-action transition.
                self.assertEqual(policy_trace[0]["action"]["tool"], RUN_WLS)
                self.assertIs(episode["terminal"], True)
                self.assertIn(
                    episode["terminal_outcome"], {"resolved", "operator_escalation"}
                )
                self.assertIs(episode["healthy_preservation_known"], True)
                self.assertIs(episode["healthy_components_preserved"], True)
                self.assertEqual(episode["false_finalization_count"], 0)
                self.assertEqual(episode["invalid_action_count"], 0)
                self.assertIs(episode["loop_detected"], False)
                self.assertEqual(
                    [
                        row["error_code"]
                        for row in policy_trace
                        if row["execution_status"] == "failure"
                    ],
                    [],
                )
                if episode["terminal_outcome"] == "operator_escalation":
                    self.assertEqual(
                        policy_trace[-1]["action"]["tool"], ASK_FOR_MORE_EVIDENCE
                    )
                    self.assertEqual(
                        policy_trace[-1]["action"]["arguments"]["request"],
                        "operator_escalation:recovery_options_exhausted",
                    )
                else:
                    self.assertIs(episode["final_physical_success"], True)
                    self.assertEqual(
                        policy_trace[-1]["action"]["tool"], FINALIZE_DIAGNOSIS
                    )


class DeterministicCaseLoaderTests(unittest.TestCase):
    def test_loader_uses_the_single_production_parser(self) -> None:
        parsed = {
            "baseMVA": 100.0,
            "bus": [[1.0]],
            "gen": [[1.0]],
            "branch": [[1.0, 2.0]],
        }
        with mock.patch(
            "mcp_server.matpower_server._load_python_case", return_value=parsed
        ) as loader:
            actual = factories.deterministic_case_loader(
                {"case_path": "case14"}
            )
        loader.assert_called_once_with(
            str((factories._REPO_ROOT / "mcp_server" / "case14.m").resolve())
        )
        self.assertEqual(actual, parsed)

    def test_loader_rejects_ambiguous_or_incomplete_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            factories.deterministic_case_loader(
                {"case_path": "case14.m", "fallback": "case30.m"}
            )
        with self.assertRaisesRegex(TypeError, "non-empty path"):
            factories.deterministic_case_loader("")
        with (
            mock.patch(
                "mcp_server.matpower_server._load_python_case",
                return_value={"bus": [], "gen": [], "branch": []},
            ),
            self.assertRaisesRegex(ValueError, "baseMVA"),
        ):
            factories.deterministic_case_loader("case14")


def _write_adapter_tree(root: Path) -> str:
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": factories.BASE_MODEL_ID,
                "peft_type": "LORA",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    return factories.checkpoint_tree_sha256(root)


class CheckpointTreeIdentityTests(unittest.TestCase):
    def test_digest_is_order_independent_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.bin").write_bytes(b"z")
            nested = root / "nested"
            nested.mkdir()
            (nested / "a.json").write_text("{}", encoding="utf-8")
            first = factories.checkpoint_tree_sha256(root)
            second = factories.checkpoint_tree_sha256(root)
            self.assertEqual(first, second)
            self.assertRegex(first, r"^[0-9a-f]{64}$")
            (nested / "a.json").write_text('{"changed":true}', encoding="utf-8")
            self.assertNotEqual(first, factories.checkpoint_tree_sha256(root))

    def test_digest_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "weights.bin"
            target.write_bytes(b"weights")
            link = root / "alias.bin"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("filesystem does not permit symlinks")
            with self.assertRaisesRegex(ValueError, "symlink"):
                factories.checkpoint_tree_sha256(root)

    def test_digest_rejects_symlinked_parent_and_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            real_parent = parent / "real"
            checkpoint = real_parent / "checkpoint"
            checkpoint.mkdir(parents=True)
            weights = checkpoint / "weights.bin"
            weights.write_bytes(b"weights")
            linked_parent = parent / "linked"
            try:
                linked_parent.symlink_to(real_parent, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("filesystem does not permit symlinks")
            with self.assertRaisesRegex(ValueError, "path contains a symlink"):
                factories.checkpoint_tree_sha256(linked_parent / "checkpoint")

            linked_parent.unlink()
            duplicate = checkpoint / "duplicate.bin"
            try:
                duplicate.hardlink_to(weights)
            except (OSError, NotImplementedError):
                self.skipTest("filesystem does not permit hardlinks")
            with self.assertRaisesRegex(ValueError, "multiply linked"):
                factories.checkpoint_tree_sha256(checkpoint)

    def test_adapter_validation_requires_exact_tree_and_pinned_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = _write_adapter_tree(root)
            self.assertEqual(factories._validate_adapter_tree(str(root), digest), root)
            with self.assertRaisesRegex(GateError, "digest mismatch"):
                factories._validate_adapter_tree(str(root), "0" * 64)

            config = json.loads((root / "adapter_config.json").read_text())
            config["base_model_name_or_path"] = "other/model"
            (root / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
            changed_digest = factories.checkpoint_tree_sha256(root)
            with self.assertRaisesRegex(GateError, "base_model_name_or_path"):
                factories._validate_adapter_tree(str(root), changed_digest)

    def test_release_checkpoint_inspection_validates_adapter_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = _write_adapter_tree(root)
            inspection = factories.inspect_release_checkpoint(root)
            self.assertEqual(inspection["path"], str(root))
            self.assertEqual(inspection["tree_sha256"], digest)
            self.assertEqual(inspection["file_count"], 2)
            self.assertGreater(inspection["total_bytes"], 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trainer_state.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(GateError, "adapter_config"):
                factories.inspect_release_checkpoint(root)


class ModelLoadingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        factories._MODEL_BUNDLES.clear()

    def tearDown(self) -> None:
        factories._MODEL_BUNDLES.clear()

    def test_snapshot_tree_rejects_corrupt_weight_and_tokenizer_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            payloads = {
                "model.safetensors": b"model-weight-bytes",
                "tokenizer.json": b'{"tokenizer": true}',
            }
            for name, payload in payloads.items():
                (snapshot / name).write_bytes(payload)
            tokenizer = payloads["tokenizer.json"]
            tokenizer_git_hash = hashlib.sha1(
                f"blob {len(tokenizer)}\0".encode("ascii") + tokenizer
            ).hexdigest()
            manifest = {
                "model.safetensors": (
                    len(payloads["model.safetensors"]),
                    "sha256",
                    hashlib.sha256(payloads["model.safetensors"]).hexdigest(),
                ),
                "tokenizer.json": (
                    len(tokenizer),
                    "git_blob_sha1",
                    tokenizer_git_hash,
                ),
            }
            factories._verify_snapshot_tree(snapshot, manifest)

            for name in payloads:
                with self.subTest(name=name):
                    original = payloads[name]
                    (snapshot / name).write_bytes(b"x" * len(original))
                    with self.assertRaisesRegex(GateError, "digest mismatch"):
                        factories._verify_snapshot_tree(snapshot, manifest)
                    (snapshot / name).write_bytes(original)

    def test_base_loader_uses_only_image_text_model_and_exact_snapshot(self) -> None:
        # Import before patching sys.modules so mock.patch.dict restores the
        # already-complete torch module graph rather than leaving only its
        # lazily imported submodules behind.
        import torch  # noqa: F401

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory).resolve()

            class Processor:
                name_or_path = str(snapshot)

                def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> str:
                    return "prompt"

                def decode(self, *_args: Any, **_kwargs: Any) -> str:
                    return ""

            class Config:
                model_type = "gemma4"
                _name_or_path = str(snapshot)

            class Model:
                config = Config()

                def eval(self) -> "Model":
                    return self

            class AutoProcessor:
                calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

                @classmethod
                def from_pretrained(cls, *args: Any, **kwargs: Any) -> Processor:
                    cls.calls.append((args, kwargs))
                    return Processor()

            class AutoModelForImageTextToText:
                calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

                @classmethod
                def from_pretrained(cls, *args: Any, **kwargs: Any) -> Model:
                    cls.calls.append((args, kwargs))
                    return Model()

            class BitsAndBytesConfig:
                def __init__(self, **kwargs: Any) -> None:
                    self.kwargs = kwargs

            transformer_module = types.SimpleNamespace(
                AutoModelForImageTextToText=AutoModelForImageTextToText,
                AutoProcessor=AutoProcessor,
                BitsAndBytesConfig=BitsAndBytesConfig,
            )
            post_load_attestation = mock.Mock()
            with (
                mock.patch.object(factories, "_resolve_base_snapshot", return_value=snapshot),
                mock.patch.object(
                    factories,
                    "_verify_snapshot_tree",
                    post_load_attestation,
                ),
                mock.patch.dict(sys.modules, {"transformers": transformer_module}),
            ):
                model, processor = factories._load_base_components()

            self.assertIsInstance(model, Model)
            self.assertIsInstance(processor, Processor)
            self.assertEqual(len(AutoProcessor.calls), 1)
            self.assertEqual(len(AutoModelForImageTextToText.calls), 1)
            model_args, model_kwargs = AutoModelForImageTextToText.calls[0]
            self.assertEqual(model_args, (str(snapshot),))
            self.assertIs(model_kwargs["local_files_only"], True)
            self.assertIs(model_kwargs["trust_remote_code"], False)
            self.assertEqual(model_kwargs["device_map"], "auto")
            post_load_attestation.assert_called_once_with(
                snapshot,
                factories.BASE_SNAPSHOT_FILE_MANIFEST,
                factories.BASE_SNAPSHOT_OPTIONAL_FILE_MANIFEST,
            )

    def test_base_identity_is_exact_and_checkpoint_never_falls_back(self) -> None:
        base_model = types.SimpleNamespace(eval=lambda: None)
        processor = object()
        with mock.patch.object(
            factories, "_load_base_components", return_value=(base_model, processor)
        ) as loader:
            bundle = factories._load_model_bundle(
                factories.BASE_MODEL_ID, factories.BASE_MODEL_REVISION
            )
            self.assertIs(bundle.model, base_model)
            loader.assert_called_once_with()
        with self.assertRaisesRegex(GateError, "exactly"):
            factories._load_model_bundle(factories.BASE_MODEL_ID, "a" * 40)
        with self.assertRaisesRegex(GateError, "absolute path"):
            factories._load_model_bundle("relative/checkpoint", "a" * 64)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            digest = _write_adapter_tree(checkpoint)

            class PeftModel:
                @classmethod
                def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
                    raise RuntimeError("adapter load failed")

            with (
                mock.patch.object(
                    factories,
                    "_load_base_components",
                    return_value=(base_model, processor),
                ),
                mock.patch.dict(
                    sys.modules, {"peft": types.SimpleNamespace(PeftModel=PeftModel)}
                ),
                self.assertRaisesRegex(GateError, "raw base model was not used"),
            ):
                factories._load_model_bundle(str(checkpoint), digest)

    def test_valid_local_peft_is_loaded_and_attested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            digest = _write_adapter_tree(checkpoint)
            base_model = object()
            processor = object()
            calls: list[tuple[Any, str, dict[str, Any]]] = []

            class Loaded:
                peft_config = {"default": object()}

                def eval(self) -> "Loaded":
                    return self

            loaded = Loaded()

            class PeftModel:
                @classmethod
                def from_pretrained(
                    cls, model: Any, path: str, **kwargs: Any
                ) -> Loaded:
                    calls.append((model, path, kwargs))
                    private_weights = Path(path) / "adapter_model.safetensors"
                    self.assertEqual(private_weights.read_bytes(), b"adapter-weights")
                    # Mutation of the source during PEFT load cannot change the
                    # already verified private bytes supplied to the loader.
                    (checkpoint / "adapter_model.safetensors").write_bytes(b"mutated")
                    self.assertEqual(private_weights.read_bytes(), b"adapter-weights")
                    return loaded

            with (
                mock.patch.object(
                    factories,
                    "_load_base_components",
                    return_value=(base_model, processor),
                ),
                mock.patch.dict(
                    sys.modules, {"peft": types.SimpleNamespace(PeftModel=PeftModel)}
                ),
            ):
                bundle = factories._load_model_bundle(str(checkpoint), digest.upper())
            self.assertIs(bundle.model, loaded)
            self.assertEqual(bundle.model_revision, digest)
            self.assertEqual(calls[0][0], base_model)
            self.assertNotEqual(calls[0][1], str(checkpoint))
            self.assertEqual(calls[0][1], bundle.adapter_snapshot_path)
            self.assertIsNotNone(bundle.adapter_snapshot_owner)
            self.assertTrue(Path(bundle.adapter_snapshot_path).is_dir())
            self.assertIs(calls[0][2]["is_trainable"], False)
            self.assertIs(calls[0][2]["local_files_only"], True)

    def test_factory_caches_one_bundle_per_identity(self) -> None:
        bundle = factories._ModelBundle(
            model=object(),
            processor=object(),
            model_id=factories.BASE_MODEL_ID,
            model_revision=factories.BASE_MODEL_REVISION,
        )
        with mock.patch.object(
            factories, "_load_model_bundle", return_value=bundle
        ) as loader:
            first = factories.gemma_release_policy_factory(
                model_id=factories.BASE_MODEL_ID,
                model_revision=factories.BASE_MODEL_REVISION,
            )
            second = factories.gemma_release_policy_factory(
                model_id=factories.BASE_MODEL_ID,
                model_revision=factories.BASE_MODEL_REVISION,
            )
        loader.assert_called_once_with(
            factories.BASE_MODEL_ID, factories.BASE_MODEL_REVISION
        )
        self.assertEqual(first.release_policy_identity, second.release_policy_identity)


class GeneratedToolCallValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        tools = factories.unified_tool_schemas()
        self.schemas = {
            row["function"]["name"]: row["function"]["parameters"]
            for row in tools
        }

    def test_exactly_one_native_or_json_call_is_required(self) -> None:
        valid = '<|tool_call|>call:wls_from_path{"case_path":"active"}'
        self.assertEqual(
            factories._validated_generated_action(valid, self.schemas),
            {"tool": "wls_from_path", "arguments": {"case_path": "active"}},
        )
        with self.assertRaisesRegex(GateError, "exactly one tool call; found 2"):
            factories._validated_generated_action(valid + valid, self.schemas)

        json_call = json.dumps(
            {"name": "wls_from_path", "arguments": {"case_path": "active"}}
        )
        with self.assertRaisesRegex(GateError, "exactly one tool call; found 2"):
            factories._validated_generated_action(
                json_call + "\n" + json_call, self.schemas
            )

    def test_registry_name_required_type_and_extra_arguments_are_exact(self) -> None:
        with self.assertRaisesRegex(GateError, "not in the pinned registry"):
            factories._validated_generated_action(
                'call:not_a_tool{"case_path":"active"}', self.schemas
            )
        with self.assertRaisesRegex(GateError, "unsupported arguments"):
            factories._validated_generated_action(
                'call:wls_from_path{"case_path":"active","state_id":"active"}',
                self.schemas,
            )
        with self.assertRaisesRegex(GateError, "JSON type string"):
            factories._validated_generated_action(
                'call:wls_from_path{"case_path":7}', self.schemas
            )
        with self.assertRaisesRegex(GateError, "missing required arguments"):
            factories._validated_generated_action(
                "call:wls_from_path{}", self.schemas
            )

    def test_topology_release_parser_uses_only_one_based_line_index(self) -> None:
        valid = (
            "call:correct_topology_from_path"
            '{"case_path":"active","line_index1":4,"desired_status":false}'
        )
        self.assertEqual(
            factories._validated_generated_action(valid, self.schemas),
            {
                "tool": "correct_topology_from_path",
                "arguments": {
                    "case_path": "active",
                    "line_index1": 4,
                    "desired_status": False,
                },
            },
        )
        with self.assertRaisesRegex(GateError, "unsupported arguments.*line_index"):
            factories._validated_generated_action(
                "call:correct_topology_from_path"
                '{"case_path":"active","line_index":3,"desired_status":false}',
                self.schemas,
            )
        with self.assertRaisesRegex(GateError, "missing required arguments.*line_index1"):
            factories._validated_generated_action(
                "call:correct_topology_from_path"
                '{"case_path":"active","desired_status":false}',
                self.schemas,
            )
        with self.assertRaisesRegex(GateError, "unsupported arguments.*cb_name"):
            factories._validated_generated_action(
                "call:correct_topology_from_path"
                '{"case_path":"active","cb_name":"CB_4_5",'
                '"desired_status":false}',
                self.schemas,
            )

    def test_parameter_release_parser_requires_executable_numeric_target(
        self,
    ) -> None:
        valid = (
            "call:correct_parameters_from_path"
            '{"case_path":"active","line_index":4}'
        )
        self.assertEqual(
            factories._validated_generated_action(valid, self.schemas),
            {
                "tool": "correct_parameters_from_path",
                "arguments": {"case_path": "active", "line_index": 4},
            },
        )
        with self.assertRaisesRegex(GateError, "missing required arguments.*line_index"):
            factories._validated_generated_action(
                'call:correct_parameters_from_path{"case_path":"active"}',
                self.schemas,
            )
        with self.assertRaisesRegex(GateError, "unsupported arguments.*branch_id"):
            factories._validated_generated_action(
                "call:correct_parameters_from_path"
                '{"case_path":"active","branch_id":"L2"}',
                self.schemas,
            )

    def test_hif_search_dimensions_are_bounded_before_execution(self) -> None:
        single_prefix = "call:estimate_hif_location_magnitude_from_path"
        with self.assertRaisesRegex(GateError, "alpha_grid_size must be <= 31"):
            factories._validated_generated_action(
                single_prefix
                + '{"case_path":"active","candidate_branch_row0":2,'
                '"alpha_grid_size":32}',
                self.schemas,
            )
        with self.assertRaisesRegex(GateError, "r_grid_size must be >= 2"):
            factories._validated_generated_action(
                single_prefix
                + '{"case_path":"active","candidate_branch_row0":2,'
                '"r_grid_size":1}',
                self.schemas,
            )
        with self.assertRaisesRegex(GateError, "max_scans must be <= 10"):
            factories._validated_generated_action(
                'call:estimate_hif_location_magnitude_multiscan_from_path'
                '{"scan_window_path":"active","candidate_branch_row0":2,'
                '"max_scans":11}',
                self.schemas,
            )

    def test_release_registry_rejects_nonexecutable_canonical_options(self) -> None:
        calls = (
            (
                "correct_measurements_from_path",
                {
                    "case_path": "active",
                    "suspect_group": [7],
                    "enable_correction": True,
                },
            ),
            (
                "correct_measurements_from_path",
                {
                    "case_path": "active",
                    "suspect_group": [7],
                    "enable_correction": False,
                },
            ),
            (
                "correct_measurements_from_path",
                {
                    "case_path": "active",
                    "suspect_group": [7],
                    "max_correction_iterations": 4,
                },
            ),
            (
                "correct_measurements_from_path",
                {
                    "case_path": "active",
                    "suspect_group": [7],
                    "error_tolerance": 1e-4,
                },
            ),
            (
                "get_parameter_context",
                {"case_path": "active", "line_index": 7},
            ),
            (
                "get_verification_snapshot",
                {"stage": "post_measurement_correction"},
            ),
        )
        for tool, arguments in calls:
            rendered = f"call:{tool}{json.dumps(arguments, separators=(',', ':'))}"
            with self.subTest(tool=tool):
                with self.assertRaisesRegex(GateError, "unsupported arguments"):
                    factories._validated_generated_action(rendered, self.schemas)


class CanonicalGemmaInferenceTests(unittest.TestCase):
    def test_prompt_generation_parse_bridge_and_alias_binding_match_sft(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - repository test dependency
            self.skipTest(f"torch unavailable: {exc}")

        class Processor:
            model_input_names = ["input_ids", "attention_mask", "mm_token_type_ids"]

            def __init__(self) -> None:
                self.messages: list[dict[str, Any]] | None = None
                self.tools: list[dict[str, Any]] | None = None
                self.tokenize_kwargs: dict[str, Any] | None = None

            def apply_chat_template(
                self,
                messages: list[dict[str, Any]],
                *,
                tools: list[dict[str, Any]],
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> str:
                self.messages = copy.deepcopy(messages)
                self.tools = copy.deepcopy(tools)
                self.assertions = (tokenize, add_generation_prompt)
                return "rendered-canonical-prompt"

            def __call__(self, text: str | None = None, **kwargs: Any) -> dict[str, Any]:
                self.tokenize_kwargs = kwargs
                if text != "rendered-canonical-prompt":
                    raise AssertionError(text)
                return {
                    "input_ids": [10, 11, 12],
                    "attention_mask": [1, 1, 1],
                }

            def decode(self, _ids: Any, *, skip_special_tokens: bool) -> str:
                if skip_special_tokens:
                    raise AssertionError("release decode must retain tool delimiters")
                return '<|tool_call|>call:wls_from_path{"case_path":"active"}'

        class Model:
            def __init__(self) -> None:
                self.embedding = torch.nn.Embedding(32, 4)
                self.generated_kwargs: dict[str, Any] | None = None

            def get_input_embeddings(self) -> Any:
                return self.embedding

            def eval(self) -> "Model":
                return self

            def forward(self, input_ids: Any, mm_token_type_ids: Any) -> Any:
                del input_ids, mm_token_type_ids
                raise AssertionError("forward is not called directly during generation")

            def generate(self, **kwargs: Any) -> Any:
                self.generated_kwargs = kwargs
                prompt = kwargs["input_ids"]
                suffix = torch.tensor([[13]], dtype=torch.long, device=prompt.device)
                return torch.cat((prompt, suffix), dim=1)

        processor = Processor()
        model = Model()
        bundle = factories._ModelBundle(
            model=model,
            processor=processor,
            model_id=factories.BASE_MODEL_ID,
            model_revision=factories.BASE_MODEL_REVISION,
        )
        policy = factories.GemmaReleasePolicy(bundle)
        controller_id = "root-episode:s12345678"
        action = policy.act(
            {
                "active_state_id": controller_id,
                "remaining_budget": 24,
                "history_window": [],
            }
        )

        self.assertEqual(
            action,
            {"tool": RUN_WLS, "arguments": {"state_id": controller_id}},
        )
        self.assertEqual(
            processor.messages[0],
            {"role": "system", "content": factories.CANONICAL_DAGGER_SYSTEM_PROMPT},
        )
        user_payload = json.loads(processor.messages[1]["content"])
        self.assertEqual(user_payload["state"]["active_state_id"], "active")
        self.assertNotIn(controller_id, processor.messages[1]["content"])
        self.assertIn(
            "wls_from_path",
            {tool["function"]["name"] for tool in processor.tools},
        )
        self.assertEqual(processor.assertions, (False, True))
        self.assertIs(processor.tokenize_kwargs["return_mm_token_type_ids"], True)
        self.assertIs(model.generated_kwargs["do_sample"], False)
        self.assertEqual(
            model.generated_kwargs["max_new_tokens"],
            factories.MAX_NEW_TOKENS,
        )
        self.assertEqual(factories.MAX_NEW_TOKENS, 64)
        self.assertIs(model.generated_kwargs["use_cache"], True)
        self.assertIn("mm_token_type_ids", model.generated_kwargs)
        self.assertEqual(
            model.generated_kwargs["mm_token_type_ids"].tolist(), [[0, 0, 0]]
        )
        metrics = policy.last_action_metrics
        self.assertEqual(metrics["prompt_tokens"], 3)
        self.assertEqual(metrics["generated_tokens"], 1)
        self.assertFalse(metrics["hit_max_new_tokens"])
        self.assertEqual(metrics["last_token_id"], 13)
        self.assertGreaterEqual(metrics["generation_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
