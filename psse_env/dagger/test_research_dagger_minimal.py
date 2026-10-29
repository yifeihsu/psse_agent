from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from unittest.mock import patch

import scripts.run_dagger_research as research_module
from scripts.run_dagger_research import (
    DEFAULT_DEVELOPMENT_PLAN,
    DEFAULT_TRAIN_PLAN,
    PRIOR_ONPOLICY_CHAT_FILENAME,
    PRIOR_ONPOLICY_RAW_FILENAME,
    RESEARCH_HIF_SEARCH_BUDGET,
    RESEARCH_MIXTURE_CONTRACT,
    allocate_scenarios,
    audit_mixture_label_conflicts,
    build_research_mixture,
    load_prior_onpolicy_rounds,
    collect_resumable,
    evaluate_paired_adapters,
    export_research_rows,
    is_research_dagger_row,
    load_protected_suite_roots,
    mark_research_label_eligibility,
    parser,
    plan_preset,
    prepare_scenario_split,
    refresh_d0_training_view,
    resolve_hif_search_profile,
    resolve_scenario_sources,
)
from psse_env.providers.scenario_generator import (
    CURRENT_TELEMETRY_HIF_SAMPLE_PATHS,
    CURRENT_TELEMETRY_IMBALANCE_SAMPLE_PATH,
)
from psse_env.dagger.dataset_builder import examples_to_chat_sft
from psse_env.dagger.offline_teacher_target_audit import (
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
)


def _raw_row(root: str = "root_a", *, audit_passed: bool = True) -> dict:
    state_id = f"{root}:episode:s0"
    return {
        "example_id": f"example_{root}",
        "scenario_id": root,
        "root_scenario_id": root,
        "physical_root_fingerprint": root,
        "scenario_family": "measurement+parameter",
        "dataset_mode": "production",
        "dataset_source": "dagger_rollout",
        "state_origin": "learner_policy",
        "state_visited_by": "model",
        "collection_role": "training",
        "iteration": 1,
        "collection_beta": 0.25,
        "supervision_policy": "dagger1_observable_recovery_handoff_v2",
        "step": 1,
        "policy_observation": {
            "evidence_profile": "wls_gated_diagnostics",
            "active_state_id": state_id,
            "candidate_state_id": None,
            "candidate_parent_id": None,
            "episode_id": f"{root}:episode",
            "remaining_budget": 5,
            "history_window": [],
            "unresolved_signatures": [],
            "remaining_anomaly_score": None,
            "no_material_anomaly_remaining": False,
        },
        "history_window": [],
        "preferred_action": {
            "tool": "run_wls",
            "arguments": {"state_id": state_id},
        },
        "model_action": {
            "tool": "get_measurement_context",
            "arguments": {"state_id": state_id},
        },
        "executed_by": "model",
        "labels": {"training_decision_evidence_verified": True},
        "observable_rank_one_target_proof": {"passed": True},
        "offline_teacher_target_audit": (
            {
                "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
                "passed": True,
                "action_class": "read_only",
                "checks": {"observable_evidence_gate_passed": True},
                "reason_codes": [],
            }
            if audit_passed
            else {"passed": False}
        ),
        "production_label_eligible": False,
        "recovery_stratum": None,
        "private_truth_marker": "never-model-visible",
    }


class ResearchEligibilityTests(unittest.TestCase):
    def test_research_eligibility_does_not_require_recovery_stratum(self) -> None:
        row = _raw_row()
        self.assertIsNone(row["recovery_stratum"])
        self.assertTrue(is_research_dagger_row(row))
        stamped = mark_research_label_eligibility(row)
        self.assertTrue(stamped["research_label_eligible"])
        self.assertFalse(stamped["production_label_eligible"])

    def test_one_failed_offline_audit_quarantines_only_that_row(self) -> None:
        good = mark_research_label_eligibility(_raw_row("good"))
        bad = mark_research_label_eligibility(
            _raw_row("bad", audit_passed=False)
        )
        self.assertTrue(good["research_label_eligible"])
        self.assertFalse(bad["research_label_eligible"])
        self.assertIn(
            "offline_teacher_target_audit_failed",
            bad["research_label_ineligibility_reasons"],
        )

    def test_canonical_export_preserves_research_marker_outside_prompt(self) -> None:
        row = mark_research_label_eligibility(_raw_row())
        with tempfile.TemporaryDirectory() as directory:
            _, exported, failures = export_research_rows(
                [row], output_dir=Path(directory)
            )
        self.assertFalse(failures)
        self.assertEqual(len(exported), 1)
        self.assertTrue(exported[0]["research_label_eligible"])
        self.assertTrue(exported[0]["metadata"]["research_label_eligible"])
        self.assertNotIn(
            "never-model-visible",
            json.dumps(exported[0]["messages"], sort_keys=True),
        )


class ResearchSplitAndResumeTests(unittest.TestCase):
    def test_protected_suite_loader_unions_jsonl_and_nested_d1_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "protected.jsonl"
            jsonl_path.write_text(
                json.dumps({"metadata": {"physical_root_fingerprint": "protected_a"}})
                + "\n",
                encoding="utf-8",
            )
            nested_path = root / "d1_development_suite.json"
            nested_path.write_text(
                json.dumps(
                    {
                        "standard_success": [
                            {
                                "grouping": {
                                    "physical_root_fingerprint": "protected_b"
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = load_protected_suite_roots([jsonl_path, nested_path])

        self.assertEqual(report["physical_root_count"], 2)
        self.assertEqual(
            report["physical_roots"], ["protected_a", "protected_b"]
        )
        self.assertEqual(report["paths"], [str(jsonl_path), str(nested_path)])
        self.assertEqual(len(report["physical_roots_sha256"]), 64)
        self.assertEqual(len(report["artifacts"]), 2)

    def test_d0_is_rerendered_under_the_current_canonical_view(self) -> None:
        raw = _raw_row("d0_root")
        prior = examples_to_chat_sft(
            [raw], protocol="canonical", allow_ineligible_auxiliary=True
        )[0]
        prior["messages"][1]["content"] = "stale-observation-view"
        refreshed, report = refresh_d0_training_view([raw], [prior])
        self.assertEqual(report["rows_changed_from_input_view"], 1)
        self.assertNotEqual(refreshed[0]["messages"], prior["messages"])
        self.assertEqual(
            refreshed[0]["physical_root_fingerprint"], "d0_root"
        )

    def test_allocator_excludes_d0_and_keeps_train_dev_roots_disjoint(self) -> None:
        candidates = []
        for index in range(8):
            candidates.append(
                {
                    "grouping": {
                        "physical_root_fingerprint": f"root_{index}",
                        "scenario_family": "parameter",
                        "split": "dagger_train",
                    }
                }
            )
        training, development = allocate_scenarios(
            candidates,
            d0_roots={"root_0"},
            train_plan={"parameter": 3},
            development_plan={"parameter": 2},
            seed=7,
        )
        train_roots = {row["grouping"]["physical_root_fingerprint"] for row in training}
        dev_roots = {row["grouping"]["physical_root_fingerprint"] for row in development}
        self.assertFalse(train_roots & dev_roots)
        self.assertNotIn("root_0", train_roots | dev_roots)

    def test_allocator_excludes_protected_roots_from_both_splits(self) -> None:
        candidates = [
            {
                "grouping": {
                    "physical_root_fingerprint": f"root_{index}",
                    "scenario_family": "parameter",
                    "split": "dagger_train",
                }
            }
            for index in range(8)
        ]
        training, development = allocate_scenarios(
            candidates,
            d0_roots={"root_0"},
            protected_roots={"root_1", "root_2"},
            train_plan={"parameter": 2},
            development_plan={"parameter": 2},
            seed=7,
        )
        selected = {
            row["grouping"]["physical_root_fingerprint"]
            for row in [*training, *development]
        }
        self.assertFalse(selected & {"root_0", "root_1", "root_2"})

    def test_completed_episode_files_skip_policy_and_environment_on_resume(self) -> None:
        scenarios = [
            {
                "grouping": {
                    "physical_root_fingerprint": root,
                    "scenario_family": "measurement+parameter",
                    "split": "dagger_train",
                }
            }
            for root in ("root_a", "root_b")
        ]
        calls = {"policy": 0, "environment": 0}

        def policy_factory():
            calls["policy"] += 1
            return object()

        def environment_factory(**_kwargs):
            calls["environment"] += 1
            return SimpleNamespace(
                process_oracle=None, candidate_quality_oracle=None
            )

        class FakeCollector:
            def __init__(self, **_kwargs):
                pass

            def collect_iteration(self, *, scenarios, **_kwargs):
                return [_raw_row(scenarios[0]["grouping"]["physical_root_fingerprint"])]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows, _ = collect_resumable(
                training_scenarios=scenarios,
                development_roots={"dev_root"},
                d0_roots={"d0_root"},
                output_dir=output,
                seed=9,
                beta=0.25,
                max_steps=8,
                policy_factory=policy_factory,
                environment_factory=environment_factory,
                learner_adapter_path="adapter",
                collector_class=FakeCollector,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(calls, {"policy": 1, "environment": 2})
            collect_resumable(
                training_scenarios=scenarios,
                development_roots={"dev_root"},
                d0_roots={"d0_root"},
                output_dir=output,
                seed=9,
                beta=0.25,
                max_steps=8,
                policy_factory=policy_factory,
                environment_factory=environment_factory,
                learner_adapter_path="adapter",
                collector_class=FakeCollector,
            )
            self.assertEqual(calls, {"policy": 1, "environment": 2})

    def test_simple_mixture_hits_quarter_d1_share(self) -> None:
        d0 = [
            {
                "example_id": f"d0_{index}",
                "physical_root_fingerprint": f"d0_root_{index}",
                "metadata": {"protocol": "canonical", "evidence_profile": "wls_gated_diagnostics"},
            }
            for index in range(20)
        ]
        d1 = [
            {
                "example_id": f"d1_{index}",
                "physical_root_fingerprint": f"d1_root_{index}",
                "metadata": {"protocol": "canonical", "evidence_profile": "wls_gated_diagnostics"},
            }
            for index in range(4)
        ]
        mixture, report = build_research_mixture(
            d0, d1, d1_share=0.25, d1_cap=None, seed=3
        )
        self.assertEqual(len(mixture), 16)
        self.assertEqual(report["d0_selected"], 12)
        self.assertEqual(report["actual_d1_share"], 0.25)

    def test_paired_evaluation_reuses_the_exact_development_roots(self) -> None:
        scenarios = [
            {
                "grouping": {
                    "physical_root_fingerprint": root,
                    "scenario_family": "parameter",
                }
            }
            for root in ("dev_a", "dev_b")
        ]
        observed = []

        def policy_loader(path, **_kwargs):
            return str(path)

        class Result:
            def __init__(self, resolved):
                self.resolved = resolved

            def as_dict(self):
                return {
                    "score": float(self.resolved),
                    "metrics": {},
                    "suite_metrics": {
                        "overall": {
                            "episodes": 2,
                            "resolved_episodes": self.resolved,
                        }
                    },
                }

        loaders = []

        def evaluator(suites, *, policy_factory, **_kwargs):
            roots = [
                row["grouping"]["physical_root_fingerprint"]
                for row in suites["standard_success"]
            ]
            policy = policy_factory()
            observed.append((policy, roots))
            loaders.append(_kwargs.get("case_loader"))
            return Result(0 if policy.endswith("bc0") else 1)

        with tempfile.TemporaryDirectory() as directory:
            comparison = evaluate_paired_adapters(
                development_scenarios=scenarios,
                bc0_adapter=Path("bc0"),
                r1_adapter=Path("r1"),
                base_model="gemma",
                base_revision="f" * 40,
                output_dir=Path(directory),
                seed=4,
                max_steps=8,
                policy_loader=policy_loader,
                environment_factory=lambda **_kwargs: object(),
                evaluator=evaluator,
                expert_policy_factory=lambda: "expert",
            )
            written = sorted(path.name for path in (Path(directory) / "evaluation").iterdir())
        self.assertEqual(observed[0][1], observed[1][1])
        self.assertEqual(observed[0][1], ["dev_a", "dev_b"])
        self.assertEqual(comparison["r1_minus_bc0"]["resolved_episodes"], 1.0)
        # The teacher itself is rolled out on the same roots as the ceiling.
        self.assertEqual([policy for policy, _ in observed], ["bc0", "r1", "expert"])
        self.assertEqual(observed[2][1], ["dev_a", "dev_b"])
        self.assertEqual(comparison["expert_overall"]["resolved_episodes"], 1)
        self.assertIn("expert_eval.json", written)
        # The strict audit needs a case loader to compare parameter and
        # topology corrections against the clean case; the production parser
        # is the default so those families are never scored evidence-missing.
        from psse_env.dagger.release_factories import deterministic_case_loader

        self.assertEqual(loaders, [deterministic_case_loader] * 3)


class ResearchEpisodeBudgetTests(unittest.TestCase):
    """The research environment keeps the production factory's episode horizon."""

    def test_research_budget_matches_the_production_factory(self) -> None:
        import inspect
        import re

        from psse_env.dagger import release_factories
        from scripts.run_dagger_research import RESEARCH_EPISODE_BUDGET, parser

        source = inspect.getsource(release_factories.production_environment_factory)
        self.assertIn("max_steps=DEFAULT_EPISODE_ACTION_LIMIT", source)
        production = release_factories.DEFAULT_EPISODE_ACTION_LIMIT
        self.assertEqual(RESEARCH_EPISODE_BUDGET, production)
        self.assertEqual(RESEARCH_EPISODE_BUDGET, 40)
        self.assertEqual(parser().get_default("eval_max_steps"), production)

    def test_branch_first_partial_option_reaches_the_candidate_oracle(self) -> None:
        from scripts import run_dagger_research as research

        self.assertFalse(parser().get_default("branch_first_partial"))
        self.assertEqual(parser().get_default("eval_output_name"), "evaluation")
        original = dict(research.RESEARCH_ENVIRONMENT_OPTIONS)
        try:
            research.RESEARCH_ENVIRONMENT_OPTIONS["branch_first_partial"] = True
            env = research.research_diagnostic_environment_factory()
            self.assertTrue(env.candidate_quality_oracle.branch_first_partial)
            research.RESEARCH_ENVIRONMENT_OPTIONS["branch_first_partial"] = False
            env = research.research_diagnostic_environment_factory()
            self.assertFalse(env.candidate_quality_oracle.branch_first_partial)
        finally:
            research.RESEARCH_ENVIRONMENT_OPTIONS.clear()
            research.RESEARCH_ENVIRONMENT_OPTIONS.update(original)


class FixedScenarioSuiteTests(unittest.TestCase):
    """A saved suite is adopted as-is and pinned in the run configuration."""

    @staticmethod
    def _scenario(family: str, root: str) -> dict:
        return {
            "scenario_id": f"{family}-{root}",
            "grouping": {"scenario_family": family, "physical_root_fingerprint": root},
        }

    def test_fixed_suites_are_written_and_pinned_without_generation(self) -> None:
        from scripts.run_dagger_research import prepare_scenario_split

        training = [self._scenario("hif", "t1"), self._scenario("harmonic", "t2")]
        development = [self._scenario("hif", "d1")]
        suite = {"training": {"sha256": "a" * 64}, "development": {"sha256": "b" * 64}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            common = dict(
                output_dir=output,
                d0_raw_path=output / "d0.jsonl",
                d0_roots={"d0"},
                train_plan={"hif": 1, "harmonic": 1},
                development_plan={"hif": 1},
                candidate_multiplier=3,
                seed=7,
                run_descriptor={"adapter_path": "x"},
                protected_roots={"p"},
            )
            got_training, got_development = prepare_scenario_split(
                **common,
                fixed_training=training,
                fixed_development=development,
                scenario_suite=suite,
            )
            self.assertEqual([r["scenario_id"] for r in got_training], ["hif-t1", "harmonic-t2"])
            self.assertEqual([r["scenario_id"] for r in got_development], ["hif-d1"])
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["scenario_suite"], suite)
            self.assertEqual(config["training_roots"], ["t1", "t2"])
            # Resume with the same suite reads the stored files back.
            again, _ = prepare_scenario_split(
                **common,
                fixed_training=training,
                fixed_development=development,
                scenario_suite=suite,
            )
            self.assertEqual(again, got_training)
            with self.assertRaisesRegex(RuntimeError, "differs on: scenario_suite"):
                prepare_scenario_split(
                    **common,
                    fixed_training=training,
                    fixed_development=development,
                    scenario_suite={**suite, "training": {"sha256": "c" * 64}},
                )

    def test_fixed_suites_reject_d0_protected_and_shared_roots(self) -> None:
        from scripts.run_dagger_research import prepare_scenario_split

        base = dict(
            d0_roots={"d0"},
            train_plan={"hif": 1},
            development_plan={"hif": 1},
            candidate_multiplier=3,
            seed=7,
            run_descriptor={},
            protected_roots={"p"},
        )
        cases = {
            "D0 or protected": ([self._scenario("hif", "d0")], [self._scenario("hif", "x")]),
            "D0 or protected roots": ([self._scenario("hif", "t")], [self._scenario("hif", "p")]),
            "share roots": ([self._scenario("hif", "s")], [self._scenario("hif", "s")]),
        }
        for message, (training, development) in cases.items():
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, message):
                    prepare_scenario_split(
                        output_dir=Path(temp),
                        d0_raw_path=Path(temp) / "d0.jsonl",
                        fixed_training=training,
                        fixed_development=development,
                        scenario_suite={},
                        **base,
                    )
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "come together"):
                prepare_scenario_split(
                    output_dir=Path(temp),
                    d0_raw_path=Path(temp) / "d0.jsonl",
                    fixed_training=[self._scenario("hif", "t")],
                    **base,
                )


class DiagnosticFamilyPresetTests(unittest.TestCase):
    def test_core_preset_is_the_legacy_default_plan(self) -> None:
        train, development = plan_preset("core")
        self.assertEqual(train, DEFAULT_TRAIN_PLAN)
        self.assertEqual(development, DEFAULT_DEVELOPMENT_PLAN)
        with self.assertRaises(ValueError):
            plan_preset("unknown")

    def test_diagnostic_preset_covers_explanation_only_families_and_control(self) -> None:
        train, development = plan_preset("diagnostic")
        self.assertEqual(
            set(train),
            {
                "hif",
                "measurement+hif",
                "three_phase_unbalance",
                "harmonic",
                "telemetry_no_disturbance",
            },
        )
        self.assertEqual(set(train), set(development))
        for family, count in train.items():
            self.assertGreater(count, development[family])
        combined_train, combined_dev = plan_preset("combined")
        self.assertEqual(set(combined_train), set(DEFAULT_TRAIN_PLAN) | set(train))
        self.assertEqual(set(combined_dev), set(DEFAULT_DEVELOPMENT_PLAN) | set(development))

    def test_core_plan_keeps_generator_default_corpora(self) -> None:
        sources = resolve_scenario_sources(plan_families=set(DEFAULT_TRAIN_PLAN))
        self.assertEqual(sources["evidence_profile"], "wls_gated_diagnostics")
        self.assertEqual(
            resolve_scenario_sources(plan_families=set(DEFAULT_TRAIN_PLAN), evidence_profile="scada_only")["evidence_profile"],
            "scada_only",
        )
        self.assertIsNone(sources["hif_sample_paths"])

    def test_diagnostic_plan_defaults_to_branch_current_corpora(self) -> None:
        sources = resolve_scenario_sources(plan_families={"three_phase_unbalance"})
        self.assertIsNotNone(sources)
        assert sources is not None
        self.assertEqual(
            sources["hif_sample_paths"],
            [str(Path(path).resolve()) for path in CURRENT_TELEMETRY_HIF_SAMPLE_PATHS],
        )
        self.assertEqual(
            sources["imbalance_sample_path"],
            str(Path(CURRENT_TELEMETRY_IMBALANCE_SAMPLE_PATH).resolve()),
        )
        for path in sources["hif_sample_paths"]:
            self.assertIn("currents", path)

    def test_signature_modes_are_recorded_with_the_corpora(self) -> None:
        sources = resolve_scenario_sources(
            plan_families={"three_phase_unbalance"},
            evidence_profile="auxiliary_diagnostics",
            signature_modes={"three_phase_unbalance": "discovered", "hif": "flagged"},
        )
        assert sources is not None
        self.assertEqual(
            sources["signature_modes"],
            {"hif": "flagged", "three_phase_unbalance": "discovered"},
        )
        with self.assertRaises(ValueError):
            resolve_scenario_sources(
                plan_families={"hif"}, signature_modes={"hif": "guessed"}
            )
        with self.assertRaises(ValueError):
            resolve_scenario_sources(
                plan_families={"hif"}, signature_modes={"unknown_family": "flagged"}
            )
        self.assertEqual(
            resolve_scenario_sources(
                plan_families=set(DEFAULT_TRAIN_PLAN),
                signature_modes={"three_phase_unbalance": "discovered"},
            )["evidence_profile"], "wls_gated_diagnostics"
        )

    def test_harmonic_only_plan_records_modes_without_telemetry_corpora(self) -> None:
        sources = resolve_scenario_sources(plan_families={"harmonic"})
        self.assertEqual(
            sources,
            {
                "hif_sample_paths": None,
                "imbalance_sample_path": None,
                "signature_modes": {"harmonic": "discovered"},
                "evidence_profile": "wls_gated_diagnostics",
                "system": "case14",
            },
        )
        flagged = resolve_scenario_sources(
            plan_families={"harmonic"}, signature_modes={"harmonic": "flagged"}, evidence_profile="auxiliary_diagnostics"
        )
        self.assertEqual(flagged["signature_modes"], {"harmonic": "flagged"})
        self.assertNotEqual(flagged, sources)
        with self.assertRaises(ValueError):
            resolve_scenario_sources(
                plan_families={"harmonic"}, signature_modes={"harmonic": "guessed"}
            )

    def test_harmonic_mode_cli_defaults_to_discovered_and_allows_legacy_flag(self) -> None:
        required = [
            "--d0-raw", "raw.jsonl", "--d0-train", "train.jsonl",
            "--adapter-path", "adapter", "--output-dir", "output",
        ]
        self.assertEqual(parser().parse_args(required).harmonic_signature_mode, "discovered")
        self.assertEqual(
            parser().parse_args(required + ["--harmonic-signature-mode", "flagged"])
            .harmonic_signature_mode,
            "flagged",
        )

    def test_explicit_corpus_paths_win_and_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "samples.jsonl"
            corpus.write_text("", encoding="utf-8")
            sources = resolve_scenario_sources(
                plan_families={"parameter"},
                hif_sample_paths=[corpus],
                imbalance_sample_path=None,
            )
            assert sources is not None
            self.assertEqual(sources["hif_sample_paths"], [str(corpus.resolve())])
            self.assertIsNone(sources["imbalance_sample_path"])
            with self.assertRaises(FileNotFoundError):
                resolve_scenario_sources(
                    plan_families={"hif"},
                    imbalance_sample_path=Path(tmp) / "missing.jsonl",
                )

    def test_hif_search_profile_auto_follows_the_plan(self) -> None:
        self.assertEqual(resolve_hif_search_profile("auto", {"hif"}), "research")
        self.assertEqual(resolve_hif_search_profile("auto", {"measurement+hif"}), "research")
        self.assertEqual(resolve_hif_search_profile("auto", {"three_phase_unbalance"}), "release")
        self.assertEqual(resolve_hif_search_profile("release", {"hif"}), "release")
        self.assertEqual(resolve_hif_search_profile("research", {"parameter"}), "research")
        with self.assertRaises(ValueError):
            resolve_hif_search_profile("fast", {"hif"})

    def test_research_budget_is_within_the_hard_search_limits(self) -> None:
        from hif_search_limits import validate_hif_search_limits

        alpha, radius, scans = validate_hif_search_limits(
            alpha_grid_size=RESEARCH_HIF_SEARCH_BUDGET["hif_alpha_grid_size"],
            r_grid_size=RESEARCH_HIF_SEARCH_BUDGET["hif_r_grid_size"],
            max_scans=RESEARCH_HIF_SEARCH_BUDGET["hif_max_scans"],
        )
        self.assertEqual((alpha, radius, scans), (7, 9, 10))

    def test_legacy_config_resumes_without_a_recorded_profile(self) -> None:
        # A run recorded before the research profile existed must still
        # resume under the core preset, and a diagnostic request against it
        # must be refused instead of silently reusing its scenarios.
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            d0_raw = output_dir / "d0.raw.jsonl"
            d0_raw.write_text("", encoding="utf-8")
            descriptor = {"adapter_path": "adapter"}
            base_config = {
                "contract": "research_dagger_minimal_v1",
                "seed": 7,
                "train_plan": {"parameter": 1},
                "development_plan": {"parameter": 1},
                "d0_raw_path": str(d0_raw.resolve()),
                "run_descriptor": descriptor,
            }
            rows = [
                {"grouping": {"physical_root_fingerprint": f"root_{index}", "scenario_family": "parameter", "split": split}}
                for index, split in enumerate(("dagger_train", "development"))
            ]
            (output_dir / "training_scenarios.json").write_text(json.dumps(rows[:1]), encoding="utf-8")
            (output_dir / "development_scenarios.json").write_text(json.dumps(rows[1:]), encoding="utf-8")
            (output_dir / "config.json").write_text(json.dumps(base_config), encoding="utf-8")
            training, development = prepare_scenario_split(
                output_dir=output_dir,
                d0_raw_path=d0_raw,
                d0_roots=set(),
                train_plan={"parameter": 1},
                development_plan={"parameter": 1},
                candidate_multiplier=1,
                seed=7,
                run_descriptor=descriptor,
            )
            self.assertEqual(len(training), 1)
            self.assertEqual(len(development), 1)
            with self.assertRaises(RuntimeError) as caught:
                prepare_scenario_split(
                    output_dir=output_dir,
                    d0_raw_path=d0_raw,
                    d0_roots=set(),
                    train_plan={"parameter": 1},
                    development_plan={"parameter": 1},
                    candidate_multiplier=1,
                    seed=7,
                    run_descriptor=descriptor,
                    research_profile={
                        "plan_preset": "diagnostic",
                        "hif_search_profile": "research",
                        "scenario_sources": None,
                    },
                )
            self.assertIn("research_profile", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class SystemSwitchSourceTests(unittest.TestCase):
    """--system threads a fresh balanced corpus through the research generator."""

    @staticmethod
    def _fresh(tmp: str) -> tuple[Path, Path]:
        corpus = Path(tmp) / "corpus.jsonl"
        corpus.write_text("", encoding="utf-8")
        artifacts = Path(tmp) / "artifacts"
        artifacts.mkdir()
        return corpus, artifacts

    def test_default_system_leaves_legacy_sources_untouched(self) -> None:
        self.assertEqual(resolve_scenario_sources(plan_families={"measurement"}, system="case14")["evidence_profile"], "wls_gated_diagnostics")
        sources = resolve_scenario_sources(plan_families={"three_phase_unbalance"}, system="ieee14")
        assert sources is not None
        self.assertEqual(sources["system"], "case14")

    def test_case57_sources_need_a_fresh_corpus_and_balanced_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, artifacts = self._fresh(tmp)
            sources = resolve_scenario_sources(
                plan_families={"measurement", "parameter"},
                system="case57",
                measurement_corpus=corpus,
                balanced_artifact_dir=artifacts,
                admission_mode="physical",
            )
            assert sources is not None
            self.assertEqual(sources["system"], "case57")
            self.assertEqual(sources["measurement_corpus"], str(corpus.resolve()))
            self.assertEqual(sources["balanced_artifact_dir"], str(artifacts.resolve()))
            self.assertEqual(sources["admission_mode"], "physical")
            self.assertIsNone(sources["hif_sample_paths"])
            self.assertIsNone(sources["imbalance_sample_path"])
            self.assertIsNone(sources["signature_modes"])
            with self.assertRaisesRegex(ValueError, "does not support families"):
                resolve_scenario_sources(
                    plan_families={"hif"}, system="case57",
                    measurement_corpus=corpus, balanced_artifact_dir=artifacts,
                )
            with self.assertRaisesRegex(ValueError, "IEEE 14 sources"):
                resolve_scenario_sources(
                    plan_families={"measurement"}, system="case57",
                    measurement_corpus=corpus, balanced_artifact_dir=artifacts,
                    hif_sample_paths=[corpus],
                )
            with self.assertRaisesRegex(ValueError, "fresh balanced corpus"):
                resolve_scenario_sources(plan_families={"measurement"}, system="case57")
            with self.assertRaises(FileNotFoundError):
                resolve_scenario_sources(
                    plan_families={"measurement"}, system="case57",
                    measurement_corpus=Path(tmp) / "missing.jsonl",
                    balanced_artifact_dir=artifacts,
                )
            with self.assertRaisesRegex(ValueError, "admission mode"):
                resolve_scenario_sources(
                    plan_families={"measurement"}, system="case57",
                    measurement_corpus=corpus, balanced_artifact_dir=artifacts,
                    admission_mode="teacher",
                )

    def test_fresh_sources_reach_the_generator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, artifacts = self._fresh(tmp)
            sources = resolve_scenario_sources(
                plan_families={"measurement"}, system="case57",
                measurement_corpus=corpus, balanced_artifact_dir=artifacts,
                admission_mode="physical",
            )
            captured: dict = {}
            with patch.object(
                research_module, "Round0ScenarioGenerator",
                side_effect=lambda **kwargs: captured.update(kwargs),
            ):
                research_module.research_scenario_generator(
                    seed=3, research_profile={"scenario_sources": sources}
                )
        self.assertEqual(captured["system"], "case57")
        self.assertEqual(captured["corpus_path"], corpus.resolve())
        self.assertEqual(captured["balanced_artifact_dir"], artifacts.resolve())
        self.assertEqual(captured["admission_mode"], "physical")
        self.assertEqual(captured["source_partition"], "train")
        self.assertEqual(captured["normalized_residual_threshold"], 4.0)
        self.assertNotIn("hif_sample_paths", captured)
        self.assertNotIn("imbalance_sample_path", captured)

    def test_system_cli_defaults_to_ieee14(self) -> None:
        required = [
            "--d0-raw", "raw.jsonl", "--d0-train", "train.jsonl",
            "--adapter-path", "adapter", "--output-dir", "output",
        ]
        args = parser().parse_args(required)
        self.assertEqual(args.system, "case14")
        self.assertIsNone(args.measurement_corpus)
        self.assertIsNone(args.balanced_artifact_dir)
        self.assertIsNone(args.admission_mode)
        args = parser().parse_args(required + ["--system", "case57", "--admission-mode", "physical"])
        self.assertEqual((args.system, args.admission_mode), ("case57", "physical"))


class DetectorRuleTests(unittest.TestCase):
    """The combined chi-square/normalized-residual rule reaches admission and the environment."""

    def test_cli_defaults_to_the_four_sigma_residual_test(self) -> None:
        self.assertEqual(parser().get_default("normalized_residual_threshold"), 4.0)
        self.assertFalse(parser().get_default("chi_square_only"))
        self.assertEqual(research_module.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"], 4.0)

    def test_option_reaches_the_environment_and_the_generator(self) -> None:
        original = dict(research_module.RESEARCH_ENVIRONMENT_OPTIONS)
        try:
            research_module.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
            env = research_module.research_diagnostic_environment_factory()
            self.assertEqual(env.wls_runner.__self__.normalized_residual_threshold, 4.0)
            captured: dict = {}
            with patch.object(
                research_module, "Round0ScenarioGenerator",
                side_effect=lambda **kwargs: captured.update(kwargs),
            ):
                research_module.research_scenario_generator(seed=1)
                self.assertEqual(captured["normalized_residual_threshold"], 4.0)
                research_module.research_scenario_generator(seed=1, normalized_residual_threshold=None)
                self.assertIsNone(captured["normalized_residual_threshold"])
            research_module.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = None
            env = research_module.research_diagnostic_environment_factory()
            self.assertIsNone(env.wls_runner.__self__.normalized_residual_threshold)
            with patch.object(
                research_module, "Round0ScenarioGenerator",
                side_effect=lambda **kwargs: captured.update(kwargs),
            ):
                research_module.research_scenario_generator(seed=1)
            self.assertIsNone(captured["normalized_residual_threshold"])
        finally:
            research_module.RESEARCH_ENVIRONMENT_OPTIONS.clear()
            research_module.RESEARCH_ENVIRONMENT_OPTIONS.update(original)


class StudentOnlyEvaluationTests(unittest.TestCase):
    """A zero-shot reading rolls out the student and the expert, with no candidate."""

    def test_student_only_evaluation_skips_the_candidate(self) -> None:
        scenarios = [
            {"grouping": {"physical_root_fingerprint": root, "scenario_family": "measurement"}}
            for root in ("dev_a", "dev_b")
        ]
        observed = []

        class Result:
            def __init__(self, resolved):
                self.resolved = resolved

            def as_dict(self):
                return {"suite_metrics": {"overall": {"episodes": 2, "resolved_episodes": self.resolved}}}

        def evaluator(suites, *, policy_factory, **_kwargs):
            policy = policy_factory()
            observed.append(policy)
            return Result(1 if policy == "expert" else 0)

        with tempfile.TemporaryDirectory() as directory:
            comparison = research_module.evaluate_paired_adapters(
                development_scenarios=scenarios,
                bc0_adapter=Path("frozen"),
                r1_adapter=None,
                base_model="gemma",
                base_revision="f" * 40,
                output_dir=Path(directory),
                seed=4,
                max_steps=8,
                policy_loader=lambda path, **_kwargs: str(path),
                environment_factory=lambda **_kwargs: object(),
                evaluator=evaluator,
                expert_policy_factory=lambda: "expert",
                evaluation_dirname="zeroshot",
            )
            written = sorted(path.name for path in (Path(directory) / "zeroshot").iterdir())
        self.assertEqual(observed, ["frozen", "expert"])
        self.assertEqual(written, ["bc0_eval.json", "comparison.json", "expert_eval.json"])
        self.assertIsNone(comparison["r1_adapter"])
        self.assertIsNone(comparison["r1_overall"])
        self.assertIsNone(comparison["r1_minus_bc0"])
        self.assertEqual(comparison["bc0_overall"]["resolved_episodes"], 0)
        self.assertEqual(comparison["expert_overall"]["resolved_episodes"], 1)

    def test_student_only_cli_flag(self) -> None:
        required = [
            "--d0-raw", "raw.jsonl", "--d0-train", "train.jsonl",
            "--adapter-path", "adapter", "--output-dir", "output",
        ]
        self.assertFalse(parser().parse_args(required).eval_student_only)
        self.assertTrue(parser().parse_args(required + ["--eval-student-only"]).eval_student_only)


def _chat_row(example_id: str, root: str, *, profile: str = "wls_gated_diagnostics") -> dict:
    return {
        "example_id": example_id,
        "physical_root_fingerprint": root,
        "metadata": {"protocol": "canonical", "evidence_profile": profile},
    }


class CumulativeAggregateTests(unittest.TestCase):
    """The 2026-10-29 mixture: earlier rounds' safe rows stay in full.

    Under the 2026-09-07 recipe round 2 mixed its new rows 1:1 with a uniform
    sample of D0 plus D1, so the corrections R1 had learned at its own failure
    states reached R2's training pass only in proportion to D1's share of the
    pool (about one row in seven).  The cumulative mixture is the DAgger
    aggregate: new rows plus every earlier round's rows against a D0 sample.
    """

    def test_cumulative_mixture_keeps_every_prior_round_row(self) -> None:
        d0 = [_chat_row(f"d0_{i}", f"d0_root_{i}") for i in range(40)]
        d1 = [_chat_row(f"d1_{i}", f"d1_root_{i}") for i in range(4)]
        prior = [
            dict(_chat_row(f"p_{i}", f"p_root_{i}"), research_prior_onpolicy_round=1)
            for i in range(12)
        ]
        mixture, report = build_research_mixture(
            d0, d1, d1_share=0.5, d1_cap=None, seed=3, prior_onpolicy_rows=prior
        )
        self.assertEqual(report["contract"], RESEARCH_MIXTURE_CONTRACT)
        self.assertEqual(len(mixture), 32)
        self.assertEqual(report["d0_selected"], 16)
        self.assertEqual(report["d1_selected"], 4)
        self.assertEqual(report["prior_onpolicy_available"], 12)
        self.assertEqual(report["prior_onpolicy_selected"], 12)
        self.assertEqual(report["prior_onpolicy_retained_fraction"], 1.0)
        self.assertEqual(report["prior_onpolicy_rows_by_round"], {"1": 12})
        self.assertEqual(report["onpolicy_selected"], 16)
        self.assertEqual(report["actual_onpolicy_share"], 0.5)
        self.assertEqual(report["actual_d1_share"], 4 / 32)
        self.assertEqual(report["mixture_rows"], 32)
        sources = {}
        for row in mixture:
            sources.setdefault(row["research_mixture_source"], set()).add(row["example_id"])
            self.assertEqual(row["metadata"]["research_mixture_source"], row["research_mixture_source"])
        self.assertEqual(sources["d1"], {f"d1_{i}" for i in range(4)})
        self.assertEqual(sources["d1_prior"], {f"p_{i}" for i in range(12)})
        self.assertEqual(len(sources["d0"]), 16)
        # The report carries the (empty) exact-conflict audit for rows whose
        # prompt is not the JSON observation contract.
        self.assertEqual(report["label_conflicts"]["conflicting_observations"], 0)
        self.assertEqual(report["label_conflicts"]["unaudited_rows"], 32)

    def test_pool_sampling_drops_most_prior_rows_while_cumulative_keeps_them(self) -> None:
        d0 = [_chat_row(f"d0_{i}", f"d0_root_{i}") for i in range(360)]
        d1 = [_chat_row(f"d1_{i}", f"d1_root_{i}") for i in range(60)]
        prior = [
            dict(_chat_row(f"p_{i}", f"p_root_{i}"), research_prior_onpolicy_round=1)
            for i in range(60)
        ]
        # 2026-09-07 recipe: the earlier round's rows compete with D0 inside
        # the sampled pool, so their expected survival is their pool share.
        pooled, pooled_report = build_research_mixture(
            d0 + prior, d1, d1_share=0.5, d1_cap=None, seed=11, audit_conflicts=False
        )
        survived = sum(row["example_id"].startswith("p_") for row in pooled)
        self.assertEqual(pooled_report["d0_selected"], 60)
        self.assertLess(survived, 20)
        cumulative, report = build_research_mixture(
            d0, d1, d1_share=0.5, d1_cap=None, seed=11,
            prior_onpolicy_rows=prior, audit_conflicts=False,
        )
        self.assertEqual(sum(row["example_id"].startswith("p_") for row in cumulative), 60)
        self.assertEqual(report["d0_selected"], 120)
        self.assertEqual(len(cumulative), 240)
        self.assertNotIn("label_conflicts", report)

    def test_without_prior_rows_the_mixture_is_unchanged(self) -> None:
        d0 = [_chat_row(f"d0_{i}", f"d0_root_{i}") for i in range(20)]
        d1 = [_chat_row(f"d1_{i}", f"d1_root_{i}") for i in range(4)]
        mixture, report = build_research_mixture(d0, d1, d1_share=0.25, d1_cap=None, seed=3)
        again, _ = build_research_mixture(d0, d1, d1_share=0.25, d1_cap=None, seed=3, prior_onpolicy_rows=[])
        self.assertEqual([row["example_id"] for row in mixture], [row["example_id"] for row in again])
        self.assertEqual(report["prior_onpolicy_available"], 0)
        self.assertIsNone(report["prior_onpolicy_retained_fraction"])
        self.assertEqual(report["actual_onpolicy_share"], report["actual_d1_share"])

    def test_prior_rows_must_keep_roots_disjoint_from_d0_and_the_new_round(self) -> None:
        d0 = [_chat_row("d0", "root_d0")]
        d1 = [_chat_row("d1", "root_d1")]
        for prior_root in ("root_d0", "root_d1"):
            with self.assertRaisesRegex(ValueError, "Prior on-policy rows overlap"):
                build_research_mixture(
                    d0, d1, d1_share=0.5, d1_cap=None, seed=1,
                    prior_onpolicy_rows=[_chat_row("p", prior_root)],
                )
        with self.assertRaisesRegex(ValueError, "Prior on-policy chat rows must retain"):
            build_research_mixture(
                d0, d1, d1_share=0.5, d1_cap=None, seed=1,
                prior_onpolicy_rows=[{"example_id": "rootless", "metadata": {"evidence_profile": "wls_gated_diagnostics"}}],
            )
        with self.assertRaisesRegex(ValueError, "evidence profile"):
            build_research_mixture(
                d0, d1, d1_share=0.5, d1_cap=None, seed=1,
                prior_onpolicy_rows=[_chat_row("p", "root_p", profile="scada_only")],
            )

    def test_exact_conflict_audit_reports_one_observation_with_two_targets(self) -> None:
        first = _raw_row("root_a")
        state_id = first["policy_observation"]["active_state_id"]
        other_target = dict(first, example_id="example_root_a_other")
        other_target["preferred_action"] = {
            "tool": "get_measurement_context",
            "arguments": {"state_id": state_id},
        }
        same_target = dict(first, example_id="example_root_a_same")
        rows = examples_to_chat_sft(
            [first, other_target, same_target], protocol="canonical", allow_ineligible_auxiliary=True
        )
        report = audit_mixture_label_conflicts(rows)
        self.assertEqual(report["contract"], "research_mixture_exact_label_conflicts_v1")
        self.assertEqual(report["audited_rows"], 3)
        self.assertEqual(report["unaudited_rows"], 0)
        self.assertEqual(report["unique_observations"], 1)
        self.assertEqual(report["conflicting_observations"], 1)
        self.assertEqual(report["conflicting_rows"], 3)
        self.assertEqual(report["conflict_rate"], 1.0)
        self.assertEqual(
            report["conflicts"][0]["example_ids"],
            ["example_root_a", "example_root_a_other", "example_root_a_same"],
        )
        self.assertEqual(len(report["conflicts"][0]["semantic_actions"]), 2)
        agreeing = audit_mixture_label_conflicts([rows[0], rows[2]])
        self.assertEqual(agreeing["conflicting_observations"], 0)
        self.assertEqual(agreeing["unique_observations"], 1)
        # Through the mixture builder the audit lands in the report.
        d0 = [dict(rows[2], physical_root_fingerprint="root_d0", example_id="d0_row")]
        _mixture, mixture_report = build_research_mixture(
            d0, [rows[0]], d1_share=0.5, d1_cap=None, seed=1,
            prior_onpolicy_rows=[dict(rows[1], physical_root_fingerprint="root_prior")],
        )
        self.assertEqual(mixture_report["label_conflicts"]["conflicting_observations"], 1)
        self.assertEqual(mixture_report["label_conflicts"]["conflicting_rows"], 3)

    def _write_round(self, directory: Path, roots: list[str], *, ineligible: str | None = None) -> list[dict]:
        raw = [mark_research_label_eligibility(_raw_row(root)) for root in roots]
        if ineligible is not None:
            raw.append(mark_research_label_eligibility(_raw_row(ineligible, audit_passed=False)))
        chat = examples_to_chat_sft(
            [row for row in raw if row["research_label_eligible"]],
            protocol="canonical",
            allow_ineligible_auxiliary=True,
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / PRIOR_ONPOLICY_RAW_FILENAME).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw), encoding="utf-8"
        )
        (directory / PRIOR_ONPOLICY_CHAT_FILENAME).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in chat), encoding="utf-8"
        )
        return chat

    def test_prior_round_loader_rebinds_rows_and_tags_their_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "r1" / "collection"
            second = Path(temp) / "r2" / "collection"
            self._write_round(first, ["root_a", "root_b"], ineligible="root_q")
            self._write_round(second, ["root_c"])
            rows, reports = load_prior_onpolicy_rounds([first, second])
            self.assertEqual([row["example_id"] for row in rows], ["example_root_a", "example_root_b", "example_root_c"])
            self.assertEqual([row["research_prior_onpolicy_round"] for row in rows], [1, 1, 2])
            self.assertEqual(rows[0]["metadata"]["research_prior_onpolicy_round"], 1)
            self.assertEqual(rows[2]["metadata"]["research_prior_onpolicy_dir"], str(second.resolve()))
            self.assertEqual({row["physical_root_fingerprint"] for row in rows}, {"root_a", "root_b", "root_c"})
            self.assertEqual([report["ordinal"] for report in reports], [1, 2])
            self.assertEqual(reports[0]["raw_rows"], 3)
            self.assertEqual(reports[0]["eligible_raw_rows"], 2)
            self.assertEqual(reports[0]["chat_rows"], 2)
            self.assertEqual(reports[0]["physical_roots"], 2)
            self.assertEqual(reports[0]["observation_view"]["rerendered_rows"], 2)
            self.assertEqual(len(reports[0]["chat_sha256"]), 64)
            # The quarantined row never enters the aggregate.
            self.assertNotIn("example_root_q", {row["example_id"] for row in rows})
            # The rows carry no private collection fields into the prompt.
            for row in rows:
                self.assertNotIn("private_truth_marker", json.dumps(row["messages"]))

    def test_prior_round_loader_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing"
            missing.mkdir()
            with self.assertRaisesRegex(ValueError, "lacks"):
                load_prior_onpolicy_rounds([missing])
            round_dir = Path(temp) / "r1"
            chat = self._write_round(round_dir, ["root_a"])
            # A chat row the ledger does not mark eligible is refused.
            (round_dir / PRIOR_ONPOLICY_CHAT_FILENAME).write_text(
                json.dumps(dict(chat[0], example_id="example_root_zz")) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not mark research_label_eligible"):
                load_prior_onpolicy_rounds([round_dir])
            # The same round listed twice repeats example ids.
            self._write_round(round_dir, ["root_a"])
            with self.assertRaisesRegex(ValueError, "repeat example_id"):
                load_prior_onpolicy_rounds([round_dir, round_dir])
            self.assertEqual(load_prior_onpolicy_rounds([]), ([], []))

    def test_fixed_suites_reject_roots_of_earlier_rounds(self) -> None:
        from scripts.run_dagger_research import prepare_scenario_split

        scenario = lambda root: {
            "scenario_id": f"hif-{root}",
            "grouping": {"physical_root_fingerprint": root, "scenario_family": "hif"},
        }
        with tempfile.TemporaryDirectory() as temp:
            # run() passes earlier rounds' roots as protected roots, so a
            # training suite that reuses a round-1 root is refused.
            with self.assertRaisesRegex(ValueError, "D0 or protected roots"):
                prepare_scenario_split(
                    output_dir=Path(temp),
                    d0_raw_path=Path(temp) / "d0.jsonl",
                    d0_roots={"d0"},
                    train_plan={"hif": 1},
                    development_plan={"hif": 1},
                    candidate_multiplier=3,
                    seed=7,
                    run_descriptor={},
                    protected_roots={"p"} | {"round1_root"},
                    fixed_training=[scenario("round1_root")],
                    fixed_development=[scenario("dev")],
                    scenario_suite={},
                )
