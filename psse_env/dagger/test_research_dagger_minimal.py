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
    RESEARCH_HIF_SEARCH_BUDGET,
    allocate_scenarios,
    build_research_mixture,
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
            "evidence_profile": "scada_only",
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
                "metadata": {"protocol": "canonical", "evidence_profile": "scada_only"},
            }
            for index in range(20)
        ]
        d1 = [
            {
                "example_id": f"d1_{index}",
                "physical_root_fingerprint": f"d1_root_{index}",
                "metadata": {"protocol": "canonical", "evidence_profile": "scada_only"},
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
        self.assertEqual(sources["evidence_profile"], "scada_only")
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
            )["evidence_profile"], "scada_only"
        )

    def test_harmonic_only_plan_records_modes_without_telemetry_corpora(self) -> None:
        sources = resolve_scenario_sources(plan_families={"harmonic"})
        self.assertEqual(
            sources,
            {
                "hif_sample_paths": None,
                "imbalance_sample_path": None,
                "signature_modes": {"harmonic": "discovered"},
                "evidence_profile": "scada_only",
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
        self.assertEqual(resolve_scenario_sources(plan_families={"measurement"}, system="case14")["evidence_profile"], "scada_only")
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
