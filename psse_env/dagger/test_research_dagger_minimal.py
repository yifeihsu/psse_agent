from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_dagger_research import (
    allocate_scenarios,
    build_research_mixture,
    collect_resumable,
    evaluate_paired_adapters,
    export_research_rows,
    is_research_dagger_row,
    load_protected_suite_roots,
    mark_research_label_eligibility,
    refresh_d0_training_view,
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
                "metadata": {"protocol": "canonical"},
            }
            for index in range(20)
        ]
        d1 = [
            {
                "example_id": f"d1_{index}",
                "physical_root_fingerprint": f"d1_root_{index}",
                "metadata": {"protocol": "canonical"},
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

        def evaluator(suites, *, policy_factory, **_kwargs):
            roots = [
                row["grouping"]["physical_root_fingerprint"]
                for row in suites["standard_success"]
            ]
            policy = policy_factory()
            observed.append((policy, roots))
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
            )
        self.assertEqual(observed[0][1], observed[1][1])
        self.assertEqual(observed[0][1], ["dev_a", "dev_b"])
        self.assertEqual(comparison["r1_minus_bc0"]["resolved_episodes"], 1.0)


if __name__ == "__main__":
    unittest.main()
