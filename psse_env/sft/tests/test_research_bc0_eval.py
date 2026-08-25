from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from psse_env.dagger.release_factories import deterministic_case_loader
from psse_env.research_models import GEMMA4_12B
from psse_env.sft import research_bc0_eval
from psse_env.sft.research_bc0_eval import (
    DEFAULT_D1_PLAN,
    EXPECTED_D0_STANDARD_ROOTS,
    RESEARCH_BC0_EVAL_CONTRACT,
    _d0_predecessor,
    adapter_content_fingerprint,
    build_d1_development_suite,
    evaluate_research_suite,
    load_d0_training_roots,
    load_frozen_standard_suite,
    summarize_closed_loop_outcomes,
    summarize_policy_behavior,
)


def _scenario(root: str, family: str, *, scenario_id: str | None = None) -> dict[str, Any]:
    identifier = scenario_id or f"scenario_{root}"
    return {
        "scenario_schema_version": 1,
        "execution": {
            "scenario_id": identifier,
            "case": "case14",
            "measurements": [],
        },
        "audit": {
            "truth": {"truth_complete": True},
            "evaluation_intervention": {
                "intervention_schema_version": 1,
                "kind": "none",
            },
        },
        "grouping": {
            "root_scenario_id": identifier,
            "physical_root_fingerprint": root,
            "scenario_family": family,
            "error_cardinality": 1,
            "case_id": "case14",
            "split": "development",
            "source_tier": "test",
        },
    }


class _FakeGenerator:
    rows: list[dict[str, Any]] = []
    constructor_kwargs: dict[str, Any] = {}
    requested: dict[str, int] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).constructor_kwargs = dict(kwargs)

    def build(self, requested: Mapping[str, int]) -> list[dict[str, Any]]:
        type(self).requested = dict(requested)
        return copy.deepcopy(type(self).rows)


def _partition(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    envelope = _scenario(str(row["root"]), str(row["family"]))
    envelope["grouping"]["split"] = split
    return envelope


class ResearchBc0DevelopmentSuiteTests(unittest.TestCase):
    def test_builds_exact_deterministic_root_disjoint_6_6_3_mix(self) -> None:
        rows: list[dict[str, Any]] = []
        for family in DEFAULT_D1_PLAN:
            rows.extend(
                {"family": family, "root": f"fresh_{family}_{index:02d}"}
                for index in range(12)
            )
            rows.extend(
                (
                    {"family": family, "root": "d0_training_root"},
                    {"family": family, "root": "frozen_standard_root"},
                )
            )
        _FakeGenerator.rows = rows

        selected, report = build_d1_development_suite(
            d0_training_roots={"d0_training_root"},
            frozen_standard_roots={"frozen_standard_root"},
            seed=17,
            candidate_multiplier=3,
            generator_factory=_FakeGenerator,
            partitioner=_partition,
        )
        reversed_rows = list(reversed(rows))
        _FakeGenerator.rows = reversed_rows
        selected_again, _ = build_d1_development_suite(
            d0_training_roots={"d0_training_root"},
            frozen_standard_roots={"frozen_standard_root"},
            seed=17,
            candidate_multiplier=3,
            generator_factory=_FakeGenerator,
            partitioner=_partition,
        )

        self.assertEqual(len(selected), 15)
        self.assertEqual(
            Counter(row["grouping"]["scenario_family"] for row in selected),
            Counter(DEFAULT_D1_PLAN),
        )
        roots = {
            row["grouping"]["physical_root_fingerprint"] for row in selected
        }
        self.assertFalse(roots & {"d0_training_root", "frozen_standard_root"})
        self.assertEqual(selected, selected_again)
        self.assertEqual(report["root_isolation"]["d0_training_overlap"], [])
        self.assertEqual(report["root_isolation"]["d0_standard_overlap"], [])
        self.assertEqual(
            _FakeGenerator.constructor_kwargs,
            {
                "seed": 17,
                "source_partition": "train",
                "parameter_ranking_dominance_threshold": 1.0,
            },
        )
        self.assertEqual(
            _FakeGenerator.requested,
            {family: count * 3 for family, count in sorted(DEFAULT_D1_PLAN.items())},
        )

    def test_rejects_insufficient_fresh_family_roots(self) -> None:
        _FakeGenerator.rows = [
            {"family": family, "root": f"{family}_{index}"}
            for family, count in DEFAULT_D1_PLAN.items()
            for index in range(count - 1)
        ]
        with self.assertRaisesRegex(Exception, "fresh roots"):
            build_d1_development_suite(
                d0_training_roots={"d0"},
                frozen_standard_roots={"holdout"},
                generator_factory=_FakeGenerator,
                partitioner=_partition,
            )


class _FakePolicy:
    def __init__(self) -> None:
        self.last_action_metrics: dict[str, Any] = {}

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self.last_action_metrics = {
            "prompt_tokens": 128,
            "original_prompt_tokens": 128,
            "truncated_input_tokens": 0,
            "generated_tokens": 9,
            "hit_max_new_tokens": False,
            "repetition_loop_detected": False,
        }
        return {
            "tool": "run_wls",
            "arguments": {"state_id": observation["active_state_id"]},
        }


class _FakeExpert:
    def next_actions(
        self, observation: Mapping[str, Any], history: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        del history
        return [
            {
                "tool": "run_wls",
                "arguments": {"state_id": observation["active_state_id"]},
            }
        ]


class _FakeEvaluationResult:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = copy.deepcopy(dict(payload))

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


class ResearchBc0EvaluationTests(unittest.TestCase):
    def test_loads_explicit_adapter_with_exact_pinned_12b_identity(self) -> None:
        calls: dict[str, Any] = {}

        def loader(path: Path, **kwargs: Any) -> _FakePolicy:
            calls["path"] = path
            calls["kwargs"] = kwargs
            return _FakePolicy()

        def evaluator(suites: Mapping[str, Any], **kwargs: Any) -> _FakeEvaluationResult:
            calls["evaluator_kwargs"] = kwargs
            policy = kwargs["policy_factory"]()
            observation = {
                "active_state_id": "active",
                "candidate_state_id": None,
                "remaining_budget": 5,
                "history_window": [],
            }
            action = policy.act(observation)
            episode = {
                "evaluator_error": None,
                "trace": [
                    {
                        "intervention": False,
                        "policy_observation": observation,
                        "action": action,
                    }
                ],
            }
            return _FakeEvaluationResult(
                {
                    "score": 0.0,
                    "metrics": {},
                    "suite_metrics": {"episodes": [episode]},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            payload, behavior = evaluate_research_suite(
                [_scenario("root", "parameter")],
                adapter_path=adapter,
                seed=23,
                max_steps=24,
                policy_loader=loader,
                evaluator=evaluator,
                environment_factory=lambda **_kwargs: None,
                expert_factory=_FakeExpert,
                progress_callback=None,
            )

        self.assertEqual(calls["path"], adapter)
        self.assertEqual(
            calls["kwargs"],
            {
                "base_model": GEMMA4_12B.model_id,
                "base_revision": GEMMA4_12B.revision,
                "architecture": GEMMA4_12B.architecture,
                "prompt_profile": GEMMA4_12B.prompt_profile,
                "load_in_4bit": True,
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
        self.assertEqual(calls["evaluator_kwargs"]["max_steps"], 24)
        self.assertIs(
            calls["evaluator_kwargs"]["case_loader"], deterministic_case_loader
        )
        self.assertFalse(calls["evaluator_kwargs"]["require_release_environment"])
        self.assertFalse(calls["evaluator_kwargs"]["require_policy_identity"])
        self.assertIn("suite_metrics", payload)
        self.assertEqual(behavior["schema_valid_action_rate"], 1.0)
        self.assertEqual(behavior["observable_expert_tool_agreement_rate"], 1.0)
        self.assertEqual(
            behavior["observable_expert_exact_action_agreement_rate"], 1.0
        )
        self.assertEqual(behavior["input_truncated_steps"], 0)
        self.assertEqual(behavior["maximum_original_prompt_tokens"], 128)

    def test_behavior_summary_distinguishes_schema_validity_from_tool_failure(self) -> None:
        evaluation = {
            "suite_metrics": {
                "episodes": [
                    {
                        "evaluator_error": None,
                        "trace": [
                            {
                                "intervention": False,
                                "policy_observation": {
                                    "active_state_id": "active",
                                    "history_window": [],
                                },
                                "action": {
                                    "tool": "run_wls",
                                    "arguments": {"state_id": "active"},
                                },
                                "execution_status": "failure",
                            },
                            {
                                "intervention": False,
                                "policy_observation": {
                                    "active_state_id": "active",
                                    "history_window": [],
                                },
                                "action": {
                                    "tool": "__invalid_action__",
                                    "arguments": {"error_code": "schema_error"},
                                },
                                "execution_status": "failure",
                            },
                        ],
                    }
                ]
            }
        }
        behavior = summarize_policy_behavior(
            evaluation, [], expert_factory=_FakeExpert
        )
        self.assertEqual(behavior["policy_steps"], 2)
        self.assertEqual(behavior["schema_valid_actions"], 1)
        self.assertEqual(behavior["schema_valid_action_rate"], 0.5)

    def test_explicit_case_loader_override_reaches_closed_loop_evaluator(self) -> None:
        calls: dict[str, Any] = {}

        def sentinel_loader(value: Any) -> Any:
            return value

        def evaluator(_suites: Mapping[str, Any], **kwargs: Any) -> _FakeEvaluationResult:
            calls.update(kwargs)
            return _FakeEvaluationResult(
                {
                    "score": 0.0,
                    "metrics": {},
                    "suite_metrics": {"episodes": []},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            evaluate_research_suite(
                [_scenario("root", "parameter")],
                adapter_path=Path(directory),
                seed=23,
                max_steps=24,
                policy_loader=lambda *_args, **_kwargs: _FakePolicy(),
                evaluator=evaluator,
                environment_factory=lambda **_kwargs: None,
                case_loader=sentinel_loader,
                expert_factory=_FakeExpert,
                progress_callback=None,
            )

        self.assertIs(calls["case_loader"], sentinel_loader)

    def test_outcome_summary_does_not_conflate_resolution_and_handoff(self) -> None:
        summary = summarize_closed_loop_outcomes(
            {
                "suite_metrics": {
                    "overall": {
                        "episodes": 4,
                        "final_physical_success_episodes": 1,
                        "final_physical_success_rate": 0.25,
                        "audited_post_correction_handoff_episodes": 2,
                        "audited_post_correction_handoff_rate": 0.5,
                        "audited_completion_episodes": 3,
                        "audited_completion_rate": 0.75,
                    }
                }
            }
        )

        self.assertEqual(
            summary["strict_resolved_physical_success"],
            {
                "episodes": 1,
                "rate": 0.25,
                "terminal_outcome": "resolved",
                "requires_strict_physical_audit": True,
            },
        )
        self.assertEqual(
            summary["audited_post_correction_handoff"],
            {
                "episodes": 2,
                "rate": 0.5,
                "terminal_outcome": "operator_escalation",
                "requires_versioned_safety_clean_assessment": True,
            },
        )
        self.assertEqual(
            summary["audited_completion_union"],
            {"episodes": 3, "rate": 0.75},
        )


class ResearchBc0InputTests(unittest.TestCase):
    def test_cli_returns_nonzero_when_readiness_fails(self) -> None:
        report = {
            "passed": False,
            "phase": "d0",
            "closed_loop_outcomes": {},
            "readiness_gate": {"passed": False, "failures": ["schema"]},
            "policy_behavior": {"schema_valid_action_rate": 0.5},
        }
        with patch.object(research_bc0_eval, "run", return_value=report):
            code = research_bc0_eval.main(
                [
                    "--phase",
                    "d0",
                    "--adapter-path",
                    "adapter",
                    "--d0-train",
                    "train.jsonl",
                    "--output-dir",
                    "output",
                ]
            )
        self.assertEqual(code, 3)

    def test_d1_predecessor_rejects_same_path_adapter_weight_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "lora"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA"}), encoding="utf-8"
            )
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"first adapter bytes")
            original = adapter_content_fingerprint(adapter)
            report_path = root / "d0.json"
            report_path.write_text(
                json.dumps(
                    {
                        "contract": RESEARCH_BC0_EVAL_CONTRACT,
                        "phase": "d0",
                        "passed": True,
                        "evaluation_completed": True,
                        "readiness_gate": {"passed": True},
                        "adapter": original,
                    }
                ),
                encoding="utf-8",
            )
            _d0_predecessor(report_path, original["content_sha256"])
            weights.write_bytes(b"second, different adapter bytes")
            changed = adapter_content_fingerprint(adapter)
            self.assertNotEqual(original, changed)
            with self.assertRaisesRegex(Exception, "same adapter bytes"):
                _d0_predecessor(report_path, changed["content_sha256"])

    def test_packaged_standard_holdout_is_exactly_21_unique_roots(self) -> None:
        rows = load_frozen_standard_suite()
        roots = {row["grouping"]["physical_root_fingerprint"] for row in rows}
        self.assertEqual(len(rows), EXPECTED_D0_STANDARD_ROOTS)
        self.assertEqual(len(roots), EXPECTED_D0_STANDARD_ROOTS)

    def test_training_root_loader_accepts_metadata_and_top_level_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            rows = [
                {
                    "example_id": "a",
                    "physical_root_fingerprint": "root_a",
                },
                {
                    "example_id": "b",
                    "metadata": {"physical_root_fingerprint": "root_b"},
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            roots, count = load_d0_training_roots(path)
        self.assertEqual(roots, {"root_a", "root_b"})
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
