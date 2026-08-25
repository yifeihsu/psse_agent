from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from psse_env.sft.gates import GateError
from psse_env.sft import research_bc0_checkpoint_compare
from psse_env.sft.research_bc0_checkpoint_compare import (
    COMPARISON_REPORT_NAME,
    AdapterSpec,
    compare_checkpoints,
    load_frozen_suite,
    parse_adapter_spec,
)


def _scenario(root: str) -> dict[str, Any]:
    return {
        "grouping": {
            "physical_root_fingerprint": root,
            "scenario_family": "parameter",
        }
    }


def _write_suite(path: Path, roots: list[str]) -> None:
    path.write_text(
        json.dumps({"standard_success": [_scenario(root) for root in roots]}),
        encoding="utf-8",
    )


def _adapter(path: Path, marker: str) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"marker": marker}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_bytes(marker.encode("utf-8"))
    return path


class _FakePolicy:
    def __init__(self, profile: Mapping[str, Any]) -> None:
        self.profile = copy.deepcopy(dict(profile))
        self.last_action_metrics: dict[str, Any] = {}

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self.last_action_metrics = {
            "prompt_tokens": 128,
            "original_prompt_tokens": 128,
            "truncated_input_tokens": 0,
            "generated_tokens": 8,
            "hit_max_new_tokens": False,
        }
        return {
            "tool": str(self.profile.get("tool", "run_wls")),
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


class _Harness:
    def __init__(self, profiles: Mapping[str, Mapping[str, Any]]) -> None:
        self.profiles = {
            str(Path(path).resolve()): copy.deepcopy(dict(profile))
            for path, profile in profiles.items()
        }
        self.loader_calls: list[str] = []
        self.evaluator_calls: list[str] = []
        self.case_loaders: list[Any] = []

    def loader(self, path: Path, **_kwargs: Any) -> _FakePolicy:
        key = str(path.resolve())
        self.loader_calls.append(key)
        return _FakePolicy(self.profiles[key])

    def evaluator(
        self, suites: Mapping[str, Any], **kwargs: Any
    ) -> _FakeEvaluationResult:
        policy = kwargs["policy_factory"]()
        profile = policy.policy.profile
        label = str(profile["label"])
        self.evaluator_calls.append(label)
        self.case_loaders.append(kwargs["case_loader"])
        episodes: list[dict[str, Any]] = []
        for index, _scenario_row in enumerate(suites["standard_success"]):
            observation = {
                "active_state_id": f"state_{index}",
                "candidate_state_id": None,
                "remaining_budget": 5,
                "history_window": [],
            }
            episodes.append(
                {
                    "evaluator_error": None,
                    "trace": [
                        {
                            "intervention": False,
                            "policy_observation": observation,
                            "action": policy.act(observation),
                        }
                    ],
                }
            )
        count = len(episodes)
        overall = {
            "episodes": count,
            "audited_completion_episodes": profile["audited"],
            "audited_completion_rate": profile["audited"] / count,
            "audited_post_correction_handoff_episodes": profile["handoff"],
            "audited_post_correction_handoff_rate": profile["handoff"] / count,
            "final_physical_success_episodes": profile["strict"],
            "final_physical_success_rate": profile["strict"] / count,
            "invalid_action_count": profile["invalid"],
            "episodes_with_invalid_actions": int(profile["invalid"] > 0),
            "false_commit_count": profile["false_commits"],
            "false_commit_episodes": int(profile["false_commits"] > 0),
            "false_commit_rate": int(profile["false_commits"] > 0) / count,
            "loop_episodes": profile["loops"],
            "loop_rate": profile["loops"] / count,
        }
        return _FakeEvaluationResult(
            {
                "score": 0.0,
                "metrics": {},
                "suite_metrics": {"overall": overall, "episodes": episodes},
            }
        )


class CheckpointComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.suite = self.root / "suite.json"
        _write_suite(self.suite, [f"root_{index}" for index in range(4)])
        self.alpha = _adapter(self.root / "alpha", "alpha")
        self.beta = _adapter(self.root / "beta", "beta")
        self.output = self.root / "output"
        self.profiles = {
            str(self.alpha): {
                "label": "alpha",
                "audited": 3,
                "handoff": 2,
                "strict": 1,
                "invalid": 1,
                "false_commits": 0,
                "loops": 0,
                "tool": "run_wls",
            },
            str(self.beta): {
                "label": "beta",
                "audited": 2,
                "handoff": 0,
                "strict": 2,
                "invalid": 0,
                "false_commits": 0,
                "loops": 0,
                "tool": "get_measurement_context",
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, harness: _Harness, **overrides: Any) -> dict[str, Any]:
        options = {
            "suite_json": self.suite,
            "adapters": [
                AdapterSpec("alpha", self.alpha),
                AdapterSpec("beta", self.beta),
            ],
            "output_dir": self.output,
            "seed": 17,
            "max_steps": 24,
            "policy_loader": harness.loader,
            "evaluator": harness.evaluator,
            "environment_factory": lambda **_kwargs: None,
            "case_loader": lambda value: value,
            "expert_factory": _FakeExpert,
            "progress_callback": None,
        }
        options.update(overrides)
        with patch.object(
            research_bc0_checkpoint_compare, "_configure_input_ceiling"
        ):
            return compare_checkpoints(**options)

    def test_evaluates_each_adapter_and_ranks_audited_completion_first(self) -> None:
        harness = _Harness(self.profiles)
        comparison = self._run(harness)

        self.assertEqual(harness.evaluator_calls, ["alpha", "beta"])
        self.assertEqual(
            [row["label"] for row in comparison["ranking"]], ["alpha", "beta"]
        )
        alpha = comparison["ranking"][0]
        beta = comparison["ranking"][1]
        self.assertEqual(alpha["audited_completion"], {"episodes": 3, "rate": 0.75})
        self.assertEqual(alpha["audited_post_correction_handoff"]["rate"], 0.5)
        self.assertEqual(alpha["strict_resolved_physical_success"]["rate"], 0.25)
        self.assertEqual(alpha["schema_valid_action_rate"], 1.0)
        self.assertEqual(alpha["observable_expert_exact_action_agreement_rate"], 1.0)
        self.assertEqual(alpha["invalid_actions"]["count"], 1)
        self.assertEqual(beta["strict_resolved_physical_success"]["rate"], 0.5)
        self.assertEqual(beta["observable_expert_tool_agreement_rate"], 0.0)
        self.assertTrue((self.output / COMPARISON_REPORT_NAME).is_file())
        self.assertTrue((self.output / "research_bc0_checkpoint_alpha.json").is_file())
        self.assertTrue((self.output / "research_bc0_checkpoint_beta.json").is_file())

    def test_reuses_only_completed_matching_adapter_report(self) -> None:
        harness = _Harness(self.profiles)
        self._run(harness)
        first_call_count = len(harness.evaluator_calls)
        reused = self._run(harness)
        self.assertEqual(len(harness.evaluator_calls), first_call_count)
        self.assertEqual(reused["reused_labels"], ["alpha", "beta"])

        alpha_report = self.output / "research_bc0_checkpoint_alpha.json"
        payload = json.loads(alpha_report.read_text(encoding="utf-8"))
        payload["evaluation_completed"] = False
        alpha_report.write_text(json.dumps(payload), encoding="utf-8")
        resumed = self._run(harness)
        self.assertEqual(harness.evaluator_calls[-1], "alpha")
        self.assertEqual(resumed["evaluated_labels"], ["alpha"])
        self.assertEqual(resumed["reused_labels"], ["beta"])

        payload = json.loads(alpha_report.read_text(encoding="utf-8"))
        payload["suite"]["content_sha256"] = "mismatched"
        alpha_report.write_text(json.dumps(payload), encoding="utf-8")
        mismatched = self._run(harness)
        self.assertEqual(harness.evaluator_calls[-1], "alpha")
        self.assertEqual(mismatched["evaluated_labels"], ["alpha"])

        payload = json.loads(alpha_report.read_text(encoding="utf-8"))
        del payload["policy_behavior"]["schema_valid_action_rate"]
        alpha_report.write_text(json.dumps(payload), encoding="utf-8")
        incomplete_metrics = self._run(harness)
        self.assertEqual(harness.evaluator_calls[-1], "alpha")
        self.assertEqual(incomplete_metrics["evaluated_labels"], ["alpha"])

    def test_case_loader_and_gpu_policy_loader_remain_injectable(self) -> None:
        harness = _Harness(self.profiles)

        def sentinel_case_loader(value: Any) -> Any:
            return value

        comparison = self._run(harness, case_loader=sentinel_case_loader)
        self.assertEqual(len(harness.loader_calls), 2)
        self.assertEqual(
            harness.case_loaders, [sentinel_case_loader, sentinel_case_loader]
        )
        self.assertTrue(comparison["comparison_completed"])

    def test_rejects_duplicate_labels_and_adapter_fingerprints(self) -> None:
        harness = _Harness(self.profiles)
        with self.assertRaisesRegex(GateError, "labels must be unique"):
            self._run(
                harness,
                adapters=[
                    AdapterSpec("same", self.alpha),
                    AdapterSpec("SAME", self.beta),
                ],
            )

        duplicate = _adapter(self.root / "duplicate", "alpha")
        with self.assertRaisesRegex(GateError, "fingerprints must be unique"):
            self._run(
                harness,
                adapters=[
                    AdapterSpec("alpha", self.alpha),
                    AdapterSpec("copy", duplicate),
                ],
            )

    def test_rejects_missing_or_duplicate_suite_roots(self) -> None:
        duplicate_suite = self.root / "duplicate_suite.json"
        _write_suite(duplicate_suite, ["same", "same"])
        with self.assertRaisesRegex(GateError, "roots are not unique"):
            load_frozen_suite(duplicate_suite)

        missing_suite = self.root / "missing_suite.json"
        missing_suite.write_text(
            json.dumps({"standard_success": [{"grouping": {}}]}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(GateError, "needs a physical root"):
            load_frozen_suite(missing_suite)

    def test_parses_label_path_and_rejects_unsafe_labels(self) -> None:
        parsed = parse_adapter_spec(f"alpha={self.alpha}")
        self.assertEqual(parsed, AdapterSpec("alpha", self.alpha))
        with self.assertRaisesRegex(Exception, "LABEL=ADAPTER_PATH"):
            parse_adapter_spec("alpha")
        with self.assertRaisesRegex(Exception, "adapter label"):
            parse_adapter_spec(f"../alpha={self.alpha}")


if __name__ == "__main__":
    unittest.main()
