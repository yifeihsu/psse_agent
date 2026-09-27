"""The parallel paired evaluation reproduces the sequential one exactly.

The torch eval jobs roll the environment out on the CPU between model calls,
so one process leaves the GPU idle about half the time and the cluster
cancels the job.  ``ParallelEvaluation`` shards every policy's development
roots over worker processes; these tests run a small real suite both ways,
with the expert standing in for the adapters, and require the same reports,
and check that a requeued job resumes from its episode checkpoints.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import scripts.run_dagger_research as research_module
from psse_env.dagger.evaluator import (
    ClosedLoopRolloutEvaluator,
    EpisodeEvaluation,
    evaluate_rollout_suites,
)
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.evidence_profile import DEFAULT_EVIDENCE_PROFILE
from psse_env.providers.scenario_generator import Round0ScenarioGenerator

PLAN = {"measurement": 2, "parameter": 1, "topology": 1, "no_error": 1}
#: When set, every environment build appends the building process id here, so
#: a test can count the episodes each worker process actually rolled out.
ENVIRONMENT_COUNTER = "PSSE_TEST_ENVIRONMENT_COUNTER"


def environment_factory(**_kwargs):
    counter = os.environ.get(ENVIRONMENT_COUNTER)
    if counter:
        with open(counter, "a", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
    return research_module.resolve_environment_factory("release", DEFAULT_EVIDENCE_PROFILE)()


def expert_as_adapter(_adapter, **_kwargs):
    """Policy loader stand-in: the expert, whatever adapter is named."""

    return research_module.research_expert_policy(environment_factory)


def _scenarios():
    generator = Round0ScenarioGenerator(seed=20260927, normalized_residual_threshold=4.0)
    return [
        partition_release_scenario_v1(row, split="dagger_train")
        for row in generator.build(PLAN)
    ]


def _parallel(workers: int) -> research_module.ParallelEvaluation:
    return research_module.ParallelEvaluation(
        workers_per_policy=workers,
        environment={
            "hif_search_profile": "release",
            "evidence_profile": DEFAULT_EVIDENCE_PROFILE,
            "options": dict(research_module.RESEARCH_ENVIRONMENT_OPTIONS),
        },
        policy_loader="psse_env.dagger.test_parallel_evaluation:expert_as_adapter",
        environment_factory="psse_env.dagger.test_parallel_evaluation:environment_factory",
    )


def _evaluate(scenarios, output_dir: Path, parallel=None):
    return research_module.evaluate_paired_adapters(
        development_scenarios=scenarios,
        bc0_adapter=Path("student"),
        r1_adapter=Path("candidate"),
        base_model="gemma",
        base_revision="f" * 40,
        output_dir=output_dir,
        seed=4,
        max_steps=research_module.RESEARCH_EPISODE_BUDGET,
        policy_loader=expert_as_adapter,
        environment_factory=environment_factory,
        evaluator=evaluate_rollout_suites,
        expert_policy_factory=lambda: research_module.research_expert_policy(environment_factory),
        parallel=parallel,
    )


def _reports(directory: Path) -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "evaluation").glob("*.json"))
    }


class ParallelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = _scenarios()
        assert len(cls.scenarios) == sum(PLAN.values())

    def test_parallel_reports_match_the_sequential_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequential = _evaluate(self.scenarios, root / "sequential")
            parallel = _evaluate(self.scenarios, root / "parallel", parallel=_parallel(2))
            self.assertEqual(sequential, parallel)
            expected = _reports(root / "sequential")
            observed = _reports(root / "parallel")
            self.assertEqual(
                sorted(expected), ["bc0_eval.json", "comparison.json", "expert_eval.json", "r1_eval.json"]
            )
            self.assertEqual(expected, observed)
            # The checkpoints are removed once the merged reports are written.
            self.assertFalse((root / "parallel" / "evaluation" / "shards").exists())

    def test_a_requeued_job_resumes_from_its_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = _evaluate(self.scenarios, root / "reference", parallel=_parallel(1))
            reference_reports = _reports(root / "reference")
            # Replay a preempted job: the student's episodes are checkpointed
            # except the last one; the candidate and expert start from scratch.
            planner = ClosedLoopRolloutEvaluator(
                env_factory=research_module._never_called,
                policy_factory=research_module._never_called,
                max_steps=research_module.RESEARCH_EPISODE_BUDGET,
                seed=4,
                **research_module.PAIRED_EVALUATION_CONTRACT,
            )
            plan = planner.plan_episodes({"standard_success": list(self.scenarios)})
            episodes = {
                episode["episode_key"]: episode
                for episode in reference_reports["bc0_eval.json"]["suite_metrics"]["episodes"]
            }
            resumed = root / "resumed"
            self._seed_checkpoints(resumed, plan, episodes)
            counter = root / "environments.txt"
            os.environ[ENVIRONMENT_COUNTER] = str(counter)
            try:
                comparison = _evaluate(self.scenarios, resumed, parallel=_parallel(1))
            finally:
                os.environ.pop(ENVIRONMENT_COUNTER, None)
            self.assertEqual(comparison, reference)
            self.assertEqual(_reports(resumed), reference_reports)
            # One build for each worker's policy oracle plus one per episode:
            # the resumed student rolled out only its missing episode.
            builds = Counter(counter.read_text(encoding="utf-8").split())
            episodes_total = len(plan.episodes)
            self.assertEqual(
                sorted(builds.values()), [2, 1 + episodes_total, 1 + episodes_total]
            )

    def _seed_checkpoints(self, output_dir, plan, episodes):
        label_dir = output_dir / "evaluation" / "shards" / "bc0"
        identity = {
            "label": "bc0",
            "adapter": "student",
            "adapter_files": research_module._adapter_identity(Path("student")),
            "policy_kwargs": {
                "base_model": "gemma",
                "base_revision": "f" * 40,
                "load_in_4bit": True,
                "local_files_only": True,
                "trust_remote_code": False,
                "prompt_profile": None,
                "architecture": None,
            },
            "environment": dict(_parallel(1).environment),
            "seed": 4,
            "max_steps": research_module.RESEARCH_EPISODE_BUDGET,
            "episode_keys": [item.episode_key for item in plan.episodes],
        }
        identity = json.loads(json.dumps(identity, sort_keys=True, default=str))
        label_dir.mkdir(parents=True)
        (label_dir / "plan.json").write_text(json.dumps(identity), encoding="utf-8")
        for item in list(plan.episodes)[:-1]:
            research_module._atomic_bytes(
                research_module._episode_checkpoint(label_dir, item.ordinal),
                pickle.dumps(EpisodeEvaluation(**episodes[item.episode_key])),
            )


if __name__ == "__main__":
    unittest.main()
