"""Wiring check of the rollout and update phases against the real research environment.

Rolls the observable expert out (``policy_source: expert``) on two development
roots of the local classifier suite under the classifier-gated profile with
the tracked triage model, through the driver's own phases.  Skips when the
local suite is absent (it is gitignored output; see the memory note on
local audit-artifact tests).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = REPO_ROOT / "output" / "classifier_triage_20261004" / "suites" / "development.json"
TRIAGE = REPO_ROOT / "psse_env" / "oracle" / "models" / "triage_gnn_ieee14_20261004"


@unittest.skipUnless(SUITE.is_file() and TRIAGE.is_dir(), "local development suite or triage model missing")
class ExpertRolloutIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        # run_phase points the research runner's module-level options at this
        # cell (classifier profile, triage model, expert variant); restore them
        # so later tests build their environments from the defaults.
        import scripts.run_dagger_research as research

        self.options = [
            (options, dict(options))
            for options in (research.RESEARCH_ENVIRONMENT_OPTIONS, research.RESEARCH_EXPERT_OPTIONS)
        ]

    def tearDown(self) -> None:
        for options, original in self.options:
            options.clear()
            options.update(original)

    def test_rollout_and_update_phases_run_on_real_roots_with_the_expert(self) -> None:
        from psse_env.rl.train import Manifest, RLConfig, run_phase

        rows = json.loads(SUITE.read_text(encoding="utf-8"))
        chosen = [row for row in rows if row["grouping"]["scenario_family"] in {"no_error", "measurement"}][:2]
        self.assertEqual(len(chosen), 2)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            suite = base / "suite.json"
            suite.write_text(json.dumps(chosen), encoding="utf-8")
            adapter = base / "adapter0"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            config = RLConfig(
                output_dir=str(base / "run"), training_suite=str(suite), development_suite=None,
                initial_adapter=str(adapter), base_model="google/gemma-4-12B-it", base_revision="0" * 40,
                evidence_profile="classifier_gated_diagnostics", triage_classifier=str(TRIAGE),
                samples_per_root=2, max_steps=40, policy_source="expert", evaluate_every_iteration=False,
            )
            os.environ.setdefault("PSSE_LOCAL_DIAGNOSTIC_BUILD", "1")
            rollout = run_phase(config, "rollout")
            self.assertEqual(rollout["completed_groups"], 2)
            manifest = Manifest.load(base / "run" / "manifest.json", initial_adapter=str(adapter))
            self.assertEqual(manifest.phase, "update")
            groups_dir = base / "run" / "iteration_000" / "trajectories" / "groups"
            self.assertEqual(len(list(groups_dir.glob("*.jsonl"))), 2)
            first = json.loads(next(groups_dir.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(first["evaluation"]["truth_audited_task_success"], first["reward"])
            self.assertFalse(first["pairing_ok"])
            self.assertGreater(first["reward"]["credit"], 0.0)
            update = run_phase(config, "update")
            self.assertIsNone(update["update"])
            self.assertTrue((base / "run" / "iteration_000" / "adapter" / "adapter_config.json").is_file())
            evaluate = run_phase(config, "evaluate")
            self.assertIsNone(evaluate)
            manifest = Manifest.load(base / "run" / "manifest.json", initial_adapter=str(adapter))
            self.assertEqual((manifest.iteration, manifest.phase), (1, "rollout"))
            self.assertTrue(manifest.current_adapter.endswith("adapter"))


if __name__ == "__main__":
    unittest.main()
