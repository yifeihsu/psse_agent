"""Episode horizons agree across runtime entry points and active launchers."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from psse_env.episode_budget import DEFAULT_EPISODE_ACTION_LIMIT
from research.evaluate import run_episode

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    "submit_research_gemma4_dagger_round1.sh", "submit_research_dagger_trace_update.sh",
    "submit_research_dagger_update.sh", "submit_research_dagger_repair.sh",
    "submit_research_dagger_demo.sh", "submit_eval_v3.sh",
    "submit_research_gemma4_bc0_eval.sh", "submit_research_gemma4_bc0_replay_compare.sh",
    "submit_research_gemma4_smoke.sh",
    "submit_dagger_release_eval.sh",
    "research/hpc/occupancy_cell_20260827/run_arm.sh", "research/hpc/occupancy_cell_20260827/audit.sh",
    "research/hpc/exposure_curve_20260828/run_arm.sh", "research/hpc/exposure_curve_20260828/audit.sh",
)


@pytest.mark.parametrize("module_name", ["research.collect", "research.evaluate", "research.run_dagger", "research.physical_outcome_audit"])
def test_research_cli_episode_default_is_forty_without_changing_training_default(monkeypatch, module_name):
    captured = []

    class ParserCaptured(Exception):
        pass

    def capture(parser, *_args, **_kwargs):
        captured.append(parser)
        raise ParserCaptured()

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    module = importlib.import_module(module_name)
    with pytest.raises(ParserCaptured):
        module.main([])
    assert captured[0].get_default("max_steps") == DEFAULT_EPISODE_ACTION_LIMIT == 40
    if module_name == "research.run_dagger":
        assert captured[0].get_default("train_max_steps") == -1


@pytest.mark.parametrize("module_name", [
    "scripts.run_dagger_research", "psse_env.sft.research_bc0_eval", "psse_env.sft.research_bc0_checkpoint_compare",
])
def test_public_parser_factories_share_the_episode_default(module_name):
    parser = importlib.import_module(module_name).parser()
    assert parser.get_default("max_steps") == DEFAULT_EPISODE_ACTION_LIMIT
    if module_name == "scripts.run_dagger_research":
        assert parser.get_default("eval_max_steps") == DEFAULT_EPISODE_ACTION_LIMIT


@pytest.mark.parametrize("limit", [3, DEFAULT_EPISODE_ACTION_LIMIT])
def test_research_runner_binds_visible_budget_before_reset_and_respects_explicit_smoke_override(limit):
    class Env:
        max_steps = 65
        terminal = False

        def reset(self, _scenario):
            self.reset_budget = self.max_steps
            self.calls = 0

        def get_policy_observation(self, _history):
            return {"remaining_budget": self.max_steps - self.calls}

        def step(self, _action):
            self.calls += 1
            return {}, {"execution_status": "success", "tool_metrics": {}}

    class Policy:
        seen = []

        def act(self, observation):
            self.seen.append(observation["remaining_budget"])
            return {"tool": "run_wls", "arguments": {}}

    env, policy = Env(), Policy()
    report = run_episode(env, policy, {"scenario_id": "budget_probe"}, max_steps=limit)
    assert env.reset_budget == limit
    assert env.calls == report["steps"] == limit
    assert policy.seen == list(range(limit, 0, -1))
    assert report["termination_reason"] == "step_horizon"


def _bash():
    bundled = Path("C:/Program Files/Git/bin/bash.exe")
    candidate = str(bundled) if bundled.exists() else shutil.which("bash")
    if candidate is None:
        pytest.skip("Bash is unavailable")
    return candidate


@pytest.mark.parametrize("relative", LAUNCHERS)
def test_active_shell_launchers_default_to_forty_and_remain_valid_bash(relative):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    assert "EPISODE_MAX_STEPS=${EPISODE_MAX_STEPS:-40}" in text
    assert "--max-steps 24" not in text
    subprocess.run([_bash(), "-n", path.as_posix()], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("relative,fields", [
    ("research/hpc/full_pipeline_20260907/pipeline.env", ["D0_MAX_STEPS", "COLLECTION_MAX_STEPS", "EVAL_MAX_STEPS"]),
    ("research/hpc/diagnostic_round_20260903/round.env", ["COLLECTION_MAX_STEPS", "EVAL_MAX_STEPS"]),
])
@pytest.mark.parametrize("override", [None, "7"])
def test_scheduled_stages_resolve_one_shared_episode_budget(relative, fields, override):
    env = dict(os.environ)
    env.pop("EPISODE_MAX_STEPS", None)
    if override is not None:
        env["EPISODE_MAX_STEPS"] = override
    command = 'source "$1"\nprintf "%s\\n" ' + " ".join(f'"${name}"' for name in fields)
    result = subprocess.run([_bash(), "-c", command, "episode_budget_probe", (ROOT / relative).as_posix()],
                            check=True, capture_output=True, text=True, env=env)
    assert result.stdout.splitlines() == [override or "40"] * len(fields)


def test_training_update_limits_are_not_replaced_by_episode_budget():
    round1 = (ROOT / "submit_research_gemma4_dagger_round1.sh").read_text()
    assert "TRAIN_MAX_STEPS=32" in round1
    assert "SAVE_EVAL_STEPS=8" in round1
    assert '--max-steps "$TRAIN_MAX_STEPS"' in round1
    occupancy = (ROOT / "research/hpc/occupancy_cell_20260827/run_arm.sh").read_text()
    assert '--max-steps "$UPDATES"' in occupancy
    exposure = (ROOT / "research/hpc/exposure_curve_20260828/run_arm.sh").read_text()
    assert '--max-steps "$MAX_STEPS"' in exposure


@pytest.mark.parametrize("relative,root_name,override_name,fields", [
    ("research/hpc/full_pipeline_20260907/pipeline.env", "PIPE", "pipeline.overrides.env",
     ["D0_MAX_STEPS", "COLLECTION_MAX_STEPS", "EVAL_MAX_STEPS"]),
    ("research/hpc/diagnostic_round_20260903/round.env", "ROUND", "round.overrides.env",
     ["COLLECTION_MAX_STEPS", "EVAL_MAX_STEPS"]),
])
def test_old_independent_stage_overrides_cannot_desynchronize_horizons(tmp_path, relative, root_name, override_name, fields):
    text = (ROOT / relative).read_text()
    assignment = next(line for line in text.splitlines() if line.startswith(f"{root_name}="))
    staged = tmp_path / "settings.env"
    staged.write_text(text.replace(assignment, f'{root_name}="{tmp_path.as_posix()}"', 1), newline="\n")
    (tmp_path / override_name).write_text("COLLECTION_MAX_STEPS=12\nEVAL_MAX_STEPS=24\nD0_MAX_STEPS=8\n")
    env = dict(os.environ)
    env.pop("EPISODE_MAX_STEPS", None)
    command = 'source "$1"\nprintf "%s\\n" ' + " ".join(f'"${name}"' for name in fields)
    result = subprocess.run([_bash(), "-c", command, "episode_budget_probe", staged.as_posix()],
                            check=True, capture_output=True, text=True, env=env)
    assert result.stdout.splitlines() == ["40"] * len(fields)


@pytest.mark.parametrize("name", ["submit_dagger_release_eval.sh", "submit_dagger_sft_round0.sh"])
def test_active_release_launchers_select_current_study_template(name):
    text = (ROOT / name).read_text()
    assert "STUDY_MANIFEST=${STUDY_MANIFEST:-psse_env/dagger/studies/dagger_multiseed_study_v2.json}" in text


def test_research_smoke_uses_forty_episode_actions_and_preserves_optimizer_smoke_size():
    from psse_env.sft.research_smoke import parser

    assert parser().get_default("closed_loop_max_steps") == DEFAULT_EPISODE_ACTION_LIMIT
    assert parser().get_default("overfit_steps") == 20
    launcher = (ROOT / "submit_research_gemma4_smoke.sh").read_text()
    assert '--closed-loop-max-steps "$EPISODE_MAX_STEPS"' in launcher
    assert "--overfit-steps 20" in launcher


def test_round1_report_identity_and_gate_use_configured_episode_horizons():
    launcher = (ROOT / "submit_research_gemma4_dagger_round1.sh").read_text()
    assert '"max_steps": collection_max_steps' in launcher
    assert '"paired_development_evaluation": {"roots": 15, "max_steps": evaluation_max_steps}' in launcher
    assert 'comparison.get("max_steps") != int(sys.argv[4])' in launcher
    assert '"max_steps": 4,' not in launcher
    assert 'comparison.get("max_steps") != 24' not in launcher
    assert '"max_steps": 32' in launcher  # Optimizer updates remain separate.


@pytest.mark.parametrize("module_name", ["generate_baseline", "generate_sft_pilot"])
def test_example_generators_expose_default_budget_in_generated_observations(tmp_path, monkeypatch, module_name):
    import json

    module = importlib.import_module(f"psse_env.examples.{module_name}")
    if module_name == "generate_sft_pilot":
        # Exercise the actual generator's environment/collector wiring without
        # making this budget test depend on its separate synthetic teacher gate.
        captured = []

        class BudgetCaptured(Exception):
            pass

        def capture(collector, **kwargs):
            collector.env.reset(kwargs["scenarios"][0])
            captured.append((collector.env.max_steps, kwargs["max_steps"],
                             collector.env.get_policy_observation([]).remaining_budget))
            raise BudgetCaptured()

        monkeypatch.setattr(module.DaggerRolloutCollector, "collect_iteration", capture)
        with pytest.raises(BudgetCaptured):
            module.generate(tmp_path)
        assert captured == [(DEFAULT_EPISODE_ACTION_LIMIT,) * 3]
        return
    module.generate(tmp_path)
    filename = "sample_rollout.jsonl" if module_name == "generate_baseline" else "pilot.raw.jsonl"
    rows = [json.loads(line) for line in (tmp_path / filename).read_text().splitlines()]
    first_rows = [row for row in rows if row.get("step", 0) == 0]
    assert first_rows
    assert all(row["policy_observation"]["remaining_budget"] == DEFAULT_EPISODE_ACTION_LIMIT for row in first_rows)
