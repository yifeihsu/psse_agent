"""Episode horizons agree across runtime entry points and active launchers."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from psse_env.episode_budget import DEFAULT_EPISODE_ACTION_LIMIT

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module_name", ["scripts.run_dagger_research"])
def test_public_parser_factories_share_the_episode_default(module_name):
    parser = importlib.import_module(module_name).parser()
    assert parser.get_default("max_steps") == DEFAULT_EPISODE_ACTION_LIMIT
    if module_name == "scripts.run_dagger_research":
        assert parser.get_default("eval_max_steps") == DEFAULT_EPISODE_ACTION_LIMIT


def _bash():
    bundled = Path("C:/Program Files/Git/bin/bash.exe")
    candidate = str(bundled) if bundled.exists() else shutil.which("bash")
    if candidate is None:
        pytest.skip("Bash is unavailable")
    return candidate


@pytest.mark.parametrize("relative,fields", [
    ("research/hpc/full_pipeline_20260907/pipeline.env", ["D0_MAX_STEPS", "COLLECTION_MAX_STEPS", "EVAL_MAX_STEPS"]),
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


@pytest.mark.parametrize("relative,root_name,override_name,fields", [
    ("research/hpc/full_pipeline_20260907/pipeline.env", "PIPE", "pipeline.overrides.env",
     ["D0_MAX_STEPS", "COLLECTION_MAX_STEPS", "EVAL_MAX_STEPS"]),
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
