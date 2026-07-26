from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "submit_dagger_sft_round0.sh"
SETUP = REPO_ROOT / "setup_unsloth_env.sh"
SFT_REQUIREMENTS = REPO_ROOT / "psse_env" / "requirements-sft.txt"
WANDB_REQUIREMENTS = REPO_ROOT / "psse_env" / "requirements-wandb.txt"
README = REPO_ROOT / "psse_env" / "README.md"


def _requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_wandb_dependency_is_exactly_pinned_and_optional() -> None:
    assert _requirements(WANDB_REQUIREMENTS) == ["wandb==0.27.0"]
    assert all(
        not requirement.lower().startswith("wandb")
        for requirement in _requirements(SFT_REQUIREMENTS)
    )


def test_setup_installs_wandb_only_when_explicitly_requested() -> None:
    setup = SETUP.read_text(encoding="utf-8")
    assert "INSTALL_WANDB=${INSTALL_WANDB:-0}" in setup
    assert 'if [[ "$INSTALL_WANDB" == "1" ]]; then' in setup
    assert 'requirements-wandb.txt"' in setup
    assert '--constraint "$REPO_ROOT/psse_env/requirements-sft.txt"' in setup
    assert "verified optional W&B monitoring dependency" in setup


def test_launcher_activates_wandb_only_for_round0() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "ENABLE_WANDB=${ENABLE_WANDB:-0}" in launcher
    assert "WANDB_ACTIVE=0" in launcher
    assert (
        'if [[ "$ENABLE_WANDB" == "1" && "$STAGE" == "round0" ]]; then'
        in launcher
    )
    assert (
        "inactive for STAGE=$STAGE (monitoring starts only at round0)"
        in launcher
    )
    assert re.search(
        r'if \[\[ "\$WANDB_ACTIVE" == "1" \]\]; then\s+'
        r'COMMON_ARGS\+=\(--report-to wandb --run-name "\$WANDB_NAME"\)',
        launcher,
    )


def test_launcher_sets_bounded_nonsecret_wandb_contract() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for contract in (
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_RUN_GROUP",
        "WANDB_TAGS",
        "WANDB_JOB_TYPE",
        "WANDB_RESUME=allow",
        "WANDB_LOG_MODEL=false",
        "WANDB_WATCH=false",
        "WANDB_MODE",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_DATA_DIR",
        "WANDB_CONFIG_DIR",
        "WANDB_ARTIFACT_DIR",
    ):
        assert contract in launcher
    assert (
        "WANDB_RUN_ID_DEFAULT=bc0-r0-$WANDB_SOURCE_SHORT-$WANDB_SLURM_JOB_ID"
        in launcher
    )
    assert "WANDB_RUN_ID=${WANDB_RUN_ID:-$WANDB_RUN_ID_DEFAULT}" in launcher
    assert "WANDB_NAME=${WANDB_NAME:-$WANDB_RUN_ID}" in launcher
    assert "WANDB_SOURCE_SHORT=${REVIEWED_SOURCE_COMMIT:0:12}" in launcher
    assert "WANDB_SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}" in launcher
    assert "mkdir -p \\" in launcher
    assert "requirements-wandb.txt" in launcher
    assert "import wandb" in launcher
    assert "WANDB_API_KEY" not in launcher


def test_documentation_covers_login_submission_and_offline_sync() -> None:
    readme = README.read_text(encoding="utf-8")
    for contract in (
        "INSTALL_WANDB=1 bash setup_unsloth_env.sh",
        "wandb login",
        "ENABLE_WANDB=1",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_RUN_GROUP",
        "WANDB_TAGS",
        "WANDB_JOB_TYPE",
        "WANDB_MODE=offline",
        "wandb sync",
    ):
        assert contract in readme


def test_modified_shell_scripts_parse() -> None:
    for script in (LAUNCHER, SETUP):
        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
