from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_bc0_eval_launcher_is_phase_gated_offline_and_bounded() -> None:
    launcher = (ROOT / "submit_research_gemma4_bc0_eval.sh").read_text(
        encoding="utf-8"
    )
    for expected in (
        "#SBATCH --constraint=rtx6000",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --requeue",
        '#SBATCH --comment="preemption=yes;requeue=true"',
        '${RESEARCH_EVAL_PHASE:?set RESEARCH_EVAL_PHASE to d0 or d1}',
        "d0|d1)",
        'FULL_BC0="$RESEARCH_RUN_ROOT/bc0/full"',
        'POSTFLIGHT="$FULL_BC0/full_bc0_postflight.json"',
        'ADAPTER_DIR="$FULL_BC0/lora"',
        'OUTPUT_DIR="$RESEARCH_RUN_ROOT/evaluation/bc0_12b"',
        'exec 9>"$OUTPUT_DIR/.${RESEARCH_EVAL_PHASE}.lock"',
        "flock -n 9",
        'postflight.get("passed") is not True',
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export RESEARCH_MAX_INPUT_TOKENS=32768",
        "python -m psse_env.sft research-cache",
        "--model-choice 12b",
        "nvidia_smi_${RESEARCH_EVAL_PHASE}_${SLURM_JOB_ID}.csv",
        "python -m psse_env.sft research-bc0-eval",
        '--phase "$RESEARCH_EVAL_PHASE"',
        '--max-steps 24',
        'eval_args+=(--d0-report "$D0_REPORT")',
    ):
        assert expected in launcher
    assert "git diff" not in launcher
    assert "git status" not in launcher
    assert "sha256" not in launcher.lower()
    assert "sbatch" not in launcher
