from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_replay_compare_launcher_is_single_gpu_resumable_and_nontraining() -> None:
    launcher = (ROOT / "submit_research_gemma4_bc0_replay_compare.sh").read_text(
        encoding="utf-8"
    )
    for expected in (
        "#SBATCH --constraint=rtx6000",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --requeue",
        'CHECKPOINT_STEPS="${RESEARCH_CHECKPOINT_STEPS:-64 128 192 256 320}"',
        'exec 9>"$OUTPUT_ROOT/.replay_compare.lock"',
        "flock -n 9",
        "report_ready()",
        'report.get("contract") == RESEARCH_BC0_EVAL_CONTRACT',
        "python -m psse_env.sft research-bc0-eval",
        "--phase d0",
        "--phase d1",
        "python -m psse_env.sft.research_bc0_checkpoint_compare",
        '--suite-json "$REPLAY_DIR/d1_development_suite.json"',
        'checkpoint_args+=(--adapter "checkpoint-$step=$checkpoint")',
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export RESEARCH_MAX_INPUT_TOKENS=32768",
        "nvidia_smi_${SLURM_JOB_ID}.csv",
    ):
        assert expected in launcher
    assert "research-train" not in launcher
    assert "sbatch" not in launcher
    assert "rm -" not in launcher
