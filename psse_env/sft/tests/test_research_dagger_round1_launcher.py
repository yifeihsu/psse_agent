from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "submit_research_gemma4_dagger_round1.sh"


def _launcher() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_round1_launcher_is_authorized_duplicate_guarded_and_nonspawning() -> None:
    launcher = _launcher()

    for expected in (
        "RESEARCH_DAGGER_R1_FALLBACK_AUTHORIZED",
        '[[ "$RESEARCH_DAGGER_R1_FALLBACK_AUTHORIZED" != "YES" ]]',
        "squeue -h",
        "16347744,16347745",
        'exec 8>"$GLOBAL_LOCK"',
        "flock -n 8",
        'exec 9>"$OUTPUT_ROOT/.job.lock"',
        "flock -n 9",
    ):
        assert expected in launcher

    # The queue matcher must recognize both descriptive Gemma names and the
    # shorter d12b/12b spellings used by the independently launched pipeline.
    assert "gemma.?4" in launcher
    assert "d12b" in launcher
    assert re.search(r"12b.*dagger|dagger.*12b|12b.*round|round.*12b", launcher)

    executable = "\n".join(
        line for line in launcher.splitlines() if not line.lstrip().startswith("#")
    )
    assert re.search(r"^\s*(?:sbatch|scontrol)\b", executable, re.MULTILINE) is None


def test_round1_launcher_freezes_inputs_collection_and_training_recipe() -> None:
    launcher = _launcher()

    for expected in (
        "cb10b81d184409bde395eb6686cb5738ad25cfb2378039761f5e679399f44f2a",
        "/scratch/yx3882/research_dagger_trace_20260823/trace_validation.jsonl",
        "/scratch/yx3882/research_gemma4_small_20260824_fe94580/evaluation/"
        "bc0_12b_replay_compare_v2/published_replay/d1_development_suite.json",
        "PROTECTED_D1_SHA256=",
        "COLLECTION_MAX_STEPS=4",
        "D1_CAP=20",
        "D1_SHARE=0.5",
        "CANDIDATE_MULTIPLIER=6",
        "TRAIN_PLAN='{\"measurement+parameter\":2,\"multi_measurement\":2,"
        "\"parameter\":1}'",
        "TRAIN_MAX_STEPS=32",
        "SAVE_EVAL_STEPS=8",
        "--learning-rate 3e-5",
        "--select-best-eval-loss",
        "research_checkpoint",
        '--expected-base-model "$MODEL_ID"',
    ):
        assert expected in launcher

    assert launcher.count("--protected-suite") == 2
    assert '--protected-suite "$VALIDATION"' in launcher
    assert '--protected-suite "$PROTECTED_D1"' in launcher
    assert 'd0_selected != d1_selected' in launcher
    assert 'mixture_report.get("actual_d1_share") != 0.5' in launcher
    assert 'metrics.get("episodes_completed") != 5' in launcher
    assert 'for step in (8, 16, 24, 32)' in launcher
    assert '"max_steps": 32' in launcher
    assert '"save_steps": 8' in launcher
    assert '"eval_steps": 8' in launcher
    assert '"learning_rate": 3e-5' in launcher
    assert 'physical-root overlap between {left} and {right}' in launcher
    assert '"protected_d1_roots": sorted(protected_d1_roots)' in launcher


def test_round1_launcher_gates_paired_generated_evaluation_and_attests_gpu() -> None:
    launcher = _launcher()

    for expected in (
        "#SBATCH --constraint=rtx6000",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --requeue",
        "release_hardware --require-class rtx6000",
        "nvidia-smi",
        "nvidia_smi_${SLURM_JOB_ID:-manual}.csv",
        "preflight_one baseline",
        "preflight_one candidate",
        "psse_env.dagger.research_action_preflight compare",
        "if (( PREFLIGHT_STATUS == 2 ))",
        "RESEARCH_GEMMA4_DAGGER_ROUND1_GATED_STOP",
        "DEVELOPMENT_PLAN='{\"measurement+parameter\":6,"
        "\"multi_measurement\":6,\"parameter\":3}'",
        "EVAL_MAX_STEPS=24",
        '--eval-r1-adapter "$CANDIDATE"',
        'len(roots) != 15 or len(set(roots)) != 15',
        "stage_status.json",
        "stage_receipt.json",
    ):
        assert expected in launcher

    gate_stop = launcher.index("if (( PREFLIGHT_STATUS == 2 ))")
    gated_exit = launcher.index("exit 0", gate_stop)
    paired_evaluation = launcher.index('--eval-r1-adapter "$CANDIDATE"')
    assert gate_stop < gated_exit < paired_evaluation
