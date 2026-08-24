from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_mini_bc0_launcher_is_bounded_and_resumable() -> None:
    launcher = (ROOT / "submit_research_gemma4_mini_bc0.sh").read_text(
        encoding="utf-8"
    )
    for expected in (
        "#SBATCH --constraint=rtx6000",
        '#SBATCH --comment="preemption=yes;requeue=true"',
        "mini.train128.jsonl",
        "mini.validation32.jsonl",
        "research_smoke.json",
        "--model-choice 12b",
        "--gradient-accumulation-steps 4",
        "--learning-rate 1e-4",
        "--epochs 1",
        "--save-steps 8",
        "--eval-steps 8",
        "python -m psse_env.sft.research_checkpoint",
        '--expected-base-model "google/gemma-4-12B-it"',
        '--resume-from-checkpoint "$checkpoint_path"',
        "expected_eval_steps = {8, 16, 24, 32}",
        "global_step == 32",
        'splits.get("train_rows") == 128',
        'splits.get("validation_rows") == 32',
        'model_selection.get("model_id") == "google/gemma-4-12B-it"',
        'model_selection.get("revision")',
        'int(adapter_delta.get("changed_tensors") or 0) > 0',
        'reload.get("fresh_base_reconstructed") is True',
        'reload.get("canaries_passed") == 1',
        "mini_bc0_postflight.json",
    ):
        assert expected in launcher
    assert "run_dagger" not in launcher
    assert "research-dagger" not in launcher
    assert "research_dagger" not in launcher
