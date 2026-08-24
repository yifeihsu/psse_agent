from __future__ import annotations

import json
import tempfile
from pathlib import Path

from psse_env.research_models import GEMMA4_12B
from psse_env.sft.research_bc0_postflight import build_full_bc0_postflight


ROOT = Path(__file__).resolve().parents[3]


def test_full_bc0_launcher_is_fresh_bounded_and_resumable() -> None:
    launcher = (ROOT / "submit_research_gemma4_full_bc0.sh").read_text(
        encoding="utf-8"
    )
    for expected in (
        "#SBATCH --constraint=rtx6000",
        '#SBATCH --comment="preemption=yes;requeue=true"',
        "aggregate.train_view.jsonl",
        "aggregate.validation.jsonl",
        "mini_bc0_postflight.json",
        "train={len(train)}, validation={len(validation)}",
        "research_gemma4_full_bc0_resume_identity_v1",
        "full-BC0 resume identity differs",
        "full-BC0 output has training artifacts but no run identity",
        "28b733db96c6ce05dbdc8d43484bdbb14445e1105958a78f9a35024aa5b3844a",
        "2ea1c79d8bc85faf40a8ea5edd2352688c141e658b98407fcfba04ba49a9e4ca",
        "--model-choice 12b",
        "--max-length 32768",
        "--gradient-accumulation-steps 4",
        "--learning-rate 1e-4",
        "--epochs 1",
        "--save-steps 64",
        "--eval-steps 64",
        "--select-best-eval-loss",
        "python -m psse_env.sft.research_checkpoint",
        '--expected-base-model "google/gemma-4-12B-it"',
        '--resume-from-checkpoint "$checkpoint_path"',
        "python -m psse_env.sft.research_bc0_postflight",
        "--expected-train-rows 1280",
        "--expected-validation-rows 304",
        "--expected-eval-step 320",
        "--minimum-global-step 320",
    ):
        assert expected in launcher
    assert "--initial-adapter" not in launcher
    assert "run_dagger" not in launcher
    assert "research-dagger" not in launcher
    assert "research_dagger" not in launcher


def test_full_bc0_postflight_selects_best_eval_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        lora = root / "lora"
        lora.mkdir()
        (lora / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "base_model_name_or_path": GEMMA4_12B.model_id,
                }
            ),
            encoding="utf-8",
        )
        (lora / "adapter_model.safetensors").write_bytes(b"adapter")
        for step in (64, 128, 192, 256, 320):
            checkpoint = root / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        stage = {
            "settings": {
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            },
            "training_metrics": {"train_loss": 0.2},
            "adapter_delta": {"changed_tensors": 12},
        }
        (root / "research_run.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "completion_errors": [],
                    "model_selection": {
                        "model_id": GEMMA4_12B.model_id,
                        "revision": GEMMA4_12B.revision,
                        "architecture": GEMMA4_12B.architecture,
                    },
                    "data": {
                        "splits": {
                            "train_rows": 1280,
                            "validation_rows": 304,
                            "train_roots": 182,
                            "validation_roots": 44,
                            "overlap": [],
                        }
                    },
                    "preserved_training_stage": stage,
                    "reload": {
                        "canary_mode": "parseable_single_tool_call_after_reload",
                        "fresh_base_reconstructed": True,
                        "adapter_reloaded": True,
                        "canaries_requested": 1,
                        "canaries_selected": 1,
                        "canaries_passed": 1,
                        "generation_canary_pass": True,
                        "canaries": [{"passed": True, "exact_action_match": False}],
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": 320,
                    "best_model_checkpoint": str(root / "checkpoint-192"),
                    "best_metric": 0.1,
                    "log_history": [
                        {"step": 64, "eval_loss": 0.4},
                        {"step": 128, "eval_loss": 0.2},
                        {"step": 192, "eval_loss": 0.1},
                        {"step": 256, "eval_loss": 0.12},
                        {"step": 320, "eval_loss": 0.11},
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = build_full_bc0_postflight(
            root,
            expected_train_rows=1280,
            expected_validation_rows=304,
            expected_eval_steps=(64, 128, 192, 256, 320),
            minimum_global_step=320,
            maximum_global_step=320,
        )
        mismatched_state = json.loads(
            (root / "trainer_state.json").read_text(encoding="utf-8")
        )
        mismatched_state["best_model_checkpoint"] = str(root / "checkpoint-128")
        (root / "trainer_state.json").write_text(
            json.dumps(mismatched_state), encoding="utf-8"
        )
        mismatched = build_full_bc0_postflight(
            root,
            expected_train_rows=1280,
            expected_validation_rows=304,
            expected_eval_steps=(64, 128, 192, 256, 320),
            minimum_global_step=320,
            maximum_global_step=320,
        )

    assert report["passed"] is True
    assert report["best_checkpoint_step"] == 192
    assert report["minimum_eval_loss"] == 0.1
    assert report["reload"]["canaries"][0]["exact_action_match"] is False
    assert mismatched["passed"] is False
    assert mismatched["checks"]["best_eval_checkpoint"] is False
