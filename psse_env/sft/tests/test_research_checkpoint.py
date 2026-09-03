from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from psse_env.sft import research_checkpoint


BASE_MODEL = "google/gemma-4-12B-it"


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {"peft_type": "LORA", "base_model_name_or_path": BASE_MODEL}
        ),
        encoding="utf-8",
    )
    for name in (
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / name).write_bytes(b"complete-marker")
    return checkpoint


def test_selects_newest_complete_checkpoint(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 8)
    newest = _checkpoint(tmp_path, 16)
    with patch.object(research_checkpoint, "_validate_adapter_weights"), patch.object(
        research_checkpoint, "_load_torch_mapping", return_value={"state": 1}
    ):
        selected = research_checkpoint.select_newest_complete_checkpoint(
            tmp_path, expected_base_model=BASE_MODEL
        )
    assert selected == newest.resolve()


def test_corrupt_newest_checkpoint_falls_back_to_complete_older(
    tmp_path: Path,
) -> None:
    older = _checkpoint(tmp_path, 8)
    newest = _checkpoint(tmp_path, 16)
    (newest / "adapter_config.json").write_text("{truncated", encoding="utf-8")
    with patch.object(research_checkpoint, "_validate_adapter_weights"), patch.object(
        research_checkpoint, "_load_torch_mapping", return_value={"state": 1}
    ):
        selected = research_checkpoint.select_newest_complete_checkpoint(
            tmp_path, expected_base_model=BASE_MODEL
        )
    assert selected == older.resolve()


def test_corrupt_serialized_state_falls_back_to_complete_older(
    tmp_path: Path,
) -> None:
    older = _checkpoint(tmp_path, 8)
    _checkpoint(tmp_path, 16)

    def load_state(path: Path):
        if path.parent.name == "checkpoint-16" and path.name == "optimizer.pt":
            raise RuntimeError("truncated optimizer")
        return {"state": 1}

    with patch.object(research_checkpoint, "_validate_adapter_weights"), patch.object(
        research_checkpoint, "_load_torch_mapping", side_effect=load_state
    ):
        selected = research_checkpoint.select_newest_complete_checkpoint(
            tmp_path, expected_base_model=BASE_MODEL
        )
    assert selected == older.resolve()


def test_real_serialized_artifacts_reject_truncated_newest(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    def real_checkpoint(step: int) -> Path:
        checkpoint = _checkpoint(tmp_path, step)
        safetensors_torch.save_file(
            {"lora_A": torch.ones(2, dtype=torch.float32)},
            str(checkpoint / "adapter_model.safetensors"),
        )
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            torch.save({"state": torch.tensor([step])}, checkpoint / name)
        return checkpoint

    older = real_checkpoint(8)
    newest = real_checkpoint(16)
    (newest / "optimizer.pt").write_bytes(b"truncated")
    selected = research_checkpoint.select_newest_complete_checkpoint(
        tmp_path, expected_base_model=BASE_MODEL
    )
    assert selected == older.resolve()


def test_checkpoint_candidates_fail_closed_when_none_are_complete(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, 8)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        research_checkpoint.IncompleteCheckpointError,
        match="checkpoint directories exist but none is complete",
    ):
        research_checkpoint.select_newest_complete_checkpoint(
            tmp_path, expected_base_model=BASE_MODEL
        )


def test_no_checkpoint_is_a_clean_start(tmp_path: Path) -> None:
    assert (
        research_checkpoint.select_newest_complete_checkpoint(
            tmp_path, expected_base_model=BASE_MODEL
        )
        is None
    )
