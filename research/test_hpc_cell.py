"""CPU-only tests for the immutable occupancy-cell cache gates."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from unittest import mock

import pytest

from research.hpc.occupancy_cell_20260827 import build as cell


def _snapshot(cache: Path, lane: str = "e2b") -> Path:
    model_id, revision = cell.MODELS[lane]
    repository = "models--" + model_id.replace("/", "--")
    snapshot = cache / "hub" / repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"model weights")
    return snapshot


def test_snapshot_manifest_hashes_exposed_bytes(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    first = cell.snapshot_manifest(tmp_path, "e2b")
    assert first["files"] == 2
    assert first["bytes"] == sum(path.stat().st_size for path in snapshot.iterdir())

    (snapshot / "model.safetensors").write_bytes(b"modified weights")
    second = cell.snapshot_manifest(tmp_path, "e2b")
    assert second["sha256"] != first["sha256"]


def test_snapshot_manifest_rejects_broken_entry(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    link = snapshot / "tokenizer.json"
    try:
        os.symlink(snapshot / "missing-blob", link)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(ValueError, match="broken pinned snapshot entry"):
        cell.snapshot_manifest(tmp_path, "e2b")


def test_environment_receipt_requires_configured_hf_home(tmp_path: Path) -> None:
    expected = {"python": platform.python_version()}
    config = {"hf_home": str(tmp_path)}
    with (
        mock.patch.dict(cell.EXPECTED_ENVIRONMENT, expected, clear=True),
        mock.patch.dict(os.environ, {"HF_HOME": str(tmp_path / "wrong")}),
        pytest.raises(ValueError, match="HF_HOME must be"),
    ):
        cell.environment_receipt(config)

    with (
        mock.patch.dict(cell.EXPECTED_ENVIRONMENT, expected, clear=True),
        mock.patch.dict(os.environ, {"HF_HOME": str(tmp_path)}),
    ):
        receipt = cell.environment_receipt(config)
    assert receipt["hf_home"] == str(tmp_path.resolve())


@pytest.mark.parametrize(
    ("train_rows", "updates", "trained", "collated"),
    [
        (2566, 666, 2662, 2663),
        (2533, 662, 2645, 2646),
        (3630, 1811, 7242, 7243),
        (1332, 247, 988, 989),
    ],
    ids=("arm-a", "arm-b", "arm-c", "arm-d"),
)
def test_expected_train_exposure_locks_occupancy_arms(
    train_rows: int,
    updates: int,
    trained: int,
    collated: int,
) -> None:
    exposure = cell.expected_train_exposure(train_rows, updates, 1, 4)

    assert exposure["training_step_rows"] == trained
    assert exposure["training_step_batches"] == trained
    assert exposure["collated_rows"] == collated
    assert exposure["collated_batches"] == collated


@pytest.mark.parametrize(
    ("train_rows", "updates", "trained", "collated"),
    [
        (10, 2, 8, 9),
        (10, 3, 10, 10),
        (10, 4, 14, 15),
        (8, 2, 8, 8),
    ],
)
def test_expected_train_exposure_distinguishes_mid_epoch_lookahead(
    train_rows: int,
    updates: int,
    trained: int,
    collated: int,
) -> None:
    exposure = cell.expected_train_exposure(train_rows, updates, 1, 4)

    assert exposure["training_step_rows"] == trained
    assert exposure["collated_rows"] == collated


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
        (True, 1, 1, 1),
    ],
)
def test_expected_train_exposure_rejects_invalid_inputs(
    values: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        cell.expected_train_exposure(*values)
