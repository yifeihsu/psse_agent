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
