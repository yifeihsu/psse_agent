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
        (3630, 1811, 7242, 7243),
    ],
    ids=("arm-a", "arm-b", "arm-c", "arm-d", "arm-e-model-only"),
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


def test_arm_e_is_a_model_only_match_to_arm_c() -> None:
    assert cell.ARMS["E"] == {
        "lane": "12b",
        "inclusion": "full_occupancy",
        "updates": 1811,
        "matched_corpus_arm": "C",
    }
    assert cell.ARMS["C"]["lane"] == "e2b"
    assert cell.ARMS["E"]["inclusion"] == cell.ARMS["C"]["inclusion"]
    assert cell.ARMS["E"]["updates"] == cell.ARMS["C"]["updates"]


def test_matched_corpus_binding_locks_paths_hashes_inclusion_and_updates() -> None:
    source = {
        "train": "/immutable/train.jsonl",
        "train_sha256": cell.MATCHED_CORPUS_BASELINES["C"]["train_sha256"],
        "validation": "/immutable/validation.jsonl",
        "validation_sha256": cell.MATCHED_CORPUS_BASELINES["C"][
            "validation_sha256"
        ],
        "inclusion": "full_occupancy",
        "updates": 1811,
        "lane": "e2b",
    }
    matched = {**source, "lane": "12b", "matched_corpus_arm": "C"}
    arms = {"C": source, "E": matched}
    assert cell.matched_corpus_binding(arms, "E")

    for key in cell.MATCHED_CORPUS_FIELDS:
        drifted = {"C": source, "E": {**matched, key: "drift"}}
        assert not cell.matched_corpus_binding(drifted, "E")

    assert not cell.matched_corpus_binding(
        {"E": {**matched, "matched_corpus_arm": "missing"}}, "E"
    )
    assert not cell.matched_corpus_binding(
        {"C": source, "E": {key: value for key, value in matched.items() if key != "matched_corpus_arm"}},
        "E",
    )
    assert not cell.matched_corpus_binding(
        {"C": source, "D": source, "E": {**matched, "matched_corpus_arm": "D"}},
        "E",
    )


def test_selected_arms_are_explicit_unique_and_closed_over_known_arms() -> None:
    assert cell.parse_selected_arms("E") == ["E"]
    assert cell.parse_selected_arms("A,C,E") == ["A", "C", "E"]
    for invalid in ("", "E,E", "F", "E, A"):
        with pytest.raises(ValueError, match="selected_arms"):
            cell.parse_selected_arms(invalid)


def test_submission_finish_requires_exact_selected_job_graph() -> None:
    payload = {
        "selected_arms": ["E"],
        "jobs": {"build": "1", "arm:E": "2", "audit:E": "3"},
    }
    cell.validate_submission_jobs(payload)

    for jobs in (
        {"build": "1", "arm:E": "2"},
        {**payload["jobs"], "arm:A": "4"},
        {"build": "1", "arm:E": "1", "audit:E": "3"},
        {"build": "not-a-job", "arm:E": "2", "audit:E": "3"},
    ):
        with pytest.raises(ValueError, match="submission job"):
            cell.validate_submission_jobs({"selected_arms": ["E"], "jobs": jobs})
