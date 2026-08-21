"""Exact-snapshot binding for the preliminary E2B warm start."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
import sys

import pytest

from psse_env.sft.preliminary_adapter import (
    PINNED_MODEL_NAME,
    PINNED_MODEL_REVISION,
    PreliminaryAdapterError,
    prepare_pinned_initial_adapter,
)


def _install_fake_hub(monkeypatch: pytest.MonkeyPatch, snapshot: Path) -> None:
    module = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        assert kwargs == {
            "repo_id": PINNED_MODEL_NAME,
            "revision": PINNED_MODEL_REVISION,
            "local_files_only": True,
        }
        return str(snapshot)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def _source_adapter(root: Path) -> Path:
    source = root / "bc0" / "lora"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": PINNED_MODEL_NAME}),
        encoding="utf-8",
    )
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    return source


def test_prepares_and_revalidates_exact_snapshot_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "hub" / "snapshots" / PINNED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    _install_fake_hub(monkeypatch, snapshot)
    source = _source_adapter(tmp_path)
    destination = tmp_path / "pinned_init"
    first = prepare_pinned_initial_adapter(
        source_adapter=source,
        destination=destination,
    )
    second = prepare_pinned_initial_adapter(
        source_adapter=source,
        destination=destination,
    )
    assert first == second
    assert first["release_eligible"] is False
    config = json.loads(
        (destination / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert config["base_model_name_or_path"] == str(snapshot.resolve())


def test_binding_rejects_source_adapter_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshots" / PINNED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    _install_fake_hub(monkeypatch, snapshot)
    source = _source_adapter(tmp_path)
    destination = tmp_path / "pinned_init"
    prepare_pinned_initial_adapter(
        source_adapter=source,
        destination=destination,
    )
    (source / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(PreliminaryAdapterError, match="binding differs"):
        prepare_pinned_initial_adapter(
            source_adapter=source,
            destination=destination,
        )


def test_rejects_adapter_for_a_different_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshots" / PINNED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    _install_fake_hub(monkeypatch, snapshot)
    source = _source_adapter(tmp_path)
    config_path = source / "adapter_config.json"
    config_path.write_text(
        json.dumps({"base_model_name_or_path": "floating/model"}),
        encoding="utf-8",
    )
    with pytest.raises(PreliminaryAdapterError, match="pinned E2B"):
        prepare_pinned_initial_adapter(
            source_adapter=source,
            destination=tmp_path / "pinned_init",
        )
