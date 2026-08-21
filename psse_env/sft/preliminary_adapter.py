"""Prepare a warm-start adapter whose base points to the exact E2B snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PINNED_MODEL_NAME = "unsloth/gemma-4-E2B-it"
PINNED_MODEL_REVISION = "f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae"
PINNED_INIT_CONTRACT = "preliminary_dagger_pinned_init_adapter_v1"


class PreliminaryAdapterError(ValueError):
    """The small-model warm-start snapshot or adapter failed closed."""


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise PreliminaryAdapterError(f"adapter is not a regular directory: {root}")
    entries = []
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise PreliminaryAdapterError(f"adapter contains a symlink: {child}")
        if child.is_file():
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "sha256": _file_hash(child),
                    "size": child.stat().st_size,
                }
            )
    if not entries:
        raise PreliminaryAdapterError("adapter tree is empty")
    return _stable_hash(entries)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreliminaryAdapterError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PreliminaryAdapterError(f"{label} must contain one JSON object")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _binding_path(destination: Path) -> Path:
    return destination.parent / f"{destination.name}.binding.json"


def _is_pinned_base_reference(value: Any) -> bool:
    if value == PINNED_MODEL_NAME:
        return True
    if not isinstance(value, str) or not value:
        return False
    snapshot = Path(value)
    return (
        snapshot.name.lower() == PINNED_MODEL_REVISION
        and snapshot.is_dir()
        and (snapshot / "config.json").is_file()
    )


def _validate_existing(
    *,
    source: Path,
    destination: Path,
    snapshot: Path,
) -> dict[str, Any]:
    binding_path = _binding_path(destination)
    if not destination.is_dir() or destination.is_symlink():
        raise PreliminaryAdapterError("pinned init adapter is missing or a symlink")
    if binding_path.is_symlink() or not binding_path.is_file():
        raise PreliminaryAdapterError("pinned init adapter lacks its binding receipt")
    config = _json_object(
        destination / "adapter_config.json", label="pinned adapter config"
    )
    if config.get("base_model_name_or_path") != str(snapshot):
        raise PreliminaryAdapterError(
            "pinned adapter base_model_name_or_path is not the exact snapshot"
        )
    expected = {
        "contract": PINNED_INIT_CONTRACT,
        "release_eligible": False,
        "model": PINNED_MODEL_NAME,
        "model_revision": PINNED_MODEL_REVISION,
        "snapshot_path": str(snapshot),
        "source_adapter_tree_sha256": _tree_hash(source),
        "prepared_adapter_tree_sha256": _tree_hash(destination),
    }
    recorded = _json_object(binding_path, label="pinned adapter binding")
    if recorded != expected:
        raise PreliminaryAdapterError(
            "pinned init adapter binding differs from current source or snapshot"
        )
    return {**expected, "adapter_path": str(destination)}


def prepare_pinned_initial_adapter(
    *,
    source_adapter: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    allow_download: bool = False,
) -> dict[str, Any]:
    """Copy BC0 and point its PEFT base reference at one exact HF snapshot."""

    source = Path(source_adapter).resolve(strict=True)
    target = Path(destination).resolve(strict=False)
    try:
        if target == source or target.is_relative_to(source) or source.is_relative_to(
            target
        ):
            raise PreliminaryAdapterError(
                "source and pinned init adapter directories must not overlap"
            )
    except AttributeError:  # pragma: no cover - Python >=3.12 in the SFT env
        pass
    source_config = _json_object(
        source / "adapter_config.json", label="source adapter config"
    )
    if not _is_pinned_base_reference(
        source_config.get("base_model_name_or_path")
    ):
        raise PreliminaryAdapterError(
            "BC0 adapter does not name the pinned E2B base model"
        )
    _tree_hash(source)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised in the HPC env
        raise PreliminaryAdapterError("huggingface_hub is required") from exc
    snapshot = Path(
        snapshot_download(
            repo_id=PINNED_MODEL_NAME,
            revision=PINNED_MODEL_REVISION,
            local_files_only=not allow_download,
        )
    ).resolve(strict=True)
    if snapshot.name.lower() != PINNED_MODEL_REVISION or not (
        snapshot / "config.json"
    ).is_file():
        raise PreliminaryAdapterError(
            "Hugging Face did not resolve the exact pinned E2B snapshot"
        )

    binding_path = _binding_path(target)
    if target.exists() or target.is_symlink() or binding_path.exists() or binding_path.is_symlink():
        return _validate_existing(
            source=source, destination=target, snapshot=snapshot
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    staged = staging_root / "adapter"
    try:
        shutil.copytree(source, staged, symlinks=False)
        config_path = staged / "adapter_config.json"
        config = _json_object(config_path, label="staged adapter config")
        config["base_model_name_or_path"] = str(snapshot)
        _write_json(config_path, config)
        os.rename(staged, target)
        expected = {
            "contract": PINNED_INIT_CONTRACT,
            "release_eligible": False,
            "model": PINNED_MODEL_NAME,
            "model_revision": PINNED_MODEL_REVISION,
            "snapshot_path": str(snapshot),
            "source_adapter_tree_sha256": _tree_hash(source),
            "prepared_adapter_tree_sha256": _tree_hash(target),
        }
        _write_json(binding_path, expected)
    except FileExistsError as exc:
        raise PreliminaryAdapterError(
            "pinned init adapter publication raced with an existing path"
        ) from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return _validate_existing(source=source, destination=target, snapshot=snapshot)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare exact-snapshot E2B BC0 warm-start adapter."
    )
    parser.add_argument("--source-adapter", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_pinned_initial_adapter(
        source_adapter=args.source_adapter,
        destination=args.destination,
        allow_download=args.allow_download,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
