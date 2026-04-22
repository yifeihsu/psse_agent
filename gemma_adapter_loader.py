from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


TOKENIZER_ARTIFACT_FILENAMES = frozenset(
    {
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "added_tokens.json",
        "processor_config.json",
        "chat_template.jinja",
    }
)


def adapter_tokenizer_files(adapter_path: Path) -> list[str]:
    if not adapter_path.is_dir():
        return []
    return sorted(
        child.name for child in adapter_path.iterdir() if child.name in TOKENIZER_ARTIFACT_FILENAMES
    )


def adapter_has_saved_tokenizer(adapter_path: Path) -> bool:
    return (adapter_path / "tokenizer_config.json").exists()


def resolve_tokenizer_source(
    adapter_path: Path,
    *,
    base_model_name: str,
    prefer_base_tokenizer: bool,
) -> tuple[str, str, list[str]]:
    tokenizer_files = adapter_tokenizer_files(adapter_path)
    tokenizer_name = base_model_name
    source_label = "base model"
    if not prefer_base_tokenizer and adapter_has_saved_tokenizer(adapter_path):
        tokenizer_name = str(adapter_path)
        source_label = "adapter directory"
    return tokenizer_name, source_label, tokenizer_files


def prepare_unsloth_adapter_path(
    adapter_path: Path,
    *,
    prefer_base_tokenizer: bool,
) -> tuple[str, tempfile.TemporaryDirectory[str] | None, list[str]]:
    tokenizer_files = adapter_tokenizer_files(adapter_path)
    if not prefer_base_tokenizer or not tokenizer_files:
        return str(adapter_path), None, tokenizer_files

    tempdir = tempfile.TemporaryDirectory(prefix="unsloth_adapter_notokenizer_")
    temp_path = Path(tempdir.name)
    for child in adapter_path.iterdir():
        if child.name in TOKENIZER_ARTIFACT_FILENAMES:
            continue
        target = temp_path / child.name
        try:
            target.symlink_to(child.resolve(), target_is_directory=child.is_dir())
        except Exception:
            if child.is_dir():
                shutil.copytree(child, target, symlinks=True)
            else:
                shutil.copy2(child, target)
    return str(temp_path), tempdir, tokenizer_files


def format_unsloth_tokenizer_load_message(
    *,
    prefer_base_tokenizer: bool,
    tokenizer_files: list[str],
) -> str | None:
    if not tokenizer_files:
        return None
    if prefer_base_tokenizer:
        action = "masking adapter-local tokenizer artifacts to prefer the base tokenizer"
    else:
        action = "using adapter-local tokenizer artifacts"
    return "  Unsloth adapter load: " + action + ": " + ", ".join(tokenizer_files)
