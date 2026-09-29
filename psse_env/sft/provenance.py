"""Content hashes and git source state for provenance records."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_source_state(repo_root: str | Path) -> dict[str, Any]:
    """Return the exact source commit and whether local edits affected the gate."""
    root = Path(repo_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {
            "source_commit": None,
            "source_worktree_dirty": None,
            "tracked_diff_hash": None,
            "release_eligible_source": False,
        }
    untracked_source_suffixes = {
        ".py",
        ".pyi",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".json5",
        ".ini",
        ".cfg",
    }
    ignored_data_roots = ("data/", "artifacts/", "outputs/", "diagonostic/")
    untracked_source_files = []
    for line in untracked_status:
        if not line.startswith("?? "):
            continue
        relative = line[3:].strip()
        if relative.startswith(ignored_data_roots):
            continue
        if Path(relative).suffix.lower() in untracked_source_suffixes:
            untracked_source_files.append(relative)
    dirty = bool(status.strip()) or bool(untracked_source_files)
    return {
        "source_commit": commit or None,
        "source_worktree_dirty": dirty,
        "tracked_diff_hash": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_source_files": sorted(untracked_source_files),
        "release_eligible_source": bool(commit) and not dirty,
    }


__all__ = [
    "file_sha256",
    "git_source_state",
    "stable_json_sha256",
]
