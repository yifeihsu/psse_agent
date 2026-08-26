"""Reproducibility metadata for live tokenizer and SFT gates."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


AGGREGATE_MANIFEST_FILENAME = "aggregate.manifest.json"
ROUND1_AGGREGATE_BUILDER_CONTRACT = (
    "deterministic_d0_d1_probe_balanced_union_v2"
)


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


def validate_aggregate_manifest_binding(
    provenance: Mapping[str, Any],
    *,
    aggregate_dir: str | Path,
) -> dict[str, Any]:
    """Verify the immutable private aggregate manifest against provenance.

    Round-0's ``aggregate.manifest.json`` contains the episode-level audit
    evidence used by later DAgger builders.  Treating only the JSONL bytes as
    immutable would leave that evidence replaceable after generation, so every
    release consumer must require the sibling manifest and its recorded hash.
    """

    manifest_path = Path(aggregate_dir) / AGGREGATE_MANIFEST_FILENAME
    failures: list[str] = []
    dataset_hashes = provenance.get("dataset_hashes")
    dataset_hashes = (
        dataset_hashes if isinstance(dataset_hashes, Mapping) else {}
    )
    expected_hash = dataset_hashes.get(AGGREGATE_MANIFEST_FILENAME)
    if not manifest_path.is_file():
        failures.append(f"{AGGREGATE_MANIFEST_FILENAME} is missing.")
        actual_hash = None
    else:
        actual_hash = file_sha256(manifest_path)
    if not (
        isinstance(expected_hash, str)
        and len(expected_hash) == 64
        and expected_hash == expected_hash.lower()
        and all(character in "0123456789abcdef" for character in expected_hash)
    ):
        failures.append(
            "Generation provenance lacks a valid aggregate.manifest.json hash."
        )
    elif actual_hash is not None and actual_hash != expected_hash:
        failures.append(
            "aggregate.manifest.json hash does not match generation provenance."
        )
    return {
        "passed": not failures,
        "failures": failures,
        "manifest_path": str(manifest_path),
        "recorded_sha256": expected_hash,
        "computed_sha256": actual_hash,
    }


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


def tool_schema_hashes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            stable_json_sha256(row.get("tools"))
            for row in rows
            if isinstance(row.get("tools"), list)
        }
    )


def build_gate_provenance(
    *,
    repo_root: str | Path,
    processor_revision: str,
    datasets: Mapping[str, str | Path],
    rows: Sequence[Mapping[str, Any]],
    exporter_files: Sequence[str | Path],
) -> dict[str, Any]:
    """Describe the source, schemas, exporter, processor, and exact datasets."""
    schema_hashes = tool_schema_hashes(rows)
    payload: dict[str, Any] = {
        **git_source_state(repo_root),
        "processor_revision": processor_revision,
        "schema_registry_hash": schema_hashes[0] if len(schema_hashes) == 1 else None,
        "schema_registry_hashes": schema_hashes,
        "exporter_hashes": {
            str(Path(path).resolve().relative_to(Path(repo_root).resolve())): file_sha256(path)
            for path in exporter_files
        },
        "dataset_hashes": {
            name: file_sha256(path) for name, path in sorted(datasets.items())
        },
    }
    return payload


def validate_release_gate_report(
    report_path: str | Path,
    *,
    model: str,
    revision: str,
    source_commit: str,
    datasets: Mapping[str, str | Path],
    max_length: int,
) -> dict[str, Any]:
    """Bind the round-0 prerequisite gate to exact release inputs.

    The standalone data gate intentionally records an ``AutoTokenizer``
    fallback for diagnostics.  Full BC0 training is stricter: it may proceed
    only from an eligible gate that used ``AutoProcessor`` on the same commit,
    model revision, maximum length, and split bytes.
    """

    failures: list[str] = []
    source = Path(report_path)
    payload: Mapping[str, Any] = {}
    try:
        decoded = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(
            f"Release gate report is unreadable: {type(exc).__name__}: {exc}"
        )
    else:
        if isinstance(decoded, Mapping):
            payload = decoded
        else:
            failures.append("Release gate report must contain a JSON object.")

    if not payload:
        failures.append("Release gate report has no gate payload.")
    else:
        if payload.get("passed") is not True:
            failures.append("Release gate report did not pass.")
        if payload.get("release_eligible") is not True:
            failures.append("Release gate report is not release eligible.")
        if payload.get("processor_loader") != "AutoProcessor":
            failures.append(
                "Release gate must use AutoProcessor; got "
                f"{payload.get('processor_loader')!r}."
            )
        if payload.get("processor_loader_passed") is not True:
            failures.append(
                "Release gate did not enforce its AutoProcessor requirement."
            )
        if payload.get("processor_loader_requirement") != "AutoProcessor":
            failures.append("Release gate did not declare AutoProcessor as required.")
        if payload.get("model") != model:
            failures.append("Release gate model identity does not match round0.")
        if payload.get("revision") != revision:
            failures.append("Release gate model revision does not match round0.")
        if payload.get("max_length") != max_length:
            failures.append("Release gate max_length does not match round0.")

        provenance = payload.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        if provenance.get("release_eligible_source") is not True:
            failures.append("Release gate source was not clean and release eligible.")
        if provenance.get("source_commit") != source_commit:
            failures.append("Release gate source commit does not match round0.")
        if provenance.get("processor_revision") != revision:
            failures.append("Release gate processor revision does not match round0.")

        generation = payload.get("generation_provenance")
        generation = generation if isinstance(generation, Mapping) else {}
        if generation.get("passed") is not True:
            failures.append("Release aggregate generation provenance did not pass.")
        if generation.get("release_eligible") is not True:
            failures.append("Release aggregate generation provenance is ineligible.")
        if generation.get("source_commit") != source_commit:
            failures.append(
                "Release aggregate generation commit does not match round0."
            )

        expected_splits = {"train", "validation", "test"}
        if set(datasets) != expected_splits:
            failures.append(
                "Release gate validation requires train, validation, and test splits."
            )
        recorded_hashes = provenance.get("dataset_hashes")
        recorded_hashes = (
            recorded_hashes if isinstance(recorded_hashes, Mapping) else {}
        )
        for split_name, dataset_path in sorted(datasets.items()):
            path = Path(dataset_path)
            if not path.is_file():
                failures.append(f"Release dataset is missing: {path}.")
                continue
            if recorded_hashes.get(split_name) != file_sha256(path):
                failures.append(
                    f"Release gate dataset hash does not match {split_name}."
                )
            split_report = payload.get(split_name)
            split_report = (
                split_report if isinstance(split_report, Mapping) else {}
            )
            if split_report.get("passed") is not True:
                failures.append(f"Release gate {split_name} audit did not pass.")
            length_audit = split_report.get("length_audit")
            length_audit = (
                length_audit if isinstance(length_audit, Mapping) else {}
            )
            if length_audit.get("prompt_truncated_rows") != 0:
                failures.append(
                    f"Release gate {split_name} contains prompt truncation."
                )
            if length_audit.get("target_truncated_rows") != 0:
                failures.append(
                    f"Release gate {split_name} contains target truncation."
                )

    return {
        "passed": not failures,
        "failures": failures,
        "report_path": str(source),
    }


def validate_generation_provenance(
    *,
    repo_root: str | Path,
    datasets: Mapping[str, str | Path],
    rows: Sequence[Mapping[str, Any]],
    dataset_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify split provenance against paths or caller-held immutable bytes.

    Checkpoint-producing training passes hashes from the same one-read snapshot
    used to build examples.  This prevents a transient path replacement from
    making provenance authenticate different bytes than the model consumes.
    """
    failures: list[str] = []
    authenticated_hashes: dict[str, str] = {}
    if dataset_sha256 is not None:
        if set(dataset_sha256) != set(datasets) or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in dataset_sha256.values()
        ):
            failures.append("Authenticated dataset snapshot hashes are malformed.")
        else:
            authenticated_hashes = dict(dataset_sha256)
    parents = {Path(path).resolve().parent for path in datasets.values()}
    if len(parents) != 1:
        failures.append("Dataset splits do not share one provenance directory.")
        provenance_path = None
    else:
        provenance_path = next(iter(parents)) / "aggregate.generation_provenance.json"
    payload: dict[str, Any] = {}
    if provenance_path is None or not provenance_path.is_file():
        failures.append("aggregate.generation_provenance.json is missing.")
    else:
        try:
            decoded = json.loads(provenance_path.read_text(encoding="utf-8"))
            payload = decoded if isinstance(decoded, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Generation provenance is unreadable: {type(exc).__name__}: {exc}")
    current_source = git_source_state(repo_root)
    from psse_env.dagger.suite_builder import local_diagnostic_build_enabled

    _diagnostic = local_diagnostic_build_enabled()
    if not current_source.get("release_eligible_source") and not _diagnostic:
        failures.append("Current source worktree is not release eligible.")
    if payload:
        if payload.get("release_eligible") is not True and not _diagnostic:
            failures.append("Dataset generation provenance is not release eligible.")
        descriptor = payload.get("generation_descriptor")
        descriptor = descriptor if isinstance(descriptor, Mapping) else {}
        expected_id = payload.get("generation_provenance_id")
        if not descriptor or stable_json_sha256(descriptor) != expected_id:
            failures.append("Generation provenance identifier is invalid.")
        generated_source = descriptor.get("source_state")
        generated_source = generated_source if isinstance(generated_source, Mapping) else {}
        if generated_source.get("release_eligible_source") is not True and not _diagnostic:
            failures.append("Dataset was generated from a non-release source worktree.")
        if generated_source.get("source_commit") != current_source.get(
            "source_commit"
        ) and not _diagnostic:
            failures.append("Dataset source commit does not match the current gate commit.")
        generator_hashes = descriptor.get("generator_hashes")
        generator_hashes = (
            generator_hashes if isinstance(generator_hashes, Mapping) else {}
        )
        if not generator_hashes:
            failures.append("Generation provenance has no generator/exporter hashes.")
        for relative, expected_hash in sorted(generator_hashes.items()):
            source_path = Path(repo_root) / str(relative)
            if not source_path.is_file() or file_sha256(source_path) != expected_hash:
                failures.append(
                    f"Generator/exporter source hash mismatch for {relative}."
                )
        row_ids = {row.get("generation_provenance_id") for row in rows}
        if not expected_id or row_ids != {expected_id}:
            failures.append(
                "Rows do not all bind to the generation provenance identifier."
            )
        recorded_hashes = payload.get("dataset_hashes")
        recorded_hashes = recorded_hashes if isinstance(recorded_hashes, Mapping) else {}
        for split_name, path in datasets.items():
            filename = Path(path).name
            expected_hash = recorded_hashes.get(filename)
            actual_hash = authenticated_hashes.get(split_name)
            if actual_hash is None:
                actual_hash = file_sha256(path)
            if expected_hash != actual_hash:
                failures.append(f"Dataset hash mismatch or missing provenance for {filename}.")
        schema_hashes = tool_schema_hashes(rows)
        if schema_hashes != [descriptor.get("schema_registry_hash")]:
            failures.append("Row tool schemas do not match generation provenance.")
        if (
            len(parents) == 1
            and descriptor.get("builder_contract")
            != ROUND1_AGGREGATE_BUILDER_CONTRACT
        ):
            aggregate_manifest = validate_aggregate_manifest_binding(
                payload,
                aggregate_dir=next(iter(parents)),
            )
            failures.extend(aggregate_manifest["failures"])
    return {
        "passed": not failures,
        "failures": failures,
        "provenance_path": str(provenance_path) if provenance_path is not None else None,
        "generation_provenance_id": payload.get("generation_provenance_id"),
        "source_commit": (payload.get("source_state") or {}).get("source_commit")
        if isinstance(payload.get("source_state"), Mapping)
        else None,
        "release_eligible": payload.get("release_eligible") is True and not failures,
    }


__all__ = [
    "AGGREGATE_MANIFEST_FILENAME",
    "ROUND1_AGGREGATE_BUILDER_CONTRACT",
    "build_gate_provenance",
    "file_sha256",
    "git_source_state",
    "stable_json_sha256",
    "tool_schema_hashes",
    "validate_aggregate_manifest_binding",
    "validate_generation_provenance",
    "validate_release_gate_report",
]
