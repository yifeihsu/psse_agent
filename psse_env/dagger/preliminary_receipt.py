"""Validation and stage receipts for the non-release preliminary DAgger run.

This module deliberately does not reuse the production Round-1 source gate.
The preliminary corpus may contain legacy diagnostic rows that are useful for
debugging but are not production-label eligible.  Its distinct receipt
contract prevents those rows or checkpoints from being mistaken for release
evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.sft.preliminary_hardware import (
    load_preliminary_hardware_attestation,
)


DATASET_RECEIPT_CONTRACT = "preliminary_dagger_dataset_receipt_v1"
DATASET_ARTIFACT_TYPE = "preliminary_dagger_nonrelease_dataset"
BUILDER_SOURCE_ATTESTATION_CONTRACT = (
    "preliminary_dagger_builder_source_attestation_v1"
)
STAGE_RECEIPT_CONTRACT = "preliminary_dagger_training_stage_receipt_v1"
STAGE_PLAN_CONTRACT = "preliminary_dagger_training_stage_plan_v1"
TRAINING_SOURCE_ATTESTATION_CONTRACT = (
    "preliminary_dagger_training_source_attestation_v1"
)
PINNED_MODEL_NAME = "unsloth/gemma-4-E2B-it"
PINNED_MODEL_REVISION = "f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae"
PINNED_MAX_SEQ_LENGTH = 8192

EXPECTED_SPLITS = {
    "bc0_train": "preliminary.bc0_train.jsonl",
    "bc0_validation": "preliminary.bc0_validation.jsonl",
    "d1_train": "preliminary.d1_train.jsonl",
    "d1_validation": "preliminary.d1_validation.jsonl",
    "d1_test": "preliminary.d1_test.jsonl",
    "d1_eval_combined": "preliminary.d1_eval_combined.jsonl",
    "mixed_train": "preliminary.mixed_train.jsonl",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_BUILDER_SOURCE_PATHS = {
    "psse_env/dagger/preliminary_dataset.py",
    "scripts/build_preliminary_dagger_dataset.py",
}
_TRAINING_SOURCE_PATHS = {
    "gemma_adapter_loader.py",
    "gpt_oss_power_sft_revised_v3.py",
    "hif_search_limits.py",
    "psse_env/dagger/preliminary_e2b_eval.py",
    "psse_env/dagger/preliminary_receipt.py",
    "psse_env/dagger/preliminary_tool_gate.py",
    "psse_env/sft/preliminary_adapter.py",
    "psse_env/sft/preliminary_hardware.py",
    "psse_env/sft/release_hardware.py",
    "submit_preliminary_dagger_e2b.sh",
    "trace_protocol.py",
}


class PreliminaryReceiptError(ValueError):
    """The explicitly non-release preliminary evidence failed closed."""


def _is_pinned_model_reference(value: Any) -> bool:
    """Accept the canonical Hub id or its exact, locally cached revision."""

    if value == PINNED_MODEL_NAME:
        return True
    if not isinstance(value, str) or not value:
        return False
    snapshot = Path(value)
    return bool(
        snapshot.is_absolute()
        and snapshot.name.lower() == PINNED_MODEL_REVISION
        and snapshot.is_dir()
        and (snapshot / "config.json").is_file()
    )


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreliminaryReceiptError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreliminaryReceiptError(f"{label} must be one JSON object")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreliminaryReceiptError(f"{field} must be an object")
    return value


def _canonical_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PreliminaryReceiptError(f"{field} must be lowercase 64-hex")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PreliminaryReceiptError(f"{field} must be a positive integer")
    return value


def _load_jsonl(path: Path, *, split: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    roots: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                text = raw_line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise PreliminaryReceiptError(
                        f"{split} has invalid JSON at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise PreliminaryReceiptError(
                        f"{split} line {line_number} is not a JSON object"
                    )
                root = row.get("physical_root_fingerprint")
                if not isinstance(root, str) or not root.strip():
                    raise PreliminaryReceiptError(
                        f"{split} line {line_number} lacks physical_root_fingerprint"
                    )
                roots.add(root.strip())
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        raise PreliminaryReceiptError(f"cannot read {split} file {path}: {exc}") from exc
    if not rows:
        raise PreliminaryReceiptError(f"{split} must contain at least one row")
    return rows, sorted(roots)


def _row_multiset(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(_stable_json_sha256(dict(row)) for row in rows)


def _row_multiset_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    expanded = sorted(
        digest for digest, count in _row_multiset(rows).items() for _ in range(count)
    )
    return _stable_json_sha256(expanded)


def _declared_roots(entry: Mapping[str, Any], *, field: str) -> list[str]:
    value = entry.get("physical_roots")
    if not isinstance(value, list) or any(
        not isinstance(root, str) or not root.strip() for root in value
    ):
        raise PreliminaryReceiptError(f"{field}.physical_roots must be a string list")
    normalized = [root.strip() for root in value]
    if normalized != sorted(set(normalized)):
        raise PreliminaryReceiptError(
            f"{field}.physical_roots must be sorted and duplicate-free"
        )
    return normalized


def _core_file_identity(entry: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    return {
        "sha256": _canonical_sha256(entry.get("sha256"), field=f"{field}.sha256"),
        "row_count": _positive_int(entry.get("row_count"), field=f"{field}.row_count"),
        "physical_root_count": _positive_int(
            entry.get("physical_root_count"),
            field=f"{field}.physical_root_count",
        ),
        "physical_root_set_sha256": _canonical_sha256(
            entry.get("physical_root_set_sha256"),
            field=f"{field}.physical_root_set_sha256",
        ),
    }


def _validate_training_source_attestation(value: Any) -> dict[str, Any]:
    attestation = _mapping(value, field="training_source_attestation")
    if attestation.get("contract") != TRAINING_SOURCE_ATTESTATION_CONTRACT:
        raise PreliminaryReceiptError(
            "training_source_attestation.contract must be "
            f"{TRAINING_SOURCE_ATTESTATION_CONTRACT!r}"
        )
    if attestation.get("release_eligible") is not False:
        raise PreliminaryReceiptError(
            "training_source_attestation must be release_eligible=false"
        )
    source_commit = attestation.get("source_commit")
    if not isinstance(source_commit, str) or _COMMIT_RE.fullmatch(source_commit) is None:
        raise PreliminaryReceiptError(
            "training_source_attestation.source_commit must be lowercase 40-hex"
        )
    if attestation.get("tracked_files_match_head") is not True:
        raise PreliminaryReceiptError(
            "training_source_attestation.tracked_files_match_head must be true"
        )
    tracked_files = _mapping(
        attestation.get("tracked_files"),
        field="training_source_attestation.tracked_files",
    )
    if set(tracked_files) != _TRAINING_SOURCE_PATHS:
        raise PreliminaryReceiptError(
            "training_source_attestation.tracked_files must contain exactly the "
            "preliminary launcher, validators, adapter/hardware helpers, Trainer, "
            "and direct local dependencies"
        )
    for source_path in sorted(_TRAINING_SOURCE_PATHS):
        source = _mapping(
            tracked_files.get(source_path),
            field=f"training_source_attestation.tracked_files.{source_path}",
        )
        blob_oid = source.get("git_blob_oid")
        if not isinstance(blob_oid, str) or _GIT_OBJECT_RE.fullmatch(blob_oid) is None:
            raise PreliminaryReceiptError(
                "training_source_attestation tracked git_blob_oid must be "
                "lowercase 40- or 64-hex"
            )
        _canonical_sha256(
            source.get("sha256"),
            field=f"training_source_attestation.tracked_files.{source_path}.sha256",
        )
        _positive_int(
            source.get("size_bytes"),
            field=(
                "training_source_attestation.tracked_files."
                f"{source_path}.size_bytes"
            ),
        )
    return dict(attestation)


def _training_source_attestation(repo_root: Path) -> dict[str, Any]:
    """Require the preliminary training implementation to equal committed bytes."""

    root = repo_root.resolve(strict=True)
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        top_level.returncode != 0
        or Path(top_level.stdout.strip()).resolve(strict=True) != root
    ):
        raise PreliminaryReceiptError(
            "training source repo_root must be the current Git worktree root"
        )
    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = head_result.stdout.strip().lower()
    if head_result.returncode != 0 or _COMMIT_RE.fullmatch(head) is None:
        raise PreliminaryReceiptError(
            "preliminary training must run from a Git worktree with a concrete HEAD"
        )

    tracked: dict[str, dict[str, Any]] = {}
    for relative in sorted(_TRAINING_SOURCE_PATHS):
        source_path = root / relative
        if source_path.is_symlink() or not source_path.is_file():
            raise PreliminaryReceiptError(
                f"missing preliminary training source: {relative}"
            )
        tracked_result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if tracked_result.returncode != 0:
            raise PreliminaryReceiptError(
                f"preliminary training source is not tracked at HEAD: {relative}"
            )
        clean_result = subprocess.run(
            ["git", "diff", "--quiet", "--no-ext-diff", "HEAD", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if clean_result.returncode != 0:
            raise PreliminaryReceiptError(
                f"preliminary training source differs from HEAD: {relative}"
            )
        blob_result = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        blob_oid = blob_result.stdout.strip().lower()
        if (
            blob_result.returncode != 0
            or _GIT_OBJECT_RE.fullmatch(blob_oid) is None
        ):
            raise PreliminaryReceiptError(
                f"cannot resolve committed training blob: {relative}"
            )
        committed_result = subprocess.run(
            ["git", "cat-file", "blob", f"HEAD:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        working_bytes = source_path.read_bytes()
        if (
            committed_result.returncode != 0
            or committed_result.stdout != working_bytes
        ):
            raise PreliminaryReceiptError(
                f"preliminary training source bytes differ from HEAD: {relative}"
            )
        if not working_bytes:
            raise PreliminaryReceiptError(
                f"preliminary training source is empty: {relative}"
            )
        tracked[relative] = {
            "git_blob_oid": blob_oid,
            "sha256": hashlib.sha256(working_bytes).hexdigest(),
            "size_bytes": len(working_bytes),
        }
    return _validate_training_source_attestation(
        {
            "contract": TRAINING_SOURCE_ATTESTATION_CONTRACT,
            "release_eligible": False,
            "source_commit": head,
            "tracked_files_match_head": True,
            "tracked_files": tracked,
        }
    )


def validate_preliminary_dataset_receipt(
    receipt_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Rehash and validate one explicitly non-release preliminary corpus."""

    path = Path(receipt_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise PreliminaryReceiptError(
            "dataset receipt must be an existing regular file, not a symlink"
        )
    path = path.resolve(strict=True)
    receipt = _json_object(path, label="dataset receipt")
    if receipt.get("contract") != DATASET_RECEIPT_CONTRACT:
        raise PreliminaryReceiptError(
            f"dataset receipt contract must be {DATASET_RECEIPT_CONTRACT!r}"
        )
    if receipt.get("artifact_type") != DATASET_ARTIFACT_TYPE:
        raise PreliminaryReceiptError(
            f"dataset artifact_type must be {DATASET_ARTIFACT_TYPE!r}"
        )
    if receipt.get("release_eligible") is not False:
        raise PreliminaryReceiptError(
            "preliminary dataset receipt must explicitly set release_eligible=false"
        )
    reasons = receipt.get("release_ineligibility_reasons")
    if not isinstance(reasons, list) or not reasons or any(
        not isinstance(reason, str) or not reason.strip() for reason in reasons
    ):
        raise PreliminaryReceiptError(
            "release_ineligibility_reasons must be a non-empty string list"
        )

    source_commits = _mapping(receipt.get("source_commits"), field="source_commits")
    for name in ("d0_generation", "diagnostic_collection", "builder"):
        commit = source_commits.get(name)
        if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
            raise PreliminaryReceiptError(
                f"source_commits.{name} must be lowercase 40-hex"
            )

    builder_source = _mapping(
        receipt.get("builder_source_attestation"),
        field="builder_source_attestation",
    )
    if builder_source.get("contract") != BUILDER_SOURCE_ATTESTATION_CONTRACT:
        raise PreliminaryReceiptError(
            "builder_source_attestation.contract must be "
            f"{BUILDER_SOURCE_ATTESTATION_CONTRACT!r}"
        )
    if builder_source.get("source_commit") != source_commits["builder"]:
        raise PreliminaryReceiptError(
            "builder_source_attestation.source_commit must equal "
            "source_commits.builder"
        )
    if builder_source.get("tracked_files_match_head") is not True:
        raise PreliminaryReceiptError(
            "builder_source_attestation.tracked_files_match_head must be true"
        )
    tracked_files = _mapping(
        builder_source.get("tracked_files"),
        field="builder_source_attestation.tracked_files",
    )
    if set(tracked_files) != _BUILDER_SOURCE_PATHS:
        raise PreliminaryReceiptError(
            "builder_source_attestation.tracked_files must contain exactly the "
            "builder module and wrapper"
        )
    for source_path in sorted(_BUILDER_SOURCE_PATHS):
        source = _mapping(
            tracked_files.get(source_path),
            field=f"builder_source_attestation.tracked_files.{source_path}",
        )
        blob_oid = source.get("git_blob_oid")
        if not isinstance(blob_oid, str) or _GIT_OBJECT_RE.fullmatch(blob_oid) is None:
            raise PreliminaryReceiptError(
                "builder_source_attestation tracked git_blob_oid must be lowercase "
                "40- or 64-hex"
            )
        _canonical_sha256(
            source.get("sha256"),
            field=(
                "builder_source_attestation.tracked_files."
                f"{source_path}.sha256"
            ),
        )
        _positive_int(
            source.get("size_bytes"),
            field=(
                "builder_source_attestation.tracked_files."
                f"{source_path}.size_bytes"
            ),
        )

    outputs = _mapping(receipt.get("outputs"), field="outputs")
    splits = _mapping(receipt.get("splits"), field="splits")
    resolved_paths: dict[str, str] = {}
    roots_by_split: dict[str, list[str]] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    file_hashes: dict[str, str] = {}

    for split, expected_filename in EXPECTED_SPLITS.items():
        split_entry = _mapping(splits.get(split), field=f"splits.{split}")
        if split_entry.get("filename") != expected_filename:
            raise PreliminaryReceiptError(
                f"splits.{split}.filename must be {expected_filename!r}"
            )
        output_entry = _mapping(
            outputs.get(expected_filename), field=f"outputs.{expected_filename}"
        )
        split_identity = _core_file_identity(split_entry, field=f"splits.{split}")
        output_identity = _core_file_identity(
            output_entry, field=f"outputs.{expected_filename}"
        )
        if split_identity != output_identity:
            raise PreliminaryReceiptError(
                f"splits.{split} identity differs from outputs.{expected_filename}"
            )

        data_path = path.parent / expected_filename
        if data_path.is_symlink() or not data_path.is_file():
            raise PreliminaryReceiptError(
                f"{split} must resolve beside the receipt as a regular non-symlink file"
            )
        actual_sha256 = _file_sha256(data_path)
        if actual_sha256 != split_identity["sha256"]:
            raise PreliminaryReceiptError(f"{split} SHA-256 does not match its receipt")
        rows, actual_roots = _load_jsonl(data_path, split=split)
        declared_roots = _declared_roots(split_entry, field=f"splits.{split}")
        if declared_roots != actual_roots:
            raise PreliminaryReceiptError(
                f"splits.{split}.physical_roots do not match the JSONL rows"
            )
        if split_identity["row_count"] != len(rows):
            raise PreliminaryReceiptError(f"{split} row_count does not match")
        if split_identity["physical_root_count"] != len(actual_roots):
            raise PreliminaryReceiptError(f"{split} physical_root_count does not match")
        if split_identity["physical_root_set_sha256"] != _stable_json_sha256(
            actual_roots
        ):
            raise PreliminaryReceiptError(
                f"{split} physical_root_set_sha256 does not match"
            )
        if split.startswith("d1_") and any(
            row.get("production_label_eligible") is not False for row in rows
        ):
            raise PreliminaryReceiptError(
                f"{split} must keep production_label_eligible=false"
            )
        resolved_paths[split] = str(data_path.resolve(strict=True))
        roots_by_split[split] = actual_roots
        rows_by_split[split] = rows
        row_counts[split] = len(rows)
        file_hashes[split] = actual_sha256

    root_sets = {name: set(values) for name, values in roots_by_split.items()}
    disjoint_pairs = (
        ("bc0_train", "bc0_validation"),
        ("d1_train", "d1_validation"),
        ("d1_train", "d1_test"),
        ("d1_validation", "d1_test"),
    )
    for left, right in disjoint_pairs:
        overlap = sorted(root_sets[left] & root_sets[right])
        if overlap:
            raise PreliminaryReceiptError(
                f"{left} and {right} physical roots overlap: {overlap[:5]}"
            )
    bc0_roots = root_sets["bc0_train"] | root_sets["bc0_validation"]
    d1_roots = (
        root_sets["d1_train"]
        | root_sets["d1_validation"]
        | root_sets["d1_test"]
    )
    if bc0_roots & d1_roots:
        raise PreliminaryReceiptError("BC0 and D1 physical roots must be disjoint")
    if root_sets["d1_eval_combined"] != (
        root_sets["d1_validation"] | root_sets["d1_test"]
    ):
        raise PreliminaryReceiptError(
            "d1_eval_combined roots must equal D1 validation plus test roots"
        )
    if root_sets["mixed_train"] != (
        root_sets["bc0_train"] | root_sets["d1_train"]
    ):
        raise PreliminaryReceiptError(
            "mixed_train roots must equal BC0 train plus D1 train roots"
        )
    expected_combined_rows = [
        *rows_by_split["d1_validation"],
        *rows_by_split["d1_test"],
    ]
    if _row_multiset(rows_by_split["d1_eval_combined"]) != _row_multiset(
        expected_combined_rows
    ):
        raise PreliminaryReceiptError(
            "d1_eval_combined rows must exactly equal D1 validation plus test rows"
        )
    expected_mixed_rows = [
        *rows_by_split["bc0_train"],
        *rows_by_split["d1_train"],
    ]
    if _row_multiset(rows_by_split["mixed_train"]) != _row_multiset(
        expected_mixed_rows
    ):
        raise PreliminaryReceiptError(
            "mixed_train rows must exactly equal BC0 train plus D1 train rows"
        )

    audits = _mapping(receipt.get("audits"), field="audits")
    required_audits = (
        "strict_failure_bundle_admission",
        "d0_generation_binding",
        "evaluation_root_reservation",
        "evaluation_root_partition",
        "canonical_d1_training_selection",
        "root_disjointness",
        "content_composition",
    )
    for audit_name in required_audits:
        audit = _mapping(
            audits.get(audit_name), field=f"audits.{audit_name}"
        )
        if audit.get("passed") is not True:
            raise PreliminaryReceiptError(
                f"audits.{audit_name}.passed must be true"
            )
    root_audit = _mapping(
        audits.get("root_disjointness"), field="audits.root_disjointness"
    )
    recorded_overlaps = _mapping(
        root_audit.get("required_pairwise_overlaps"),
        field="audits.root_disjointness.required_pairwise_overlaps",
    )
    if not recorded_overlaps or any(value != [] for value in recorded_overlaps.values()):
        raise PreliminaryReceiptError(
            "audits.root_disjointness.required_pairwise_overlaps must contain "
            "only empty overlap lists"
        )
    content_audit = _mapping(
        audits.get("content_composition"), field="audits.content_composition"
    )
    expected_composition = {
        "d1_validation_plus_test_row_multiset_sha256": _row_multiset_sha256(
            expected_combined_rows
        ),
        "d1_eval_combined_row_multiset_sha256": _row_multiset_sha256(
            rows_by_split["d1_eval_combined"]
        ),
        "bc0_plus_d1_train_row_multiset_sha256": _row_multiset_sha256(
            expected_mixed_rows
        ),
        "mixed_train_row_multiset_sha256": _row_multiset_sha256(
            rows_by_split["mixed_train"]
        ),
    }
    for field, expected in expected_composition.items():
        if content_audit.get(field) != expected:
            raise PreliminaryReceiptError(
                f"audits.content_composition.{field} does not match dataset rows"
            )

    return {
        "contract": DATASET_RECEIPT_CONTRACT,
        "artifact_type": DATASET_ARTIFACT_TYPE,
        "release_eligible": False,
        "receipt_path": str(path),
        "receipt_sha256": _file_sha256(path),
        "paths": resolved_paths,
        "row_counts": row_counts,
        "file_sha256": file_hashes,
        "physical_roots": roots_by_split,
    }


def _tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise PreliminaryReceiptError(f"adapter tree is not a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise PreliminaryReceiptError(f"adapter tree contains symlink: {child}")
        if child.is_file():
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "sha256": _file_sha256(child),
                    "size": child.stat().st_size,
                }
            )
    if not entries:
        raise PreliminaryReceiptError("adapter tree is empty")
    return _stable_json_sha256(entries)


def _latest_trainer_metrics(output_dir: Path) -> dict[str, Any]:
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.glob("checkpoint-*/trainer_state.json"):
        suffix = child.parent.name.removeprefix("checkpoint-")
        if suffix.isdigit() and child.is_file() and not child.is_symlink():
            candidates.append((int(suffix), child))
    if not candidates:
        return {}
    step, state_path = max(candidates, key=lambda item: item[0])
    state = _json_object(state_path, label="trainer state")
    history = state.get("log_history")
    latest: dict[str, Any] = {
        "checkpoint_step": step,
        "trainer_global_step": state.get("global_step"),
    }
    latest_eval_step = -1
    if isinstance(history, list):
        for record in history:
            if not isinstance(record, Mapping):
                continue
            for key in ("epoch", "loss", "eval_loss", "learning_rate"):
                value = record.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key == "eval_loss":
                        record_step = record.get("step")
                        if (
                            isinstance(record_step, int)
                            and not isinstance(record_step, bool)
                            and record_step >= latest_eval_step
                        ):
                            latest_eval_step = record_step
                            latest[key] = value
                            latest["eval_step"] = record_step
                    else:
                        latest[key] = value
    return latest


def _stage_plan_payload(
    *,
    stage: str,
    dataset_receipt: Path,
    train_file: Path,
    validation_file: Path,
    output_dir: Path,
    repo_root: Path,
    initial_adapter: Path | None,
    training_seed: int,
    max_train_rows: int,
    max_valid_rows: int,
    max_steps: int,
    max_seq_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    save_steps: int,
    eval_steps: int,
    save_total_limit: int,
    dataloader_workers: int,
    report_to: str,
) -> dict[str, Any]:
    if stage not in {"bc0", "dagger"}:
        raise PreliminaryReceiptError("stage plan must be bc0 or dagger")
    dataset = validate_preliminary_dataset_receipt(dataset_receipt)
    train_path = train_file.resolve(strict=True)
    validation_path = validation_file.resolve(strict=True)
    expected_train_split = "bc0_train" if stage == "bc0" else "mixed_train"
    if train_path != Path(dataset["paths"][expected_train_split]):
        raise PreliminaryReceiptError(
            f"{stage} stage plan must use receipt split {expected_train_split}"
        )
    if validation_path != Path(dataset["paths"]["d1_validation"]):
        raise PreliminaryReceiptError(
            "both stages must use the receipt's held-out D1 validation split"
        )
    train_rows, _ = _load_jsonl(train_path, split=f"{stage} plan train")
    validation_rows, _ = _load_jsonl(
        validation_path, split=f"{stage} plan validation"
    )
    integer_values = {
        "training_seed": training_seed,
        "max_train_rows": max_train_rows,
        "max_valid_rows": max_valid_rows,
        "max_steps": max_steps,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "save_total_limit": save_total_limit,
        "dataloader_workers": dataloader_workers,
    }
    for field, value in integer_values.items():
        minimum = 0 if field in {"training_seed", "dataloader_workers"} else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise PreliminaryReceiptError(
                f"stage plan {field} must be an integer >= {minimum}"
            )
    if not isinstance(learning_rate, float) or not math.isfinite(
        learning_rate
    ) or learning_rate <= 0:
        raise PreliminaryReceiptError(
            "stage plan learning_rate must be a positive finite float"
        )
    if report_to not in {"none", "wandb"}:
        raise PreliminaryReceiptError("stage plan report_to must be none or wandb")
    if max_seq_length != PINNED_MAX_SEQ_LENGTH:
        raise PreliminaryReceiptError(
            "preliminary E2B stages must use the pinned 8192-token context so "
            "reviewed prompts cannot be silently left-truncated"
        )
    if max_valid_rows < len(validation_rows):
        raise PreliminaryReceiptError(
            "stage plan must evaluate every held-out D1 validation row"
        )
    if max_steps % eval_steps != 0 or max_steps % save_steps != 0:
        raise PreliminaryReceiptError(
            "stage max_steps must be divisible by eval_steps and save_steps so "
            "final-step evaluation evidence is persisted"
        )
    if stage == "bc0" and initial_adapter is not None:
        raise PreliminaryReceiptError("BC0 stage plan cannot have an initial adapter")
    if stage == "dagger" and initial_adapter is None:
        raise PreliminaryReceiptError("DAgger stage plan requires an initial adapter")
    optimizer_rows = train_rows[:max_train_rows]
    d1_training_roots = set(dataset["physical_roots"]["d1_train"])
    optimizer_d1_rows = [
        row
        for row in optimizer_rows
        if row.get("physical_root_fingerprint") in d1_training_roots
    ]
    optimizer_d1_roots = sorted(
        {
            str(row["physical_root_fingerprint"])
            for row in optimizer_d1_rows
        }
    )
    if stage == "dagger" and not optimizer_d1_rows:
        raise PreliminaryReceiptError(
            "DAgger optimizer-visible prefix must contain at least one D1 row/root"
        )
    initial_path: str | None = None
    initial_tree: str | None = None
    if initial_adapter is not None:
        initial = initial_adapter.resolve(strict=True)
        initial_path = str(initial)
        initial_tree = _tree_sha256(initial)
    run_name = f"prelim-e2b-{stage}-seed{training_seed}"
    training_source = _training_source_attestation(repo_root)
    from psse_env.dagger.preliminary_tool_gate import gate_plan_contract

    return {
        "contract": STAGE_PLAN_CONTRACT,
        "artifact_type": "preliminary_dagger_nonrelease_training_plan",
        "release_eligible": False,
        "training_source_attestation": training_source,
        "stage": stage,
        "model": PINNED_MODEL_NAME,
        "model_revision": PINNED_MODEL_REVISION,
        "dataset_receipt_sha256": dataset["receipt_sha256"],
        "train_file": str(train_path),
        "train_file_sha256": _file_sha256(train_path),
        "train_file_row_count": len(train_rows),
        "validation_file": str(validation_path),
        "validation_file_sha256": _file_sha256(validation_path),
        "validation_file_row_count": len(validation_rows),
        "initial_adapter": initial_path,
        "initial_adapter_tree_sha256": initial_tree,
        "training_arguments": {
            **integer_values,
            "learning_rate": learning_rate,
            "num_train_epochs": 1.0,
            "dataset_num_proc": 1,
            "per_device_eval_batch_size": 1,
            "warmup_steps": 4,
            "logging_steps": 2,
            "lora_dropout": 0.0,
            "lora_target_scope": "language_model",
            "weight_decay": 0.001,
            "lr_scheduler_type": "linear",
            "load_in_4bit": False,
            "load_in_16bit": True,
            "drop_too_long_targets": True,
            "report_to": report_to,
            "run_name": run_name,
        },
        "prompt_arguments": {
            "include_tool_schemas": True,
            "tools_file": "",
            "phase_gated_prompt": False,
            "inject_empty_thought_channel": True,
            "preserve_system_text": True,
            "keep_debug_text": False,
            "repeat_first_tool_call": 1,
            "repeat_later_tool_call": 1,
            "repeat_final": 1,
            "fail_on_prompt_truncation": True,
            "sanity_check_samples": 0,
        },
        "generation_gate": gate_plan_contract(stage),
        "optimizer_visible_train_rows": len(optimizer_rows),
        "optimizer_visible_d1_row_count": len(optimizer_d1_rows),
        "optimizer_visible_d1_root_count": len(optimizer_d1_roots),
        "optimizer_visible_d1_root_set_sha256": _stable_json_sha256(
            optimizer_d1_roots
        ),
        "evaluated_validation_rows": len(validation_rows),
        "output_dir": str(output_dir.resolve(strict=True)),
    }


def _publish_write_once_json(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists() or path.is_symlink():
        raise PreliminaryReceiptError(f"write-once artifact already exists: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        os.chmod(path, 0o400)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except FileExistsError as exc:
        raise PreliminaryReceiptError(
            f"write-once publication raced with an existing artifact: {path}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def ensure_preliminary_stage_plan(
    *,
    stage: str,
    dataset_receipt: str | os.PathLike[str],
    train_file: str | os.PathLike[str],
    validation_file: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    initial_adapter: str | os.PathLike[str] | None,
    training_seed: int,
    max_train_rows: int,
    max_valid_rows: int,
    max_steps: int,
    max_seq_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    save_steps: int,
    eval_steps: int,
    save_total_limit: int,
    dataloader_workers: int,
    report_to: str,
) -> Path:
    """Publish or revalidate the plan before accepting any Trainer state."""

    output = Path(output_dir).resolve(strict=True)
    plan_path = output / "preliminary_stage_plan.json"
    expected = _stage_plan_payload(
        stage=stage,
        dataset_receipt=Path(dataset_receipt),
        train_file=Path(train_file),
        validation_file=Path(validation_file),
        output_dir=output,
        repo_root=Path(repo_root),
        initial_adapter=(
            Path(initial_adapter) if initial_adapter is not None else None
        ),
        training_seed=training_seed,
        max_train_rows=max_train_rows,
        max_valid_rows=max_valid_rows,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        dataloader_workers=dataloader_workers,
        report_to=report_to,
    )
    if plan_path.exists() or plan_path.is_symlink():
        if plan_path.is_symlink() or not plan_path.is_file():
            raise PreliminaryReceiptError("stage plan is not a regular file")
        if _json_object(plan_path, label="stage plan") != expected:
            raise PreliminaryReceiptError(
                "existing stage plan differs from current data or training arguments"
            )
        return plan_path
    stale = [
        path.name
        for path in output.iterdir()
        if path.name == "run_config.json"
        or path.name == "lora"
        or path.name == "preliminary_stage_receipt.json"
        or (path.name.startswith("checkpoint-") and path.name[11:].isdigit())
    ]
    if stale:
        raise PreliminaryReceiptError(
            "refusing preexisting Trainer state without an immutable stage plan: "
            + ", ".join(sorted(stale))
        )
    return _publish_write_once_json(plan_path, expected)


def _load_stage_plan(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreliminaryReceiptError("stage plan is missing or is a symlink")
    plan = _json_object(path, label="stage plan")
    if plan.get("contract") != STAGE_PLAN_CONTRACT:
        raise PreliminaryReceiptError("invalid preliminary stage plan contract")
    if plan.get("release_eligible") is not False:
        raise PreliminaryReceiptError("stage plan must be release_eligible=false")
    _validate_training_source_attestation(plan.get("training_source_attestation"))
    return plan


def _validate_run_config(
    *, plan: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    path = output_dir / "run_config.json"
    if path.is_symlink() or not path.is_file():
        raise PreliminaryReceiptError("Trainer run_config.json is missing or a symlink")
    config = _json_object(path, label="Trainer run config")
    training = _mapping(plan.get("training_arguments"), field="plan.training_arguments")
    prompt = _mapping(plan.get("prompt_arguments"), field="plan.prompt_arguments")
    requested_model_name = config.get("requested_model_name")
    if not _is_pinned_model_reference(requested_model_name):
        raise PreliminaryReceiptError(
            "Trainer run config requested_model_name differs from the pinned model"
        )
    expected_top = {
        "requested_model_revision": PINNED_MODEL_REVISION,
        "output_dir": str(output_dir),
        "train_file": plan.get("train_file"),
        "valid_file": plan.get("validation_file"),
        "init_adapter": plan.get("initial_adapter") or "",
    }
    for field, expected in expected_top.items():
        if config.get(field) != expected:
            raise PreliminaryReceiptError(
                f"Trainer run config {field} differs from stage plan"
            )
    expected_sft = {
        "max_seq_length": training.get("max_seq_length"),
        "dataset_num_proc": training.get("dataset_num_proc"),
        "per_device_train_batch_size": training.get("batch_size"),
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size"),
        "gradient_accumulation_steps": training.get(
            "gradient_accumulation_steps"
        ),
        "warmup_steps": training.get("warmup_steps"),
        "max_steps": training.get("max_steps"),
        "num_train_epochs": training.get("num_train_epochs"),
        "learning_rate": training.get("learning_rate"),
        "logging_steps": training.get("logging_steps"),
        "save_steps": training.get("save_steps"),
        "eval_steps": training.get("eval_steps"),
        "save_total_limit": training.get("save_total_limit"),
        "weight_decay": training.get("weight_decay"),
        "lr_scheduler_type": training.get("lr_scheduler_type"),
        "dataloader_num_workers": training.get("dataloader_workers"),
        "drop_too_long_targets": training.get("drop_too_long_targets"),
        "load_in_4bit": training.get("load_in_4bit"),
        "load_in_16bit": training.get("load_in_16bit"),
        "lora_r": training.get("lora_r"),
        "lora_alpha": training.get("lora_alpha"),
        "lora_dropout": training.get("lora_dropout"),
        "lora_target_scope": training.get("lora_target_scope"),
        "report_to": training.get("report_to"),
        "run_name": training.get("run_name"),
        "seed": training.get("training_seed"),
    }
    sft = _mapping(config.get("sft_args"), field="run_config.sft_args")
    if any(sft.get(field) != expected for field, expected in expected_sft.items()):
        mismatches = sorted(
            field
            for field, expected in expected_sft.items()
            if sft.get(field) != expected
        )
        raise PreliminaryReceiptError(
            "Trainer SFT arguments differ from stage plan: " + ", ".join(mismatches)
        )
    expected_prompt = {
        field: value
        for field, value in prompt.items()
        if field != "sanity_check_samples"
    }
    recorded_prompt = _mapping(
        config.get("prompt_args"), field="run_config.prompt_args"
    )
    if any(
        recorded_prompt.get(field) != expected
        for field, expected in expected_prompt.items()
    ):
        raise PreliminaryReceiptError(
            "Trainer prompt arguments differ from stage plan"
        )
    summary = _mapping(
        config.get("dataset_summary"), field="run_config.dataset_summary"
    )
    expected_train_rows = int(plan["optimizer_visible_train_rows"])
    expected_validation_rows = int(plan["evaluated_validation_rows"])
    if summary.get("raw_train_conversations") != expected_train_rows:
        raise PreliminaryReceiptError(
            "Trainer raw train row count differs from stage plan"
        )
    if summary.get("processed_train_samples") != expected_train_rows:
        raise PreliminaryReceiptError(
            "Trainer processed train rows differ from the bound optimizer prefix"
        )
    for field in (
        "raw_validation_conversations",
        "processed_validation_samples",
    ):
        if summary.get(field) != expected_validation_rows:
            raise PreliminaryReceiptError(
                f"Trainer {field} does not prove full D1 validation"
            )
    if summary.get("repeated_train_samples") != summary.get(
        "processed_train_samples"
    ):
        raise PreliminaryReceiptError(
            "Trainer repeated train rows conflict with repeat factor one"
        )
    base_reference = config.get("adapter_base_model_name_or_path")
    if plan.get("stage") == "bc0":
        if not _is_pinned_model_reference(base_reference):
            raise PreliminaryReceiptError(
                "BC0 run config lacks the pinned base model"
            )
    else:
        initial_config = _json_object(
            Path(str(plan["initial_adapter"])) / "adapter_config.json",
            label="planned initial adapter config",
        )
        if base_reference != initial_config.get("base_model_name_or_path"):
            raise PreliminaryReceiptError(
                "DAgger run config base differs from pinned initial adapter"
            )
    return config


def validate_preliminary_resume_checkpoint(
    *, stage_plan: str | os.PathLike[str], checkpoint: str | os.PathLike[str]
) -> dict[str, Any]:
    """Refuse a numeric checkpoint unless plan, run config, and state agree."""

    plan_path = Path(stage_plan).resolve(strict=True)
    plan = _load_stage_plan(plan_path)
    output = Path(str(plan.get("output_dir") or "")).resolve(strict=True)
    if plan_path != output / "preliminary_stage_plan.json":
        raise PreliminaryReceiptError(
            "resume stage plan must be the fixed plan inside its output directory"
        )
    candidate = Path(checkpoint)
    if candidate.is_symlink() or not candidate.is_dir():
        raise PreliminaryReceiptError("resume checkpoint is not a regular directory")
    candidate = candidate.resolve(strict=True)
    suffix = candidate.name.removeprefix("checkpoint-")
    if (
        candidate.parent != output
        or not candidate.name.startswith("checkpoint-")
        or not suffix.isdigit()
    ):
        raise PreliminaryReceiptError(
            "resume checkpoint must be one numeric child of the planned output"
        )
    _validate_run_config(plan=plan, output_dir=output)
    state_path = candidate / "trainer_state.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise PreliminaryReceiptError("resume checkpoint lacks trainer_state.json")
    state = _json_object(state_path, label="resume trainer state")
    step = int(suffix)
    if state.get("global_step") != step or not (
        0 < step <= int(_mapping(plan["training_arguments"], field="training")["max_steps"])
    ):
        raise PreliminaryReceiptError(
            "resume trainer state step does not match checkpoint/plan"
        )
    return {
        "stage_plan_sha256": _file_sha256(plan_path),
        "checkpoint": str(candidate),
        "checkpoint_step": step,
        "checkpoint_tree_sha256": _tree_sha256(candidate),
        "run_config_sha256": _file_sha256(output / "run_config.json"),
    }


def _stage_payload(
    *,
    stage: str,
    dataset_receipt: Path,
    train_file: Path,
    validation_file: Path,
    output_dir: Path,
    hardware_attestation: Path,
    stage_plan: Path,
) -> dict[str, Any]:
    if stage not in {"bc0", "dagger"}:
        raise PreliminaryReceiptError("stage must be bc0 or dagger")
    dataset = validate_preliminary_dataset_receipt(dataset_receipt)
    plan_path = stage_plan.resolve(strict=True)
    plan = _load_stage_plan(plan_path)
    if plan_path != output_dir.resolve(strict=True) / "preliminary_stage_plan.json":
        raise PreliminaryReceiptError(
            "stage receipt must use the fixed plan inside its output directory"
        )
    if (
        plan.get("stage") != stage
        or plan.get("model") != PINNED_MODEL_NAME
        or plan.get("model_revision") != PINNED_MODEL_REVISION
        or plan.get("dataset_receipt_sha256") != dataset["receipt_sha256"]
        or plan.get("train_file_sha256")
        != _file_sha256(train_file.resolve(strict=True))
        or plan.get("validation_file_sha256")
        != _file_sha256(validation_file.resolve(strict=True))
        or plan.get("output_dir") != str(output_dir.resolve(strict=True))
    ):
        raise PreliminaryReceiptError(
            "stage plan does not bind the current stage/data/model/output"
        )
    if plan.get("initial_adapter") is not None:
        initial = Path(str(plan["initial_adapter"])).resolve(strict=True)
        if plan.get("initial_adapter_tree_sha256") != _tree_sha256(initial):
            raise PreliminaryReceiptError(
                "planned initial adapter tree changed before completion"
            )
    adapter_dir = output_dir / "lora"
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise PreliminaryReceiptError("completed stage lacks lora/adapter_config.json")
    adapter_config = _json_object(config_path, label="adapter config")
    base_reference = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_reference, str) or not base_reference:
        raise PreliminaryReceiptError("adapter config lacks its base model reference")
    if stage == "bc0" and not _is_pinned_model_reference(base_reference):
        raise PreliminaryReceiptError(
            "BC0 adapter does not bind the pinned E2B model or snapshot"
        )
    if stage == "dagger":
        snapshot = Path(base_reference)
        if (
            snapshot.name.lower() != PINNED_MODEL_REVISION
            or not snapshot.is_dir()
            or not (snapshot / "config.json").is_file()
        ):
            raise PreliminaryReceiptError(
                "DAgger adapter does not bind the exact pinned E2B snapshot"
            )
    hardware = load_preliminary_hardware_attestation(hardware_attestation)
    run_config_path = output_dir / "run_config.json"
    _validate_run_config(plan=plan, output_dir=output_dir)
    metrics = _latest_trainer_metrics(output_dir)
    planned_max_steps = int(
        _mapping(plan["training_arguments"], field="plan.training_arguments")[
            "max_steps"
        ]
    )
    eval_loss = metrics.get("eval_loss")
    if (
        not isinstance(eval_loss, (int, float))
        or isinstance(eval_loss, bool)
        or not math.isfinite(float(eval_loss))
        or metrics.get("eval_step") != planned_max_steps
        or metrics.get("checkpoint_step") != planned_max_steps
        or metrics.get("trainer_global_step") != planned_max_steps
    ):
        raise PreliminaryReceiptError(
            "completed stage lacks finite full-validation eval_loss at final max step"
        )
    from psse_env.dagger.preliminary_tool_gate import (
        GATE_FILENAME,
        PreliminaryToolGateError,
        validate_gate_report,
    )

    gate_path = output_dir / GATE_FILENAME
    try:
        generation_gate = validate_gate_report(
            report_path=gate_path,
            stage_plan_path=plan_path,
            adapter_path=adapter_dir,
            validation_path=validation_file.resolve(strict=True),
            require_passed=True,
        )
    except PreliminaryToolGateError as exc:
        raise PreliminaryReceiptError(
            f"completed stage lacks a passing tool generation gate: {exc}"
        ) from exc
    return {
        "contract": STAGE_RECEIPT_CONTRACT,
        "artifact_type": "preliminary_dagger_nonrelease_checkpoint",
        "release_eligible": False,
        "release_ineligibility_reasons": [
            "small-model preliminary debugging run",
            "input dataset is explicitly non-release",
        ],
        "stage": stage,
        "model": PINNED_MODEL_NAME,
        "model_revision": PINNED_MODEL_REVISION,
        "dataset_receipt_sha256": dataset["receipt_sha256"],
        "train_file_sha256": _file_sha256(train_file.resolve(strict=True)),
        "validation_file_sha256": _file_sha256(
            validation_file.resolve(strict=True)
        ),
        "adapter_tree_sha256": _tree_sha256(adapter_dir.resolve(strict=True)),
        "adapter_base_model_name_or_path": base_reference,
        "accelerator_class": hardware["accelerator_class"],
        "hardware_attestation_sha256": _file_sha256(
            hardware_attestation.resolve(strict=True)
        ),
        "hardware_attestation": hardware,
        "stage_plan_sha256": _file_sha256(plan_path),
        "run_config_sha256": _file_sha256(run_config_path),
        "full_validation_row_count": plan["evaluated_validation_rows"],
        "latest_trainer_metrics": metrics,
        "generation_gate_report_sha256": _file_sha256(gate_path),
        "generation_gate_summary": generation_gate["summary"],
    }


def write_preliminary_stage_receipt(
    *,
    stage: str,
    dataset_receipt: str | os.PathLike[str],
    train_file: str | os.PathLike[str],
    validation_file: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    hardware_attestation: str | os.PathLike[str],
    stage_plan: str | os.PathLike[str],
) -> Path:
    """Atomically publish a write-once non-release stage completion receipt."""

    output = Path(output_dir).resolve(strict=True)
    final_path = output / "preliminary_stage_receipt.json"
    if final_path.exists() or final_path.is_symlink():
        raise PreliminaryReceiptError(f"stage receipt already exists: {final_path}")
    payload = _stage_payload(
        stage=stage,
        dataset_receipt=Path(dataset_receipt),
        train_file=Path(train_file),
        validation_file=Path(validation_file),
        output_dir=output,
        hardware_attestation=Path(hardware_attestation),
        stage_plan=Path(stage_plan),
    )
    return _publish_write_once_json(final_path, payload)


def validate_preliminary_stage_receipt(
    *,
    stage: str,
    dataset_receipt: str | os.PathLike[str],
    train_file: str | os.PathLike[str],
    validation_file: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    hardware_attestation: str | os.PathLike[str],
    stage_plan: str | os.PathLike[str],
) -> dict[str, Any]:
    output = Path(output_dir).resolve(strict=True)
    receipt_path = output / "preliminary_stage_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise PreliminaryReceiptError("stage receipt is missing or is a symlink")
    recorded = _json_object(receipt_path, label="stage receipt")
    expected = _stage_payload(
        stage=stage,
        dataset_receipt=Path(dataset_receipt),
        train_file=Path(train_file),
        validation_file=Path(validation_file),
        output_dir=output,
        hardware_attestation=Path(hardware_attestation),
        stage_plan=Path(stage_plan),
    )
    if recorded != expected:
        raise PreliminaryReceiptError(
            "stage receipt does not match current dataset, model, or adapter bytes"
        )
    return recorded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate non-release preliminary DAgger receipts."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset")
    dataset.add_argument("--receipt", required=True, type=Path)

    stage_plan = commands.add_parser("stage-plan")
    stage_plan.add_argument("--stage", required=True, choices=("bc0", "dagger"))
    stage_plan.add_argument("--dataset-receipt", required=True, type=Path)
    stage_plan.add_argument("--train-file", required=True, type=Path)
    stage_plan.add_argument("--validation-file", required=True, type=Path)
    stage_plan.add_argument("--output-dir", required=True, type=Path)
    stage_plan.add_argument("--repo-root", required=True, type=Path)
    stage_plan.add_argument("--initial-adapter", type=Path)
    stage_plan.add_argument("--training-seed", required=True, type=int)
    stage_plan.add_argument("--max-train-rows", required=True, type=int)
    stage_plan.add_argument("--max-valid-rows", required=True, type=int)
    stage_plan.add_argument("--max-steps", required=True, type=int)
    stage_plan.add_argument("--max-seq-length", required=True, type=int)
    stage_plan.add_argument("--batch-size", required=True, type=int)
    stage_plan.add_argument(
        "--gradient-accumulation-steps", required=True, type=int
    )
    stage_plan.add_argument("--learning-rate", required=True, type=float)
    stage_plan.add_argument("--lora-r", required=True, type=int)
    stage_plan.add_argument("--lora-alpha", required=True, type=int)
    stage_plan.add_argument("--save-steps", required=True, type=int)
    stage_plan.add_argument("--eval-steps", required=True, type=int)
    stage_plan.add_argument("--save-total-limit", required=True, type=int)
    stage_plan.add_argument("--dataloader-workers", required=True, type=int)
    stage_plan.add_argument(
        "--report-to", required=True, choices=("none", "wandb")
    )

    resume = commands.add_parser("resume-check")
    resume.add_argument("--stage-plan", required=True, type=Path)
    resume.add_argument("--checkpoint", required=True, type=Path)

    for name in ("stage-check", "stage-write"):
        stage = commands.add_parser(name)
        stage.add_argument("--stage", required=True, choices=("bc0", "dagger"))
        stage.add_argument("--dataset-receipt", required=True, type=Path)
        stage.add_argument("--train-file", required=True, type=Path)
        stage.add_argument("--validation-file", required=True, type=Path)
        stage.add_argument("--output-dir", required=True, type=Path)
        stage.add_argument("--hardware-attestation", required=True, type=Path)
        stage.add_argument("--stage-plan", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dataset":
        result = validate_preliminary_dataset_receipt(args.receipt)
    elif args.command == "stage-plan":
        plan_path = ensure_preliminary_stage_plan(
            stage=args.stage,
            dataset_receipt=args.dataset_receipt,
            train_file=args.train_file,
            validation_file=args.validation_file,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            initial_adapter=args.initial_adapter,
            training_seed=args.training_seed,
            max_train_rows=args.max_train_rows,
            max_valid_rows=args.max_valid_rows,
            max_steps=args.max_steps,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            dataloader_workers=args.dataloader_workers,
            report_to=args.report_to,
        )
        result = {"stage_plan_path": str(plan_path)}
    elif args.command == "resume-check":
        result = validate_preliminary_resume_checkpoint(
            stage_plan=args.stage_plan,
            checkpoint=args.checkpoint,
        )
    else:
        kwargs = {
            "stage": args.stage,
            "dataset_receipt": args.dataset_receipt,
            "train_file": args.train_file,
            "validation_file": args.validation_file,
            "output_dir": args.output_dir,
            "hardware_attestation": args.hardware_attestation,
            "stage_plan": args.stage_plan,
        }
        if args.command == "stage-write":
            receipt_path = write_preliminary_stage_receipt(**kwargs)
            result = {"receipt_path": str(receipt_path)}
        else:
            result = validate_preliminary_stage_receipt(**kwargs)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
