"""Fail-closed reuse of one approved historical expert evaluation closure.

The historical expert artifact was produced at an older source commit and was
subsequently accepted by a narrowly scoped dual-source validator.  A later SFT
freeze must not relabel that artifact as current-source evidence.  This module
instead authenticates the immutable gate receipt, authenticates a caller-held
copy of the exact artifact, and proves that the semantic evaluation sources are
unchanged in a clean, merge-free descendant of the receipt's validator.

The tracked manifest is the authority for an approved closure.  Callers do not
supply their own receipt digest or source pins.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.dagger.evaluation_gate import current_registry_sha256

from .provenance import git_source_state, stable_json_sha256


DEFAULT_HISTORICAL_EXPERT_CLOSURE_ID = (
    "bc0-expert-ef30899-validator-99b5c59-v1"
)
HISTORICAL_EXPERT_CLOSURE_MANIFEST = Path(
    "psse_env/sft/bc0_historical_expert_closures_v1.json"
)
HISTORICAL_EXPERT_CLOSURE_VALIDATION_CONTRACT = (
    "bc0_historical_expert_closure_validation_v1"
)

_MANIFEST_CONTRACT = "bc0_historical_expert_closure_registry_v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_BLOB = re.compile(r"[0-9a-f]{40,64}")
_PROTECTED_POLICY_PATH = "psse_env/dagger/bc0_evaluation_policy.json"
_PROTECTED_SUITE_PATH = "psse_env/dagger/suites/bc0_eval_suite_v1.json"
_DUAL_ATTESTATION_KEYS = {
    "artifact_is_ancestor",
    "artifact_source_commit",
    "contract",
    "gate_tree_sources",
    "git_replacements_disabled",
    "grafts_present",
    "history_commits",
    "history_path_hex",
    "history_paths",
    "merge_commits",
    "protected_blob_ids",
    "protected_sources",
    "replace_refs",
    "tracked_tree_matches_validator",
    "tree_delta",
    "validator_source_commit",
}


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    mode: int
    device: int
    inode: int
    size: int
    mtime_ns: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "mode": f"{self.mode:04o}",
            "size_bytes": self.size,
            "device": self.device,
            "inode": self.inode,
        }


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_path(
    value: str | Path,
    *,
    repo_root: Path,
    require_absolute: bool,
) -> Path:
    requested = Path(value).expanduser()
    if require_absolute and not requested.is_absolute():
        raise ValueError("historical evidence path must be absolute")
    if not requested.is_absolute():
        requested = repo_root / requested
    parent = requested.parent.resolve(strict=True)
    target = parent / requested.name
    if target.resolve(strict=True) != target:
        raise ValueError(f"evidence path contains a symlink component: {target}")
    return target


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _read_regular_snapshot(
    value: str | Path,
    *,
    repo_root: Path,
    require_absolute: bool = False,
    require_outside_repo: bool = False,
    required_posix_mode: int | None = None,
) -> _FileSnapshot:
    """Read one regular file once, binding its bytes to its opened identity."""

    target = _resolve_path(
        value,
        repo_root=repo_root,
        require_absolute=require_absolute,
    )
    if require_outside_repo and _is_within(target, repo_root):
        raise ValueError("historical evidence must be outside the consumer repository")
    before = os.lstat(target)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"historical evidence is not a regular file: {target}")
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"historical evidence cannot be a symlink: {target}")
    mode = stat.S_IMODE(before.st_mode)
    if os.name == "posix" and required_posix_mode is not None and mode != required_posix_mode:
        raise ValueError(
            f"historical receipt mode must be {required_posix_mode:04o}; got {mode:04o}"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"opened evidence is not a regular file: {target}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"historical evidence identity changed before read: {target}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)
    if size != opened.st_size:
        raise ValueError(f"historical evidence size changed during read: {target}")
    return _FileSnapshot(
        path=target,
        raw=b"".join(chunks),
        sha256=digest.hexdigest(),
        mode=stat.S_IMODE(opened.st_mode),
        device=opened.st_dev,
        inode=opened.st_ino,
        size=size,
        mtime_ns=opened.st_mtime_ns,
    )


def _snapshot_identity_unchanged(snapshot: _FileSnapshot) -> bool:
    try:
        observed = os.lstat(snapshot.path)
    except OSError:
        return False
    return (
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_dev == snapshot.device
        and observed.st_ino == snapshot.inode
        and stat.S_IMODE(observed.st_mode) == snapshot.mode
        and observed.st_size == snapshot.size
        and observed.st_mtime_ns == snapshot.mtime_ns
    )


def _decode_json_object(snapshot: _FileSnapshot, *, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(snapshot.raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {type(exc).__name__}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return decoded


def _git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    text: bool = False,
) -> bytes | str:
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "--no-replace-objects", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=text,
        env=environment,
    )
    return completed.stdout


def _git_lines(repo_root: Path, arguments: Sequence[str]) -> list[str]:
    output = _git(repo_root, arguments, text=True)
    if not isinstance(output, str):  # pragma: no cover - type narrowing
        raise TypeError("text Git command returned bytes")
    return output.splitlines()


def _tree_source_descriptor(
    repo_root: Path,
    commit: str,
    path: str,
) -> dict[str, str]:
    raw = _git(repo_root, ["ls-tree", "-z", commit, "--", path])
    if not isinstance(raw, bytes):  # pragma: no cover - type narrowing
        raise TypeError("binary Git command returned text")
    records = raw.split(b"\0")
    if len(records) != 2 or records[-1] != b"" or not records[0]:
        raise ValueError(f"Git tree does not contain exactly one entry for {path}")
    metadata, separator, raw_path = records[0].partition(b"\t")
    if separator != b"\t" or raw_path != path.encode("utf-8"):
        raise ValueError(f"Git tree returned a non-exact path for {path}")
    fields = metadata.split(b" ")
    if len(fields) != 3:
        raise ValueError(f"Git tree descriptor is malformed for {path}")
    mode, object_type, object_id = (field.decode("ascii") for field in fields)
    if mode != "100644" or object_type != "blob" or _GIT_BLOB.fullmatch(object_id) is None:
        raise ValueError(f"Git tree entry is not a regular 100644 blob for {path}")
    blob = _git(repo_root, ["cat-file", "blob", object_id])
    if not isinstance(blob, bytes):  # pragma: no cover - type narrowing
        raise TypeError("binary Git command returned text")
    return {
        "path": path,
        "mode": mode,
        "git_blob_id": object_id,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def _descriptor_core(value: object) -> dict[str, str]:
    descriptor = _mapping(value)
    return {
        "mode": str(descriptor.get("mode") or ""),
        "git_blob_id": str(descriptor.get("git_blob_id") or ""),
        "sha256": str(descriptor.get("sha256") or ""),
    }


def _range_paths(repo_root: Path, commits: Sequence[str]) -> list[str]:
    observed: set[str] = set()
    for commit in commits:
        raw = _git(
            repo_root,
            ["diff-tree", "--no-commit-id", "--name-only", "-r", "-z", commit],
        )
        if not isinstance(raw, bytes):  # pragma: no cover - type narrowing
            raise TypeError("binary Git command returned text")
        records = raw.split(b"\0")
        if records[-1] != b"":
            raise ValueError("Git history path stream is not NUL terminated")
        for record in records[:-1]:
            try:
                path = record.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError("Git history contains a non-UTF-8 path") from exc
            if not path or "\x00" in path:
                raise ValueError("Git history contains an invalid path")
            observed.add(path)
    return sorted(observed)


def _raw_tree_delta(
    repo_root: Path,
    old_commit: str,
    new_commit: str,
) -> list[dict[str, str]]:
    raw = _git(
        repo_root,
        ["diff", "--raw", "--no-abbrev", "-z", old_commit, new_commit, "--"],
    )
    if not isinstance(raw, bytes):  # pragma: no cover - type narrowing
        raise TypeError("binary Git command returned text")
    parts = raw.split(b"\0")
    if parts[-1] != b"":
        raise ValueError("raw Git delta is not NUL terminated")
    parts = parts[:-1]
    if len(parts) % 2:
        raise ValueError("raw Git delta has an incomplete path record")
    result: list[dict[str, str]] = []
    for offset in range(0, len(parts), 2):
        header = parts[offset]
        raw_path = parts[offset + 1]
        fields = header.split(b" ")
        if len(fields) != 5 or not fields[0].startswith(b":"):
            raise ValueError("raw Git delta descriptor is malformed")
        old_mode = fields[0][1:].decode("ascii")
        new_mode = fields[1].decode("ascii")
        old_blob = fields[2].decode("ascii")
        new_blob = fields[3].decode("ascii")
        status_code = fields[4].decode("ascii")
        try:
            path = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("raw Git delta contains a non-UTF-8 path") from exc
        if status_code != "M":
            raise ValueError(f"historical validator delta is not content-only M: {path}")
        if old_mode != "100644" or new_mode != "100644":
            raise ValueError(f"historical validator delta changed file mode: {path}")
        if old_blob == new_blob:
            raise ValueError(f"historical validator delta did not change blob: {path}")
        result.append(
            {
                "path": path,
                "path_hex": raw_path.hex(),
                "status": status_code,
                "old_mode": old_mode,
                "new_mode": new_mode,
                "old_blob_id": old_blob,
                "new_blob_id": new_blob,
            }
        )
    return result


def _grafts_present(repo_root: Path) -> bool:
    lines = _git_lines(repo_root, ["rev-parse", "--git-path", "info/grafts"])
    if len(lines) != 1 or not lines[0]:
        raise ValueError("Git did not return one graft path")
    path = Path(lines[0])
    if not path.is_absolute():
        path = repo_root / path
    try:
        return path.stat().st_size > 0
    except FileNotFoundError:
        return False


def _manifest_entry(
    payload: Mapping[str, Any],
    closure_id: str,
) -> Mapping[str, Any]:
    if payload.get("schema_version") != 1 or payload.get("contract") != _MANIFEST_CONTRACT:
        raise ValueError("historical expert closure manifest contract is not approved")
    closures = _mapping(payload.get("closures"))
    if set(closures) != {DEFAULT_HISTORICAL_EXPERT_CLOSURE_ID}:
        raise ValueError("historical expert closure manifest contains an unreviewed closure set")
    entry = _mapping(closures.get(closure_id))
    if not entry:
        raise ValueError(f"historical expert closure ID is not approved: {closure_id!r}")
    required = {"artifact", "evaluation", "protected_sources", "receipt", "validator"}
    if set(entry) != required:
        raise ValueError("historical expert closure entry has unexpected fields")
    receipt_sha = str(_mapping(entry["receipt"]).get("sha256") or "")
    artifact = _mapping(entry["artifact"])
    validator = _mapping(entry["validator"])
    evaluation = _mapping(entry["evaluation"])
    for label, value, pattern in (
        ("receipt SHA-256", receipt_sha, _SHA256),
        ("artifact SHA-256", str(artifact.get("sha256") or ""), _SHA256),
        (
            "artifact content SHA-256",
            str(artifact.get("content_sha256") or ""),
            _SHA256,
        ),
        ("artifact source commit", str(artifact.get("source_commit") or ""), _COMMIT),
        ("validator source commit", str(validator.get("source_commit") or ""), _COMMIT),
        (
            "evaluation registry SHA-256",
            str(evaluation.get("registry_sha256") or ""),
            _SHA256,
        ),
        ("evaluation suite SHA-256", str(evaluation.get("suite_sha256") or ""), _SHA256),
        (
            "evaluation policy SHA-256",
            str(evaluation.get("gate_policy_sha256") or ""),
            _SHA256,
        ),
    ):
        if pattern.fullmatch(value) is None:
            raise ValueError(f"historical expert closure has malformed {label}")
    protected = _mapping(entry["protected_sources"])
    validator_sources = _mapping(validator.get("validator_sources"))
    if set(protected) != {
        "psse_env/dagger/evaluator.py",
        "psse_env/dagger/release_factories.py",
        _PROTECTED_POLICY_PATH,
        _PROTECTED_SUITE_PATH,
    }:
        raise ValueError("historical expert closure protected-source set changed")
    expected_validator_paths = set(str(path) for path in validator.get("history_paths") or [])
    if set(validator_sources) != expected_validator_paths or len(validator_sources) != 2:
        raise ValueError("historical expert validator-source set changed")
    for path, raw_descriptor in {**protected, **validator_sources}.items():
        descriptor = _descriptor_core(raw_descriptor)
        if (
            descriptor["mode"] != "100644"
            or _GIT_BLOB.fullmatch(descriptor["git_blob_id"]) is None
            or _SHA256.fullmatch(descriptor["sha256"]) is None
        ):
            raise ValueError(f"historical expert source descriptor is malformed: {path}")
    return entry


def _source_attestation(
    *,
    repo_root: Path,
    manifest_snapshot: _FileSnapshot,
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    attestation: dict[str, Any] = {}
    try:
        source_state = git_source_state(repo_root)
        head_lines = _git_lines(repo_root, ["rev-parse", "HEAD"])
        if len(head_lines) != 1 or _COMMIT.fullmatch(head_lines[0]) is None:
            raise ValueError("consumer source HEAD is not one exact commit")
        head = head_lines[0]
        attestation["source_state"] = copy.deepcopy(source_state)
        attestation["source_commit"] = head
        if (
            source_state.get("release_eligible_source") is not True
            or source_state.get("source_commit") != head
        ):
            failures.append("consumer source is not one clean release-eligible commit")

        replace_refs = _git_lines(
            repo_root,
            ["for-each-ref", "--format=%(refname)", "refs/replace"],
        )
        grafts = _grafts_present(repo_root)
        attestation["replace_refs"] = replace_refs
        attestation["grafts_present"] = grafts
        attestation["git_replacements_disabled"] = True
        if replace_refs:
            failures.append("consumer repository contains Git replacement refs")
        if grafts:
            failures.append("consumer repository contains Git grafts")

        artifact = _mapping(entry["artifact"])
        validator = _mapping(entry["validator"])
        artifact_commit = str(artifact.get("source_commit") or "")
        validator_commit = str(validator.get("source_commit") or "")
        historical_commits = _git_lines(
            repo_root,
            ["rev-list", "--reverse", f"{artifact_commit}..{validator_commit}"],
        )
        expected_historical_commits = [
            str(value) for value in validator.get("history_commits") or []
        ]
        historical_merges = _git_lines(
            repo_root,
            ["rev-list", "--merges", f"{artifact_commit}..{validator_commit}"],
        )
        historical_paths = _range_paths(repo_root, historical_commits)
        expected_historical_paths = sorted(
            str(value) for value in validator.get("history_paths") or []
        )
        historical_delta = _raw_tree_delta(
            repo_root,
            artifact_commit,
            validator_commit,
        )
        attestation["historical_validator"] = {
            "artifact_is_ancestor": (
                subprocess.run(
                    [
                        "git",
                        "--no-replace-objects",
                        "merge-base",
                        "--is-ancestor",
                        artifact_commit,
                        validator_commit,
                    ],
                    cwd=repo_root,
                    check=False,
                    capture_output=True,
                    env={
                        **os.environ,
                        "GIT_NO_REPLACE_OBJECTS": "1",
                        "GIT_OPTIONAL_LOCKS": "0",
                    },
                ).returncode
                == 0
            ),
            "history_commits": historical_commits,
            "merge_commits": historical_merges,
            "history_paths": historical_paths,
            "tree_delta": historical_delta,
        }
        if not attestation["historical_validator"]["artifact_is_ancestor"]:
            failures.append("historical artifact commit is not an ancestor of its validator")
        if historical_commits != expected_historical_commits:
            failures.append("historical validator commit sequence changed")
        if historical_merges:
            failures.append("historical validator history contains a merge")
        if historical_paths != expected_historical_paths:
            failures.append("historical validator path closure changed")
        if (
            len(historical_delta) != len(expected_historical_paths)
            or {row["path"] for row in historical_delta}
            != set(expected_historical_paths)
        ):
            failures.append("historical validator raw tree delta changed")

        consumer_ancestor = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "merge-base",
                "--is-ancestor",
                validator_commit,
                head,
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            env={
                **os.environ,
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
            },
        ).returncode == 0
        consumer_commits = _git_lines(
            repo_root,
            ["rev-list", "--reverse", f"{validator_commit}..{head}"],
        )
        consumer_merges = _git_lines(
            repo_root,
            ["rev-list", "--merges", f"{validator_commit}..{head}"],
        )
        consumer_paths = _range_paths(repo_root, consumer_commits)
        attestation["consumer_history"] = {
            "validator_is_ancestor": consumer_ancestor,
            "commits": consumer_commits,
            "merge_commits": consumer_merges,
            "paths": consumer_paths,
        }
        if not consumer_ancestor or head == validator_commit:
            failures.append("consumer source is not a strict descendant of the validator")
        if consumer_merges:
            failures.append("consumer history after the validator contains a merge")

        protected = _mapping(entry["protected_sources"])
        validator_sources = _mapping(validator.get("validator_sources"))
        all_pins = {**protected, **validator_sources}
        changed_pins = sorted(set(consumer_paths) & set(all_pins))
        attestation["consumer_history"]["changed_pinned_paths"] = changed_pins
        if changed_pins:
            failures.append(
                "consumer history changed a closure-pinned source: "
                + ", ".join(changed_pins)
            )

        source_descriptors: dict[str, Any] = {}
        for path, manifest_descriptor_raw in sorted(all_pins.items()):
            manifest_descriptor = _descriptor_core(manifest_descriptor_raw)
            validator_descriptor = _tree_source_descriptor(
                repo_root,
                validator_commit,
                path,
            )
            consumer_descriptor = _tree_source_descriptor(repo_root, head, path)
            working_snapshot = _read_regular_snapshot(
                repo_root / path,
                repo_root=repo_root,
            )
            working_descriptor = {
                "path": path,
                "mode": f"{working_snapshot.mode:06o}",
                "sha256": working_snapshot.sha256,
            }
            source_descriptors[path] = {
                "manifest": manifest_descriptor,
                "validator": validator_descriptor,
                "consumer": consumer_descriptor,
                "working": working_descriptor,
            }
            if _descriptor_core(validator_descriptor) != manifest_descriptor:
                failures.append(f"validator tree source differs from closure pin: {path}")
            if _descriptor_core(consumer_descriptor) != manifest_descriptor:
                failures.append(f"consumer tree source differs from closure pin: {path}")
            if (
                (os.name == "posix" and working_snapshot.mode != 0o644)
                or working_snapshot.sha256 != manifest_descriptor["sha256"]
            ):
                failures.append(f"consumer working source differs from closure pin: {path}")
            if path in protected:
                artifact_descriptor = _tree_source_descriptor(
                    repo_root,
                    artifact_commit,
                    path,
                )
                source_descriptors[path]["artifact"] = artifact_descriptor
                if _descriptor_core(artifact_descriptor) != manifest_descriptor:
                    failures.append(f"artifact tree protected source differs from closure pin: {path}")
        attestation["pinned_sources"] = source_descriptors

        manifest_relative = HISTORICAL_EXPERT_CLOSURE_MANIFEST.as_posix()
        manifest_tree = _tree_source_descriptor(repo_root, head, manifest_relative)
        attestation["manifest_source"] = {
            "tree": manifest_tree,
            "working": manifest_snapshot.descriptor(),
        }
        if manifest_tree["sha256"] != manifest_snapshot.sha256:
            failures.append("closure manifest bytes differ from the consumer Git tree")
        if manifest_tree["mode"] != "100644" or (
            os.name == "posix" and manifest_snapshot.mode != 0o644
        ):
            failures.append("closure manifest is not a regular tracked 100644 source")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        failures.append(f"consumer source attestation failed: {type(exc).__name__}: {exc}")
    attestation["failures"] = _unique(failures)
    attestation["passed"] = not failures
    return attestation, _unique(failures)


def _validate_artifact(
    artifact: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    expected_artifact = _mapping(entry["artifact"])
    expected_evaluation = _mapping(entry["evaluation"])
    if artifact.get("content_sha256") != expected_artifact.get("content_sha256"):
        failures.append("expert artifact content SHA-256 differs from the approved closure")
    content_payload = dict(artifact)
    recorded_content = content_payload.pop("content_sha256", None)
    if recorded_content is not None and stable_json_sha256(content_payload) != recorded_content:
        failures.append("expert artifact content SHA-256 does not authenticate its payload")
    if artifact.get("release_eligible") is not True or artifact.get("release_failures") != []:
        failures.append("expert artifact is not release eligible")
    provenance = _mapping(artifact.get("provenance"))
    source_state = _mapping(provenance.get("source_state"))
    if (
        source_state.get("release_eligible_source") is not True
        or source_state.get("source_commit") != expected_artifact.get("source_commit")
    ):
        failures.append("expert artifact producer source differs from the approved closure")
    suite = _mapping(provenance.get("input_suite"))
    if suite.get("sha256") != expected_evaluation.get("suite_sha256"):
        failures.append("expert artifact suite differs from the approved closure")
    registry = _mapping(provenance.get("protocol_registry"))
    if (
        registry.get("protocol") != expected_evaluation.get("protocol")
        or registry.get("registry_sha256") != expected_evaluation.get("registry_sha256")
    ):
        failures.append("expert artifact protocol registry differs from the approved closure")
    if provenance.get("policy_identity") != expected_evaluation.get("policy_identity"):
        failures.append("expert artifact policy identity differs from the approved closure")
    return {
        "content_sha256": artifact.get("content_sha256"),
        "source_commit": source_state.get("source_commit"),
        "suite_sha256": suite.get("sha256"),
        "protocol": registry.get("protocol"),
        "registry_sha256": registry.get("registry_sha256"),
        "policy_identity": copy.deepcopy(provenance.get("policy_identity")),
        "release_eligible": artifact.get("release_eligible") is True,
    }, failures


def _validate_receipt(
    receipt: Mapping[str, Any],
    entry: Mapping[str, Any],
    source_attestation: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    expected_artifact = _mapping(entry["artifact"])
    expected_evaluation = _mapping(entry["evaluation"])
    expected_validator = _mapping(entry["validator"])

    expected_top = {
        "passed": True,
        "failures": [],
        "validation_role": expected_evaluation.get("role"),
        "evidence_passed": True,
        "performance_passed": True,
        "performance_enforced": True,
        "artifact_sha256": expected_artifact.get("sha256"),
        "artifact_content_sha256": expected_artifact.get("content_sha256"),
        "source_commit": expected_artifact.get("source_commit"),
        "frozen_suite_sha256": expected_evaluation.get("suite_sha256"),
        "protocol": expected_evaluation.get("protocol"),
        "registry_sha256": expected_evaluation.get("registry_sha256"),
        "evaluated_policy_identity": expected_evaluation.get("policy_identity"),
        "gate_policy_id": expected_evaluation.get("gate_policy_id"),
        "gate_policy_sha256": expected_evaluation.get("gate_policy_sha256"),
    }
    for field, expected in expected_top.items():
        if receipt.get(field) != expected:
            failures.append(f"historical expert receipt field changed: {field}")

    observed = _mapping(receipt.get("observed"))
    if observed.get("episodes") != expected_evaluation.get("episodes"):
        failures.append("historical expert receipt episode count changed")
    expected_safety = _mapping(expected_evaluation.get("safety_counts"))
    for field, expected in expected_safety.items():
        if observed.get(field) != expected:
            failures.append(f"historical expert receipt safety count changed: {field}")

    attestation = _mapping(receipt.get("validator_source_attestation"))
    expected_attestation = {
        "schema_version": 1,
        "contract": expected_validator.get("contract"),
        "artifact_source_commit": expected_artifact.get("source_commit"),
        "validator_source_commit": expected_validator.get("source_commit"),
        "artifact_is_ancestor": True,
        "git_replacements_disabled": True,
        "replace_refs": [],
        "grafts_present": False,
        "tracked_tree_matches_validator": True,
        "history_commits": expected_validator.get("history_commits"),
        "merge_commits": [],
        "history_paths": expected_validator.get("history_paths"),
        "history_path_hex": [
            str(path).encode("utf-8").hex()
            for path in expected_validator.get("history_paths") or []
        ],
        "current_source_enforced": True,
        "passed": True,
        "failures": [],
        "final_dual_source_failures": [],
    }
    for field, expected in expected_attestation.items():
        if attestation.get(field) != expected:
            failures.append(f"historical expert validator attestation changed: {field}")

    source_state = _mapping(attestation.get("current_source_state"))
    final_source_state = _mapping(attestation.get("final_source_state"))
    if (
        source_state.get("source_commit") != expected_validator.get("source_commit")
        or source_state.get("release_eligible_source") is not True
        or dict(source_state) != dict(final_source_state)
    ):
        failures.append("historical expert validator source state is not exact and stable")

    historical = _mapping(source_attestation.get("historical_validator"))
    receipt_delta = attestation.get("tree_delta")
    if receipt_delta != historical.get("tree_delta"):
        failures.append("historical expert receipt raw tree delta differs from live Git")
    gate_sources = _mapping(attestation.get("gate_tree_sources"))
    protected_sources = _mapping(attestation.get("protected_sources"))
    protected_ids = _mapping(attestation.get("protected_blob_ids"))
    pinned = _mapping(source_attestation.get("pinned_sources"))
    expected_validator_sources = _mapping(expected_validator.get("validator_sources"))
    expected_protected = _mapping(entry.get("protected_sources"))

    for path, expected_raw in expected_validator_sources.items():
        endpoints = _mapping(gate_sources.get(path))
        validator_descriptor = _descriptor_core(endpoints.get("validator"))
        artifact_descriptor = _descriptor_core(endpoints.get("artifact"))
        live = _mapping(pinned.get(path))
        if validator_descriptor != _descriptor_core(expected_raw):
            failures.append(f"historical receipt validator source changed: {path}")
        if validator_descriptor != _descriptor_core(live.get("validator")):
            failures.append(f"historical receipt validator source differs from live Git: {path}")
        live_delta = {
            row.get("path"): row for row in historical.get("tree_delta") or []
        }.get(path, {})
        if (
            artifact_descriptor.get("git_blob_id") != live_delta.get("old_blob_id")
            or validator_descriptor.get("git_blob_id") != live_delta.get("new_blob_id")
        ):
            failures.append(f"historical receipt endpoint blobs differ from raw delta: {path}")

    for path, expected_raw in expected_protected.items():
        endpoints = _mapping(protected_sources.get(path))
        artifact_descriptor = _descriptor_core(endpoints.get("artifact"))
        validator_descriptor = _descriptor_core(endpoints.get("validator"))
        expected_descriptor = _descriptor_core(expected_raw)
        if artifact_descriptor != expected_descriptor or validator_descriptor != expected_descriptor:
            failures.append(f"historical receipt protected source changed: {path}")
        if protected_ids.get(path) != expected_descriptor.get("git_blob_id"):
            failures.append(f"historical receipt protected blob ID changed: {path}")

    validator_gate = _mapping(attestation.get("validator_gate_source"))
    gate_path = "psse_env/dagger/evaluation_gate.py"
    if (
        str(validator_gate.get("sha256") or "")
        != _descriptor_core(expected_validator_sources.get(gate_path))["sha256"]
    ):
        failures.append("historical receipt executing validator gate source changed")

    final_dual = _mapping(attestation.get("final_dual_source_attestation"))
    if set(final_dual) != _DUAL_ATTESTATION_KEYS:
        failures.append("historical receipt final dual-source attestation is incomplete")
    else:
        for field in _DUAL_ATTESTATION_KEYS:
            if final_dual.get(field) != attestation.get(field):
                failures.append(
                    f"historical receipt final dual-source re-attestation changed: {field}"
                )

    return {
        "validation_role": receipt.get("validation_role"),
        "passed": receipt.get("passed") is True,
        "evidence_passed": receipt.get("evidence_passed") is True,
        "performance_passed": receipt.get("performance_passed") is True,
        "performance_enforced": receipt.get("performance_enforced") is True,
        "producer_artifact_path": receipt.get("artifact_path"),
        "artifact_sha256": receipt.get("artifact_sha256"),
        "artifact_content_sha256": receipt.get("artifact_content_sha256"),
        "artifact_source_commit": receipt.get("source_commit"),
        "validator_source_commit": attestation.get("validator_source_commit"),
        "validator_contract": attestation.get("contract"),
        "gate_policy_id": receipt.get("gate_policy_id"),
        "gate_policy_sha256": receipt.get("gate_policy_sha256"),
        "frozen_suite_sha256": receipt.get("frozen_suite_sha256"),
        "protocol": receipt.get("protocol"),
        "registry_sha256": receipt.get("registry_sha256"),
        "evaluated_policy_identity": copy.deepcopy(
            receipt.get("evaluated_policy_identity")
        ),
        "episodes": observed.get("episodes"),
        "safety_counts": {
            field: observed.get(field) for field in expected_safety
        },
    }, failures


def _failure_payload(
    *,
    closure_id: str,
    failures: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract": HISTORICAL_EXPERT_CLOSURE_VALIDATION_CONTRACT,
        "closure_id": closure_id,
        "passed": False,
        "failures": _unique(failures),
        "manifest": {},
        "receipt": {},
        "artifact": {},
        "expert": {},
        "consumer_source_attestation": {
            "initial": {},
            "final": {},
            "unchanged": False,
        },
    }


def validate_historical_expert_closure(
    receipt_path: str | Path,
    *,
    expert_artifact_path: str | Path,
    repo_root: str | Path,
    expected_suite_path: str | Path,
    expected_policy_path: str | Path,
    expected_policy_identity: str,
    expected_protocol: str = "canonical",
    expected_registry_sha256: str | None = None,
    closure_id: str = DEFAULT_HISTORICAL_EXPERT_CLOSURE_ID,
) -> dict[str, Any]:
    """Authenticate an approved expert receipt for a clean descendant freeze.

    The returned mapping is JSON-ready and never promotes the historical
    artifact's producer or validator commit to the consumer commit.
    """

    failures: list[str] = []
    try:
        root = Path(repo_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("consumer repository root is not a directory")
    except (OSError, ValueError) as exc:
        return _failure_payload(
            closure_id=closure_id,
            failures=[f"consumer repository is unavailable: {type(exc).__name__}: {exc}"],
        )

    manifest_snapshot: _FileSnapshot | None = None
    entry: Mapping[str, Any] = {}
    try:
        manifest_snapshot = _read_regular_snapshot(
            root / HISTORICAL_EXPERT_CLOSURE_MANIFEST,
            repo_root=root,
        )
        manifest_payload = _decode_json_object(
            manifest_snapshot,
            label="historical expert closure manifest",
        )
        entry = _manifest_entry(manifest_payload, closure_id)
    except (OSError, ValueError) as exc:
        return _failure_payload(
            closure_id=closure_id,
            failures=[f"closure manifest validation failed: {type(exc).__name__}: {exc}"],
        )

    if manifest_snapshot is None:  # pragma: no cover - guarded above
        return _failure_payload(
            closure_id=closure_id,
            failures=["closure manifest snapshot is missing"],
        )

    initial_source, source_failures = _source_attestation(
        repo_root=root,
        manifest_snapshot=manifest_snapshot,
        entry=entry,
    )
    failures.extend(source_failures)

    expected_evaluation = _mapping(entry["evaluation"])
    actual_registry: str | None = None
    try:
        actual_registry = current_registry_sha256("canonical")
    except (OSError, ValueError, RuntimeError) as exc:
        failures.append(
            f"current canonical registry is unavailable: {type(exc).__name__}: {exc}"
        )
    if actual_registry != expected_evaluation.get("registry_sha256"):
        failures.append("current canonical registry differs from the approved closure")
    if (
        expected_registry_sha256 is not None
        and expected_registry_sha256 != actual_registry
    ):
        failures.append("caller's expected registry differs from the current registry")
    if expected_protocol != expected_evaluation.get("protocol"):
        failures.append("caller's expected protocol differs from the approved closure")
    policy_identity = _mapping(expected_evaluation.get("policy_identity"))
    if expected_policy_identity != policy_identity.get("explicit_policy_identity"):
        failures.append("caller's expert policy identity differs from the approved closure")

    try:
        suite = _resolve_path(
            expected_suite_path,
            repo_root=root,
            require_absolute=False,
        )
        policy = _resolve_path(
            expected_policy_path,
            repo_root=root,
            require_absolute=False,
        )
        if suite != root / _PROTECTED_SUITE_PATH:
            failures.append("caller's evaluation suite is not the closure-pinned suite")
        if policy != root / _PROTECTED_POLICY_PATH:
            failures.append("caller's evaluation policy is not the closure-pinned policy")
    except (OSError, ValueError) as exc:
        failures.append(f"closure input path validation failed: {type(exc).__name__}: {exc}")

    receipt_snapshot: _FileSnapshot | None = None
    artifact_snapshot: _FileSnapshot | None = None
    receipt_facts: dict[str, Any] = {}
    artifact_facts: dict[str, Any] = {}
    try:
        receipt_snapshot = _read_regular_snapshot(
            receipt_path,
            repo_root=root,
            require_absolute=True,
            require_outside_repo=True,
            required_posix_mode=0o400,
        )
        if receipt_snapshot.sha256 != _mapping(entry["receipt"]).get("sha256"):
            failures.append("historical expert receipt SHA-256 differs from the approved closure")
        receipt_payload = _decode_json_object(
            receipt_snapshot,
            label="historical expert gate receipt",
        )
        receipt_facts, receipt_failures = _validate_receipt(
            receipt_payload,
            entry,
            initial_source,
        )
        failures.extend(receipt_failures)
    except (OSError, ValueError) as exc:
        failures.append(
            f"historical expert receipt validation failed: {type(exc).__name__}: {exc}"
        )

    try:
        artifact_snapshot = _read_regular_snapshot(
            expert_artifact_path,
            repo_root=root,
            require_absolute=True,
            require_outside_repo=True,
        )
        if receipt_snapshot is not None and artifact_snapshot.path == receipt_snapshot.path:
            failures.append("expert artifact aliases the historical gate receipt")
        if artifact_snapshot.sha256 != _mapping(entry["artifact"]).get("sha256"):
            failures.append("expert artifact file SHA-256 differs from the approved closure")
        artifact_payload = _decode_json_object(
            artifact_snapshot,
            label="historical expert artifact",
        )
        artifact_facts, artifact_failures = _validate_artifact(
            artifact_payload,
            entry,
        )
        failures.extend(artifact_failures)
    except (OSError, ValueError) as exc:
        failures.append(
            f"historical expert artifact validation failed: {type(exc).__name__}: {exc}"
        )

    if receipt_facts and artifact_facts:
        for receipt_field, artifact_field in (
            ("artifact_content_sha256", "content_sha256"),
            ("artifact_source_commit", "source_commit"),
            ("frozen_suite_sha256", "suite_sha256"),
            ("protocol", "protocol"),
            ("registry_sha256", "registry_sha256"),
            ("evaluated_policy_identity", "policy_identity"),
        ):
            if receipt_facts.get(receipt_field) != artifact_facts.get(artifact_field):
                failures.append(
                    "historical receipt and caller-held artifact disagree: "
                    + receipt_field
                )

    if receipt_snapshot is not None and not _snapshot_identity_unchanged(receipt_snapshot):
        failures.append("historical expert receipt identity changed during validation")
    if artifact_snapshot is not None and not _snapshot_identity_unchanged(artifact_snapshot):
        failures.append("historical expert artifact identity changed during validation")

    try:
        final_manifest_snapshot = _read_regular_snapshot(
            root / HISTORICAL_EXPERT_CLOSURE_MANIFEST,
            repo_root=root,
        )
        final_source, final_source_failures = _source_attestation(
            repo_root=root,
            manifest_snapshot=final_manifest_snapshot,
            entry=entry,
        )
    except (OSError, ValueError) as exc:
        final_source = {}
        final_source_failures = [
            f"final consumer source re-attestation failed: {type(exc).__name__}: {exc}"
        ]
    failures.extend(final_source_failures)
    source_unchanged = bool(final_source) and final_source == initial_source
    if not source_unchanged:
        failures.append("consumer source changed during historical closure validation")

    try:
        final_registry = current_registry_sha256("canonical")
    except (OSError, ValueError, RuntimeError) as exc:
        final_registry = None
        failures.append(
            f"final canonical registry re-attestation failed: {type(exc).__name__}: {exc}"
        )
    if final_registry != actual_registry:
        failures.append("canonical registry changed during historical closure validation")

    unique_failures = _unique(failures)
    return {
        "schema_version": 1,
        "contract": HISTORICAL_EXPERT_CLOSURE_VALIDATION_CONTRACT,
        "closure_id": closure_id,
        "passed": not unique_failures,
        "failures": unique_failures,
        "manifest": {
            **manifest_snapshot.descriptor(),
            "contract": _MANIFEST_CONTRACT,
        },
        "receipt": (
            receipt_snapshot.descriptor() if receipt_snapshot is not None else {}
        ),
        "artifact": (
            {
                **artifact_snapshot.descriptor(),
                **artifact_facts,
            }
            if artifact_snapshot is not None
            else {}
        ),
        "expert": receipt_facts,
        "consumer_source_attestation": {
            "initial": initial_source,
            "final": final_source,
            "unchanged": source_unchanged,
        },
    }


__all__ = [
    "DEFAULT_HISTORICAL_EXPERT_CLOSURE_ID",
    "HISTORICAL_EXPERT_CLOSURE_MANIFEST",
    "HISTORICAL_EXPERT_CLOSURE_VALIDATION_CONTRACT",
    "validate_historical_expert_closure",
]
