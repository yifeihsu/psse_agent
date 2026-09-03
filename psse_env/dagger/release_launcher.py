"""Fail-closed path validation for the BC0 HPC release launcher."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Callable, Iterable

import psse_env.dagger.evaluation_gate as evaluation_gate


def _resolved(repo_root: Path, value: str | Path, *, strict: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=strict)


def validate_release_evaluation_paths(
    *,
    repo_root: str | Path,
    mode: str,
    artifact: str | Path,
    report: str | Path,
    protected_inputs: Iterable[str | Path],
    reference_artifact: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, str]:
    """Validate that release outputs cannot overwrite inputs or base evidence.

    Release evaluation artifacts deliberately live under one fixed repository
    directory.  Gate reports are write-once publications and therefore must use
    an already-created canonical parent outside the source repository. Existing
    symlink aliases are resolved before comparison, and a checkpoint is
    forbidden from living inside the artifact evidence directory.
    """

    root = Path(repo_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"expert", "base", "checkpoint"}:
        raise ValueError(f"unsupported release evaluation mode: {mode!r}")

    output_root = (root / "artifacts" / "evaluations").resolve(strict=False)
    requested_artifact = Path(artifact).expanduser()
    if any(marker in str(requested_artifact) for marker in ("\r", "\n")):
        raise ValueError("evaluation artifact path contains a forbidden line break")
    if not requested_artifact.is_absolute():
        requested_artifact = root / requested_artifact
    if os.path.lexists(requested_artifact):
        raise ValueError("evaluation artifact already exists; refusing to overwrite it")
    artifact_path = requested_artifact.resolve(strict=False)
    requested_report = Path(report).expanduser()
    if any(marker in str(requested_report) for marker in ("\r", "\n")):
        raise ValueError("gate report path contains a forbidden line break")
    if not requested_report.is_absolute() or not requested_report.name:
        raise ValueError("gate report must be an absolute path naming a new file")
    report_parent = requested_report.parent.resolve(strict=True)
    if not report_parent.is_dir():
        raise ValueError("gate report parent is not a directory")
    report_path = report_parent / requested_report.name
    if os.path.lexists(report_path):
        raise ValueError("gate report already exists; refusing to overwrite it")
    try:
        report_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("gate report must be outside the source repository")

    outputs = {"artifact": artifact_path, "report": report_path}
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("evaluation artifact and gate report must be different paths")
    for label, path in {"artifact": artifact_path}.items():
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(
                f"{label} must remain under the fixed evidence directory {output_root}"
            ) from exc
        if path == output_root or (path.exists() and not path.is_file()):
            raise ValueError(f"{label} is not a regular evidence-file path: {path}")

    protected = {
        _resolved(root, value, strict=True) for value in protected_inputs
    }
    if normalized_mode == "checkpoint":
        if reference_artifact is None:
            raise ValueError("checkpoint mode requires a base reference artifact")
        reference = _resolved(root, reference_artifact, strict=True)
        if not reference.is_file():
            raise ValueError(f"base reference is not a regular file: {reference}")
        protected.add(reference)
    elif reference_artifact is not None:
        raise ValueError("only checkpoint mode may provide a comparison reference")

    collisions = {
        label: str(path) for label, path in outputs.items() if path in protected
    }
    if collisions:
        raise ValueError(f"release output collides with protected evidence: {collisions}")

    checkpoint: Path | None = None
    if normalized_mode == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("checkpoint mode requires checkpoint_path")
        checkpoint = _resolved(root, checkpoint_path, strict=True)
        if not checkpoint.is_dir():
            raise ValueError(f"checkpoint path is not a directory: {checkpoint}")
        try:
            checkpoint.relative_to(output_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "checkpoint input must not live inside the release evidence directory"
            )
        for label, path in outputs.items():
            try:
                path.relative_to(checkpoint)
            except ValueError:
                continue
            raise ValueError(f"{label} must not be written inside the checkpoint tree")
    elif checkpoint_path is not None:
        raise ValueError("only checkpoint mode may provide checkpoint_path")

    return {
        "artifact": str(outputs["artifact"]),
        "report": str(outputs["report"]),
        "output_root": str(output_root),
        "report_parent": str(report_parent),
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
    }


def publish_external_release_report(
    *,
    repo_root: str | Path,
    report: str | Path,
    rendered: str,
    protected_inputs: Iterable[str | Path | None],
    postpublication_revalidate: Callable[[], None],
) -> dict[str, str | int]:
    """Publish one immutable external report, then re-attest its evidence.

    The hardened evaluation-gate publisher owns the no-clobber, no-follow,
    fsync, byte-verification, and POSIX-0400 contract.  A caller-supplied
    re-attestation runs only after those bytes exist; failure removes only the
    exact inode just created and is never converted into a successful report.
    """

    root = Path(repo_root).expanduser().resolve(strict=True)
    target = evaluation_gate._prepare_report_output(
        report,
        repo_root=root,
        protected_inputs=tuple(protected_inputs),
    )
    identity = evaluation_gate._publish_new_report(target, rendered)
    payload = rendered.encode("utf-8")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        postpublication_revalidate()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
        try:
            observed = os.fstat(descriptor)
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != identity
            or size != len(payload)
            or digest.hexdigest() != expected_sha256
            or (os.name == "posix" and stat.S_IMODE(observed.st_mode) != 0o400)
        ):
            raise OSError("release report changed during post-publication re-attestation")
    except Exception as exc:
        removed = evaluation_gate._unlink_created_report(target, identity)
        message = (
            "release report post-publication re-attestation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        if not removed:
            message += "; the newly created report could not be safely removed"
        raise ValueError(message) from exc
    return {
        "path": str(target),
        "sha256": expected_sha256,
        "size_bytes": len(payload),
        "mode": "0400" if os.name == "posix" else "platform-read-only",
    }


__all__ = [
    "publish_external_release_report",
    "validate_release_evaluation_paths",
]
