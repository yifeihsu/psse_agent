"""Fail-closed path validation for the BC0 HPC release launcher."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


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
    directory.  Existing symlink aliases are resolved before comparison, and a
    checkpoint is forbidden from living inside that evidence directory.
    """

    root = Path(repo_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"base", "checkpoint"}:
        raise ValueError(f"unsupported release evaluation mode: {mode!r}")

    output_root = (root / "artifacts" / "evaluations").resolve(strict=False)
    outputs = {
        "artifact": _resolved(root, artifact, strict=False),
        "report": _resolved(root, report, strict=False),
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("evaluation artifact and gate report must be different paths")
    for label, path in outputs.items():
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
        raise ValueError("base mode must not provide a comparison reference")

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
        raise ValueError("base mode must not provide checkpoint_path")

    return {
        "artifact": str(outputs["artifact"]),
        "report": str(outputs["report"]),
        "output_root": str(output_root),
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
    }


__all__ = ["validate_release_evaluation_paths"]
