"""Fail-closed builder and receipt gates for the 2026-08-27 screening cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT = "research_occupancy_screening_cell_v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
EXPECTED_ENVIRONMENT = {
    "python": "3.12.12",
    "torch": "2.10.0",
    "transformers": "5.15.1",
    "trl": "1.10.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.49.2",
    "datasets": "5.0.1",
}
MODELS = {
    "e2b": (
        "unsloth/gemma-4-E2B-it",
        "f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae",
    ),
    "12b": (
        "unsloth/gemma-4-12B-it",
        "55cdba0740a9765956f49501f689a66b098feda3",
    ),
}
ARMS = {
    "A": {"lane": "e2b", "inclusion": "learner_full", "updates": 666},
    "B": {"lane": "12b", "inclusion": "learner_full", "updates": 662},
    "C": {
        "lane": "e2b",
        "inclusion": "full_occupancy",
        "updates": 1811,
    },
    "D": {"lane": "12b", "inclusion": "selective", "updates": 247},
}
EXPECTED_LEARNER_FULL = {
    "A": {
        "round1_rows_collected": 3696,
        "supervisable_rows": 3695,
        "teacher_abstention_rows": 49,
        "round1_audited_rows": 3646,
        "round1_rows_selected": 2323,
        "invalid_learner_action_rows_retained": 11,
        "round1_rows_added_to_train": 1870,
        "round1_rows_added_to_validation": 453,
        "round1_split_universe_episodes": 333,
        "train_rows": 2566,
        "validation_rows": 656,
    },
    "B": {
        "round1_rows_collected": 3628,
        "supervisable_rows": 3619,
        "teacher_abstention_rows": 18,
        "round1_audited_rows": 3601,
        "round1_rows_selected": 2283,
        "invalid_learner_action_rows_retained": 13,
        "round1_rows_added_to_train": 1837,
        "round1_rows_added_to_validation": 446,
        "round1_split_universe_episodes": 333,
        "train_rows": 2533,
        "validation_rows": 649,
    },
}
INPUT_KEYS = {
    "e2b_round0_train",
    "e2b_round0_validation",
    "e2b_round1_rows",
    "12b_round0_train",
    "12b_round0_validation",
    "12b_round1_rows",
    "e2b_legacy_full_train",
    "e2b_legacy_full_validation",
    "12b_legacy_selective_train",
    "12b_legacy_selective_validation",
    "12b_legacy_selective_receipt",
    "evaluation_scenarios",
}
TREE_EXCLUDES = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    actual = {"python": platform.python_version()}
    for distribution in EXPECTED_ENVIRONMENT:
        if distribution != "python":
            actual[distribution] = importlib.metadata.version(distribution)
    if actual != EXPECTED_ENVIRONMENT:
        raise ValueError(
            f"research environment drift: expected {EXPECTED_ENVIRONMENT}, got {actual}"
        )
    expected_hf_home = Path(config["hf_home"]).resolve(strict=True)
    configured_hf_home = Path(os.environ.get("HF_HOME", "")).resolve()
    if configured_hf_home != expected_hf_home:
        raise ValueError(
            f"HF_HOME must be the configured historical cache {expected_hf_home}, "
            f"got {configured_hf_home}"
        )
    return {
        "versions": actual,
        "python_executable": str(Path(sys.executable).resolve()),
        "hf_home": str(expected_hf_home),
        "validation": "exact_distribution_versions",
    }


def snapshot_manifest(hf_home: str | Path, lane: str) -> dict[str, Any]:
    """Hash every file exposed by one pinned Hub snapshot.

    Hugging Face snapshots normally consist of symlinks into the repository's
    content-addressed blob directory.  Resolve each entry strictly so broken
    links fail, require every target to remain inside the configured cache,
    and hash the actual bytes rather than trusting directory or blob names.
    """

    if lane not in MODELS:
        raise ValueError(f"unknown model lane: {lane}")
    cache = Path(hf_home).resolve(strict=True)
    model_id, revision = MODELS[lane]
    repository = "models--" + model_id.replace("/", "--")
    snapshot = cache / "hub" / repository / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(f"missing pinned {lane} model snapshot: {snapshot}")

    entries: list[tuple[str, int, str]] = []
    for entry in sorted(snapshot.rglob("*"), key=lambda path: path.relative_to(snapshot).as_posix()):
        relative = entry.relative_to(snapshot).as_posix()
        if entry.is_symlink():
            try:
                target = entry.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(f"broken pinned snapshot entry: {relative}") from exc
            if not target.is_file():
                raise ValueError(f"pinned snapshot entry is not a file: {relative}")
        elif entry.is_file():
            target = entry.resolve(strict=True)
        elif entry.is_dir():
            continue
        else:
            raise ValueError(f"unsupported pinned snapshot entry: {relative}")
        if not target.is_relative_to(cache):
            raise ValueError(f"pinned snapshot entry escapes HF_HOME: {relative}")
        entries.append((relative, target.stat().st_size, sha256(target)))
    if not entries:
        raise ValueError(f"pinned {lane} model snapshot is empty: {snapshot}")
    names = {name for name, _, _ in entries}
    if "config.json" not in names:
        raise ValueError(f"pinned {lane} model snapshot lacks config.json")
    if not any(
        name == "model.safetensors"
        or name == "model.safetensors.index.json"
        or name.startswith("model-") and name.endswith(".safetensors")
        for name in names
    ):
        raise ValueError(f"pinned {lane} model snapshot lacks model weights")

    digest = hashlib.sha256(b"occupancy-cell-hf-snapshot-v1\0")
    total_bytes = 0
    for relative, size, file_sha in entries:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_sha))
        total_bytes += size
    return {
        "lane": lane,
        "model_id": model_id,
        "revision": revision,
        "snapshot": str(snapshot),
        "sha256": digest.hexdigest(),
        "files": len(entries),
        "bytes": total_bytes,
        "validation": "all_snapshot_entries_resolved_and_content_hashed",
    }


def verify_model_snapshot(config: Mapping[str, Any], lane: str) -> dict[str, Any]:
    manifest = snapshot_manifest(config["hf_home"], lane)
    expected = config["model_snapshot_sha256"][lane]
    if manifest["sha256"] != expected:
        raise ValueError(
            f"pinned {lane} model snapshot hash mismatch: "
            f"expected {expected}, got {manifest['sha256']}"
        )
    return manifest


def tree_digest(root: str | Path) -> str:
    base = Path(root).resolve(strict=True)
    digest = hashlib.sha256(b"occupancy-cell-source-tree-v1\0")
    files: list[Path] = []
    for path in base.rglob("*"):
        relative = path.relative_to(base)
        if any(part in TREE_EXCLUDES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"source tree contains symlink: {relative}")
        if path.is_file() and path.suffix != ".pyc":
            files.append(path)
    for path in sorted(files, key=lambda value: value.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return value


def _absolute(value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or any(char in str(path) for char in "\n\r\t,"):
        raise ValueError(f"{label} must be an absolute export-safe path")
    return path


def validate_config(path: str | Path, expected_sha: str | None = None) -> tuple[dict[str, Any], str]:
    config_path = Path(path).resolve(strict=True)
    config_sha = sha256(config_path)
    if expected_sha and config_sha != expected_sha.lower():
        raise ValueError("configuration SHA-256 changed after submission")
    config = load_json(config_path)
    if config.get("contract") != CONTRACT:
        raise ValueError(f"config contract must be {CONTRACT!r}")
    if not HEX40.fullmatch(str(config.get("source_commit", "")).lower()):
        raise ValueError("source_commit must be a 40-character commit")
    source = _absolute(config.get("source_root"), "source_root").resolve(strict=True)
    root = _absolute(config.get("cell_root"), "cell_root").resolve()
    python = _absolute(config.get("python"), "python")
    python.resolve(strict=True)
    if not os.access(python, os.X_OK):
        raise ValueError(f"configured Python is not executable: {python}")
    hf_home = _absolute(config.get("hf_home"), "hf_home").resolve(strict=True)
    snapshot_hashes = config.get("model_snapshot_sha256")
    if not isinstance(snapshot_hashes, dict) or set(snapshot_hashes) != set(MODELS):
        raise ValueError(
            f"model_snapshot_sha256 must contain exactly {sorted(MODELS)}"
        )
    for lane, digest in snapshot_hashes.items():
        if not HEX64.fullmatch(str(digest).lower()):
            raise ValueError(f"model_snapshot_sha256.{lane} must be a SHA-256")
        snapshot_hashes[lane] = str(digest).lower()
    expected_tree = str(config.get("source_tree_sha256", "")).lower()
    if not HEX64.fullmatch(expected_tree) or tree_digest(source) != expected_tree:
        raise ValueError("source tree SHA-256 mismatch")
    if root == source or root.is_relative_to(source):
        raise ValueError("cell_root must not be inside source_root")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != INPUT_KEYS:
        raise ValueError(f"inputs must contain exactly {sorted(INPUT_KEYS)}")
    for name, record in inputs.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"inputs.{name} is not an object")
        item = _absolute(record.get("path"), f"inputs.{name}.path").resolve(strict=True)
        expected = str(record.get("sha256", "")).lower()
        if not HEX64.fullmatch(expected) or sha256(item) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    receipt = load_json(inputs["12b_legacy_selective_receipt"]["path"])
    if receipt.get("inclusion") != "selective":
        raise ValueError("12b_legacy_selective_receipt does not attest inclusion=selective")
    for split, rows in (("train", 1332), ("validation", 331)):
        receipt_source = _absolute(
            receipt.get(split), f"12b_legacy_selective_receipt.{split}"
        ).resolve(strict=True)
        configured = inputs[f"12b_legacy_selective_{split}"]
        if sha256(receipt_source) != configured["sha256"]:
            raise ValueError(
                f"12B selective receipt {split} is not bound to the configured data"
            )
        if receipt.get(f"{split}_rows") != rows:
            raise ValueError(f"12B selective receipt {split} row-count drift")
    sys.path.insert(0, str(source))
    from research.collect import load_scenarios

    scenarios = load_scenarios(inputs["evaluation_scenarios"]["path"])
    if len(scenarios) != 65:
        raise ValueError(f"evaluation suite must contain exactly 65 scenarios, got {len(scenarios)}")
    config["source_root"] = str(source)
    config["cell_root"] = str(root)
    config["python"] = str(python)
    config["hf_home"] = str(hf_home)
    return config, config_sha


def input_path(config: Mapping[str, Any], key: str) -> Path:
    return Path(config["inputs"][key]["path"]).resolve(strict=True)


def _row_digest(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_e2b_legacy_full(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that arm C is D0 plus every audited E2B Round-1 row.

    The historical E2B full-occupancy directory did not retain its original
    aggregate report.  A new self-asserted receipt would not repair that gap,
    so the screening build reconstructs the expected D1 chat rows and compares
    their complete multiset with the immutable legacy train/validation tails.
    """

    from psse_env.dagger.dataset_builder import examples_to_chat_sft

    from research.train import read_jsonl

    base_train = read_jsonl(input_path(config, "e2b_round0_train"))
    base_validation = read_jsonl(input_path(config, "e2b_round0_validation"))
    legacy_train = read_jsonl(input_path(config, "e2b_legacy_full_train"))
    legacy_validation = read_jsonl(input_path(config, "e2b_legacy_full_validation"))
    if len(legacy_train) != 3630 or len(legacy_validation) != 915:
        raise ValueError("E2B legacy full aggregate row-count drift")
    if legacy_train[: len(base_train)] != base_train:
        raise ValueError("E2B legacy full train does not preserve the D0 prefix")
    if legacy_validation[: len(base_validation)] != base_validation:
        raise ValueError("E2B legacy full validation does not preserve the D0 prefix")

    raw_rows = read_jsonl(input_path(config, "e2b_round1_rows"))
    supervisable = [row for row in raw_rows if row.get("preferred_action")]
    audited = [
        row
        for row in supervisable
        if (row.get("offline_teacher_target_audit") or {}).get("passed")
    ]
    if (len(raw_rows), len(supervisable), len(audited)) != (3696, 3695, 3646):
        raise ValueError("E2B source occupancy count drift")
    expected = examples_to_chat_sft(
        audited,
        protocol="canonical",
        require_derived_provenance=False,
        allow_ineligible_auxiliary=True,
    )
    added_train = legacy_train[len(base_train) :]
    added_validation = legacy_validation[len(base_validation) :]
    observed = [*added_train, *added_validation]
    if Counter(map(_row_digest, observed)) != Counter(map(_row_digest, expected)):
        raise ValueError("E2B legacy full D1 rows do not equal all audited occupancy")

    def roots(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        values = {
            str(row.get("root_scenario_id") or row.get("scenario_id") or "")
            for row in rows
        }
        if "" in values:
            raise ValueError("E2B legacy full row lacks physical-root identity")
        return values

    train_roots = roots(added_train)
    validation_roots = roots(added_validation)
    if train_roots & validation_roots:
        raise ValueError("E2B legacy full split leaks physical roots")
    if len(train_roots | validation_roots) != 333 or len(validation_roots) != 66:
        raise ValueError("E2B legacy full root assignment drift")
    origins = Counter(str(row.get("state_origin")) for row in audited)
    if origins != {"initial": 333, "learner_policy": 2323, "expert_policy": 990}:
        raise ValueError(f"E2B legacy full origin drift: {dict(origins)}")
    return {
        "validation_method": "reconstructed_complete_audited_d1_multiset_v1",
        "inclusion": "full_occupancy",
        "raw_rows": len(raw_rows),
        "supervisable_rows": len(supervisable),
        "audited_rows": len(audited),
        "d1_train_rows": len(added_train),
        "d1_validation_rows": len(added_validation),
        "train_root_count": len(train_roots),
        "validation_root_count": len(validation_roots),
        "state_origin_breakdown": dict(sorted(origins.items())),
    }


def _validate_12b_legacy_selective(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct and bind the legacy selective aggregate used by arm D."""

    from psse_env.dagger.dataset_builder import examples_to_chat_sft

    from research.train import read_jsonl

    base_train = read_jsonl(input_path(config, "12b_round0_train"))
    base_validation = read_jsonl(input_path(config, "12b_round0_validation"))
    legacy_train = read_jsonl(input_path(config, "12b_legacy_selective_train"))
    legacy_validation = read_jsonl(
        input_path(config, "12b_legacy_selective_validation")
    )
    if len(legacy_train) != 1332 or len(legacy_validation) != 331:
        raise ValueError("12B legacy selective aggregate row-count drift")
    if legacy_train[: len(base_train)] != base_train:
        raise ValueError("12B legacy selective train does not preserve the D0 prefix")
    if legacy_validation[: len(base_validation)] != base_validation:
        raise ValueError(
            "12B legacy selective validation does not preserve the D0 prefix"
        )

    raw_rows = read_jsonl(input_path(config, "12b_round1_rows"))
    supervisable = [row for row in raw_rows if row.get("preferred_action")]
    audited = [
        row
        for row in supervisable
        if (row.get("offline_teacher_target_audit") or {}).get("passed")
    ]
    selected = [row for row in audited if row.get("production_label_eligible")]
    if (len(raw_rows), len(supervisable), len(audited), len(selected)) != (
        3628,
        3619,
        3601,
        764,
    ):
        raise ValueError("12B selective source occupancy count drift")
    expected = examples_to_chat_sft(
        selected,
        protocol="canonical",
        require_derived_provenance=False,
        allow_ineligible_auxiliary=False,
    )
    added_train = legacy_train[len(base_train) :]
    added_validation = legacy_validation[len(base_validation) :]
    if Counter(map(_row_digest, [*added_train, *added_validation])) != Counter(
        map(_row_digest, expected)
    ):
        raise ValueError("12B legacy selective D1 rows do not match reconstructed rows")

    def roots(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        values = {
            str(row.get("root_scenario_id") or row.get("scenario_id") or "")
            for row in rows
        }
        if "" in values:
            raise ValueError("12B legacy selective row lacks physical-root identity")
        return values

    train_roots = roots(added_train)
    validation_roots = roots(added_validation)
    if train_roots & validation_roots:
        raise ValueError("12B legacy selective split leaks physical roots")
    receipt = load_json(input_path(config, "12b_legacy_selective_receipt"))
    recorded_validation = set(receipt.get("round1_validation_episodes") or [])
    if (
        len(train_roots | validation_roots) != 210
        or len(validation_roots) != 42
        or validation_roots != recorded_validation
    ):
        raise ValueError("12B legacy selective root assignment drift")
    origins = Counter(str(row.get("state_origin")) for row in selected)
    invalid = sum(
        (row.get("executed_action") or {}).get("tool") == "__invalid_action__"
        for row in selected
    )
    if origins != {"learner_policy": 764} or invalid != 8:
        raise ValueError("12B legacy selective retention semantics drift")
    return {
        "validation_method": "reconstructed_selective_d1_multiset_v1",
        "inclusion": "selective",
        "raw_rows": len(raw_rows),
        "supervisable_rows": len(supervisable),
        "audited_rows": len(audited),
        "selected_rows": len(selected),
        "retained_invalid_learner_actions": invalid,
        "d1_train_rows": len(added_train),
        "d1_validation_rows": len(added_validation),
        "train_root_count": len(train_roots),
        "validation_root_count": len(validation_roots),
        "state_origin_breakdown": dict(origins),
    }


def _validate_exposure(report: Mapping[str, Any], train: Path, validation: Path, lane: str) -> None:
    model_id, revision = MODELS[lane]
    if (report.get("model_id"), report.get("model_revision")) != (model_id, revision):
        raise ValueError(f"{lane} exposure used the wrong model identity")
    for split, path in (("train", train), ("validation", validation)):
        item = report.get(split)
        if not isinstance(item, Mapping):
            raise ValueError(f"missing {split} exposure")
        if item.get("sha256") != sha256(path):
            raise ValueError(f"{lane} {split} exposure hash mismatch")
        if item.get("preparation_failure_count") != 0:
            raise ValueError(f"{lane} {split} has preparation failures")
        prepared = item.get("prepared") or {}
        if prepared.get("rows") != item.get("input_rows") or not prepared.get("rows"):
            raise ValueError(f"{lane} {split} exposure is incomplete or empty")


def build(config_path: Path, expected_sha: str, attempt: Path) -> dict[str, Any]:
    config, config_sha = validate_config(config_path, expected_sha)
    environment = environment_receipt(config)
    environment["model_snapshots"] = {
        lane: verify_model_snapshot(config, lane) for lane in MODELS
    }
    cell_root = Path(config["cell_root"])
    attempt = attempt.resolve()
    if not attempt.is_relative_to(cell_root / "build"):
        raise ValueError("build attempt must be under <cell_root>/build")
    attempt.mkdir(parents=True, exist_ok=False)
    source = Path(config["source_root"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from research.exposure import summarize_dataset
    from research.model import load_processor
    from research.run_dagger import build_aggregate

    built: dict[str, dict[str, Any]] = {}
    for arm, prefix in (("A", "e2b"), ("B", "12b")):
        destination = attempt / f"aggregate_{arm}"
        report = build_aggregate(
            round0_train=input_path(config, f"{prefix}_round0_train"),
            round0_validation=input_path(config, f"{prefix}_round0_validation"),
            round1_rows=input_path(config, f"{prefix}_round1_rows"),
            output_dir=destination,
            inclusion="learner_full",
            seed=20260823,
        )
        if report.get("round1_split_universe") != "all_audited_rows_before_inclusion":
            raise ValueError(f"arm {arm} did not use the frozen audited split universe")
        origins = report.get("round1_selected_state_origin_breakdown")
        if not isinstance(origins, dict) or set(origins) != {"learner_policy"}:
            raise ValueError(f"arm {arm} retained non-learner occupancy: {origins}")
        count_drift = {
            key: (report.get(key), expected)
            for key, expected in EXPECTED_LEARNER_FULL[arm].items()
            if report.get(key) != expected
        }
        if count_drift:
            raise ValueError(f"arm {arm} learner-full count drift: {count_drift}")
        if len(report.get("round1_split_assignment_validation_episodes", [])) != 66:
            raise ValueError(f"arm {arm} must retain exactly 66 validation roots")
        report_path = destination / "aggregate_report.json"
        atomic_json(report_path, report, exclusive=True)
        built[arm] = {
            "train": report["train"],
            "train_sha256": report["train_sha256"],
            "validation": report["validation"],
            "validation_sha256": report["validation_sha256"],
            "aggregate_report": str(report_path),
            "aggregate_report_sha256": sha256(report_path),
        }
    if (
        load_json(built["A"]["aggregate_report"])["round1_split_assignment_validation_episodes"]
        != load_json(built["B"]["aggregate_report"])["round1_split_assignment_validation_episodes"]
    ):
        raise ValueError("E2B and 12B learner-full arms do not share the same root split")

    e2b_legacy_validation = _validate_e2b_legacy_full(config)
    built["C"] = {
        "train": str(input_path(config, "e2b_legacy_full_train")),
        "train_sha256": config["inputs"]["e2b_legacy_full_train"]["sha256"],
        "validation": str(input_path(config, "e2b_legacy_full_validation")),
        "validation_sha256": config["inputs"]["e2b_legacy_full_validation"]["sha256"],
        "legacy_inclusion_validation": e2b_legacy_validation,
    }
    b12_legacy_validation = _validate_12b_legacy_selective(config)
    built["D"] = {
        "train": str(input_path(config, "12b_legacy_selective_train")),
        "train_sha256": config["inputs"]["12b_legacy_selective_train"]["sha256"],
        "validation": str(input_path(config, "12b_legacy_selective_validation")),
        "validation_sha256": config["inputs"]["12b_legacy_selective_validation"]["sha256"],
        "legacy_receipt": str(input_path(config, "12b_legacy_selective_receipt")),
        "legacy_receipt_sha256": config["inputs"]["12b_legacy_selective_receipt"]["sha256"],
        "legacy_inclusion_validation": b12_legacy_validation,
    }

    processors: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        lane = spec["lane"]
        if lane not in processors:
            processors[lane], resolved_id, resolved_revision = load_processor(
                model_id=MODELS[lane][0], revision=MODELS[lane][1]
            )
            if (resolved_id, resolved_revision) != MODELS[lane]:
                raise ValueError(f"processor identity drift for {lane}")
        train, validation = Path(built[arm]["train"]), Path(built[arm]["validation"])
        exposure = {
            "model_id": MODELS[lane][0],
            "model_revision": MODELS[lane][1],
            "max_length": 8192,
            "train": summarize_dataset(train, processors[lane], max_length=8192, label="train"),
            "validation": summarize_dataset(
                validation, processors[lane], max_length=8192, label="validation"
            ),
            "release_evidence": False,
        }
        _validate_exposure(exposure, train, validation, lane)
        exposure_path = attempt / f"exposure_{arm}.json"
        atomic_json(exposure_path, exposure, exclusive=True)
        built[arm]["exposure"] = str(exposure_path)
        built[arm]["exposure_sha256"] = sha256(exposure_path)
        built[arm].update(ARMS[arm])
        built[arm]["model_id"], built[arm]["model_revision"] = MODELS[lane]

    receipt = {
        "contract": CONTRACT,
        "stage": "build_complete",
        "config": str(config_path.resolve()),
        "config_sha256": config_sha,
        "source_commit": config["source_commit"],
        "source_commit_binding": "informational_label_tree_sha256_authoritative",
        "source_tree_sha256": config["source_tree_sha256"],
        "environment": environment,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "arms": built,
        "release_evidence": False,
    }
    receipt_path = attempt / "build_receipt.json"
    atomic_json(receipt_path, receipt, exclusive=True)
    pointer = {
        "contract": CONTRACT,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "config_sha256": config_sha,
        "source_tree_sha256": config["source_tree_sha256"],
    }
    atomic_json(cell_root / "build" / "completed.json", pointer, exclusive=True)
    return pointer


def verify_build(config_path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config, config_sha = validate_config(config_path, expected_sha)
    pointer = load_json(Path(config["cell_root"]) / "build" / "completed.json")
    receipt_path = Path(pointer["receipt"]).resolve(strict=True)
    if pointer.get("receipt_sha256") != sha256(receipt_path):
        raise ValueError("build receipt hash mismatch")
    receipt = load_json(receipt_path)
    if receipt.get("config_sha256") != config_sha or receipt.get("source_tree_sha256") != config["source_tree_sha256"]:
        raise ValueError("build receipt is not bound to this configuration/source")
    current_environment = environment_receipt(config)
    recorded_environment = receipt.get("environment") or {}
    if any(
        recorded_environment.get(key) != value
        for key, value in current_environment.items()
    ):
        raise ValueError("runtime environment changed after the build")
    for arm in ARMS:
        record = receipt["arms"][arm]
        for key in ("train", "validation", "exposure"):
            path = Path(record[key]).resolve(strict=True)
            if sha256(path) != record[f"{key}_sha256"]:
                raise ValueError(f"arm {arm} {key} changed after build")
    return config, pointer, receipt


def arm_values(config_path: Path, expected_sha: str, arm: str) -> list[str]:
    _, _, receipt = verify_build(config_path, expected_sha)
    record = receipt["arms"][arm]
    return [
        record["model_id"], record["model_revision"], str(record["updates"]),
        record["train"], record["validation"], record["train_sha256"],
        record["validation_sha256"], record["lane"], record["inclusion"],
        record["exposure"], record["exposure_sha256"],
    ]


def gate_training(config_path: Path, expected_sha: str, arm: str, summary_path: Path, output: Path) -> dict[str, Any]:
    _, _, receipt = verify_build(config_path, expected_sha)
    record = receipt["arms"][arm]
    summary = load_json(summary_path)
    exposure = load_json(record["exposure"])
    # Trainer closes a short gradient-accumulation group at each epoch boundary.
    # This makes the exact collated-row count slightly smaller than updates * 4
    # when the corpus size is not divisible by four.
    train_rows = exposure["train"]["prepared"]["rows"]
    full_epoch_updates = (train_rows + 3) // 4
    full_epochs, residual_updates = divmod(record["updates"], full_epoch_updates)
    expected_sampled_rows = full_epochs * train_rows + min(
        residual_updates * 4, train_rows
    )
    sampled_train = summary.get("sampled_train_exposure") or {}
    gpu_inventory = summary_path.parent.parent / "gpu_inventory.csv"
    allocated_gpu_path = summary_path.parent.parent / "allocated_gpu.json"
    allocated_gpu = load_json(allocated_gpu_path) if allocated_gpu_path.is_file() else {}
    expected_gpu_families = os.environ.get("CELL_EXPECTED_GPU_FAMILY", "").split("|")
    observed_gpu_family = os.environ.get("CELL_OBSERVED_GPU_FAMILY")
    slurmd_nodename = os.environ.get("SLURMD_NODENAME")
    expected_gpu_feature = os.environ.get("CELL_EXPECTED_GPU_FEATURE")
    slurm_job_constraint = os.environ.get("CELL_SCHEDULER_GPU_CONSTRAINT")
    slurm_job_constraint_source = os.environ.get(
        "CELL_SCHEDULER_GPU_CONSTRAINT_SOURCE"
    )
    checks = {
        "model_identity": (summary.get("model_id"), summary.get("model_revision"))
        == (record["model_id"], record["model_revision"]),
        "fresh_base_start": summary.get("initial_adapter") is None,
        "train_hash": summary.get("train_sha256") == record["train_sha256"],
        "validation_hash": summary.get("validation_sha256") == record["validation_sha256"],
        "updates": summary.get("optimizer_updates_completed") == record["updates"]
        and summary.get("max_steps") == record["updates"],
        "recipe": summary.get("seed") == 20260823
        and summary.get("max_length") == 8192
        and summary.get("epochs") == 2.0
        and summary.get("batch_size") == 1
        and summary.get("gradient_accumulation_steps") == 4
        and summary.get("learning_rate") == 0.0001
        and summary.get("lora_rank") == 16
        and summary.get("lora_alpha") == 16
        and summary.get("lora_dropout") == 0.0
        and summary.get("quantization")
        == {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "single_process": summary.get("world_size") == 1,
        "zero_preparation_failures": not summary.get("train_preparation_failures")
        and not summary.get("validation_preparation_failures"),
        "prepared_train_rows": summary.get("prepared_train_rows")
        == exposure["train"]["prepared"]["rows"],
        "prepared_validation_rows": summary.get("prepared_validation_rows")
        == exposure["validation"]["prepared"]["rows"],
        "train_corpus_tokens": summary.get("train_target_token_corpus_summary")
        == exposure["train"]["prepared"],
        "sampled_train_rows": sampled_train.get("rows") == expected_sampled_rows,
        "sampled_train_batches": sampled_train.get("batches") == expected_sampled_rows,
        "sampled_train_tokens_recorded": sampled_train.get("supervised_tokens", 0) > 0,
        "finite_training_loss": isinstance(summary.get("training_loss"), (int, float))
        and math.isfinite(float(summary["training_loss"])),
        "gpu_inventory": gpu_inventory.is_file() and gpu_inventory.stat().st_size > 0,
        "allocated_gpu_identity": allocated_gpu.get("torch_cuda_device_count") == 1
        and allocated_gpu.get("torch_device_index") == 0
        and allocated_gpu.get("cuda_visible_devices")
        == os.environ.get("CUDA_VISIBLE_DEVICES")
        and allocated_gpu.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")
        and bool(slurmd_nodename)
        and allocated_gpu.get("slurmd_nodename") == slurmd_nodename
        and allocated_gpu.get("slurm_job_constraint") == slurm_job_constraint
        and allocated_gpu.get("slurm_job_constraint_source")
        == slurm_job_constraint_source
        and isinstance(allocated_gpu.get("name"), str)
        and allocated_gpu.get("total_memory_bytes", 0) > 0,
        "gpu_family_attested": observed_gpu_family in expected_gpu_families,
        "slurm_constraint_attested": bool(expected_gpu_feature)
        and slurm_job_constraint_source in {"SLURM_JOB_CONSTRAINTS", "sacct"}
        and expected_gpu_feature == slurm_job_constraint,
    }
    adapter = summary_path.parent
    adapter_files = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter / name
        if not path.is_file():
            checks[f"adapter_file_{name}"] = False
        else:
            checks[f"adapter_file_{name}"] = True
            adapter_files[name] = sha256(path)
    if not all(checks.values()):
        raise ValueError(f"training gate failed: {[key for key, value in checks.items() if not value]}")
    gate = {
        "contract": CONTRACT, "stage": "training_gate", "arm": arm,
        "checks": checks, "summary": str(summary_path), "summary_sha256": sha256(summary_path),
        "adapter": str(adapter.resolve()), "adapter_files": adapter_files,
        "sampled_train_exposure": summary["sampled_train_exposure"],
        "sampled_validation_exposure": summary["sampled_validation_exposure"],
        "expected_sampled_train_rows": expected_sampled_rows,
        "gpu_inventory": str(gpu_inventory),
        "gpu_inventory_sha256": sha256(gpu_inventory),
        "allocated_gpu": str(allocated_gpu_path),
        "allocated_gpu_sha256": sha256(allocated_gpu_path),
        "allocated_gpu_identity": allocated_gpu,
        "expected_gpu_feature": expected_gpu_feature,
        "expected_gpu_family": os.environ.get("CELL_EXPECTED_GPU_FAMILY"),
        "observed_gpu_family": observed_gpu_family,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_constraint": slurm_job_constraint,
        "slurm_job_constraint_source": slurm_job_constraint_source,
        "release_evidence": False,
    }
    atomic_json(output, gate, exclusive=True)
    return gate


def gate_evaluation(config_path: Path, expected_sha: str, arm: str, training_gate_path: Path, evaluation_path: Path, output: Path) -> dict[str, Any]:
    config, _, receipt = verify_build(config_path, expected_sha)
    record = receipt["arms"][arm]
    training_gate = load_json(training_gate_path)
    if training_gate.get("summary_sha256") != sha256(training_gate["summary"]):
        raise ValueError("training summary changed after its gate")
    for name, digest in training_gate["adapter_files"].items():
        if sha256(Path(training_gate["adapter"]) / name) != digest:
            raise ValueError("adapter changed after training gate")
    report = load_json(evaluation_path)
    episodes = report.get("per_episode")
    scenarios = config["inputs"]["evaluation_scenarios"]
    checks = {
        "schema_v2": report.get("evaluation_schema_version") == 2,
        "label": report.get("label") == f"occupancy-cell-{arm}",
        "model_identity": (report.get("model_id"), report.get("model_revision"))
        == (record["model_id"], record["model_revision"]),
        "adapter": Path(str(report.get("adapter"))).resolve() == Path(training_gate["adapter"]).resolve(),
        "scenario_hash": report.get("scenarios_sha256") == scenarios["sha256"],
        "episode_count": report.get("episodes") == 65 and isinstance(episodes, list) and len(episodes) == 65,
        "full_action_arguments": isinstance(episodes, list) and all(
            isinstance(ep.get("actions"), list)
            and all(
                isinstance(action, dict)
                and isinstance(action.get("arguments"), dict)
                for action in ep["actions"]
            )
            for ep in episodes
        ),
        "per_episode_identity": isinstance(episodes, list)
        and all(
            ep.get("scenario_id")
            and ep.get("root_scenario_id")
            and ep.get("physical_root_fingerprint")
            for ep in episodes
        ),
        "research_only": report.get("release_evidence") is False and report.get("guards_enabled") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"evaluation gate failed: {[key for key, value in checks.items() if not value]}")
    gate = {
        "contract": CONTRACT, "stage": "evaluation_gate", "arm": arm,
        "checks": checks, "evaluation": str(evaluation_path),
        "evaluation_sha256": sha256(evaluation_path), "training_gate": str(training_gate_path),
        "training_gate_sha256": sha256(training_gate_path), "release_evidence": False,
    }
    atomic_json(output, gate, exclusive=True)
    return gate


def publish_arm(config_path: Path, expected_sha: str, arm: str, job_id: str, attempt: Path) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    train_gate = attempt / "training_gate.json"
    eval_gate = attempt / "evaluation_gate.json"
    for path in (train_gate, eval_gate):
        if not path.is_file():
            raise ValueError(f"missing arm gate: {path}")
    evaluation = load_json(eval_gate)["evaluation"]
    pointer = {
        "contract": CONTRACT, "stage": "arm_complete", "arm": arm, "job_id": job_id,
        "attempt": str(attempt.resolve()), "training_gate": str(train_gate),
        "training_gate_sha256": sha256(train_gate), "evaluation_gate": str(eval_gate),
        "evaluation_gate_sha256": sha256(eval_gate), "evaluation": evaluation,
        "evaluation_sha256": sha256(evaluation), "release_evidence": False,
    }
    target = Path(config["cell_root"]) / "runs" / arm / f"job-{job_id}" / "completed.json"
    atomic_json(target, pointer, exclusive=True)
    return pointer


def verify_arm(config_path: Path, expected_sha: str, arm: str, job_id: str) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    pointer = load_json(Path(config["cell_root"]) / "runs" / arm / f"job-{job_id}" / "completed.json")
    if pointer.get("arm") != arm or pointer.get("job_id") != job_id:
        raise ValueError("arm completion pointer identity mismatch")
    for key in ("training_gate", "evaluation_gate", "evaluation"):
        if sha256(pointer[key]) != pointer[f"{key}_sha256"]:
            raise ValueError(f"completed arm {key} hash mismatch")
    return pointer


def gate_audit(config_path: Path, expected_sha: str, arm: str, parent_job: str, full: Path, summary: Path, output: Path) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    completed = verify_arm(config_path, expected_sha, arm, parent_job)
    report = load_json(full)
    compact = load_json(summary)
    evaluation = report.get("evaluations", {}).get(f"occupancy-cell-{arm}", {})
    metrics = evaluation.get("summary", {})
    source = evaluation.get("source_evaluation", {})
    checks = {
        "scenario_hash": report.get("scenarios_sha256") == config["inputs"]["evaluation_scenarios"]["sha256"],
        "evaluation_hash": source.get("sha256") == completed["evaluation_sha256"],
        "episodes": metrics.get("episodes") == 65 and metrics.get("physical_assessable_episodes") == 65,
        "replay_exact": metrics.get("replay_mismatch_episodes") == 0,
        "classified": metrics.get("unclassified_episodes") == 0,
        "expert_baseline": compact.get("expert_baseline_problem_episodes") == 0,
        "source_identity_binding": evaluation.get("source_scenario_binding")
        == "sha256_and_per_episode_identity",
        "episode_identity_alignment": metrics.get("scenario_alignment_counts")
        == {"validated_physical_root_fingerprint": 65},
        "compact_bound": compact.get("full_report_sha256") == sha256(full),
        "research_only": report.get("release_evidence") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"physical audit gate failed: {[key for key, value in checks.items() if not value]}")
    gate = {
        "contract": CONTRACT, "stage": "physical_audit_gate", "arm": arm,
        "checks": checks, "full_report": str(full), "full_report_sha256": sha256(full),
        "summary_report": str(summary), "summary_report_sha256": sha256(summary),
        "physical_summary": metrics, "release_evidence": False,
    }
    atomic_json(output, gate, exclusive=True)
    return gate


def submission_update(config_path: Path, expected_sha: str, action: str, **kwargs: str) -> dict[str, Any]:
    config, config_sha = validate_config(config_path, expected_sha)
    target = Path(config["cell_root"]) / "submission.json"
    if action == "begin":
        payload = {
            "contract": CONTRACT, "status": "submitting", "config": str(config_path.resolve()),
            "config_sha256": config_sha, "source_commit": config["source_commit"],
            "source_commit_binding": "informational_label_tree_sha256_authoritative",
            "source_tree_sha256": config["source_tree_sha256"],
            "gpu_constraints": kwargs["gpu_constraints"], "jobs": {}, "release_evidence": False,
        }
        atomic_json(target, payload, exclusive=True)
        return payload
    payload = load_json(target)
    if payload.get("config_sha256") != config_sha:
        raise ValueError("submission receipt configuration mismatch")
    if action == "job":
        key = kwargs["role"] + (f":{kwargs['arm']}" if kwargs.get("arm") else "")
        if key in payload["jobs"]:
            raise ValueError(f"submission job already recorded: {key}")
        payload["jobs"][key] = kwargs["job_id"]
    elif action in {"finish", "fail"}:
        payload["status"] = "submitted" if action == "finish" else "submission_failed"
        if kwargs.get("detail"):
            payload["detail"] = kwargs["detail"]
    atomic_json(target, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    tree = sub.add_parser("tree-digest")
    tree.add_argument("--root", required=True, type=Path)
    digest = sub.add_parser("sha256")
    digest.add_argument("--path", required=True, type=Path)
    environment = sub.add_parser("environment")
    environment.add_argument("--config", required=True, type=Path)
    environment.add_argument("--expected-config-sha", required=True)
    snapshot = sub.add_parser("snapshot-digest")
    snapshot.add_argument("--hf-home", required=True, type=Path)
    snapshot.add_argument("--lane", required=True, choices=MODELS)
    cache = sub.add_parser("verify-model-cache")
    cache.add_argument("--config", required=True, type=Path)
    cache.add_argument("--expected-config-sha", required=True)
    cache.add_argument("--arm", required=True, choices=ARMS)
    for name in ("config-values", "verify-build"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True, type=Path)
        item.add_argument("--expected-config-sha", required=True)
    item = sub.add_parser("build")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--attempt", required=True, type=Path)
    item = sub.add_parser("arm-values")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item = sub.add_parser("gate-training")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--summary", required=True, type=Path)
    item.add_argument("--output", required=True, type=Path)
    item = sub.add_parser("gate-evaluation")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--training-gate", required=True, type=Path)
    item.add_argument("--evaluation", required=True, type=Path)
    item.add_argument("--output", required=True, type=Path)
    item = sub.add_parser("publish-arm")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--job-id", required=True)
    item.add_argument("--attempt", required=True, type=Path)
    item = sub.add_parser("verify-arm")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--job-id", required=True)
    item = sub.add_parser("gate-audit")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--parent-job", required=True)
    item.add_argument("--full", required=True, type=Path)
    item.add_argument("--summary", required=True, type=Path)
    item.add_argument("--output", required=True, type=Path)
    for name in ("submission-begin", "submission-job", "submission-finish", "submission-fail"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True, type=Path)
        item.add_argument("--expected-config-sha", required=True)
        if name == "submission-begin":
            item.add_argument("--gpu-constraints", required=True)
        elif name == "submission-job":
            item.add_argument("--role", required=True, choices=("build", "arm", "audit"))
            item.add_argument("--arm", choices=ARMS)
            item.add_argument("--job-id", required=True)
        elif name == "submission-fail":
            item.add_argument("--detail", default="sbatch failed")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "tree-digest":
        print(tree_digest(args.root))
        return 0
    if args.command == "sha256":
        print(sha256(args.path))
        return 0
    if args.command == "environment":
        config, _ = validate_config(args.config, args.expected_config_sha)
        print(json.dumps(environment_receipt(config), sort_keys=True))
        return 0
    if args.command == "snapshot-digest":
        print(json.dumps(snapshot_manifest(args.hf_home, args.lane), sort_keys=True))
        return 0
    if args.command == "verify-model-cache":
        config, _ = validate_config(args.config, args.expected_config_sha)
        lane = ARMS[args.arm]["lane"]
        print(json.dumps(verify_model_snapshot(config, lane), sort_keys=True))
        return 0
    if args.command == "config-values":
        config, _ = validate_config(args.config, args.expected_config_sha)
        print(config["cell_root"])
        print(config["source_root"])
        print(config["python"])
        print(config["hf_home"])
        return 0
    if args.command == "verify-build":
        _, pointer, _ = verify_build(args.config, args.expected_config_sha)
        print(json.dumps(pointer))
        return 0
    if args.command == "build":
        print(json.dumps(build(args.config, args.expected_config_sha, args.attempt)))
        return 0
    if args.command == "arm-values":
        print("\n".join(arm_values(args.config, args.expected_config_sha, args.arm)))
        return 0
    if args.command == "gate-training":
        gate_training(
            args.config, args.expected_config_sha, args.arm, args.summary, args.output
        )
        return 0
    if args.command == "gate-evaluation":
        gate_evaluation(
            args.config,
            args.expected_config_sha,
            args.arm,
            args.training_gate,
            args.evaluation,
            args.output,
        )
        return 0
    if args.command == "publish-arm":
        print(
            json.dumps(
                publish_arm(
                    args.config,
                    args.expected_config_sha,
                    args.arm,
                    args.job_id,
                    args.attempt,
                )
            )
        )
        return 0
    if args.command == "verify-arm":
        print(
            json.dumps(
                verify_arm(
                    args.config, args.expected_config_sha, args.arm, args.job_id
                )
            )
        )
        return 0
    if args.command == "gate-audit":
        gate_audit(
            args.config,
            args.expected_config_sha,
            args.arm,
            args.parent_job,
            args.full,
            args.summary,
            args.output,
        )
        return 0
    action = args.command.removeprefix("submission-")
    values = vars(args)
    payload = submission_update(
        args.config, args.expected_config_sha, action,
        **{key: str(values[key]) for key in ("gpu_constraints", "role", "arm", "job_id", "detail") if values.get(key) is not None},
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
