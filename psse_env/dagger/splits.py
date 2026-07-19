from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHYSICAL_FINGERPRINT_VERSION = 2
_PHYSICAL_ROOT_FIELDS = (
    "case",
    "case_path",
    "base_case_version",
    "model_version",
    "clean_case",
    "measurements",
    "z_obs",
    "clean_measurements",
    "measurement_placement",
    "measurement_covariance",
    "noise_seed",
    "true_measurement_errors",
    "true_parameter_errors",
    "true_topology_errors",
    "metadata",
    "telemetry_channel_configuration",
    "tool_failure_realization",
)
_NONPHYSICAL_REFERENCE_KEYS = frozenset(
    {
        "scenario_id",
        "root_scenario_id",
        "scenario_family",
        "oracle_action_hints",
        "suggested_actions",
        "semantic_field_provenance",
        "labels",
        "state_class",
        "preferred_action",
        "expert_actions",
        "target_evidence",
        "verification_output",
        "remaining_suspect_count",
        "unresolved_signatures",
        "dataset_split",
        "dataset_source",
        "production_label_eligible",
        "generation_provenance_id",
    }
)


def _file_identity(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    # A scalar path is a reference to bytes, not a physical identifier.  Do
    # not include its basename: copied realizations must remain one split
    # group even when their staging paths differ.
    return {"file_sha256": digest.hexdigest()}


def _directory_identity(path: Path) -> dict[str, Any]:
    files = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": str(child.relative_to(path)),
                **_file_identity(child),
            }
        )
    # Preserve only the layout below the referenced root.  The top-level
    # directory name is operational packaging, not physical realization.
    return {"files": files}


def _canonical_physical_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            str(child_key): _canonical_physical_value(child, key=str(child_key))
            for child_key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if str(child_key) not in _NONPHYSICAL_REFERENCE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_physical_value(child, key=key) for child in value]
    if isinstance(value, Path):
        if value.is_file():
            return _file_identity(value)
        if value.is_dir():
            return _directory_identity(value)
        return str(value)
    if isinstance(value, str) and (
        key.endswith("_path")
        or key.endswith("_dir")
        or key in {"case", "clean_case"}
    ):
        path = Path(value)
        if path.is_file():
            return _file_identity(path)
        if path.is_dir():
            return _directory_identity(path)
        if key.endswith("_path") or key.endswith("_dir"):
            # Runtime aliases such as case_path/state bindings are often
            # branch-specific names.  If no persisted content can be hashed,
            # retain only a common unresolved marker so aliases cannot split
            # byte-identical physical realizations.
            return {"unresolved_runtime_reference": True}
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Physical-root fingerprint cannot contain non-finite numbers.")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    # NumPy scalar/array-like values are not expected in persisted scenarios,
    # but converting through tolist/item keeps the helper reusable at source.
    if callable(getattr(value, "tolist", None)):
        return _canonical_physical_value(value.tolist(), key=key)
    if callable(getattr(value, "item", None)):
        return _canonical_physical_value(value.item(), key=key)
    raise TypeError(f"Unsupported physical-root value at {key!r}: {type(value).__name__}")


def physical_root_fingerprint(scenario: Mapping[str, Any]) -> str:
    """Hash the physical realization independently of scenario/branch IDs.

    The payload includes the model/case identity, operating and measurement
    realization, placement/covariance/noise fields when supplied, injected
    error set, telemetry payload/configuration, and tool-failure realization.
    Descendant DAgger/counterfactual IDs and supervision hints are excluded.
    """
    payload = {
        "fingerprint_version": PHYSICAL_FINGERPRINT_VERSION,
        **{
            field: _canonical_physical_value(scenario[field], key=field)
            for field in _PHYSICAL_ROOT_FIELDS
            if field in scenario
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"physical_v{PHYSICAL_FINGERPRINT_VERSION}_{hashlib.sha256(encoded).hexdigest()}"


def grouped_scenario_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Keep all descendants of a physical root in one deterministic split."""
    if train_fraction < 0 or validation_fraction < 0 or train_fraction + validation_fraction > 1:
        raise ValueError("invalid split fractions")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root_to_physical: dict[str, str] = {}
    for row in rows:
        copied = dict(row)
        root_id = str(row.get("root_scenario_id", row.get("scenario_id")))
        physical = row.get("physical_root_fingerprint")
        group_id = str(physical) if isinstance(physical, str) and physical else root_id
        previous = root_to_physical.setdefault(root_id, group_id)
        if previous != group_id:
            raise ValueError(
                f"Root scenario {root_id!r} carries multiple physical fingerprints."
            )
        groups[group_id].append(copied)
    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for group_id in sorted(groups):
        digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(2**64)
        split = "train" if fraction < train_fraction else (
            "validation" if fraction < train_fraction + validation_fraction else "test"
        )
        result[split].extend(groups[group_id])
    return result


def audit_physical_split_disjointness(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    require_fingerprints: bool = True,
) -> dict[str, Any]:
    ownership: dict[str, set[str]] = defaultdict(set)
    missing: list[str] = []
    for split_name, rows in splits.items():
        for index, row in enumerate(rows):
            value = row.get("physical_root_fingerprint")
            if not isinstance(value, str) or not value:
                if require_fingerprints:
                    missing.append(str(row.get("example_id", f"{split_name}[{index}]")))
                continue
            ownership[value].add(split_name)
    overlaps = {
        fingerprint: sorted(split_names)
        for fingerprint, split_names in sorted(ownership.items())
        if len(split_names) > 1
    }
    return {
        "passed": not missing and not overlaps,
        "fingerprint_count": len(ownership),
        "missing_fingerprint_count": len(missing),
        "missing_fingerprint_examples": missing[:100],
        "overlapping_fingerprints": overlaps,
    }


__all__ = [
    "PHYSICAL_FINGERPRINT_VERSION",
    "audit_physical_split_disjointness",
    "grouped_scenario_split",
    "physical_root_fingerprint",
]
