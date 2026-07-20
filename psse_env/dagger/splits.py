from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHYSICAL_FINGERPRINT_VERSION = 2
SPLIT_NAMES = ("train", "validation", "test")
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
_TRUTH_ERROR_FIELDS = (
    "true_measurement_errors",
    "true_parameter_errors",
    "true_topology_errors",
    "true_harmonic_errors",
    "true_hif_errors",
    "true_unbalance_errors",
)


@dataclass(frozen=True, order=True)
class ScenarioSplitStratum:
    """The release-relevant stratum of one independent physical root."""

    case_id: str
    error_family_combination: str
    error_cardinality: int
    source_tier: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "error_family_combination": self.error_family_combination,
            "error_cardinality": self.error_cardinality,
            "source_tier": self.source_tier,
        }


class StratifiedSplitError(ValueError):
    """Fail-closed split error with a machine-readable diagnostic report."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        self.diagnostics = dict(diagnostics)
        rendered = json.dumps(self.diagnostics, sort_keys=True, default=str)
        super().__init__(f"{message}: {rendered}")


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


def _row_label(row: Mapping[str, Any], index: int | None = None) -> str:
    for field in ("example_id", "scenario_id", "root_scenario_id"):
        value = row.get(field)
        if value is not None and str(value):
            return str(value)
    return f"row[{index}]" if index is not None else "unknown_row"


def _required_text(value: Any, *, field: str) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a non-empty scalar identifier")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty scalar identifier")
    return text


def _canonical_error_family(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        parts = [_required_text(item, field="error_family_combination") for item in value]
    else:
        text = _required_text(value, field="error_family_combination")
        parts = [part.strip() for part in text.split("+")]
    if not parts or any(not part for part in parts):
        raise ValueError("error_family_combination contains an empty family")
    return "+".join(sorted(set(parts)))


def _explicit_or_derived_cardinality(
    row: Mapping[str, Any], *, family: str
) -> int:
    explicit = row.get("error_cardinality")
    if explicit is not None:
        if isinstance(explicit, bool):
            raise ValueError("error_cardinality must be a non-negative integer")
        try:
            cardinality = int(explicit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "error_cardinality must be a non-negative integer"
            ) from exc
        if cardinality < 0 or str(cardinality) != str(explicit).strip():
            raise ValueError("error_cardinality must be a non-negative integer")
        return cardinality

    hidden_truth = row.get("hidden_truth")
    if hidden_truth is not None:
        if not isinstance(hidden_truth, Mapping):
            raise ValueError("hidden_truth must be a mapping when supplied")
    else:
        hidden_truth = {}
    found_truth_field = False
    cardinality = 0
    for field in _TRUTH_ERROR_FIELDS:
        # Prefer a promoted top-level field over its hidden-truth copy so a
        # serialization that retains both representations is not double-counted.
        if field in row:
            errors = row[field]
        elif field in hidden_truth:
            errors = hidden_truth[field]
        else:
            continue
        found_truth_field = True
        if errors is None:
            continue
        if not isinstance(errors, Sequence) or isinstance(
            errors, (str, bytes, bytearray)
        ):
            raise ValueError(f"{field} must be a sequence when supplied")
        cardinality += len(errors)
    if found_truth_field:
        return cardinality
    if family == "no_error":
        return 0
    raise ValueError(
        "error_cardinality is required when no recognized truth-error lists are present"
    )


def scenario_split_stratum(row: Mapping[str, Any]) -> ScenarioSplitStratum:
    """Return the strict case x family x cardinality x source stratum.

    ``network_case``/``case`` and ``scenario_family`` are accepted as
    compatibility aliases for ``case_id`` and ``error_family_combination``.
    Cardinality can be derived from recognized top-level/``hidden_truth``
    error lists.  ``source_tier`` is intentionally required (directly or in
    ``metadata``) so an absent tier cannot silently collapse measured and
    synthetic physical roots.
    """

    case_value = row.get("case_id")
    if case_value is None:
        case_value = row.get("network_case", row.get("case"))
    family_value = row.get(
        "error_family_combination", row.get("scenario_family")
    )
    metadata = row.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping when supplied")
    source_value = row.get("source_tier")
    if source_value is None and isinstance(metadata, Mapping):
        source_value = metadata.get("source_tier")

    case_id = _required_text(case_value, field="case_id")
    family = _canonical_error_family(family_value)
    source_tier = _required_text(source_value, field="source_tier")
    cardinality = _explicit_or_derived_cardinality(row, family=family)
    return ScenarioSplitStratum(
        case_id=case_id,
        error_family_combination=family,
        error_cardinality=cardinality,
        source_tier=source_tier,
    )


def _split_fractions(
    train_fraction: float, validation_fraction: float
) -> dict[str, float]:
    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction < 0
        or validation_fraction < 0
        or train_fraction + validation_fraction > 1
    ):
        raise ValueError("invalid split fractions")
    return {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": 1.0 - train_fraction - validation_fraction,
    }


def _coverage_requirements(
    minimum_roots_per_critical_family: Mapping[str, int] | None,
) -> dict[str, int]:
    requirements = {split: 0 for split in SPLIT_NAMES}
    if minimum_roots_per_critical_family is None:
        return requirements
    unknown = sorted(set(minimum_roots_per_critical_family) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"unknown split names in coverage requirements: {unknown}")
    for split, value in minimum_roots_per_critical_family.items():
        if isinstance(value, bool):
            raise ValueError("minimum root requirements must be non-negative integers")
        try:
            required = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "minimum root requirements must be non-negative integers"
            ) from exc
        if required < 0 or str(required) != str(value).strip():
            raise ValueError("minimum root requirements must be non-negative integers")
        requirements[split] = required
    return requirements


def _critical_family_names(values: Iterable[str]) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("critical_families must be an iterable of family names")
    return sorted({_canonical_error_family(family) for family in values})


def _largest_remainder_counts(
    size: int, fractions: Mapping[str, float]
) -> dict[str, int]:
    ideals = {split: size * fractions[split] for split in SPLIT_NAMES}
    counts = {split: int(math.floor(ideals[split])) for split in SPLIT_NAMES}
    remaining = size - sum(counts.values())
    ranked = sorted(
        SPLIT_NAMES,
        key=lambda split: (-(ideals[split] - counts[split]), SPLIT_NAMES.index(split)),
    )
    for split in ranked[:remaining]:
        counts[split] += 1
    return counts


def grouped_scenario_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Keep all descendants of a physical root in one deterministic split."""
    _split_fractions(train_fraction, validation_fraction)
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


def audit_stratified_split_coverage(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    critical_families: Iterable[str] = (),
    minimum_roots_per_critical_family: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Audit physical disjointness and independent-root family coverage."""

    requirements = _coverage_requirements(minimum_roots_per_critical_family)
    canonical_critical = _critical_family_names(critical_families)
    configuration_errors = []
    if any(requirements.values()) and not canonical_critical:
        configuration_errors.append(
            "critical_families must be non-empty when minimum roots are required"
        )
    disjointness = audit_physical_split_disjointness(
        splits, require_fingerprints=True
    )
    supplied_split_names = {str(split) for split in splits}
    missing_split_names = sorted(set(SPLIT_NAMES) - supplied_split_names)
    unexpected_split_names = sorted(supplied_split_names - set(SPLIT_NAMES))
    missing_strata: list[dict[str, str]] = []
    strata_by_fingerprint: dict[str, set[ScenarioSplitStratum]] = defaultdict(set)
    roots_by_family_split: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    roots_by_stratum_split: dict[
        ScenarioSplitStratum, dict[str, set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    for split_name, rows in splits.items():
        for index, row in enumerate(rows):
            fingerprint = row.get("physical_root_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                continue
            try:
                stratum = scenario_split_stratum(row)
            except (TypeError, ValueError) as exc:
                missing_strata.append(
                    {
                        "row": _row_label(row, index),
                        "split": str(split_name),
                        "error": str(exc),
                    }
                )
                continue
            family = stratum.error_family_combination
            strata_by_fingerprint[fingerprint].add(stratum)
            roots_by_family_split[family][str(split_name)].add(fingerprint)
            roots_by_stratum_split[stratum][str(split_name)].add(fingerprint)

    inconsistent_root_strata = {
        fingerprint: [stratum.as_dict() for stratum in sorted(strata)]
        for fingerprint, strata in sorted(strata_by_fingerprint.items())
        if len(strata) > 1
    }
    root_counts = {
        family: {
            split: len(roots_by_family_split[family].get(split, set()))
            for split in SPLIT_NAMES
        }
        for family in sorted(set(roots_by_family_split) | set(canonical_critical))
    }
    stratum_counts = [
        {
            **stratum.as_dict(),
            "root_counts_by_split": {
                split: len(roots_by_stratum_split[stratum].get(split, set()))
                for split in SPLIT_NAMES
            },
        }
        for stratum in sorted(roots_by_stratum_split)
    ]
    deficits = []
    for family in canonical_critical:
        for split in SPLIT_NAMES:
            required = requirements[split]
            actual = root_counts[family][split]
            if actual < required:
                deficits.append(
                    {
                        "family": family,
                        "split": split,
                        "required": required,
                        "actual": actual,
                        "deficit": required - actual,
                    }
                )
    missing_critical_families = [
        family
        for family in canonical_critical
        if sum(root_counts[family].values()) == 0
    ]
    return {
        "passed": bool(disjointness["passed"])
        and not configuration_errors
        and not missing_split_names
        and not unexpected_split_names
        and not missing_strata
        and not inconsistent_root_strata
        and not deficits
        and not missing_critical_families,
        "physical_split_disjointness": disjointness,
        "configuration_errors": configuration_errors,
        "critical_families": canonical_critical,
        "minimum_roots_per_critical_family": requirements,
        "missing_split_names": missing_split_names,
        "unexpected_split_names": unexpected_split_names,
        "root_counts_by_family_and_split": root_counts,
        "root_counts_by_stratum_and_split": stratum_counts,
        "coverage_deficits": deficits,
        "missing_critical_families": missing_critical_families,
        "missing_or_invalid_strata": missing_strata[:100],
        "missing_or_invalid_strata_count": len(missing_strata),
        "inconsistent_root_strata": inconsistent_root_strata,
    }


def stratified_grouped_scenario_split(
    rows: Iterable[Mapping[str, Any]],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    critical_families: Iterable[str] = (),
    minimum_roots_per_critical_family: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split physical roots within release-relevant strata.

    The indivisible group is ``physical_root_fingerprint``.  Every group must
    carry exactly one :class:`ScenarioSplitStratum`, so descendants cannot
    drift between case, family, cardinality, or source-tier buckets.  Initial
    per-stratum quotas use largest-remainder apportionment.  When critical
    family floors are configured, the quotas are minimally rebalanced while
    retaining stratification; infeasible coverage raises
    :class:`StratifiedSplitError` with diagnostics instead of emitting a
    release-ineligible split.
    """

    fractions = _split_fractions(train_fraction, validation_fraction)
    requirements = _coverage_requirements(minimum_roots_per_critical_family)
    canonical_critical = _critical_family_names(critical_families)
    if any(requirements.values()) and not canonical_critical:
        raise ValueError(
            "critical_families must be non-empty when minimum roots are required"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stratum_by_group: dict[str, ScenarioSplitStratum] = {}
    root_to_physical: dict[str, str] = {}
    input_errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        copied = dict(row)
        fingerprint = row.get("physical_root_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            input_errors.append(
                {
                    "code": "missing_physical_root_fingerprint",
                    "row": _row_label(row, index),
                }
            )
            continue
        try:
            stratum = scenario_split_stratum(row)
        except (TypeError, ValueError) as exc:
            input_errors.append(
                {
                    "code": "invalid_split_stratum",
                    "row": _row_label(row, index),
                    "physical_root_fingerprint": fingerprint,
                    "error": str(exc),
                }
            )
            continue

        root_value = row.get("root_scenario_id", row.get("scenario_id"))
        if root_value is not None:
            root_id = str(root_value)
            previous_physical = root_to_physical.setdefault(root_id, fingerprint)
            if previous_physical != fingerprint:
                input_errors.append(
                    {
                        "code": "root_has_multiple_physical_fingerprints",
                        "row": _row_label(row, index),
                        "root_scenario_id": root_id,
                        "physical_root_fingerprints": sorted(
                            {previous_physical, fingerprint}
                        ),
                    }
                )
                continue

        previous_stratum = stratum_by_group.setdefault(fingerprint, stratum)
        if previous_stratum != stratum:
            input_errors.append(
                {
                    "code": "physical_root_has_multiple_strata",
                    "row": _row_label(row, index),
                    "physical_root_fingerprint": fingerprint,
                    "strata": [previous_stratum.as_dict(), stratum.as_dict()],
                }
            )
            continue
        groups[fingerprint].append(copied)

    if input_errors:
        raise StratifiedSplitError(
            "Stratified grouped split input is invalid",
            {
                "passed": False,
                "input_error_count": len(input_errors),
                "input_errors": input_errors[:100],
            },
        )

    groups_by_stratum: dict[ScenarioSplitStratum, list[str]] = defaultdict(list)
    for fingerprint, stratum in stratum_by_group.items():
        groups_by_stratum[stratum].append(fingerprint)
    quota_by_stratum = {
        stratum: _largest_remainder_counts(len(fingerprints), fractions)
        for stratum, fingerprints in groups_by_stratum.items()
    }

    available_by_family: dict[str, int] = defaultdict(int)
    strata_by_family: dict[str, list[ScenarioSplitStratum]] = defaultdict(list)
    for stratum, fingerprints in groups_by_stratum.items():
        family = stratum.error_family_combination
        available_by_family[family] += len(fingerprints)
        strata_by_family[family].append(stratum)
    infeasible = []
    total_required = sum(requirements.values())
    for family in canonical_critical:
        available = available_by_family.get(family, 0)
        if available < total_required:
            infeasible.append(
                {
                    "family": family,
                    "available_independent_roots": available,
                    "required_independent_roots": total_required,
                    "minimum_by_split": requirements,
                }
            )
    if infeasible:
        raise StratifiedSplitError(
            "Critical-family coverage is infeasible",
            {
                "passed": False,
                "infeasible_critical_family_coverage": infeasible,
                "available_roots_by_family": dict(sorted(available_by_family.items())),
            },
        )

    # Meet family floors by transferring one quota at a time from a family
    # split above its own floor.  The squared-error delta chooses the transfer
    # that least disturbs the original per-stratum fractional targets.
    for family in canonical_critical:
        family_strata = sorted(strata_by_family[family])
        family_totals = {
            split: sum(quota_by_stratum[stratum][split] for stratum in family_strata)
            for split in SPLIT_NAMES
        }
        for target in SPLIT_NAMES:
            while family_totals[target] < requirements[target]:
                candidates: list[
                    tuple[float, str, ScenarioSplitStratum, str]
                ] = []
                for stratum in family_strata:
                    size = len(groups_by_stratum[stratum])
                    quota = quota_by_stratum[stratum]
                    ideal_target = size * fractions[target]
                    for donor in SPLIT_NAMES:
                        if donor == target or quota[donor] <= 0:
                            continue
                        if family_totals[donor] <= requirements[donor]:
                            continue
                        ideal_donor = size * fractions[donor]
                        before = (
                            (quota[target] - ideal_target) ** 2
                            + (quota[donor] - ideal_donor) ** 2
                        )
                        after = (
                            (quota[target] + 1 - ideal_target) ** 2
                            + (quota[donor] - 1 - ideal_donor) ** 2
                        )
                        tie = hashlib.sha256(
                            json.dumps(
                                [seed, family, target, donor, stratum.as_dict()],
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        candidates.append((after - before, tie, stratum, donor))
                if not candidates:
                    raise StratifiedSplitError(
                        "Critical-family quota rebalance failed closed",
                        {
                            "passed": False,
                            "family": family,
                            "target_split": target,
                            "required": requirements[target],
                            "actual": family_totals[target],
                            "family_totals": family_totals,
                        },
                    )
                _, _, stratum, donor = min(candidates)
                quota_by_stratum[stratum][donor] -= 1
                quota_by_stratum[stratum][target] += 1
                family_totals[donor] -= 1
                family_totals[target] += 1

    result: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLIT_NAMES
    }
    for stratum in sorted(groups_by_stratum):
        fingerprints = sorted(
            groups_by_stratum[stratum],
            key=lambda fingerprint: (
                hashlib.sha256(
                    json.dumps(
                        [seed, stratum.as_dict(), fingerprint],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                fingerprint,
            ),
        )
        cursor = 0
        for split in SPLIT_NAMES:
            next_cursor = cursor + quota_by_stratum[stratum][split]
            for fingerprint in fingerprints[cursor:next_cursor]:
                result[split].extend(groups[fingerprint])
            cursor = next_cursor
        if cursor != len(fingerprints):  # defensive invariant
            raise RuntimeError("Internal stratified quota did not exhaust its roots")

    audit = audit_stratified_split_coverage(
        result,
        critical_families=canonical_critical,
        minimum_roots_per_critical_family=requirements,
    )
    if not audit["passed"]:
        raise StratifiedSplitError(
            "Constructed stratified grouped split failed its release audit", audit
        )
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
    "SPLIT_NAMES",
    "ScenarioSplitStratum",
    "StratifiedSplitError",
    "audit_physical_split_disjointness",
    "audit_stratified_split_coverage",
    "grouped_scenario_split",
    "physical_root_fingerprint",
    "scenario_split_stratum",
    "stratified_grouped_scenario_split",
]
