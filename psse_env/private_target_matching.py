"""Private-only correction-target matching shared by truth and release audits.

This module must never be called from policy construction.  It combines a
model-visible correction target with private scenario truth and the physical
parent/candidate states in order to answer one offline question: which declared
faults did this candidate actually repair within the scenario's release
tolerance?  Keeping that logic here prevents the transactional truth ledger and
the strict release audit from drifting on index bases, grouped measurements, or
compact parameter actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
)


CORRECTION_FAMILY = {
    CORRECT_MEASUREMENTS: "measurement",
    "correct_measurements_from_path": "measurement",
    CORRECT_PARAMETERS: "parameter",
    "correct_parameters_from_path": "parameter",
    CORRECT_TOPOLOGY: "topology",
    "correct_topology_from_path": "topology",
}

PARAMETER_BRANCH_COLUMNS = {
    "r": (2,),
    "x": (3,),
    "rx": (2, 3),
    "b": (4,),
    "tap": (8,),
    "shift": (9,),
}

_PARAMETER_COLUMN_NAMES = {
    2: ("r", "br_r"),
    3: ("x", "br_x"),
    4: ("b", "br_b"),
    8: ("tap", "ratio"),
    9: ("shift", "angle"),
}
_PARAMETER_CLEAN_FIELDS = {
    2: "clean_r",
    3: "clean_x",
    4: "clean_b",
    8: "clean_tap",
    9: "clean_shift",
}
_MEASUREMENT_SCALAR_TARGET_KEYS = (
    "measurement_index",
    "measurement_id",
    "index",
    "index0",
    "target",
    "meter",
)


@dataclass(frozen=True)
class PrivateTargetTolerances:
    """Release tolerances relevant to correction retirement."""

    measurement_abs: float = 1e-6
    measurement_rel: float = 1e-6
    final_case_abs: float = 1e-9
    final_case_rel: float = 1e-9


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"private target tolerance {name!r} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"private target tolerance {name!r} must be numeric"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"private target tolerance {name!r} must be finite and nonnegative"
        )
    return numeric


def private_target_tolerances(
    value: Mapping[str, Any] | PrivateTargetTolerances | None,
) -> PrivateTargetTolerances:
    """Extract the four retirement tolerances from a release-audit profile.

    ``value`` may be the complete private truth payload, the ``release_audit``
    mapping, its nested ``tolerances`` mapping, or an already validated profile.
    Unrelated diagnostic tolerances are intentionally ignored here; the strict
    release audit validates the complete profile.
    """

    if isinstance(value, PrivateTargetTolerances):
        return value
    source: Mapping[str, Any] = value if isinstance(value, Mapping) else {}
    release_audit = source.get("release_audit")
    if isinstance(release_audit, Mapping):
        source = release_audit
    nested = source.get("tolerances")
    if isinstance(nested, Mapping):
        source = nested
    defaults = PrivateTargetTolerances()
    fields = {
        name: _finite_nonnegative(source.get(name, getattr(defaults, name)), name=name)
        for name in (
            "measurement_abs",
            "measurement_rel",
            "final_case_abs",
            "final_case_rel",
        )
    }
    return PrivateTargetTolerances(**fields)


def as_sequence(value: Any) -> list[Any] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None
    return list(value) if isinstance(value, Sequence) else None


def nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None
        result = int(numeric)
    else:
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    return result if result >= 0 else None


def correction_family(action: Mapping[str, Any]) -> str | None:
    return CORRECTION_FAMILY.get(str(action.get("tool") or action.get("name") or ""))


def measurement_action_targets(arguments: Mapping[str, Any]) -> set[int] | None:
    """Return every explicitly declared zero-based measurement target."""

    targets: set[int] = set()
    group = arguments.get("suspect_group")
    if group is not None:
        rows = as_sequence(group)
        if rows is None:
            return None
        for raw in rows:
            index = nonnegative_integer(raw)
            if index is None:
                return None
            targets.add(index)

    updates = arguments.get("measurement_updates")
    if updates is not None:
        if isinstance(updates, Mapping):
            raw_indices = list(updates)
        else:
            update_rows = as_sequence(updates)
            if update_rows is None:
                return None
            raw_indices = []
            for item in update_rows:
                if not isinstance(item, Mapping):
                    return None
                raw_indices.append(item.get("index", item.get("index0")))
        for raw in raw_indices:
            index = nonnegative_integer(raw)
            if index is None:
                return None
            targets.add(index)

    for key in _MEASUREMENT_SCALAR_TARGET_KEYS:
        if arguments.get(key) is None:
            continue
        index = nonnegative_integer(arguments[key])
        if index is None:
            return None
        targets.add(index)
    return targets


def measurement_fault_target(fault: Mapping[str, Any]) -> int | None:
    for key in ("index", "index0", "measurement_index"):
        if fault.get(key) is not None:
            return nonnegative_integer(fault[key])
    return None


def canonical_branch_target(item: Mapping[str, Any]) -> tuple[str, Any] | None:
    """Canonicalize numeric branch aliases to a zero-based physical row.

    ``branch_row0`` is zero-based.  Both production spellings ``line_index``
    and ``line_index1`` are one-based.  Supplying contradictory aliases is
    malformed and fails closed.
    """

    rows: list[int] = []
    if item.get("branch_row0") is not None:
        row0 = nonnegative_integer(item["branch_row0"])
        if row0 is None:
            return None
        rows.append(row0)
    for key in ("line_index1", "line_index"):
        if item.get(key) is None:
            continue
        line1 = nonnegative_integer(item[key])
        if line1 is None or line1 < 1:
            return None
        rows.append(line1 - 1)
    if rows:
        return ("branch_row0", rows[0]) if len(set(rows)) == 1 else None

    named: list[str] = []
    for key in ("branch_id", "cb_name", "dss_element"):
        if item.get(key) is None:
            continue
        text = str(item[key]).strip()
        if not text:
            return None
        named.append(text)
    if named:
        return ("branch_id", named[0]) if len(set(named)) == 1 else None
    return None


def _load_case(value: Any, case_loader: Callable[[Any], Any] | None) -> Any | None:
    if isinstance(value, Mapping):
        return value
    if case_loader is None:
        return None
    try:
        loaded = case_loader(value)
    except Exception:
        return None
    return loaded if isinstance(loaded, Mapping) else None


def _case_rows(value: Any, case_loader: Callable[[Any], Any] | None) -> list[Any] | None:
    loaded = _load_case(value, case_loader)
    return as_sequence(loaded.get("branch")) if isinstance(loaded, Mapping) else None


def _resolve_named_row(rows: list[Any] | None, reference: str) -> int | None:
    if rows is None:
        return None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        if any(
            row.get(key) is not None and str(row[key]).strip() == reference
            for key in ("branch_id", "id", "name", "cb_name", "dss_element")
        ):
            return index
    return None


def _target_row0(target: tuple[str, Any], rows: list[Any] | None) -> int | None:
    if target[0] == "branch_row0":
        row0 = int(target[1])
        return row0 if rows is None or 0 <= row0 < len(rows) else None
    return _resolve_named_row(rows, str(target[1]))


def _topology_status_field(
    descriptor: Mapping[str, Any], row: Any
) -> str:
    if descriptor.get("status_field") is not None:
        return str(descriptor["status_field"])
    if isinstance(row, Mapping) and "br_status" in row:
        return "br_status"
    return "status"


def action_targets_private_fault(
    action: Mapping[str, Any],
    fault: Mapping[str, Any],
    *,
    parent_state: Mapping[str, Any] | None = None,
    case_loader: Callable[[Any], Any] | None = None,
) -> bool:
    """Check same-family target identity without inspecting target values."""

    family = correction_family(action)
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    if family == "measurement":
        targets = measurement_action_targets(arguments)
        target = measurement_fault_target(fault)
        return bool(targets and target is not None and target in targets)
    if family not in {"parameter", "topology"}:
        return False
    action_target = canonical_branch_target(arguments)
    fault_target = canonical_branch_target(fault)
    if action_target is None or fault_target is None:
        return False
    parent_case = (parent_state or {}).get("case")
    rows = _case_rows(parent_case, case_loader)
    action_row = _target_row0(action_target, rows)
    fault_row = _target_row0(fault_target, rows)
    targets_match = action_target == fault_target or bool(
        action_row is not None
        and fault_row is not None
        and action_row == fault_row
    )
    if not targets_match:
        return False
    if family == "topology" and fault.get("status_field") is not None:
        parent_row = (
            rows[action_row]
            if rows is not None
            and action_row is not None
            and 0 <= action_row < len(rows)
            else None
        )
        return _topology_status_field(arguments, parent_row) == str(
            fault["status_field"]
        )
    return True


def _close(
    actual: Any,
    expected: Any,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
    exact: bool = False,
) -> bool | None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected if exact else None
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError, OverflowError):
        return actual == expected if exact else None
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return math.isclose(
        left,
        right,
        abs_tol=0.0 if exact else abs_tolerance,
        rel_tol=0.0 if exact else rel_tolerance,
    )


def _measurement_value(state: Mapping[str, Any], index: int) -> Any | None:
    values = as_sequence(state.get("measurements"))
    return values[index] if values is not None and 0 <= index < len(values) else None


def _row_value(row: Any, column: int) -> Any | None:
    if isinstance(row, Mapping):
        if column == 10:
            for key in ("br_status", "status"):
                if row.get(key) is not None:
                    return row[key]
            return None
        for key in _PARAMETER_COLUMN_NAMES.get(column, ()):
            if row.get(key) is not None:
                return row[key]
        return None
    values = as_sequence(row)
    return values[column] if values is not None and 0 <= column < len(values) else None


def _topology_value(row: Any, field: str) -> Any | None:
    if isinstance(row, Mapping):
        return row.get(field)
    return _row_value(row, 10) if field in {"status", "br_status"} else None


def _branch_row(
    state: Mapping[str, Any],
    target: tuple[str, Any],
    case_loader: Callable[[Any], Any] | None,
) -> Any | None:
    rows = _case_rows(state.get("case"), case_loader)
    row0 = _target_row0(target, rows)
    return rows[row0] if rows is not None and row0 is not None else None


def _parameter_columns(fault: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw = fault.get("parameter", fault.get("field"))
    if raw is None:
        raw = "rx" if fault.get("clean_r") is not None or fault.get("clean_x") is not None else "x"
    return PARAMETER_BRANCH_COLUMNS.get(str(raw).strip().lower())


def _clean_parameter_value(
    fault: Mapping[str, Any],
    column: int,
    *,
    truth: Mapping[str, Any],
    target: tuple[str, Any],
    case_loader: Callable[[Any], Any] | None,
) -> Any | None:
    field = _PARAMETER_CLEAN_FIELDS[column]
    if fault.get(field) is not None:
        return fault[field]
    if len(_parameter_columns(fault) or ()) == 1:
        for key in ("clean", "clean_value", "true_value", "expected_value", "correct_value"):
            if fault.get(key) is not None:
                return fault[key]
    clean_row = _branch_row({"case": truth.get("clean_case")}, target, case_loader)
    return _row_value(clean_row, column)


def correction_matches_private_fault(
    action: Mapping[str, Any],
    fault: Mapping[str, Any],
    *,
    truth: Mapping[str, Any],
    parent_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    case_loader: Callable[[Any], Any] | None = None,
    tolerances: Mapping[str, Any] | PrivateTargetTolerances | None = None,
) -> bool | None:
    """Return whether one targeted fault was physically retired by a candidate.

    ``None`` means the private comparison evidence is unavailable or malformed;
    callers retire only an explicit ``True``.
    """

    if not action_targets_private_fault(
        action,
        fault,
        parent_state=parent_state,
        case_loader=case_loader,
    ):
        return False
    profile = private_target_tolerances(tolerances or truth)
    family = correction_family(action)
    arguments = action.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}

    if family == "measurement":
        index = measurement_fault_target(fault)
        if index is None:
            return None
        expected = next(
            (
                fault[key]
                for key in (
                    "clean",
                    "clean_value",
                    "true_value",
                    "expected_value",
                    "correct_value",
                )
                if fault.get(key) is not None
            ),
            None,
        )
        if expected is None:
            clean = as_sequence(truth.get("clean_measurements"))
            expected = clean[index] if clean is not None and 0 <= index < len(clean) else None
        before = _measurement_value(parent_state, index)
        after = _measurement_value(candidate_state, index)
        after_close = _close(
            after,
            expected,
            abs_tolerance=profile.measurement_abs,
            rel_tolerance=profile.measurement_rel,
        )
        before_close = _close(
            before,
            expected,
            abs_tolerance=profile.measurement_abs,
            rel_tolerance=profile.measurement_rel,
        )
        if after_close is None or before_close is None:
            return None
        return bool(after_close and not before_close)

    action_target = canonical_branch_target(arguments)
    fault_target = canonical_branch_target(fault)
    if action_target is None or fault_target is None:
        return None
    parent_rows = _case_rows(parent_state.get("case"), case_loader)
    action_row0 = _target_row0(action_target, parent_rows)
    if action_row0 is None:
        return None
    effective_target = ("branch_row0", action_row0)
    parent_row = _branch_row(parent_state, effective_target, case_loader)
    candidate_row = _branch_row(candidate_state, effective_target, case_loader)
    if parent_row is None or candidate_row is None:
        return None

    if family == "parameter":
        columns = _parameter_columns(fault)
        if not columns:
            return None
        after_results: list[bool] = []
        before_results: list[bool] = []
        for column in columns:
            expected = _clean_parameter_value(
                fault,
                column,
                truth=truth,
                target=fault_target,
                case_loader=case_loader,
            )
            after_close = _close(
                _row_value(candidate_row, column),
                expected,
                abs_tolerance=profile.final_case_abs,
                rel_tolerance=profile.final_case_rel,
            )
            before_close = _close(
                _row_value(parent_row, column),
                expected,
                abs_tolerance=profile.final_case_abs,
                rel_tolerance=profile.final_case_rel,
            )
            if after_close is None or before_close is None:
                return None
            after_results.append(after_close)
            before_results.append(before_close)
        # A compact production action has no parameter field because the
        # executor estimates and writes both r and x.  The physical diff, not a
        # fabricated default of "x", determines whether an rx truth target was
        # actually resolved.
        return bool(all(after_results) and not all(before_results))

    if family == "topology":
        status_field = _topology_status_field(arguments, parent_row)
        expected = next(
            (
                fault[key]
                for key in ("expected_status", "clean", "true_value")
                if fault.get(key) is not None
            ),
            None,
        )
        after_close = _close(
            _topology_value(candidate_row, status_field),
            expected,
            abs_tolerance=0.0,
            rel_tolerance=0.0,
            exact=True,
        )
        before_close = _close(
            _topology_value(parent_row, status_field),
            expected,
            abs_tolerance=0.0,
            rel_tolerance=0.0,
            exact=True,
        )
        if after_close is None or before_close is None:
            return None
        return bool(after_close and not before_close)
    return False


def matched_private_fault_indices(
    action: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    parent_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    case_loader: Callable[[Any], Any] | None = None,
    tolerances: Mapping[str, Any] | PrivateTargetTolerances | None = None,
) -> list[int]:
    family = correction_family(action)
    if family is None:
        return []
    faults = truth.get(f"true_{family}_errors")
    rows = as_sequence(faults)
    if rows is None:
        return []
    return [
        index
        for index, fault in enumerate(rows)
        if isinstance(fault, Mapping)
        and correction_matches_private_fault(
            action,
            fault,
            truth=truth,
            parent_state=parent_state,
            candidate_state=candidate_state,
            case_loader=case_loader,
            tolerances=tolerances,
        )
        is True
    ]


__all__ = [
    "CORRECTION_FAMILY",
    "PARAMETER_BRANCH_COLUMNS",
    "PrivateTargetTolerances",
    "action_targets_private_fault",
    "as_sequence",
    "canonical_branch_target",
    "correction_family",
    "correction_matches_private_fault",
    "matched_private_fault_indices",
    "measurement_action_targets",
    "measurement_fault_target",
    "nonnegative_integer",
    "private_target_tolerances",
]
