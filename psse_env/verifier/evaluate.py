"""Evaluation metrics for structured process verifiers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .dataset import extract_transition_labels


def _label(value: Any) -> str:
    return str(getattr(value, "value", value))


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _prediction_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {"candidate_disposition": _label(value)}


def classification_metrics(
    truth: Sequence[Any],
    predictions: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return confusion counts and standard per-class/macro metrics."""

    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length.")
    actual = [_label(value) for value in truth]
    predicted = [_label(value) for value in predictions]
    class_labels = list(labels or sorted(set(actual) | set(predicted)))
    confusion = {
        expected: {observed: 0 for observed in class_labels}
        for expected in class_labels
    }
    for expected, observed in zip(actual, predicted):
        if expected not in confusion:
            confusion[expected] = {candidate: 0 for candidate in class_labels}
            class_labels.append(expected)
            for row in confusion.values():
                row.setdefault(expected, 0)
        if observed not in class_labels:
            class_labels.append(observed)
            for row in confusion.values():
                row[observed] = 0
        confusion[expected][observed] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for candidate in class_labels:
        true_positive = confusion.get(candidate, {}).get(candidate, 0)
        false_negative = sum(
            count for observed, count in confusion.get(candidate, {}).items() if observed != candidate
        )
        false_positive = sum(
            row.get(candidate, 0) for expected, row in confusion.items() if expected != candidate
        )
        support = true_positive + false_negative
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        per_class[candidate] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_f1 = (
        sum(float(per_class[label]["f1"]) for label in class_labels) / len(class_labels)
        if class_labels
        else 0.0
    )
    accuracy = _safe_divide(sum(a == b for a, b in zip(actual, predicted)), len(actual))
    return {
        "labels": class_labels,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "total": len(actual),
    }


def false_accept_final_rate(truth: Sequence[Any], predictions: Sequence[Any]) -> float:
    """False-positive rate for ``ACCEPT_FINAL`` among truly non-final rows."""

    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length.")
    pairs = [(_label(actual), _label(predicted)) for actual, predicted in zip(truth, predictions)]
    negatives = sum(actual != "ACCEPT_FINAL" for actual, _ in pairs)
    false_finals = sum(
        actual != "ACCEPT_FINAL" and predicted == "ACCEPT_FINAL"
        for actual, predicted in pairs
    )
    return _safe_divide(false_finals, negatives)


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[Any],
    *,
    bins: int = 10,
) -> float:
    """Binary equal-width expected calibration error."""

    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length.")
    if bins < 1:
        raise ValueError("bins must be at least one.")
    if not probabilities:
        return 0.0

    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for probability, outcome in zip(probabilities, outcomes):
        try:
            value = float(probability)
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        value = min(max(value, 0.0), 1.0)
        index = min(int(value * bins), bins - 1)
        binary = float(_label(outcome) == "ACCEPT_FINAL" or outcome is True or outcome == 1)
        buckets[index].append((value, binary))

    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(value for value, _ in bucket) / len(bucket)
        frequency = sum(outcome for _, outcome in bucket) / len(bucket)
        error += len(bucket) / len(probabilities) * abs(confidence - frequency)
    return error


def _action_types(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if not isinstance(value, Sequence):
        return set()
    result: set[str] = set()
    for item in value:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, Mapping):
            tool = item.get("tool") or item.get("name") or item.get("tool_name")
            function = item.get("function")
            if not tool and isinstance(function, Mapping):
                tool = function.get("name")
            if tool:
                result.add(str(tool))
    return result


def next_action_validity_precision(
    valid_actions: Sequence[Any],
    predicted_actions: Sequence[Any],
) -> float:
    """Micro precision of proposed next action types against valid types."""

    if len(valid_actions) != len(predicted_actions):
        raise ValueError("valid_actions and predicted_actions must have the same length.")
    true_positive = 0
    proposed = 0
    for valid, predicted in zip(valid_actions, predicted_actions):
        valid_types = _action_types(valid)
        predicted_types = _action_types(predicted)
        true_positive += len(valid_types & predicted_types)
        proposed += len(predicted_types)
    return _safe_divide(true_positive, proposed)


def evaluate_predictions(
    truths: Sequence[Mapping[str, Any] | Any],
    predictions: Sequence[Mapping[str, Any] | Any],
    *,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate already-materialized verifier outputs."""

    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must have the same length.")
    truth_rows = [dict(value) if isinstance(value, Mapping) else {"candidate_disposition": value} for value in truths]
    prediction_rows = [_prediction_mapping(value) for value in predictions]
    true_labels = [_label(row.get("candidate_disposition")) for row in truth_rows]
    predicted_labels = [_label(row.get("candidate_disposition")) for row in prediction_rows]
    classification = classification_metrics(true_labels, predicted_labels)

    false_final_count = sum(
        actual != "ACCEPT_FINAL" and predicted == "ACCEPT_FINAL"
        for actual, predicted in zip(true_labels, predicted_labels)
    )
    nonfinal_count = sum(label != "ACCEPT_FINAL" for label in true_labels)
    predicted_final_count = sum(label == "ACCEPT_FINAL" for label in predicted_labels)

    final_probabilities: list[float] = []
    for label, row in zip(predicted_labels, prediction_rows):
        probabilities = row.get("candidate_disposition_probabilities")
        if isinstance(probabilities, Mapping) and probabilities.get("ACCEPT_FINAL") is not None:
            probability = probabilities["ACCEPT_FINAL"]
        elif row.get("terminal_success_probability") is not None:
            probability = row["terminal_success_probability"]
        else:
            probability = float(label == "ACCEPT_FINAL")
        try:
            final_probabilities.append(float(probability))
        except (TypeError, ValueError, OverflowError):
            final_probabilities.append(0.0)

    valid_actions = [
        row.get("valid_next_action_types", row.get("valid_next_actions", []))
        for row in truth_rows
    ]
    proposed_actions = [row.get("valid_next_action_types", []) for row in prediction_rows]
    per_class = classification["per_class"]
    reject_recall = float(per_class.get("REJECT", {}).get("recall", 0.0))
    partial_recall = float(per_class.get("ACCEPT_PARTIAL", {}).get("recall", 0.0))

    return {
        "candidate_disposition_macro_f1": classification["macro_f1"],
        "reject_recall": reject_recall,
        "accept_partial_recall": partial_recall,
        # Primary safety metric: false-positive rate over truly non-final rows.
        "false_accept_final_rate": _safe_divide(false_final_count, nonfinal_count),
        "false_accept_final_count": false_final_count,
        "non_accept_final_count": nonfinal_count,
        # Also expose false discovery among predicted finals to remove ambiguity.
        "false_accept_final_share_of_predictions": _safe_divide(
            false_final_count, predicted_final_count
        ),
        "calibration_error": expected_calibration_error(
            final_probabilities, true_labels, bins=calibration_bins
        ),
        "next_action_validity_precision": next_action_validity_precision(
            valid_actions, proposed_actions
        ),
        "classification": classification,
        "total_examples": len(true_labels),
    }


def _call_verifier(verifier: Any, transition: Mapping[str, Any]) -> Any:
    if hasattr(verifier, "verify"):
        return verifier.verify(transition)
    if hasattr(verifier, "predict"):
        return verifier.predict(transition)
    if callable(verifier):
        return verifier(transition)
    raise TypeError("verifier must be callable or expose verify/predict.")


def evaluate_verifier(
    verifier: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    calibration_bins: int = 10,
    skip_unlabeled: bool = True,
) -> dict[str, Any]:
    """Run a verifier over labeled transition rows and compute roadmap metrics."""

    truths: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        materialized = dict(row)
        labels = extract_transition_labels(materialized)
        if not labels.get("candidate_disposition"):
            if skip_unlabeled:
                skipped += 1
                continue
            raise ValueError("A row is missing labels.candidate_disposition.")
        prediction = _call_verifier(verifier, materialized)
        truths.append(labels)
        predictions.append(_prediction_mapping(prediction))
    metrics = evaluate_predictions(truths, predictions, calibration_bins=calibration_bins)
    metrics["skipped_unlabeled_examples"] = skipped
    return metrics


__all__ = [
    "classification_metrics",
    "evaluate_predictions",
    "evaluate_verifier",
    "expected_calibration_error",
    "false_accept_final_rate",
    "next_action_validity_precision",
]
