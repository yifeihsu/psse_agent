"""A small standard-library numerical model for transition quality.

The learned component is multinomial logistic regression over the fixed
structured feature vector.  ``StructuredVerifierModel.verify`` combines it
with deterministic rules: hard process/physics decisions remain authoritative,
and the model may refine only inconclusive cases.  In particular, a learned
score cannot bypass the rule verifier's observable-evidence gate for
``ACCEPT_FINAL``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .features import FEATURE_NAMES, extract_transition_features, feature_vector, normalize_transition
from .rules import DISPOSITIONS, RuleBasedVerifier


def _label_value(value: Any) -> str | None:
    if value is None:
        return None
    candidate = getattr(value, "value", value)
    text = str(candidate)
    return text if text in DISPOSITIONS else None


def _extract_label(row: Mapping[str, Any]) -> str | None:
    labels = row.get("labels")
    if isinstance(labels, Mapping):
        label = _label_value(labels.get("candidate_disposition"))
        if label:
            return label
    transition_label = row.get("transition_label")
    if isinstance(transition_label, Mapping):
        label = _label_value(transition_label.get("candidate_disposition"))
        if label:
            return label
    return _label_value(row.get("candidate_disposition") or row.get("label"))


def _features(row: Mapping[str, Any]) -> dict[str, float]:
    def finite_feature(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    supplied = row.get("features")
    if isinstance(supplied, Mapping):
        return {name: finite_feature(supplied.get(name, 0.0)) for name in FEATURE_NAMES}
    if all(name in row for name in FEATURE_NAMES):
        return {name: finite_feature(row.get(name, 0.0)) for name in FEATURE_NAMES}
    return extract_transition_features(row)


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    maximum = max(logits)
    exponentials = [math.exp(min(max(value - maximum, -700.0), 700.0)) for value in logits]
    denominator = sum(exponentials)
    if denominator <= 0.0 or not math.isfinite(denominator):
        return [1.0 / len(logits)] * len(logits)
    return [value / denominator for value in exponentials]


class StructuredVerifierModel:
    """Deterministic softmax model with rule-based safety constraints."""

    def __init__(
        self,
        *,
        rule_verifier: RuleBasedVerifier | None = None,
        learning_rate: float = 0.08,
        iterations: int = 300,
        l2: float = 1e-4,
        class_balanced: bool = True,
    ) -> None:
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if iterations < 1:
            raise ValueError("iterations must be at least one.")
        if not math.isfinite(float(l2)) or l2 < 0.0:
            raise ValueError("l2 must be nonnegative.")
        self.rule_verifier = rule_verifier or RuleBasedVerifier()
        self.learning_rate = float(learning_rate)
        self.iterations = int(iterations)
        self.l2 = float(l2)
        self.class_balanced = bool(class_balanced)
        self.feature_names = tuple(FEATURE_NAMES)
        self.classes = tuple(DISPOSITIONS)
        self.means = [0.0] * len(self.feature_names)
        self.scales = [1.0] * len(self.feature_names)
        self.weights = [[0.0] * (len(self.feature_names) + 1) for _ in self.classes]
        self.temperature = 1.0
        self.training_examples = 0
        self.is_fitted = False

    def fit(
        self,
        rows: Iterable[Mapping[str, Any]],
        labels: Sequence[Any] | None = None,
    ) -> "StructuredVerifierModel":
        materialized = [dict(row) for row in rows]
        if labels is not None and len(labels) != len(materialized):
            raise ValueError("labels must have the same length as rows.")

        vectors: list[list[float]] = []
        targets: list[int] = []
        class_index = {label: index for index, label in enumerate(self.classes)}
        for index, row in enumerate(materialized):
            label = _label_value(labels[index]) if labels is not None else _extract_label(row)
            if label is None:
                continue
            vectors.append(feature_vector(_features(row), self.feature_names))
            targets.append(class_index[label])
        if not vectors:
            raise ValueError("No rows contain a recognized candidate_disposition label.")

        self.training_examples = len(vectors)
        dimension = len(self.feature_names)
        self.means = [sum(vector[j] for vector in vectors) / len(vectors) for j in range(dimension)]
        self.scales = []
        for j, mean in enumerate(self.means):
            variance = sum((vector[j] - mean) ** 2 for vector in vectors) / len(vectors)
            self.scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
        normalized = [self._standardize(vector) for vector in vectors]

        counts = [targets.count(index) for index in range(len(self.classes))]
        sample_weights = []
        for target in targets:
            if self.class_balanced and counts[target]:
                sample_weights.append(len(targets) / (len(self.classes) * counts[target]))
            else:
                sample_weights.append(1.0)

        self.weights = [[0.0] * (dimension + 1) for _ in self.classes]
        # Prior biases improve behavior for small datasets and absent classes.
        for class_id, count in enumerate(counts):
            self.weights[class_id][0] = math.log((count + 1.0) / (len(targets) + len(self.classes)))

        for iteration in range(self.iterations):
            gradients = [[0.0] * (dimension + 1) for _ in self.classes]
            total_weight = sum(sample_weights)
            for vector, target, sample_weight in zip(normalized, targets, sample_weights):
                augmented = [1.0, *vector]
                probabilities = _softmax(
                    [sum(weight * value for weight, value in zip(row, augmented)) for row in self.weights]
                )
                for class_id in range(len(self.classes)):
                    error = (probabilities[class_id] - float(class_id == target)) * sample_weight
                    for j, value in enumerate(augmented):
                        gradients[class_id][j] += error * value

            rate = self.learning_rate / math.sqrt(1.0 + iteration / 25.0)
            for class_id in range(len(self.classes)):
                for j in range(dimension + 1):
                    regularization = 0.0 if j == 0 else self.l2 * self.weights[class_id][j]
                    gradient = gradients[class_id][j] / max(total_weight, 1e-12) + regularization
                    self.weights[class_id][j] -= rate * gradient

        self.temperature = 1.0
        self.is_fitted = True
        return self

    def _standardize(self, vector: Sequence[float]) -> list[float]:
        return [
            (float(value) - self.means[index]) / self.scales[index]
            for index, value in enumerate(vector)
        ]

    def _logits(self, row: Mapping[str, Any]) -> list[float]:
        vector = feature_vector(_features(row), self.feature_names)
        normalized = self._standardize(vector)
        augmented = [1.0, *normalized]
        return [sum(weight * value for weight, value in zip(weights, augmented)) for weights in self.weights]

    def predict_proba(self, row: Mapping[str, Any]) -> dict[str, float]:
        if not self.is_fitted:
            rule = self.rule_verifier.verify(row)
            label = str(rule["candidate_disposition"])
            return {candidate: float(candidate == label) for candidate in self.classes}
        temperature = max(self.temperature, 1e-6)
        probabilities = _softmax([value / temperature for value in self._logits(row)])
        return dict(zip(self.classes, probabilities))

    def predict_learned_disposition(self, row: Mapping[str, Any]) -> str:
        """Return the raw learned class before deterministic safety gates."""

        probabilities = self.predict_proba(row)
        # DISPOSITIONS is ordered from conservative to permissive, so exact
        # ties resolve away from false finalization.
        return max(self.classes, key=lambda label: probabilities[label])

    def predict(self, row: Mapping[str, Any]) -> str:
        """Return the safety-constrained disposition for estimator-style callers."""

        return str(self.verify(row)["candidate_disposition"])

    def calibrate(
        self,
        rows: Iterable[Mapping[str, Any]],
        labels: Sequence[Any] | None = None,
        *,
        temperatures: Sequence[float] | None = None,
    ) -> float:
        """Fit a deterministic scalar temperature by validation NLL."""

        if not self.is_fitted:
            raise RuntimeError("fit must be called before calibrate.")
        materialized = [dict(row) for row in rows]
        if labels is not None and len(labels) != len(materialized):
            raise ValueError("labels must have the same length as rows.")
        candidates = (
            [0.5 + 0.05 * index for index in range(51)]
            if temperatures is None
            else [float(value) for value in temperatures]
        )
        if not candidates or any(not math.isfinite(value) or value <= 0.0 for value in candidates):
            raise ValueError("temperatures must contain positive values.")
        class_index = {label: index for index, label in enumerate(self.classes)}
        labeled: list[tuple[list[float], int]] = []
        for index, row in enumerate(materialized):
            label = _label_value(labels[index]) if labels is not None else _extract_label(row)
            if label is not None:
                labeled.append((self._logits(row), class_index[label]))
        if not labeled:
            raise ValueError("No calibration rows contain a recognized label.")

        def nll(temperature: float) -> float:
            loss = 0.0
            for logits, target in labeled:
                probability = _softmax([value / temperature for value in logits])[target]
                loss -= math.log(max(probability, 1e-15))
            return loss / len(labeled)

        self.temperature = min(candidates, key=lambda value: (nll(float(value)), float(value)))
        return self.temperature

    def verify(self, transition: Mapping[str, Any]) -> dict[str, Any]:
        """Return rule-augmented structured predictions for one transition."""

        base = self.rule_verifier.verify(transition)
        probabilities = self.predict_proba(transition)
        learned = max(self.classes, key=lambda label: probabilities[label])
        disposition = str(base["candidate_disposition"])

        # Deterministic decisions carry hard observable evidence.  The model is
        # used only to resolve an inconclusive rule output, and even there it
        # cannot assert final success without the rule gate.
        if disposition == "INCONCLUSIVE":
            if learned in {"REJECT", "ACCEPT_PARTIAL"}:
                disposition = learned
            elif learned == "ACCEPT_FINAL":
                nonfinal = [label for label in self.classes if label != "ACCEPT_FINAL"]
                alternative = max(nonfinal, key=lambda label: probabilities[label])
                disposition = alternative if alternative != "ACCEPT_FINAL" else "INCONCLUSIVE"

        base["candidate_disposition"] = disposition
        base["candidate_disposition_probabilities"] = probabilities
        terminal_probability = probabilities.get("ACCEPT_FINAL", 0.0)
        if disposition != "ACCEPT_FINAL":
            terminal_probability = min(terminal_probability, 0.49)
        base["terminal_success_probability"] = terminal_probability

        item = normalize_transition(transition)
        base["valid_next_action_types"] = self.rule_verifier.valid_next_action_types(
            item,
            disposition=disposition,
            process_valid=bool(base["process_valid"]),
            process_reason=(base.get("rationale_codes") or [None])[0],
        )
        return base

    def to_dict(self) -> dict[str, Any]:
        """Serialize learned numerical parameters without pickle."""

        return {
            "version": 1,
            "feature_names": list(self.feature_names),
            "classes": list(self.classes),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": [list(row) for row in self.weights],
            "temperature": self.temperature,
            "training_examples": self.training_examples,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        rule_verifier: RuleBasedVerifier | None = None,
    ) -> "StructuredVerifierModel":
        if payload.get("version") != 1:
            raise ValueError("Unsupported serialized verifier model version.")
        if tuple(payload.get("feature_names", ())) != tuple(FEATURE_NAMES):
            raise ValueError("Serialized feature schema does not match this verifier version.")
        if tuple(payload.get("classes", ())) != tuple(DISPOSITIONS):
            raise ValueError("Serialized class schema does not match this verifier version.")
        model = cls(rule_verifier=rule_verifier)
        dimension = len(FEATURE_NAMES)
        means = [float(value) for value in payload.get("means", ())]
        scales = [float(value) for value in payload.get("scales", ())]
        weights = [[float(value) for value in row] for row in payload.get("weights", ())]
        if len(means) != dimension or len(scales) != dimension:
            raise ValueError("Serialized scaler has an invalid dimension.")
        if len(weights) != len(DISPOSITIONS) or any(len(row) != dimension + 1 for row in weights):
            raise ValueError("Serialized weights have an invalid dimension.")
        if any(not math.isfinite(value) for value in means):
            raise ValueError("Serialized means must be finite.")
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("Serialized scales must be finite and positive.")
        if any(not math.isfinite(value) for row in weights for value in row):
            raise ValueError("Serialized weights must be finite.")
        temperature = float(payload.get("temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Serialized temperature must be finite and positive.")
        training_examples = int(payload.get("training_examples", 0))
        if training_examples < 0:
            raise ValueError("Serialized training_examples must be nonnegative.")
        model.means = means
        model.scales = scales
        model.weights = weights
        model.temperature = temperature
        model.training_examples = training_examples
        model.is_fitted = bool(payload.get("is_fitted", True))
        return model


LinearVerifierModel = StructuredVerifierModel
HybridProcessVerifier = StructuredVerifierModel


__all__ = [
    "HybridProcessVerifier",
    "LinearVerifierModel",
    "StructuredVerifierModel",
]
