"""Same prompt, different learner: gradient-boosted trees on the fields an LLM triage prompt shows.

    python -m research.classifier_triage.benchmark ... --prompt-control prompt_top5=output/.../llm/prompt_top5

``gbm_llm_view`` approximates the prompt from the WLS payload.  This control
removes the approximation: every feature is parsed from the user message of
the rows ``llm_dataset`` rendered (the text the fine-tune read), the model is
fitted on the fine-tune's own training rows and targets, and it is read both
ways an LLM is read: its largest class as a decision (``_argmax``, the
counterpart of a greedy first action) and its request score thresholded on
the calibration split.  A gap between this control and the LLM is therefore
a gap of the learner, not of what the prompt shows.

Parsed fields: the chi-square ratio, the largest normalized residual, the
anomaly breadth, the listed residuals (channel, offset in the channel, value)
and the listed branch multipliers (line, R or X, value).  The derived
features (both ends of one line listed, their signs, the top residual on the
top multiplier's line) are functions of those fields only.  A prompt that
carries the bus and branch tables (``prompt_tables``) adds, per table row in
the prompt's own order, the row's values and, for a branch, the residuals of
its two end buses read off the bus table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.classifier_triage.features import BLOCKS
from research.classifier_triage.llm_dataset import target_class

RESIDUAL_SLOTS = 10
MULTIPLIER_SLOTS = 5
TABLE_BRANCH_SLOTS = 6
TABLE_BUS_SLOTS = 6
CLASSES = ("request", "measurement", "parameter", "topology")
ID_PREFIX = "triage_"


def prompt_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """The model-visible state in a rendered row's user message."""
    return json.loads(row["messages"][1]["content"])["state"]


def prompt_features(state: Mapping[str, Any]) -> dict[str, float]:
    """Features of the opening WLS as the prompt states it."""
    metrics = (state.get("last_tool_output") or {}).get("observable_metrics") or {}
    summary = metrics.get("wls_summary") or {}
    wls = (state.get("fresh_context_evidence") or {}).get("wls") or {}
    # The model view caps a list and states how many entries it left out (``_omitted_items``).
    listed = list(summary.get("top_residuals") or [])
    residuals = [item for item in listed if "channel" in item][:RESIDUAL_SLOTS]
    multipliers = [item for item in (summary.get("top_lagrange") or []) if "line_row0" in item][:MULTIPLIER_SLOTS]
    features: dict[str, float] = {
        "chi_square_ratio_log": float(np.log(max(float(metrics.get("chi_square_ratio") or 0.0), 1e-9))),
        "max_residual_log1p": float(np.log1p(float(metrics.get("max_normalized_residual") or 0.0))),
        "anomaly_breadth": float(wls.get("anomaly_breadth") or 0.0),
        "residuals_listed": float(len(residuals)),
        "residuals_omitted": float(sum(int(item.get("_omitted_items") or 0) for item in listed if "channel" not in item)),
        "multipliers_listed": float(len(multipliers)),
    }
    flows: dict[tuple[int, str], dict[str, float]] = {}
    buses: set[int] = set()
    branches: set[int] = set()
    for slot in range(RESIDUAL_SLOTS):
        item = residuals[slot] if slot < len(residuals) else None
        channel = str(item["channel"]) if item else ""
        value = float(item["value"]) if item else 0.0
        offset = int(item["channel_offset"]) if item else -1
        features[f"r{slot}_log1p"] = float(np.log1p(abs(value)))
        features[f"r{slot}_sign"] = float(np.sign(value))
        features[f"r{slot}_offset"] = float(offset)
        for name in BLOCKS:
            features[f"r{slot}_{name}"] = float(channel == name)
        if not item:
            continue
        if channel in ("Pf", "Qf", "Pt", "Qt"):
            branches.add(offset)
            flows.setdefault((offset, channel[0]), {})[channel[1]] = value
        else:
            buses.add(offset)
    pairs = [ends for ends in flows.values() if len(ends) == 2]
    features["flow_pair_listed"] = float(bool(pairs))
    features["flow_pair_same_sign"] = float(any(ends["f"] * ends["t"] > 0 for ends in pairs))
    features["flow_pair_opposite_sign"] = float(any(ends["f"] * ends["t"] < 0 for ends in pairs))
    features["distinct_buses_listed"] = float(len(buses))
    features["distinct_branches_listed"] = float(len(branches))
    lines: set[int] = set()
    for slot in range(MULTIPLIER_SLOTS):
        item = multipliers[slot] if slot < len(multipliers) else None
        value = float(item["value"]) if item else 0.0
        features[f"l{slot}_log1p"] = float(np.log1p(abs(value)))
        features[f"l{slot}_sign"] = float(np.sign(value))
        features[f"l{slot}_is_x"] = float(bool(item) and str(item.get("parameter")) == "X")
        features[f"l{slot}_line"] = float(item["line_row0"]) if item else -1.0
        if item:
            lines.add(int(item["line_row0"]))
    features["distinct_multiplier_branches"] = float(len(lines))
    top_residual = abs(float(residuals[0]["value"])) if residuals else 0.0
    top_multiplier = abs(float(multipliers[0]["value"])) if multipliers else 0.0
    features["residual_over_multiplier_log"] = float(np.log((top_residual + 1e-6) / (top_multiplier + 1e-6)))
    features["top_residual_on_top_multiplier_line"] = float(
        bool(residuals) and bool(multipliers) and str(residuals[0]["channel"]) in ("Pf", "Qf", "Pt", "Qt")
        and int(residuals[0]["channel_offset"]) == int(multipliers[0]["line_row0"]))
    if summary.get("branch_table") is not None or summary.get("bus_table") is not None:
        features.update(table_features(summary))
    return features


def _signed_log(value: Any) -> float:
    if value is None:
        return 0.0
    value = float(value)
    return float(np.sign(value) * np.log1p(abs(value)))


def table_features(summary: Mapping[str, Any]) -> dict[str, float]:
    """Features of the bus and branch tables as the prompt lists them (``psse_env.providers.wls_tables``)."""
    buses = [item for item in (summary.get("bus_table") or []) if isinstance(item, Mapping) and "bus" in item]
    branches = [item for item in (summary.get("branch_table") or []) if isinstance(item, Mapping) and "line" in item]
    omitted = summary.get("omitted") if isinstance(summary.get("omitted"), Mapping) else {}
    by_bus = {int(item["bus"]): item for item in buses}
    features: dict[str, float] = {
        "table_buses": float(len(buses)), "table_branches": float(len(branches)),
        "table_buses_omitted": float(omitted.get("buses") or 0), "table_branches_omitted": float(omitted.get("branches") or 0),
    }
    for slot in range(TABLE_BRANCH_SLOTS):
        item = branches[slot] if slot < len(branches) else None
        for name in ("pf", "qf", "pt", "qt", "lr", "lx"):
            features[f"bt{slot}_{name}"] = _signed_log(item.get(name)) if item else 0.0
        pf, pt = (float(item.get("pf") or 0.0), float(item.get("pt") or 0.0)) if item else (0.0, 0.0)
        features[f"bt{slot}_same_sign_p"] = float(bool(item) and pf * pt > 0 and min(abs(pf), abs(pt)) >= 2.0)
        features[f"bt{slot}_opposite_sign_p"] = float(bool(item) and pf * pt < 0 and min(abs(pf), abs(pt)) >= 2.0)
        features[f"bt{slot}_xfmr"] = float(bool(item and item.get("xfmr")))
        features[f"bt{slot}_out"] = float(bool(item and item.get("out")))
        for end in ("from", "to"):
            bus = by_bus.get(int(item[end])) if item and item.get(end) is not None else None
            for name in ("vm", "p", "q"):
                features[f"bt{slot}_{end}_{name}"] = _signed_log(bus.get(name)) if bus else 0.0
            features[f"bt{slot}_{end}_listed"] = float(bus is not None)
    for slot in range(TABLE_BUS_SLOTS):
        item = buses[slot] if slot < len(buses) else None
        for name in ("vm", "p", "q"):
            features[f"bb{slot}_{name}"] = _signed_log(item.get(name)) if item else 0.0
        features[f"bb{slot}_zero_injection"] = float(bool(item) and item.get("p") is None)
        for kind in ("PQ", "PV", "ref"):
            features[f"bb{slot}_{kind}"] = float(bool(item) and str(item.get("type")) == kind)
    return features


def load_variant(directory: Path) -> dict[str, Any]:
    """Features by row id for a rendered prompt variant, and the ids of the fine-tune's training rows."""
    table: dict[str, dict[str, float]] = {}
    train_ids: list[str] = []
    for name in ("train", "validation", "score"):
        with (Path(directory) / f"{name}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row.get("id") or str(row["example_id"])[len(ID_PREFIX):])
                table[row_id] = prompt_features(prompt_state(row))
                if name == "train":
                    train_ids.append(row_id)
    return {"features": table, "train_ids": train_ids}


def argmax_decisions(table: Mapping[str, Mapping[str, float]], train: Sequence[Mapping[str, Any]],
                     groups: Mapping[str, Sequence[Mapping[str, Any]]], seed: int) -> dict[str, list[str]]:
    """The four-way first action of a boosted model fitted on the fine-tune's rows and targets."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    names = sorted(table[str(train[0]["id"])])

    def matrix(rows):
        return np.asarray([[float(table[str(row["id"])][name]) for name in names] for row in rows], dtype=float)

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                           l2_regularization=1.0, random_state=seed)
    model.fit(matrix(train), np.asarray([CLASSES.index(target_class(row["triage"])) for row in train]))
    return {name: [CLASSES[int(k)] for k in model.predict(matrix(rows))] if rows else [] for name, rows in groups.items()}
