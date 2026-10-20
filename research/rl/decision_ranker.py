"""Predictability test of the decision audit (phase 3 of docs/decision_audit_plan_20261019.md).

    python -m research.rl.decision_ranker --states output/decision_audit_20261019/train168/states.jsonl \\
        --suite output/rl_hard_draw_20261015/hard_training_screen.json --output-dir output/decision_audit_20261019/ranker

For every audited correction decision (a ``correct_parameters`` or
``correct_measurements`` choice) the candidates the audit branched on are
the chosen action and its alternatives; each carries the observable features
the audit stored and a known outcome (the reference's for the chosen one,
the branch's for an alternative).  A classifier over the candidate features
predicts "this candidate leads to success"; the decision picks the candidate
with the highest predicted probability among the audited ones.  Physical
parents are held out (grouped cross-validation by root, every training root
being its own parent on this draw), so a parent never scores its own
decision.  Reported per stratum: roots solved by the fixed rule (the
expert's choice), by the ranker, and by the audited oracle (the best audited
candidate, retrospective and privileged), plus the ranker's AUC on the
candidates.  A ranker is worth training only where it beats the rule
without losing the roots the rule solves.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RANKER_CONTRACT = "rl_decision_ranker_v1"
DECISION_TOOLS = ("correct_parameters", "correct_measurements")
CHANNELS = ("Pinj", "Qinj", "Pf", "Pt", "Qf", "Qt", "Vm")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def candidate_vector(features: Mapping[str, Any], *, tool: str) -> list[float]:
    """The numeric view of one candidate, the same for every candidate of a decision kind."""

    wls = features.get("wls") or {}
    common = [
        _num(features.get("rank")), _num(features.get("step")), _num(features.get("remaining_budget")),
        _num(features.get("accepted_corrections")), _num(features.get("rejected_hypotheses")), _num(features.get("tried_actions")),
        _num(features.get("remaining_anomaly_score")), _num(wls.get("max_normalized_residual")), _num(wls.get("anomaly_breadth")),
        1.0 if wls.get("chi_square_alarm") else 0.0, 1.0 if wls.get("normalized_residual_alarm") else 0.0,
    ]
    if tool == "correct_parameters":
        candidates = features.get("parameter_candidates") or []
        scores = [_num(c.get("abs_lambda_score") or c.get("abs_lambda")) for c in candidates if isinstance(c, Mapping)]
        top = scores[0] if scores else 0.0
        second = scores[1] if len(scores) > 1 else 0.0
        own = _num(features.get("candidate_abs_lambda"))
        return common + [
            own, _num(features.get("candidate_lambda_ratio_to_top")), _num(features.get("candidate_context_rank"), 9.0),
            _num(features.get("parameter_dominance_ratio")), 1.0 if features.get("parameter_dominant") else 0.0,
            _num(features.get("parameter_top_abs_lambda")), _num(features.get("parameter_runner_up_abs_lambda")),
            (own - second) if own else 0.0, (top - own) if own else 0.0, float(len(scores)),
            1.0 if str(features.get("parameter_route_status_reason") or "").endswith("ambiguous_top_candidates") else 0.0,
        ]
    channels = [str(c) for c in (features.get("meter_channels") or [])]
    return common + [
        _num(features.get("meter_value")), _num(features.get("meter_value_ratio_to_top")), _num(features.get("meter_context_rank"), 9.0),
        _num(features.get("meter_group_size"), 1.0), _num(features.get("meter_same_branch_partner")), _num(features.get("measurement_finding_count")),
        *[1.0 if any(c == name for c in channels) else 0.0 for name in CHANNELS],
    ]


def decision_rows(states: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per decision kind, one row per audited correction decision with its candidates, features, labels and outcomes."""

    rows: dict[str, list[dict[str, Any]]] = {tool: [] for tool in DECISION_TOOLS}
    for state in states:
        chosen = state.get("chosen") or {}
        tool = str(chosen.get("tool"))
        if tool not in rows or "bin" not in state:
            continue
        alternatives = [a for a in state.get("alternatives") or [] if str((a.get("action") or {}).get("tool")) == tool and a.get("outcome")]
        reference = state.get("reference") or {}
        chosen_features = None
        if alternatives:
            # The chosen candidate's own features: the same state view with its own arguments.
            template = dict(alternatives[0]["features"])
            template.update({"rank": state.get("chosen_rank") or 0, "line_index": (chosen.get("arguments") or {}).get("line_index"),
                             "suspect_group": (chosen.get("arguments") or {}).get("suspect_group")})
            chosen_features = template
        candidates = []
        if chosen_features is not None:
            candidates.append({"candidate": "chosen", "action": chosen, "features": chosen_features, "success": bool(reference.get("success")),
                               "steps": int(reference.get("steps") or 0)})
        for a in alternatives:
            candidates.append({"candidate": f"alt{a.get('rank')}", "action": a["action"], "features": a["features"],
                               "success": bool(a["outcome"]["success"]), "steps": int(a["outcome"].get("steps") or 0)})
        if len(candidates) < 2:
            continue
        rows[tool].append({"root": state["root"], "stratum": state.get("stratum"), "step": state.get("step"), "bin": state.get("bin"),
                           "candidates": candidates, "reference_success": bool(reference.get("success"))})
    return rows


def _refresh_chosen_features(row: Mapping[str, Any], tool: str) -> None:
    """Recompute the chosen candidate's own scores from the shared context (the audit stored them for alternatives only)."""

    chosen = row["candidates"][0]
    if chosen["candidate"] != "chosen":
        return
    features = chosen["features"]
    if tool == "correct_parameters":
        line = features.get("line_index")
        ranked = features.get("parameter_candidates") or []
        scores = {int(c["line_index1"]): _num(c.get("abs_lambda_score") or c.get("abs_lambda")) for c in ranked if isinstance(c, Mapping) and c.get("line_index1") is not None}
        own = scores.get(int(line)) if line is not None else None
        top = max(scores.values()) if scores else None
        features["candidate_abs_lambda"] = own
        features["candidate_lambda_ratio_to_top"] = (own / top) if own is not None and top else None
        features["candidate_context_rank"] = next((i for i, c in enumerate(ranked) if isinstance(c, Mapping) and c.get("line_index1") == line), None)
    else:
        group = features.get("suspect_group") or []
        findings = features.get("measurement_findings") or []
        by_index = {}
        for position, item in enumerate(findings):
            for key in ("index0", "index", "measurement_index"):
                if isinstance(item, Mapping) and item.get(key) is not None:
                    by_index[int(item[key])] = (position, item)
                    break
        own = [by_index[int(i)] for i in group if int(i) in by_index]
        values = [_num(item.get("value")) for _p, item in own]
        top = max((_num(item.get("value")) for item in findings if isinstance(item, Mapping)), default=None)
        features["meter_context_rank"] = min((p for p, _i in own), default=None)
        features["meter_value"] = max(values) if values else None
        features["meter_value_ratio_to_top"] = (max(values) / top) if values and top else None
        features["meter_channels"] = [str(item.get("channel")) for _p, item in own]
        features["meter_group_size"] = len(group)
        partners = 0
        for _p, item in own:
            partner = {"Pf": "Pt", "Pt": "Pf", "Qf": "Qt", "Qt": "Qf"}.get(str(item.get("channel")))
            if partner and any(isinstance(o, Mapping) and o.get("channel_offset") == item.get("channel_offset") and str(o.get("channel")) == partner for o in findings):
                partners += 1
        features["meter_same_branch_partner"] = partners


def evaluate_kind(rows: Sequence[Mapping[str, Any]], *, tool: str, seed: int = 0) -> dict[str, Any]:
    """Grouped-by-root cross-validation of the candidate classifier; per-stratum roots solved by rule, ranker and audited oracle."""

    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for row in rows:
        _refresh_chosen_features(row, tool)
    X, y, groups, owner = [], [], [], []
    for index, row in enumerate(rows):
        for candidate in row["candidates"]:
            X.append(candidate_vector(candidate["features"], tool=tool))
            y.append(int(candidate["success"]))
            groups.append(row["root"])
            owner.append(index)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    owner = np.asarray(owner)
    report: dict[str, Any] = {"tool": tool, "decisions": len(rows), "candidates": int(len(y)), "positives": int(y.sum())}
    if len(rows) < 5 or y.sum() == 0 or y.sum() == len(y):
        report["note"] = "too few decisions or no label variation"
        return report
    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        "gbm": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.9, random_state=seed),
    }
    folds = min(10, len(set(groups)))
    results = {}
    for name, make in models.items():
        probabilities = np.zeros(len(y), dtype=float)
        for train_index, test_index in GroupKFold(n_splits=folds).split(X, y, groups):
            model = make()
            model.fit(X[train_index], y[train_index])
            probabilities[test_index] = model.predict_proba(X[test_index])[:, 1]
        picks = {}
        for index, row in enumerate(rows):
            members = np.where(owner == index)[0]
            best = members[int(np.argmax(probabilities[members]))]
            position = int(best - members[0])
            picks[index] = row["candidates"][position]
        by_stratum: dict[str, Counter] = defaultdict(Counter)
        for index, row in enumerate(rows):
            entry = by_stratum[str(row["stratum"])]
            entry["decisions"] += 1
            entry["rule"] += int(row["candidates"][0]["success"])
            entry["ranker"] += int(picks[index]["success"])
            entry["oracle"] += int(any(c["success"] for c in row["candidates"]))
            entry["ranker_lost"] += int(row["candidates"][0]["success"] and not picks[index]["success"])
            entry["ranker_gained"] += int((not row["candidates"][0]["success"]) and picks[index]["success"])
        totals = Counter()
        for entry in by_stratum.values():
            totals.update(entry)
        try:
            auc = float(roc_auc_score(y, probabilities))
        except ValueError:
            auc = None
        results[name] = {"auc": auc, "totals": dict(totals), "by_stratum": {s: dict(c) for s, c in sorted(by_stratum.items())}}
    report["models"] = results
    return report


def markdown(report: Mapping[str, Any]) -> str:
    lines = []
    for tool, kind in report["kinds"].items():
        lines.append(f"### {tool}: {kind.get('decisions')} decisions, {kind.get('candidates')} candidates ({kind.get('positives')} successful)")
        if kind.get("note"):
            lines.append(kind["note"]); continue
        for name, result in kind["models"].items():
            t = result["totals"]
            lines.append(f"- **{name}** (AUC {result['auc']:.3f}): rule {t.get('rule', 0)} / ranker {t.get('ranker', 0)} / oracle {t.get('oracle', 0)} of {t.get('decisions', 0)} decisions; ranker gained {t.get('ranker_gained', 0)}, lost {t.get('ranker_lost', 0)}")
            lines.append("  | stratum | decisions | rule | ranker | oracle | gained | lost |")
            lines.append("  | --- | --- | --- | --- | --- | --- | --- |")
            for stratum, c in result["by_stratum"].items():
                lines.append(f"  | {stratum} | {c.get('decisions', 0)} | {c.get('rule', 0)} | {c.get('ranker', 0)} | {c.get('oracle', 0)} | {c.get('ranker_gained', 0)} | {c.get('ranker_lost', 0)} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", required=True, nargs="+", help="states.jsonl of one or more audit runs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    states: list[dict[str, Any]] = []
    for path in args.states:
        states.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    rows = decision_rows(states)
    report: dict[str, Any] = {"contract": RANKER_CONTRACT, "states": len(states), "kinds": {}}
    for tool in DECISION_TOOLS:
        report["kinds"][tool] = evaluate_kind(rows[tool], tool=tool, seed=args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ranker_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "ranker_report.md").write_text(markdown(report) + "\n", encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
