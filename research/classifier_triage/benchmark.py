"""IEEE 14 offline benchmark of WLS-only triage classifiers against the physics screen.

    python -m research.classifier_triage.benchmark --output-dir output/classifier_triage_20261004

Every classifier answers the same two questions on the same alarmed states
(the step-4 study rows, split by physical parent): request phase-resolved
measurements or not, and which balanced family to investigate first.  Labels
are truth-derived (decision T1); the request decision is thresholded on the
classifier's score (decision G1), with the threshold set on the calibration
split at the physics rule's recall, so every row of the table is read at the
same recall.

Classifiers:

* ``screen_rule``: the exhaustive balanced screen's rule v3 (HIF won, a
  voltage meter set aside, or unexplained): the reference, about 1.1 s;
* ``gbm_screen``: gradient-boosted trees on the screen's outputs (step 4);
* ``gbm_wls``: the same model on the 45 offline features of the WLS alone;
* ``gbm_llm_view``: the same model on what the agent's prompt shows of a WLS
  solve (the information-matched bound for an LLM that reads the prompt);
* ``gbm_prompt_top10_signed``, ``gbm_prompt_top20_signed``: the same on richer
  prompts (the ten or twenty largest residuals with their signs);
* ``gbm_full_vector``: the same model on every channel's residual and every
  multiplier as one flat vector (nothing hidden, no graph structure, fixed to
  this network's size);
* ``gnn_residual``: the size-agnostic GNN on residual-only graph features;
* ``gnn_values``: the GNN with observed and fitted values added (ablation);
* ``gnn_residual_decorrelated``: ``gnn_residual`` trained with the
  background probe rows of train parents (bad meters on healthy OpenDSS and
  OPF windows) as extra negatives.

The background probe (``data.build_probe``) asks what a classifier keys on:
bad power meters on healthy OpenDSS references and on clean OPF windows need
no phasors on either background, so the two request rates should match.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import data, features  # noqa: E402
from research.hypothesis_ranking import ranker  # noqa: E402

SEED = 20261004
FAMILIES = ("measurement", "parameter", "topology", "hif", "unbalance", "harmonic")
SCREEN_PREFIXES = ("screen_valid", "s1_", "w1_", "n_accepted", "acc0_", "acc1_", "explained", "unexplained", "suspected",
                   "vm_channels_n", "n_rounds", "ct1_", "final_", "hif_alpha", "hif_G_log1p")
SCREEN_SECONDS = 1.1  # median on the 144 alarmed development roots (output/hypothesis_ranking_20260930/screen_cost)


def screen_derived(name: str) -> bool:
    return name.startswith(SCREEN_PREFIXES) or (name.endswith("_rank_gap_log") and name != "lambda_top_gap_log")


def family_label(row: Mapping[str, Any], family: str) -> int:
    return int(bool((row["record"].get("truth") or {}).get(family)))


# ------------------------------------------------------------- tabular models

def gbm_scores(feature_of: Callable[[Mapping[str, Any]], Mapping[str, float]], train: Sequence[Mapping[str, Any]],
               groups: Mapping[str, Sequence[Mapping[str, Any]]], names: Sequence[str] | None = None) -> dict[str, Any]:
    """Gradient-boosted ``needs_aux`` and family scores for each row group (the step-4 model settings)."""
    vectors = {id(row): feature_of(row) for row in list(train) + [r for group in groups.values() for r in group]}
    names = list(names) if names is not None else sorted(vectors[id(train[0])])

    def matrix(rows):
        return np.asarray([[float(vectors[id(row)].get(name, 0.0)) for name in names] for row in rows], dtype=float)

    x_train = matrix(train)
    targets = {"needs_aux": np.asarray([row["triage"]["needs_aux"] for row in train])}
    targets.update({family: np.asarray([family_label(row, family) for row in train]) for family in FAMILIES})
    started = time.perf_counter()
    out: dict[str, Any] = {name: {} for name in groups}
    for target, y in targets.items():
        fitted = ranker.fit_models(x_train, y, SEED).get("gbm")
        for name, rows in groups.items():
            out[name][target] = (fitted.predict_proba(matrix(rows))[:, 1] if fitted is not None and rows
                                 else np.zeros(len(rows)))
    # The first balanced family: one multiclass model on the rows that have one.
    from sklearn.ensemble import HistGradientBoostingClassifier

    labelled = [i for i, row in enumerate(train) if row["triage"]["first"] in data.BALANCED_FAMILIES]
    first_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                                 l2_regularization=1.0, random_state=SEED)
    first_model.fit(x_train[labelled], np.asarray([data.BALANCED_FAMILIES.index(train[i]["triage"]["first"]) for i in labelled]))
    for name, rows in groups.items():
        probabilities = np.zeros((len(rows), len(data.BALANCED_FAMILIES)))
        if rows:
            probabilities[:, first_model.classes_] = first_model.predict_proba(matrix(rows))
        out[name]["first"] = probabilities
    fit_seconds = time.perf_counter() - started
    one = matrix(train[:1])
    started = time.perf_counter()
    for _ in range(20):
        fitted.predict_proba(one)
    out["_meta"] = {"features": len(names), "fit_seconds": fit_seconds,
                    "seconds_per_decision": (time.perf_counter() - started) / 20}
    return out


# -------------------------------------------------------------------- metrics

def request_study(cal: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], s_cal: np.ndarray,
                  s_test: np.ndarray) -> dict[str, Any]:
    """The step-4 acquisition study: recall and unneeded requests at the rule's calibration recall and false rate."""
    study = ranker.acquisition_study(list(cal), list(test), np.asarray(s_cal, float), np.asarray(s_test, float), SEED)
    keep = ("auc", "rule_recall", "rule_false_rate", "learned_recall_at_rule_recall", "learned_false_rate_at_rule_recall",
            "learned_recall_at_rule_false_rate", "learned_false_rate_at_rule_false_rate", "thresholds", "by_family",
            "n_test_positive", "n_test_no_phasor_roots")
    return {key: study[key] for key in keep}


def probe_study(probe: Sequence[Mapping[str, Any]], flags: np.ndarray) -> dict[str, Any]:
    """Request rates on the bad-meter probes, one study per probe kind."""
    flags = np.asarray(flags, dtype=float)
    kinds = sorted({str(row.get("probe_kind") or "meter") for row in probe})
    study = {}
    for kind in kinds:
        positions = [i for i, row in enumerate(probe) if str(row.get("probe_kind") or "meter") == kind]
        study[kind] = _probe_kind_study([probe[i] for i in positions], flags[positions])
    return study


def _probe_kind_study(probe: Sequence[Mapping[str, Any]], flags: np.ndarray) -> dict[str, Any]:
    """Request rate on one probe kind, by background; the difference is the background effect."""
    background = np.asarray([row["family"].endswith("opendss") for row in probe])
    parents = [str(row["parent"]) for row in probe]

    def rate(indices, mask):
        chosen = indices[mask[indices]]
        return float(flags[chosen].mean()) if chosen.size else None

    def difference(indices):
        a, b = rate(indices, background), rate(indices, ~background)
        return None if a is None or b is None else a - b

    return {
        "opendss_request_rate": ranker.parent_bootstrap(parents, lambda i: rate(i, background), seed=SEED),
        "opf_request_rate": ranker.parent_bootstrap(parents, lambda i: rate(i, ~background), seed=SEED),
        "background_effect": ranker.parent_bootstrap(parents, difference, seed=SEED),
        "n_opendss": int(background.sum()), "n_opf": int((~background).sum()),
    }


def order_study(test: Sequence[Mapping[str, Any]], pick: Sequence[str | None]) -> dict[str, Any]:
    """First balanced family on test rows that need no phasors: hit when the pick is a true family."""
    rows = [(row, choice) for row, choice in zip(test, pick)
            if row["triage"]["needs_aux"] == 0 and row["triage"]["families"]]
    parents = [str(row["parent"]) for row, _ in rows]
    hit = np.asarray([choice in row["triage"]["families"] for row, choice in rows], dtype=float)
    branch = {"parameter", "topology"}
    coarse = np.asarray([
        (choice == "measurement" and "measurement" in row["triage"]["families"])
        or (choice in branch | {"branch"} and bool(branch & set(row["triage"]["families"])))
        for row, choice in rows
    ], dtype=float)
    oracle = np.asarray([choice == row["triage"]["first"] for row, choice in rows], dtype=float)
    mixed = np.asarray([len(row["triage"]["families"]) > 1 for row, _ in rows])
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({row["family"] for row, _ in rows}):
        mask = np.asarray([row["family"] == family for row, _ in rows])
        by_family[family] = {"n": int(mask.sum()), "hit": float(hit[mask].mean()), "coarse_hit": float(coarse[mask].mean())}
    mean_of = lambda values: (lambda i: float(values[i].mean()) if i.size else None)  # noqa: E731
    return {
        "hit": ranker.parent_bootstrap(parents, mean_of(hit), seed=SEED),
        "meter_or_branch_hit": ranker.parent_bootstrap(parents, mean_of(coarse), seed=SEED),
        "oracle_first_agreement_mixed": (float(oracle[mixed].mean()) if mixed.any() else None),
        "n": len(rows), "n_mixed": int(mixed.sum()), "by_family": by_family,
    }


def screen_first_pick(row: Mapping[str, Any]) -> str | None:
    screen = (row["record"].get("analysis") or {}).get("screen") or {}
    rounds = [r for r in (screen.get("rounds") or []) if isinstance(r, Mapping) and r.get("winner")]
    winner = str(rounds[0]["winner"]) if rounds else None
    return {"meter": "measurement", "parameter": "parameter", "topology": "topology"}.get(winner)


def dominance_pick(row: Mapping[str, Any]) -> str:
    wls = (row["record"].get("analysis") or {}).get("wls") or {}
    return "measurement" if wls.get("measurement_dominant") or not wls.get("branch_dominant") else "branch"


def score_pick(scores: Mapping[str, np.ndarray], index: int) -> str:
    """The first-family head's choice, else the largest balanced family score."""
    if "first" in scores:
        return data.BALANCED_FAMILIES[int(np.argmax(scores["first"][index]))]
    return max(data.BALANCED_FAMILIES, key=lambda name: float(scores[name][index]))


def family_auc(test: Sequence[Mapping[str, Any]], scores: Mapping[str, np.ndarray]) -> dict[str, float | None]:
    return {family: ranker.auc(np.asarray(scores[family], float), np.asarray([family_label(row, family) for row in test]))
            for family in FAMILIES if family in scores}


def evaluate(name: str, groups: Mapping[str, Sequence[Mapping[str, Any]]], scores: Mapping[str, Mapping[str, np.ndarray]],
             seconds: float | None) -> dict[str, Any]:
    """All studies for one scored classifier; ``scores[group][target]`` aligns with ``groups[group]``."""
    cal, test, probe = groups["calibration"], groups["test"], groups["probe"]
    request = request_study(cal, test, scores["calibration"]["needs_aux"], scores["test"]["needs_aux"])
    threshold = float(request["thresholds"]["at_rule_recall"])
    result: dict[str, Any] = {"name": name, "request": request, "seconds_per_decision": seconds}
    if probe:
        result["probe"] = probe_study(probe, np.asarray(scores["probe"]["needs_aux"]) >= threshold)
    if all(family in scores["test"] for family in data.BALANCED_FAMILIES):
        result["order"] = order_study(test, [score_pick(scores["test"], i) for i in range(len(test))])
        result["family_auc"] = family_auc(test, scores["test"])
    if "hif" in scores["test"]:
        mimic = ranker.hif_mimic_study(list(cal), list(test), np.asarray(scores["calibration"]["hif"], float),
                                       np.asarray(scores["test"]["hif"], float), SEED)
        result["hif_vs_mimic"] = {key: mimic[key] for key in (
            "screen_hif_recall", "learned_hif_recall_at_threshold", "screen_same_sign_mimic_flag_rate",
            "learned_same_sign_mimic_flag_rate", "auc_hif_vs_mimics", "n_test_hif", "n_test_same_sign_mimics")}
    return result


def evaluate_hard(name: str, groups: Mapping[str, Sequence[Mapping[str, Any]]], test_flags: Sequence[float],
                  probe_flags: Sequence[float], picks: Sequence[str | None], seconds: float | None) -> dict[str, Any]:
    """Studies for a classifier that makes decisions, not scores: request flags and first-family picks."""
    test, probe = groups["test"], groups["probe"]
    flags = np.asarray(test_flags, dtype=float)
    y = np.asarray([row["triage"]["needs_aux"] for row in test])
    negative = np.asarray([row["needs_no_phasors_family"] for row in test])
    parents = [str(row["parent"]) for row in test]
    by_family: dict[str, dict[str, int]] = {}
    for family in sorted({row["family"] for row in test}):
        mask = np.asarray([row["family"] == family for row in test])
        by_family[family] = {"n": int(mask.sum()), "requested": int(flags[mask].sum())}
    result: dict[str, Any] = {
        "name": name, "hard": True, "seconds_per_decision": seconds,
        "request": {
            "recall": ranker.parent_bootstrap(parents, lambda i: float(flags[i[y[i] == 1]].mean()), seed=SEED),
            "unneeded": ranker.parent_bootstrap(parents, lambda i: float(flags[i[negative[i]]].mean()), seed=SEED),
            "n_test_positive": int((y == 1).sum()), "n_test_no_phasor_roots": int(negative.sum()), "by_family": by_family,
        },
        "order": order_study(test, list(picks)),
    }
    if probe:
        result["probe"] = probe_study(probe, np.asarray(probe_flags, dtype=float))
    return result


def evaluate_rule(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """The physics screen: its rule on the test split and the probe, its first-round winner as the family pick."""
    test, probe = groups["test"], groups["probe"]
    result = evaluate_hard("screen_rule", groups, [row["rule_v3"] for row in test], [row["rule_v3"] for row in probe],
                           [screen_first_pick(row) for row in test], SCREEN_SECONDS)
    result["order_dominance_rule"] = order_study(test, [dominance_pick(row) for row in test])
    return result


def evaluate_llm(name: str, groups: Mapping[str, Sequence[Mapping[str, Any]]], path: Path) -> dict[str, Any]:
    """An LLM's first actions (``llm_score``): a request is the phasor tool, a pick is a balanced context."""
    decisions = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]

    def decision(row: Mapping[str, Any]) -> str:
        return str((decisions.get(str(row["id"])) or {}).get("decision") or "missing")

    test = groups["test"]
    # Every test row must be scored (a missing one counts as no request); the probe is read on the scored subset.
    probe = [row for row in groups["probe"] if str(row["id"]) in decisions]
    seconds = [float(item["seconds"]) for item in decisions.values() if item.get("seconds") is not None]
    result = evaluate_hard(name, {**groups, "probe": probe}, [decision(row) == "request" for row in test],
                           [decision(row) == "request" for row in probe],
                           [decision(row) if decision(row) in data.BALANCED_FAMILIES else None for row in test],
                           float(np.median(seconds)) if seconds else None)
    counts: dict[str, int] = {}
    for row in list(test) + list(probe):
        counts[decision(row)] = counts.get(decision(row), 0) + 1
    result["decisions"] = counts
    result["probe_rows_scored"] = len(probe)
    return result


# --------------------------------------------------------------------- report

def _ci(value: Mapping[str, Any] | None, percent: bool = True) -> str:
    if not isinstance(value, Mapping) or value.get("point") is None:
        return "n/a"
    scale, unit = (100.0, "%") if percent else (1.0, "")
    text = f"{scale * value['point']:.1f}{unit}"
    if value.get("low") is not None:
        text += f" [{scale * value['low']:.1f}, {scale * value['high']:.1f}]"
    return text


def write_report(path: Path, results: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> None:
    lines = ["# IEEE 14 triage benchmark: WLS-only classifiers against the physics screen\n",
             f"Rows: train {meta['train']} (with children), calibration {meta['calibration']}, test {meta['test']} "
             f"({meta['test_positive']} need phase-resolved measurements, {meta['test_negative']} need none); "
             f"probe {meta['probe']} bad-meter rows on healthy backgrounds (parents outside the train split). "
             "Brackets are 95% parent-bootstrap intervals. Scored classifiers are thresholded on the calibration "
             "split at the screen rule's recall; the screen rule and an LLM's first action are read at their own decision.\n",
             "## Request decision\n",
             "| classifier | AUC | recall | unneeded requests | seconds per alarm |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        req = r["request"]
        seconds = r.get("seconds_per_decision")
        cost = "n/a" if seconds is None else (f"{seconds:.2f}" if seconds >= 0.01 else f"{seconds * 1000:.1f} ms")
        if r.get("hard"):
            lines.append(f"| {r['name']} | n/a | {_ci(req['recall'])} | {_ci(req['unneeded'])} | {cost} |")
        else:
            lines.append(f"| {r['name']} | {_ci(req['auc'])} | {_ci(req['learned_recall_at_rule_recall'])} | "
                         f"{_ci(req['learned_false_rate_at_rule_recall'])} | {cost} |")
    titles = {"meter": "bad power meter, 10 to 20 sigma", "vm_large": "bad voltage meter, 10 to 20 sigma",
              "vm_small": "bad voltage meter, 4.5 to 9 sigma (below the training population)"}
    kinds = sorted({kind for r in results for kind in (r.get("probe") or {})})
    for kind in kinds:
        sizes = next(r["probe"][kind] for r in results if kind in (r.get("probe") or {}))
        lines += ["", f"## Probe on healthy windows: {titles.get(kind, kind)}\n",
                  f"No row needs phasors ({sizes['n_opendss']} on OpenDSS references, {sizes['n_opf']} on OPF windows). "
                  "A classifier that reads the event requests equally often on both backgrounds.\n",
                  "| classifier | requests, OpenDSS background | requests, OPF background | difference |",
                  "| --- | --- | --- | --- |"]
        for r in results:
            probe = (r.get("probe") or {}).get(kind)
            if probe:
                lines.append(f"| {r['name']} | {_ci(probe['opendss_request_rate'])} | {_ci(probe['opf_request_rate'])} | "
                             f"{_ci(probe['background_effect'])} |")
    lines += ["", "## First balanced family (test rows that need no phasors)\n",
              "| classifier | pick is a true family | meter-or-branch correct | agrees with the oracle order on mixed roots |",
              "| --- | --- | --- | --- |"]
    for r in results:
        for key, label in (("order", r["name"]), ("order_dominance_rule", "residual-versus-multiplier rule")):
            order = r.get(key)
            if order:
                agreement = order.get("oracle_first_agreement_mixed")
                lines.append(f"| {label} | {_ci(order['hit']) if key == 'order' else 'n/a'} | {_ci(order['meter_or_branch_hit'])} | "
                             f"{'n/a' if agreement is None or key != 'order' else f'{100 * agreement:.1f}%'} |")
    lines += ["", "## HIF score against same-sign flow-meter pairs\n",
              "| classifier | HIF recall | same-sign pairs flagged | AUC, HIF against pairs |", "| --- | --- | --- | --- |"]
    for r in results:
        mimic = r.get("hif_vs_mimic")
        if mimic:
            lines.append(f"| {r['name']} | {_ci(mimic['learned_hif_recall_at_threshold'])} | "
                         f"{_ci(mimic['learned_same_sign_mimic_flag_rate'])} | {_ci(mimic['auc_hif_vs_mimics'])} |")
    first = next((r.get("hif_vs_mimic") for r in results if r.get("hif_vs_mimic")), None)
    if first:
        lines.append(f"| screen_rule | {_ci(first['screen_hif_recall'])} | {_ci(first['screen_same_sign_mimic_flag_rate'])} | n/a |")
    lines += ["", "## Family scores (diagnostic)\n",
              "Test AUC of each family head. Unbalance, harmonic and a bad voltage meter are not separable on balanced "
              "data except through simulator artifacts, and the topology roots carry a corpus format cue, so these "
              "numbers are not a claim of fault-type classification.\n",
              "| classifier | " + " | ".join(FAMILIES) + " |", "| --- |" + " --- |" * len(FAMILIES)]
    for r in results:
        aucs = r.get("family_auc")
        if aucs:
            lines.append(f"| {r['name']} | " + " | ".join("n/a" if aucs.get(f) is None else f"{100 * aucs[f]:.1f}%" for f in FAMILIES) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


# ----------------------------------------------------------------------- main

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(data.DEFAULT_DATASET))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--probe-replicas", type=int, default=2)
    parser.add_argument("--probe-limit", type=int, default=600, help="probe attempts per background")
    parser.add_argument("--probe-vm-limit", type=int, default=600, help="voltage-meter probe attempts per background and kind")
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--reuse-gnn", action="store_true", help="load saved GNN scores instead of training again")
    parser.add_argument("--llm-scores", action="append", default=[], metavar="NAME=PATH",
                        help="an LLM's first actions from llm_score (repeatable)")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rows = data.load_rows(Path(args.dataset_dir), workers=args.workers)
    split = data.split_rows(rows)
    probe_path = out / "probe.jsonl"
    if not probe_path.is_file():
        records = data.build_probe(Path(args.dataset_dir), replicas=args.probe_replicas,
                                   limit_per_background=args.probe_limit, workers=args.workers)
        with probe_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(_json_safe(record)) + "\n")
    with probe_path.open(encoding="utf-8") as stream:
        probe_all = data.probe_rows(json.loads(line) for line in stream if line.strip())
    # The voltage-meter kinds are evaluation only (the decorrelated variant trains on the power-meter kind).
    for kind in ("vm_large", "vm_small"):
        kind_path = out / f"probe_{kind}.jsonl"
        if not kind_path.is_file():
            records = data.build_probe(Path(args.dataset_dir), replicas=args.probe_replicas, kind=kind,
                                       limit_per_background=args.probe_vm_limit, workers=args.workers)
            with kind_path.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(_json_safe(record)) + "\n")
        with kind_path.open(encoding="utf-8") as stream:
            probe_all += data.probe_rows(json.loads(line) for line in stream if line.strip())
    probe_all = data.attach_payloads(probe_all, out / "probe_payloads.pkl", workers=args.workers)
    probe = [row for row in probe_all if row["split"] != "train"]
    probe_train = [row for row in probe_all if row["split"] == "train" and row["probe_kind"] == "meter"]
    groups = {"calibration": split["calibration"], "test": split["test"], "probe": probe}
    print(f"[benchmark] rows train {len(split['train'])}, calibration {len(split['calibration'])}, test {len(split['test'])}; "
          f"probe {len(probe)} scored + {len(probe_train)} on train parents ({time.perf_counter() - started:.0f} s)", flush=True)

    results: list[dict[str, Any]] = [evaluate_rule(groups)]
    offline_names = sorted(rows[0]["features"])
    tabular = {
        "gbm_screen": (lambda row: row["features"], offline_names),
        "gbm_wls": (lambda row: row["features"], [n for n in offline_names if not screen_derived(n)]),
        "gbm_llm_view": (lambda row: features.llm_visible_features(str(row["record"]["case"]), row["payload"]), None),
        "gbm_prompt_top10_signed": (lambda row: features.llm_visible_features(
            str(row["record"]["case"]), row["payload"], 10, signed=True), None),
        "gbm_prompt_top20_signed": (lambda row: features.llm_visible_features(
            str(row["record"]["case"]), row["payload"], 20, signed=True), None),
        "gbm_full_vector": (lambda row: features.flat_features(str(row["record"]["case"]), row["payload"]), None),
    }
    for name, (feature_of, names) in tabular.items():
        scores = gbm_scores(feature_of, split["train"], groups, names)
        meta = scores.pop("_meta")
        results.append({**evaluate(name, groups, scores, meta["seconds_per_decision"]), "features": meta["features"]})
        print(f"[benchmark] {name}: {meta['features']} features, fitted in {meta['fit_seconds']:.0f} s", flush=True)

    if not args.skip_gnn:
        from research.classifier_triage import train_gnn

        variants = {"gnn_residual": ("residual", ()), "gnn_values": ("values", ()),
                    "gnn_residual_decorrelated": ("residual", probe_train)}
        for name, (view, augment) in variants.items():
            saved = out / name / "scores.json"
            if args.reuse_gnn and saved.is_file():
                scored = train_gnn.load_scores(saved)
                known = set(scored["ids"])
                fresh = [row for group_rows in groups.values() for row in group_rows if str(row["id"]) not in known]
                if fresh:  # rows added after training (new probe kinds): score them with the saved models
                    extra_scores = train_gnn.score_with_checkpoints(out / name / "checkpoints", fresh, view)
                    scored["ids"] = list(scored["ids"]) + [str(row["id"]) for row in fresh]
                    for key in ("needs_aux", "family", "first"):
                        scored[key] = np.concatenate((scored[key], extra_scores[key]))
            else:
                scored = train_gnn.train_and_score(rows, probe, view=view, seeds=range(args.seeds), device=args.device,
                                                   augment=augment, checkpoint_dir=out / name / "checkpoints")
                train_gnn.save_scores(scored, saved)
            index = {row_id: position for position, row_id in enumerate(scored["ids"])}
            scores = {}
            for group, group_rows in groups.items():
                positions = [index[str(row["id"])] for row in group_rows]
                scores[group] = {"needs_aux": scored["needs_aux"][positions], "first": scored["first"][positions]}
                scores[group].update({family: scored["family"][positions, k] for k, family in enumerate(FAMILIES)})
            evaluated = evaluate(name, groups, scores, scored["seconds_per_decision_cpu"])
            by_seed = [ranker.auc(np.asarray(scored["needs_aux_by_seed"])[[index[str(r["id"])] for r in split["test"]], k],
                                  np.asarray([r["triage"]["needs_aux"] for r in split["test"]]))
                       for k in range(len(scored["seeds"]))]
            results.append({**evaluated, "parameters": scored["parameters"], "auc_by_seed": by_seed,
                            "history": scored["history"], "train_rows": scored["train_rows"]})
            print(f"[benchmark] {name}: AUC by seed {[round(a, 4) for a in by_seed]}", flush=True)

    for item in args.llm_scores:
        name, _, path = item.partition("=")
        results.append(evaluate_llm(name, groups, Path(path)))
        print(f"[benchmark] {name}: decisions {results[-1]['decisions']}", flush=True)

    meta = {"train": len(split["train"]), "calibration": len(split["calibration"]), "test": len(split["test"]),
            "test_positive": int(sum(r["triage"]["needs_aux"] for r in split["test"])),
            "test_negative": int(sum(r["needs_no_phasors_family"] for r in split["test"])),
            "probe": len(probe), "probe_train_parents": len(probe_train), "seed": SEED,
            "dataset_dir": str(args.dataset_dir), "seconds": time.perf_counter() - started}
    (out / "benchmark.json").write_text(json.dumps(_json_safe({"meta": meta, "results": results}), indent=1), encoding="utf-8")
    write_report(out / "report.md", results, meta)
    print(f"[benchmark] wrote {out / 'report.md'} in {meta['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
