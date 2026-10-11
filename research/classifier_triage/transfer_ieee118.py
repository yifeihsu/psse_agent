"""Zero-shot transfer of the IEEE 14 triage GNN to IEEE 118, and where it breaks.

    python -m research.classifier_triage.transfer_ieee118 --dataset-dir output/classifier_triage_20261007/ieee118 \\
        --output-dir output/classifier_triage_20261007/transfer

Every alarmed IEEE 118 row of ``ieee118_dataset`` (admitted roots, healthy
windows and flow-pair mimics that alarm) is scored by the tracked IEEE 14
export (``psse_env/oracle/models/triage_gnn_ieee14_20261004``) with its
weights and request threshold unchanged, and reported like the IEEE 14
benchmark: recall on the roots that need phase-resolved data (HIF, unbalance,
harmonic) and the request rate on the rows that do not.  The IEEE 14 test
split is scored through the same code as the reference.

Each candidate cause is then probed by changing one thing at inference:

* ``recalibrated``: the request threshold re-chosen on the IEEE 118
  calibration split for the IEEE 14 recall target (ranking versus
  calibration; the threshold-free AUC answers the same question);
* ``globals14``: the five graph-level inputs re-expressed on IEEE 14's
  degrees of freedom and channel count (the chi-square ratio of the same
  excess objective, the >3 and >4 sigma counts over 122 channels);
* ``subgraph14``: the 14-bus breadth-first neighbourhood of the most anomalous
  bus, with ``globals14``, so mean pooling sees an IEEE-14-sized graph;
* ``magnitude10``: residuals and multipliers scaled so the largest is at most
  10.2 (the IEEE 14 HIF training rows' 95th percentile), the objective's excess
  by the square;
* ``static14``: degree, shunts and branch r, x, b, tap clamped to the IEEE 14
  network's range;
* ``all``: every change at once.

Last, the same GNN is trained on the IEEE 118 train split (in-domain ceiling)
and on IEEE 14 + IEEE 118 together, and scored on both test splits: if the
ceiling is high, the information is in the WLS solve and the loss is a domain
shift; if not, the IEEE 118 WLS evidence itself is ambiguous.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.wls_admission import admitted as admission_admitted  # noqa: E402
from research.classifier_triage import data, features, train_gnn  # noqa: E402
from research.classifier_triage.model import FAMILY_HEADS, FIRST_CLASSES  # noqa: E402
from research.classifier_triage.runtime import DEFAULT_TRIAGE_CLASSIFIER, load_triage_classifier  # noqa: E402
from research.hypothesis_ranking import ranker  # noqa: E402

SEED = 20261007
IEEE14_NZ, IEEE14_DOF = 122, 95
IEEE14_HIF_MAX_RESIDUAL_Q95 = 10.2
SUBGRAPH_BUSES = 14
WAVEFORM = ("hif", "measurement+hif", "three_phase_unbalance", "harmonic")
ROW_FILES = (("roots.jsonl", "root"), ("healthy_alarms.jsonl", "healthy"), ("mimic.jsonl", "mimic"))


# ------------------------------------------------------------------------ rows

def load_ieee118_rows(dataset_dir: Path, *, include_alarm_only: bool = False) -> list[dict[str, Any]]:
    """Alarmed IEEE 118 rows with truth-derived triage labels and their saved WLS payloads."""
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "payloads.pkl").open("rb") as stream:
        payloads = pickle.load(stream)
    sources = list(ROW_FILES) + ([("alarm_only.jsonl", "alarm_only")] if include_alarm_only else [])
    rows = []
    for name, kind in sources:
        for record in ranker._read(dataset_dir / name):
            if not ranker.alarmed(record):
                continue
            payload = payloads.get(str(record["root_id"]))
            if payload is None:
                continue
            rows.append({"id": str(record["root_id"]), "kind": kind, "family": record["family"],
                         "parent": str(record["parent_id"]), "split": record.get("split") or "test", "record": record,
                         "triage": data.triage_labels(record), "payload": payload})
    return rows


def ieee14_test_rows(workers: int) -> list[dict[str, Any]]:
    return [row for row in data.load_rows(workers=workers) if row["split"] == "test" and row["kind"] != "child"]


def admitted_rows(rows: Sequence[dict[str, Any]], rule: str | None) -> list[dict[str, Any]]:
    """Roots re-filtered by an admission rule on their own solve; healthy windows, mimics and children keep the operator alarm.

    The IEEE 14 study was generated under the chi-square-margin rule; re-filtering it by the
    excess-energy rule gives the population that rule would have admitted (on IEEE 14 the two
    rules differ only between excess 67.5 and 70).
    """
    if rule is None:
        return list(rows)
    kept = []
    for row in rows:
        if row["kind"] != "root":
            kept.append(row)
            continue
        payload = row["payload"]
        max_r = float(np.max(np.abs(payload["signed_normalized_residual"])))
        if admission_admitted(float(payload["objective"]), int(payload["dof"]), max_r, rule=rule):
            kept.append(row)
    return kept


# --------------------------------------------------------------------- scoring

def export_models(device: str):
    classifier = load_triage_classifier(DEFAULT_TRIAGE_CLASSIFIER)
    return classifier, [copy.deepcopy(model).to(device).eval() for model in classifier.models]


def score_graphs(models: Sequence[Any], graphs: Sequence[Mapping[str, np.ndarray]], device: str) -> dict[str, np.ndarray]:
    """Mean of the seeds' probabilities (the runtime's score) on prebuilt graphs."""
    runs = [train_gnn._predict(model, list(graphs), device) for model in models]
    return {"needs_aux": np.mean([r[0] for r in runs], axis=0), "family": np.mean([r[1] for r in runs], axis=0),
            "first": np.mean([r[2] for r in runs], axis=0)}


def graph_of(row: Mapping[str, Any], view: str = "residual") -> dict[str, np.ndarray]:
    return features.build_graph(str(row["record"]["case"]), row["payload"], view)


# --------------------------------------------------------------- interventions

def globals14(graph: Mapping[str, np.ndarray], payload: Mapping[str, Any], nz: int) -> dict[str, np.ndarray]:
    """The five graph-level inputs re-expressed on IEEE 14's dof and channel count (same excess objective)."""
    out = dict(graph)
    u = np.array(graph["u"], dtype=float, copy=True)
    excess = float(payload["objective"]) - float(payload["dof"])
    threshold14 = float(chi2.ppf(1.0 - features.CHI2_ALPHA, IEEE14_DOF))
    u[0] = np.log(max(IEEE14_DOF + excess, 1e-9) / threshold14)
    u[2] = min(1.0, u[2] * nz / IEEE14_NZ)
    u[3] = min(1.0, u[3] * nz / IEEE14_NZ)
    out["u"] = u
    return out


def _center_bus(graph: Mapping[str, np.ndarray], payload: Mapping[str, Any], net: Mapping[str, Any]) -> int:
    """The bus whose own channels or incident branch channels hold the largest |normalized residual|."""
    nb, nl = int(net["nb"]), int(net["nl"])
    magnitude = np.abs(features.neutral_residuals(payload, net))
    score = np.max(np.stack([magnitude[:nb], magnitude[nb:2 * nb], magnitude[2 * nb:3 * nb]]), axis=0)
    flows = np.max(np.stack([magnitude[3 * nb + k * nl:3 * nb + (k + 1) * nl] for k in range(4)]), axis=0)
    np.maximum.at(score, net["source"], flows)
    np.maximum.at(score, net["destination"], flows)
    return int(np.argmax(score))


def subgraph(graph: Mapping[str, np.ndarray], center: int, size: int = SUBGRAPH_BUSES) -> dict[str, np.ndarray]:
    """The ``size``-bus breadth-first neighbourhood of ``center`` with every branch inside it."""
    edge_index = np.asarray(graph["edge_index"])
    neighbours: dict[int, list[int]] = {}
    for a, b in edge_index.T:
        neighbours.setdefault(int(a), []).append(int(b))
    order, seen, queue = [], {center}, deque([center])
    while queue and len(order) < size:
        node = queue.popleft()
        order.append(node)
        for other in sorted(set(neighbours.get(node, []))):
            if other not in seen:
                seen.add(other)
                queue.append(other)
    keep = np.asarray(sorted(order), dtype=np.int64)
    position = {int(node): k for k, node in enumerate(keep)}
    inside = np.asarray([int(a) in position and int(b) in position for a, b in edge_index.T])
    pairs = np.asarray(graph["edge_pair"])[inside]
    relabel = {int(p): k for k, p in enumerate(dict.fromkeys(pairs.tolist()))}
    return {"x": np.asarray(graph["x"])[keep], "u": np.asarray(graph["u"]),
            "edge_index": np.asarray([[position[int(a)] for a in edge_index[0][inside]],
                                      [position[int(b)] for b in edge_index[1][inside]]], dtype=np.int64).reshape(2, -1),
            "edge_attr": np.asarray(graph["edge_attr"])[inside],
            "edge_pair": np.asarray([relabel[int(p)] for p in pairs], dtype=np.int64)}


def magnitude_capped(payload: Mapping[str, Any], cap: float = IEEE14_HIF_MAX_RESIDUAL_Q95) -> dict[str, Any]:
    """Residuals and multipliers scaled so the largest residual is at most ``cap``; the objective's excess by the square."""
    residual = np.asarray(payload["signed_normalized_residual"], dtype=float)
    factor = min(1.0, cap / max(float(np.max(np.abs(residual))), 1e-12))
    out = dict(payload)
    out["signed_normalized_residual"] = residual * factor
    out["lambda_normalized"] = np.asarray(payload["lambda_normalized"], dtype=float) * factor
    dof = float(payload["dof"])
    out["objective"] = dof + factor ** 2 * (float(payload["objective"]) - dof)
    return out


def static_ranges(case: str = "case14") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    net = features.network(case)
    return {"node": (net["bus_static"].min(axis=0), net["bus_static"].max(axis=0)),
            "edge": (net["branch_static"].min(axis=0), net["branch_static"].max(axis=0))}


def static_clamped(graph: Mapping[str, np.ndarray], ranges: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Degree, shunts, r, x, b and tap clamped to the IEEE 14 range (one-hots and status untouched)."""
    node_names = features.NODE_FEATURES["residual"]
    edge_names = features.EDGE_FEATURES["residual"]
    node_static0 = node_names.index("bus_pq")
    edge_static0 = edge_names.index("r_pu")
    out = dict(graph)
    x = np.array(graph["x"], dtype=float, copy=True)
    for name in ("log1p_degree", "g_shunt_pu", "b_shunt_pu"):
        column = node_names.index(name)
        low, high = ranges["node"][0][column - node_static0], ranges["node"][1][column - node_static0]
        x[:, column] = np.clip(x[:, column], low, high)
    attr = np.array(graph["edge_attr"], dtype=float, copy=True)
    for name in ("r_pu", "x_pu", "b_pu", "tap"):
        column = edge_names.index(name)
        low, high = ranges["edge"][0][column - edge_static0], ranges["edge"][1][column - edge_static0]
        attr[:, column] = np.clip(attr[:, column], low, high)
    out["x"], out["edge_attr"] = x, attr
    return out


def intervention_graphs(rows: Sequence[Mapping[str, Any]], name: str, view: str = "residual") -> list[dict[str, np.ndarray]]:
    """Graphs of ``rows`` under one intervention; the legacy-globals change applies to the legacy view only."""
    ranges = static_ranges()
    graphs = []
    for row in rows:
        case = str(row["record"]["case"])
        net = features.network(case)
        nz = 3 * int(net["nb"]) + 4 * int(net["nl"])
        payload = row["payload"]
        if name in ("magnitude10", "all"):
            payload = magnitude_capped(payload)
        graph = features.build_graph(case, payload, view)
        if view == "residual" and name in ("globals14", "subgraph14", "all"):
            graph = globals14(graph, payload, nz)
        if name in ("subgraph14", "all"):
            graph = subgraph(graph, _center_bus(graph, payload, net))
        if name in ("static14", "all"):
            graph = static_clamped(graph, ranges)
        graphs.append(graph)
    return graphs


# --------------------------------------------------------------------- metrics

def _rate(parents: Sequence[str], flags: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    flags, mask = np.asarray(flags, dtype=float), np.asarray(mask, dtype=bool)
    return ranker.parent_bootstrap(parents, lambda i: float(flags[i][mask[i]].mean()) if mask[i].any() else None, seed=SEED)


def summarize(rows: Sequence[Mapping[str, Any]], score: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray([row["triage"]["needs_aux"] for row in rows], dtype=int)
    parents = [str(row["parent"]) for row in rows]
    flags = np.asarray(score) >= threshold
    out = {"rows": len(rows), "positives": int(y.sum()), "negatives": int((1 - y).sum()), "threshold": float(threshold),
           "recall": _rate(parents, flags, y == 1), "unneeded_request_rate": _rate(parents, flags, y == 0),
           "auc": ranker.parent_bootstrap(parents, lambda i: ranker.auc(np.asarray(score)[i], y[i]), seed=SEED),
           "by_family": {}}
    for family in sorted({row["family"] for row in rows}):
        mask = np.asarray([row["family"] == family for row in rows])
        out["by_family"][family] = {"requested": int(flags[mask].sum()), "n": int(mask.sum())}
    return out


def recall_threshold(score: np.ndarray, y: np.ndarray, target: float) -> float:
    """The largest threshold whose recall on (score, y) reaches ``target``."""
    positives = np.sort(np.asarray(score)[np.asarray(y) == 1])
    if positives.size == 0:
        return 0.5
    allowed = int(np.floor((1.0 - target) * positives.size + 1e-9))
    return float(positives[min(allowed, positives.size - 1)])


def false_rate_threshold(score: np.ndarray, y: np.ndarray, target: float) -> float:
    """The smallest threshold whose request rate on the negatives is at most ``target``."""
    negatives = np.sort(np.asarray(score)[np.asarray(y) == 0])[::-1]
    if negatives.size == 0:
        return 0.5
    allowed = int(np.floor(target * negatives.size + 1e-9))
    return float(np.nextafter(negatives[allowed], np.inf)) if allowed < negatives.size else float(negatives[-1])


def _bin(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def strata(row: Mapping[str, Any]) -> dict[str, str]:
    """Physical and WLS strata of one root (empty for rows without a waveform truth)."""
    truth = row["record"]["truth"]
    admission = row["record"].get("admission") or {}
    out: dict[str, str] = {}
    max_r = float(admission.get("max_normalized_residual") or np.max(np.abs(row["payload"]["signed_normalized_residual"])))
    out["max_residual"] = _bin(max_r, (10, 20, 40), ("<10", "10-20", "20-40", ">=40"))
    ratio = float(admission.get("log_chi_square_ratio") or 0.0)
    out["log_chi_square_ratio"] = _bin(ratio, (0.0, 0.5, 1.0, 2.0), ("<0", "0-0.5", "0.5-1", "1-2", ">=2"))
    if truth.get("hif"):
        hif = truth["hif"][0]
        out["kv"] = f"{float(hif['kv_ll']):g} kV"
        out["r_band_ohm"] = f"{hif['band_low_ohm']:g}-{hif['band_high_ohm']:g}"
        out["r_hif_pu"] = _bin(float(hif["r_hif_pu"]), (1.0, 2.0, 4.0), ("<1", "1-2", "2-4", ">=4"))
    if truth.get("unbalance"):
        out["vuf_at_bus"] = _bin(float(truth["unbalance"][0]["vuf_at_bus"]), (0.01, 0.02, 0.04), ("<1%", "1-2%", "2-4%", ">=4%"))
    if truth.get("harmonic"):
        out["max_thd"] = _bin(float(truth["harmonic"][0]["maximum_voltage_thd"]), (0.2, 0.3, 0.5), ("<0.2", "0.2-0.3", "0.3-0.5", ">=0.5"))
    return out


def stratified(rows: Sequence[Mapping[str, Any]], flags: np.ndarray, families: Sequence[str], key: str) -> dict[str, str]:
    counts: dict[str, list[int]] = {}
    for row, flag in zip(rows, flags):
        if row["family"] not in families:
            continue
        label = strata(row).get(key)
        if label is None:
            continue
        cell = counts.setdefault(label, [0, 0])
        cell[0] += int(flag)
        cell[1] += 1
    return {label: f"{hit}/{n}" for label, (hit, n) in sorted(counts.items())}


def family_heads_on_misses(rows: Sequence[Mapping[str, Any]], scored: Mapping[str, np.ndarray], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in WAVEFORM:
        idx = [i for i, row in enumerate(rows) if row["family"] == family and scored["needs_aux"][i] < threshold]
        out[family] = {"misses": len(idx),
                       "top_family_head": dict(Counter(FAMILY_HEADS[int(np.argmax(scored["family"][i]))] for i in idx)),
                       "first_head": dict(Counter(FIRST_CLASSES[int(np.argmax(scored["first"][i]))] for i in idx))}
    return out


def input_ranges(rows: Sequence[Mapping[str, Any]], graphs: Sequence[Mapping[str, np.ndarray]], view: str = "residual") -> dict[str, Any]:
    """Quantiles of the graph-level inputs and the largest residual by family (5, 50, 95 %)."""
    out: dict[str, Any] = {}
    for family in sorted({row["family"] for row in rows}):
        idx = [i for i, row in enumerate(rows) if row["family"] == family]
        u = np.asarray([graphs[i]["u"] for i in idx])
        out[family] = {name: np.quantile(u[:, k], [0.05, 0.5, 0.95]).round(3).tolist()
                       for k, name in enumerate(features.GLOBAL_FEATURES[view])}
        out[family]["n"] = len(idx)
    return out


def signature(row: Mapping[str, Any]) -> dict[str, float]:
    """How concentrated one alarm is: channels above 3 sigma, the top-2 share of their squared residual, the assets they touch."""
    net = features.network(str(row["record"]["case"]))
    nb, nl = int(net["nb"]), int(net["nl"])
    residual = np.abs(features.neutral_residuals(row["payload"], net))
    hot = np.flatnonzero(residual > 3.0)
    energy = np.sort(residual[hot] ** 2)[::-1]
    assets = set()
    for index in hot:
        if index < 3 * nb:
            assets.add(("bus", int(index % nb)))
        else:
            assets.add(("branch", int((index - 3 * nb) % nl)))
    block = np.searchsorted(np.cumsum([nb, nb, nb, nl, nl, nl, nl]), hot, side="right")  # 0 Vm, 1 Pinj, 2 Qinj, 3.. flows
    reactive = np.isin(block, (2, 4, 6))
    return {"channels_gt3": float(hot.size), "top2_share": float(energy[:2].sum() / energy.sum()) if energy.size else 0.0,
            "assets_gt3": float(len(assets)), "max_residual": float(residual.max()),
            "vm_share_gt3": float(np.mean(block == 0)) if hot.size else 0.0,
            "reactive_share_gt3": float(np.mean(reactive)) if hot.size else 0.0}


def signature_table(rows: Sequence[Mapping[str, Any]], groups: Mapping[str, Callable[[Mapping[str, Any]], bool]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, keep in groups.items():
        values = [signature(row) for row in rows if keep(row)]
        if not values:
            continue
        out[name] = {"n": len(values), **{key: np.quantile([v[key] for v in values], [0.25, 0.5, 0.75]).round(3).tolist()
                                         for key in values[0]}}
    return out


# ------------------------------------------------------------------- ceilings

def in_domain(rows118: Sequence[dict[str, Any]], rows14: Sequence[dict[str, Any]], *, device: str, seeds: Sequence[int],
              target: float, view: str = "residual", log=print) -> dict[str, Any]:
    """The same GNN trained on IEEE 14 alone, IEEE 118 alone and both; every model scored on both test splits.

    Thresholds come from each network's own calibration split at ``target`` recall.  ``ieee14_only``
    scored on IEEE 118 is the forward transfer with freshly trained weights (the export is the
    legacy-view version of it); ``ieee118_only`` scored on IEEE 14 is the reverse transfer.
    """
    def evaluation(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if r["split"] in ("calibration", "test") and r["kind"] != "child"]

    out: dict[str, Any] = {}
    variants = (("ieee14_only", list(rows14), evaluation(rows118)), ("ieee118_only", list(rows118), evaluation(rows14)),
                ("ieee14_plus_ieee118", list(rows14) + list(rows118), []))
    for name, train_rows, extra in variants:
        started = time.perf_counter()
        result = train_gnn.train_and_score(train_rows, extra=extra, view=view, seeds=seeds, device=device, log=log)
        by_id = {rid: k for k, rid in enumerate(result["ids"])}
        entry: dict[str, Any] = {"train_rows": result["train_rows"], "seconds": time.perf_counter() - started, "view": view,
                                 "scores": {}}
        thresholds: dict[str, float] = {}
        tests: dict[str, tuple[list[dict[str, Any]], np.ndarray]] = {}
        for label, rows in (("ieee118", rows118), ("ieee14", rows14)):
            calibration = [r for r in rows if r["split"] == "calibration" and r["kind"] != "child" and r["id"] in by_id]
            test = [r for r in rows if r["split"] == "test" and r["kind"] != "child" and r["id"] in by_id]
            if not calibration or not test:
                continue
            cal_score = np.asarray([result["needs_aux"][by_id[r["id"]]] for r in calibration])
            cal_y = np.asarray([r["triage"]["needs_aux"] for r in calibration])
            thresholds[label] = recall_threshold(cal_score, cal_y, target)
            test_score = np.asarray([result["needs_aux"][by_id[r["id"]]] for r in test])
            tests[label] = (test, test_score)
            entry[label] = summarize(test, test_score, thresholds[label])
            entry["scores"][label] = {"ids": [str(r["id"]) for r in test], "needs_aux": test_score.round(6).tolist()}
        # Zero-shot reading of a single-network model: the other network's test split at the training network's threshold.
        source = {"ieee14_only": "ieee14", "ieee118_only": "ieee118"}.get(name)
        if source in thresholds:
            for label, (test, test_score) in tests.items():
                if label != source:
                    entry[f"{label}_at_{source}_threshold"] = summarize(test, test_score, thresholds[source])
        out[name] = entry
    return out


# ------------------------------------------------------------------------ main

def _cell(value: Any) -> str:
    if not isinstance(value, Mapping) or value.get("point") is None:
        return "n/a"
    if value.get("low") is None:
        return f"{100 * value['point']:.1f}%"
    return f"{100 * value['point']:.1f}% [{100 * value['low']:.1f}, {100 * value['high']:.1f}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, default=5, help="seeds of the trained models (0 skips them)")
    parser.add_argument("--view", choices=features.RESIDUAL_VIEWS, default="residual_v2",
                        help="feature view of the trained models (the export keeps its own)")
    parser.add_argument("--admission-rule", choices=("excess_energy", "chi_square_margin", "none"), default="excess_energy",
                        help="re-filter both networks' roots by this rule on their own solve ('none' keeps them as generated)")
    args = parser.parse_args(argv)
    rule = None if args.admission_rule == "none" else args.admission_rule
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    classifier, models = export_models(device)
    threshold = float(classifier.threshold)
    target = 0.9927536231884058  # the IEEE 14 calibration recall target behind the export's threshold

    rows = admitted_rows(load_ieee118_rows(Path(args.dataset_dir)), rule)
    alarm_only = [r for r in load_ieee118_rows(Path(args.dataset_dir), include_alarm_only=True) if r["kind"] == "alarm_only"]
    rows14_all = admitted_rows(data.load_rows(workers=args.workers), rule)
    rows14 = [r for r in rows14_all if r["split"] == "test" and r["kind"] != "child"]
    train14 = [r for r in rows14_all if r["split"] == "train" and r["kind"] != "child"]
    print(f"[transfer] IEEE 118 rows {len(rows)} ({dict(Counter(r['family'] for r in rows))}), alarm-only {len(alarm_only)}, "
          f"IEEE 14 test rows {len(rows14)}; admission rule {args.admission_rule}, trained view {args.view}", flush=True)

    export_view = classifier.view
    graphs = [graph_of(r, export_view) for r in rows]
    scored = score_graphs(models, graphs, device)
    graphs14 = [graph_of(r, export_view) for r in rows14]
    scored14 = score_graphs(models, graphs14, device)
    summary: dict[str, Any] = {
        "export": {"model_id": classifier.model_id, "threshold": threshold, "calibration_recall_target": target, "view": export_view},
        "recipe": {"admission_rule": args.admission_rule, "trained_view": args.view, "dataset_dir": str(args.dataset_dir)},
        "ieee14_test": summarize(rows14, scored14["needs_aux"], threshold),
        "ieee118_zero_shot": summarize(rows, scored["needs_aux"], threshold),
    }
    flags = scored["needs_aux"] >= threshold
    summary["ieee118_strata"] = {
        "hif_by_kv": stratified(rows, flags, ("hif",), "kv"),
        "hif_by_r_band_ohm": stratified(rows, flags, ("hif",), "r_band_ohm"),
        "hif_by_r_hif_pu": stratified(rows, flags, ("hif",), "r_hif_pu"),
        "waveform_by_max_residual": {f: stratified(rows, flags, (f,), "max_residual") for f in WAVEFORM},
        "waveform_by_log_chi_square_ratio": {f: stratified(rows, flags, (f,), "log_chi_square_ratio") for f in WAVEFORM},
        "unbalance_by_vuf": stratified(rows, flags, ("three_phase_unbalance",), "vuf_at_bus"),
        "harmonic_by_max_thd": stratified(rows, flags, ("harmonic",), "max_thd"),
    }
    summary["family_heads_on_misses"] = family_heads_on_misses(rows, scored, threshold)
    summary["input_ranges"] = {"ieee118": input_ranges(rows, graphs, export_view),
                               "ieee14_train": input_ranges(train14, [graph_of(r, export_view) for r in train14], export_view)}
    requested = {row["id"]: bool(flag) for row, flag in zip(rows, flags)}
    summary["signatures"] = {
        "ieee14_train": signature_table(train14, {family: (lambda r, f=family: r["family"] == f) for family in
                                                  ("hif", "measurement+hif", "three_phase_unbalance", "harmonic", "measurement",
                                                   "multi_measurement", "parameter", "mimic_flow_pair_same_sign")}),
        "ieee118": signature_table(rows, {
            **{f"{family} requested": (lambda r, f=family: r["family"] == f and requested[r["id"]]) for family in WAVEFORM},
            **{f"{family} missed": (lambda r, f=family: r["family"] == f and not requested[r["id"]]) for family in WAVEFORM},
            **{family: (lambda r, f=family: r["family"] == f) for family in
               ("measurement", "multi_measurement", "parameter", "mimic_flow_pair_same_sign", "healthy_window")}}),
    }
    if alarm_only:
        alarm_scores = score_graphs(models, [graph_of(r, export_view) for r in alarm_only], device)["needs_aux"]
        summary["alarm_only_secondary"] = {
            family: f"{int(np.sum(alarm_scores[[i for i, r in enumerate(alarm_only) if r['family'] == family]] >= threshold))}"
                    f"/{sum(r['family'] == family for r in alarm_only)}" for family in sorted({r["family"] for r in alarm_only})}

    # Calibration versus ranking: a threshold re-chosen on the IEEE 118 calibration split.
    calibration = np.asarray([r["split"] == "calibration" for r in rows])
    test = np.asarray([r["split"] == "test" for r in rows])
    y = np.asarray([r["triage"]["needs_aux"] for r in rows])
    test_rows = [r for r, keep in zip(rows, test) if keep]
    interventions: dict[str, Any] = {}
    for name in ("none", "globals14", "subgraph14", "magnitude10", "static14", "all"):
        if name == "none":
            score = scored["needs_aux"]
        else:
            score = score_graphs(models, intervention_graphs(rows, name, export_view), device)["needs_aux"]
        recalibrated = recall_threshold(score[calibration], y[calibration], target)
        entry = {"ieee14_threshold": summarize(rows, score, threshold),
                 "recalibrated_on_ieee118": {**summarize(test_rows, score[test], recalibrated),
                                             "rule": "largest threshold reaching the IEEE 14 recall target on the IEEE 118 calibration split"},
                 "hif_by_r_band_ohm": stratified(rows, score >= threshold, ("hif",), "r_band_ohm"),
                 "waveform_by_max_residual": {f: stratified(rows, score >= threshold, (f,), "max_residual") for f in WAVEFORM}}
        interventions[name] = entry
        print(f"[transfer] {name}: recall {_cell(entry['ieee14_threshold']['recall'])}, unneeded "
              f"{_cell(entry['ieee14_threshold']['unneeded_request_rate'])}, AUC {_cell(entry['ieee14_threshold']['auc'])}", flush=True)
    summary["interventions"] = interventions
    if args.seeds > 0:
        summary["ceilings"] = in_domain(rows, rows14_all, device=device, seeds=range(args.seeds), target=target, view=args.view)
    summary["seconds"] = time.perf_counter() - started
    (out / "transfer_ieee118.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    write_report(summary, out / "transfer_ieee118.md")
    print((out / "transfer_ieee118.md").read_text(encoding="utf-8"))
    return 0


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    zs, ref = summary["ieee118_zero_shot"], summary["ieee14_test"]
    lines = ["# IEEE 14 triage GNN on IEEE 118 (zero-shot)\n",
             f"Export {summary['export']['model_id']}, request threshold {summary['export']['threshold']:.4f} (IEEE 14).\n",
             "| set | rows | recall (needs phasors) | unneeded requests | AUC |", "| --- | --- | --- | --- | --- |",
             f"| IEEE 14 test | {ref['rows']} | {_cell(ref['recall'])} | {_cell(ref['unneeded_request_rate'])} | {_cell(ref['auc'])} |",
             f"| IEEE 118, all alarmed rows | {zs['rows']} | {_cell(zs['recall'])} | {_cell(zs['unneeded_request_rate'])} | {_cell(zs['auc'])} |",
             "", "## Requested by family (IEEE 118)\n", "| family | requested |", "| --- | --- |"]
    lines += [f"| {family} | {cell['requested']}/{cell['n']} |" for family, cell in zs["by_family"].items()]
    lines += ["", "## Interventions (one change at inference)\n",
              "| change | recall at IEEE 14 threshold | unneeded at IEEE 14 threshold | AUC | test recall, recalibrated | test unneeded, recalibrated |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name, entry in summary["interventions"].items():
        a, b = entry["ieee14_threshold"], entry["recalibrated_on_ieee118"]
        lines.append(f"| {name} | {_cell(a['recall'])} | {_cell(a['unneeded_request_rate'])} | {_cell(a['auc'])} | "
                     f"{_cell(b['recall'])} | {_cell(b['unneeded_request_rate'])} |")
    if summary.get("ceilings"):
        recipe = summary.get("recipe") or {}
        lines += ["", f"## Trained models (test splits, thresholds from each network's calibration split; view {recipe.get('trained_view')}, "
                      f"admission rule {recipe.get('admission_rule')})\n",
                  "| training set | scored on | recall | unneeded requests | AUC |", "| --- | --- | --- | --- | --- |"]
        for name, entry in summary["ceilings"].items():
            for label in ("ieee118", "ieee14", "ieee118_at_ieee14_threshold", "ieee14_at_ieee118_threshold"):
                if label in entry:
                    s = entry[label]
                    scored_on = f"{label} test" if "_at_" not in label else f"{label.split('_at_')[0]} test, zero-shot ({label.split('_at_')[1].replace('_threshold', '')} threshold)"
                    lines.append(f"| {name} | {scored_on} | {_cell(s['recall'])} | {_cell(s['unneeded_request_rate'])} | {_cell(s['auc'])} |")
    lines += ["", "## Strata (requested / rows, IEEE 14 threshold)\n", "```", json.dumps(summary["ieee118_strata"], indent=1), "```",
              "", "## Family heads on missed waveform roots\n", "```", json.dumps(summary["family_heads_on_misses"], indent=1), "```",
              "", "## Alarm signatures (quartiles: channels > 3 sigma, top-2 share of their energy, assets touched, max |r|)\n",
              "```", json.dumps(summary.get("signatures"), indent=1), "```",
              "", "## Graph-level inputs (5/50/95 %)\n", "```", json.dumps(summary.get("input_ranges"), indent=1), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
