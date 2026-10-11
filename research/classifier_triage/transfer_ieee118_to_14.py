"""The reverse transfer: a triage GNN trained on IEEE 118 only, scored on IEEE 14.

    python -m research.classifier_triage.transfer_ieee118_to_14 --dataset-dir output/classifier_triage_20261007/ieee118 \\
        --output-dir output/classifier_triage_20261008/ieee118_to_14

The GNN of the IEEE 14 export (same features, architecture and training
recipe, five seeds) is trained on the IEEE 118 train split of
``ieee118_dataset``, early-stopped on IEEE 118 train parents, and given its
request threshold on the IEEE 118 calibration split (the IEEE 14 export's
recall target, 99.3%).  Nothing of IEEE 14 enters training, model selection
or the threshold.  It is then scored on the IEEE 14 study's test split, the
same 463 rows the IEEE 14 benchmark reports.

References on the same rows: the IEEE 14 export (trained on IEEE 14) and the
model trained on IEEE 14 and IEEE 118 together (``transfer_ieee118`` ceilings).
The IEEE 118 set has no topology roots (the breaker-level model exists for
IEEE 14 only), so the IEEE 14 topology and meter-plus-topology rows are
outside the training families; their requests are reported apart.

Each candidate cause is probed by changing one thing at inference, the
inverse of ``transfer_ieee118``:

* ``recalibrated``: the threshold re-chosen on the IEEE 14 calibration split
  (ranking versus calibration);
* ``globals118``: the five graph-level inputs re-expressed on IEEE 118's
  degrees of freedom and channel count (the same excess objective, the >3
  and >4 sigma counts over 1,098 channels);
* ``pool118``: the IEEE 14 graph placed beside a quiet IEEE 118 window
  (disjoint union, globals kept), so mean pooling dilutes the anomaly as on
  IEEE 118;
* ``embed118``: the same union with globals recomputed over both, i.e. the
  IEEE 14 state seen inside a larger healthy system;
* ``static118``: branch r, x and tap clamped to the IEEE 118 range.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import data, features, train_gnn  # noqa: E402
from research.classifier_triage.model import FAMILY_HEADS, FIRST_CLASSES  # noqa: E402
from research.classifier_triage.transfer_ieee118 import (  # noqa: E402
    _cell, admitted_rows, graph_of, load_ieee118_rows, recall_threshold, score_graphs, summarize,
)

SEED = 20261008
RECALL_TARGET = 0.9927536231884058  # the IEEE 14 export's calibration recall target
IEEE118_NZ, IEEE118_DOF = 1098, 863
TOPOLOGY_FAMILIES = ("topology", "measurement+topology")
WAVEFORM_FAMILIES = ("hif", "measurement+hif", "three_phase_unbalance", "harmonic")
QUIET_WINDOWS = 24
DEFAULT_JOINT_REPORT = REPO_ROOT / "output" / "classifier_triage_20261007" / "transfer" / "transfer_ieee118.json"


# ----------------------------------------------------------------- models

def load_checkpoints(directory: Path, device: str) -> list[Any]:
    import torch

    from research.classifier_triage.model import TriageGNN

    models = []
    for path in sorted(Path(directory).glob("seed_*.pt")):
        saved = torch.load(path, map_location=device, weights_only=False)
        config = saved["config"]
        model = TriageGNN(config["node_dim"], config["edge_dim"], config["global_dim"], hidden_dim=config["hidden_dim"],
                          layers=config["layers"], dropout=config["dropout"]).to(device)
        model.load_state_dict(saved["state_dict"])
        models.append(model.eval())
    if not models:
        raise FileNotFoundError(f"no seed checkpoints under {directory}")
    return models


# ----------------------------------------------------------- interventions

def quiet_ieee118_payloads(dataset_dir: Path, count: int = QUIET_WINDOWS) -> list[dict[str, Any]]:
    """WLS payloads of clean IEEE 118 corpus windows that do not alarm (the healthy system the IEEE 14 state is placed in)."""
    from research.classifier_triage.ieee118_dataset import wls_check

    rng = np.random.default_rng(SEED)
    rows = []
    for path in sorted((Path(dataset_dir) / "balanced").glob("shard_*/corpus/samples.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows += [json.loads(line) for line in stream if '"no_error"' in line]
    rows = [row for row in rows if row.get("scenario") == "no_error"]
    out = []
    for position in rng.permutation(len(rows)):
        row = rows[int(position)]
        payload, stats = wls_check(row["z_obs"], row["sigma_z"])
        if payload is not None and not stats["alarm"]:
            out.append(payload)
        if len(out) >= count:
            break
    return out


def globals118(graph: Mapping[str, np.ndarray], payload: Mapping[str, Any], nz: int) -> dict[str, np.ndarray]:
    """The five graph-level inputs re-expressed on IEEE 118's dof and channel count (same excess objective)."""
    out = dict(graph)
    u = np.array(graph["u"], dtype=float, copy=True)
    excess = float(payload["objective"]) - float(payload["dof"])
    threshold = float(chi2.ppf(1.0 - features.CHI2_ALPHA, IEEE118_DOF))
    u[0] = np.log(max(IEEE118_DOF + excess, 1e-9) / threshold)
    u[2] = u[2] * nz / IEEE118_NZ
    u[3] = u[3] * nz / IEEE118_NZ
    out["u"] = u
    return out


def union(graph: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray], u: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Disjoint union of two graphs (no edge between them); ``u`` replaces the first graph's globals when given."""
    nodes = np.asarray(graph["x"]).shape[0]
    branches = int(np.asarray(graph["edge_pair"]).max()) + 1 if np.asarray(graph["edge_pair"]).size else 0
    return {
        "x": np.vstack((graph["x"], other["x"])),
        "edge_index": np.hstack((graph["edge_index"], np.asarray(other["edge_index"]) + nodes)),
        "edge_attr": np.vstack((graph["edge_attr"], other["edge_attr"])),
        "edge_pair": np.concatenate((graph["edge_pair"], np.asarray(other["edge_pair"]) + branches)),
        "u": np.asarray(graph["u"] if u is None else u, dtype=float),
    }


def combined_globals(payload: Mapping[str, Any], case: str, quiet: Mapping[str, Any], quiet_case: str = "case118") -> np.ndarray:
    """The five globals of the two solves read as one system: objectives and dof add, fractions over both channel sets."""
    residual = np.abs(np.concatenate((features.neutral_residuals(payload, features.network(case)),
                                      features.neutral_residuals(quiet, features.network(quiet_case)))))
    lambdas = np.concatenate((np.asarray(payload["lambda_normalized"], dtype=float), np.asarray(quiet["lambda_normalized"], dtype=float)))
    objective = float(payload["objective"]) + float(quiet["objective"])
    dof = int(payload["dof"]) + int(quiet["dof"])
    threshold = float(chi2.ppf(1.0 - features.CHI2_ALPHA, dof))
    return np.asarray([np.log(max(objective, 1e-9) / threshold), np.log1p(residual.max()), float(np.mean(residual > 3.0)),
                       float(np.mean(residual > 4.0)), np.log1p(float(np.max(np.abs(lambdas))) if lambdas.size else 0.0)])


def static118(graph: Mapping[str, np.ndarray], ranges: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Branch r, x and tap clamped to the IEEE 118 range (IEEE 14's other static inputs already lie inside it)."""
    names = features.EDGE_FEATURES["residual"]
    start = names.index("r_pu")
    attr = np.array(graph["edge_attr"], dtype=float, copy=True)
    for name in ("r_pu", "x_pu", "tap"):
        column = names.index(name)
        attr[:, column] = np.clip(attr[:, column], ranges["edge"][0][column - start], ranges["edge"][1][column - start])
    return {**graph, "edge_attr": attr}


def intervention_graphs(rows: Sequence[Mapping[str, Any]], name: str, quiet: Sequence[Mapping[str, Any]],
                        view: str = "residual") -> list[dict[str, np.ndarray]]:
    from research.classifier_triage.transfer_ieee118 import static_ranges

    ranges = static_ranges("case118")
    quiet_graphs = [features.build_graph("case118", payload, view) for payload in quiet]
    rng = np.random.default_rng(SEED + 1)
    picks = rng.integers(len(quiet), size=len(rows)) if quiet else np.zeros(len(rows), dtype=int)
    graphs = []
    for row, pick in zip(rows, picks):
        case = str(row["record"]["case"])
        net = features.network(case)
        nz = 3 * int(net["nb"]) + 4 * int(net["nl"])
        graph = features.build_graph(case, row["payload"], view)
        if name == "globals118" and view == "residual":
            graph = globals118(graph, row["payload"], nz)
        elif name == "pool118":
            graph = union(graph, quiet_graphs[int(pick)])
        elif name == "embed118":
            graph = union(graph, quiet_graphs[int(pick)], combined_globals(row["payload"], case, quiet[int(pick)]))
        elif name == "static118":
            graph = static118(graph, ranges)
        elif name != "none":
            raise ValueError(f"unknown intervention {name!r}")
        graphs.append(graph)
    return graphs


# ------------------------------------------------------------------ reports

def family_table(rows: Sequence[Mapping[str, Any]], score: np.ndarray, threshold: float) -> dict[str, str]:
    counts: dict[str, list[int]] = {}
    for row, value in zip(rows, score):
        cell = counts.setdefault(row["family"], [0, 0])
        cell[0] += int(value >= threshold)
        cell[1] += 1
    return {family: f"{hit}/{n}" for family, (hit, n) in sorted(counts.items())}


def heads(rows: Sequence[Mapping[str, Any]], scored: Mapping[str, np.ndarray], threshold: float, *, positives: bool) -> dict[str, Any]:
    """Top family head and first head on the misses (positives) or on the false requests (negatives), by family."""
    out: dict[str, Any] = {}
    for i, row in enumerate(rows):
        need = bool(row["triage"]["needs_aux"])
        flagged = scored["needs_aux"][i] >= threshold
        if need != positives or flagged == positives:
            continue
        entry = out.setdefault(row["family"], {"count": 0, "top_family_head": Counter(), "first_head": Counter()})
        entry["count"] += 1
        entry["top_family_head"][FAMILY_HEADS[int(np.argmax(scored["family"][i]))]] += 1
        entry["first_head"][FIRST_CLASSES[int(np.argmax(scored["first"][i]))]] += 1
    return {family: {"count": e["count"], "top_family_head": dict(e["top_family_head"]), "first_head": dict(e["first_head"])}
            for family, e in sorted(out.items())}


def without_topology(rows: Sequence[Mapping[str, Any]], score: np.ndarray, threshold: float) -> dict[str, Any]:
    keep = [i for i, row in enumerate(rows) if row["family"] not in TOPOLOGY_FAMILIES]
    return summarize([rows[i] for i in keep], np.asarray(score)[keep], threshold)


def hif_strata(rows: Sequence[Mapping[str, Any]], score: np.ndarray, threshold: float) -> dict[str, str]:
    counts: dict[str, list[int]] = {}
    for row, value in zip(rows, score):
        if row["family"] not in ("hif", "measurement+hif"):
            continue
        hif = row["record"]["truth"]["hif"][0]
        r_pu = float(hif.get("r_hif_pu") or np.nan)
        label = f"{row['family']}, R {'<3' if r_pu < 3 else '3-4' if r_pu < 4 else '>=4'} pu"
        cell = counts.setdefault(label, [0, 0])
        cell[0] += int(value >= threshold)
        cell[1] += 1
    return {label: f"{hit}/{n}" for label, (hit, n) in sorted(counts.items())}


DEFAULT_PROBE_DIR = REPO_ROOT / "output" / "classifier_triage_20261004" / "ieee14"
PROBE_FILES = {"meter": "probe.jsonl", "vm_large": "probe_vm_large.jsonl", "vm_small": "probe_vm_small.jsonl"}


def probe_rates(models: Sequence[Any], threshold: float, device: str, probe_dir: Path = DEFAULT_PROBE_DIR,
                workers: int = 8, view: str = "residual") -> dict[str, Any]:
    """Request rates on the IEEE 14 background probes (a quiet window plus one bad meter; none needs phasors).

    No IEEE 14 parent is in this model's training, so every probe row counts.
    """
    out: dict[str, Any] = {}
    for kind, name in PROBE_FILES.items():
        path = Path(probe_dir) / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            rows = data.probe_rows(json.loads(line) for line in stream if line.strip())
        rows = data.attach_payloads(rows, Path(probe_dir) / "probe_payloads.pkl", workers=workers)
        scored = score_graphs(models, [graph_of(row, view) for row in rows], device)["needs_aux"]
        for background in ("opendss", "opf"):
            mask = np.asarray([row["background"] == background for row in rows])
            if mask.any():
                out[f"{kind}_{background}"] = {"requested": int(np.sum(scored[mask] >= threshold)), "rows": int(mask.sum()),
                                               "rate": float(np.mean(scored[mask] >= threshold))}
    return out


# --------------------------------------------------------------------- main

def probes_only(output_dir: Path, device: str, workers: int) -> int:
    """Score the IEEE 14 probes with the checkpoints and threshold of a finished run."""
    summary_path = Path(output_dir) / "transfer_ieee118_to_14.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    threshold = float(summary["training"]["threshold_from_ieee118_calibration"])
    view = str(summary["training"].get("view") or "residual")
    rates = probe_rates(load_checkpoints(Path(output_dir) / "checkpoints", device), threshold, device, workers=workers, view=view)
    summary["ieee14_probes"] = rates
    summary_path.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rates, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--joint-report", default=str(DEFAULT_JOINT_REPORT))
    parser.add_argument("--probes-only", action="store_true", help="score the IEEE 14 probes with a finished run's checkpoints")
    parser.add_argument("--view", choices=features.RESIDUAL_VIEWS, default="residual")
    parser.add_argument("--admission-rule", choices=("excess_energy", "chi_square_margin", "none"), default="none",
                        help="re-filter both networks' roots by this rule on their own solve")
    args = parser.parse_args(argv)
    rule = None if args.admission_rule == "none" else args.admission_rule
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    if args.probes_only:
        return probes_only(out, device, args.workers)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rows118 = admitted_rows(load_ieee118_rows(Path(args.dataset_dir)), rule)
    rows14 = admitted_rows(data.load_rows(workers=args.workers), rule)
    split118 = data.split_rows(rows118)
    cal14 = [r for r in rows14 if r["split"] == "calibration" and r["kind"] != "child"]
    test14 = [r for r in rows14 if r["split"] == "test" and r["kind"] != "child"]
    print(f"[reverse] IEEE 118 rows {len(rows118)} (train {len(split118['train'])}, calibration {len(split118['calibration'])}, "
          f"test {len(split118['test'])}); IEEE 14 calibration {len(cal14)}, test {len(test14)}", flush=True)

    checkpoints = out / "checkpoints"
    result = train_gnn.train_and_score(rows118, extra=cal14 + test14, view=args.view, seeds=range(args.seeds), device=device,
                                       checkpoint_dir=checkpoints, log=lambda text: print(text, flush=True))
    n_cal, n_test = len(split118["calibration"]), len(split118["test"])
    blocks = {"cal118": slice(0, n_cal), "test118": slice(n_cal, n_cal + n_test),
              "cal14": slice(n_cal + n_test, n_cal + n_test + len(cal14)), "test14": slice(n_cal + n_test + len(cal14), None)}
    expected = [str(r["id"]) for r in split118["calibration"] + split118["test"] + cal14 + test14]
    if result["ids"] != expected:
        raise RuntimeError("scored rows are not in the expected order")
    score = np.asarray(result["needs_aux"])
    scored14 = {"needs_aux": score[blocks["test14"]], "family": np.asarray(result["family"])[blocks["test14"]],
                "first": np.asarray(result["first"])[blocks["test14"]]}

    def labels(rows):
        return np.asarray([r["triage"]["needs_aux"] for r in rows])

    threshold = recall_threshold(score[blocks["cal118"]], labels(split118["calibration"]), RECALL_TARGET)
    threshold14 = recall_threshold(score[blocks["cal14"]], labels(cal14), RECALL_TARGET)
    summary: dict[str, Any] = {
        "training": {"rows": result["train_rows"], "validation_rows": result["validation_rows"], "history": result["history"],
                     "seeds": args.seeds, "threshold_from_ieee118_calibration": threshold, "view": args.view,
                     "admission_rule": args.admission_rule},
        "ieee118_test": summarize(split118["test"], score[blocks["test118"]], threshold),
        "ieee14_test_zero_shot": summarize(test14, scored14["needs_aux"], threshold),
        "ieee14_test_zero_shot_without_topology": without_topology(test14, scored14["needs_aux"], threshold),
        "ieee14_test_recalibrated_on_ieee14": {**summarize(test14, scored14["needs_aux"], threshold14), "threshold": threshold14},
        "ieee14_families": family_table(test14, scored14["needs_aux"], threshold),
        "ieee14_hif_strata": hif_strata(test14, scored14["needs_aux"], threshold),
        "heads_on_missed_positives": heads(test14, scored14, threshold, positives=True),
        "heads_on_false_requests": heads(test14, scored14, threshold, positives=False),
        "score_quantiles_ieee14_test": {
            family: np.quantile(scored14["needs_aux"][[i for i, r in enumerate(test14) if r["family"] == family]],
                                [0.1, 0.5, 0.9]).round(4).tolist()
            for family in sorted({r["family"] for r in test14})},
    }
    joint = Path(args.joint_report)
    if joint.is_file():
        report = json.loads(joint.read_text(encoding="utf-8"))
        summary["references"] = {"ieee14_export_on_ieee14_test": report.get("ieee14_test"),
                                 "joint_on_ieee14_test": ((report.get("ceilings") or {}).get("ieee14_plus_ieee118") or {}).get("ieee14")}

    models = load_checkpoints(checkpoints, device)
    quiet = quiet_ieee118_payloads(Path(args.dataset_dir))
    interventions: dict[str, Any] = {}
    for name in ("none", "globals118", "pool118", "embed118", "static118"):
        if name == "globals118" and args.view != "residual":
            continue  # the v2 globals are network-free by construction
        scored = score_graphs(models, intervention_graphs(test14, name, quiet, args.view), device)
        entry = summarize(test14, scored["needs_aux"], threshold)
        entry["families"] = family_table(test14, scored["needs_aux"], threshold)
        interventions[name] = entry
        print(f"[reverse] {name}: recall {_cell(entry['recall'])}, unneeded {_cell(entry['unneeded_request_rate'])}, "
              f"AUC {_cell(entry['auc'])}", flush=True)
    summary["interventions"] = interventions
    summary["quiet_ieee118_windows"] = len(quiet)
    summary["seconds"] = time.perf_counter() - started
    (out / "transfer_ieee118_to_14.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    write_report(summary, out / "transfer_ieee118_to_14.md")
    print((out / "transfer_ieee118_to_14.md").read_text(encoding="utf-8"))
    return 0


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    zs, own, recal = summary["ieee14_test_zero_shot"], summary["ieee118_test"], summary["ieee14_test_recalibrated_on_ieee14"]
    lines = ["# Triage GNN trained on IEEE 118, scored on IEEE 14\n",
             f"Threshold {summary['training']['threshold_from_ieee118_calibration']:.4f} from the IEEE 118 calibration split; "
             f"{summary['training']['rows']} IEEE 118 training rows, {summary['training']['seeds']} seeds.\n",
             "| model | scored on | recall | unneeded requests | AUC |", "| --- | --- | --- | --- | --- |",
             f"| trained on IEEE 118 | IEEE 118 test | {_cell(own['recall'])} | {_cell(own['unneeded_request_rate'])} | {_cell(own['auc'])} |",
             f"| trained on IEEE 118 | IEEE 14 test | {_cell(zs['recall'])} | {_cell(zs['unneeded_request_rate'])} | {_cell(zs['auc'])} |",
             f"| trained on IEEE 118, threshold from IEEE 14 calibration | IEEE 14 test | {_cell(recal['recall'])} | "
             f"{_cell(recal['unneeded_request_rate'])} | {_cell(recal['auc'])} |"]
    references = summary.get("references") or {}
    for label, key in (("IEEE 14 export", "ieee14_export_on_ieee14_test"), ("trained on IEEE 14 + 118", "joint_on_ieee14_test")):
        ref = references.get(key)
        if ref:
            lines.append(f"| {label} | IEEE 14 test | {_cell(ref['recall'])} | {_cell(ref['unneeded_request_rate'])} | {_cell(ref['auc'])} |")
    nt = summary["ieee14_test_zero_shot_without_topology"]
    lines += ["", f"Without the topology families (not in the IEEE 118 training set): unneeded requests {_cell(nt['unneeded_request_rate'])}.",
              "", "## IEEE 14 test by family (requested / rows)\n", "| family | requested |", "| --- | --- |"]
    lines += [f"| {family} | {cell} |" for family, cell in summary["ieee14_families"].items()]
    lines += ["", "## Interventions on the IEEE 14 test rows (threshold from IEEE 118)\n",
              "| change | recall | unneeded requests | AUC |", "| --- | --- | --- | --- |"]
    for name, entry in summary["interventions"].items():
        lines.append(f"| {name} | {_cell(entry['recall'])} | {_cell(entry['unneeded_request_rate'])} | {_cell(entry['auc'])} |")
    lines += ["", "```", json.dumps({name: entry["families"] for name, entry in summary["interventions"].items()}, indent=1), "```",
              "", "## Heads\n", "```", json.dumps({"missed_positives": summary["heads_on_missed_positives"],
                                                    "false_requests": summary["heads_on_false_requests"]}, indent=1), "```",
              "", "## HIF strata and score quantiles\n", "```",
              json.dumps({"hif": summary["ieee14_hif_strata"], "score_quantiles": summary["score_quantiles_ieee14_test"]}, indent=1), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
