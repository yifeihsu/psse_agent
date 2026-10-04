"""Does the triage GNN recognize a waveform event at a location it never trained on?

    python -m research.classifier_triage.location_study --output-dir output/classifier_triage_20261004/ieee14_location

A model that moves between networks without retraining has to read the
event, not the place.  Within IEEE 14 this is tested by holding locations
out: the waveform roots (HIF by faulted line, unbalance and harmonic by
source bus) are split into three folds by location, each fold's model trains
without the roots at its held-out locations, and is then scored on them.
The request threshold is set on the calibration rows at seen locations only
(at the physics rule's calibration recall), so nothing about the held-out
locations reaches the decision.  Reported per family: recall at unseen
locations against recall at seen ones, and the unneeded-request rate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import data, train_gnn  # noqa: E402
from research.hypothesis_ranking import ranker  # noqa: E402

FOLDS = 3
SEED = 20261004


def location(row: Mapping[str, Any]) -> tuple[str, int] | None:
    """The waveform event's family and place (faulted branch row or source bus), else ``None``."""
    truth = row["record"].get("truth") or {}
    if truth.get("hif"):
        return "hif", int(truth["hif"][0]["branch_row0"])
    if truth.get("unbalance"):
        return "unbalance", int(truth["unbalance"][0]["unbalance_bus"])
    if truth.get("harmonic"):
        return "harmonic", int(truth["harmonic"][0]["bus_1based"])
    return None


def fold_of(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], int]:
    """Round-robin folds over each family's sorted locations, so every fold holds out some of each family."""
    places: dict[str, set[int]] = {}
    for row in rows:
        place = location(row)
        if place is not None:
            places.setdefault(place[0], set()).add(place[1])
    return {(family, value): index % FOLDS for family, values in places.items() for index, value in enumerate(sorted(values))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(data.DEFAULT_DATASET))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", default="residual")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = data.load_rows(Path(args.dataset_dir), workers=args.workers)
    folds = fold_of(rows)
    split = data.split_rows(rows)
    rule_recall = float(np.mean([r["rule_v3"] for r in split["calibration"] if r["triage"]["needs_aux"] == 1]))
    print(f"[location] {len(folds)} waveform locations in {FOLDS} folds; rule calibration recall {rule_recall:.4f}", flush=True)

    records: list[dict[str, Any]] = []  # one per scored positive, plus the test negatives of each fold
    fold_summaries = []
    for fold in range(FOLDS):
        held = {place for place, index in folds.items() if index == fold}
        kept = [r for r in rows if not (r["split"] == "train" and location(r) in held)]
        dropped = len(rows) - len(kept)
        scored = train_gnn.train_and_score(kept, view=args.view, seeds=range(args.seeds), device=args.device,
                                           log=lambda message: print(message, flush=True))
        score = dict(zip(scored["ids"], scored["needs_aux"]))
        seen_cal = [float(score[str(r["id"])]) for r in split["calibration"]
                    if r["triage"]["needs_aux"] == 1 and location(r) not in held]
        threshold = ranker.threshold_at_recall(np.asarray(seen_cal), rule_recall)
        negatives = [r for r in split["test"] if r["needs_no_phasors_family"]]
        unneeded = float(np.mean([score[str(r["id"])] >= threshold for r in negatives]))
        for r in split["calibration"] + split["test"]:
            place = location(r)
            if place is None:
                continue
            unseen = place in held
            if not unseen and r["split"] != "test":
                continue  # seen-location calibration positives set the threshold
            records.append({"fold": fold, "id": str(r["id"]), "parent": str(r["parent"]), "family": place[0],
                            "place": place[1], "unseen": unseen, "requested": bool(score[str(r["id"])] >= threshold),
                            "score": float(score[str(r["id"])])})
        fold_summaries.append({"fold": fold, "held_out": sorted(f"{family}:{value}" for family, value in held),
                               "train_rows_dropped": dropped, "threshold": threshold, "unneeded_request_rate": unneeded,
                               "test_negatives": len(negatives)})
        print(f"[location] fold {fold}: held out {len(held)} locations ({dropped} train rows dropped), "
              f"threshold {threshold:.4f}, unneeded {100 * unneeded:.1f}%", flush=True)

    def recall(family: str | None, unseen: bool) -> dict[str, Any]:
        chosen = [x for x in records if x["unseen"] == unseen and (family is None or x["family"] == family)]
        flags = np.asarray([x["requested"] for x in chosen], dtype=float)
        parents = [x["parent"] for x in chosen]
        interval = ranker.parent_bootstrap(parents, lambda i: float(flags[i].mean()) if i.size else None, seed=SEED) \
            if chosen else {"point": None, "low": None, "high": None}
        return {**interval, "n": len(chosen), "missed": int(len(chosen) - flags.sum())}

    summary = {
        "view": args.view, "seeds": args.seeds, "folds": fold_summaries,
        "recall": {name: {"unseen_location": recall(family, True), "seen_location": recall(family, False)}
                   for name, family in (("all", None), ("hif", "hif"), ("unbalance", "unbalance"), ("harmonic", "harmonic"))},
        "unneeded_request_rate_mean": float(np.mean([f["unneeded_request_rate"] for f in fold_summaries])),
        "missed_unseen": [x for x in records if x["unseen"] and not x["requested"]],
        "seconds": time.perf_counter() - started,
    }
    (out / "location_study.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = ["# Triage GNN at locations held out of training (IEEE 14)\n",
             f"Three folds over waveform locations, {args.seeds} seeds per fold, view `{args.view}`. The threshold is set on "
             "calibration rows at seen locations, at the physics rule's calibration recall. Brackets are 95% "
             "parent-bootstrap intervals.\n",
             "| family | recall, unseen locations | recall, seen locations |", "| --- | --- | --- |"]
    for name, item in summary["recall"].items():
        cells = []
        for key in ("unseen_location", "seen_location"):
            value = item[key]
            cells.append("n/a" if value["point"] is None else
                         f"{100 * value['point']:.1f}% [{100 * value['low']:.1f}, {100 * value['high']:.1f}] ({value['n'] - value['missed']}/{value['n']})")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines += ["", f"Unneeded requests on the test negatives, mean over folds: {100 * summary['unneeded_request_rate_mean']:.1f}%.", ""]
    lines += ["| fold | held-out locations | train rows dropped | unneeded requests |", "| --- | --- | --- | --- |"]
    for item in fold_summaries:
        lines.append(f"| {item['fold']} | {', '.join(item['held_out'])} | {item['train_rows_dropped']} | "
                     f"{100 * item['unneeded_request_rate']:.1f}% |")
    (out / "location_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[location] wrote {out / 'location_study.md'} in {summary['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
