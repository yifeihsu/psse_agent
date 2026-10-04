"""A first zero-shot look: the IEEE 14 triage GNN on the existing IEEE 57 study rows, no retraining.

    python -m research.classifier_triage.transfer_look --benchmark-dir output/classifier_triage_20261004/ieee14

The weights and the request threshold are the IEEE 14 ones.  The IEEE 57
study set (``output/hypothesis_ranking_20260930/ieee57_v3``) holds only HIF
and unbalance roots and healthy windows, and its HIFs are the normalized
per-unit sweep rather than the physical-ohm corpora, so this is a look at
whether the request score moves to another network at all, not the transfer
benchmark: there are no meter, parameter, topology or harmonic roots to
measure unneeded requests on.  The physics screen's rule on the same rows is
the reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import data, train_gnn  # noqa: E402
from research.hypothesis_ranking import ranker  # noqa: E402

DEFAULT_TRANSFER = REPO_ROOT / "output" / "hypothesis_ranking_20260930" / "ieee57_v3"
SEED = 20261004


def load_transfer_rows(dataset_dir: Path, workers: int) -> list[dict[str, Any]]:
    rows = []
    for record in ranker._read(Path(dataset_dir) / "roots.jsonl"):
        screen = (record.get("analysis") or {}).get("screen") or {}
        if not ranker.alarmed(record) or screen.get("status") != "valid":
            continue
        rows.append({"id": record["root_id"], "kind": "root", "family": record["family"], "parent": record["parent_id"],
                     "split": "test", "labels": ranker.labels(record), "rule_v3": ranker.rule_v3(record),
                     "record": record, "triage": data.triage_labels(record)})
    return data.attach_payloads(rows, Path(dataset_dir).parent / "classifier_triage" / "payloads_ieee57_v3.pkl", workers=workers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--transfer-dir", default=str(DEFAULT_TRANSFER))
    parser.add_argument("--models", nargs="*", default=["gnn_residual", "gnn_values", "gnn_residual_decorrelated"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    bench = Path(args.benchmark_dir)
    benchmark = json.loads((bench / "benchmark.json").read_text(encoding="utf-8"))
    rows = load_transfer_rows(Path(args.transfer_dir), args.workers)
    y = np.asarray([row["triage"]["needs_aux"] for row in rows])
    rule = np.asarray([row["rule_v3"] for row in rows], dtype=float)
    parents = [str(row["parent"]) for row in rows]
    families = Counter(row["family"] for row in rows)
    print(f"[transfer] {len(rows)} alarmed IEEE 57 rows: {dict(families)}", flush=True)

    def rates(flags: np.ndarray) -> dict[str, Any]:
        flags = np.asarray(flags, dtype=float)
        result = {
            "recall": ranker.parent_bootstrap(parents, lambda i: float(flags[i[y[i] == 1]].mean()) if (y[i] == 1).any() else None, seed=SEED),
            "healthy_request_rate": ranker.parent_bootstrap(parents, lambda i: float(flags[i[y[i] == 0]].mean()) if (y[i] == 0).any() else None, seed=SEED),
        }
        for family in sorted(families):
            mask = np.asarray([row["family"] == family for row in rows])
            result[f"requested_{family}"] = f"{int(flags[mask].sum())}/{int(mask.sum())}"
        return result

    summary: dict[str, Any] = {"rows": len(rows), "families": dict(families), "screen_rule": rates(rule), "models": {}}
    for name in args.models:
        entry = next((r for r in benchmark["results"] if r["name"] == name), None)
        if entry is None or not (bench / name / "checkpoints").is_dir():
            continue
        view = "values" if name == "gnn_values" else "residual"
        threshold = float(entry["request"]["thresholds"]["at_rule_recall"])
        score = train_gnn.score_with_checkpoints(bench / name / "checkpoints", rows, view)["needs_aux"]
        summary["models"][name] = {
            "threshold_from_ieee14": threshold,
            "auc": ranker.parent_bootstrap(parents, lambda i, s=score: ranker.auc(s[i], y[i]), seed=SEED),
            **rates(score >= threshold),
        }
    out = bench / "transfer_look_ieee57.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    def cell(value):
        return "n/a" if not isinstance(value, dict) or value.get("point") is None else \
            f"{100 * value['point']:.1f}% [{100 * value['low']:.1f}, {100 * value['high']:.1f}]"

    names = sorted(families)
    lines = ["# Zero-shot look: IEEE 14 triage GNN on the IEEE 57 study rows\n",
             f"{len(rows)} alarmed rows ({', '.join(f'{families[n]} {n}' for n in names)}); weights and threshold from IEEE 14. "
             "No balanced-fault roots exist on this network yet, so unneeded requests are measured on healthy windows only.\n",
             "| classifier | AUC | recall | requests on healthy windows | " + " | ".join(f"requested, {n}" for n in names) + " |",
             "| --- | --- | --- | --- |" + " --- |" * len(names)]
    rule_rates = summary["screen_rule"]
    lines.append(f"| screen_rule | n/a | {cell(rule_rates['recall'])} | {cell(rule_rates['healthy_request_rate'])} | "
                 + " | ".join(rule_rates[f"requested_{n}"] for n in names) + " |")
    for name, item in summary["models"].items():
        lines.append(f"| {name} | {cell(item['auc'])} | {cell(item['recall'])} | {cell(item['healthy_request_rate'])} | "
                     + " | ".join(item[f"requested_{n}"] for n in names) + " |")
    (bench / "transfer_look_ieee57.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
