"""Train the triage GNN on IEEE 14 and IEEE 118 together and export it as a runtime classifier.

    python -m research.classifier_triage.export_joint --dataset-dir output/classifier_triage_20261011/ieee118_v2 \\
        --output psse_env/oracle/models/triage_gnn_joint_20261011

The training rows are the IEEE 14 study rows (``data.load_rows``) and the
IEEE 118 set of ``ieee118_dataset``, both re-filtered by the excess-energy
admission rule, under the ``residual_v2`` view.  One request threshold per
network comes from that network's calibration split at the IEEE 14 export's
recall target (99.3%); the manifest carries them as ``request_thresholds``
and the IEEE 14 one as the fallback ``request_threshold``.  The test splits
of both networks are reported beside the export.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import data, runtime, train_gnn  # noqa: E402
from research.classifier_triage.transfer_ieee118 import (  # noqa: E402
    admitted_rows, load_ieee118_rows, recall_threshold, summarize,
)

RECALL_TARGET = 0.9927536231884058


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="the IEEE 118 set (ieee118_dataset)")
    parser.add_argument("--output", required=True, help="runtime directory to write")
    parser.add_argument("--train-dir", default=None,
                        help="where the training checkpoints and the report go (default: output/classifier_triage_20261011/"
                             "<output name>_train, outside the tracked model directory)")
    parser.add_argument("--view", default="residual_v2")
    parser.add_argument("--admission-rule", default="excess_energy")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    started = time.perf_counter()
    rule = None if args.admission_rule == "none" else args.admission_rule
    rows14 = admitted_rows(data.load_rows(workers=args.workers), rule)
    rows118 = admitted_rows(load_ieee118_rows(Path(args.dataset_dir)), rule)
    print(f"[joint] IEEE 14 rows {len(rows14)}, IEEE 118 rows {len(rows118)}; view {args.view}, rule {args.admission_rule}", flush=True)
    train_dir = Path(args.train_dir) if args.train_dir else Path("output/classifier_triage_20261011") / f"{output.name}_train"
    checkpoints = train_dir / "checkpoints"
    result = train_gnn.train_and_score(list(rows14) + list(rows118), view=args.view, seeds=range(args.seeds), device=device,
                                       checkpoint_dir=checkpoints, log=lambda text: print(text, flush=True))
    by_id = {rid: k for k, rid in enumerate(result["ids"])}
    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {}
    for key, rows in (("case14", rows14), ("case118", rows118)):
        calibration = [r for r in rows if r["split"] == "calibration" and r["kind"] != "child" and r["id"] in by_id]
        test = [r for r in rows if r["split"] == "test" and r["kind"] != "child" and r["id"] in by_id]
        score = np.asarray([result["needs_aux"][by_id[r["id"]]] for r in calibration])
        labels = np.asarray([r["triage"]["needs_aux"] for r in calibration])
        thresholds[key] = recall_threshold(score, labels, RECALL_TARGET)
        test_score = np.asarray([result["needs_aux"][by_id[r["id"]]] for r in test])
        report[key] = {"calibration_rows": len(calibration), "test": summarize(test, test_score, thresholds[key])}
        print(f"[joint] {key}: threshold {thresholds[key]:.4f}; test recall {100 * report[key]['test']['recall']['point']:.1f}%, "
              f"unneeded {100 * report[key]['test']['unneeded_request_rate']['point']:.1f}%", flush=True)
    manifest = runtime.export_checkpoints(
        checkpoints, output, threshold=thresholds["case14"], thresholds=thresholds,
        threshold_rule=(f"per network, the recall target {RECALL_TARGET:.4f} of the IEEE 14 export on that network's "
                        "calibration split; mean of the seeds' request probabilities"),
        source={"ieee14_dataset": str(data.DEFAULT_DATASET), "ieee118_dataset": str(Path(args.dataset_dir).resolve()),
                "admission_rule": args.admission_rule, "train_rows": result["train_rows"], "validation_rows": result["validation_rows"],
                "parameters": result["parameters"], "seeds": args.seeds, "history": result["history"],
                "test": {key: {"recall": report[key]["test"]["recall"], "unneeded_request_rate": report[key]["test"]["unneeded_request_rate"],
                               "auc": report[key]["test"]["auc"], "by_family": report[key]["test"]["by_family"]} for key in report},
                "seconds": time.perf_counter() - started},
    )
    (train_dir / "report.json").write_text(json.dumps({"thresholds": thresholds, "report": report}, indent=1, default=float),
                                                   encoding="utf-8")
    print(json.dumps({"model_id": manifest["model_id"], "view": manifest["view"], "request_thresholds": manifest["request_thresholds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
