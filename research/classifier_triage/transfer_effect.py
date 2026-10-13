"""Zero-shot GNN transfer in both directions on effect-targeted datasets (2026-10-12).

    python -m research.classifier_triage.transfer_effect --dataset case14=output/.../ieee14_effect \\
        --dataset case118=output/.../ieee118_effect --output-dir output/.../transfer_effect --view residual_v3

Both datasets have the layout of ``research.classifier_triage.ieee118_dataset``
(roots, healthy alarms, mimics, saved WLS payloads, parent-based splits) and
were generated with the same effect band, so a root's magnitude carries no
family or network information.  For every dataset the triage GNN is trained
on its train split and its request threshold is set on its calibration split
at the IEEE 14 export's recall target; it is then scored on its own test split
and zero-shot on every other dataset's test split (at its own threshold, and
at a threshold re-chosen on the target's calibration split, which bounds what
a threshold alone could recover).  A joint model trained on all datasets gives
the in-domain ceiling.  Roots are kept when their own solve's effect ratio is
in the band (the band is the admission); healthy windows and mimics keep the
operator alarm.
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
from research.classifier_triage.effect_sampler import DEFAULT_BAND, effect_ratio, parse_band  # noqa: E402
from research.classifier_triage.transfer_ieee118 import load_ieee118_rows, recall_threshold, summarize  # noqa: E402
from research.hypothesis_ranking.features import json_safe  # noqa: E402

RECALL_TARGET = 0.9927536231884058


def load_dataset(name: str, directory: Path) -> list[dict[str, Any]]:
    """Rows of one effect-targeted dataset, tagged with the dataset name."""
    rows = load_ieee118_rows(directory)
    for row in rows:
        row["dataset"] = name
        row["id"] = f"{name}:{row['id']}"
        row["parent"] = f"{name}:{row['parent']}"
    return rows


def in_band(rows: Sequence[dict[str, Any]], band: tuple[float, float], slack: float = 0.15) -> list[dict[str, Any]]:
    """Roots whose own solve's effect ratio is in the band (with slack for the sampler's tolerance)."""
    low, high = band
    kept = []
    for row in rows:
        if row["kind"] != "root":
            kept.append(row)
            continue
        payload = row["payload"]
        ratio = effect_ratio(float(payload["objective"]), int(payload["dof"]))
        if low * (1.0 - slack) <= ratio <= high * (1.0 + slack):
            kept.append(row)
    return kept


def family_rates(rows: Sequence[Mapping[str, Any]], score: np.ndarray, threshold: float) -> dict[str, str]:
    flags = np.asarray(score) >= threshold
    out = {}
    for family in sorted({row["family"] for row in rows}):
        mask = np.asarray([row["family"] == family for row in rows])
        out[family] = f"{int(flags[mask].sum())}/{int(mask.sum())}"
    return out


def ratio_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Median effect ratio per family of the roots (what the band produced)."""
    out: dict[str, list[float]] = {}
    for row in rows:
        if row["kind"] != "root":
            continue
        payload = row["payload"]
        out.setdefault(row["family"], []).append(effect_ratio(float(payload["objective"]), int(payload["dof"])))
    return {family: {"n": len(v), "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}
            for family, v in sorted(out.items())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, help="name=directory (two or more)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", default="residual_v3")
    parser.add_argument("--band", default=f"{DEFAULT_BAND[0]},{DEFAULT_BAND[1]}")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-joint", action="store_true")
    args = parser.parse_args(argv)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    band = parse_band(args.band)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    datasets: dict[str, list[dict[str, Any]]] = {}
    for item in args.dataset:
        name, _, directory = item.partition("=")
        datasets[name] = in_band(load_dataset(name, Path(directory)), band)
        split = data.split_rows(datasets[name])
        print(f"[effect] {name}: {len(datasets[name])} rows in band (train {len(split['train'])}, calibration "
              f"{len(split['calibration'])}, test {len(split['test'])}); roots by family: "
              f"{ {f: v['n'] for f, v in ratio_table(datasets[name]).items()} }", flush=True)
    seeds = list(range(args.seeds))
    report: dict[str, Any] = {"view": args.view, "band": list(band), "seeds": seeds, "device": device,
                              "datasets": {name: {"rows": len(rows), "ratio_by_family": ratio_table(rows)} for name, rows in datasets.items()},
                              "models": {}}
    names = list(datasets)
    tests = {name: data.split_rows(rows)["test"] for name, rows in datasets.items()}
    for name in names:
        others = [n for n in names if n != name]
        extra = [row for n in others for row in tests[n]]
        result = train_gnn.train_and_score(datasets[name], extra=extra, view=args.view, seeds=seeds, device=device,
                                           checkpoint_dir=out / f"checkpoints_{name}", log=lambda text: print(text, flush=True))
        by_id = {rid: k for k, rid in enumerate(result["ids"])}
        split = data.split_rows(datasets[name])
        calibration = [r for r in split["calibration"] if r["id"] in by_id]
        score_c = np.asarray([result["needs_aux"][by_id[r["id"]]] for r in calibration])
        y_c = np.asarray([r["triage"]["needs_aux"] for r in calibration])
        threshold = recall_threshold(score_c, y_c, RECALL_TARGET)
        entry: dict[str, Any] = {"threshold": threshold, "train_rows": result["train_rows"], "scored": {}}
        for target in names:
            rows = [r for r in tests[target] if r["id"] in by_id]
            score = np.asarray([result["needs_aux"][by_id[r["id"]]] for r in rows])
            y = np.asarray([r["triage"]["needs_aux"] for r in rows])
            scored = {"at_own_threshold": summarize(rows, score, threshold),
                      "families_at_own_threshold": family_rates(rows, score, threshold)}
            if target != name:
                cal_t = data.split_rows(datasets[target])["calibration"]
                # the target's calibration rows were not scored by this model; score them now
                cal_scores = train_gnn.score_with_checkpoints(out / f"checkpoints_{name}", cal_t, args.view, device)["needs_aux"]
                t_target = recall_threshold(np.asarray(cal_scores), np.asarray([r["triage"]["needs_aux"] for r in cal_t]), RECALL_TARGET)
                scored["at_target_threshold"] = summarize(rows, score, t_target)
                scored["target_threshold"] = t_target
            entry["scored"][target] = scored
            line = scored["at_own_threshold"]
            print(f"[effect] trained on {name} -> {target}: recall {100 * line['recall']['point']:.1f}%, unneeded "
                  f"{100 * line['unneeded_request_rate']['point']:.1f}%, AUC {100 * line['auc']['point']:.1f}%"
                  + (f" (recall {100 * scored['at_target_threshold']['recall']['point']:.1f}% at the target's threshold)" if target != name else ""),
                  flush=True)
        report["models"][name] = entry
    if not args.no_joint and len(names) > 1:
        joint_rows = [row for rows in datasets.values() for row in rows]
        result = train_gnn.train_and_score(joint_rows, view=args.view, seeds=seeds, device=device,
                                           checkpoint_dir=out / "checkpoints_joint", log=lambda text: print(text, flush=True))
        by_id = {rid: k for k, rid in enumerate(result["ids"])}
        entry = {"train_rows": result["train_rows"], "scored": {}}
        for target in names:
            split = data.split_rows(datasets[target])
            calibration = [r for r in split["calibration"] if r["id"] in by_id]
            threshold = recall_threshold(np.asarray([result["needs_aux"][by_id[r["id"]]] for r in calibration]),
                                         np.asarray([r["triage"]["needs_aux"] for r in calibration]), RECALL_TARGET)
            rows = [r for r in tests[target] if r["id"] in by_id]
            score = np.asarray([result["needs_aux"][by_id[r["id"]]] for r in rows])
            entry["scored"][target] = {"threshold": threshold, "at_own_threshold": summarize(rows, score, threshold),
                                       "families_at_own_threshold": family_rates(rows, score, threshold)}
            line = entry["scored"][target]["at_own_threshold"]
            print(f"[effect] joint -> {target}: recall {100 * line['recall']['point']:.1f}%, unneeded "
                  f"{100 * line['unneeded_request_rate']['point']:.1f}%, AUC {100 * line['auc']['point']:.1f}%", flush=True)
        report["models"]["joint"] = entry
    report["seconds"] = time.perf_counter() - started
    (out / "transfer_effect.json").write_text(json.dumps(json_safe(report), indent=1, sort_keys=True), encoding="utf-8")
    lines = [f"# Zero-shot GNN transfer on effect-targeted datasets (view {args.view}, band {band[0]}-{band[1]})", "",
             "| model | scored on | recall | unneeded | AUC | recall at target threshold |", "| --- | --- | --- | --- | --- | --- |"]
    for model, entry in report["models"].items():
        for target, scored in entry["scored"].items():
            line = scored["at_own_threshold"]
            alt = scored.get("at_target_threshold")
            lines.append(f"| {model} | {target} | {100 * line['recall']['point']:.1f}% | {100 * line['unneeded_request_rate']['point']:.1f}% | "
                         f"{100 * line['auc']['point']:.1f}% | {(f'{100 * alt['recall']['point']:.1f}%') if alt else '-'} |")
    (out / "transfer_effect.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
