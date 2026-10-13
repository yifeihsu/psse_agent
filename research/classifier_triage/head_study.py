"""What the other heads of the effect-run GNNs know: waveform-family routing, first-family ordering,
and what the misses look like. Scores the saved transfer_v3 checkpoints on the effect test splits."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.classifier_triage import data, train_gnn  # noqa: E402
from research.classifier_triage.effect_sampler import DEFAULT_BAND  # noqa: E402
from research.classifier_triage.model import FAMILY_HEADS, FIRST_CLASSES  # noqa: E402
from research.classifier_triage.transfer_effect import in_band, load_dataset  # noqa: E402
from research.hypothesis_ranking import ranker  # noqa: E402

RUN = REPO / "output/classifier_triage_20261012/transfer_v3"
VIEW = "residual_v3"
WAVE = ("hif", "unbalance", "harmonic")
WIDX = [FAMILY_HEADS.index(w) for w in WAVE]
report = json.load(open(RUN / "transfer_effect.json", encoding="utf-8"))
device = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"
except Exception:
    pass

datasets = {
    "case14": in_band(load_dataset("case14", REPO / "output/classifier_triage_20261012/ieee14_effect"), DEFAULT_BAND),
    "case118": in_band(load_dataset("case118", REPO / "output/classifier_triage_20261012/ieee118_effect"), DEFAULT_BAND),
}
tests = {name: data.split_rows(rows)["test"] for name, rows in datasets.items()}
out = {}
for model in ("case14", "case118", "joint"):
    ck = RUN / f"checkpoints_{model}"
    for target, rows in tests.items():
        thr = report["models"][model].get("threshold") or report["models"][model]["scored"][target]["threshold"]
        scored = train_gnn.score_with_checkpoints(ck, rows, VIEW, device)
        aux = np.asarray(scored["needs_aux"]); fam = np.asarray(scored["family"]); first = np.asarray(scored["first"])
        y = np.asarray([r["triage"]["needs_aux"] for r in rows]); flags = aux >= thr
        wave_true = [tuple(r["triage"]["waveform"]) for r in rows]
        pos = np.where(y == 1)[0]
        # (a) routing: argmax over the three waveform heads vs the true waveform family (positives with one waveform family)
        confusion = {w: {v: 0 for v in WAVE} for w in WAVE}
        route_hit = {w: [0, 0] for w in WAVE}
        route_hit_flagged = {w: [0, 0] for w in WAVE}
        for i in pos:
            if len(wave_true[i]) != 1:
                continue
            truth = wave_true[i][0]
            guess = WAVE[int(np.argmax(fam[i, WIDX]))]
            confusion[truth][guess] += 1
            route_hit[truth][1] += 1; route_hit[truth][0] += int(guess == truth)
            if flags[i]:
                route_hit_flagged[truth][1] += 1; route_hit_flagged[truth][0] += int(guess == truth)
        # (b) the waveform heads as an alternative gate: max of the three, at the same unneeded rate as needs_aux
        alt = fam[:, WIDX].max(axis=1)
        neg = np.where(y == 0)[0]
        unneeded = float(flags[neg].mean()) if neg.size else 0.0
        # threshold on alt that gives the same unneeded rate
        neg_alt = np.sort(alt[neg])[::-1]
        k = int(np.floor(unneeded * neg.size + 1e-9))
        alt_thr = float(np.nextafter(neg_alt[k], np.inf)) if k < neg.size else float(neg_alt[-1])
        alt_flags = alt >= alt_thr
        # (c) misses: how the needs_aux score and the heads look on missed positives
        missed = [i for i in pos if not flags[i]]
        miss_desc = {}
        for i in missed:
            fam_name = rows[i]["family"]
            d = miss_desc.setdefault(fam_name, {"n": 0, "score_median": [], "below_half_thr": 0, "head_says_waveform": 0, "head_guess": {}})
            d["n"] += 1; d["score_median"].append(float(aux[i]))
            d["below_half_thr"] += int(aux[i] < 0.5 * thr)
            d["head_says_waveform"] += int(alt[i] >= 0.5)
            g = WAVE[int(np.argmax(fam[i, WIDX]))]; d["head_guess"][g] = d["head_guess"].get(g, 0) + 1
        for d in miss_desc.values():
            d["score_median"] = float(np.median(d["score_median"]))
        # (d) first-family head on balanced roots with a first label
        first_rows = [(i, r["triage"]["first"]) for i, r in enumerate(rows) if r["triage"]["first"] in FIRST_CLASSES and y[i] == 0]
        first_acc = {}
        for i, truth in first_rows:
            guess = FIRST_CLASSES[int(np.argmax(first[i]))]
            d = first_acc.setdefault(truth, [0, 0]); d[1] += 1; d[0] += int(guess == truth)
        mixed = [(i, t) for i, t in first_rows if len(rows[i]["triage"]["families"]) > 1]
        mixed_acc = [int(FIRST_CLASSES[int(np.argmax(first[i]))] == t) for i, t in mixed]
        out[f"{model}->{target}"] = {
            "threshold": thr, "rows": len(rows), "positives": int(pos.size),
            "needs_aux": {"recall": float(flags[pos].mean()), "unneeded": unneeded, "auc": float(ranker.auc(aux, y))},
            "max_waveform_head_as_gate": {"recall_at_same_unneeded": float(alt_flags[pos].mean()), "unneeded": float(alt_flags[neg].mean()),
                                          "auc": float(ranker.auc(alt, y))},
            "routing_all_positives": {w: f"{a}/{b}" for w, (a, b) in route_hit.items()},
            "routing_flagged_positives": {w: f"{a}/{b}" for w, (a, b) in route_hit_flagged.items()},
            "routing_confusion(true->guess)": confusion,
            "misses": miss_desc,
            "first_family_accuracy": {w: f"{a}/{b}" for w, (a, b) in first_acc.items()},
            "first_family_on_mixed": f"{sum(mixed_acc)}/{len(mixed_acc)}",
        }
        print(f"scored {model}->{target}", flush=True)
json.dump(out, open(RUN / "head_study.json", "w", encoding="utf-8"), indent=1)
print("done")
