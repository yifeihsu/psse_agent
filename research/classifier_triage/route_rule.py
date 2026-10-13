"""How the tracked triage exports' waveform heads would route a spectra-first request.

For each tracked export and each test split it was calibrated on: among the rows the gate flags,
how many of each family have the harmonic head as the largest waveform head (argmax rule), and
how many have harmonic >= 0.5 (absolute rule)."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.classifier_triage import data, train_gnn  # noqa: E402
from research.classifier_triage.model import FAMILY_HEADS  # noqa: E402
from research.classifier_triage.transfer_ieee118 import admitted_rows, load_ieee118_rows  # noqa: E402

WAVE = ("hif", "unbalance", "harmonic")
WIDX = [FAMILY_HEADS.index(w) for w in WAVE]
H = FAMILY_HEADS.index("harmonic")
device = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"
except Exception:
    pass

rows14 = admitted_rows(data.load_rows(workers=8), "excess_energy")
rows118 = admitted_rows(load_ieee118_rows(REPO / "output/classifier_triage_20261011/ieee118_v2"), "excess_energy")
tests = {"case14": data.split_rows(rows14)["test"], "case118": data.split_rows(rows118)["test"]}
# the IEEE 14 export was calibrated on the un-refiltered study rows; its test split is the same parents
exports = {
    "ieee14_20261004": (REPO / "psse_env/oracle/models/triage_gnn_ieee14_20261004", ["case14"]),
    "joint_20261011": (REPO / "psse_env/oracle/models/triage_gnn_joint_20261011", ["case14", "case118"]),
}
(REPO / "output/classifier_triage_20261013").mkdir(parents=True, exist_ok=True)
out = {}
for name, (directory, targets) in exports.items():
    manifest = json.load(open(directory / "runtime.json", encoding="utf-8"))
    for target in targets:
        rows = tests[target]
        thr = (manifest.get("request_thresholds") or {}).get(target, manifest["request_threshold"])
        scored = train_gnn.score_with_checkpoints(directory, rows, manifest["view"], device)
        aux = np.asarray(scored["needs_aux"]); fam = np.asarray(scored["family"])
        flags = aux >= thr
        argmax_h = np.asarray([WAVE[int(np.argmax(fam[i, WIDX]))] == "harmonic" for i in range(len(rows))])
        abs_h = fam[:, H] >= 0.5
        table = {}
        for family in sorted({r["family"] for r in rows}):
            mask = np.asarray([r["family"] == family for r in rows])
            f = mask & flags
            table[family] = {"n": int(mask.sum()), "flagged": int(f.sum()),
                             "spectra_first_argmax": int((f & argmax_h).sum()), "spectra_first_abs0.5": int((f & abs_h).sum()),
                             "harmonic_head_median_flagged": float(np.median(fam[f, H])) if f.any() else None}
        out[f"{name}->{target}"] = {"threshold": thr, "view": manifest["view"], "table": table}
        print(f"\n== {name} scored on {target} (threshold {thr:.4f}, view {manifest['view']})")
        print("family | n | flagged | spectra-first (argmax) | spectra-first (harmonic>=0.5) | harmonic head median on flagged")
        for family, t in table.items():
            med = "-" if t["harmonic_head_median_flagged"] is None else f"{t['harmonic_head_median_flagged']:.3f}"
            print(f"{family} | {t['n']} | {t['flagged']} | {t['spectra_first_argmax']} | {t['spectra_first_abs0.5']} | {med}")
json.dump(out, open(REPO / "output/classifier_triage_20261013/route_rule.json", "w", encoding="utf-8"), indent=1)
print("\ndone")
