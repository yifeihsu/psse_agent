"""Step 1 transfer check: the balanced screen on the IEEE 57 HIF, unbalance and healthy study roots.

Reads the 2026-09-11 IEEE 57 validation study
(``output/ieee57_hif_unbalance_20260911_verified``): for every recorded root
the nominal-noise 491-channel SCADA vector is solved with the repository WLS
on the canonical case57 (sigma 0.001 pu for voltages, 0.01 pu for powers, the
study's declaration) and, when the alarm rule fires (chi-square at alpha 0.01
or a normalized residual at or above 4), screened with the same balanced
screen as IEEE 14.  case57 carries no base kV, so ``default_hif_lines`` gives
the screen no candidate lines there; the study passes every in-service
untapped branch instead and records that rule, because that is the gap a
transfer would have to close.

    python -m research.hypothesis_ranking.ieee57_screen --output-dir output/hypothesis_ranking_20260930/ieee57 --workers 16
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.systems.registry import resolve_system  # noqa: E402
from research.hypothesis_ranking.features import analyze_state, candidate_lines, json_safe  # noqa: E402
from research.hypothesis_ranking.report import _pct, _rate, _table, alarmed, final_class, screen_of, suspicion_v2  # noqa: E402

STUDY_DIR = REPO_ROOT / "output" / "ieee57_hif_unbalance_20260911_verified"
MODELS = ("normalized_diagonal_100", "normalized_diagonal_080", "coupled_sensitivity_100", "coupled_sensitivity_080")


def _records(study_dir: Path, models: Sequence[str], case_path: str, sigma: list[float]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model in models:
        results_path = study_dir / model / "results.json"
        if not results_path.is_file():
            continue
        for entry in json.loads(results_path.read_text(encoding="utf-8")):
            observations = study_dir / str(entry["observations_path"]).replace("\\", "/")
            if not observations.is_file():
                continue
            with gzip.open(observations, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            z = payload["noisy_nominal"]["measurement_vector"]
            truth = dict(entry.get("truth") or {})
            family = {"hif": "hif", "unbalance": "three_phase_unbalance", "healthy": "healthy_window"}.get(str(entry.get("family")), str(entry.get("family")))
            records.append({
                "root_id": f"{model}:{entry['scenario_id']}", "family": family, "model": model,
                "parent_id": f"ieee57_study:{entry.get('physical_root') or entry['scenario_id']}",
                "source_id": str(entry["scenario_id"]), "case": case_path,
                "measurements": [float(v) for v in z], "metadata": {"sigma_z": sigma},
                "truth": {
                    "measurement": [], "parameter": [], "topology": [],
                    "hif": [truth] if entry.get("family") == "hif" else [],
                    "unbalance": [truth] if entry.get("family") == "unbalance" else [],
                    "harmonic": [],
                },
                "study_wls_noisy": entry.get("wls", {}).get("noisy"),
            })
    return records


_LINES: list[int] | None = None
_RULE: str | None = None


def _analyze(record: Mapping[str, Any]) -> dict[str, Any]:
    global _LINES, _RULE
    out = dict(record)
    try:
        if _LINES is None:
            _LINES, _RULE = candidate_lines(resolve_system("case57").load_case())
        out["analysis"] = analyze_state(record, lines=_LINES)
        if isinstance(out["analysis"].get("screen"), dict):
            out["analysis"]["screen"]["candidate_line_rule"] = _RULE
    except Exception as exc:
        out["analysis"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-dir", default=str(STUDY_DIR))
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--limit", type=int, default=None, help="analyze only the first N records (smoke)")
    parser.add_argument("--summary-only", action="store_true", help="rebuild report.md/metrics.json from an existing roots.jsonl")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    system = resolve_system("case57")
    sigma = [float(v) for v in system.measurement_sigma()]
    started = time.perf_counter()
    lines, rule = candidate_lines(system.load_case())
    if args.summary_only:
        with (out / "roots.jsonl").open(encoding="utf-8") as stream:
            analyzed = [json.loads(line) for line in stream if line.strip()]
        print(f"[ieee57] summary only: {len(analyzed)} saved roots", flush=True)
    else:
        records = _records(Path(args.study_dir), args.models, system.case_path, sigma)
        if args.limit:
            records = records[: args.limit]
        print(f"[ieee57] {len(records)} study roots; candidate lines {len(lines)} by {rule}", flush=True)
        if args.workers <= 1:
            analyzed = [_analyze(r) for r in records]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                analyzed = list(pool.map(_analyze, records, chunksize=2))
        with (out / "roots.jsonl").open("w", encoding="utf-8") as stream:
            for row in analyzed:
                stream.write(json.dumps(json_safe(row), sort_keys=True) + "\n")

    # --------------------------------------------------------------- summary
    from scipy.stats import chi2 as _chi2

    reproduction = Counter()
    for row in analyzed:
        study = row.get("study_wls_noisy") or {}
        wls = (row.get("analysis") or {}).get("wls") or {}
        if study.get("chi_square_statistic") is not None and wls.get("chi_square_statistic") is not None:
            reproduction["compared"] += 1
            # The 0.80-load models store their statistics with about six significant digits.
            reproduction["chi_square_within_1e-5_relative"] += abs(wls["chi_square_statistic"] - study["chi_square_statistic"]) <= 1e-5 * max(1.0, abs(study["chi_square_statistic"]))
            threshold_05 = float(_chi2.ppf(0.95, max(int(wls["chi_square_dof"]), 1)))
            alarm_05 = wls["chi_square_statistic"] >= threshold_05 or wls["max_normalized_residual"] >= 4.0
            reproduction["study_alarms_alpha_0.05"] += bool(study.get("alarm"))
            reproduction["my_alarms_alpha_0.05"] += bool(alarm_05)
            reproduction["alarm_agreement_alpha_0.05"] += bool(alarm_05) == bool(study.get("alarm"))
            reproduction["my_alarms_alpha_0.01"] += bool(wls.get("alarm"))
    by_family: dict[str, Counter] = defaultdict(Counter)
    by_resistance: dict[str, Counter] = defaultdict(Counter)
    by_delta: dict[str, Counter] = defaultdict(Counter)
    for row in analyzed:
        family = row["family"]
        c = by_family[family]
        c["n"] += 1
        c["alarmed"] += alarmed(row)
        if alarmed(row):
            c["final_" + final_class(row)] += 1
            c["suspicion_v1"] += bool((screen_of(row) or {}).get("suspected"))
            c["suspicion_v2"] += suspicion_v2(row)
            if family == "hif" and row["truth"]["hif"]:
                truth = row["truth"]["hif"][0]
                c["hif_right_line"] += bool((screen_of(row) or {}).get("suspected")) and int((screen_of(row) or {}).get("branch_row0", -1)) == int(truth.get("branch_row0", -2))
        if family == "hif" and row["truth"]["hif"]:
            key = str(row["truth"]["hif"][0].get("resistance_pu"))
            by_resistance[key]["n"] += 1
            by_resistance[key]["alarmed"] += alarmed(row)
            if alarmed(row):
                by_resistance[key]["final_" + final_class(row)] += 1
                by_resistance[key]["suspicion_v2"] += suspicion_v2(row)
        if family == "three_phase_unbalance" and row["truth"]["unbalance"]:
            key = str(row["truth"]["unbalance"][0].get("delta"))
            by_delta[key]["n"] += 1
            by_delta[key]["alarmed"] += alarmed(row)
            if alarmed(row):
                by_delta[key]["final_" + final_class(row)] += 1
                by_delta[key]["suspicion_v2"] += suspicion_v2(row)
    failures = Counter()
    for row in analyzed:
        analysis = row.get("analysis") or {}
        if analysis.get("error"):
            failures["analysis_error"] += 1
        elif (analysis.get("wls") or {}).get("success") is False:
            failures["wls_failure"] += 1
        elif alarmed(row) and (screen_of(row) or {}).get("status") not in (None, "valid"):
            failures["screen_" + str((screen_of(row) or {}).get("status"))] += 1
    screen_times = [(r.get("analysis") or {}).get("screen_seconds") for r in analyzed if (r.get("analysis") or {}).get("screen_seconds")]
    metrics = {
        "study_dir": str(args.study_dir), "models": list(args.models), "records": len(analyzed),
        "candidate_lines": {"count": len(lines), "rule": rule},
        "wls_reproduction": dict(reproduction), "analysis_failures": dict(failures),
        "by_family": {k: dict(v) for k, v in by_family.items()},
        "hif_by_resistance_pu": {k: dict(v) for k, v in by_resistance.items()},
        "unbalance_by_delta": {k: dict(v) for k, v in by_delta.items()},
        "screen_seconds_mean": (sum(screen_times) / len(screen_times)) if screen_times else None,
        "total_seconds": time.perf_counter() - started,
    }
    (out / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8")

    def finals(c: Counter) -> str:
        return ", ".join(f"{k[6:]} {v}" for k, v in sorted(c.items()) if k.startswith("final_"))

    parts = ["# Balanced screen on the IEEE 57 study roots\n",
             f"{len(analyzed)} roots from {', '.join(args.models)}; candidate HIF lines: {len(lines)} ({rule}). "
             f"WLS reproduction against the study's saved statistics (the study alarmed at alpha 0.05; this run uses the repository rule, alpha 0.01 or a normalized residual at or above 4): {dict(reproduction)}. Failures: {dict(failures) or 'none'}. "
             f"Mean screen time {metrics['screen_seconds_mean'] and round(metrics['screen_seconds_mean'], 1)} s.\n",
             _table("By family", ["family", "n", "alarmed", "final classes (alarmed)", "suspicion v1", "suspicion v2", "HIF on true line"],
                    [[f, c["n"], c["alarmed"], finals(c), c["suspicion_v1"], c["suspicion_v2"], c.get("hif_right_line", "")] for f, c in sorted(by_family.items())]),
             _table("HIF by resistance (pu)", ["R", "n", "alarmed", "final classes (alarmed)", "suspicion v2"],
                    [[k, c["n"], c["alarmed"], finals(c), c["suspicion_v2"]] for k, c in sorted(by_resistance.items(), key=lambda kv: float(kv[0]))]),
             _table("Unbalance by delta", ["delta", "n", "alarmed", "final classes (alarmed)", "suspicion v2"],
                    [[k, c["n"], c["alarmed"], finals(c), c["suspicion_v2"]] for k, c in sorted(by_delta.items(), key=lambda kv: float(kv[0]))])]
    (out / "report.md").write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
