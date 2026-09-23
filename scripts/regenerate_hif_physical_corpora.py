"""Regenerate the IEEE-14 physical HIF corpora and their WLS-detectable subsets.

Stages (run in order, each resumable on its own):

1. ``generate``   six corpora with the recipes and seeds of the 2026-09-19/21 corpora
2. ``validate``   sample validator, strict-physics multiscan replay, branch-current localization
3. ``audit``      operator WLS and discovered-mode scenario admission (scripts/audit_hif_physical_corpora.py)
4. ``subset``     detectable subsets of the four main corpora: every healthy control plus the
                  admitted HIF windows, rows copied verbatim, meta.json gains ``detectable_filter``
5. ``reaudit``    audit of the subsets (every HIF window must be admitted)

Seeds are unchanged, so fault labels and operating points repeat the earlier corpora; only the
simulated physics differs. New directories carry ``--tag``; nothing existing is overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN_BANDS = ["--r-hif-ohm-bands", "100:200,200:500,500:1000", "--r-hif-ohm-band-weights", "1,1,1", "--voltage-stratum", "69kv"]
# (name stem, HIF windows, healthy controls, seed, extra arguments, detectable-subset stem or None)
RECIPES = (
    ("hif_physical69_main_train_84x10", 84, 20, 20260919, MAIN_BANDS, "hif_physical69_main_train_detectable"),
    ("hif_physical69_main_valid_21x10", 21, 5, 20260920, MAIN_BANDS, "hif_physical69_main_valid_detectable"),
    ("hif_physical69_detection_limit_21x10", 21, 0, 20260921,
     ["--r-hif-ohm-min", "1000", "--r-hif-ohm-max", "5000", "--voltage-stratum", "69kv"], None),
    ("hif_physical_sweep_eval_336x10", 336, 0, 20260922,
     ["--r-hif-ohm-sweep", "50,100,200,500,1000,2000,5000", "--voltage-stratum", "all_same_voltage"], None),
    ("hif_physical69_main_train_extra_252x10", 252, 60, 20260923, MAIN_BANDS, "hif_physical69_main_train_extra_detectable"),
    ("hif_physical69_main_valid_extra_63x10", 63, 15, 20260924, MAIN_BANDS, "hif_physical69_main_valid_extra_detectable"),
)
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
ADMISSION = ("scenario generator discovered-mode WLS admission, anomaly margin 1.25 "
             "(chi-square 0.01 OR normalized residual 4.0), reference scan")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(command: list[str], log: Path) -> dict:
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
    return {"command": command, "exit_code": result.returncode, "seconds": round(time.monotonic() - start, 1),
            "log": str(log)}


def _write_receipts(path: Path, receipts: list[dict]) -> None:
    path.write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")


def generate(artifacts: Path, out: Path, tag: str, workers: int) -> None:
    def job(recipe):
        stem, n_hif, n_clean, seed, extra, _ = recipe
        target = artifacts / f"{stem}_{tag}"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        command = [sys.executable, "Transmission/generate_measurements_hif_ieee14.py", "--out", str(target),
                   "--n-hif", str(n_hif), "--n-no-error", str(n_clean), "--seed", str(seed),
                   "--scans-per-window", "10", "--resistance-units", "ohm", *extra]
        receipt = {"corpus": target.name, **_run(command, out / f"{target.name}_generation.log")}
        if receipt["exit_code"] == 0:
            rows = _rows(target / "samples.jsonl")
            receipt.update(actual_hif=sum(r["scenario"] == "high_impedance_fault" for r in rows),
                           actual_healthy=sum(r["scenario"] == "no_error" for r in rows),
                           actual_scans=sum(len(r.get("scans") or []) for r in rows))
        receipt.update(expected_hif=n_hif, expected_healthy=n_clean)
        print(json.dumps(receipt), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(job, RECIPES))
    _write_receipts(out / "generation_receipts.json", receipts)
    bad = [r["corpus"] for r in receipts if r["exit_code"] or r.get("actual_hif") != r["expected_hif"]
           or r.get("actual_healthy") != r["expected_healthy"]]
    if bad:
        raise SystemExit(f"incomplete generation: {bad}")


def validate(artifacts: Path, out: Path, tag: str, workers: int, names: list[str] | None = None) -> None:
    names = names or [f"{recipe[0]}_{tag}" for recipe in RECIPES]
    tasks = []
    for name in names:
        d = artifacts / name
        tasks += [
            (name, 0, ["scripts/validate_hif_samples.py", str(d / "samples.jsonl"), "--meta", str(d / "meta.json"),
                       "--allow-non-top3-detectability", "weak,extreme"]),
            (name, 1, ["scripts/validate_hif_multiscan_dataset.py", str(d / "samples.jsonl"), "--meta", str(d / "meta.json"),
                       "--strict-physics", "--output", str(d / "quality_report.json")]),
            (name, 2, ["scripts/validate_branch_current_localization.py", str(d / "samples.jsonl"),
                       "--output", str(d / "branch_current_localization_report.json")]),
        ]

    def job(task):
        name, index, command = task
        receipt = {"corpus": name, "validator": command[0],
                   **_run([sys.executable, *command], out / f"{name}_validation_{index}.log")}
        print(json.dumps(receipt), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(job, tasks))
    _write_receipts(out / "validation_receipts.json", receipts)
    failed = [(r["corpus"], r["validator"]) for r in receipts if r["exit_code"]]
    if failed:
        print(json.dumps({"validation_failures": failed}), flush=True)


def audit(artifacts: Path, out: Path, tag: str, audit_name: str = "detection_audit", names: list[str] | None = None) -> None:
    names = names or [f"{recipe[0]}_{tag}" for recipe in RECIPES]
    target = out / audit_name
    command = [sys.executable, "scripts/audit_hif_physical_corpora.py", "--corpus-dirs",
               *[str(artifacts / name) for name in names], "--out", str(target)]
    log = out / f"{audit_name}.log"
    receipt = _run(command, log)
    print(json.dumps(receipt), flush=True)
    _write_receipts(out / f"{audit_name}_receipt.json", [receipt])
    if receipt["exit_code"]:
        raise SystemExit(f"audit failed; see {log}")


def subset(artifacts: Path, out: Path, tag: str) -> list[str]:
    summary = json.loads((out / "detection_audit" / "summary.json").read_text(encoding="utf-8"))
    repo = artifacts.parents[1]
    audit_dir = out / "detection_audit"
    audit_ref = (audit_dir.relative_to(repo).as_posix() if audit_dir.is_relative_to(repo) else audit_dir.as_posix())
    written = []
    for stem, _, _, _, _, subset_stem in RECIPES:
        if subset_stem is None:
            continue
        name = f"{stem}_{tag}"
        source = artifacts / name
        admitted = {row["id"] for row in summary["cohorts"][name]["admission_rows"] if row["admitted"]}
        lines = [line for line in (source / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        kept, controls, windows = [], 0, 0
        for line in lines:
            row = json.loads(line)
            if row["scenario"] == "no_error":
                kept.append(line)
                controls += 1
            elif row["scenario"] == "high_impedance_fault":
                windows += 1
                if row["id"] in admitted:
                    kept.append(line)
        target = artifacts / f"{subset_stem}_{len(admitted)}x10_{tag}"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        target.mkdir(parents=True)
        (target / "samples.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
        meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
        meta["detectable_filter"] = {
            "schema": "discovered_mode_admission_filter_v1", "source_corpus": name,
            "source_samples_sha256": hashlib.sha256((source / "samples.jsonl").read_bytes()).hexdigest(),
            "criterion": ADMISSION, "audit": audit_ref,
            "kept_windows": len(admitted), "source_windows": windows, "controls_kept": controls,
            "note": "controls are kept unchanged; only disturbance windows that the operator WLS can discover are retained"}
        (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for report in ("quality_report.json", "branch_current_localization_report.json"):
            shutil.copy2(source / report, target / report)
        written.append(target.name)
        print(json.dumps({"subset": target.name, "kept_windows": len(admitted), "source_windows": windows,
                          "controls": controls}), flush=True)
    (out / "detectable_subsets.json").write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="date suffix for the new corpus directories")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts" / "measurements")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", default="generate,validate,audit,subset,reaudit")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    artifacts, out = args.artifacts_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "generate" in stages:
        generate(artifacts, out, args.tag, args.workers)
    if "validate" in stages:
        validate(artifacts, out, args.tag, args.workers * 3)
    if "audit" in stages:
        audit(artifacts, out, args.tag)
    if "subset" in stages:
        subset(artifacts, out, args.tag)
    if "reaudit" in stages:
        names = json.loads((out / "detectable_subsets.json").read_text(encoding="utf-8"))
        audit(artifacts, out, args.tag, audit_name="detectable_audit", names=names)


if __name__ == "__main__":
    main()
