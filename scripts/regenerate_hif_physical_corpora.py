"""Regenerate the IEEE-14 physical HIF and unbalance corpora and their WLS-detectable subsets.

Stages (run in order, each resumable on its own; ``--families`` selects hif, unbalance or both):

1. ``generate``   six HIF corpora with the recipes and seeds of the 2026-09-19/21 corpora, and the
                  440-window unbalance corpus with its 2026-09-21 seed
2. ``validate``   HIF: sample validator, strict-physics multiscan replay, branch-current localization;
                  unbalance: exact replay of every balanced reference and unbalanced sensor mean
3. ``audit``      operator WLS and discovered-mode scenario admission
                  (scripts/audit_hif_physical_corpora.py, scripts/audit_unbalance_physical_corpus.py)
4. ``subset``     detectable subsets: every healthy control plus the admitted disturbance windows,
                  rows copied verbatim, meta.json gains ``detectable_filter``
5. ``reaudit``    audit of the subsets (every disturbance window must be admitted)

Seeds are unchanged, so fault labels and operating points repeat the earlier corpora; only the
simulated physics differs. New directories carry ``--tag``; nothing existing is overwritten. An
implementation manifest records the commit, dirty files and digests of the physics sources.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # in-process replay imports repository packages
MAIN_BANDS =["--r-hif-ohm-bands", "100:200,200:500,500:1000", "--r-hif-ohm-band-weights", "1,1,1", "--voltage-stratum", "69kv"]
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
# The three-phase unbalance corpus (Transmission/generate_measurements_imbalance.py), its
# discovered-mode audit (scripts/audit_unbalance_physical_corpus.py) and detectable subset.
UNBALANCE = {"stem": "out_measurements_imbalance_currents_ybus_440", "windows": 440, "controls": 60, "seed": 20260925,
             "subset_stem": "out_measurements_imbalance_currents_ybus_detectable"}
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
    fatal = []
    for receipt in receipts:
        if not receipt["exit_code"]:
            continue
        reason = _tolerable_sample_validator_failure(Path(receipt["log"])) if receipt["validator"].endswith(
            "validate_hif_samples.py") else None
        if reason:
            receipt["tolerated"] = reason
        else:
            fatal.append((receipt["corpus"], receipt["validator"]))
    _write_receipts(out / "validation_receipts.json", receipts)
    tolerated = [(r["corpus"], r["tolerated"]) for r in receipts if r.get("tolerated")]
    if tolerated:
        print(json.dumps({"tolerated_validator_failures": tolerated}), flush=True)
    if fatal:
        raise SystemExit(f"validation failed: {fatal}")


def _tolerable_sample_validator_failure(log: Path) -> str | None:
    """Only legacy-NLM top-1 ranking misses with the target still in the top 3 are tolerated.

    The legacy three-phase NLM diagnostic ranks candidate lines from a load-scale-only
    model that never sees the operating point, so these misses do not describe the
    regenerated physics; every other validator issue is fatal.
    """
    try:
        summary = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    except (IndexError, ValueError):
        return None
    issues = summary.get("issues") or {}
    hif_rows = (summary.get("scenario_counts") or {}).get("high_impedance_fault")
    if set(issues) == {"target_not_top1"} and summary.get("top3_count") == hif_rows:
        return f"{issues['target_not_top1']} legacy-NLM top-1 ranking misses, all targets in the top 3"
    return None


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
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    if not (summary.get("success") and summary.get("complete")) or summary.get("failures"):
        raise SystemExit(f"audit {audit_name} incomplete or unsuccessful; see {target / 'summary.json'}")
    for name in names:
        cohort = summary["cohorts"][name]
        recorded = cohort.get("source_sha256")
        actual = hashlib.sha256((artifacts / name / "samples.jsonl").read_bytes()).hexdigest()
        if recorded and recorded != actual:
            raise SystemExit(f"audit {audit_name} read a different {name}/samples.jsonl than the one on disk")


def subset(artifacts: Path, out: Path, tag: str) -> list[str]:
    summary = json.loads((out / "detection_audit" / "summary.json").read_text(encoding="utf-8"))
    if not (summary.get("success") and summary.get("complete")):
        raise SystemExit("detection audit did not complete successfully; refusing to write subsets")
    repo = artifacts.parents[1]
    audit_dir = out / "detection_audit"
    audit_ref = (audit_dir.relative_to(repo).as_posix() if audit_dir.is_relative_to(repo) else audit_dir.as_posix())
    written = []
    for stem, _, _, _, _, subset_stem in RECIPES:
        if subset_stem is None:
            continue
        name = f"{stem}_{tag}"
        source = artifacts / name
        cohort = summary["cohorts"][name]
        source_sha = hashlib.sha256((source / "samples.jsonl").read_bytes()).hexdigest()
        if cohort.get("source_sha256") not in (None, source_sha):
            raise SystemExit(f"{name}: audit digest differs from samples.jsonl on disk")
        admitted = {row["id"] for row in cohort["admission_rows"] if row["admitted"]}
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
            "source_samples_sha256": source_sha,
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


def reaudit(artifacts: Path, out: Path, tag: str) -> None:
    names = json.loads((out / "detectable_subsets.json").read_text(encoding="utf-8"))
    audit(artifacts, out, tag, audit_name="detectable_audit", names=names)
    summary = json.loads((out / "detectable_audit" / "summary.json").read_text(encoding="utf-8"))
    for name in names:
        counts = summary["cohorts"][name]["counts"]
        if counts.get("scenario_admitted") != counts.get("hif_windows"):
            raise SystemExit(f"{name}: re-audit admits {counts.get('scenario_admitted')} of {counts.get('hif_windows')} windows")
    print(json.dumps({"reaudit": {name: summary["cohorts"][name]["counts"]["scenario_admitted"] for name in names}}), flush=True)


# ----------------------------------------------------------------- unbalance family

def _unbalance_name(tag: str) -> str:
    return f"{UNBALANCE['stem']}_{tag}"


def generate_unbalance(artifacts: Path, out: Path, tag: str) -> None:
    target = artifacts / _unbalance_name(tag)
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    command = [sys.executable, "Transmission/generate_measurements_imbalance.py", "--out", str(target),
               "--n-imbalance", str(UNBALANCE["windows"]), "--n-no-error", str(UNBALANCE["controls"]),
               "--seed", str(UNBALANCE["seed"])]
    receipt = {"corpus": target.name, **_run(command, out / f"{target.name}_generation.log")}
    if receipt["exit_code"] == 0:
        rows = _rows(target / "samples.jsonl")
        receipt.update(actual_unbalance=sum(r["scenario"] == "three_phase_imbalance" for r in rows),
                       actual_healthy=sum(r["scenario"] == "no_error" for r in rows))
    receipt.update(expected_unbalance=UNBALANCE["windows"], expected_healthy=UNBALANCE["controls"])
    print(json.dumps(receipt), flush=True)
    _write_receipts(out / "generation_receipts_unbalance.json", [receipt])
    if receipt["exit_code"] or receipt.get("actual_unbalance") != UNBALANCE["windows"] \
            or receipt.get("actual_healthy") != UNBALANCE["controls"]:
        raise SystemExit(f"incomplete unbalance generation: {receipt}")


def validate_unbalance(artifacts: Path, out: Path, tag: str) -> None:
    """Strict replay: every row's balanced reference and unbalanced sensor mean re-solve exactly."""
    import numpy as np
    from Transmission import generate_measurements_imbalance as gi
    from IEEE_14_OpenDSS.export_measurement_series import extract_measurement_series
    name = _unbalance_name(tag)
    rows = [r for r in _rows(artifacts / name / "samples.jsonl") if r["scenario"] == "three_phase_imbalance"]
    repo = str(ROOT / "IEEE_14_OpenDSS")
    gi._compile_ieee14_opendss(repo)
    base_loads = gi._read_base_loads()
    worst_true = worst_clean = 0.0
    for row in rows:
        scale = float(row["op_point"]["load_scale"])
        split = row["label"]["load_split"]
        gi._compile_ieee14_opendss(repo)
        gi._scale_all_loads(base_loads, scale)
        gi._solve_or_raise()
        z_true = np.asarray(extract_measurement_series(shunt_convention="ybus")[0], dtype=float)
        worst_true = max(worst_true, float(np.max(np.abs(z_true - np.asarray(row["z_true"])))))
        gi._compile_ieee14_opendss(repo)
        gi._set_loads_scaled_with_bus_unbalance(base_loads, target_bus=split["bus"], load_scale=scale,
                                                bus_fracs=tuple(split["fractions"][p] for p in ("a", "b", "c")))
        gi._solve_or_raise()
        z_clean = np.asarray(extract_measurement_series(shunt_convention="ybus")[0], dtype=float)
        worst_clean = max(worst_clean, float(np.max(np.abs(z_clean - np.asarray(row["z_clean"])))))
    report = {"corpus": name, "rows": len(rows), "max_abs_z_true_replay_error_pu": worst_true,
              "max_abs_z_clean_replay_error_pu": worst_clean, "tolerance_pu": 1e-9}
    (out / f"{name}_replay.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if max(worst_true, worst_clean) > 1e-9:
        raise SystemExit(f"unbalance corpus does not replay: {report}")


def audit_unbalance(artifacts: Path, out: Path, tag: str, *, corpus: str | None = None, audit_name: str = "unbalance_audit") -> dict:
    corpus = corpus or _unbalance_name(tag)
    target = out / audit_name
    command = [sys.executable, "scripts/audit_unbalance_physical_corpus.py", "--corpus", str(artifacts / corpus), "--out", str(target)]
    receipt = _run(command, out / f"{audit_name}.log")
    print(json.dumps(receipt), flush=True)
    _write_receipts(out / f"{audit_name}_receipt.json", [receipt])
    if receipt["exit_code"]:
        raise SystemExit(f"unbalance audit failed; see {receipt['log']}")
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256((artifacts / corpus / "samples.jsonl").read_bytes()).hexdigest()
    if summary.get("source_sha256") not in (None, actual):
        raise SystemExit(f"unbalance audit {audit_name} read a different samples.jsonl than the one on disk")
    return summary


def subset_unbalance(artifacts: Path, out: Path, tag: str) -> str:
    name = _unbalance_name(tag)
    source = artifacts / name
    manifest = [json.loads(line) for line in (out / "unbalance_audit" / "unbalance_admission_manifest.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
    source_sha = hashlib.sha256((source / "samples.jsonl").read_bytes()).hexdigest()
    if any(entry.get("source_sha256") not in (None, source_sha) for entry in manifest):
        raise SystemExit(f"{name}: admission manifest digest differs from samples.jsonl on disk")
    admitted = {entry["id"] for entry in manifest if entry["admitted"]}
    lines = [line for line in (source / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, controls, windows = [], 0, 0
    for line in lines:
        row = json.loads(line)
        if row["scenario"] == "no_error":
            kept.append(line)
            controls += 1
        elif row["scenario"] == "three_phase_imbalance":
            windows += 1
            if row["id"] in admitted:
                kept.append(line)
    target = artifacts / f"{UNBALANCE['subset_stem']}_{len(admitted)}_{tag}"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    target.mkdir(parents=True)
    (target / "samples.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    repo = artifacts.parents[1]
    audit_dir = out / "unbalance_audit"
    meta["detectable_filter"] = {
        "schema": "discovered_mode_admission_filter_v1", "source_corpus": name, "source_samples_sha256": source_sha,
        "criterion": ADMISSION,
        "audit": audit_dir.relative_to(repo).as_posix() if audit_dir.is_relative_to(repo) else audit_dir.as_posix(),
        "kept_windows": len(admitted), "source_windows": windows, "controls_kept": controls,
        "note": "controls are kept unchanged; only disturbance windows that the operator WLS can discover are retained"}
    (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out / "detectable_unbalance_subset.json").write_text(json.dumps(target.name) + "\n", encoding="utf-8")
    print(json.dumps({"subset": target.name, "kept_windows": len(admitted), "source_windows": windows, "controls": controls}), flush=True)
    return target.name


def reaudit_unbalance(artifacts: Path, out: Path, tag: str) -> None:
    name = json.loads((out / "detectable_unbalance_subset.json").read_text(encoding="utf-8"))
    summary = audit_unbalance(artifacts, out, tag, corpus=name, audit_name="detectable_unbalance_audit")
    counts = summary["counts"]
    if counts.get("scenario_admitted") != counts.get("unbalance_rows"):
        raise SystemExit(f"{name}: re-audit admits {counts.get('scenario_admitted')} of {counts.get('unbalance_rows')} windows")
    print(json.dumps({"reaudit_unbalance": {name: counts.get("scenario_admitted")}}), flush=True)


UNBALANCE_STAGES = {"generate": generate_unbalance, "validate": validate_unbalance, "audit": audit_unbalance,
                    "subset": subset_unbalance, "reaudit": reaudit_unbalance}


PROVENANCE_FILES = (
    "three_phase_nlm/hif_operating_point.py", "three_phase_nlm/hif_parameter_estimator.py",
    "Transmission/generate_measurements_hif_ieee14.py", "IEEE_14_OpenDSS/IEEE14Gen.DSS", "IEEE_14_OpenDSS/IEEE14Lines.DSS",
    "IEEE_14_OpenDSS/IEEE14Loads.DSS", "IEEE_14_OpenDSS/Run_IEEE14Bus.dss", "scripts/audit_hif_physical_corpora.py",
    "scripts/validate_hif_samples.py", "scripts/validate_hif_multiscan_dataset.py",
    "scripts/validate_branch_current_localization.py", "scripts/regenerate_hif_physical_corpora.py",
    "Transmission/generate_measurements_imbalance.py", "scripts/audit_unbalance_physical_corpus.py",
    "IEEE_14_OpenDSS/IEEE14BusMaster.dss", "IEEE_14_OpenDSS/IEEE14Cap.DSS", "IEEE_14_OpenDSS/IEEE14Trafo.DSS",
)


def write_provenance(out: Path, args: argparse.Namespace) -> None:
    """Record the code that produced the corpora: commit, dirty files and digests of the physics sources."""
    def git(*command):
        try:
            return subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"
    import opendssdirect
    import platform
    manifest = {
        "recorded_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "repository_root": str(ROOT), "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain").splitlines(),
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "python": platform.python_version(), "opendssdirect": getattr(opendssdirect, "__version__", "unknown"),
        "file_sha256": {name: (hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).is_file() else None)
                        for name in PROVENANCE_FILES},
        "recipes": [{"stem": r[0], "hif_windows": r[1], "healthy_controls": r[2], "seed": r[3], "arguments": r[4]} for r in RECIPES],
    }
    path = out / "implementation_manifest.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    existing.append(manifest)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


STAGES = {"generate": generate, "validate": validate, "audit": audit, "subset": subset, "reaudit": reaudit}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="date suffix for the new corpus directories")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts" / "measurements")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--families", default="hif,unbalance", help="hif, unbalance, or both")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    artifacts, out = args.artifacts_dir.resolve(), args.output_dir.resolve()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [s for s in stages if s not in STAGES] + [f for f in families if f not in ("hif", "unbalance")]
    if unknown:
        parser.error(f"unknown stages or families {unknown}; stages {list(STAGES)}, families hif, unbalance")
    out.mkdir(parents=True, exist_ok=True)
    write_provenance(out, args)
    for stage in STAGES:
        if stage not in stages:
            continue
        if "hif" in families:
            if stage == "generate":
                generate(artifacts, out, args.tag, args.workers)
            elif stage == "validate":
                validate(artifacts, out, args.tag, args.workers * 3)
            else:
                STAGES[stage](artifacts, out, args.tag)
        if "unbalance" in families:
            UNBALANCE_STAGES[stage](artifacts, out, args.tag)


if __name__ == "__main__":
    main()
