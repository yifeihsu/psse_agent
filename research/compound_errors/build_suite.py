"""Draw the compound-error suite (stage 1 of the compound-errors extension, 2026-10-27).

    python -m research.compound_errors.build_suite --output-dir output/compound_errors_20261027 --seed 20261027 [--workers 8]

One generator per class, seeded from the suite seed and the class name, builds
its roots in a worker process with the physical admission of the compound
synthesizer (``psse_env.providers.scenario_generator``: the reported model
alarms, the true model is clean, no teacher-solvability gate), under the
``scada_only`` profile with the 10-sigma meter floor of the conventional cell.
Every root is partitioned into the release envelope; the offline ``compound``
block (atoms, electrical relations, solo effects, deployed rankings) rides
under ``audit`` beside the parameter-ranking stratum.  The suite is a JSON
list of envelopes; the manifest records the seed, every class seed, the plan,
the generator reports and the source commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUITE_CONTRACT = "compound_errors_suite_v1"
#: The study plan: about 150 roots per class, the corridor and same-line pairs
#: split so the assessment's flagged pairs are measured on their own, two
#: breakers by effect pair, and 150 single-error controls over three families.
DEFAULT_PLAN: dict[str, int] = {
    "rx2_adjacent": 150,
    "rx2_distant": 150,
    "rx_cb_same_line": 75,
    "rx_cb_shared_bus": 75,
    "rx_cb_distant": 150,
    "meter_rx2": 150,
    "meter_on_rx": 75,
    "rx_cb_meter": 150,
    "cb2_dangling": 60,
    "cb2_mixed": 45,
    "cb2_split": 45,
    "control_rx": 50,
    "control_cb": 50,
    "control_meter": 50,
}
GENERATOR_OPTIONS: dict[str, Any] = {
    "evidence_profile": "scada_only",
    "admission_mode": "physical",
    "normalized_residual_threshold": 4.0,
    "min_measurement_error_sigma": 10.0,
    "validate": True,
}
#: The production dominance threshold stratifies the deployed parameter ranking.
STRATUM_DOMINANCE_THRESHOLD = 1.2


def class_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def parameter_stratum(ranking: Mapping[str, Any] | None, *, threshold: float = STRATUM_DOMINANCE_THRESHOLD) -> dict[str, Any]:
    """``dominant`` / ``ambiguous`` / ``misranked`` from the deployed ranking's view of the true lines."""
    if not isinstance(ranking, Mapping):
        return {"stratum": "not_applicable", "dominance_ratio": None, "true_line_rank": None, "true_line_ranks": None}
    rank = ranking.get("true_line_rank")
    ratio = ranking.get("parameter_ranking_dominance_ratio")
    try:
        ratio_value = float(ratio) if ratio is not None else None
    except (TypeError, ValueError):
        ratio_value = None
    singleton = ranking.get("parameter_ranking_singleton") is True
    if rank is None:
        stratum = "unranked"
    elif int(rank) != 1:
        stratum = "misranked"
    elif singleton or (ratio_value is not None and ratio_value >= threshold):
        stratum = "dominant"
    else:
        stratum = "ambiguous"
    return {
        "stratum": stratum,
        "dominance_ratio": ratio_value,
        "true_line_rank": rank,
        "true_line_ranks": dict(ranking.get("true_line_ranks") or {}),
        "threshold": float(threshold),
    }


def envelope_of(row: Mapping[str, Any]) -> dict[str, Any]:
    from psse_env.dagger.suite_builder import partition_release_scenario_v1

    envelope = partition_release_scenario_v1(row)
    compound = dict(row["compound"])
    envelope["audit"]["compound"] = compound
    envelope["audit"]["parameter_ranking"] = {
        **parameter_stratum(compound.get("parameter_ranking")),
        "generation": dict(row["parameter_ranking"]) if isinstance(row.get("parameter_ranking"), Mapping) else None,
    }
    # The grouping block is validated field by field by the evaluator; the
    # class lives under ``audit.compound`` (``hard_draw_report.root_stratum``).
    return envelope


def atoms_record(envelope: Mapping[str, Any]) -> dict[str, Any]:
    compound = envelope["audit"]["compound"]
    truth = envelope["audit"]["truth"]
    return {
        "scenario_id": envelope["execution"]["scenario_id"],
        "class": compound["class"],
        "atoms": compound["atoms"],
        "relations": compound["relations"],
        "meters_on_faulted_elements": compound["meters_on_faulted_elements"],
        "detection": compound["detection"],
        "solo_effects": compound["solo_effects"],
        "parameter_ranking": compound.get("parameter_ranking"),
        "breaker_ranking": compound.get("breaker_ranking"),
        "stratum": envelope["audit"]["parameter_ranking"]["stratum"],
        "error_cardinality": envelope["grouping"]["error_cardinality"],
        "true_measurement_indices": [int(item["index"]) for item in truth.get("true_measurement_errors") or []],
        "true_lines": [int(item["line_index1"]) for item in truth.get("true_parameter_errors") or []],
        "true_breakers": [str(item["cb_name"]) for item in truth.get("true_topology_errors") or []],
        "load_scale": compound["load_scale"],
    }


def build_class(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Worker: build one class with its own generator and seed."""
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator

    started = time.perf_counter()
    name, count, seed = str(payload["name"]), int(payload["count"]), int(payload["seed"])
    generator = Round0ScenarioGenerator(
        seed=seed,
        derived_case_dir=str(payload["derived_case_dir"]),
        compound_solo_effects=bool(payload.get("solo_effects", True)),
        **dict(payload.get("generator_options") or GENERATOR_OPTIONS),
    )
    rows = generator.build_compound({name: count})
    envelopes = [envelope_of(row) for row in rows]
    report = generator.report()
    return {
        "class": name,
        "seed": seed,
        "requested": count,
        "built": len(envelopes),
        "rows": envelopes,
        "skipped_by_reason": dict(Counter(entry["reason"] for entry in generator.skipped)),
        "attempts": len(generator.skipped) + len(envelopes),
        "seconds": round(time.perf_counter() - started, 1),
        "generator": {key: report[key] for key in (
            "seed", "admission_mode", "evidence_profile", "chi2_alpha", "normalized_residual_threshold",
            "anomaly_detection_rule", "measurement_error_floor_sigma", "compound_solo_effects")},
    }


def source_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--plan", default=None, help="JSON object class -> count (default: the study plan)")
    parser.add_argument("--classes", nargs="*", default=None, help="restrict the plan to these classes")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-solo-effects", action="store_true", help="skip the per-atom solo-effect probes")
    args = parser.parse_args(argv)

    from psse_env.providers.scenario_generator import DEFAULT_COMPOUND_CLASSES

    plan = dict(DEFAULT_PLAN) if not args.plan else {str(k): int(v) for k, v in json.loads(args.plan).items()}
    if args.classes:
        plan = {name: plan[name] for name in args.classes}
    known = {spec.name for spec in DEFAULT_COMPOUND_CLASSES}
    unknown = sorted(set(plan) - known)
    if unknown:
        raise SystemExit(f"unknown classes {unknown}; known: {sorted(known)}")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    derived = output / "derived_cases"
    derived.mkdir(parents=True, exist_ok=True)
    payloads = [
        {"name": name, "count": count, "seed": class_seed(args.seed, name), "derived_case_dir": str(derived),
         "solo_effects": not args.no_solo_effects, "generator_options": GENERATOR_OPTIONS}
        for name, count in plan.items() if count > 0
    ]
    print(f"[suite] {len(payloads)} classes, {sum(plan.values())} roots planned, seed {args.seed}, workers {args.workers}", flush=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(build_class, payloads):
                results.append(result)
                print(json.dumps({k: result[k] for k in ("class", "built", "requested", "attempts", "seconds", "skipped_by_reason")}), flush=True)
    else:
        for payload in payloads:
            result = build_class(payload)
            results.append(result)
            print(json.dumps({k: result[k] for k in ("class", "built", "requested", "attempts", "seconds", "skipped_by_reason")}), flush=True)
    order = {spec.name: index for index, spec in enumerate(DEFAULT_COMPOUND_CLASSES)}
    results.sort(key=lambda item: order[item["class"]])
    suite = [row for result in results for row in result["rows"]]
    ids = [row["execution"]["scenario_id"] for row in suite]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate scenario ids across classes")
    fingerprints = [row["grouping"]["physical_root_fingerprint"] for row in suite]
    suite_path = output / "suite.json"
    suite_path.write_text(json.dumps(suite, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "atoms.jsonl").open("w", encoding="utf-8") as stream:
        for row in suite:
            stream.write(json.dumps(atoms_record(row), sort_keys=True) + "\n")
    manifest = {
        "contract": SUITE_CONTRACT,
        "seed": int(args.seed),
        "plan": plan,
        "generator_options": GENERATOR_OPTIONS,
        "solo_effects": not args.no_solo_effects,
        "stratum_dominance_threshold": STRATUM_DOMINANCE_THRESHOLD,
        "classes": [spec.as_dict() for spec in DEFAULT_COMPOUND_CLASSES if spec.name in plan],
        "class_results": [{k: v for k, v in result.items() if k != "rows"} for result in results],
        "roots": len(suite),
        "roots_by_class": {result["class"]: result["built"] for result in results},
        "unique_physical_roots": len(set(fingerprints)),
        "suite": {"path": str(suite_path), "sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest()},
        "derived_case_dir": str(derived),
        "source_commit": source_commit(),
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"roots": len(suite), "by_class": manifest["roots_by_class"], "seconds": manifest["total_seconds"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
