"""Draw the compound-error suite (stage 1 of the compound-errors extension, 2026-10-27).

    python -m research.compound_errors.build_suite --output-dir output/compound_errors_20261027 --seed 20261027 [--workers 8]
    python -m research.compound_errors.build_suite --system case118 --output-dir output/compound_errors_ieee118_20261028 \\
        --seed 20261027 --shards 10 --workers 28

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

``--system`` selects a registered system (2026-10-28).  IEEE 14 keeps the
node/breaker physics and the fourteen classes; a system without a
node/breaker model (IEEE 118) draws the classes that need no breaker on the
bus-branch model, every atom admitted at an effect ratio of at least
``--effect-floor`` on its own (1.5, the low end of the effect-targeted band of
``research.classifier_triage``).  ``--shards`` splits each class over that many
generators (seeded from the suite seed, the class name and the shard), so a
slow system's large classes spread over the workers; one shard is the
original seeding.
"""
from __future__ import annotations

import argparse
import dataclasses
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
#: The bus-branch suite (IEEE 118, 2026-10-28): the classes that need no
#: breaker, at the IEEE 14 study's counts.
BUS_BRANCH_PLAN: dict[str, int] = {
    "rx2_adjacent": 150,
    "rx2_distant": 150,
    "meter_rx2": 150,
    "meter_on_rx": 75,
    "control_rx": 50,
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


def class_shards(seed: int, name: str, count: int, shards: int) -> list[tuple[int, int, int]]:
    """(shard, seed, count) of one class; one shard keeps the class seed of the original suites."""
    shards = max(1, min(int(shards), int(count)))
    if shards == 1:
        return [(0, class_seed(seed, name), int(count))]
    base, extra = divmod(int(count), shards)
    return [(k, class_seed(seed, f"{name}#{k}"), base + (1 if k < extra else 0)) for k in range(shards)]


def resolve_physics(system: str, physics: str) -> str:
    from psse_env.systems import resolve_system

    case_id = resolve_system(system).case_id
    if physics == "auto":
        return "node_breaker" if case_id == "case14" else "bus_branch"
    return physics


def floor_rejections(skipped: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Draws refused for an atom below the effect floor, and how many of them the physical admission alone keeps."""
    out: Counter = Counter()
    for entry in skipped:
        if entry.get("reason") != "compound_atom_below_effect_floor":
            continue
        metrics = entry.get("metrics") or {}
        out["total"] += 1
        out["joint_alarm"] += bool(metrics.get("joint_alarm"))
        out["physically_admissible"] += bool(metrics.get("joint_alarm") and metrics.get("true_model_clean"))
        kinds = metrics.get("kinds") or []
        for position in metrics.get("atoms_below") or []:
            out[f"below_{kinds[position] if position < len(kinds) else 'atom'}"] += 1
    return dict(out)


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
    """Worker: build one class (or one shard of it) with its own generator and seed."""
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator

    started = time.perf_counter()
    name, count, seed = str(payload["name"]), int(payload["count"]), int(payload["seed"])
    system = dict(payload.get("system") or {})
    generator = Round0ScenarioGenerator(
        seed=seed,
        derived_case_dir=str(payload["derived_case_dir"]),
        compound_solo_effects=bool(payload.get("solo_effects", True)),
        **system,
        **dict(payload.get("generator_options") or GENERATOR_OPTIONS),
    )
    rows = generator.build_compound({name: count})
    envelopes = [envelope_of(row) for row in rows]
    report = generator.report()
    keys = ["seed", "admission_mode", "evidence_profile", "chi2_alpha", "normalized_residual_threshold",
            "anomaly_detection_rule", "measurement_error_floor_sigma", "compound_solo_effects"]
    if system:
        keys += ["compound_physics", "compound_effect_floor"]
    result = {
        "class": name,
        "seed": seed,
        "requested": count,
        "built": len(envelopes),
        "rows": envelopes,
        "skipped_by_reason": dict(Counter(entry["reason"] for entry in generator.skipped)),
        "attempts": len(generator.skipped) + len(envelopes),
        "seconds": round(time.perf_counter() - started, 1),
        "generator": {key: report[key] for key in keys},
    }
    if system:
        result["shard"] = int(payload.get("shard", 0))
        result["system"] = report["system"]["case_id"]
        result["floor_rejections"] = floor_rejections(generator.skipped)
    return result


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
    parser.add_argument("--system", default="case14", help="registered system (case14, case57, case118)")
    parser.add_argument("--physics", default="auto", choices=("auto", "node_breaker", "bus_branch"),
                        help="auto: node/breaker on IEEE 14, bus-branch elsewhere")
    parser.add_argument("--effect-floor", default=None,
                        help="per-atom effect-ratio floor of a bus-branch suite (default 1.5; 'none' admits physically only)")
    parser.add_argument("--shards", type=int, default=1, help="generators per class (one keeps the original class seeds)")
    parser.add_argument("--distant-hops", type=int, default=None,
                        help="hops of a distant pair, replacing the classes' own (e.g. 2 for an IEEE 14 bus-branch control)")
    args = parser.parse_args(argv)

    from psse_env.providers.scenario_generator import BUS_BRANCH_COMPOUND_CLASSES, DEFAULT_COMPOUND_CLASSES
    from psse_env.systems import resolve_system
    from research.classifier_triage.effect_sampler import DEFAULT_BAND

    case_id = resolve_system(args.system).case_id
    physics = resolve_physics(case_id, args.physics)
    bus_branch = physics == "bus_branch"
    classes = BUS_BRANCH_COMPOUND_CLASSES if bus_branch else DEFAULT_COMPOUND_CLASSES
    if args.distant_hops is not None:
        classes = tuple(
            dataclasses.replace(spec, distant_hops=int(args.distant_hops)) if spec.parameter_placement == "distant" else spec
            for spec in classes
        )
    if args.effect_floor is None:
        effect_floor = float(DEFAULT_BAND[0]) if bus_branch else None
    else:
        effect_floor = None if str(args.effect_floor).strip().lower() == "none" else float(args.effect_floor)
    # IEEE 14 under its own physics passes nothing new to the generator.
    system_options = (
        {"system": case_id, "compound_physics": physics, "compound_effect_floor": effect_floor,
         "compound_only": case_id != "case14"}
        if (case_id != "case14" or bus_branch) else {}
    )
    if args.distant_hops is not None:
        system_options["compound_classes"] = [spec.as_dict() for spec in classes]
    default_plan = BUS_BRANCH_PLAN if bus_branch else DEFAULT_PLAN
    plan = dict(default_plan) if not args.plan else {str(k): int(v) for k, v in json.loads(args.plan).items()}
    if args.classes:
        plan = {name: plan[name] for name in args.classes}
    known = {spec.name for spec in classes}
    unknown = sorted(set(plan) - known)
    if unknown:
        raise SystemExit(f"unknown classes {unknown}; known: {sorted(known)}")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    derived = output / "derived_cases"
    derived.mkdir(parents=True, exist_ok=True)
    payloads = [
        {"name": name, "count": shard_count, "seed": shard_seed, "shard": shard, "derived_case_dir": str(derived),
         "solo_effects": not args.no_solo_effects, "generator_options": GENERATOR_OPTIONS, "system": system_options}
        for name, count in plan.items() if count > 0
        for shard, shard_seed, shard_count in class_shards(args.seed, name, count, args.shards)
    ]
    print(f"[suite] {case_id} ({physics}), {len(plan)} classes in {len(payloads)} generators, {sum(plan.values())} roots planned, "
          f"seed {args.seed}, effect floor {effect_floor}, workers {args.workers}", flush=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    progress_keys = ("class", "shard", "built", "requested", "attempts", "seconds", "skipped_by_reason")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(build_class, payloads):
                results.append(result)
                print(json.dumps({k: result[k] for k in progress_keys if k in result}), flush=True)
    else:
        for payload in payloads:
            result = build_class(payload)
            results.append(result)
            print(json.dumps({k: result[k] for k in progress_keys if k in result}), flush=True)
    order = {spec.name: index for index, spec in enumerate(classes)}
    results.sort(key=lambda item: (order[item["class"]], int(item.get("shard", 0))))
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
    roots_by_class: Counter = Counter()
    for result in results:
        roots_by_class[result["class"]] += int(result["built"])
    manifest = {
        "contract": SUITE_CONTRACT,
        "seed": int(args.seed),
        "plan": plan,
        "generator_options": GENERATOR_OPTIONS,
        "solo_effects": not args.no_solo_effects,
        "stratum_dominance_threshold": STRATUM_DOMINANCE_THRESHOLD,
        "classes": [spec.as_dict() for spec in classes if spec.name in plan],
        "class_results": [{k: v for k, v in result.items() if k != "rows"} for result in results],
        "roots": len(suite),
        "roots_by_class": dict(roots_by_class),
        "unique_physical_roots": len(set(fingerprints)),
        "suite": {"path": str(suite_path), "sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest()},
        "derived_case_dir": str(derived),
        "source_commit": source_commit(),
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    if system_options:
        floors: Counter = Counter()
        for result in results:
            floors.update(result.get("floor_rejections") or {})
        command = ["python -m research.compound_errors.build_suite", *(argv if argv is not None else sys.argv[1:])]
        manifest.update(system=case_id, physics=physics, effect_floor=effect_floor, shards=int(args.shards),
                        command=" ".join(str(part) for part in command), floor_rejections=dict(floors))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"roots": len(suite), "by_class": manifest["roots_by_class"], "seconds": manifest["total_seconds"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
