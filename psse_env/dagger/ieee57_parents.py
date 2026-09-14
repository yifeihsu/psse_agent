"""Preassigned operating parents for a small, unfiltered IEEE57 training pilot.

The parent is the canonical network plus its load-scaling construction, before
parameter changes, OPF, meter noise, or reported-error overlays. All realizations
from a reserved construction stay together, including unused support rows and
failed physical attempts. This is a pilot sampling design, not a claim that
scalar load scaling spans the operating population.
"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.providers.balanced_corpus import build_balanced_corpus
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.systems import resolve_system


FAMILIES = ("no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter")
PARENT_CONTRACT = "ieee57_preassigned_operating_parent_v1"


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_bytes(value))


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def parent_construction_fingerprint(base_case_hash: str, load_scale: float) -> str:
    """Identity deliberately excludes seed, slot name, split and fault labels."""
    if not str(base_case_hash).strip():
        raise ValueError("base_case_hash is required")
    if isinstance(load_scale, bool) or not math.isfinite(load_scale) or not 0.8 <= load_scale <= 1.0:
        raise ValueError("parent load scale must be finite and in [0.8, 1.0]")
    return _digest({"contract": PARENT_CONTRACT, "base_case_hash": base_case_hash, "load_scale": float(load_scale)})


def _relative(path: str | Path, output: Path) -> str:
    return Path(path).resolve().relative_to(output).as_posix()


def _portable(value: Any, output: Path) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _portable(item, output) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item, output) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(output).as_posix()
            except ValueError:
                pass
    return value


def _plan(output: Path, *, seed: int, per_family: int, splits: Sequence[str]) -> tuple[dict[str, Any], str]:
    spec = resolve_system("case57")
    rng = random.Random(seed)
    slots: list[dict[str, Any]] = []
    reserved: set[str] = set()
    for split in splits:
        for family in FAMILIES:
            for replicate in range(per_family):
                while True:
                    scale = rng.uniform(0.8, 1.0)
                    fingerprint = parent_construction_fingerprint(spec.base_case_hash, scale)
                    if fingerprint not in reserved:
                        break
                reserved.add(fingerprint)
                index = len(slots)
                counts = {"no_error": 1, "measurement_error": 0, "parameter_error": 0}
                if family in {"measurement", "multi_measurement"}:
                    counts["measurement_error"] = 2 if family == "multi_measurement" else 1
                elif family in {"parameter", "measurement+parameter"}:
                    counts["parameter_error"] = 1
                slots.append({
                    "slot_id": f"parent_{index:04d}", "split": split, "family": family,
                    "replicate": replicate, "load_scale": scale,
                    "base_case_hash": spec.base_case_hash,
                    "parent_construction_fingerprint": fingerprint,
                    "source_realization_id": f"ieee57_parent_{fingerprint}",
                    "physical_seed": rng.randrange(0, 2**63),
                    "descendant_seed": rng.randrange(0, 2**63),
                    "requested_raw_counts": counts,
                    "requested_scenario_count": 1,
                })
    plan = {
        "contract": PARENT_CONTRACT, "network_case": "case57", "base_case_hash": spec.base_case_hash,
        "seed": seed, "per_family": per_family, "splits": list(splits), "slots": slots,
        "assignment_stage": "before_parameter_changes_opf_noise_and_descendant_generation",
        "parent_identity_fields": ["contract", "base_case_hash", "load_scale"],
        "scope": "all raw attempts and counterfactuals in a slot share its reserved parent",
    }
    path = output / "parent_plan.json"
    _write(path, plan)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "parent_plan.sha256").write_text(digest + "\n", encoding="ascii")
    return plan, digest


def _locked_plan(output: Path, expected_hash: str) -> dict[str, Any]:
    payload = (output / "parent_plan.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("reserved parent plan changed after assignment")
    if (output / "parent_plan.sha256").read_text(encoding="ascii").strip() != expected_hash:
        raise ValueError("reserved parent-plan lock changed")
    return json.loads(payload)


def validate_parent_assignment(
    output_dir: str | Path, manifest: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check locked assignments and every persisted source/descendant link."""
    output = Path(output_dir).resolve()
    expected = str(manifest["parent_plan_sha256"])
    plan = _locked_plan(output, expected)
    slots: dict[str, Mapping[str, Any]] = {}
    parents: dict[str, str] = {}
    for slot in plan["slots"]:
        slot_id = slot["slot_id"]
        fingerprint = parent_construction_fingerprint(slot["base_case_hash"], slot["load_scale"])
        if slot_id in slots or fingerprint in parents:
            raise ValueError("duplicate reserved slot or physical parent construction")
        if slot["parent_construction_fingerprint"] != fingerprint or slot["source_realization_id"] != f"ieee57_parent_{fingerprint}":
            raise ValueError("parent construction identity does not match its physical inputs")
        if slot["base_case_hash"] != plan["base_case_hash"] or slot["split"] not in plan["splits"]:
            raise ValueError("slot base system or split disagrees with the locked plan")
        slots[slot_id] = slot
        parents[fingerprint] = slot["split"]

    source_path = output / str(manifest["source_population_path"])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest["source_population_sha256"]:
        raise ValueError("source population changed after generation")
    population = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line]
    source_owners: dict[str, str] = {}
    electrical_owners: dict[str, str] = {}
    raw_files: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in population:
        slot = slots[row["parent_slot_id"]]
        fingerprint = slot["parent_construction_fingerprint"]
        if any(row.get(key) != slot[value] for key, value in (
            ("dataset_split", "split"), ("source_realization_id", "source_realization_id"),
            ("parent_construction_fingerprint", "parent_construction_fingerprint"),
        )):
            raise ValueError("raw source was relabeled away from its reserved parent")
        source_id = row["original_source_realization_id"]
        relative_path = row["raw_source_path"]
        path = (output / relative_path).resolve()
        if not path.is_relative_to(output):
            raise ValueError("raw source path escaped the generated artifact directory")
        if relative_path not in raw_files:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            raw_files[relative_path] = {raw["source_realization_id"]: raw for raw in records}
            if len(raw_files[relative_path]) != len(records):
                raise ValueError("duplicate source ID within a raw corpus")
        original_record = raw_files[relative_path].get(source_id)
        if original_record is None or _digest(original_record) != row["raw_source_record_sha256"]:
            raise ValueError("original raw source content changed or is missing")
        if original_record.get("base_case_hash") != slot["base_case_hash"] or original_record.get("op_point", {}).get("load_scale") != slot["load_scale"]:
            raise ValueError("raw electrical source no longer matches its reserved construction")
        if row["noiseless_measurement_sha256"] != _digest({"base_case_hash": slot["base_case_hash"], "z_true": original_record["z_true"]}):
            raise ValueError("raw noiseless content fingerprint is inconsistent")
        if source_id in source_owners and source_owners[source_id] != fingerprint:
            raise ValueError("raw source identity crosses parent constructions")
        source_owners[source_id] = fingerprint
        electrical_hash = row["noiseless_measurement_sha256"]
        if electrical_hash in electrical_owners and electrical_owners[electrical_hash] != fingerprint:
            raise ValueError("same noiseless electrical realization crosses parent constructions")
        electrical_owners[electrical_hash] = fingerprint

    scenario_ids: set[str] = set()
    counts: Counter[str] = Counter()
    per_slot: Counter[str] = Counter()
    for envelope in scenarios:
        grouping = envelope["grouping"]
        slot = slots[grouping["parent_slot_id"]]
        expected_fields = {
            "split": slot["split"], "dataset_split": slot["split"],
            "source_realization_id": slot["source_realization_id"],
            "parent_construction_fingerprint": slot["parent_construction_fingerprint"],
            "physical_root_fingerprint": grouping.get("original_physical_root_fingerprint"),
            "scenario_family": slot["family"], "parent_plan_sha256": expected,
        }
        if any(grouping.get(key) != value for key, value in expected_fields.items()):
            raise ValueError("scenario split or parent lineage differs from the preassigned plan")
        if not grouping.get("physical_root_fingerprint"):
            raise ValueError("scenario exact physical-root fingerprint is missing")
        original = grouping.get("original_source_realization_id")
        if source_owners.get(original) != slot["parent_construction_fingerprint"]:
            raise ValueError("scenario original source is absent from its parent's raw population")
        scenario_id = envelope["execution"]["scenario_id"]
        if scenario_id in scenario_ids:
            raise ValueError("duplicate scenario ID")
        scenario_ids.add(scenario_id)
        per_slot[slot["slot_id"]] += 1
        if per_slot[slot["slot_id"]] > slot["requested_scenario_count"]:
            raise ValueError("more descendants than the locked slot requested")
        linked = [row for row in population if row["parent_slot_id"] == slot["slot_id"] and row["original_source_realization_id"] == original]
        if len(linked) != 1 or scenario_id not in linked[0]["selected_scenario_ids"]:
            raise ValueError("scenario is missing its persisted raw-source lineage link")
        measurements = envelope["execution"]["measurements"]
        if len(measurements) != 491 or any(isinstance(value, bool) or not math.isfinite(value) for value in measurements):
            raise ValueError("IEEE57 descendants require 491 finite measurement channels")
        counts[slot["split"]] += 1
    if _digest(list(scenarios)) != manifest["scenario_sha256"]:
        raise ValueError("scenario payload differs from the generated manifest")
    if hashlib.sha256((output / str(manifest["scenario_path"])).read_bytes()).hexdigest() != manifest["scenario_sha256"]:
        raise ValueError("persisted scenario payload changed after generation")
    return {
        "passed": True, "reserved_parent_count": len(slots), "raw_source_count": len(population),
        "scenario_count": len(scenarios), "scenario_count_by_split": dict(counts),
        "split_disjoint_by_parent_construction": True,
        "source_and_noiseless_content_cross_parent_overlap": 0,
    }


def generate_parent_assigned_scenarios(
    output_dir: str | Path, *, seed: int = 20260913, per_family: int = 1,
    splits: Sequence[str] = ("train", "validation", "test"),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate requested pilot roots with an immutable, prior split plan.

    Returned paths are relative to ``output_dir``. Canonical envelopes embed
    private reference cases and retain every requested physically admitted root,
    independent of WLS/teacher success. Raw support rows are retained separately;
    their count must not be reported as evaluated trajectories. Missing slots are
    recorded without choosing a replacement operating parent.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(per_family, bool) or not isinstance(per_family, int) or per_family < 1:
        raise ValueError("per_family must be a positive integer")
    if isinstance(splits, (str, bytes)):
        raise ValueError("splits must be a sequence")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(split not in {"train", "validation", "test", "development"} for split in splits):
        raise ValueError("splits must be distinct train/validation/test/development names")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan, plan_hash = _plan(output, seed=seed, per_family=per_family, splits=splits)
    scenarios: list[dict[str, Any]] = []
    population: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for slot in plan["slots"]:
        _locked_plan(output, plan_hash)
        directory = output / "parents" / slot["slot_id"]
        directory.mkdir(parents=True)
        report: dict[str, Any] = {**copy.deepcopy(slot), "status": "started", "scenario_ids": []}
        reports.append(report)
        _write(output / "slot_results.json", reports)
        try:
            corpus = build_balanced_corpus(
                directory / "corpus", system="case57", seed=slot["physical_seed"],
                counts=slot["requested_raw_counts"], num_scans=3,
                load_scale_range=(slot["load_scale"], slot["load_scale"]),
            )
            report["corpus_manifest"] = _portable(corpus, output)
            raw_path = Path(corpus["corpus_path"])
            raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
            report["raw_admitted_count"] = len(raw_rows)
            report["raw_physical_rejected_count"] = sum(corpus.get("rejected", {}).values())
            for raw in raw_rows:
                if raw.get("base_case_hash") != slot["base_case_hash"] or raw.get("op_point", {}).get("load_scale") != slot["load_scale"]:
                    raise ValueError("physical builder departed from the reserved parent construction")
                original = str(raw["source_realization_id"])
                population.append({
                    "parent_slot_id": slot["slot_id"], "dataset_split": slot["split"],
                    "source_realization_id": slot["source_realization_id"],
                    "parent_construction_fingerprint": slot["parent_construction_fingerprint"],
                    "original_source_realization_id": original,
                    "raw_family": raw["scenario"], "raw_source_path": _relative(raw_path, output),
                    "raw_source_record_sha256": _digest(raw),
                    "noiseless_measurement_sha256": _digest({"base_case_hash": slot["base_case_hash"], "z_true": raw["z_true"]}),
                    "selected_scenario_ids": [],
                })
            generator = Round0ScenarioGenerator(
                system="case57", admission_mode="physical", seed=slot["descendant_seed"],
                corpus_path=corpus["corpus_path"], balanced_artifact_dir=corpus["artifact_dir"],
                derived_case_dir=directory / "derived_cases", chi2_alpha=0.05,
                min_measurement_error_sigma=10.0,
            )
            generated = generator.build({slot["family"]: 1})
            _write(directory / "generator_report.json", generator.report())
            _write(directory / "generator_rows.json", generated)
            report["generator_report_path"] = _relative(directory / "generator_report.json", output)
            for scenario in generated:
                original = scenario["source_realization_id"]
                envelope = partition_release_scenario_v1(scenario, split=slot["split"])
                grouping = envelope["grouping"]
                grouping["original_physical_root_fingerprint"] = grouping["physical_root_fingerprint"]
                grouping.update({
                    "parent_slot_id": slot["slot_id"], "dataset_split": slot["split"],
                    "source_realization_id": slot["source_realization_id"],
                    "original_source_realization_id": original,
                    "parent_construction_fingerprint": slot["parent_construction_fingerprint"],
                    "parent_plan_sha256": plan_hash,
                })
                scenarios.append(envelope)
                scenario_id = envelope["execution"]["scenario_id"]
                report["scenario_ids"].append(scenario_id)
                for source in population:
                    if source["parent_slot_id"] == slot["slot_id"] and source["original_source_realization_id"] == original:
                        source["selected_scenario_ids"].append(scenario_id)
            report["status"] = "complete" if len(generated) == 1 else "no_scenario_admitted"
        except Exception as exc:
            report.update(status="generation_failed", error_type=type(exc).__name__, error=str(exc))
        _locked_plan(output, plan_hash)
        _write(output / "slot_results.json", reports)
        _jsonl(output / "source_population.jsonl", population)
        _write(output / "scenarios.json", scenarios)

    manifest = {
        "contract": PARENT_CONTRACT, "network_case": "case57", "measurement_count": 491,
        "parent_plan_path": "parent_plan.json", "parent_plan_sha256": plan_hash,
        "source_population_path": "source_population.jsonl",
        "source_population_sha256": hashlib.sha256((output / "source_population.jsonl").read_bytes()).hexdigest(),
        "scenario_path": "scenarios.json", "scenario_sha256": hashlib.sha256((output / "scenarios.json").read_bytes()).hexdigest(),
        "slot_results_path": "slot_results.json", "splits": list(splits),
        "requested_parent_count": len(plan["slots"]), "generated_scenario_count": len(scenarios),
        "complete": len(scenarios) == len(plan["slots"]) and all(row["status"] == "complete" for row in reports),
        "counts_by_split_family": {split: dict(Counter(row["grouping"]["scenario_family"] for row in scenarios if row["grouping"]["split"] == split)) for split in splits},
        "raw_admitted_source_count": len(population),
        "requested_raw_source_count": sum(sum(slot["requested_raw_counts"].values()) for slot in plan["slots"]),
        "raw_source_generation_complete": all(row.get("corpus_manifest", {}).get("complete") is True for row in reports),
        "raw_support_sources_without_scenario": sum(not row["selected_scenario_ids"] for row in population),
        "raw_physical_rejected_attempt_count": sum(row.get("raw_physical_rejected_count", 0) for row in reports),
        "failed_or_missing_parent_slots": [row["slot_id"] for row in reports if row["status"] != "complete"],
        "raw_source_population_unfiltered_retained": True,
        "requested_scenario_population_selected_on_teacher_outcomes": False,
        "supervision_filtering_performed": False, "evaluation_performed": False,
        "all_source_counts_interpretation": "Raw support controls and alternation rows are retained but are not additional evaluated trajectories.",
        "failure_policy": "Keep fixed parent assignment and failures; no replacement operating parent is drawn.",
    }
    manifest["independence_validation"] = validate_parent_assignment(output, manifest, scenarios)
    _write(output / "manifest.json", manifest)
    return manifest, scenarios


__all__ = ["FAMILIES", "PARENT_CONTRACT", "generate_parent_assigned_scenarios", "parent_construction_fingerprint", "validate_parent_assignment"]
