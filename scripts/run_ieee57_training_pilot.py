"""Generate parent-assigned IEEE57 scenarios, audit targets, export and replay.

All requested physical roots remain in evaluation. Only audited TRAIN actions
enter the canonical SFT file. This validates collection mechanics, not a
qualified five-family corpus or trained model.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from threadpoolctl import threadpool_limits
from psse_env.dagger.ieee57_parents import generate_parent_assigned_scenarios, validate_parent_assignment
from psse_env.dagger.ieee57_runtime import ieee57_runtime_manifest
from psse_env.dagger.ieee57_training import collect_episode, export_audited_rows, replay_episode, supervision_counts, _model_view, _canonical
from scripts.validate_balanced_transfer import evaluate_scenarios

def source_hashes():
    # Freeze participating production code, not merely this orchestration file.
    paths = set(ROOT.glob("*.py"))
    for directory in ("psse_env", "tools", "Transmission", "mcp_server"):
        paths.update(p for p in (ROOT/directory).rglob("*.py")
                     if not p.name.startswith("test_") and "__pycache__" not in p.parts)
    paths.update([Path(__file__), ROOT/"scripts/validate_balanced_transfer.py", ROOT/"mcp_server/case57.m"])
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def evaluation_view(scenario):
    """Keep the parent-aware envelope intact; project legacy audit schema only."""
    result = copy.deepcopy(scenario)
    for key in ("dataset_split", "original_physical_root_fingerprint", "original_source_realization_id",
                "parent_construction_fingerprint", "parent_plan_sha256", "parent_slot_id"):
        result["grouping"].pop(key, None)
    return result


def assert_persisted(path, expected, *, jsonl=False):
    text = path.read_text(encoding="utf-8")
    loaded = [json.loads(line) for line in text.splitlines() if line] if jsonl else json.loads(text)
    native = json.loads(json.dumps(expected, sort_keys=True, allow_nan=False))
    if loaded != native:
        raise ValueError(f"Persisted artifact differs from gated evidence: {path.name}")


def assert_evaluation_matches(rows, evaluation):
    """Check collector and independent evaluator use the same visible policy."""
    by_step = {(row["scenario_id"], row["step"]): row for row in rows}
    seen = set()
    for episode in evaluation["evaluation"]["suite_metrics"]["episodes"]:
        for step in episode["trace"]:
            key = (episode["scenario_id"], step["step"])
            if key in seen or key not in by_step:
                raise ValueError("Evaluation trajectory coverage differs from collection")
            seen.add(key)
            raw = by_step[key]
            view, aliases = _model_view(step["policy_observation"])
            if raw["canonical_action"] is None:
                action_matches = step["action"] == raw["preferred_action"] and step["action"]["tool"] == "__invalid_action__"
            else:
                action_matches = _canonical(step["action"], aliases) == raw["canonical_action"]
            if view != raw["model_view"] or not action_matches:
                raise ValueError(f"Collection/evaluation visible decision differs: {key}")
    if seen != set(by_step):
        raise ValueError("Evaluation omitted collected decisions")
    return {"passed": True, "decisions_compared": len(seen)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False)+"\n" for row in rows), encoding="utf-8")


def build_pilot(output_dir, *, seed=20260913, per_family=1):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"contract": "ieee57_audited_balanced_training_pilot_v1",
                "started_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
                "runtime": ieee57_runtime_manifest(), "seed": seed,
                "model_training_performed": False, "qualified_five_family_training_corpus": False,
                "source_sha256": source_hashes()}
    write_json(output/"manifest.json", manifest)
    try:
        parents, scenarios = generate_parent_assigned_scenarios(
            output/"parents", seed=seed, per_family=per_family,
        )
        if not parents["complete"] or not parents["raw_source_generation_complete"]:
            raise ValueError("Parent population incomplete; evidence retained without release")
        validate_parent_assignment(output/"parents", parents, scenarios)
        if source_hashes() != manifest["source_sha256"]:
            raise ValueError("Source changed during pilot; retain attempt without releasing targets")
        all_rows, episode_rows = [], {}
        for scenario in scenarios:
            sid = scenario["execution"]["scenario_id"]
            rows = collect_episode(scenario)
            episode_rows[sid] = rows
            all_rows.extend(rows)
            write_jsonl(output/"raw_targets.jsonl", all_rows)
            print(f"Collected {sid}: {len(rows)} targets", flush=True)
        # Evaluate the complete physically admitted requested population. Neither
        # target quarantine nor export eligibility selects this population.
        evaluation = evaluate_scenarios([evaluation_view(s) for s in scenarios], seed=seed, max_steps=40)
        write_json(output/"evaluation.json", evaluation)
        if evaluation["infrastructure_errors"]:
            raise ValueError("Complete-population evaluation has infrastructure errors")
        if len(evaluation["episodes"]) != len(scenarios):
            raise ValueError("Evaluation silently omitted a requested root")
        evaluation_consistency = assert_evaluation_matches(all_rows, evaluation)
        exported, export_audit = export_audited_rows(all_rows)
        write_json(output/"export_audit.json", export_audit)
        replays = []
        for scenario in scenarios:
            sid = scenario["execution"]["scenario_id"]
            subset = [row for row in exported if row["scenario_id"] == sid]
            replays.append(replay_episode(scenario, episode_rows[sid], subset))
            write_json(output/"replay.json", replays)
            print(f"Replayed {sid}: {len(subset)} exported targets", flush=True)
        replay_count = sum(row["exported_targets_replayed"] for row in replays)
        if not exported or replay_count != len(exported):
            raise ValueError("No export, or not every exported target replayed")
        if {row["dataset_split"] for row in exported} != {"train"}:
            raise ValueError("Holdout target entered training export")
        parent_to_split = {s["grouping"]["parent_construction_fingerprint"]: s["grouping"]["split"] for s in scenarios}
        if any(parent_to_split[row["parent_construction_fingerprint"]] != "train" for row in exported):
            raise ValueError("Exported parent ownership changed")
        validate_parent_assignment(output/"parents", parents, scenarios)
        counts = supervision_counts(all_rows)
        write_json(output/"supervision_counts.json", counts)
        train_rows = [row for row in all_rows if row["dataset_split"] == "train"]
        if source_hashes() != manifest["source_sha256"]:
            raise ValueError("Source changed during pilot; retain evidence without publishing targets")
        pending = output/"train.canonical.pending.jsonl"
        write_jsonl(pending, exported)
        for name, expected, is_jsonl in (
            ("raw_targets.jsonl", all_rows, True), ("evaluation.json", evaluation, False),
            ("export_audit.json", export_audit, False), ("replay.json", replays, False),
            ("supervision_counts.json", counts, False), (pending.name, exported, True),
        ):
            assert_persisted(output/name, expected, jsonl=is_jsonl)
        if source_hashes() != manifest["source_sha256"]:
            raise ValueError("Source changed at publication boundary")
        # Atomic final-name publication after every gate, then manifest last.
        os.replace(pending, output/"train.canonical.jsonl")
        assert_persisted(output/"train.canonical.jsonl", exported, jsonl=True)
        manifest.update(
            complete=True, mechanics_release_passed=True,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            parent_count=len(parent_to_split), requested_scenarios=len(scenarios),
            raw_targets=len(all_rows), training_targets_considered=len(train_rows),
            exported_training_targets=len(exported), quarantined_training_targets=export_audit["quarantined"],
            all_requested_roots_evaluated=True, exported_targets_replayed=replay_count,
            collection_evaluation_consistency=evaluation_consistency,
            source_population=parents, family_outcomes=evaluation["family_summary"],
            supervision_counts=counts,
            accepted_training_tools=dict(Counter(row["preferred_action"]["tool"] for row in train_rows if row["production_label_eligible"])),
            split_scope="Fresh mechanics-pilot train/validation/test parents, not a sealed final benchmark",
            filtering_scope="Individual audited TRAIN targets; failed trajectories retained in raw/evaluation",
            replay_scope="All raw prefixes including quarantined actions; exported targets re-audited and executed with fresh aliases",
            artifacts={name: {"path": name, "sha256": hashlib.sha256((output/name).read_bytes()).hexdigest()}
                       for name in ("raw_targets.jsonl", "evaluation.json", "export_audit.json", "replay.json", "train.canonical.jsonl", "supervision_counts.json")},
        )
        if source_hashes() != manifest["source_sha256"]:
            raise ValueError("Source changed during pilot; do not qualify release")
        write_json(output/"manifest.json", manifest)
        return manifest
    except Exception as exc:
        published = output/"train.canonical.jsonl"
        if published.exists():
            os.replace(published, output/"train.canonical.quarantined.jsonl")
        manifest.update(complete=False, failure={"type": type(exc).__name__, "detail": str(exc)}, mechanics_release_passed=False)
        write_json(output/"manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--per-family", type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = build_pilot(args.output_dir, seed=args.seed, per_family=args.per_family)
    print(json.dumps({key: result[key] for key in ("complete", "parent_count", "raw_targets", "exported_training_targets", "quarantined_training_targets", "exported_targets_replayed")}, indent=2))


if __name__ == "__main__":
    main()
