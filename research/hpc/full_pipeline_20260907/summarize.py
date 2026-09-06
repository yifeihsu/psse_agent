#!/usr/bin/env python3
"""Per-family outcome tables for one DAgger round and for the whole pipeline.

A round summary reads the research run report and the paired evaluation of
its student (the adapter that collected) against its candidate (the adapter
it trained).  The pipeline summary joins the rounds: BC0 and R1 from round
1, R1 and R2 from round 2, all on the same development roots, with a check
that R1 scores identically in both rounds.  Nothing here touches hidden
truth; every field is read from evaluator receipts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

OUTCOME_FIELDS = (
    "terminal",
    "terminal_outcome",
    "final_physical_success",
    "truth_audited_task_success",
    "false_commit_count",
    "invalid_action_count",
    "loop_detected",
    "steps",
)
STUDENT_LABEL = {"r1": "bc0", "r2": "r1"}
CANDIDATE_LABEL = {"r1": "r1", "r2": "r2"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_root(row: Mapping[str, Any]) -> str:
    for container in (row, row.get("grouping") or {}, row.get("metadata") or {}):
        if isinstance(container, Mapping):
            root = str(container.get("physical_root_fingerprint") or "").strip()
            if root:
                return root
    return ""


def family_by_root(development: list[Mapping[str, Any]]) -> dict[str, str]:
    result = {}
    for row in development:
        grouping = row.get("grouping") if isinstance(row.get("grouping"), Mapping) else {}
        family = str(grouping.get("scenario_family") or row.get("scenario_family") or "")
        result[_row_root(row)] = family
    return result


def _episodes(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        episodes = payload.get("episodes")
        if isinstance(episodes, list):
            return [e for e in episodes if isinstance(e, Mapping)]
        for value in payload.values():
            found = _episodes(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _episodes(value)
            if found:
                return found
    return []


def _nested_flag(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _nested_flag(child, key)
            if found is not None:
                return found
    return None


def per_family_outcomes(payload: Any, families: Mapping[str, str]) -> dict[str, Any]:
    tables: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    counts: Counter = Counter()
    unmatched = 0
    for episode in _episodes(payload):
        root = str(episode.get("physical_root") or _row_root(episode) or "")
        family = families.get(root) or str(episode.get("family") or "")
        if not family:
            unmatched += 1
            continue
        counts[family] += 1
        for field in OUTCOME_FIELDS:
            tables[family][field][str(episode.get(field))] += 1
        matched = _nested_flag(episode.get("audit"), "diagnostic_truth_matched")
        if matched is not None:
            tables[family]["audit.diagnostic_truth_matched"][str(matched)] += 1
        success = _nested_flag(episode.get("audit"), "truth_audited_task_success")
        if success is not None:
            tables[family]["truth_audited_task_success"][str(bool(success))] += 1
    return {
        "episodes_per_family": dict(sorted(counts.items())),
        "outcomes": {
            family: {field: dict(sorted(values.items())) for field, values in sorted(table.items())}
            for family, table in sorted(tables.items())
        },
        "unmatched_episodes": unmatched,
    }


def success_table(outcomes: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """family -> {successes, episodes} from a per-family outcome block."""

    table = {}
    for family, count in outcomes["episodes_per_family"].items():
        successes = outcomes["outcomes"].get(family, {}).get("truth_audited_task_success", {})
        table[family] = {"successes": int(successes.get("True", 0)), "episodes": int(count)}
    return table


def round_summary(round_dir: Path, round_name: str) -> dict[str, Any]:
    collection = round_dir / "collection"
    report = _read_json(collection / "research_run_report.json")
    comparison = _read_json(collection / "evaluation" / "comparison.json")
    development = _read_json(collection / "development_scenarios.json")
    families = family_by_root(development)
    per_adapter: dict[str, Any] = {}
    for file_label, name in (("bc0", STUDENT_LABEL[round_name]), ("r1", CANDIDATE_LABEL[round_name])):
        path = collection / "evaluation" / f"{file_label}_eval.json"
        if path.is_file():
            per_adapter[name] = per_family_outcomes(_read_json(path), families)
    summary = {
        "contract": "research_full_pipeline_round_summary_v1",
        "round": round_name,
        "student": STUDENT_LABEL[round_name],
        "candidate": CANDIDATE_LABEL[round_name],
        "release_evidence": False,
        "research_profile": report.get("research_profile"),
        "collection_metrics": report.get("collection_metrics"),
        "mixture": report.get("mixture"),
        "development_families": dict(sorted(Counter(families.values()).items())),
        "paired_evaluation": {
            "paired_physical_roots": len(comparison.get("paired_physical_roots") or []),
            "student_overall": comparison.get("bc0_overall"),
            "candidate_overall": comparison.get("r1_overall"),
            "candidate_minus_student": comparison.get("r1_minus_bc0"),
        },
        "per_family": per_adapter,
        "success_by_family": {name: success_table(block) for name, block in per_adapter.items()},
    }
    training_done = round_dir / "training.done"
    if training_done.is_file():
        summary["training"] = _read_json(training_done)
    return summary


def pipeline_summary(out_dir: Path) -> dict[str, Any]:
    rounds = {}
    for name in ("r1", "r2"):
        path = out_dir / name / "round_summary.json"
        if path.is_file():
            rounds[name] = _read_json(path)
    adapters: dict[str, Any] = {}
    consistency: dict[str, Any] = {}
    for name, summary in rounds.items():
        for adapter, table in summary.get("success_by_family", {}).items():
            if adapter in adapters and adapters[adapter] != table:
                consistency[adapter] = {"first": adapters[adapter], "second": table}
            adapters.setdefault(adapter, table)
    overall = {
        adapter: {
            "successes": sum(v["successes"] for v in table.values()),
            "episodes": sum(v["episodes"] for v in table.values()),
        }
        for adapter, table in adapters.items()
    }
    for value in overall.values():
        value["rate"] = value["successes"] / value["episodes"] if value["episodes"] else None
    result = {
        "contract": "research_full_pipeline_summary_v1",
        "release_evidence": False,
        "rounds": {name: summary.get("paired_evaluation") for name, summary in rounds.items()},
        "success_by_family": adapters,
        "overall": overall,
        "adapter_consistency_across_rounds": consistency or "consistent",
    }
    for extra in ("d0.done", "suite.done", "bc0.done"):
        path = out_dir / extra
        if path.is_file():
            result[extra.replace(".done", "")] = _read_json(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("round")
    one.add_argument("--round-dir", required=True, type=Path)
    one.add_argument("--round", required=True, choices=("r1", "r2"))
    one.add_argument("--output", required=True, type=Path)
    whole = sub.add_parser("pipeline")
    whole.add_argument("--out-dir", required=True, type=Path)
    whole.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "round":
        summary = round_summary(args.round_dir, args.round)
        shown = summary["paired_evaluation"]
    else:
        summary = pipeline_summary(args.out_dir)
        shown = summary["overall"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(shown, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
