"""Per-family summary of the diagnostic research round.

Reads the research script's paired-evaluation payloads and collection report
and writes one JSON summary: overall suite metrics for the BC0 baseline and
the round candidate, their difference, and per-family episode outcome counts
on the saved development roots.  It never touches hidden truth; every field
comes from the evaluator's own reports.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

#: Outcome-bearing fields of ``psse_env.dagger.evaluator.EpisodeEvaluation``.
#: The record carries more (traces, attestations); these are the ones a
#: per-family table needs.
OUTCOME_FIELDS = (
    "terminal",
    "terminal_outcome",
    "final_physical_correct",
    "final_physical_success",
    "truth_audited_task_success",
    "healthy_components_preserved",
    "invalid_action_count",
    "loop_detected",
    "specialized_tool_calls",
    "steps",
    "policy_steps",
    "evaluator_error",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_root(row: Mapping[str, Any]) -> str:
    # Scenario envelopes spell the root ``physical_root_fingerprint`` (top
    # level or under ``grouping``); evaluator episodes spell it ``physical_root``.
    for container in (
        row,
        row.get("grouping") if isinstance(row.get("grouping"), Mapping) else {},
        row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
        row.get("scenario") if isinstance(row.get("scenario"), Mapping) else {},
    ):
        for key in ("physical_root_fingerprint", "physical_root"):
            root = str(container.get(key) or "").strip()
            if root:
                return root
    return ""


def family_by_root(development: list[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in development:
        grouping = row.get("grouping") if isinstance(row.get("grouping"), Mapping) else {}
        family = str(grouping.get("scenario_family") or row.get("scenario_family") or "unknown")
        root = _row_root(row)
        if root:
            result[root] = family
    return result


def _episodes(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    episodes = payload.get("episodes")
    if isinstance(episodes, list):
        return [row for row in episodes if isinstance(row, Mapping)]
    suites = payload.get("suites")
    found: list[Mapping[str, Any]] = []
    if isinstance(suites, Mapping):
        for suite in suites.values():
            if isinstance(suite, Mapping) and isinstance(suite.get("episodes"), list):
                found.extend(row for row in suite["episodes"] if isinstance(row, Mapping))
    return found


def per_family_outcomes(
    payload: Mapping[str, Any], families: Mapping[str, str]
) -> dict[str, Any]:
    """Outcome field counts per scenario family for one adapter's evaluation."""
    tables: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    episodes_per_family: Counter = Counter()
    unmatched = 0
    for episode in _episodes(payload):
        root = _row_root(episode)
        family = families.get(root) or str(
            episode.get("family") or episode.get("scenario_family") or ""
        )
        if not family:
            unmatched += 1
            family = "unknown"
        episodes_per_family[family] += 1
        for field in OUTCOME_FIELDS:
            if field in episode:
                value = episode[field]
                key = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
                tables[family][field][key] += 1
    return {
        "episodes_per_family": dict(sorted(episodes_per_family.items())),
        "unmatched_episodes": unmatched,
        "outcomes": {
            family: {field: dict(sorted(counter.items())) for field, counter in sorted(fields.items())}
            for family, fields in sorted(tables.items())
        },
    }


def build_summary(
    *,
    collection_dir: Path,
    training_done: Path | None,
    prerequisites: Path | None,
    mixture_filter: Path | None = None,
) -> dict[str, Any]:
    report = _read_json(collection_dir / "research_run_report.json")
    comparison = _read_json(collection_dir / "evaluation" / "comparison.json")
    development = _read_json(collection_dir / "development_scenarios.json")
    families = family_by_root(development)
    per_adapter: dict[str, Any] = {}
    for label in ("bc0", "r1"):
        path = collection_dir / "evaluation" / f"{label}_eval.json"
        if path.is_file():
            per_adapter[label] = per_family_outcomes(_read_json(path), families)
    summary = {
        "contract": "research_diagnostic_round_summary_v1",
        "release_evidence": False,
        "research_profile": report.get("research_profile"),
        "collection_metrics": report.get("collection_metrics"),
        "mixture": report.get("mixture"),
        "development_families": dict(sorted(Counter(families.values()).items())),
        "paired_evaluation": {
            "paired_physical_roots": len(comparison.get("paired_physical_roots") or []),
            "bc0_overall": comparison.get("bc0_overall"),
            "r1_overall": comparison.get("r1_overall"),
            "r1_minus_bc0": comparison.get("r1_minus_bc0"),
        },
        "per_family": per_adapter,
    }
    if training_done is not None and training_done.is_file():
        summary["training"] = _read_json(training_done)
    if prerequisites is not None and prerequisites.is_file():
        summary["prerequisites"] = _read_json(prerequisites)
    if mixture_filter is not None and mixture_filter.is_file():
        summary["mixture_filter"] = _read_json(mixture_filter)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", required=True, type=Path)
    parser.add_argument("--training-done", type=Path)
    parser.add_argument("--prerequisites", type=Path)
    parser.add_argument("--mixture-filter", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = build_summary(
        collection_dir=args.collection_dir,
        training_done=args.training_done,
        prerequisites=args.prerequisites,
        mixture_filter=args.mixture_filter,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["paired_evaluation"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
