"""Join a rollout that ran in parts into one run (2026-10-29).

    python -m research.compound_errors.join_runs --suite output/compound_errors_ieee118_20261029/suite.json \\
        --part output/compound_errors_ieee118_20261029/meter_route_16/episodes_head.jsonl \\
        --part output/compound_errors_ieee118_20261029/meter_route_16_rest/episodes.jsonl \\
        --metadata output/compound_errors_ieee118_20261029/meter_route_16_rest/summary.json \\
        --output-dir output/compound_errors_ieee118_20261029/meter_route_16

A rollout stopped partway (a machine restart, or a single-worker start moved
to more workers) has written its first roots to ``episodes.jsonl``;
``rollout.py --roots`` runs the others.  The parts must cover the suite's
roots once each.  The joined ``episodes.jsonl`` is in suite order, and
``summary.{json,md}`` is rebuilt by ``rollout.summarize`` with the run
settings of ``--metadata`` (one part's own summary).  A part is never
overwritten: move the head aside (``episodes_head.jsonl``) before joining
into its folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.compound_errors.budget_headroom import load_jsonl  # noqa: E402
from research.compound_errors.rollout import load_suite, markdown, summarize  # noqa: E402

RUN_SETTINGS = ("contract", "suite", "system", "multi_scan_rcond", "profile", "expert", "progress_basis", "max_steps", "seed",
                "env_options", "expert_options")


def join(order: Sequence[str], parts: Sequence[Sequence[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
    """The parts' episodes in suite order; every suite root exactly once."""
    by_id: dict[str, Mapping[str, Any]] = {}
    for part in parts:
        for record in part:
            sid = str(record["scenario_id"])
            if sid in by_id:
                raise ValueError(f"root {sid} is in more than one part")
            by_id[sid] = record
    missing = [sid for sid in order if sid not in by_id]
    extra = sorted(set(by_id) - set(order))
    if missing or extra:
        raise ValueError(f"the parts do not cover the suite: {len(missing)} missing, {len(extra)} not in the suite")
    return [by_id[sid] for sid in order]


def joined_summary(records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], part_sizes: Sequence[int]) -> dict[str, Any]:
    summary = summarize(records)
    summary.update({key: metadata[key] for key in RUN_SETTINGS if key in metadata})
    summary.update(roots=len(records), parts=list(part_sizes), workers=None, total_seconds=None)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--part", action="append", required=True, type=Path, help="an episodes.jsonl of one part, in run order")
    parser.add_argument("--metadata", required=True, type=Path, help="a part's summary.json (the run settings)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    out = args.output_dir
    target = (out / "episodes.jsonl").resolve()
    if any(part.resolve() == target for part in args.part):
        raise SystemExit(f"{target} is a part; move it aside (episodes_head.jsonl) before joining into {out}")
    order = [str(root["execution"]["scenario_id"]) for root in load_suite(args.suite)]
    parts = [load_jsonl(part) for part in args.part]
    records = join(order, parts)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    summary = joined_summary(records, metadata, [len(part) for part in parts])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    title = f"Expert {summary.get('expert')} under {summary.get('profile')} on {len(records)} compound roots"
    (out / "summary.md").write_text(markdown(summary, title=title) + "\n", encoding="utf-8")
    print(json.dumps({"solved": summary["totals"]["success"], "roots": len(records), "parts": summary["parts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
