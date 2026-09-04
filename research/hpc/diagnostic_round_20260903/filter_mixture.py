"""Rebuild the round's 1:1 training mixture from a D0 pool without stale families.

The round-0 aggregate was collected before per-phase branch currents existed,
so every HIF and measurement+HIF episode in it was taught to end in an
operator handoff after both estimators were exhausted.  The diagnostic D1
rows teach the opposite for the same signature.  This tool drops those stale
D0 families *before* the mixture is sampled, so the exact 1:1 D0/D1 ratio is
preserved over the cleaned pool, and records exactly what was removed.

It reuses ``build_research_mixture`` from the research script unchanged, so
the only difference from the collection stage's mixture is the D0 pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

FILTER_CONTRACT = "research_diagnostic_round_mixture_filter_v1"


def row_family(row: Mapping[str, Any]) -> str:
    """Scenario family of a chat row, wherever the exporter placed it."""
    for container in (
        row,
        row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
        row.get("grouping") if isinstance(row.get("grouping"), Mapping) else {},
    ):
        family = str(container.get("scenario_family") or "").strip()
        if family:
            return family
    return ""


def filter_rows(
    rows: Iterable[Mapping[str, Any]], stale_families: Sequence[str]
) -> tuple[list[dict[str, Any]], Counter]:
    """Drop rows whose family is stale; fail closed on a row with no family."""
    stale = {str(item).strip() for item in stale_families if str(item).strip()}
    if not stale:
        raise ValueError("at least one stale family is required")
    kept: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for index, row in enumerate(rows):
        family = row_family(row)
        if not family:
            raise ValueError(f"D0 row {row.get('example_id') or index} carries no scenario family")
        if family in stale:
            dropped[family] += 1
            continue
        kept.append(dict(row))
    return kept, dropped


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    *,
    d0_path: Path,
    d1_path: Path,
    stale_families: Sequence[str],
    expected_dropped: int | None,
    d1_share: float,
    d1_cap: int | None,
    seed: int,
    output: Path,
    report_path: Path,
    mixture_builder: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if mixture_builder is None:
        from scripts.run_dagger_research import build_research_mixture

        mixture_builder = build_research_mixture
    d0_rows = load_jsonl(d0_path)
    d1_rows = load_jsonl(d1_path)
    kept, dropped = filter_rows(d0_rows, stale_families)
    dropped_total = sum(dropped.values())
    if expected_dropped is not None and dropped_total != int(expected_dropped):
        raise ValueError(
            f"expected to drop {expected_dropped} stale D0 rows, found {dropped_total}: {dict(dropped)}"
        )
    if not kept:
        raise ValueError("filtering removed every D0 row")
    mixture, mixture_report = mixture_builder(
        kept, d1_rows, d1_share=float(d1_share), d1_cap=d1_cap, seed=int(seed)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in mixture
    )
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(output)
    report = {
        "contract": FILTER_CONTRACT,
        "stale_families": sorted({str(item) for item in stale_families}),
        "d0_path": str(d0_path),
        "d0_sha256": _sha256(d0_path),
        "d1_path": str(d1_path),
        "d1_sha256": _sha256(d1_path),
        "d0_rows_before": len(d0_rows),
        "d0_rows_after": len(kept),
        "dropped_by_family": dict(sorted(dropped.items())),
        "dropped_total": dropped_total,
        "d0_families_after": dict(sorted(Counter(row_family(row) for row in kept).items())),
        "mixture_report": mixture_report,
        "mixture_rows": len(mixture),
        "mixture_sources": dict(
            sorted(Counter(str(row.get("research_mixture_source")) for row in mixture).items())
        ),
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0", required=True, type=Path, help="re-rendered D0 training view JSONL")
    parser.add_argument("--d1", required=True, type=Path, help="exported safe D1 chat rows JSONL")
    parser.add_argument("--stale-families", required=True, help="comma-separated families to drop from D0")
    parser.add_argument("--expected-dropped", type=int, help="fail unless exactly this many D0 rows drop")
    parser.add_argument("--d1-share", required=True, type=float)
    parser.add_argument("--d1-cap", type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build(
        d0_path=args.d0,
        d1_path=args.d1,
        stale_families=[item for item in args.stale_families.split(",") if item.strip()],
        expected_dropped=args.expected_dropped,
        d1_share=args.d1_share,
        d1_cap=args.d1_cap,
        seed=args.seed,
        output=args.output,
        report_path=args.report,
    )
    print(json.dumps({key: report[key] for key in ("dropped_by_family", "d0_rows_after", "mixture_rows", "mixture_sources", "output_sha256")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
