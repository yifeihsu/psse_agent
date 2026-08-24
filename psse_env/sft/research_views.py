"""Deterministic root-disjoint D0 views for bounded research GPU runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .gates import GateError, load_jsonl, validate_current_tool_registry


VIEW_CONTRACT = "research_d0_bounded_views_v1"


def _root(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(
        row.get("physical_root_fingerprint")
        or metadata.get("physical_root_fingerprint")
        or ""
    ).strip()


def _target_tool(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    target = messages[-1] if isinstance(messages, list) and messages else None
    calls = target.get("tool_calls") if isinstance(target, Mapping) else None
    call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
    function = call.get("function") if isinstance(call, Mapping) else None
    return str(function.get("name") or "").strip() if isinstance(function, Mapping) else ""


def _rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def _deduplicate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows):
        row = copy.deepcopy(dict(source))
        example_id = str(row.get("example_id") or "").strip()
        key = example_id or hashlib.sha256(
            json.dumps(row, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        if not _root(row):
            raise GateError(f"D0 row {example_id or index} lacks physical_root_fingerprint")
        if not _target_tool(row):
            raise GateError(f"D0 row {example_id or index} lacks one tool-call target")
        unique.append(row)
    return unique


def _select_diverse(
    rows: Sequence[Mapping[str, Any]], *, count: int, seed: int, namespace: str
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("view row count must be positive")
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tool[_target_tool(row)].append(copy.deepcopy(dict(row)))
    for tool, bucket in by_tool.items():
        bucket.sort(
            key=lambda row: (
                _rank(seed, f"{namespace}:{tool}", str(row.get("example_id") or "")),
                str(row.get("example_id") or ""),
            )
        )
    selected: list[dict[str, Any]] = []
    tool_names = sorted(by_tool)
    depth = 0
    while len(selected) < count:
        progressed = False
        for tool in tool_names:
            if depth < len(by_tool[tool]):
                selected.append(by_tool[tool][depth])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        depth += 1
    if len(selected) != count:
        raise GateError(
            f"{namespace}: selected {len(selected)}/{count} rows from the available roots"
        )
    return selected


def build_research_views(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260720,
    smoke_train_rows: int = 16,
    smoke_validation_rows: int = 8,
    mini_train_rows: int = 128,
    mini_validation_rows: int = 32,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build nested smoke/mini views from one stable root partition."""

    materialized = [copy.deepcopy(dict(row)) for row in rows]
    registry_failures = validate_current_tool_registry(materialized)
    if registry_failures:
        raise GateError("D0 source registry is stale: " + " | ".join(registry_failures))
    unique = _deduplicate(materialized)
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        by_root[_root(row)].append(row)
    roots = sorted(by_root, key=lambda value: (_rank(seed, "root", value), value))

    validation_roots: set[str] = set()
    validation_available = 0
    for root in roots:
        validation_roots.add(root)
        validation_available += len(by_root[root])
        if validation_available >= mini_validation_rows:
            break
    train_roots = set(roots) - validation_roots
    train_pool = [row for row in unique if _root(row) in train_roots]
    validation_pool = [row for row in unique if _root(row) in validation_roots]
    if len(train_pool) < mini_train_rows or len(validation_pool) < mini_validation_rows:
        raise GateError(
            "D0 root partition is too small for requested mini train/validation views"
        )

    mini_train = _select_diverse(
        train_pool, count=mini_train_rows, seed=seed, namespace="mini_train"
    )
    mini_validation = _select_diverse(
        validation_pool,
        count=mini_validation_rows,
        seed=seed,
        namespace="mini_validation",
    )
    smoke_train = _select_diverse(
        mini_train, count=smoke_train_rows, seed=seed, namespace="smoke_train"
    )
    smoke_validation = _select_diverse(
        mini_validation,
        count=smoke_validation_rows,
        seed=seed,
        namespace="smoke_validation",
    )
    views = {
        "smoke_train16": smoke_train,
        "smoke_validation8": smoke_validation,
        "mini_train128": mini_train,
        "mini_validation32": mini_validation,
    }
    report = {
        "contract": VIEW_CONTRACT,
        "seed": int(seed),
        "source_rows": len(materialized),
        "unique_example_ids": len(unique),
        "source_roots": len(by_root),
        "partition": {
            "train_roots": len(train_roots),
            "validation_roots": len(validation_roots),
            "overlap": sorted(train_roots & validation_roots),
        },
        "views": {
            name: {
                "rows": len(view),
                "roots": len({_root(row) for row in view}),
                "tool_counts": dict(sorted(Counter(_target_tool(row) for row in view).items())),
            }
            for name, view in views.items()
        },
    }
    return views, report


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build bounded root-disjoint D0 research views")
    result.add_argument("--d0", required=True, type=Path)
    result.add_argument(
        "--probe-source",
        required=True,
        type=Path,
        help="Canonical chat-SFT D1/recovery rows used only to complete ten fixed probes",
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--seed", type=int, default=20260720)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.d0.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    d0_rows = load_jsonl(source)
    views, report = build_research_views(d0_rows, seed=args.seed)
    probe_source = args.probe_source.expanduser().resolve(strict=True)
    from .research_smoke import select_probe_rows

    probe_rows = select_probe_rows(
        [
            *d0_rows,
            *load_jsonl(probe_source),
        ]
    )
    filenames = {
        "smoke_train16": "smoke.train16.jsonl",
        "smoke_validation8": "smoke.validation8.jsonl",
        "mini_train128": "mini.train128.jsonl",
        "mini_validation32": "mini.validation32.jsonl",
    }
    for name, filename in filenames.items():
        _write_jsonl(output / filename, views[name])
    _write_jsonl(output / "smoke.probes10.jsonl", probe_rows)
    report["source"] = str(source)
    report["probe_source"] = str(probe_source)
    report["outputs"] = {
        name: str((output / filename).resolve()) for name, filename in filenames.items()
    }
    report["outputs"]["smoke_probes10"] = str(
        (output / "smoke.probes10.jsonl").resolve()
    )
    report["probe_stages"] = [row["_research_probe_stage"] for row in probe_rows]
    _write_json(output / "view_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(parser().parse_args(argv))
    except Exception as exc:
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
