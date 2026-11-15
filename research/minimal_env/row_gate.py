#!/usr/bin/env python3
"""The tokenizer row gate on rows of the minimal interface, alone, and the inputs of a smoke run.

    row_gate.py gate --train T.jsonl --validation V.jsonl --model-choice 12b --max-length 32768 --output-dir DIR
    row_gate.py rows --train T.jsonl --validation V.jsonl --out-dir DIR --train-rows 200 --validation-rows 32 --seed N
    row_gate.py suite --suite SUITE.json --roots 8 --output SLICE.json

``gate`` runs what research-train runs before it trains (``train_sft.install`` and then the
trainer's own row preparation): every row is rendered with its own tool list by the
model's chat template, tokenized, and its tool call read back from the rendering; no
prompt may be cut.  It prints the time, the token lengths and the action counts.  With
``MINIMAL_GATE_CACHE`` set the result is kept for the training job's restarts.  The local
checks made when the rows were written cover the schemas only: this is the first time the
rows meet the tokenizer.

``rows`` cuts a small training and validation file for a smoke run: the longest row of the
view first (the trainer's optimizer smoke steps on the first row, so the smoke meets the
largest prompt), then the step kinds and actions in turn, in a seeded order.

``suite`` cuts the first cases of a suite, the classes in turn.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def _row_group(row: Mapping[str, Any]) -> tuple[str, str, str]:
    labels = row.get("minimal") if isinstance(row.get("minimal"), Mapping) else {}
    return str(labels.get("demonstration")), str(labels.get("kind")), str(labels.get("tool"))


def _row_length(row: Mapping[str, Any]) -> int:
    return sum(len(str(message.get("content") or "")) for message in row.get("messages") or [] if isinstance(message, Mapping))


def choose_rows(lines: Sequence[str], count: int, seed: int) -> list[int]:
    """Indexes of ``count`` rows: the longest first, then the (demonstration, kind, action) groups in turn, seeded."""
    rows = [json.loads(line) for line in lines]
    longest = max(range(len(rows)), key=lambda index: _row_length(rows[index]))
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if index != longest:
            groups[_row_group(row)].append(index)
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    chosen = [longest]
    while len(chosen) < min(count, len(rows)):
        for key in sorted(groups):
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    return chosen


def write_rows(view: Path, count: int, seed: int, output: Path) -> dict[str, Any]:
    with view.open(encoding="utf-8") as stream:
        lines = [line.rstrip("\n") for line in stream if line.strip()]
    chosen = choose_rows(lines, count, seed)
    _atomic_write(output, "".join(lines[index] + "\n" for index in chosen))
    groups: dict[str, int] = defaultdict(int)
    for index in chosen:
        groups[" ".join(_row_group(json.loads(lines[index])))] += 1
    return {"source": str(view), "rows": len(chosen), "of": len(lines), "groups": dict(sorted(groups.items()))}


def disjoint_smoke_rows(train: Path, validation: Path, out_dir: Path, *, train_rows: int, validation_rows: int, seed: int) -> dict[str, Any]:
    """The smoke's two files.  The views hold different cases already, so the cuts share none either."""
    return {"train": write_rows(train, train_rows, seed, out_dir / "train.jsonl"),
            "validation": write_rows(validation, validation_rows, seed, out_dir / "validation.jsonl")}


def choose_roots(roots: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    """The first ``count`` cases, the compound classes in turn (the family where a case has no class)."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for root in roots:
        compound = (root.get("audit") or {}).get("compound") or {}
        groups[str(compound.get("class") or (root.get("grouping") or {}).get("scenario_family"))].append(root)
    chosen: list[Mapping[str, Any]] = []
    position = 0
    while len(chosen) < min(count, len(roots)):
        for key in sorted(groups):
            if position < len(groups[key]) and len(chosen) < count:
                chosen.append(groups[key][position])
        position += 1
    return chosen


def gate_timing(args: argparse.Namespace) -> dict[str, Any]:
    from psse_env.research_models import get_research_model_spec
    from psse_env.sft import research_cli
    from psse_env.sft.research_cli import ResearchTrainerSettings
    from research.minimal_env import train_sft

    train_sft.install()
    spec = get_research_model_spec(args.model_choice)
    settings = ResearchTrainerSettings(model_name=spec.model_id, revision=spec.revision, architecture=spec.architecture,
                                       output_dir=str(args.output_dir), max_length=args.max_length, allow_prompt_truncation=False)
    started = time.perf_counter()
    # Through the module: install() put the minimal checks and the cached gate there.
    _processor, train, validation, report = research_cli.prepare_research_examples(
        train_file=args.train, validation_file=args.validation, settings=settings)
    seconds = time.perf_counter() - started
    rows = len(train) + len(validation)
    return {
        "rows_gated": rows, "train_rows": len(train), "validation_rows": len(validation),
        "gate_seconds": round(seconds, 1), "seconds_per_row": round(seconds / rows, 4),
        "max_length": int(args.max_length),
        "length_audit": {"train": report["train"]["length_audit"], "validation": report["validation"]["length_audit"]},
        "action_distribution": report["train"]["action_distribution"],
        "row_normalization": report.get("row_normalization"),
        "row_gate_cache": report.get("row_gate_cache"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    rows = commands.add_parser("rows")
    rows.add_argument("--train", required=True, type=Path)
    rows.add_argument("--validation", required=True, type=Path)
    rows.add_argument("--out-dir", required=True, type=Path)
    rows.add_argument("--train-rows", type=int, default=200)
    rows.add_argument("--validation-rows", type=int, default=32)
    rows.add_argument("--seed", type=int, default=0)
    suite = commands.add_parser("suite")
    suite.add_argument("--suite", required=True, type=Path)
    suite.add_argument("--roots", type=int, default=8)
    suite.add_argument("--output", required=True, type=Path)
    gate = commands.add_parser("gate")
    gate.add_argument("--train", required=True, type=Path)
    gate.add_argument("--validation", required=True, type=Path)
    gate.add_argument("--model-choice", default="12b")
    gate.add_argument("--max-length", type=int, default=32768)
    gate.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "rows":
        result = disjoint_smoke_rows(args.train, args.validation, args.out_dir, train_rows=args.train_rows,
                                     validation_rows=args.validation_rows, seed=args.seed)
    elif args.command == "suite":
        roots = json.loads(args.suite.read_text(encoding="utf-8"))
        chosen = choose_roots(roots, args.roots)
        _atomic_write(args.output, json.dumps(chosen, sort_keys=True) + "\n")
        result = {"source": str(args.suite), "roots": len(chosen), "of": len(roots)}
    else:
        result = gate_timing(args)
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
