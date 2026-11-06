#!/usr/bin/env python3
"""The inputs of the cell's smoke jobs, cut from the real files, and the row gate's timing.

    smoke_inputs.py rows --train VIEW --validation VIEW --out-dir DIR --train-rows 200 --validation-rows 32 --seed N
    smoke_inputs.py suite --suite SUITE.json --roots 8 --output SLICE.json
    smoke_inputs.py gate --train DIR/train.jsonl --validation DIR/validation.jsonl --full-train VIEW --tools runtime --model-choice 12b --output-dir DIR

``rows`` covers every system and action, declines apart, in a seeded order.  The longest row of the
whole view comes first: the optimizer smoke of research-train steps on the first row
(psse_env/sft/smoke.py), so the smoke meets the largest prompt the pass will train on.

``gate`` runs the tokenizer-based row gate alone on the smoke rows and prints its time per row and
what that projects for the full view.  research-train repeats the gate over all rows every time a
preempted job restarts, before its first step, so this is the restart cost that decides how much
of a preemption slot is left for training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def _row_group(row: Mapping[str, Any]) -> tuple[str, str]:
    selection = row.get("cloning_selection") if isinstance(row.get("cloning_selection"), Mapping) else {}
    label = row.get("cloning") if isinstance(row.get("cloning"), Mapping) else {}
    action = "decline" if selection.get("decline_row") else str(label.get("tool"))
    return str(selection.get("system")), action


def _row_length(row: Mapping[str, Any]) -> int:
    return sum(len(str(message.get("content") or "")) for message in row.get("messages") or [] if isinstance(message, Mapping))


def choose_rows(lines: Sequence[str], count: int, seed: int) -> list[int]:
    """Indexes of ``count`` rows: the longest first, then round robin over (system, action) in a seeded order."""
    rows = [json.loads(line) for line in lines]
    longest = max(range(len(rows)), key=lambda index: _row_length(rows[index]))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
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


def choose_roots(roots: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    """The first ``count`` roots round robin over the compound classes (the family where a root has none)."""
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

    if args.tools == "runtime":
        spec_file = Path(__file__).resolve().with_name("train_runtime_tools.py")
        loader = importlib.util.spec_from_file_location(spec_file.stem, spec_file)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        module.install()
    spec = get_research_model_spec(args.model_choice)
    settings = ResearchTrainerSettings(model_name=spec.model_id, revision=spec.revision, architecture=spec.architecture,
                                       output_dir=str(args.output_dir), max_length=args.max_length, allow_prompt_truncation=False)
    started = time.perf_counter()
    # Through the module: install() above may have put the cached gate there.
    _processor, train, validation, report = research_cli.prepare_research_examples(train_file=args.train, validation_file=args.validation, settings=settings)
    seconds = time.perf_counter() - started
    rows = len(train) + len(validation)
    with Path(args.full_train).open(encoding="utf-8") as stream:
        full = sum(1 for line in stream if line.strip())
    return {
        "rows_gated": rows, "gate_seconds": round(seconds, 1), "seconds_per_row": round(seconds / rows, 4),
        "full_view_rows": full, "projected_restart_gate_minutes": round(seconds / rows * full / 60, 1),
        "length_audit": {"train": report["train"]["length_audit"], "validation": report["validation"]["length_audit"]},
        "action_distribution": report["train"]["action_distribution"],
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
    gate.add_argument("--full-train", required=True, type=Path)
    gate.add_argument("--tools", choices=("runtime", "registry"), default="runtime")
    gate.add_argument("--model-choice", default="12b")
    gate.add_argument("--max-length", type=int, default=32768)
    gate.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "rows":
        result = {"train": write_rows(args.train, args.train_rows, args.seed, args.out_dir / "train.jsonl"),
                  "validation": write_rows(args.validation, args.validation_rows, args.seed, args.out_dir / "validation.jsonl")}
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
