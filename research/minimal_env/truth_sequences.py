"""The truth-built repair of every training case that has one, as a list of actions.

    python -m research.minimal_env.truth_sequences \\
        --episodes output/minimal_env_20261115/demonstrations_train_ieee14/episodes.jsonl \\
        --output research/hpc/minimal_rl_20261116/train_truth_sequences.json

Reads the episode records ``research.minimal_env.build_demonstrations`` wrote and keeps, for every case whose
success path ended repaired (the object-level audit), that path's actions in order, its ``finish`` included.  The
recovery paths and the cases whose success path does not repair are left out.

The file says which training cases a truth-built sequence repairs, and how.  Two things read it:

* ``research.minimal_env.evaluate --cases`` evaluates an adapter on these cases alone;
* ``research.minimal_env.rl_train`` (``focus``) draws its cases from them and, for a drawn case none of whose
  samples is repaired, replays the sequence in the environment as the example its update trains on.

It is built from the truth and is training data: nothing of it reaches a prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SEQUENCES_CONTRACT = "minimal_truth_sequences_v1"


def build(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The file's payload from demonstration episode records."""
    cases: dict[str, dict[str, Any]] = {}
    counts: Counter = Counter()
    for record in records:
        if record.get("demonstration") != "success":
            continue
        counts["success_paths"] += 1
        if not record.get("solved"):
            counts["not_repaired"] += 1
            continue
        case = str(record["scenario_id"])
        if case in cases:
            raise ValueError(f"two success paths for {case}")
        actions = []
        for step in record["steps"]:
            if (step.get("result") or {}).get("status") != "ok":
                raise ValueError(f"the success path of {case} holds a call the environment refused (step {step.get('step')})")
            action = step["action"]
            actions.append([str(action["tool"]), dict(action.get("arguments") or {})])
        if not actions or actions[-1] != ["finish", {"outcome": "resolved"}]:
            raise ValueError(f"the success path of {case} does not end by declaring the case resolved")
        cases[case] = {"class": str(record.get("compound_class") or record.get("class") or ""), "actions": actions}
        counts["kept"] += 1
    by_class = Counter(entry["class"] for entry in cases.values())
    return {"contract": SEQUENCES_CONTRACT, "counts": dict(sorted(counts.items())), "by_class": dict(sorted(by_class.items())),
            "cases": dict(sorted(cases.items()))}


def dump(payload: Mapping[str, Any], path: Path) -> None:
    """One case per line, so a diff of the file reads case by case."""
    head = {key: value for key, value in payload.items() if key != "cases"}
    lines = ["{"]
    for key, value in head.items():
        lines.append(f" {json.dumps(key)}: {json.dumps(value, sort_keys=True)},")
    lines.append(' "cases": {')
    cases = list(payload["cases"].items())
    for number, (case, entry) in enumerate(cases):
        end = "," if number + 1 < len(cases) else ""
        lines.append(f"  {json.dumps(case)}: {json.dumps(entry, sort_keys=True, separators=(',', ':'))}{end}")
    lines += [" }", "}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """``{case: [{"tool", "arguments"}, ...]}`` of a file this module wrote."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("contract") != SEQUENCES_CONTRACT:
        raise ValueError(f"{path} is not a file of truth-built sequences ({SEQUENCES_CONTRACT})")
    sequences: dict[str, list[dict[str, Any]]] = {}
    for case, entry in (payload.get("cases") or {}).items():
        actions = [{"tool": str(tool), "arguments": dict(arguments)} for tool, arguments in entry["actions"]]
        if not actions or actions[-1]["tool"] != "finish":
            raise ValueError(f"the sequence of {case} in {path} does not end with finish")
        sequences[str(case)] = actions
    if not sequences:
        raise ValueError(f"{path} holds no sequence")
    return sequences


def case_ids(path: str | Path) -> set[str]:
    """The cases a file names: a file of this module, a JSON list of ids, or an object keyed by id."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload["cases"] if isinstance(payload.get("cases"), (Mapping, list)) else payload
    ids = {str(item) for item in payload}
    if not ids:
        raise ValueError(f"{path} names no case")
    return ids


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, type=Path, help="episodes.jsonl of build_demonstrations")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    with args.episodes.open("r", encoding="utf-8") as stream:
        payload = build(json.loads(line) for line in stream if line.strip())
    dump(payload, args.output)
    print(json.dumps({**payload["counts"], "by_class": payload["by_class"], "bytes": args.output.stat().st_size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
