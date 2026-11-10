"""What a job of the series ran on: the source commit, the config and the inputs, by digest (workplan of 2026-11-10, ticket 2).

    python -m research.hpc.rl_improvement_20261110.provenance --config $WORK/configs/arm_a_bounded.json

The job script calls this once before its first phase.  The record goes to
standard output and, as one more line, to ``<output_dir>/provenance.jsonl``:

* the commit of the source the job runs, and whether its tree has changes;
* the config file's digest;
* the digest of every input the config names (the suites, the reserved
  roots, the start adapter's weights) and, where the series planned on a
  digest (the config's ``series.planned_inputs``), whether it is the same.

A difference is reported, on standard error and in the record; the job goes
on (``--strict`` makes it the exit code).  This is a record of what ran, for
reading two arms as matched afterwards, not a gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVENANCE_CONTRACT = "rl_series_provenance_v1"
PROVENANCE_FILE = "provenance.jsonl"
#: The files of an adapter directory that are the adapter.
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def file_sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit(root: Path = REPO_ROOT) -> dict[str, Any]:
    """The commit of the source tree and whether tracked files differ from it; empty when it is not a git checkout."""

    def git(*arguments: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "tracked_changes": None}
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "tracked_changes": None if status is None else len([line for line in status.splitlines() if line.strip()])}


def record(config_path: Path, *, root: Path = REPO_ROOT, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The provenance record of one job on ``config_path``."""

    environment = os.environ if environment is None else environment
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    series = config.get("series") if isinstance(config.get("series"), Mapping) else {}
    planned = series.get("planned_inputs") if isinstance(series.get("planned_inputs"), Mapping) else {}
    inputs: dict[str, Any] = {}
    for name in ("training_suite", "development_suite", "reserved_roots", "expert_reference"):
        # The scorecard's expert record is named in the series block when the reward does not read one.
        path = config.get(name) or (series.get(name) if name == "expert_reference" else None)
        if not path:
            continue
        found = file_sha256(Path(path))
        expected = planned.get(f"{name}_sha256")
        inputs[name] = {"path": str(path), "sha256": found, "planned_sha256": expected,
                        "as_planned": None if expected is None else found == expected}
    adapter = Path(str(config.get("initial_adapter") or ""))
    inputs["initial_adapter"] = {"path": str(adapter), "files": {name: file_sha256(adapter / name) for name in ADAPTER_FILES}}
    differing = sorted(name for name, entry in inputs.items() if entry.get("as_planned") is False)
    missing = sorted(name for name, entry in inputs.items() if "sha256" in entry and entry["sha256"] is None)
    return {
        "contract": PROVENANCE_CONTRACT,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job": environment.get("SLURM_JOB_ID"),
        "restart": environment.get("SLURM_RESTART_COUNT"),
        "node": environment.get("SLURMD_NODENAME"),
        "source": source_commit(root),
        "config": {"path": str(config_path), "sha256": file_sha256(Path(config_path))},
        "series": {key: series.get(key) for key in ("name", "arm", "config_file")},
        "prompt_limit": {"max_input_tokens": config.get("max_input_tokens"), "max_length": config.get("max_length"),
                         "observation_mode": config.get("observation_mode")},
        "inputs": inputs,
        "inputs_not_as_planned": differing,
        "inputs_missing": missing,
    }


def append(entry: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PROVENANCE_FILE
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--strict", action="store_true", help="exit 3 when an input is missing or is not the one the series planned on")
    parser.add_argument("--no-write", action="store_true", help="print the record only")
    args = parser.parse_args(argv)
    entry = record(args.config)
    if not args.no_write:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        append(entry, Path(config["output_dir"]))
    print(json.dumps(entry, sort_keys=True))
    for name in entry["inputs_not_as_planned"]:
        print(f"[provenance] {name} is not the file the series planned on: {entry['inputs'][name]}", file=sys.stderr)
    for name in entry["inputs_missing"]:
        print(f"[provenance] {name} is missing: {entry['inputs'][name]['path']}", file=sys.stderr)
    return 3 if args.strict and (entry["inputs_not_as_planned"] or entry["inputs_missing"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
