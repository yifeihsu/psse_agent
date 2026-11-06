#!/usr/bin/env python3
"""research-train, rendering every row with the tool list of its run-time prompt.

    train_runtime_tools.py [the options of python -m psse_env.sft research-train]

The trainer replaces each row's tools with the whole sanitized registry before its row gate
(psse_env/sft/research_cli.py prepare_research_examples, psse_env/sft/research_rows.py).  That made
training match inference when inference rendered the whole registry.  Inference now renders
tool_schemas_for_observation(registry, observation) (research_policy_factory.py), so under scada_only
with the lifecycle choice a training prompt would carry 20 tools and the plain rollback description,
and the run-time prompt 13 tools and the description that says a candidate may be declined.

This entry point calls the same trainer with one change: after the trainer's normalization, each row
gets the list the policy would render for that row's own observation.  Everything else (row gate,
optimizer, checkpoints, reload) is research_cli's.  cell.env CLONE_TOOLS=registry runs the stock
command instead.  Nothing under psse_env changes; a fix there would make this file redundant.

The row gate.  research-train renders and tokenizes every row before anything else each time the job
starts.  Measured on the full view on 2026-11-06: 239 s, of which about 33 s is loading the processor
(an earlier projection of 27 minutes wrongly scaled that load with the rows).  With CLONE_GATE_CACHE
naming a file, the gate's result is kept there, keyed by the row files, the gate's settings and the
code that renders the rows; a restart with the same key loads it, 72 s measured, which leaves a
preemption slot about three minutes more of training.  stage_row_gate.sbatch fills it on a CPU node
before any GPU time, so a failing row also stops the pass before it starts.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.dataset_builder import tool_schemas_for_observation  # noqa: E402
from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas  # noqa: E402
from psse_env.sft import research_cli  # noqa: E402
from psse_env.sft.gates import GateError  # noqa: E402

_STOCK_NORMALIZATION = research_cli.normalize_research_rows
_STOCK_PREPARE = research_cli.prepare_research_examples
GATE_CACHE_ENV = "CLONE_GATE_CACHE"
GATE_CACHE_CONTRACT = "compound_cloning_row_gate_cache_v1"


def row_observation(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The observation a row's prompt shows: the state in its user turn (the policy's payload)."""
    for message in row.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == "user":
            state = json.loads(str(message.get("content"))).get("state")
            if isinstance(state, Mapping):
                return state
    raise GateError(f"{row.get('example_id')}: no state in the row's user turn")


def normalize_with_runtime_tools(rows: Sequence[Mapping[str, Any]], *, source_label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The trainer's normalization, then each row's run-time tool list."""
    normalized, report = _STOCK_NORMALIZATION(rows, source_label=source_label)
    registry = canonical_prompt_tool_schemas()
    counts: Counter = Counter()
    for row in normalized:
        row["tools"] = tool_schemas_for_observation(registry, row_observation(row))
        counts[len(row["tools"])] += 1
    report["rendered_tools"] = {"source": "tool_schemas_for_observation", "rows_by_tool_count": {str(k): v for k, v in sorted(counts.items())}}
    return normalized, report


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate_cache_key(train_file: str | Path, validation_file: str | Path, settings: Any) -> str:
    """What the gate's result depends on: the row files, the gate's settings, the code that renders the rows."""
    sft = Path(research_cli.__file__).parent
    source = [sft / "gates.py", sft / "research_rows.py", sft.parent / "dagger" / "dataset_builder.py", Path(__file__)]
    payload = {
        "contract": GATE_CACHE_CONTRACT,
        "rows": [_file_sha256(train_file), _file_sha256(validation_file)],
        "settings": [settings.model_name, settings.revision, settings.max_length, settings.allow_prompt_truncation],
        "source": [_file_sha256(path) for path in source],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _read_gate_cache(path: Path, key: str) -> Mapping[str, Any] | None:
    """The cached gate result for this key; a missing, stale or unreadable file is a miss, never an error."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            cached = pickle.load(stream)
    except Exception as exc:
        print(f"[row gate] cache {path} unreadable ({type(exc).__name__}); gating again", file=sys.stderr, flush=True)
        return None
    if not isinstance(cached, Mapping) or cached.get("key") != key:
        print(f"[row gate] cache {path} is for other rows or code; gating again", file=sys.stderr, flush=True)
        return None
    return cached


def _write_gate_cache(path: Path, payload: Mapping[str, Any]) -> None:
    """Written whole or not at all: a preemption during the write leaves the previous state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(dict(payload), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_with_gate_cache(*, train_file: str | Path, validation_file: str | Path, settings: Any) -> tuple[Any, list[Any], list[Any], dict[str, Any]]:
    """research-train's row preparation, from the cache when its key matches (module docstring, "The row gate")."""
    target = os.environ.get(GATE_CACHE_ENV)
    if not target:
        return _STOCK_PREPARE(train_file=train_file, validation_file=validation_file, settings=settings)
    path = Path(target)
    started = time.perf_counter()
    key = gate_cache_key(train_file, validation_file, settings)
    cached = _read_gate_cache(path, key)
    if cached is not None:
        settings.validate()
        processor, loader = research_cli.load_exact_processor(
            settings.model_name, settings.revision,
            local_files_only=settings.local_files_only, trust_remote_code=settings.trust_remote_code,
        )
        if loader != "AutoProcessor" or not callable(getattr(processor, "apply_chat_template", None)):
            raise GateError("Research Gemma training requires AutoProcessor so chat rendering matches inference")
        seconds = round(time.perf_counter() - started, 1)
        print(f"[row gate] cache hit: {len(cached['train'])} + {len(cached['validation'])} rows in {seconds} s", file=sys.stderr, flush=True)
        report = dict(cached["report"])
        report["row_gate_cache"] = {"hit": True, "key": key, "seconds": seconds}
        return processor, list(cached["train"]), list(cached["validation"]), report
    processor, train, validation, report = _STOCK_PREPARE(train_file=train_file, validation_file=validation_file, settings=settings)
    seconds = round(time.perf_counter() - started, 1)
    _write_gate_cache(path, {"contract": GATE_CACHE_CONTRACT, "key": key, "train": train, "validation": validation, "report": report})
    print(f"[row gate] gated {len(train)} + {len(validation)} rows in {seconds} s; kept in {path}", file=sys.stderr, flush=True)
    report = dict(report)
    report["row_gate_cache"] = {"hit": False, "key": key, "seconds": seconds}
    return processor, train, validation, report


def install() -> None:
    research_cli.normalize_research_rows = normalize_with_runtime_tools
    research_cli.prepare_research_examples = prepare_with_gate_cache


if __name__ == "__main__":
    install()
    raise SystemExit(research_cli.main(sys.argv[1:]))
