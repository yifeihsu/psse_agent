#!/usr/bin/env python3
"""research-train on rows of the minimal interface.

    python research/minimal_env/train_sft.py [the options of python -m psse_env.sft research-train]

The stock trainer (``psse_env/sft/research_cli.py``) was written for the old interface:
it requires every row to declare ``metadata.protocol == "canonical"`` and replaces each
row's tool list with the old registry before its row gate.  A row of the minimal
interface carries its own six tools and its own protocol name, so this entry point calls
the same trainer with those two steps replaced:

* a row must declare ``metadata.protocol == "minimal_env_v1"``;
* a row keeps its own tool list, checked as the schema it is, in the form the run-time
  policy renders (``psse_env.minimal.observation.tools``).

Everything else (the tokenizer and mask gate, the optimizer, checkpoints, the reload
check) is research_cli's.  Nothing under ``psse_env/sft`` changes.

**The row gate.**  research-train renders and tokenizes every row each time the job
starts, which takes minutes on rows of about 10,000 tokens, and a pass on preemptible
nodes starts many times.  With ``MINIMAL_GATE_CACHE`` naming a file, the gate's result is
kept there, keyed by the row files, the gate's settings and the code that checks the
rows; a restart with the same key loads it.  ``row_gate.py gate`` fills it on a CPU node
before any GPU time, so a failing row also stops the pass before it starts.  (The same
arrangement as ``research/hpc/compound_cloning_20261106/train_runtime_tools.py``.)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.rows import MINIMAL_PROTOCOL  # noqa: E402
from psse_env.sft import research_cli  # noqa: E402
from psse_env.sft.gates import GateError, validate_tool_schemas  # noqa: E402
from psse_env.sft.gemma_text import sanitize_tool_schemas  # noqa: E402

NORMALIZATION_CONTRACT = "minimal_row_tools_kept_v1"
GATE_CACHE_ENV = "MINIMAL_GATE_CACHE"
GATE_CACHE_CONTRACT = "minimal_row_gate_cache_v1"
_STOCK_PREPARE = research_cli.prepare_research_examples


def _protocol(row: Mapping[str, Any]) -> Any:
    metadata = row.get("metadata")
    return metadata.get("protocol") if isinstance(metadata, Mapping) else None


def validate_minimal_splits(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The trainer's split check for minimal rows: both sides non-empty, the protocol named, no case on both sides."""
    if not train_rows or not validation_rows:
        raise GateError("training and validation splits must both be non-empty")
    roots: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for index, row in enumerate(rows):
            root = str(row.get("physical_root_fingerprint") or "").strip()
            if not root:
                raise GateError(f"{split}[{index}] lacks physical_root_fingerprint")
            roots[split].add(root)
            if _protocol(row) != MINIMAL_PROTOCOL:
                raise GateError(f"{split}[{index}] is not a row of {MINIMAL_PROTOCOL}")
    overlap = roots["train"] & roots["validation"]
    if overlap:
        raise GateError("train/validation physical roots overlap: " + ", ".join(sorted(overlap)[:8]))
    return {"train_rows": len(train_rows), "validation_rows": len(validation_rows), "train_roots": len(roots["train"]),
            "validation_roots": len(roots["validation"]), "overlap": []}


def keep_row_tools(rows: Sequence[Mapping[str, Any]], *, source_label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each row with its own tool list, validated and in the rendered form; the old registry is not put in."""
    if not rows:
        raise GateError(f"{source_label}: row set is empty")
    kept: list[dict[str, Any]] = []
    counts: Counter = Counter()
    changed = 0
    for index, row in enumerate(rows):
        copied = copy.deepcopy(dict(row))
        tools = validate_tool_schemas(copied.get("tools"), row_label=f"{source_label}[{index}]")
        rendered = sanitize_tool_schemas(tools)
        changed += int(rendered != copied.get("tools"))
        copied["tools"] = rendered
        counts[len(rendered)] += 1
        kept.append(copied)
    return kept, {"contract": NORMALIZATION_CONTRACT, "source_label": source_label, "rows": len(kept),
                  "source_registry_replaced": False, "rows_changed": changed,
                  "rows_by_tool_count": {str(key): value for key, value in sorted(counts.items())}}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate_cache_key(train_file: str | Path, validation_file: str | Path, settings: Any) -> str:
    """What the gate's result depends on: the row files, the gate's settings, the code that checks the rows."""
    sft = Path(research_cli.__file__).parent
    source = [sft / "gates.py", sft / "gemma_text.py", Path(__file__)]
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
    """research-train's row preparation, from the cache when its key matches."""
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
    """Put the minimal interface's checks and the cached row gate into the stock trainer's module."""
    research_cli.validate_research_splits = validate_minimal_splits
    research_cli.normalize_research_rows = keep_row_tools
    research_cli.prepare_research_examples = prepare_with_gate_cache


def main(argv: list[str] | None = None) -> int:
    install()
    return research_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
