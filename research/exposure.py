"""Count prepared rows and supervised target tokens without loading a model.

This is a CPU-side preflight for occupancy/exposure comparisons.  It uses the
same renderer, tokenizer and assistant-only mask as training, but loads only
the processor.  The resulting receipt is research evidence, not release
evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .model import load_processor
from .train import file_sha256, prepare_rows, read_jsonl, summarize_prepared_rows


def summarize_dataset(
    path: str | Path,
    processor: Any,
    *,
    max_length: int,
    label: str,
) -> dict[str, Any]:
    source = Path(path)
    rows = read_jsonl(source)
    prepared, failures = prepare_rows(
        rows,
        processor,
        max_length=max_length,
        label=label,
    )
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "input_rows": len(rows),
        "prepared": summarize_prepared_rows(prepared),
        "preparation_failure_count": len(failures),
        "preparation_failures": failures[:20],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    processor, resolved_id, resolved_revision = load_processor(
        model_id=args.model_id,
        revision=args.revision,
    )
    report: dict[str, Any] = {
        "model_id": resolved_id,
        "model_revision": resolved_revision,
        "max_length": args.max_length,
        "train": summarize_dataset(
            args.train,
            processor,
            max_length=args.max_length,
            label="train",
        ),
        "validation": None,
        "release_evidence": False,
    }
    if args.validation:
        report["validation"] = summarize_dataset(
            args.validation,
            processor,
            max_length=args.max_length,
            label="validation",
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
