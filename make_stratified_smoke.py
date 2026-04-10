#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a stratified smoke-test JSONL from SFT traces")
    p.add_argument(
        "--input",
        default="out_traces_balanced/sft_traces.test.jsonl",
        help="Source JSONL with conversation traces",
    )
    p.add_argument(
        "--output",
        default="outputs/stratified_smoke.jsonl",
        help="Output JSONL containing the sampled subset",
    )
    p.add_argument(
        "--per-family",
        type=int,
        default=4,
        help="How many samples to draw per error family",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic sampling",
    )
    p.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle the final sampled rows after per-family sampling",
    )
    p.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow families with fewer than --per-family rows instead of failing",
    )
    return p.parse_args()


def extract_gt_family(sample: dict[str, Any]) -> str | None:
    for msg in reversed(sample.get("messages", [])):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not content:
            continue
        try:
            obj = json.loads(content)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("verdict"), dict):
            fam = obj["verdict"].get("error_family")
            if isinstance(fam, list):
                fam = fam[0] if fam else None
            return fam
    return None


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            fam = extract_gt_family(row)
            if fam is None:
                continue
            families[fam].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    picked_counts: Counter[str] = Counter()

    for fam in sorted(families.keys()):
        pool = families[fam]
        if len(pool) < args.per_family and not args.allow_short:
            raise SystemExit(
                f"Family {fam!r} only has {len(pool)} rows, fewer than requested {args.per_family}."
            )
        k = min(args.per_family, len(pool))
        chosen = rng.sample(pool, k)
        selected.extend(chosen)
        picked_counts[fam] += k

    if args.shuffle_output:
        rng.shuffle(selected)

    with dst.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_counts = {fam: len(rows) for fam, rows in sorted(families.items())}

    print("Built stratified smoke set")
    print(f"  source: {src}")
    print(f"  total source rows: {total}")
    print(f"  families in source: {source_counts}")
    print(f"  per-family requested: {args.per_family}")
    print(f"  picked counts: {dict(sorted(picked_counts.items()))}")
    print(f"  output rows: {len(selected)}")
    print(f"  output: {dst}")


if __name__ == "__main__":
    main()
