"""Build a fresh balanced corpus for a registered system (stage 0 of a non-IEEE14 run).

The corpus is the physical source the round-0 generator draws from: solved AC-OPF
operating windows with clean, single-meter-error and parameter-error telemetry
under the fixed WLS covariance. Physical admission only; no WLS detectability or
teacher-success selection happens here (psse_env.providers.balanced_corpus).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.providers.balanced_corpus import build_balanced_corpus  # noqa: E402
from psse_env.systems import resolve_system  # noqa: E402

DEFAULT_COUNTS = {"no_error": 300, "measurement_error": 900, "parameter_error": 900}


def _counts(value: str | None) -> dict[str, int]:
    counts = dict(DEFAULT_COUNTS)
    if value:
        path = Path(value)
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(value)
        if not isinstance(loaded, dict):
            raise ValueError("--counts must be a JSON object of corpus scenario counts")
        counts = {str(key): int(count) for key, count in loaded.items()}
    unknown = sorted(set(counts) - set(DEFAULT_COUNTS))
    if unknown or any(count < 0 for count in counts.values()) or not any(counts.values()):
        raise ValueError(f"counts must be non-negative {sorted(DEFAULT_COUNTS)} with at least one positive: {counts}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="case57", help="Registered system (case14 or case57)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Corpus directory; must not exist yet")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--counts", default=None, help="JSON (inline or a file) of rows per corpus scenario")
    parser.add_argument("--num-scans", type=int, default=3, help="Independent-noise scans per parameter window")
    parser.add_argument("--load-scale-min", type=float, default=0.80)
    parser.add_argument("--load-scale-max", type=float, default=1.00)
    args = parser.parse_args(argv)
    spec = resolve_system(args.system)
    counts = _counts(args.counts)
    if args.num_scans < 1:
        raise ValueError("--num-scans must be positive")
    if not 0.0 < args.load_scale_min <= args.load_scale_max:
        raise ValueError("load scale range must be positive and ordered")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"corpus directory already exists; choose a fresh one: {output}")
    manifest = build_balanced_corpus(
        output, system=spec.case_id, seed=args.seed, counts=counts,
        load_scale_range=(args.load_scale_min, args.load_scale_max), num_scans=args.num_scans,
    )
    receipt = {
        "contract": "balanced_corpus_cli_v1",
        "system": spec.to_manifest(),
        "seed": args.seed,
        "requested_counts": counts,
        "num_scans": args.num_scans,
        "load_scale_range": [args.load_scale_min, args.load_scale_max],
        "corpus": manifest,
    }
    (output / "cli_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "corpus_path": str(manifest["corpus_path"]),
        "artifact_dir": str(manifest["artifact_dir"]),
        "counts": manifest.get("counts"),
        "complete": manifest.get("complete"),
    }, indent=2, default=str))
    return 0 if manifest.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
