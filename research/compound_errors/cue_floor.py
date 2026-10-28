"""The open-terminal cue's residual floor on compound suites (compound-error study, stage 2 item 7 step 1).

For every root's first WLS solve and every line whose terminal flows read at noise (both P flow
readings below 3 sigma, the smaller Q flow reading below 3 sigma, the cue's unchanged first part),
the script records the larger normalized P residual, the line's modelled P flow (the measurement
minus the raw residual, in sigma units, from the same solve), the solve's largest normalized
residual, and the line's branch multiplier and rank.  It then scores residual floors: fixed ones,
floors that scale with the modelled flow, and the cued line's standing among the branch
candidates.  A firing on a root without a breaker fault is false; on IEEE 14 compound roots with a
dangling terminal the open line's firing is the recall.

Usage:
  python -m research.compound_errors.cue_floor --suite ieee14=output/compound_errors_20261027/suite.json \
      --suite ieee118=<path to the IEEE 118 compound suite.json> --output-dir output/compound_errors_20261027/cue_floor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FLOW_NOISE_SIGMA = 3.0


def line_rows(payload: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The at-noise lines of one root's first solve, with the quantities the floors read."""
    os.chdir(REPO)
    from mcp_server.matpower_server import _load_python_case, _wls_json
    from psse_env.providers.matpower import measurement_index_map
    from trace_protocol import build_lambda_evidence

    label, root = payload
    execution = root["execution"]
    metadata = execution.get("metadata") or {}
    z = [float(v) for v in execution["measurements"]]
    sigma = [float(v) for v in metadata.get("sigma_z") or []]
    out: dict[str, Any] = {"suite": label, "scenario_id": execution["scenario_id"],
                           "class": (root.get("audit", {}).get("compound") or {}).get("class"), "rows": [], "ok": False}
    options: dict[str, Any] = {"measurement_sigma": sigma}
    if metadata.get("structural_zero_indices"):
        options["exact_measurement_indices"] = list(metadata["structural_zero_indices"])
    try:
        solve = _wls_json(execution["case"], z, **options)
    except Exception as exc:  # recorded, not fatal
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if not solve.get("success"):
        out["error"] = "wls_failed"
        return out
    ppc = _load_python_case(execution["case"])
    nb, nl = int(ppc["bus"].shape[0]), int(ppc["branch"].shape[0])
    index_map = measurement_index_map(nb, nl)
    normalized = [float(v) for v in solve["r"]]
    raw = [float(v) for v in solve["raw_residual"]]
    largest = max(abs(v) for v in normalized)
    multiplier: dict[int, float] = {}
    for item in build_lambda_evidence([float(v) for v in solve.get("lambdaN") or []], solve.get("branch_info") or [],
                                      k=nl, min_abs=0.0):
        if item.get("line_row0") is not None:
            row0 = int(item["line_row0"])
            multiplier[row0] = max(multiplier.get(row0, 0.0), abs(float(item["value"])))
    ranking = sorted(multiplier, key=lambda row0: -multiplier[row0])
    top = multiplier[ranking[0]] if ranking else 0.0
    truth = root.get("audit", {}).get("truth") or {}
    dangling = {int(t["branch_row0"]) for t in truth.get("true_topology_errors") or [] if t.get("branch_row0") is not None}
    for row0 in range(nl):
        index = {ch: index_map[ch].start + row0 for ch in ("Pf", "Qf", "Pt", "Qt")}
        reading = {ch: abs(z[i]) / sigma[i] for ch, i in index.items()}
        if not (max(reading["Pf"], reading["Pt"]) < FLOW_NOISE_SIGMA and min(reading["Qf"], reading["Qt"]) < FLOW_NOISE_SIGMA):
            continue
        out["rows"].append({
            "line": row0 + 1,
            "p_resid": max(abs(normalized[index["Pf"]]), abs(normalized[index["Pt"]])),
            "model_p": max(abs(z[index[ch]] - raw[index[ch]]) / sigma[index[ch]] for ch in ("Pf", "Pt")),
            "resid_share": max(abs(normalized[index["Pf"]]), abs(normalized[index["Pt"]])) / largest if largest else 0.0,
            "multiplier_ratio": multiplier.get(row0, 0.0) / top if top else 0.0,
            "rank": ranking.index(row0) + 1 if row0 in ranking else None,
            "dangling": row0 in dangling,
        })
    out.update(ok=True, has_breaker=bool(truth.get("true_topology_errors")))
    return out


Rule = Callable[[Mapping[str, Any]], bool]


def rules() -> list[tuple[str, Rule]]:
    out: list[tuple[str, Rule]] = []
    for floor in (3.0, 5.0, 8.0, 10.0):
        out.append((f"fixed {floor:g}", lambda r, f=floor: r["p_resid"] > f))
    for floor in (6.0, 8.0, 10.0):
        for scale in (0.9, 1.0, 1.05, 1.1, 1.2):
            out.append((f"max({floor:g}, {scale:g} x modelled flow)",
                        lambda r, f=floor, k=scale: r["p_resid"] > max(f, k * r["model_p"])))
    for floor in (3.0, 8.0):
        for ratio in (0.5, 0.8):
            out.append((f"fixed {floor:g} and multiplier >= {ratio:g} x top",
                        lambda r, f=floor, q=ratio: r["p_resid"] > f and r["multiplier_ratio"] >= q))
    return out


def score(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table = []
    labels = sorted({r["suite"] for r in results})
    for name, rule in rules():
        row: dict[str, Any] = {"rule": name}
        for label in labels:
            roots = [r for r in results if r["suite"] == label and r["ok"]]
            clean = [r for r in roots if not r["has_breaker"]]
            row[f"{label}:false_fires"] = sum(1 for r in clean if any(rule(line) for line in r["rows"]))
            row[f"{label}:roots_without_breaker"] = len(clean)
            open_lines = [line for r in roots for line in r["rows"] if line["dangling"]]
            if open_lines:
                row[f"{label}:open_lines_caught"] = sum(1 for line in open_lines if rule(line))
                row[f"{label}:open_lines_at_noise"] = len(open_lines)
        table.append(row)
    return table


def markdown(table: Sequence[Mapping[str, Any]]) -> str:
    labels = sorted({key.split(":")[0] for row in table for key in row if ":" in key})
    head = ["rule"]
    for label in labels:
        head.append(f"{label}: false fires")
        if any(f"{label}:open_lines_caught" in row for row in table):
            head.append(f"{label}: open lines caught")
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for row in table:
        cells = [row["rule"]]
        for label in labels:
            cells.append(f"{row[f'{label}:false_fires']} of {row[f'{label}:roots_without_breaker']}")
            if f"{label}:open_lines_caught" in row:
                cells.append(f"{row[f'{label}:open_lines_caught']} of {row[f'{label}:open_lines_at_noise']}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args(argv)
    payloads = []
    for item in args.suite:
        label, _, path = item.partition("=")
        for root in json.loads(Path(path).read_text(encoding="utf-8")):
            payloads.append((label, root))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(line_rows, payloads, chunksize=4))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rows.json").write_text(json.dumps(results), encoding="utf-8")
    table = score(results)
    (args.output_dir / "table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (args.output_dir / "table.md").write_text(markdown(table) + "\n", encoding="utf-8")
    print(markdown(table))
    failed = [r for r in results if not r["ok"]]
    print(f"{len(results)} roots, {len(failed)} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
