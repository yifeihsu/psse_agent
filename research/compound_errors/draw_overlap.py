"""How a fresh compound draw overlaps a reference draw (the out-of-sample check, 2026-10-29).

    python -m research.compound_errors.draw_overlap \\
        --pair ieee118=output/compound_errors_ieee118_20261029,output/compound_errors_ieee118_20261028 \\
        --pair ieee14_control=output/compound_errors_ieee14_busbranch_20261029,output/compound_errors_ieee14_busbranch_20261028 \\
        --output-dir output/compound_errors_ieee118_20261029/draw_overlap

Each folder is a ``build_suite.py`` output (``manifest.json``, ``atoms.jsonl``).
Per class: the generator seeds both draws use, the scenario ids both contain,
the roots identical in every atom and the load scale, and the roots whose
targets (true lines and meters) the reference already had, at other
magnitudes.  The fresh draw is disjoint when the first three are zero.  The
last is expected on a small network (the IEEE 14 control has 20 lines) and is
reported, not refused: a threshold read off the reference is tested on new
roots, not on new targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.compound_errors.budget_headroom import CLASSES, load_jsonl  # noqa: E402

CONTRACT = "compound_draw_overlap_v1"


def load_draw(folder: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folder = Path(folder)
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8")), load_jsonl(folder / "atoms.jsonl")


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, Mapping):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    return value


def root_key(row: Mapping[str, Any]) -> tuple:
    """The class, every atom (magnitudes to nine digits) and the load scale."""
    atoms = sorted(json.dumps(_rounded(atom), sort_keys=True) for atom in row["atoms"])
    return str(row["class"]), tuple(atoms), _rounded(float(row["load_scale"]))


def target_key(row: Mapping[str, Any]) -> tuple:
    """The class and the true targets, whatever their magnitudes."""
    return (str(row["class"]), tuple(sorted(int(i) for i in row.get("true_lines") or [])),
            tuple(sorted(int(i) for i in row.get("true_measurement_indices") or [])))


def overlap(fresh: tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]],
            reference: tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    (fresh_manifest, fresh_rows), (reference_manifest, reference_rows) = fresh, reference
    seeds = lambda manifest: {(str(c["class"]), int(c["seed"])) for c in manifest["class_results"]}  # noqa: E731
    shared_seeds = seeds(fresh_manifest) & seeds(reference_manifest)
    reference_ids = {str(row["scenario_id"]) for row in reference_rows}
    reference_roots = {root_key(row) for row in reference_rows}
    reference_targets = {target_key(row) for row in reference_rows}
    classes: dict[str, dict[str, int]] = {}
    for name in [c for c in CLASSES if any(row["class"] == c for row in fresh_rows)]:
        rows = [row for row in fresh_rows if row["class"] == name]
        classes[name] = {
            "n": len(rows),
            "shared_seeds": sum(1 for cls, _ in shared_seeds if cls == name),
            "shared_ids": sum(1 for row in rows if str(row["scenario_id"]) in reference_ids),
            "identical_roots": sum(1 for row in rows if root_key(row) in reference_roots),
            "same_targets": sum(1 for row in rows if target_key(row) in reference_targets),
        }
    total = Counter()
    for cell in classes.values():
        total.update(cell)
    return {
        "seeds": {"fresh": int(fresh_manifest["seed"]), "reference": int(reference_manifest["seed"])},
        "classes": classes,
        "all": dict(total),
        "disjoint": not any(total[k] for k in ("shared_seeds", "shared_ids", "identical_roots")),
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Fresh compound draw against the reference draw", ""]
    for system, entry in report.items():
        seeds = entry["seeds"]
        lines += [f"## {system}: suite seed {seeds['fresh']} against {seeds['reference']} "
                  f"({'disjoint' if entry['disjoint'] else 'NOT disjoint'})", "",
                  "| class | n | shared generator seeds | shared scenario ids | identical roots | same targets |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, cell in [*entry["classes"].items(), ("all", entry["all"])]:
            lines.append(f"| {name} | {cell['n']} | {cell['shared_seeds']} | {cell['shared_ids']} | {cell['identical_roots']} | "
                         f"{cell['same_targets']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", action="append", required=True, metavar="SYSTEM=FRESH_DIR,REFERENCE_DIR")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {}
    for item in args.pair:
        system, _, folders = item.partition("=")
        fresh, _, reference = folders.partition(",")
        report[system] = overlap(load_draw(Path(fresh)), load_draw(Path(reference)))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "draw_overlap.json").write_text(
        json.dumps({"contract": CONTRACT, "pairs": args.pair, "report": report}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = markdown(report)
    (out / "draw_overlap.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if all(entry["disjoint"] for entry in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
