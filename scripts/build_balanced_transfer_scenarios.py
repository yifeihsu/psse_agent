"""Generate fresh balanced development roots for the existing research pipeline.

Raw OPF population, generator admission, and subsequent expert evaluation are
separate artifacts. Default physical admission does not select teacher successes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.providers.balanced_corpus import build_balanced_corpus
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.systems import resolve_system

FAMILIES = ("no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_development_suite(
    output_dir: str | Path, *, system: str = "case57", seed: int = 20260910,
    per_family: int = 5, admission_mode: str = "physical", num_scans: int = 3,
    load_scale_range: tuple[float, float] = (0.80, 1.00),
) -> dict[str, Any]:
    if isinstance(per_family, bool) or int(per_family) != per_family or per_family < 1:
        raise ValueError("per_family must be a positive integer")
    if admission_mode not in {"physical", "recoverable"}:
        raise ValueError("admission_mode must be physical or recoverable")
    spec = resolve_system(system)
    output = Path(output_dir).resolve()
    # Reproducible generations must not overwrite prior numerical evidence.
    output.mkdir(parents=True, exist_ok=False)
    counts = {
        "no_error": max(20, 2 * per_family),
        "measurement_error": 4 * per_family,
        "parameter_error": 2 * per_family,
    }
    corpus = build_balanced_corpus(
        output / "corpus", system=spec.case_id, seed=seed, counts=counts,
        load_scale_range=load_scale_range, num_scans=num_scans,
    )
    generator = Round0ScenarioGenerator(
        system=spec.case_id, admission_mode=admission_mode,
        corpus_path=corpus["corpus_path"], balanced_artifact_dir=corpus["artifact_dir"],
        derived_case_dir=output / "derived_cases", seed=seed + 1,
        min_measurement_error_sigma=10.0,
    )
    plan = dict.fromkeys(FAMILIES, per_family)
    scenarios = generator.build(plan)
    envelopes = [partition_release_scenario_v1(row, split="development") for row in scenarios]
    scenario_path = output / "scenarios.json"
    _write_json(scenario_path, envelopes)
    _write_json(output / "generator_report.json", generator.report())
    _write_json(output / "generator_rows.json", scenarios)
    actual = Counter(row["grouping"]["scenario_family"] for row in envelopes)
    parents = {row["grouping"]["source_realization_id"] for row in envelopes}
    roots = {row["grouping"]["physical_root_fingerprint"] for row in envelopes}
    if len(roots) != len(envelopes):
        raise ValueError("duplicate physical roots in generated development suite")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    sources = (
        "scripts/build_balanced_transfer_scenarios.py", "psse_env/providers/balanced_corpus.py",
        "psse_env/providers/scenario_generator.py", "psse_env/systems/registry.py",
        "mcp_server/matpower_server.py", "psse_env/providers/matpower.py",
        "psse_env/dagger/suite_builder.py", "psse_env/dagger/evaluator.py",
        "psse_env/dagger/release_factories.py",
        "tools/lagrangian_port.py", "tools/correct_parameter_group_multi_scan_port.py",
        "Transmission/generate_measurements.py",
    )
    manifest = {
        "contract": "balanced_transfer_development_v1", "release_evidence": False,
        "system": spec.to_manifest(), "seed": seed, "admission_mode": admission_mode,
        "split": "development", "not_final_test": True,
        "requested_by_family": plan, "built_by_family": dict(actual),
        "counts_complete": all(actual[family] == count for family, count in plan.items()),
        "physical_root_count": len(roots), "parent_realization_count": len(parents),
        "scenario_path": str(scenario_path),
        "scenario_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        "source_corpus_manifest": corpus, "teacher_evaluation_performed": False,
        "learned_policy_evaluation_performed": False, "training_performed": False,
        "split_rule": "Keep source_realization_id together before splitting variants; this artifact is development only.",
        "git_head": git_head, "tracked_dirty_status": dirty,
        "source_sha256": {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in sources},
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system", choices=("case14", "case57"), default="case57")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--per-family", type=int, default=5)
    parser.add_argument("--num-scans", type=int, default=3)
    parser.add_argument("--load-scale-min", type=float, default=0.80)
    parser.add_argument("--load-scale-max", type=float, default=1.00)
    parser.add_argument("--admission-mode", choices=("physical", "recoverable"), default="physical")
    args = parser.parse_args(argv)
    manifest = build_development_suite(
        args.output_dir, system=args.system, seed=args.seed, per_family=args.per_family,
        admission_mode=args.admission_mode, num_scans=args.num_scans,
        load_scale_range=(args.load_scale_min, args.load_scale_max),
    )
    print(json.dumps({key: manifest[key] for key in ("scenario_path", "counts_complete", "built_by_family", "physical_root_count", "parent_realization_count")}, indent=2))
    return 0 if manifest["counts_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
