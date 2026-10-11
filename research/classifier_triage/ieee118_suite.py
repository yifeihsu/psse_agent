"""A development suite of IEEE 118 roots of every non-topology family, for the agentic transfer test.

    python -m research.classifier_triage.ieee118_suite --balanced-corpus output/.../balanced/shard_00/corpus \\
        --waveform-corpus output/classifier_triage_20261011/ieee118_pipeline/pipeline_corpus \\
        --output-dir output/classifier_triage_20261011/suite_ieee118

The round-0 generator on IEEE 118 (``psse_env.providers.scenario_generator``,
2026-10-11: waveform families from the system's own corpora), with the
pipeline's development-suite settings (classifier-gated profile, the
four-sigma residual test, ten-sigma meters, parameter ranking at 1.0 with rank
allowance 2, three HIF scans), partitioned into the release envelopes the
research runner and ``expert_e2e`` consume.  The admission rule is the
pipeline's own, so the suite is built the way the IEEE 14 suites were.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.suite_builder import partition_release_scenario_v1  # noqa: E402
from psse_env.evidence_profile import CLASSIFIER_GATED_PROFILE  # noqa: E402
from psse_env.providers.scenario_generator import Round0ScenarioGenerator  # noqa: E402

DEFAULT_PLAN = {"no_error": 8, "measurement": 16, "multi_measurement": 12, "parameter": 16, "measurement+parameter": 16,
                "harmonic": 16, "hif": 16, "measurement+hif": 8, "three_phase_unbalance": 16}


def build_generator(balanced_corpus: Path, waveform_corpus: Path, *, output_dir: Path, seed: int, profile: str,
                    admission_rule: str = "chi_square_margin") -> Round0ScenarioGenerator:
    return Round0ScenarioGenerator(
        system="case118", admission_mode="recoverable",
        corpus_path=balanced_corpus / "samples.jsonl", balanced_artifact_dir=balanced_corpus,
        derived_case_dir=output_dir / "derived_cases", seed=seed,
        hif_sample_paths=[waveform_corpus / "hif_samples.jsonl"],
        imbalance_sample_path=waveform_corpus / "unbalance_samples.jsonl",
        evidence_profile=profile, normalized_residual_threshold=4.0, hif_max_scans=3,
        min_measurement_error_sigma=10.0, parameter_ranking_dominance_threshold=1.0, parameter_target_rank_allowance=2,
        admission_rule=admission_rule,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--balanced-corpus", required=True, type=Path, help="a fresh IEEE 118 balanced corpus directory")
    parser.add_argument("--waveform-corpus", required=True, type=Path, help="pipeline_corpus of ieee118_dataset")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan", type=json.loads, default=None)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--profile", default=CLASSIFIER_GATED_PROFILE)
    parser.add_argument("--admission-rule", default="chi_square_margin")
    parser.add_argument("--split", default="development")
    args = parser.parse_args(argv)
    plan = dict(DEFAULT_PLAN)
    if args.plan:
        plan.update({str(k): int(v) for k, v in args.plan.items()})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generator = build_generator(args.balanced_corpus, args.waveform_corpus, output_dir=out, seed=args.seed,
                                profile=args.profile, admission_rule=args.admission_rule)
    scenarios = generator.build(plan)
    envelopes = [partition_release_scenario_v1(row, split=args.split) for row in scenarios]
    (out / f"{args.split}.json").write_text(json.dumps(envelopes, indent=1, sort_keys=True), encoding="utf-8")
    (out / "generator_report.json").write_text(json.dumps(generator.report(), indent=1, default=str), encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        commit = None
    families = Counter(row["grouping"]["scenario_family"] for row in envelopes)
    manifest = {"system": "case118", "profile": args.profile, "plan": plan, "built": dict(sorted(families.items())),
                "skipped_by_reason": generator.report().get("skipped_by_reason"), "seed": args.seed,
                "admission_rule": args.admission_rule, "balanced_corpus": str(args.balanced_corpus.resolve()),
                "waveform_corpus": str(args.waveform_corpus.resolve()), "git_commit": commit,
                "seconds": time.perf_counter() - started}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"built": manifest["built"], "skipped": manifest["skipped_by_reason"], "seconds": round(manifest["seconds"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
