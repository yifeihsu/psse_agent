"""Evaluate the completed comparison-cell adapter without collection or training.

Reuse the saved run's adapter, suite, model, detector, seed and action budget.
The output directory must be new, or carry an identical replay manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def prepare(run_report: Path, suite_path: Path, output: Path) -> tuple[dict, list, dict]:
    report = json.loads(run_report.read_text())
    paired = report["paired_evaluation"]
    profile = report["research_profile"]
    options = report["environment_options"]
    model = report["model_selection"]
    if profile["evidence_profile"] != "suspicion_gated_diagnostics":
        raise ValueError("expected the completed screen-profile comparison cell")
    if profile["expert_variant"] != "ledger_ranked_compare":
        raise ValueError("expected the comparison teacher")
    if options.get("branch_first_partial") is not False:
        raise ValueError("the original verifier must not be relaxed")
    suite = json.loads(suite_path.read_text())
    saved_suite = Path(report["paths"]["development_scenarios"])
    if suite != json.loads(saved_suite.read_text()):
        raise ValueError("suite contents differ from the saved evaluation scenarios")
    roots = [next((container["physical_root_fingerprint"] for container in
                   (row, row.get("grouping") or {}, row.get("metadata") or {})
                   if container.get("physical_root_fingerprint")), "") for row in suite]
    if len(roots) != 209 or len(set(roots)) != 209 or roots != paired["paired_physical_roots"]:
        raise ValueError("expected the original 209 roots in their original evaluation order")
    adapter = Path(paired["r1_adapter"]).resolve(strict=True)
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("completed r1 adapter weights are required")
    # Hash sources as well as the commit: a prompt edit must invalidate any
    # incomplete checkpoint even when weights and episode keys are unchanged.
    source = hashlib.sha256()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode().split("\0")
    source_files = {REPO / name for name in tracked if name.endswith(".py")}
    source_files.update((Path(__file__), REPO / "research/rl/compare_interface_audit.py"))
    for path in sorted(source_files):
        source.update(str(path.relative_to(REPO)).encode())
        source.update(path.read_bytes())
    manifest = {
        "contract": "saved_comparison_evaluation_v1", "training": False,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip(),
        "source_python_sha256": source.hexdigest(),
        "run_report_sha256": digest(run_report), "suite_sha256": digest(suite_path),
        "adapter": str(adapter),
        "adapter_files": {p.name: digest(p) for p in sorted(adapter.iterdir()) if p.is_file()},
        "model": model, "research_profile": profile, "environment_options": options,
        "seed": paired["seed"], "max_steps": paired["max_steps"],
        "load_in_4bit": True, "local_files_only": True, "trust_remote_code": False,
    }
    marker = output / "replay_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != manifest:
            raise ValueError("output contains another run; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return report, suite, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    report, suite, manifest = prepare(args.run_report, args.suite, args.output_dir)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    # The completed cell used these defaults. Use its HPC Python environment
    # and cached model; the driver never downloads a base model or fits weights.
    os.environ["PSSE_EXPERT_VARIANT"] = report["research_profile"]["expert_variant"]
    from scripts import run_dagger_research as runner
    from psse_env.dagger.evaluator import evaluate_rollout_suites
    from psse_env.dagger.research_policy_factory import clear_research_policy_cache, research_gemma_policy_factory

    runner.RESEARCH_ENVIRONMENT_OPTIONS.clear()
    runner.RESEARCH_ENVIRONMENT_OPTIONS.update(report["environment_options"])
    profile = report["research_profile"]
    model = report["model_selection"]
    factory = runner.resolve_environment_factory(profile["hif_search_profile"], profile["evidence_profile"])
    runner.evaluate_paired_adapters(
        development_scenarios=suite, bc0_adapter=Path(manifest["adapter"]), r1_adapter=None,
        base_model=model["model_id"], base_revision=model["revision"],
        output_dir=args.output_dir, evaluation_dirname="evaluation",
        seed=manifest["seed"], max_steps=manifest["max_steps"],
        policy_loader=research_gemma_policy_factory, environment_factory=factory,
        evaluator=evaluate_rollout_suites, load_in_4bit=True, local_files_only=True,
        trust_remote_code=False, architecture=model["architecture"], prompt_profile=model["prompt_profile"],
        policy_cache_clear=clear_research_policy_cache,
        expert_policy_factory=lambda: runner.research_expert_policy(factory),
        parallel=runner.ParallelEvaluation(workers_per_policy=args.workers, environment={
            "hif_search_profile": profile["hif_search_profile"], "evidence_profile": profile["evidence_profile"],
            "options": report["environment_options"],
        }),
    )


if __name__ == "__main__":
    main()
