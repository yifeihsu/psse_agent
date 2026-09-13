"""Run the configured observable rule expert on balanced transfer scenarios.

This is a local research check, not a learned-policy evaluation or a release
gate. Every supplied physical root is retained, including undetectable faults
and teacher failures. Physical outcomes come from the existing closed-loop
evaluator and strict offline truth audit; this module only presents evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.evaluator import (  # noqa: E402
    ClosedLoopRolloutEvaluator,
    validate_release_scenario_suites,
)
from psse_env.dagger.release_factories import (  # noqa: E402
    EXPERT_POLICY_IDENTITY,
    deterministic_case_loader,
    observable_expert_policy_factory,
    production_environment_factory,
)
from psse_env.dagger.suite_builder import partition_release_scenario_v1  # noqa: E402
from psse_env.dagger.ieee57_runtime import (  # noqa: E402
    ieee57_environment_factory, ieee57_runtime_manifest,
)

CONTRACT = "balanced_transfer_observable_expert_validation_v1"
BALANCED_FAMILIES = frozenset(
    {"no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter"}
)
SUITE = "standard_success"
RECEIPT_PATHS = (
    "scripts/validate_balanced_transfer.py",
    "psse_env/dagger/evaluator.py",
    "psse_env/dagger/release_audit.py",
    "psse_env/dagger/release_factories.py",
    "psse_env/dagger/ieee57_runtime.py",
    "psse_env/dagger/suite_builder.py",
    "psse_env/dagger/study_metrics.py",
    "psse_env/dagger/dataset_builder.py",
    "psse_env/providers/matpower.py",
    "psse_env/providers/scenario_generator.py",
    "psse_env/transactional_env.py",
    "psse_env/state_store.py",
    "psse_env/systems/registry.py",
    "psse_env/oracle/expert_policy.py",
    "psse_env/oracle/anomaly_evidence.py",
    "psse_env/oracle/termination_expert.py",
    "psse_env/oracle/measurement_expert.py",
    "psse_env/oracle/measurement_recovery_evidence.py",
    "psse_env/oracle/parameter_expert.py",
    "psse_env/oracle/candidate_quality.py",
    "psse_env/oracle/process_validity.py",
    "mcp_server/matpower_server.py",
    "mcp_server/case57.m",
)


def normalize_scenarios(payload: Any) -> list[dict[str, Any]]:
    """Accept generator rows or canonical execution/audit/grouping envelopes."""
    if isinstance(payload, Mapping):
        if "scenarios" in payload:
            payload = payload["scenarios"]
        elif SUITE in payload and set(payload) == {SUITE}:
            payload = payload[SUITE]
        else:
            raise ValueError("scenario JSON must contain a scenarios list or standard_success list")
    if not isinstance(payload, list) or not payload:
        raise ValueError("scenario JSON must contain a non-empty list")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"scenario {index} must be a mapping")
        row = (
            copy.deepcopy(dict(raw))
            if "scenario_schema_version" in raw
            else partition_release_scenario_v1(raw, split="development")
        )
        family = row.get("grouping", {}).get("scenario_family")
        if family not in BALANCED_FAMILIES:
            raise ValueError(f"scenario {index} is outside balanced transfer scope: {family!r}")
        rows.append(row)
    validate_release_scenario_suites({SUITE: rows})
    roots = [row["grouping"]["physical_root_fingerprint"] for row in rows]
    if len(roots) != len(set(roots)):
        raise ValueError("validation requires one episode per distinct physical root")
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def summarize_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Extract existing deployment evidence without changing success scoring."""
    first_wls: dict[str, Any] | None = None
    acquisitions: list[dict[str, Any]] = []
    for transition in episode.get("trace", []):
        action = _mapping(transition.get("action"))
        tool = action.get("tool")
        output = _mapping(transition.get("policy_tool_output"))
        metrics = _mapping(output.get("tool_metrics"))
        if tool == "run_wls" and first_wls is None:
            first_wls = {
                "execution_status": output.get("execution_status"),
                "error_code": output.get("error_code"),
                **{
                    key: copy.deepcopy(metrics.get(key))
                    for key in (
                        "wls_objective", "chi_square_statistic", "chi_square_threshold",
                        "chi_square_dof", "chi_square_alpha", "chi_square_ratio", "max_normalized_residual",
                        "normalized_residual_threshold", "chi_square_alarm",
                        "normalized_residual_alarm", "anomaly_detection_rule",
                        "converged", "no_material_anomaly_remaining", "physical_constraints_ok",
                    )
                },
            }
        if tool in {"get_harmonic_context", "get_three_phase_context"}:
            acquisitions.append(
                {
                    "step": transition.get("step"),
                    "tool": tool,
                    "execution_status": output.get("execution_status"),
                    "error_code": output.get("error_code"),
                    "context_status": metrics.get(
                        "harmonic_context_status" if tool == "get_harmonic_context"
                        else "three_phase_context_status"
                    ),
                    "available_evidence_channels": copy.deepcopy(
                        metrics.get("available_evidence_channels")
                    ),
                }
            )
    audit = _mapping(episode.get("audit"))
    return {
        **{key: copy.deepcopy(episode.get(key)) for key in (
            "scenario_id", "physical_root", "family", "cardinality", "case", "steps",
            "terminal", "terminal_outcome", "final_physical_success", "final_physical_correct",
            "physical_correctness_known", "truth_audited_task_success",
            "truth_audited_task_success_evidence_known", "healthy_components_preserved",
            "healthy_preservation_known", "false_commit_count", "false_finalization_count",
            "invalid_action_count", "evaluator_error",
        )},
        "initial_wls": first_wls,
        "acquisitions": acquisitions,
        "physical_outcome_audit": {
            key: copy.deepcopy(audit.get(key))
            for key in (
                "audit_mode", "evidence_complete", "problems", "remaining_true_fault_count",
                "final_physical_correct", "physical_correctness_known",
                "healthy_components_preserved", "healthy_preservation_known",
                "accepted_target_audit", "strict_release_audit",
                "post_correction_handoff_assessment", "truth_audited_task_assessment",
            )
        },
    }


def evaluate_scenarios(
    scenarios: Sequence[Mapping[str, Any]], *, seed: int = 20260910,
    max_steps: int = 40, progress_callback: Any = None,
    chi2_alpha: float = 0.05, normalized_residual_threshold: float | None = 4.0,
) -> dict[str, Any]:
    """Run all input roots once in the attested production environment."""
    if not 1 <= max_steps <= 40:
        raise ValueError("max_steps must be between 1 and the production budget of 40")
    if isinstance(chi2_alpha, bool) or not math.isfinite(chi2_alpha) or not 0 < chi2_alpha < 1:
        raise ValueError("chi2_alpha must be finite and strictly between zero and one")
    if normalized_residual_threshold is not None and (
        isinstance(normalized_residual_threshold, bool)
        or not math.isfinite(normalized_residual_threshold)
        or normalized_residual_threshold <= 0
    ):
        raise ValueError("normalized_residual_threshold must be finite and positive, or None")
    rows = normalize_scenarios(list(scenarios))
    configuration = {
        "chi2_alpha": float(chi2_alpha),
        "normalized_residual_threshold": normalized_residual_threshold,
        "normalized_residual_alarm_comparison": ">=",
        "anomaly_rule": "chi_square_only" if normalized_residual_threshold is None
        else "chi_square_or_normalized_residual",
        "parameter_and_measurement_screening_unchanged": True,
    }
    source_before = collect_source_receipt(phase="before_evaluation")
    pinned_ieee57 = chi2_alpha == 0.05 and normalized_residual_threshold == 4.0 and all(
        str(row.get("execution", {}).get("case", "")).lower() in {"case57", "ieee57"}
        or str(row.get("grouping", {}).get("case_id", "")).lower() in {"case57", "ieee57"}
        for row in rows
    )
    configuration["pinned_runtime"] = ieee57_runtime_manifest() if pinned_ieee57 else None
    environment_factory = ieee57_environment_factory if pinned_ieee57 else partial(
        production_environment_factory, chi2_alpha=chi2_alpha,
        normalized_residual_threshold=normalized_residual_threshold,
    )
    evaluator = ClosedLoopRolloutEvaluator(
        env_factory=environment_factory,
        policy_factory=observable_expert_policy_factory,
        case_loader=deterministic_case_loader,
        max_steps=max_steps,
        seed=seed,
        required_suites=(SUITE,),
        minimum_suites=1,
        minimum_episodes_per_suite=len(rows),
        minimum_roots_per_suite=len(rows),
        require_release_environment=True,
        expected_policy_identity={
            "explicit_policy_identity": EXPERT_POLICY_IDENTITY,
            "model_id": None,
            "model_revision": None,
        },
        require_policy_identity=True,
        progress_callback=progress_callback,
    )
    result = evaluator.evaluate({SUITE: rows}).as_dict()
    source_after = collect_source_receipt(phase="after_evaluation")
    report = summarize_retained_evaluation(
        result, seed=seed, max_steps=max_steps, expected_physical_roots=len(rows),
        detection_configuration=configuration,
    )
    changed = sorted(
        path for path in source_before["sha256"]
        if source_before["sha256"][path] != source_after["sha256"][path]
    )
    report["runtime_source_attestation"] = {
        "scope": "selected_core_source_files",
        "before": source_before,
        "after": source_after,
        "source_hashes_matched": not changed,
        "changed_paths": changed,
    }
    if changed:
        report["infrastructure_errors"].append({
            "error": "selected_source_files_changed_during_evaluation", "paths": changed,
        })
    return report


def collect_source_receipt(*, phase: str = "collection_after_evaluation") -> dict[str, Any]:
    """Record source-file bytes at a declared collection or execution phase."""
    if phase not in {"before_evaluation", "after_evaluation", "collection_after_evaluation"}:
        raise ValueError("invalid source-receipt phase")
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_phase": phase,
        "scope": "selected_core_sources",
        "original_runtime_source_snapshot": phase != "collection_after_evaluation",
        "note": (
            "Hashes describe files when this receipt was collected; they do not attest original evaluation runtime bytes."
            if phase == "collection_after_evaluation" else
            "Source-file snapshot at the declared phase of this evaluation; compare before and after receipts for stability."
        ),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sha256": {
            relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in RECEIPT_PATHS
        },
    }


def summarize_retained_evaluation(
    result: Mapping[str, Any], *, seed: int, max_steps: int,
    expected_physical_roots: int, created_at_utc: str | None = None,
    detection_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Present a retained evaluator result; never replay or rescore its episodes."""
    episodes = result["suite_metrics"]["episodes"]
    summaries = [summarize_episode(episode) for episode in episodes]
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    infrastructure_errors: list[dict[str, Any]] = []
    for episode in summaries:
        counts = family_counts[str(episode["family"])]
        counts["physical_roots"] += 1
        counts["truth_audited_task_successes"] += int(episode["truth_audited_task_success"] is True)
        counts["final_physical_successes"] += int(episode["final_physical_success"] is True)
        counts["final_physical_correct"] += int(episode["final_physical_correct"] is True)
        counts["healthy_components_preserved"] += int(episode["healthy_components_preserved"] is True)
        counts["healthy_preservation_known"] += int(episode["healthy_preservation_known"] is True)
        counts["teacher_task_failures"] += int(episode["truth_audited_task_success"] is not True)
        counts["unknown_task_evidence"] += int(episode["truth_audited_task_success_evidence_known"] is not True)
        counts["false_commits"] += int(episode["false_commit_count"] or 0)
        counts["false_finalizations"] += int(episode["false_finalization_count"] or 0)
        counts["invalid_actions"] += int(episode["invalid_action_count"] or 0)
        baseline = episode.get("initial_wls") or {}
        clean = baseline.get("no_material_anomaly_remaining")
        counts["initially_flagged_anomalous"] += int(clean is False)
        counts["initially_not_flagged"] += int(clean is True)
        counts["initial_detection_unknown"] += int(not isinstance(clean, bool))
        if episode.get("evaluator_error"):
            infrastructure_errors.append({
                "scenario_id": episode["scenario_id"],
                "error": episode["evaluator_error"],
            })
        elif episode["truth_audited_task_success_evidence_known"] is not True:
            infrastructure_errors.append({
                "scenario_id": episode["scenario_id"],
                "error": "truth_audited_task_success_evidence_unknown",
            })
    return {
        "contract": CONTRACT,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scope": "local_balanced_observable_expert_only",
        "learned_policy_evaluated": False,
        "all_supplied_physical_roots_retained": len(episodes) == expected_physical_roots,
        "seed": seed,
        "max_steps": max_steps,
        "detection_configuration": copy.deepcopy(detection_configuration),
        "initial_detection_summary": {
            "clean_control_count": sum(item["family"] == "no_error" for item in summaries),
            "clean_control_alarm_count": sum(
                item["family"] == "no_error"
                and (item.get("initial_wls") or {}).get("no_material_anomaly_remaining") is False
                for item in summaries
            ),
            "faulted_root_count": sum(item["family"] != "no_error" for item in summaries),
            "faulted_root_alarm_count": sum(
                item["family"] != "no_error"
                and (item.get("initial_wls") or {}).get("no_material_anomaly_remaining") is False
                for item in summaries
            ),
        },
        "outcome_semantics": {
            "truth_audited_task_success": "Existing evaluator task outcome, including audited completion after controller handoff.",
            "final_physical_success": "Existing evaluator strict resolved outcome; audited controller handoff alone does not set this field.",
            "final_physical_correct": "Existing evaluator physical correctness field and known-evidence flag, also copied directly inside physical_outcome_audit.",
        },
        "source_receipt": collect_source_receipt(),
        "family_summary": {key: dict(value) for key, value in sorted(family_counts.items())},
        "infrastructure_errors": infrastructure_errors,
        "episodes": summaries,
        "evaluation": result,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--progress", action="store_true", help="Print one progress record after each episode.")
    parser.add_argument("--chi2-alpha", type=float, default=0.05)
    residual_options = parser.add_mutually_exclusive_group()
    residual_options.add_argument("--normalized-residual-threshold", type=float, default=4.0)
    residual_options.add_argument(
        "--chi-square-only", action="store_true",
        help="Disable the normalized-residual alarm; use --chi2-alpha 0.01 for the historical baseline.",
    )
    args = parser.parse_args(argv)
    source = args.scenarios.expanduser().resolve(strict=True)
    source_bytes = source.read_bytes()
    scenarios = normalize_scenarios(json.loads(source_bytes.decode("utf-8-sig")))
    def progress(record: Mapping[str, Any]) -> None:
        if record.get("event") == "episode_complete":
            print(json.dumps(dict(record)), flush=True)
    report = evaluate_scenarios(
        scenarios, seed=args.seed, max_steps=args.max_steps,
        chi2_alpha=args.chi2_alpha,
        normalized_residual_threshold=None if args.chi_square_only else args.normalized_residual_threshold,
        progress_callback=progress if args.progress else None,
    )
    report["source"] = {
        "path": str(source), "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "physical_roots": len(scenarios),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "family_summary": report["family_summary"],
                      "infrastructure_errors": report["infrastructure_errors"]}, indent=2))
    return 2 if report["infrastructure_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
