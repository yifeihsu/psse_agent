#!/usr/bin/env python3
"""The small report file evaluate_suite.py reads, one per system of the suites this cell evaluates.

    env_report.py --suite SUITE.json --env-root DIR

evaluate_suite.py fixes its environment from CELL/out/r1/collection/research_run_report.json (the
evidence profile, the environment options, the model and the teacher it checks against
PSSE_EXPERT_VARIANT).  This cell has no collection, so it writes that file itself under
DIR/SYSTEM/out/r1/collection/ and passes --cell-dir DIR/SYSTEM; evaluate_suite.py is unchanged.

The environment is the one the rows were made in: research/rl/decision_audit.py _Harness with
--env-option lifecycle_choice=1, detailed_evidence=1, pair_refit_adjacent=1, flow_at_noise_cue=1,
route_comparison=0.  The system is read from the suite's roots; the progress basis and the
conditioning guard follow it.  The script builds the environment from the report and refuses a
flag that did not land, then prints "SYSTEM RCOND" for the stage (RCOND is the value of
PSSE_MULTI_SCAN_RCOND, or "default" for the variable unset).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_CONTRACT = "compound_cloning_environment_report_v1"
EVIDENCE_PROFILE = "scada_only"
HIF_SEARCH_PROFILE = "research"
MODEL_CHOICE = "12b"
EPISODE_ACTIONS = 40
#: Only there because evaluate_suite.py checks it against PSSE_EXPERT_VARIANT (cell.env exports the
#: same name).  The student needs no teacher, and the rows' teachers differ by system.
EXPERT_VARIANT = "baseline_compare"
#: Every option a teacher variant could otherwise set is named here, so the variant decides nothing.
STUDENT_OPTIONS: dict[str, Any] = {
    "evidence_profile": EVIDENCE_PROFILE,
    "branch_first_partial": False,
    "normalized_residual_threshold": 4.0,
    "triage_classifier": None,
    "lifecycle_choice": 1,
    "detailed_evidence": 1,
    "pair_refit_adjacent": 1,
    "flow_at_noise_cue": 1,
    "route_comparison": 0,
    "revert_fruitless_cycles": 0,
}
#: PSSE_MULTI_SCAN_RCOND by system.  A system not listed here is refused: its setting is a decision.
MULTI_SCAN_RCOND = {"case14": None, "case118": "sparse_symmetric"}


def build_report(system: str) -> dict[str, Any]:
    """The report for one system; the progress basis is the compound rollouts' rule for that system."""
    from psse_env.research_models import get_research_model_spec
    from research.compound_errors.rollout import system_progress_basis

    if system not in MULTI_SCAN_RCOND:
        raise SystemExit(f"no conditioning-guard setting for system {system}; add it to env_report.py MULTI_SCAN_RCOND")
    options = dict(STUDENT_OPTIONS)
    basis = system_progress_basis(system, None)
    if basis:
        options["progress_basis"] = basis
    model = asdict(get_research_model_spec(MODEL_CHOICE))
    return {
        "contract": REPORT_CONTRACT,
        "research_profile": {
            "evidence_profile": EVIDENCE_PROFILE,
            "hif_search_profile": HIF_SEARCH_PROFILE,
            "expert_variant": EXPERT_VARIANT,
            # evaluate_suite.py copies this profile into policies.json: the record of what ran.
            "system": system,
            "multi_scan_rcond": MULTI_SCAN_RCOND[system] or "dense_svd",
        },
        "environment_options": options,
        "model_selection": {key: model[key] for key in ("key", "model_id", "revision", "architecture", "prompt_profile", "purpose")},
    }


def report_path(env_root: Path, system: str) -> Path:
    return Path(env_root) / system / "out" / "r1" / "collection" / "research_run_report.json"


def write_report(env_root: Path, system: str) -> Path:
    """Write atomically: two evaluation jobs of one system may start together."""
    path = report_path(env_root, system)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(build_report(system), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def environment_flags(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build the environment evaluate_suite.py would build from the report and read what landed."""
    import scripts.run_dagger_research as research

    research.RESEARCH_ENVIRONMENT_OPTIONS.update(dict(report["environment_options"]))
    profile = report["research_profile"]
    env = research.resolve_environment_factory(profile["hif_search_profile"], profile["evidence_profile"])()
    providers = env.wls_runner.__self__
    return {
        "lifecycle_choice": env.lifecycle_choice,
        "detailed_evidence": env.detailed_evidence,
        "route_comparison": env.route_comparison,
        "revert_fruitless_cycles": env.revert_fruitless_cycles,
        "progress_basis": env.progress_basis,
        "max_steps": env.max_steps,
        "evidence_profile": env.evidence_profile,
        "provider_detailed_evidence": providers.detailed_evidence,
        "pair_refit_adjacent": providers.pair_refit_adjacent,
        "flow_at_noise_cue": providers.flow_at_noise_cue,
        "normalized_residual_threshold": providers.normalized_residual_threshold,
        "triage": getattr(providers, "triage", None),
    }


def expected_flags(system: str) -> dict[str, Any]:
    from psse_env.transactional_env import DEFAULT_PROGRESS_BASIS
    from research.compound_errors.rollout import system_progress_basis

    return {
        "lifecycle_choice": True,
        "detailed_evidence": True,
        "route_comparison": False,
        "revert_fruitless_cycles": 0,
        "progress_basis": system_progress_basis(system, None) or DEFAULT_PROGRESS_BASIS,
        "max_steps": EPISODE_ACTIONS,
        "evidence_profile": EVIDENCE_PROFILE,
        "provider_detailed_evidence": True,
        "pair_refit_adjacent": True,
        "flow_at_noise_cue": True,
        "normalized_residual_threshold": 4.0,
        "triage": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--env-root", required=True, type=Path)
    args = parser.parse_args(argv)
    from research.compound_errors.rollout import load_suite, suite_system

    system = suite_system(load_suite(args.suite))
    path = write_report(args.env_root, system)
    report = json.loads(path.read_text(encoding="utf-8"))
    flags = environment_flags(report)
    expected = expected_flags(system)
    wrong = {key: (flags.get(key), value) for key, value in expected.items() if flags.get(key) != value}
    if wrong:
        raise SystemExit(f"environment differs from the rows' (found, expected): {wrong}")
    print(json.dumps({"system": system, "report": str(path), "flags": flags}, sort_keys=True, default=str), file=sys.stderr)
    print(system, MULTI_SCAN_RCOND[system] or "default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
