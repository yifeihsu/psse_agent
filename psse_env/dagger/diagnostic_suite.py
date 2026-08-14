"""Build non-release diagnostic suites from failed closed-loop episodes.

The frozen release suite is immutable and must never become training data.  A
small diagnostic replay is nevertheless useful after a runtime guard changes:
it can cheaply show whether an already-trained checkpoint still repeats the
same failed transition.  This module selects the exact frozen scenarios whose
recorded episodes were nonterminal, invalid, looping, or contained a false
lifecycle decision, and writes them to a different path with a sidecar report
that is explicitly ineligible for release evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from psse_env.actions import CORRECTION_TOOLS, safe_normalize_action


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _episode_requires_diagnostic_replay(episode: Mapping[str, Any]) -> bool:
    return bool(
        episode.get("terminal") is not True
        or int(episode.get("invalid_action_count") or 0) > 0
        or int(episode.get("false_commit_count") or 0) > 0
        or int(episode.get("false_finalization_count") or 0) > 0
        or int(episode.get("false_rollback_count") or 0) > 0
        or episode.get("loop_detected") is True
    )


def _scenario_id(row: Mapping[str, Any]) -> str:
    execution = row.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("diagnostic source scenario has no execution envelope")
    scenario_id = str(execution.get("scenario_id") or "").strip()
    if not scenario_id:
        raise ValueError("diagnostic source scenario has no execution.scenario_id")
    return scenario_id


def build_failure_diagnostic_suite(
    artifact: Mapping[str, Any],
    frozen_suites: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Select failed artifact episodes from the exact frozen suite payload."""

    evaluation = artifact.get("evaluation")
    metrics = evaluation.get("suite_metrics") if isinstance(evaluation, Mapping) else None
    episodes = metrics.get("episodes") if isinstance(metrics, Mapping) else None
    if not isinstance(episodes, list):
        raise ValueError("evaluation artifact has no suite_metrics.episodes list")

    scenario_index: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_suite, raw_rows in frozen_suites.items():
        suite = str(raw_suite)
        if not isinstance(raw_rows, list):
            raise ValueError(f"frozen suite {suite!r} is not a scenario list")
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"frozen suite {suite!r} contains a non-object scenario")
            row = copy.deepcopy(dict(raw_row))
            key = (suite, _scenario_id(row))
            if key in scenario_index:
                raise ValueError(f"duplicate frozen scenario identity: {key!r}")
            scenario_index[key] = row

    selected: dict[str, list[dict[str, Any]]] = {}
    selected_keys: set[tuple[str, str]] = set()
    reasons: Counter[str] = Counter()
    families: Counter[str] = Counter()
    missing: list[str] = []
    for raw_episode in episodes:
        if not isinstance(raw_episode, Mapping) or not _episode_requires_diagnostic_replay(
            raw_episode
        ):
            continue
        suite = str(raw_episode.get("suite") or "").strip()
        scenario_id = str(
            raw_episode.get("scenario_id")
            or raw_episode.get("root_scenario_id")
            or ""
        ).strip()
        key = (suite, scenario_id)
        if not suite or not scenario_id or key not in scenario_index:
            missing.append(str(raw_episode.get("episode_key") or key))
            continue
        if key not in selected_keys:
            selected.setdefault(suite, []).append(copy.deepcopy(scenario_index[key]))
            selected_keys.add(key)
            families[str(raw_episode.get("family") or "unknown")] += 1
        if raw_episode.get("terminal") is not True:
            reasons["nonterminal"] += 1
        if int(raw_episode.get("invalid_action_count") or 0) > 0:
            reasons["invalid_action"] += 1
        if int(raw_episode.get("false_commit_count") or 0) > 0:
            reasons["false_commit"] += 1
        if int(raw_episode.get("false_finalization_count") or 0) > 0:
            reasons["false_finalization"] += 1
        if int(raw_episode.get("false_rollback_count") or 0) > 0:
            reasons["false_rollback"] += 1
        if raw_episode.get("loop_detected") is True:
            reasons["loop"] += 1

    if missing:
        raise ValueError(
            "failed episodes could not be bound to the frozen suite: "
            + ", ".join(sorted(missing))
        )
    if not selected_keys:
        raise ValueError("evaluation artifact contains no failed episodes to replay")
    for rows in selected.values():
        rows.sort(key=_scenario_id)

    report = {
        "artifact_kind": "checkpoint_failure_diagnostic_suite",
        "release_evidence_eligible": False,
        "training_eligible": False,
        "frozen_suite_modified": False,
        "selected_scenarios": len(selected_keys),
        "selected_suites": {
            suite: len(rows) for suite, rows in sorted(selected.items())
        },
        "scenario_families": dict(sorted(families.items())),
        "failure_episode_reasons": dict(sorted(reasons.items())),
        "source_commit": artifact.get("source_commit")
        or (artifact.get("provenance") or {}).get("source_commit")
        or (
            ((artifact.get("provenance") or {}).get("source_state") or {}).get(
                "source_commit"
            )
        ),
    }
    return dict(sorted(selected.items())), report


def audit_failure_diagnostic_evaluation(
    artifact: Mapping[str, Any],
    diagnostic_suites: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the temporary failure replay without creating release evidence.

    The two hard diagnostic targets are deliberately narrow: the runtime guard
    must prevent the old numerical ``parameter_scans_missing`` path, and the
    policy/evaluator combination must not repeat an identical unsupported
    correction. Lifecycle and terminality metrics remain visible but do not
    weaken or replace the frozen promotion gate.
    """

    diagnostic_identity_passed = bool(
        artifact.get("artifact_type") == "closed_loop_diagnostic_evaluation"
        and artifact.get("diagnostic_only") is True
        and artifact.get("release_evidence_eligible") is False
        and artifact.get("training_eligible") is False
        and artifact.get("release_eligible") is False
    )

    expected: set[tuple[str, str]] = set()
    for raw_suite, raw_rows in diagnostic_suites.items():
        suite = str(raw_suite)
        if not isinstance(raw_rows, list):
            raise ValueError(f"diagnostic suite {suite!r} is not a scenario list")
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError(
                    f"diagnostic suite {suite!r} contains a non-object scenario"
                )
            expected.add((suite, _scenario_id(raw_row)))
    if not expected:
        raise ValueError("diagnostic suite contains no scenarios")

    evaluation = artifact.get("evaluation")
    metrics = evaluation.get("suite_metrics") if isinstance(evaluation, Mapping) else None
    episodes = metrics.get("episodes") if isinstance(metrics, Mapping) else None
    if not isinstance(episodes, list):
        raise ValueError("diagnostic evaluation has no suite_metrics.episodes list")

    observed: set[tuple[str, str]] = set()
    parameter_scans_missing = 0
    repeated_unsupported_corrections = 0
    unsupported_corrections = 0
    nonactionable_routes = 0
    premature_commits = 0
    premature_escalations = 0
    terminal = 0
    for raw_episode in episodes:
        if not isinstance(raw_episode, Mapping):
            raise ValueError("diagnostic evaluation contains a non-object episode")
        suite = str(raw_episode.get("suite") or "").strip()
        scenario_id = str(raw_episode.get("scenario_id") or "").strip()
        if suite and scenario_id:
            observed.add((suite, scenario_id))
        if raw_episode.get("terminal") is True:
            terminal += 1
        trace = raw_episode.get("trace")
        if not isinstance(trace, list):
            raise ValueError(
                f"diagnostic episode {(suite, scenario_id)!r} has no trace list"
            )
        for raw_step in trace:
            if not isinstance(raw_step, Mapping):
                raise ValueError("diagnostic trace contains a non-object step")
            action = safe_normalize_action(raw_step.get("action") or {})
            tool = action["tool"]
            error_code = str(raw_step.get("error_code") or "").strip()
            if tool == "correct_parameters" and error_code == "parameter_scans_missing":
                parameter_scans_missing += 1
            if (
                tool in CORRECTION_TOOLS
                and error_code == "evaluation_repeated_nonadvancing_failure"
            ):
                repeated_unsupported_corrections += 1
            if error_code == "correction_not_supported_by_current_context":
                unsupported_corrections += 1
            if error_code == "correction_route_not_actionable":
                nonactionable_routes += 1
            if error_code == "candidate_lifecycle_violation" and tool == "commit_state":
                premature_commits += 1
            if error_code == "operator_escalation_precondition_not_met":
                premature_escalations += 1

    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    hard_failures: list[str] = []
    if not diagnostic_identity_passed:
        hard_failures.append(
            "evaluation artifact is not irreversibly diagnostic-only"
        )
    if missing or unexpected:
        hard_failures.append("diagnostic episode identities do not match the suite")
    if parameter_scans_missing:
        hard_failures.append("parameter_scans_missing reached the numerical executor")
    if repeated_unsupported_corrections:
        hard_failures.append("unsupported corrections were repeated")
    return {
        "artifact_kind": "checkpoint_failure_diagnostic_evaluation_audit",
        "release_evidence_eligible": False,
        "training_eligible": False,
        "passed": not hard_failures,
        "failures": hard_failures,
        "diagnostic_artifact_identity_passed": diagnostic_identity_passed,
        "expected_episode_count": len(expected),
        "observed_episode_count": len(observed),
        "missing_episode_identities": [list(item) for item in missing],
        "unexpected_episode_identities": [list(item) for item in unexpected],
        "hard_targets": {
            "parameter_scans_missing": {
                "observed": parameter_scans_missing,
                "required": 0,
            },
            "repeated_unsupported_corrections": {
                "observed": repeated_unsupported_corrections,
                "required": 0,
            },
        },
        "recovery_observations": {
            "correction_not_supported_by_current_context": unsupported_corrections,
            "correction_route_not_actionable": nonactionable_routes,
            "commit_without_verified_candidate": premature_commits,
            "operator_escalation_precondition_not_met": premature_escalations,
            "terminal_episodes": terminal,
            "nonterminal_episodes": max(len(observed) - terminal, 0),
        },
    }


def write_failure_diagnostic_evaluation_audit(
    *,
    artifact_path: str | Path,
    diagnostic_suite_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    artifact_file = Path(artifact_path).expanduser().resolve(strict=True)
    suite_file = Path(diagnostic_suite_path).expanduser().resolve(strict=True)
    output_file = Path(output_path).expanduser().resolve(strict=False)
    if output_file in {artifact_file, suite_file}:
        raise ValueError("diagnostic audit output must use a separate path")
    artifact = _load_json(artifact_file)
    suites = _load_json(suite_file)
    if not isinstance(artifact, Mapping) or not isinstance(suites, Mapping):
        raise ValueError("diagnostic artifact and suite must be JSON objects")
    report = audit_failure_diagnostic_evaluation(artifact, suites)
    report.update(
        {
            "evaluation_artifact": str(artifact_file),
            "evaluation_artifact_sha256": hashlib.sha256(
                artifact_file.read_bytes()
            ).hexdigest(),
            "diagnostic_suite": str(suite_file),
            "diagnostic_suite_sha256": hashlib.sha256(
                suite_file.read_bytes()
            ).hexdigest(),
        }
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def write_failure_diagnostic_suite(
    *,
    artifact_path: str | Path,
    frozen_suite_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    expected_scenarios: int | None = None,
) -> dict[str, Any]:
    artifact_file = Path(artifact_path).expanduser().resolve(strict=True)
    frozen_file = Path(frozen_suite_path).expanduser().resolve(strict=True)
    output_file = Path(output_path).expanduser().resolve(strict=False)
    if output_file == frozen_file:
        raise ValueError("diagnostic output must not overwrite the frozen suite")
    report_file = (
        Path(report_path).expanduser().resolve(strict=False)
        if report_path is not None
        else output_file.with_suffix(output_file.suffix + ".report.json")
    )
    if report_file in {frozen_file, output_file}:
        raise ValueError("diagnostic report requires a separate non-frozen path")

    artifact = _load_json(artifact_file)
    suites = _load_json(frozen_file)
    if not isinstance(artifact, Mapping) or not isinstance(suites, Mapping):
        raise ValueError("artifact and frozen suite must both be JSON objects")
    diagnostic, report = build_failure_diagnostic_suite(artifact, suites)
    if expected_scenarios is not None and report["selected_scenarios"] != int(
        expected_scenarios
    ):
        raise ValueError(
            "diagnostic scenario count differs from the reviewed failure set: "
            f"expected {int(expected_scenarios)}, selected "
            f"{report['selected_scenarios']}"
        )
    rendered = json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    report.update(
        {
            "source_artifact": str(artifact_file),
            "source_artifact_sha256": hashlib.sha256(
                artifact_file.read_bytes()
            ).hexdigest(),
            "frozen_suite": str(frozen_file),
            "frozen_suite_sha256": hashlib.sha256(frozen_file.read_bytes()).hexdigest(),
            "diagnostic_suite": str(output_file),
            "diagnostic_suite_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        }
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(rendered, encoding="utf-8")
    report_file.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a non-release diagnostic suite from failed episodes."
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--frozen-suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--expected-scenarios",
        type=int,
        help=(
            "Optional reviewed failure-root count. The builder fails closed "
            "rather than silently producing a different diagnostic set."
        ),
    )
    args = parser.parse_args(argv)
    report = write_failure_diagnostic_suite(
        artifact_path=args.artifact,
        frozen_suite_path=args.frozen_suite,
        output_path=args.output,
        report_path=args.report_output,
        expected_scenarios=args.expected_scenarios,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
