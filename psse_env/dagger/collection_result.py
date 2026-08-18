"""Classify a strict DAgger-1 collection job outcome for orchestration.

A fail-closed collection exits ``1`` and publishes no production data.  That is
a valid, expected scientific result, and it is *not* the same event as a crash,
an OOM, or a walltime kill — but both surface to Slurm as a non-zero exit with
no Python traceback.  The DAgger-1 round-2 chain was reported healthy for two
days on exactly that ambiguity: monitoring looked for tracebacks and CUDA
errors, found none, and never noticed that five gates had rejected the run.

This module resolves the three outcomes explicitly:

``STRICT_GO``
    Collector exited 0, production rows and manifest exist, no failure bundle.
``STRICT_NO_GO``
    Collector exited 1 with a well-formed failure bundle and no production
    outputs.  Report the failed gate names; do not call this healthy.
``INFRASTRUCTURE_FAILURE``
    Anything else: crash, OOM, timeout, missing model, invalid environment, or
    a half-written state that matches neither contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

COLLECTION_RESULT_CONTRACT = "dagger1_collection_result_classification_v1"

STRICT_GO = "STRICT_GO"
STRICT_NO_GO = "STRICT_NO_GO"
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"

#: Distinct exit codes so a scheduler can branch without parsing stdout.
EXIT_CODES = {STRICT_GO: 0, INFRASTRUCTURE_FAILURE: 1, STRICT_NO_GO: 20}

_EXPECTED_NO_GO_EXIT = 1


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def classify_collection_result(
    *,
    exit_code: int,
    production_output: Path | None = None,
    production_manifest: Path | None = None,
    failed_collection_dir: Path | None = None,
) -> dict[str, Any]:
    """Classify one collection job. Pure function over the filesystem state."""
    produced = bool(
        production_output is not None
        and production_output.is_file()
        and production_output.stat().st_size > 0
    )
    manifested = bool(
        production_manifest is not None
        and production_manifest.is_file()
        and production_manifest.stat().st_size > 0
    )
    bundle_dir = failed_collection_dir if failed_collection_dir else None
    bundle_present = bool(bundle_dir is not None and bundle_dir.is_dir())
    evidence: Mapping[str, Any] | None = None
    if bundle_present:
        assert bundle_dir is not None
        loaded = _load_json(bundle_dir / "failure_evidence.json")
        evidence = loaded if isinstance(loaded, Mapping) else None

    result: dict[str, Any] = {
        "contract": COLLECTION_RESULT_CONTRACT,
        "exit_code": int(exit_code),
        "production_outputs_present": produced and manifested,
        "failure_bundle_present": bundle_present,
        "failed_gate_names": [],
        "stopping_reason": None,
        "quarantined_rows": None,
        "executed_episode_count": None,
        "planned_episode_count": None,
        "analysis_only": False,
        "detail": None,
    }

    if evidence is not None:
        result["failed_gate_names"] = [
            str(name) for name in (evidence.get("failed_gate_names") or [])
        ]
        report = evidence.get("collection_stopping_report")
        if isinstance(report, Mapping):
            result["stopping_reason"] = report.get("stopping_reason")
            result["executed_episode_count"] = report.get("executed_episode_count")
            result["planned_episode_count"] = report.get("planned_episode_count")
            result["analysis_only"] = bool(report.get("analysis_only"))
            terminal = report.get("terminal_failure")
            if isinstance(terminal, Mapping):
                result["quarantined_rows"] = terminal.get("quarantined_rows")

    if exit_code == 0 and produced and manifested and not bundle_present:
        result["classification"] = STRICT_GO
        result["detail"] = "production outputs published and all gates passed"
    elif (
        exit_code == _EXPECTED_NO_GO_EXIT
        and bundle_present
        and evidence is not None
        and not produced
    ):
        result["classification"] = STRICT_NO_GO
        gates = ", ".join(result["failed_gate_names"]) or "unreported"
        result["detail"] = (
            f"fail-closed collection: {len(result['failed_gate_names'])} gate(s) "
            f"rejected the run [{gates}]"
        )
    else:
        result["classification"] = INFRASTRUCTURE_FAILURE
        if bundle_present and evidence is None:
            result["detail"] = "failure bundle present but evidence unreadable"
        elif produced and bundle_present:
            result["detail"] = (
                "both production outputs and a failure bundle exist; "
                "collection state is inconsistent"
            )
        elif exit_code not in (0, _EXPECTED_NO_GO_EXIT):
            result["detail"] = (
                f"collector exited {exit_code}: crash, OOM, timeout, or "
                "environment failure"
            )
        elif exit_code == 0:
            result["detail"] = "collector exited 0 without complete production outputs"
        else:
            result["detail"] = (
                "collector exited 1 without a failure bundle; the run died "
                "before the gates were evaluated"
            )
    result["exit_status"] = EXIT_CODES[result["classification"]]
    return result


def format_summary(result: Mapping[str, Any]) -> str:
    """One-line human summary for a Slurm log tail or a monitor notification."""
    parts = [f"[{result['classification']}] {result.get('detail') or ''}".rstrip()]
    executed = result.get("executed_episode_count")
    planned = result.get("planned_episode_count")
    if executed is not None and planned:
        pct = 100.0 * float(executed) / float(planned)
        parts.append(f"episodes {executed}/{planned} ({pct:.1f}% of schedule)")
    if result.get("stopping_reason"):
        parts.append(f"stopping_reason={result['stopping_reason']}")
    if result.get("quarantined_rows"):
        parts.append(f"quarantined_rows={result['quarantined_rows']}")
    if result.get("analysis_only"):
        parts.append("ANALYSIS-ONLY: never eligible for aggregate ingestion")
    return " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a DAgger-1 collection job as STRICT_GO, STRICT_NO_GO, or "
            "INFRASTRUCTURE_FAILURE. Exits 0/20/1 respectively."
        )
    )
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--production-output", type=Path)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--failed-collection-dir", type=Path)
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Emit only the JSON payload, no human summary line",
    )
    args = parser.parse_args(argv)

    result = classify_collection_result(
        exit_code=args.exit_code,
        production_output=args.production_output,
        production_manifest=args.production_manifest,
        failed_collection_dir=args.failed_collection_dir,
    )
    if not args.json_only:
        print(format_summary(result))
    print(json.dumps(result, sort_keys=True))
    return int(result["exit_status"])


if __name__ == "__main__":
    raise SystemExit(main())
