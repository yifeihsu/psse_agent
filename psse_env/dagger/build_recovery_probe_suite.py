"""Build the observable recovery-probe suite as a separate auxiliary source.

The probe suite is published beside the natural DAgger-1 corpus, never inside
it: its own JSONL, its own manifest, its own provenance, and its own replay
quota.  Nothing here may satisfy a natural on-policy floor.

Probe roots are drawn only from scenarios whose physical root is outside the
frozen evaluation suite, the development holdout, and the D0 aggregate.  Sharing
a root with the *natural* DAgger corpus is permitted and reported: a probe
deliberately visits a state the learner did not reach, so it leaks no evaluation
answer, but the overlap belongs in the record.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.dagger.collect_dagger1 import (
    _file_sha256,
    _write_fsynced_jsonl,
    frozen_physical_roots,
)
from psse_env.dagger.recovery_probes import (
    RECOVERY_PROBE_CONTRACT,
    RECOVERY_PROBE_ROOT_QUOTAS,
    generate_recovery_probes,
    recovery_probe_manifest,
)
from psse_env.dagger.release_factories import production_environment_factory
from psse_env.dagger.root_sets import (
    physical_roots_from_artifact,
    root_set_digest,
)
from psse_env.dagger.rollout_collector import classify_state_example
from psse_env.oracle.expert_policy import ExpertPolicyOracle

PROBE_SUITE_ARTIFACT_TYPE = "dagger1_observable_recovery_probe_suite"


def envelope_roots(path: Path) -> set[str]:
    """Physical roots named by a scenario list or a suite mapping.

    Delegates to the shared reader.  The real development holdout is a suite
    mapping, not the envelope list an earlier local reader assumed, so this
    stage raised before any disjointness check could run.
    """

    return physical_roots_from_artifact(path)


def aggregate_roots(directory: Path) -> set[str]:
    """Physical roots present in a D0 aggregate's raw row file."""

    return physical_roots_from_artifact(Path(directory) / "aggregate.raw.jsonl")


def build_recovery_probe_suite(
    *,
    scenarios_path: Path,
    output: Path,
    manifest_path: Path,
    source_commit: str,
    generator_identity: str,
    quotas: Mapping[str, int],
    forbidden_suite: Path | None = None,
    development_holdout: Path | None = None,
    d0_aggregate_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate, verify, and atomically publish one probe suite."""

    output = Path(output)
    manifest_path = Path(manifest_path)
    for path in (output, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite probe artifact: {path}")

    payload = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("probe scenarios must be a non-empty envelope list")

    frozen = set(frozen_physical_roots(forbidden_suite)) if forbidden_suite else set()
    development = envelope_roots(development_holdout) if development_holdout else set()
    d0 = aggregate_roots(d0_aggregate_dir) if d0_aggregate_dir else set()
    excluded = frozen | development | d0

    natural_roots: set[str] = set()
    eligible: list[Mapping[str, Any]] = []
    for envelope in payload:
        if not isinstance(envelope, Mapping):
            continue
        grouping = envelope.get("grouping")
        grouping = grouping if isinstance(grouping, Mapping) else envelope
        root = str(grouping.get("physical_root_fingerprint") or "").strip()
        if not root:
            continue
        natural_roots.add(root)
        if root not in excluded:
            eligible.append(envelope)
    if not eligible:
        raise ValueError(
            "every candidate probe root is excluded by the frozen suite, "
            "development holdout, or D0 aggregate"
        )

    env = production_environment_factory()
    oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
    rows, report = generate_recovery_probes(
        eligible,
        env=env,
        expert_oracle=oracle,
        state_class_for=lambda observation, preferred: classify_state_example(
            observation, preferred_action=preferred
        ),
        quotas=dict(quotas),
    )

    manifest = recovery_probe_manifest(
        rows,
        generator_identity=generator_identity,
        source_commit=source_commit,
        natural_roots=sorted(natural_roots),
        development_roots=sorted(development),
        frozen_evaluation_roots=sorted(frozen),
        d0_roots=sorted(d0),
    )
    manifest.update(
        {
            "artifact_type": PROBE_SUITE_ARTIFACT_TYPE,
            "generation_report": report,
            "candidate_roots_considered": len(eligible),
            "candidate_roots_excluded": len(natural_roots - {
                str(
                    (e.get("grouping") if isinstance(e.get("grouping"), Mapping) else e)
                    .get("physical_root_fingerprint")
                    or ""
                )
                for e in eligible
            }),
            "scenarios_path": str(scenarios_path),
        }
    )

    # Publish rows first so a manifest never names a file that does not exist.
    _write_fsynced_jsonl(output, rows)
    manifest["probe_rows"] = {
        "relative_path": output.name,
        "row_count": len(rows),
        "sha256": _file_sha256(output),
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the observable recovery-probe auxiliary suite. Exits 0 when "
            "the manifest passes, 20 when it does not, so orchestration can "
            "branch without parsing stdout."
        )
    )
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--generator-identity", default="observable_recovery_probe_generator_v1"
    )
    parser.add_argument("--forbidden-suite", type=Path)
    parser.add_argument("--development-holdout", type=Path)
    parser.add_argument("--d0-aggregate-dir", type=Path)
    parser.add_argument(
        "--quota-post-failure-no-candidate",
        type=int,
        default=RECOVERY_PROBE_ROOT_QUOTAS["post_failure_no_candidate"],
    )
    parser.add_argument(
        "--quota-unsupported-correction-recovery",
        type=int,
        default=RECOVERY_PROBE_ROOT_QUOTAS["unsupported_correction_recovery"],
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = build_recovery_probe_suite(
        scenarios_path=args.scenarios,
        output=args.output,
        manifest_path=args.manifest,
        source_commit=args.source_commit,
        generator_identity=args.generator_identity,
        quotas={
            "post_failure_no_candidate": args.quota_post_failure_no_candidate,
            "unsupported_correction_recovery": (
                args.quota_unsupported_correction_recovery
            ),
        },
        forbidden_suite=args.forbidden_suite,
        development_holdout=args.development_holdout,
        d0_aggregate_dir=args.d0_aggregate_dir,
    )
    support = manifest["probe_support"]["probe_strata"]
    summary = " ".join(
        f"{stratum}={entry['distinct_physical_roots']}/"
        f"{entry['minimum_distinct_physical_roots']}"
        for stratum, entry in sorted(support.items())
    )
    print(
        f"[{'PROBE_SUITE_OK' if manifest['passed'] else 'PROBE_SUITE_SHORT'}] "
        f"{summary} roots={manifest['distinct_physical_roots']} "
        f"contract={RECOVERY_PROBE_CONTRACT}"
    )
    print(json.dumps(manifest["probe_support"], sort_keys=True))
    return 0 if manifest["passed"] else 20


if __name__ == "__main__":
    raise SystemExit(main())
