"""Tag every root of a suite with what the deployment verifier can separate (plan step 1.3).

    python -m research.rl.identifiability --suite <development.json> --output-dir <dir> [--profile ...] [--top-k 3]

For a parameter-bearing root the script fetches the parameter context on the
fresh state, then verifies the true line and the top-ranked lines the context
saw (offered or not), each in its own fresh environment (the environment does
not let an accepted candidate be rolled back), and records the verifier's
disposition for each.  For a meter root it does the
same with the residual-ranked channels against the true meter.  The tag is:

* ``separable_by_verification``: only true targets pass verification;
* ``not_separable``: a wrong target passes too (the misranked ceiling);
* ``true_target_unverifiable``: no true target passes, so the root cannot be
  closed by a correction at all under this verifier;
* ``no_balanced_target``: the family has no balanced correction target
  (waveform and healthy roots), tagged without running anything;
* ``context_unavailable``: the context offered nothing to verify.

The separable fraction per family and stratum is the attainable ceiling of
any policy on the draw; a root tagged ``not_separable`` can only be scored
through a bounded handoff.  Everything here runs the observable verifier the
policy would see; the truth is read only to name which candidates are true.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import (  # noqa: E402
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    ROLLBACK_STATE,
    RUN_WLS,
)
from psse_env.dagger.evaluator import _truth_branch_targets, _truth_measurement_targets  # noqa: E402
from psse_env.evidence_profile import validate_evidence_profile  # noqa: E402

TAGS = ("separable_by_verification", "not_separable", "true_target_unverifiable", "no_balanced_target", "context_unavailable")
PARAMETER_FAMILIES = {"parameter", "measurement+parameter"}
MEASUREMENT_FAMILIES = {"measurement", "multi_measurement", "measurement+parameter", "measurement+topology", "measurement+hif"}
ACCEPTING = {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}


def _family(root: Mapping[str, Any]) -> str:
    grouping = root.get("grouping") if isinstance(root.get("grouping"), Mapping) else {}
    return str(grouping.get("scenario_family") or root.get("scenario_family") or "unknown")


def _scenario_id(root: Mapping[str, Any]) -> str:
    execution = root.get("execution") if isinstance(root.get("execution"), Mapping) else {}
    return str(execution.get("scenario_id") or root.get("scenario_id"))


def _stratum(root: Mapping[str, Any]) -> str | None:
    audit = root.get("audit") if isinstance(root.get("audit"), Mapping) else {}
    ranking = audit.get("parameter_ranking") if isinstance(audit.get("parameter_ranking"), Mapping) else {}
    value = ranking.get("stratum")
    return str(value) if value is not None else None


def _truth(root: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = root.get("audit") if isinstance(root.get("audit"), Mapping) else {}
    truth = audit.get("truth")
    return truth if isinstance(truth, Mapping) else {}


def _step(env: Any, action: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state, output = env.step(dict(action))
    return dict(state), dict(output)


def _active_id(env: Any) -> str:
    return str(env.get_policy_observation().as_dict()["active_state_id"])


def _context_tool(family: str) -> str:
    return GET_PARAMETER_CONTEXT if family == "parameter" else GET_MEASUREMENT_CONTEXT


def fetch_context(env_factory: Callable[[], Any], root: Mapping[str, Any], family: str) -> dict[str, Any]:
    """The context the policy would read first on the root's fresh state."""

    env = env_factory()
    env.reset(root["execution"])
    _step(env, {"tool": RUN_WLS, "arguments": {"state_id": _active_id(env)}})
    _state, output = _step(env, {"tool": _context_tool(family), "arguments": {"state_id": _active_id(env)}})
    metrics = output.get("tool_metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def verify_candidate(env_factory: Callable[[], Any], root: Mapping[str, Any], family: str, action: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one correction on the root's fresh state, verify it and record the verifier's verdict.

    Each candidate gets its own environment: the transactional environment
    refuses to roll back an accepted candidate (``rollback_accepted_candidate``)
    and refuses a second correction while one is open, so candidates cannot be
    compared on one state.  The policy has the same limit.
    """

    env = env_factory()
    env.reset(root["execution"])
    _step(env, {"tool": RUN_WLS, "arguments": {"state_id": _active_id(env)}})
    _step(env, {"tool": _context_tool(family), "arguments": {"state_id": _active_id(env)}})
    arguments = {**dict(action.get("arguments") or {}), "state_id": _active_id(env)}
    state, output = _step(env, {"tool": action["tool"], "arguments": arguments})
    record: dict[str, Any] = {"action": {"tool": action["tool"], "arguments": arguments}}
    if output.get("execution_status") != "success":
        record.update(status="execution_failed", error_code=output.get("error_code"),
                      error_detail=str(output.get("error_detail") or "")[:200])
        return record
    observation = env.get_policy_observation().as_dict()
    candidate = observation.get("candidate_state_id") or output.get("candidate_state_id") or state.get("candidate_state_id")
    if candidate is None:
        record["status"] = "no_candidate"
        return record
    _state, verification = _step(env, {"tool": RUN_WLS, "arguments": {"state_id": str(candidate)}})
    metrics = verification.get("tool_metrics") if isinstance(verification.get("tool_metrics"), Mapping) else {}
    oracle = env.get_oracle_state()
    disposition = getattr(oracle, "candidate_disposition", None)
    if disposition is None and isinstance(oracle, Mapping):
        disposition = oracle.get("candidate_disposition")
    disposition = getattr(disposition, "value", disposition)
    record.update(
        status="verified" if verification.get("execution_status") == "success" else "verification_failed",
        disposition=str(disposition) if disposition is not None else None,
        chi_square=metrics.get("chi_square_statistic"),
        alarm=bool(metrics.get("chi_square_alarm") or metrics.get("normalized_residual_alarm")),
    )
    return record


def parameter_candidates(context: Mapping[str, Any], *, top_k: int) -> list[int]:
    ranked = context.get("parameter_ranking_distinct_lines") or []
    lines: list[int] = []
    for item in ranked:
        if isinstance(item, Mapping) and item.get("line_index1") is not None:
            lines.append(int(item["line_index1"]))
        if len(lines) >= top_k:
            break
    return lines


def measurement_candidates(context: Mapping[str, Any], *, top_k: int) -> list[int]:
    findings = context.get("measurement_findings") or []
    indices: list[int] = []
    for item in findings:
        if not isinstance(item, Mapping):
            continue
        for key in ("index", "index0", "measurement_index", "channel_index0"):
            if item.get(key) is not None:
                indices.append(int(item[key]))
                break
        if len(indices) >= top_k:
            break
    return indices


def tag_root(env_factory: Callable[[], Any], root: Mapping[str, Any], *, top_k: int) -> dict[str, Any]:
    family = _family(root)
    truth = _truth(root)
    record: dict[str, Any] = {"scenario_id": _scenario_id(root), "family": family, "stratum": _stratum(root), "tests": []}
    true_lines = {row + 1 for row in _truth_branch_targets(truth, "parameter")} if family in PARAMETER_FAMILIES else set()
    true_meters = _truth_measurement_targets(truth) if family in MEASUREMENT_FAMILIES else set()
    if not true_lines and not true_meters:
        record["tag"] = "no_balanced_target"
        return record
    if true_lines:
        context = fetch_context(env_factory, root, "parameter")
        candidates = parameter_candidates(context, top_k=top_k)
        offered = [int(v) for v in context.get("parameter_ranking_candidate_lines") or []]
        record["parameter_context"] = {"ranked": candidates, "offered": offered,
                                       "dominance_ratio": context.get("parameter_ranking_dominance_ratio"),
                                       "true_lines": sorted(true_lines)}
        for line in dict.fromkeys(list(candidates) + sorted(true_lines)):
            verdict = verify_candidate(env_factory, root, "parameter",
                                       {"tool": CORRECT_PARAMETERS, "arguments": {"line_index": int(line)}})
            verdict.update(family="parameter", target=int(line), true=line in true_lines, offered=line in offered)
            record["tests"].append(verdict)
    if true_meters:
        context = fetch_context(env_factory, root, "measurement")
        candidates = measurement_candidates(context, top_k=top_k)
        record["measurement_context"] = {"ranked": candidates, "true_meters": sorted(true_meters)}
        for index in dict.fromkeys(list(candidates) + sorted(true_meters)):
            verdict = verify_candidate(env_factory, root, "measurement",
                                       {"tool": CORRECT_MEASUREMENTS, "arguments": {"suspect_group": [int(index)]}})
            verdict.update(family="measurement", target=int(index), true=index in true_meters, offered=index in candidates)
            record["tests"].append(verdict)
    tested_true = true_pass = wrong_pass = 0
    for test in record["tests"]:
        passed = test.get("disposition") in ACCEPTING
        if test["true"]:
            tested_true += 1
            true_pass += int(passed)
        else:
            wrong_pass += int(passed)
    if not record["tests"]:
        record["tag"] = "context_unavailable"
    elif true_pass == 0:
        record["tag"] = "true_target_unverifiable"
    elif wrong_pass:
        record["tag"] = "not_separable"
    else:
        record["tag"] = "separable_by_verification"
    true_final = any(test["true"] and test.get("disposition") == "ACCEPT_FINAL" for test in record["tests"])
    wrong_final = any((not test["true"]) and test.get("disposition") == "ACCEPT_FINAL" for test in record["tests"])
    record.update(true_targets_tested=tested_true, true_targets_passing=true_pass, wrong_targets_passing=wrong_pass,
                  # A wrong target accepted only as partial progress while the true one clears the
                  # alarm could be told apart by comparing verdicts, which needs a rollback of an
                  # accepted candidate the contract does not allow today.
                  distinguishable_by_final_verdict=bool(true_final and not wrong_final))
    return record


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Counter] = defaultdict(Counter)
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        by_family[str(record["family"])][str(record["tag"])] += 1
        if record.get("stratum"):
            by_stratum[str(record["stratum"])][str(record["tag"])] += 1
    total = Counter(str(record["tag"]) for record in records)
    balanced = [r for r in records if r["tag"] != "no_balanced_target"]
    not_separable = [r for r in records if r["tag"] == "not_separable"]
    return {
        "roots": len(records),
        "tags": dict(sorted(total.items())),
        "not_separable_but_distinguishable_by_final_verdict": sum(
            1 for r in not_separable if r.get("distinguishable_by_final_verdict")),
        "separable_fraction_of_balanced_roots": (
            sum(1 for r in balanced if r["tag"] == "separable_by_verification") / len(balanced)) if balanced else None,
        "by_family": {family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())},
        "by_stratum": {stratum: dict(sorted(counts.items())) for stratum, counts in sorted(by_stratum.items())},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="classifier_gated_diagnostics")
    parser.add_argument("--triage-model", default=None)
    parser.add_argument("--progress-basis", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    import scripts.run_dagger_research as research

    profile = validate_evidence_profile(args.profile)
    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = profile
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
    research.RESEARCH_ENVIRONMENT_OPTIONS["triage_classifier"] = args.triage_model
    if args.progress_basis:
        research.RESEARCH_ENVIRONMENT_OPTIONS["progress_basis"] = args.progress_basis
    factory = research.resolve_environment_factory("research", profile)
    roots = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    if isinstance(roots, Mapping):
        roots = roots.get("scenarios") or roots.get("rows") or []
    if args.families:
        roots = [root for root in roots if _family(root) in set(args.families)]
    if args.limit:
        roots = roots[: int(args.limit)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with (out / "tags.jsonl").open("w", encoding="utf-8") as stream:
        for index, root in enumerate(roots, start=1):
            record = tag_root(factory, root, top_k=int(args.top_k))
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            print(f"[identifiability] {index}/{len(roots)} {record['scenario_id']} {record['family']} -> {record['tag']}", flush=True)
    summary = summarize(records)
    summary.update(suite=str(args.suite), profile=profile, top_k=int(args.top_k), seconds=time.perf_counter() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["tags"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
