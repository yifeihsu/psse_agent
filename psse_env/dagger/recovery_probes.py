"""Observable recovery probes: a separately provenance-bound auxiliary source.

The complete-schedule analysis run executed all 477 predeclared episodes and
still produced only three distinct physical roots for each of
``post_failure_no_candidate`` and ``unsupported_correction_recovery``, against a
release floor of ten.  Every well-supplied stratum scaled with the 3.2x episode
increase; these two did not.  The deficit is a property of how rarely the
learner enters those states, not of the earlier early stop.

Mining fresh roots until the learner happens to make ten of each mistake would
be a poor competence gate: an improved learner would fail the release for making
*fewer* mistakes.  This module instead supplies a small predeclared intervention
suite as an explicitly separate source, so the corpus becomes

    natural DAgger corpus + observable recovery-probe auxiliary corpus

rather than a silently redefined DAgger.  The probe rows are never claimed to be
learner-visited and never count toward natural on-policy support; see
``audit_dagger1_training_support`` for the three-way report that keeps them
apart, deduplicating roots shared between the two sources.

Construction rules, in order:

1.  Build only the *intervention*, never the recovery target.
2.  Derive it from policy-visible evidence under a frozen deterministic rule.
3.  Never consult hidden truth to select or rank it.
4.  Execute it through the ordinary process-validity/environment path.
5.  Materialise the resulting real ``PolicyObservation``.
6.  Let the ordinary observable expert choose the rank-one recovery action.
7.  Use private truth only afterwards, to audit what is already fixed.
8.  Retain at most one probe row per physical root.

The vocabulary here is not new: ``DAGGER_ITERATION_1_RECOVERY_GATE_POLICY``
already lists ``observable_recovery_probe`` in both ``allowed_state_origins``
and ``allowed_dataset_sources``, and prohibits ``synthetic_counterfactual``.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    GET_MEASUREMENT_CONTEXT,
    RUN_WLS,
)
from psse_env.dagger.offline_teacher_target_audit import (
    offline_teacher_target_audit,
)
from psse_env.dagger.release_factories import select_observable_expert_actions
from psse_env.oracle.process_validity import (
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    post_correction_confirmation_required,
)
from psse_env.dagger.rollout_collector import (
    classify_dagger1_recovery_stratum,
    observable_rank_one_target_proof,
)

RECOVERY_PROBE_CONTRACT = "dagger1_observable_recovery_probe_v1"
RECOVERY_PROBE_STATE_ORIGIN = "observable_recovery_probe"
RECOVERY_PROBE_DATASET_SOURCE = "observable_recovery_probe"
RECOVERY_PROBE_COLLECTION_ROLE = "auxiliary_training"

#: Only the two strata the natural schedule provably cannot fill.  Every other
#: floor stays a natural on-policy floor.
RECOVERY_PROBE_STRATA: tuple[str, ...] = (
    "post_failure_no_candidate",
    "unsupported_correction_recovery",
)
#: Ten is the binding floor; two are reserve margin against a probe that lands
#: in a neighbouring stratum and is discarded by ``verify_probe_stratum``.
RECOVERY_PROBE_ROOT_FLOORS: dict[str, int] = {
    stratum: 10 for stratum in RECOVERY_PROBE_STRATA
}
RECOVERY_PROBE_ROOT_QUOTAS: dict[str, int] = {
    stratum: 12 for stratum in RECOVERY_PROBE_STRATA
}

#: Frozen suffix for the deliberately unbindable state reference.  Constant so
#: the intervention is reproducible from the observation alone.
_UNBOUND_STATE_SUFFIX = "probe_unbound_state"

# The former unsupported-correction probe issued a grouped two-target
# correction. It is deliberately deleted rather than deprecated: it produced two
# unranked expert actions that the rank-one proof must reject, so any reuse
# would silently reintroduce a 0/10 stratum.


def _measurement_context(observation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    context = observation.get("fresh_context_evidence")
    measurement = context.get("measurement") if isinstance(context, Mapping) else None
    return measurement if isinstance(measurement, Mapping) else None


def _supported_measurement_targets(
    measurement: Mapping[str, Any], *, state_id: str
) -> set[int]:
    """Every measurement target reachable through the same-state inventory."""

    supported = measurement.get("supported_corrections")
    targets: set[int] = set()
    if not isinstance(supported, Sequence) or isinstance(supported, (str, bytes)):
        return targets
    for entry in supported:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("tool") or "") != CORRECT_MEASUREMENTS:
            continue
        arguments = entry.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        if str(arguments.get("state_id") or "") != state_id:
            continue
        group = arguments.get("suspect_group")
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            continue
        for item in group:
            try:
                targets.add(int(item))
            except (TypeError, ValueError):
                continue
    return targets


def _observable_findings(measurement: Mapping[str, Any]) -> list[int]:
    """Policy-visible measurement indices, ascending and de-duplicated."""

    findings = measurement.get("measurement_findings")
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)):
        return []
    seen: set[int] = set()
    for item in findings:
        if not isinstance(item, Mapping):
            continue
        for key in ("index0", "index", "measurement_index"):
            if item.get(key) is not None:
                try:
                    seen.add(int(item[key]))
                except (TypeError, ValueError):
                    pass
                break
    return sorted(seen)


def post_failure_no_candidate_intervention(
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Request context bound to a state reference that cannot resolve.

    A read-only context request fails without opening a candidate, and its tool
    is neither a correction, a commit, nor an escalation, so the three earlier
    ``previous_failed`` branches of the stratum classifier cannot claim it.
    Requires that no candidate is already open: with one, the resulting state is
    a different stratum entirely.
    """

    if observation.get("has_open_candidate") or observation.get("candidate_state_id"):
        return None
    active = str(observation.get("active_state_id") or "")
    if not active:
        return None
    episode = active.split(":", 1)[0]
    return {
        "tool": GET_MEASUREMENT_CONTEXT,
        "arguments": {"state_id": f"{episode}:{_UNBOUND_STATE_SUFFIX}"},
    }


#: The same-state supported-correction inventory does not exist at episode
#: start: it is published by a context request, after the residuals that
#: motivate it.  A probe that needs to read the inventory must therefore prime
#: it first, with ordinary legal actions and no reference to hidden truth.


# Backwards-compatible name for callers/tests; the predicate itself is owned by
# the production process gate so a probe cannot drift from the guard it claims
# to exercise.
post_correction_confirmation_pending = post_correction_confirmation_required


def confirmation_violation_intervention(
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Repeat an accepted correction instead of honouring the confirmation.

    Built only from the observable accepted-correction record: an existing
    source action rebound to the current active state.  No new target and no new
    physical value is introduced, so this cannot manufacture a fault -- it models
    a learner opening another autonomous transaction when the protocol requires
    confirmation and handoff.
    """

    if not post_correction_confirmation_pending(observation):
        return None
    active_state_id = str(observation.get("active_state_id") or "").strip()
    if not active_state_id:
        return None
    accepted = observation.get("accepted_corrections") or []
    sources = [
        entry.get("source_action")
        for entry in accepted
        if isinstance(entry, Mapping)
        and isinstance(entry.get("source_action"), Mapping)
        and str(entry["source_action"].get("tool") or "") == CORRECT_MEASUREMENTS
    ]
    if not sources:
        return None
    # Frozen deterministic rule: the most recently accepted correction.
    source = sources[-1]
    arguments = dict(source.get("arguments") or {})
    arguments["state_id"] = active_state_id
    return {"tool": CORRECT_MEASUREMENTS, "arguments": arguments}


RECOVERY_PROBE_INTERVENTIONS = {
    "post_failure_no_candidate": post_failure_no_candidate_intervention,
    "unsupported_correction_recovery": confirmation_violation_intervention,
}

#: Preconditions the observable prefix must reach before an intervention may
#: fire.  Firing early is how the previous unsupported probe produced
#: missing_precondition rows instead of the intended stratum.
RECOVERY_PROBE_PRECONDITIONS = {
    "post_failure_no_candidate": lambda observation: bool(
        observation.get("active_state_id")
        and not observation.get("has_open_candidate")
        and not observation.get("candidate_state_id")
    ),
    "unsupported_correction_recovery": post_correction_confirmation_pending,
}

#: Bound on the observable-expert prefix driven before an intervention.
RECOVERY_PROBE_MAX_PREFIX_STEPS = 24


def probe_intervention_precondition(
    observation: Mapping[str, Any], *, stratum: str
) -> bool:
    """Whether this observation is the exact state the intervention targets."""

    rule = RECOVERY_PROBE_PRECONDITIONS.get(str(stratum))
    if rule is None:
        raise ValueError(f"no recovery-probe precondition for stratum {stratum!r}")
    return bool(rule(observation))


def probe_intervention(
    observation: Mapping[str, Any], *, stratum: str
) -> dict[str, Any] | None:
    """Intervention for one target stratum, or ``None`` if this root cannot host it."""

    rule = RECOVERY_PROBE_INTERVENTIONS.get(str(stratum))
    if rule is None:
        raise ValueError(f"no recovery-probe intervention for stratum {stratum!r}")
    return rule(observation)


def verify_probe_stratum(
    observation: Mapping[str, Any],
    *,
    preferred_action: Mapping[str, Any] | str | None,
    state_class: str,
    scenario_family: str,
    error_cardinality: int,
    expected_stratum: str,
) -> dict[str, Any]:
    """Classify the post-intervention state and require the intended stratum.

    An intervention that lands in a neighbouring stratum is discarded rather
    than relabelled: the classifier, not the generator, decides what a row is.
    """

    actual = classify_dagger1_recovery_stratum(
        observation,
        preferred_action=preferred_action,
        state_class=state_class,
        scenario_family=scenario_family,
        error_cardinality=int(error_cardinality),
    )
    return {
        "contract": RECOVERY_PROBE_CONTRACT,
        "expected_stratum": str(expected_stratum),
        "actual_stratum": actual,
        "passed": actual == str(expected_stratum),
    }


def stamp_recovery_probe_row(
    row: Mapping[str, Any],
    *,
    intervention: Mapping[str, Any],
    expected_stratum: str,
    verification: Mapping[str, Any],
    rank_one_proof: Mapping[str, Any] | None = None,
    teacher_target_audit: Mapping[str, Any] | None = None,
    training_decision_evidence_verified: bool = False,
) -> dict[str, Any]:
    """Stamp the auxiliary-source identity onto one fully audited probe row.

    A probe target earns its place the same way a natural DAgger label does.
    Being the expert's first returned action is not sufficient: it must carry
    the observable rank-one proof, and it must survive the private teacher-target
    audit, exactly as a learner-visited target would.

    The row is never marked learner-visited.  ``production_label_eligible``
    stays false so a probe can never satisfy a natural on-policy release floor;
    ``auxiliary_training_eligible`` is the separate, explicit permission that the
    validated probe-ingestion path checks instead.
    """

    if verification.get("passed") is not True:
        raise ValueError(
            "refusing to stamp a probe row whose stratum verification failed: "
            f"{verification.get('actual_stratum')!r} != {expected_stratum!r}"
        )
    if not training_decision_evidence_verified:
        raise ValueError(
            "refusing to stamp a probe row without verified training-decision "
            "evidence"
        )
    if (rank_one_proof or {}).get("passed") is not True:
        raise ValueError(
            "refusing to stamp a probe row whose observable rank-one target "
            f"proof failed: {(rank_one_proof or {}).get('reason')!r}"
        )
    if (teacher_target_audit or {}).get("passed") is not True:
        raise ValueError(
            "refusing to stamp a probe row quarantined by the private "
            f"teacher-target audit: {(teacher_target_audit or {}).get('reason_codes')!r}"
        )
    stamped = dict(row)
    stamped.update(
        {
            "collector_contract": RECOVERY_PROBE_CONTRACT,
            "dataset_mode": "production",
            "state_origin": RECOVERY_PROBE_STATE_ORIGIN,
            "dataset_source": RECOVERY_PROBE_DATASET_SOURCE,
            "replay_source": RECOVERY_PROBE_DATASET_SOURCE,
            "collection_role": RECOVERY_PROBE_COLLECTION_ROLE,
            "state_visited_by": RECOVERY_PROBE_STATE_ORIGIN,
            "recovery_stratum": str(expected_stratum),
            "probe_intervention": dict(intervention),
            "probe_stratum_verification": dict(verification),
            "training_decision_evidence_verified": True,
            "observable_rank_one_target_proof": dict(rank_one_proof or {}),
            "offline_teacher_target_audit": dict(teacher_target_audit or {}),
            "auxiliary_training_eligible": True,
            "production_label_eligible": False,
            "natural_on_policy_support_eligible": False,
        }
    )
    return stamped


def audit_recovery_probe_support(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Distinct-root support per probe stratum, against the probe floors."""

    roots_by_stratum: dict[str, set[str]] = {
        stratum: set() for stratum in RECOVERY_PROBE_STRATA
    }
    rows_by_root: Counter[tuple[str, str]] = Counter()
    foreign_rows = 0
    unverified_rows = 0
    for row in rows:
        if row.get("dataset_source") != RECOVERY_PROBE_DATASET_SOURCE:
            foreign_rows += 1
            continue
        verification = row.get("probe_stratum_verification")
        if (
            not isinstance(verification, Mapping)
            or verification.get("passed") is not True
        ):
            unverified_rows += 1
            continue
        stratum = str(row.get("recovery_stratum") or "")
        root = str(row.get("physical_root_fingerprint") or "")
        if stratum not in roots_by_stratum or not root:
            foreign_rows += 1
            continue
        roots_by_stratum[stratum].add(root)
        rows_by_root[(stratum, root)] += 1

    # Rule 8: one designated recovery row per physical root, so a single root
    # cannot manufacture support it does not have.
    duplicated = sorted(
        f"{stratum}:{root}" for (stratum, root), n in rows_by_root.items() if n > 1
    )
    strata = {
        stratum: {
            "distinct_physical_roots": len(roots),
            "minimum_distinct_physical_roots": RECOVERY_PROBE_ROOT_FLOORS[stratum],
            "root_shortfall": max(
                RECOVERY_PROBE_ROOT_FLOORS[stratum] - len(roots), 0
            ),
            "passed": len(roots) >= RECOVERY_PROBE_ROOT_FLOORS[stratum],
        }
        for stratum, roots in sorted(roots_by_stratum.items())
    }
    return {
        "contract": RECOVERY_PROBE_CONTRACT,
        "probe_strata": strata,
        "rows": len(rows),
        "foreign_rows": foreign_rows,
        "unverified_rows": unverified_rows,
        "roots_with_multiple_rows": duplicated,
        "passed": bool(
            strata
            and all(entry["passed"] for entry in strata.values())
            and not foreign_rows
            and not unverified_rows
            and not duplicated
        ),
    }


def recovery_probe_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    generator_identity: str,
    source_commit: str,
    natural_roots: Sequence[str] = (),
    development_roots: Sequence[str] = (),
    frozen_evaluation_roots: Sequence[str] = (),
    d0_roots: Sequence[str] = (),
) -> dict[str, Any]:
    """Manifest binding the probe suite to its own provenance and disjointness.

    Probe roots must not overlap the development holdout or the frozen
    evaluation suite, for the same reason natural roots must not.  Overlap with
    natural DAgger roots is reported but not fatal: a probe deliberately visits
    a state the learner did not reach, so sharing a physical root does not leak
    an evaluation answer.  Overlap with D0 is likewise reported, since the D0
    root set is forbidden to natural DAgger-1 collection.
    """

    support = audit_recovery_probe_support(rows)
    probe_roots = {
        str(row.get("physical_root_fingerprint") or "")
        for row in rows
        if row.get("dataset_source") == RECOVERY_PROBE_DATASET_SOURCE
    }
    probe_roots.discard("")

    def overlap(other: Sequence[str]) -> list[str]:
        return sorted(probe_roots & {str(item) for item in other})

    development_overlap = overlap(development_roots)
    evaluation_overlap = overlap(frozen_evaluation_roots)
    disjointness = {
        "development_holdout_overlap": development_overlap,
        "frozen_evaluation_overlap": evaluation_overlap,
        "natural_dagger_overlap": overlap(natural_roots),
        "d0_overlap": overlap(d0_roots),
        "passed": not development_overlap and not evaluation_overlap,
    }
    return {
        "contract": RECOVERY_PROBE_CONTRACT,
        "artifact_type": "dagger1_observable_recovery_probe_suite",
        "generator_identity": str(generator_identity),
        "source_commit": str(source_commit),
        "state_origin": RECOVERY_PROBE_STATE_ORIGIN,
        "dataset_source": RECOVERY_PROBE_DATASET_SOURCE,
        "collection_role": RECOVERY_PROBE_COLLECTION_ROLE,
        "root_quotas": dict(sorted(RECOVERY_PROBE_ROOT_QUOTAS.items())),
        "distinct_physical_roots": len(probe_roots),
        "probe_support": support,
        "root_disjointness": disjointness,
        "training_eligible": True,
        "natural_on_policy_support_eligible": False,
        "passed": bool(support["passed"] and disjointness["passed"]),
    }


def combined_recovery_support(
    natural_rows: Sequence[Mapping[str, Any]],
    probe_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Three-way support over the actual rows.

    This used to add natural and probe root counts, on the assumption that probe
    roots are drawn root-disjointly from natural ones.  The real corpus refutes
    that: all 24 pilot probe roots coincide with natural support roots, so the
    additive figure overcounted every shared root.  There is one authoritative
    implementation, and it deduplicates.
    """

    from psse_env.dagger.replay_buffer import audit_dagger1_training_support

    return audit_dagger1_training_support(natural_rows, probe_rows)

#: Grouping fields the offline audit reads for identity, mirroring the natural
#: collector's export allowlist.  These are not projected into PolicyObservation.
_AUDIT_GROUPING_FIELDS = (
    "physical_root_fingerprint",
    "scenario_family",
    "error_cardinality",
    "network_case",
    "source_tier",
    "dataset_split",
    "parameter_scans_available",
)


def prepare_scenario_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Split one envelope into the three views collection needs.

    Returns ``runtime`` (what reaches ``env.reset``), ``audit`` (runtime plus the
    private truth ledger, for the offline teacher-target audit), and
    ``grouping`` (public identity).  Keeping the split in one helper is what
    stops a probe from being audited against a ledger the natural collector
    would have hydrated: today every probe target is the read-only
    ``get_measurement_context``, for which the audit's admission condition is
    the observable-evidence gate, but a future mutating target would be judged
    on an incomplete ``OracleState`` if these views drifted apart.
    """

    runtime = envelope.get("execution")
    runtime = dict(runtime) if isinstance(runtime, Mapping) else dict(envelope)
    grouping = envelope.get("grouping")
    grouping = (
        dict(grouping) if isinstance(grouping, Mapping) else dict(envelope)
    )
    return {
        "runtime": runtime,
        "audit": probe_audit_scenario(envelope),
        "grouping": grouping,
    }


def probe_audit_scenario(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Compose the offline-audit scenario: runtime fields plus private truth.

    The envelope handed to ``env.reset`` is truth-free by construction, so the
    private teacher-target audit must receive the audit truth ledger explicitly.
    Auditing against the runtime object judges the target on an incomplete
    ledger, which would silently pass any probe whose recovery target is itself
    a correction.
    """

    runtime = envelope.get("execution")
    scenario = dict(runtime) if isinstance(runtime, Mapping) else dict(envelope)
    audit = envelope.get("audit")
    truth = audit.get("truth") if isinstance(audit, Mapping) else None
    if isinstance(truth, Mapping):
        scenario.update(copy.deepcopy(dict(truth)))
    grouping = envelope.get("grouping")
    if isinstance(grouping, Mapping):
        for key in _AUDIT_GROUPING_FIELDS:
            if key in grouping:
                scenario.setdefault(key, copy.deepcopy(grouping[key]))
    return scenario


def _default_rank_one_proof(
    observation: Mapping[str, Any],
    *,
    preferred_action: Any,
    expert_actions: Sequence[Any],
) -> dict[str, Any]:
    """The same observable proof a natural DAgger label must carry."""

    return observable_rank_one_target_proof(
        observation,
        preferred_action=preferred_action,
        expert_actions=expert_actions,
    )


def _default_teacher_audit(
    observation: Mapping[str, Any],
    *,
    preferred_action: Any,
    env: Any,
    history: Sequence[Any],
    scenario: Mapping[str, Any],
    observable_evidence_passed: bool,
) -> dict[str, Any]:
    """The same private truth audit a natural DAgger target must survive."""

    return offline_teacher_target_audit(
        preferred_action=preferred_action,
        oracle_state=env.get_oracle_state(history),
        policy_observation=observation,
        scenario=scenario,
        env=env,
        observable_evidence_passed=observable_evidence_passed,
    )


def generate_recovery_probes(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    env: Any,
    expert_oracle: Any,
    state_class_for: Any,
    quotas: Mapping[str, int] | None = None,
    rank_one_proof_for: Any | None = None,
    teacher_target_audit_for: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drive one intervention per root and keep only the verified probe rows.

    The intervention executes through the ordinary environment path, so the
    resulting observation is a real ``PolicyObservation`` and the recovery
    target is chosen by the ordinary observable expert.  Hidden truth is never
    consulted to build or rank an intervention.

    ``state_class_for(observation, preferred_action)`` supplies the replay class
    for the post-intervention state; it is injected rather than imported so this
    driver stays testable without an environment.

    Roots are consumed in the order given.  A root is skipped when its
    intervention rule declines, when the environment does not fail, or when the
    classifier places the result in a neighbouring stratum -- the reserve margin
    above each floor exists to absorb exactly those discards.
    """

    targets = dict(quotas or RECOVERY_PROBE_ROOT_QUOTAS)
    unknown = sorted(set(targets) - set(RECOVERY_PROBE_STRATA))
    if unknown:
        raise ValueError(f"unsupported recovery-probe strata: {unknown}")

    rows: list[dict[str, Any]] = []
    used_roots: dict[str, set[str]] = {stratum: set() for stratum in targets}
    skipped: Counter[str] = Counter()
    attempts: list[dict[str, Any]] = []

    for scenario in scenarios:
        remaining = [
            stratum
            for stratum in RECOVERY_PROBE_STRATA
            if stratum in targets and len(used_roots[stratum]) < int(targets[stratum])
        ]
        if not remaining:
            break
        prepared = prepare_scenario_envelope(scenario)
        grouping = prepared["grouping"]
        root = str(grouping.get("physical_root_fingerprint") or "")
        family = str(grouping.get("scenario_family") or "")
        cardinality = int(grouping.get("error_cardinality") or 0)
        if not root:
            skipped["missing_physical_root"] += 1
            continue

        for stratum in remaining:
            if root in used_roots[stratum]:
                continue
            env.reset(prepared["runtime"])
            history: list[Any] = []
            observation = env.get_policy_observation(history)
            observation = (
                observation.as_dict()
                if hasattr(observation, "as_dict")
                else dict(observation)
            )

            # Drive a legal observable-expert prefix until the intervention's
            # precondition holds.  The previous priming loop fired the
            # intervention wherever an accepted correction existed with no open
            # candidate, which precedes the confirmation guard and yields
            # missing_precondition -- a different stratum entirely.
            setup_actions: list[dict[str, Any]] = []
            for _prefix_step in range(RECOVERY_PROBE_MAX_PREFIX_STEPS):
                if probe_intervention_precondition(observation, stratum=stratum):
                    break
                prefix_selection = select_observable_expert_actions(
                    policy_observation=observation, expert_oracle=expert_oracle
                )
                prefix_action = prefix_selection.preferred_action
                if prefix_action is None:
                    break
                prefix_result = env.step(prefix_action)
                prefix_output = (
                    prefix_result[1]
                    if isinstance(prefix_result, tuple) and len(prefix_result) > 1
                    else prefix_result
                )
                setup_actions.append(prefix_action)
                history.append(
                    {"action": prefix_action, "tool_output": prefix_output}
                )
                observation = env.get_policy_observation(history)
                observation = (
                    observation.as_dict()
                    if hasattr(observation, "as_dict")
                    else dict(observation)
                )
            if not probe_intervention_precondition(observation, stratum=stratum):
                skipped[f"{stratum}:precondition_never_reached"] += 1
                continue

            intervention = probe_intervention(observation, stratum=stratum)
            if intervention is None:
                skipped[f"{stratum}:rule_declined"] += 1
                continue

            _, tool_output = env.step(intervention)
            history.append({"action": intervention, "tool_output": tool_output})
            post = env.get_policy_observation(history)
            post = post.as_dict() if hasattr(post, "as_dict") else dict(post)

            # The shared observable path, identical to natural collection.
            # Calling the rule expert directly returns nothing at a verified
            # candidate and reads an unbounded history the learner cannot see.
            selection = select_observable_expert_actions(
                policy_observation=post, expert_oracle=expert_oracle
            )
            expert_actions = list(selection.actions)
            preferred_action = selection.preferred_action
            if preferred_action is None:
                skipped[f"{stratum}:no_expert_target"] += 1
                continue

            state_class = state_class_for(post, preferred_action)
            verification = verify_probe_stratum(
                post,
                preferred_action=preferred_action,
                state_class=state_class,
                scenario_family=family,
                error_cardinality=cardinality,
                expected_stratum=stratum,
            )
            attempts.append(
                {
                    "physical_root_fingerprint": root,
                    "expected_stratum": stratum,
                    "actual_stratum": verification.get("actual_stratum"),
                    "admitted": bool(verification.get("passed")),
                }
            )
            if not verification.get("passed"):
                skipped[f"{stratum}:landed_in_{verification.get('actual_stratum')}"] += 1
                continue

            # A probe target must clear the same two audits a natural DAgger
            # label clears.  Being the expert's first action is not a reason to
            # trust it: the observable proof must rank it first on
            # policy-visible evidence, and private truth must then find the
            # already-fixed target safe.
            # Training-decision evidence is verified, never asserted.  The
            # natural collector runs this environment check and only then sets
            # its flag; stamping a caller-supplied boolean would let a probe row
            # claim an admission step it never passed.
            training_decision_evidence_verified = False
            assertion = getattr(env, "assert_training_decision_evidence", None)
            if not callable(assertion):
                # An environment that cannot attest the evidence is not an
                # environment whose probes may be admitted.  Treating a missing
                # assertion as a pass would let a stubbed environment mint rows
                # that never cleared the check the flag claims.
                skipped[f"{stratum}:training_decision_evidence_unavailable"] += 1
                continue
            try:
                assertion(preferred_action)
            except ValueError:
                skipped[f"{stratum}:training_decision_evidence_failed"] += 1
                continue
            training_decision_evidence_verified = True

            prove_rank_one = rank_one_proof_for or _default_rank_one_proof
            audit_target = teacher_target_audit_for or _default_teacher_audit
            rank_one_proof = prove_rank_one(
                post,
                preferred_action=preferred_action,
                expert_actions=expert_actions,
            )
            if rank_one_proof.get("passed") is not True:
                skipped[f"{stratum}:rank_one_proof_failed"] += 1
                continue
            teacher_audit = audit_target(
                post,
                preferred_action=preferred_action,
                env=env,
                history=history,
                # Private truth must reach the audit.  The runtime envelope is
                # truth-free by construction, so auditing against it would judge
                # the target on an incomplete ledger.
                scenario=prepared["audit"],
                observable_evidence_passed=training_decision_evidence_verified,
            )
            if teacher_audit.get("passed") is not True:
                skipped[f"{stratum}:teacher_target_quarantined"] += 1
                continue

            rows.append(
                stamp_recovery_probe_row(
                    {
                        "example_id": f"probe_{stratum}_{root}",
                        "physical_root_fingerprint": root,
                        "scenario_family": family,
                        "error_cardinality": cardinality,
                        "policy_observation": post,
                        "probe_setup_actions": list(setup_actions),
                        "preferred_action": preferred_action,
                        "state_class": state_class,
                        "scenario_id": grouping.get("scenario_id"),
                    },
                    intervention=intervention,
                    expected_stratum=stratum,
                    verification=verification,
                    rank_one_proof=rank_one_proof,
                    teacher_target_audit=teacher_audit,
                    training_decision_evidence_verified=True,
                )
            )
            used_roots[stratum].add(root)
            break

    report = {
        "contract": RECOVERY_PROBE_CONTRACT,
        "scenarios_considered": len(scenarios),
        "root_quotas": dict(sorted(targets.items())),
        "roots_admitted": {
            stratum: len(roots) for stratum, roots in sorted(used_roots.items())
        },
        "quota_met": {
            stratum: len(used_roots[stratum]) >= int(targets[stratum])
            for stratum in sorted(targets)
        },
        "skipped": dict(sorted(skipped.items())),
        "attempts": attempts,
        "probe_support": audit_recovery_probe_support(rows),
    }
    report["passed"] = bool(
        report["probe_support"]["passed"]
        and all(report["quota_met"].values())
    )
    return rows, report
