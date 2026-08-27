# Counterfactual augmentation — options, and why the decision is deferred

> **STATUS (rev 2): DEFERRED. Do not implement any option below yet.**
>
> Revision 1 recommended Option B as the unblocking path. That recommendation is
> **withdrawn**. The run that motivated it executed 151 of 477 planned episodes (2 of 6
> batches) before two quarantined commit targets triggered irreversible termination.
> The shortfalls that made augmentation look necessary were measured under that
> censorship and may not survive a complete schedule.
>
> This document is retained because it remains the right analysis **if** an uncensored
> run still misses the floors — see "When this becomes live" at the end.

---

## Immediate sequence (supersedes rev 1's sequencing)

1. **Resolve the two quarantined commits.** Forensically extract both trajectories and
   their private scenario envelopes; determine whether each is a genuine wrong-target
   commit or a legitimate refinement misjudged by a new-fault-membership rule. Do not
   weaken the zero-quarantine gate until this is answered. Add both physical roots as
   regression fixtures.
2. **Fix state-class semantics** from policy-visible verification evidence (see
   `README.md`). Add a regression test on the 159-row pattern.
3. **Add an analysis-only complete-schedule mode** — same β=0.25, same root-local seeds,
   same 477-episode schedule, continuing past quarantine *for analysis only*: every output
   training-ineligible, no production JSONL or manifest, never accepted by aggregate
   ingestion. This answers the generation-gap question without weakening production safety.
4. **Regenerate hash-bound artifacts and rerun** the original strict β=0.25 protocol with
   β, row target 300–600, aggregate target 1630, root floors, zero-quarantine requirement,
   and the 477-episode maximum all **unchanged**.

**Do not** reduce the aggregate target to 1521. That figure is the largest feasible view
at an early-stopped checkpoint, not the schedule's true capacity — substituting it would
conflate a censored measurement with a protocol amendment.

---

## What the unexecuted batches can and cannot supply

A subtlety worth carrying into the analysis-only run. The four unexecuted batches are:

```
reserve-2-measurement+parameter-r0     ← r0: introduces NEW physical roots
repeat-multi_measurement-r1            ← r1/r2: additional REPLICAS of existing roots
repeat-measurement+parameter-r1
repeat-multi_measurement-r2
```

with `maximum_rollout_replicas_by_family: {measurement+parameter: 2, multi_measurement: 3,
parameter: 1}`.

Only one of the four adds new physical roots. The other three re-run roots already in the
pool under different seeds. That still helps the floors — the floors count *distinct roots
bearing a row in that stratum*, so a second trajectory from an existing root can newly
qualify it — but the mechanism differs, and the two effects should be reported separately.

**Rough extrapolation, flagged as speculative.** At 151 episodes the raw stratum labels
were 3 and 3. Scaling linearly to 477 (×3.16) gives ~9–10 raw labels per stratum, against
a floor of **10 distinct roots**, and roots ≤ rows. That lands close enough to the boundary
that neither "it will obviously fill" nor "it obviously cannot" is defensible from current
data. This is precisely why step 3 is worth building rather than assuming an outcome.

---

## The constraint that would force a choice

If an uncensored run still misses the floors, the blocker is this eligibility chain
(`psse_env/dagger/rollout_collector.py:1136`):

```python
if collection_role == "diagnostic":      → diagnostic_beta_zero_not_training_eligible
elif state_origin != "learner_policy":   → not_learner_visited_state
elif recovery_stratum not in _DAGGER1_PRODUCTION_RECOVERY_STRATA:
                                         → not_recovery_state
elif preferred_action is None:           → missing_expert_target
...
```

Counterfactual rows die at check 2, and that check is load-bearing: DAgger's justification
is that the learner trains on the distribution *its own policy* induces. D0 already
respects this — `generate_round0_aggregate.py:2941` writes `eligible_rows` and
`auxiliary_rows` to separate files.

## The four options

| Option | Effort | What it costs | Verdict |
|---|---|---|---|
| **A** · label injected rows `learner_policy` | trivial | Falsifies provenance; breaks the DAgger guarantee invisibly | **Reject** |
| **B** · new `counterfactual_injection` origin | moderate | Changes the method to DAgger + counterfactual augmentation | **Deferred** — not yet shown necessary |
| **C** · amend gates only, rows stay auxiliary | low | Floors satisfied by rows that never train anything | **Reject** |
| **D** · revisit the floors | low code, high review | Contract change requiring sign-off | **Live question** |

**[CORRECTED] Option D's supporting argument, restated.** Rev 1 claimed
`invalid_precondition_repair` was a declared stratum with an unfillable 10-root floor.
**That was wrong** — it has no minimum-root floor at all. The floor map holds six entries
(`multi_measurement_safe_handoff`, `post_failure_no_candidate`,
`sequential_measurement_parameter_recovery`, `unsupported_correction_recovery` at 10;
`premature_commit_recovery`, `premature_escalation_recovery` at 5) and
`invalid_precondition_repair` is not among them. It is a production-eligible residual
class, and its zero firing blocks nothing. No contract change is needed for it.

D's remaining argument stands on its own merits: floors on `post_failure_no_candidate` and
`unsupported_correction_recovery` constrain situations DAgger is *designed to eliminate*,
so the requirement tightens as the learner improves. A defensible reformulation replaces
absolute on-policy occurrence floors with:

* complete execution of the finite natural schedule;
* zero unsafe labels;
* complete supervision for every naturally occurring opportunity;
* a separate ten-root **intervention-suite** requirement.

That guarantees recovery competence through a dedicated stress corpus without penalising
an improved learner for making fewer mistakes.

---

## When this becomes live

Only after an uncensored, zero-quarantine complete schedule.

**If strict collection passes:** none of this is needed. Proceed to aggregate → provenance
gate → tokenizer/mask validation → one real forward/backward → tiny-overfit → Round-1
training → development-holdout evaluation, and only then design DAgger iteration 2.

**If zero quarantine but the full schedule still misses rare floors:** a modified Option B
becomes justified, as a *separate auxiliary corpus* rather than a redefinition of DAgger:

```
dataset_source = dagger_counterfactual_recovery
state_origin   = counterfactual_injection
```

with separate JSONL and manifest; separate root-disjointness checks; separate provenance
and generator identity; separate replay quota; no claim of learner visitation; auxiliary
roots never counted toward natural on-policy support; explicit natural-versus-injected
coverage reporting; counterfactual actions passing the same observable rank-one and private
safety audits; and disjointness from development and frozen evaluation roots.

**If coverage passes but only replay capacity fails:** first check whether additional
natural roots are economical. Amend 1630 only when the full schedule ran, all safety and
root-support gates passed, maximum feasible view was recomputed from final data, and the
revised size is recorded as a deliberate protocol amendment.

---

## Mechanical scope (retained for the Outcome-B case)

Generator API, reusable as-is:

```python
generator = CounterfactualGenerator(env=env, expert_oracle=oracle)
rows = generator.generate_from_current(
    injected, root_scenario_id=..., physical_root_fingerprint=...)
```

| # | Work | Notes |
|---|---|---|
| 1 | Lift `_bounded_injected_actions` into a shared module | `generate_round0_aggregate.py:458`, ~20 lines |
| 2 | Wire generator into `collect_dagger1.py` | `--counterfactuals-per-root`; mirror `generate_round0_aggregate.py:976`; dedicated derived RNG |
| 3 | New `state_origin` + eligibility branch | stratum classifier needs no change |
| 4 | **Schema bridge — largest chunk** | D0 counterfactual rows lack `recovery_stratum`, `state_origin`, `collection_*`, `iteration`, `step`, `transition_label`, `next_valid_actions`, `terminal_outcome`, `supervision_policy`, `recovery_label_contract`, `production_label_ineligibility_reason`, `observable_rank_one_target_proof`, `offline_teacher_target_audit` |
| 5 | Gates admit the new origin | reporting natural vs injected separately |
| 6 | Manifests + SHA256SUMS | pipeline is hash-bound throughout |
| 7 | Tests | `test_collect_dagger1.py` (2208 lines), `test_dagger1_builders.py` (1855) |

**Determinism:** derive the counterfactual RNG from `--seed` so reruns stay byte-identical.
**Root safety:** branches clone from the same fresh root, so fingerprints stay in the fresh
set. **`executed_by`:** currently `"expert"`/`"model"`; injected rows need a third value.

**Known injector gap:** only two of the four unsupported codes fire in D0 —
`correction_not_supported_by_current_context` (164) and `correction_route_not_actionable`
(48). `parameter_scans_missing` may be absent deliberately; `diagnostic_suite.py:152` says
the design "must prevent the old numerical `parameter_scans_missing` path."
