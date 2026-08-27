# Implemented fixes

Against the review's "next source revision should contain only" list.
Branch: `codex/expert-hard-gate-fix`. All changes are uncommitted working-tree edits.

| # | Review item | Status |
|---|---|---|
| 0 | Forensic resolution of the two quarantines (prerequisite) | **Done** — Case A established |
| 1 | Regression-backed resolution of the two quarantined commits | **Partial** — evidence + fixtures landed; the environment change needs a design decision (below) |
| 2 | Observable state-class correction | **Done**, with regression test |
| 3 | Analysis-only complete-schedule mode | **Done**, with two tests |
| 4 | Clearer Slurm result classification | **Done**, with eight tests |

---

## 0. Forensic finding — Case A, "already corrected and retired"

Full output in `quarantine_forensics/case_analysis.txt`; scripts alongside it.

Both quarantined commits share one pattern: a **late broad correction whose
suspect group mixes one genuinely-remaining fault with targets already accepted
and retired from the private remaining-truth ledger.**

| Episode | Accepted & retired | Late correction | Genuinely new |
|---|---|---|---|
| `r0_9826886a46fd_episode126` step 17 | `[92]`, `[97]` | `[88, 92, 97]` | only `88` |
| `r0_eab044f4881b_episode143` step 21 | `[17]`, `[27]`, `[26]` | `[17, 19, 26, 27]` | only `19` |

Both audits are identical:

```json
{"action_class": "commit",
 "checks": {"candidate_exists": true, "candidate_verified": true,
            "candidate_source_truth_evidence_complete": true,
            "observable_evidence_gate_passed": true,
            "candidate_truth_safe_to_commit": false},
 "reason_codes": ["candidate_source_target_outside_remaining_truth"],
 "passed": false}
```

**Answering the review's question 9 directly: the audit detected only
target-membership mismatch, not physical harm.** No harm check fails.

**This is not Case B.** A legitimate refinement must touch *only* previously
accepted targets; both corrections add a genuinely new one (`88`, `19`)
alongside retired ones. No `accepted_target_refinement` marker is present in
either action's arguments.

**The defect is in the observable path, not the audit.** In episode 143 the
quarantined commit was `executed_by = expert`, and in *both* episodes the
teacher's rank-one target was the commit. The observable expert cannot see
remaining truth, so it commits a verified candidate the private audit rejects.
Of the review's four Case-A interventions, the evidence points squarely at
**"already exhausted targets are removed from supported inventory"** — retired
indices stay correctable, so the over-broad suspect group can be formed at all.

### Why the environment change is not implemented here

That intervention changes what corrections the observable expert may propose —
teacher semantics, not a bug fix. The review lists four candidate intervention
points and the evidence narrows it to one, but choosing where to enforce it
(context provider inventory, correction validation, or pre-commit candidate
rejection) is a design decision with different downstream consequences for the
expert's rank-one proof. It should be made deliberately, not inferred.

Both physical roots are preserved as exact regression fixtures:
`r0_9826886a46fd` and `r0_eab044f4881b`, with full trajectories in
`quarantine_forensics/full_trajectories.txt`.

---

## 2. Observable state-class correction

`psse_env/dagger/rollout_collector.py`

Added `observable_candidate_verified()` and `observable_commit_class()` as the
shared observable helper, contract `dagger1_observable_commit_class_v1`.
`classify_state_example` now delegates its `commit_state` branch to it instead
of falling through to `invalid_precondition_recovery` when disposition is
absent.

Mapping, per the review:

```
no verified candidate / invalid lifecycle  → invalid_precondition_recovery
verified + no_material_anomaly_remaining   → accepted_final_commit
verified + anomalies remaining             → accepted_partial_commit
verified + declared REJECT                 → rejected_candidate_recovery
```

An explicitly declared disposition still wins, so existing behaviour is
unchanged wherever disposition is actually populated. The 159 observed rows
(`VERIFIED_CANDIDATE`, `commit_state`, `no_material_anomaly_remaining: false`,
no disposition anywhere) now classify as `accepted_partial_commit`.

Test: `psse_env/test_production_mode.py::
test_commit_class_uses_observable_evidence_when_disposition_absent`, built from
the exact observed row shape, covering all four branches plus declared-
disposition precedence.

---

## 3. Analysis-only complete-schedule mode

`psse_env/dagger/collect_dagger1.py`

New `--analysis-only-complete-schedule` flag and a matching
`analysis_only_complete_schedule` parameter on
`collect_dagger1_rollout_schedule`.

* Runs every predeclared batch to exhaustion — neither the first quarantine nor
  a passing checkpoint stops it.
* Quarantines are still recorded; the run is still a failure.
* Report carries `analysis_only: true`, `training_eligible: false`,
  `stopping_reason: analysis_only_complete_schedule_exhausted`, and `passed`
  is forced `False` even on a passing checkpoint.
* CLI requires `--collection-pass training` and `--failed-collection-dir`, and
  forces `--require-recommended-target` so the failure-evidence path is taken
  and production outputs can never be published.

**Production output is byte-identical.** The extra `first_quarantined_batch`
key is added to `terminal_failure` only in analysis mode, so the hash-bound
strict payload is unchanged.

Tests: `test_analysis_only_mode_runs_complete_schedule_past_quarantine` and
`test_analysis_only_mode_does_not_stop_on_a_passing_checkpoint`.

Usage:

```bash
python scripts/collect_dagger1_recovery.py \
  ... same arguments as the strict run ... \
  --collection-pass training --beta 0.25 \
  --failed-collection-dir "$D1_DIR/analysis_beta025.failed-collection" \
  --analysis-only-complete-schedule
```

---

## 4. Slurm result classification

New `psse_env/dagger/collection_result.py` + `scripts/classify_collection_result.py`.

Three outcomes with distinct exit codes so a scheduler can branch without
parsing stdout:

| Classification | Exit | Condition |
|---|---|---|
| `STRICT_GO` | 0 | exit 0, production rows **and** manifest present, no failure bundle |
| `STRICT_NO_GO` | 20 | exit 1, well-formed failure bundle, no production outputs |
| `INFRASTRUCTURE_FAILURE` | 1 | anything else — crash, OOM, timeout, unreadable evidence, inconsistent state |

Verified against the real round-2 bundle:

```
[STRICT_NO_GO] fail-closed collection: 5 gate(s) rejected the run
[deterministic_collection_selection, independent_root_support,
 offline_teacher_target_quarantine_summary, round1_replay_capacity,
 targeted_state_coverage] | episodes 151/477 (31.7% of schedule) |
 stopping_reason=irreversible_truth_audit_quarantine | quarantined_rows=2
```

That one line is what monitoring should have emitted for two days instead of
"healthy". It reports the failed gate names and — critically — surfaces the
31.7% schedule execution that made every coverage figure censored.

Recommended job-script tail (replaces the bare `test -s` assertions, which
cannot distinguish a fail-closed NO-GO from a crash):

```bash
set +e
"$PY" scripts/collect_dagger1_recovery.py ... ; COLLECT_RC=$?
set -e
"$PY" scripts/classify_collection_result.py \
  --exit-code "$COLLECT_RC" \
  --production-output "$D1_DIR/training_beta025.jsonl" \
  --production-manifest "$D1_DIR/training_beta025.jsonl.manifest.json" \
  --failed-collection-dir "$D1_DIR/training_beta025.failed-collection"
# exits 0 = GO (aggregate may proceed), 20 = fail-closed NO-GO, 1 = infrastructure
```

Tests: `psse_env/dagger/test_collection_result.py`, 8 cases.

---

## Test status

```
psse_env/test_production_mode.py            passed
psse_env/dagger/test_training_view.py       passed
psse_env/dagger/test_review_regressions.py  passed
psse_env/dagger/test_collect_dagger1.py     36 passed, 2 failed *
psse_env/dagger/test_collection_result.py   8 passed
psse_env/examples/test_generate_round0_aggregate.py   1 failed *
```

\* The three failures are **pre-existing and platform-specific**, confirmed by
re-running the same suites with these changes stashed: `_fsync_directory` calls
`os.open()` on a directory, which raises `PermissionError` on Windows. They are
unrelated to these changes and would not occur on the cluster.

## Not done, deliberately

Per the review: no counterfactual rows added, no reduction of the 1630 target,
no weakening of the root floors or the zero-quarantine gate, no aggregate
ingestion, no Round-1 SFT, no DAgger iteration 2. The failed-collection evidence
is untouched.
