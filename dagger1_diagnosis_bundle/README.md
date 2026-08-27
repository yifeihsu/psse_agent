# DAgger-1 Round-2 Failure Diagnosis

**Run:** `dagger1_6b76281c-20260816T035053Z-31541-3666363`
**Source commit:** `6b76281c` ("Re-freeze BC0 suite for expert recovery contract")
**Cluster:** NYU Torch · **Collected:** 2026-08-18 · **Revised:** 2026-08-18 (rev 2)

> **Revision 2 supersedes revision 1.** Peer review correctly identified that rev 1
> inverted the causal chain and overstated three claims. Corrections are marked
> **[CORRECTED]** throughout. The underlying evidence files are unchanged.

---

## One-paragraph summary

The DAgger-1 round-2 job chain did not crash. Job `15842429` (`d1r2-train`) ran two hours
on an H200 and exited `1` by design (`expected_exit_code: 1`); the aggregate job
`15842430` was then cancelled because its dependency could never be satisfied. No
traceback, no CUDA OOM, no evaluator error — which is why automated monitoring reported
the run as healthy while the pipeline was already dead.

**The terminal event was two quarantined commit targets.** The strict zero-quarantine
gate is cumulative and irreversible, so the collector stopped after batch 2 of 6, having
executed **151 of 477 planned episodes**. Every coverage and capacity figure below was
measured on that censored 31.7% of the schedule.

---

## Job chain, as it actually ran

| Job | Name | State | Exit | Elapsed | Limit |
|---|---|---|---|---|---|
| 15842427 | `d1r2-audit` | COMPLETED | 0:0 | 00:00:38 | 01:00:00 |
| 15842428 | `d1r2-diag` | COMPLETED | 0:0 | 01:34:11 | 12:00:00 |
| 15842429 | `d1r2-train` | **FAILED** | **1:0** | 02:00:03 | 1-00:00:00 |
| 15842430 | `d1r2-aggregate` | **CANCELLED** | 0:0 | 00:00:00 | 04:00:00 |

The 2h00m runtime is **not** a Slurm timeout — the limit was 24 hours.

---

## The terminal event  **[CORRECTED — this was rev 1's cause "C", ranked last]**

From `collection_stopping_report` in `failure_evidence.json`:

```
stopping_reason:        irreversible_truth_audit_quarantine
stopped_after_batch:    reserve-1-multi_measurement-r0
executed_batch_ids:     primary-r0, reserve-1-multi_measurement-r0          (2 of 6)
executed_episode_count: 151   of   planned_episode_count: 477
terminal_failure:       gate=offline_teacher_target_quarantine_summary
                        quarantined_rows=2
                        reason=strict_zero_quarantine_gate_is_cumulative_and_irreversible
unexecuted_batch_ids:   reserve-2-measurement+parameter-r0
                        repeat-multi_measurement-r1
                        repeat-measurement+parameter-r1
                        repeat-multi_measurement-r2
```

The two quarantined examples:

```
dagger_iter1_r0_9826886a46fd_step17__order126__replica0
dagger_iter1_r0_eab044f4881b_step21__order143__replica0
action_class = commit
reason       = candidate_source_target_outside_remaining_truth
```

Stopping immediately on quarantine is **correct** production behaviour — the gate cannot
be repaired by later rows. This is the first thing to resolve, and it requires forensic
extraction of both trajectories plus their private scenario envelopes to distinguish a
genuine wrong-target commit from a legitimate refinement misjudged by a
new-fault-membership rule.

---

## What the censored run does and does not establish

**Established:** the first two batches did not supply sufficient root support before two
quarantined commit targets triggered irreversible termination.

**NOT established [CORRECTED]:** that the complete 477-episode schedule would fail to
produce the missing roots or capacity. Rev 1 called this a "verified generation gap." It
is not verified. Four batches never ran, including the only remaining new-root batch
(`reserve-2-measurement+parameter-r0`) and three repeat-replica batches.

Observed shortfalls, all measured at 31.7% of schedule:

| Stratum / cell | Roots | Floor |
|---|---|---|
| `post_failure_no_candidate` | 2 | 10 |
| `unsupported_correction_recovery` | 1 | 10 |
| `parameter_route_complete_negative` (targeted cell) | 4 | 5 |

Note `parameter_route_complete_negative` is parameter-related and
`reserve-2-measurement+parameter-r0` never executed — that shortfall in particular has an
obvious unexecuted source.

**On the D0 comparison [CORRECTED].** Rev 1 argued the situations appear only in injected
counterfactual data because D0's ordinary rows carry zero of both predicates. That
comparison is weaker than presented: D0 ordinary collection is **expert-controlled BC0**,
whereas D1 is a **learner-controlled β-mixture**. Rare learner-failure states are expected
to be absent from expert-only paths, so their absence in D0 is not evidence about D1's
full schedule. The D0 counterfactual figures (263 and 152 roots) remain accurate as a
statement about what injection *can* produce — not as proof that natural collection cannot.

---

## The state-class plumbing defect — real, and independent of the stop

159 rows reach `state_class = invalid_precondition_recovery` through a **catch-all
branch** of `classify_state_example`, not because anything is invalid. Every disposition
source is empty:

```
transition_label.candidate_disposition   absent — key not present in label
candidate_state_summary                  {}
observation.candidate_disposition        None
```

...so `disposition` resolves to `None` and the function falls through. All 159 have
`candidate_lifecycle = VERIFIED_CANDIDATE`, `preferred_action.tool = commit_state`, and
`no_material_anomaly_remaining = false` — which reads as `accepted_partial_commit`.

**[CORRECTED] This does not skew the replay mixture.** Rev 1 claimed it did. The defaults
in `replay_buffer.py` are `accepted_partial_commit: 0.10`, `accepted_final_commit: 0.10`,
`invalid_precondition_recovery: 0.10` — all three equal, so the rows are not moving
between unequal nominal weights. The defect still corrupts state-class audits,
class-conditioned reporting, deterministic selector behaviour, and potentially bucket
allocation. It did **not** cause this run to stop.

A related but insufficient bug sits at `rollout_collector.py:390`: the guard
`if target_disposition is None and preferred is None` skips the
`label["candidate_disposition"]` fallback whenever a preferred action exists. Fixing that
guard alone changes nothing, because the key is absent from the label too.

**Recommended fix** (per review): derive commit class from policy-visible verification
evidence rather than exposing private disposition — no verified candidate or invalid
lifecycle → `invalid_precondition_recovery`; verified + observable final disposition →
`accepted_final_commit`; verified + observable partial acceptance →
`accepted_partial_commit`; verified requiring rollback → `rejected_candidate_recovery`.
Share one observable helper across expert commit/rollback reconstruction, state-class
assignment, and rank-one target proof.

---

## Corrections to revision 1

| Rev 1 claim | Status |
|---|---|
| "Generation gap is the blocker; verified" | **Wrong.** Measured at 31.7% of schedule; the quarantine was the terminal event. |
| "Mislabelling skews the replay mixture" | **Wrong.** All three class weights are 0.10. |
| "`invalid_precondition_repair` has a 10-root floor and will block every run" | **Wrong.** It has no minimum-root floor; the map holds six entries and this is not one. Zero firing blocks nothing. |
| "Retarget 1630 → 1521 — cheapest fix" | **Withdrawn.** The 1521 figure is a censored-checkpoint feasibility, not the schedule's true capacity. |
| Classifier gates all terminal branches on `previous_failed` | **Stands.** Code-structure fact, unaffected by censorship. |
| 159 rows mislabelled via empty disposition | **Stands.** Row-level fact, unaffected by censorship. |
| D0 counterfactual carries 263 / 152 roots | **Stands as measured**, but does not support the inference rev 1 drew from it. |

---

## What's in this bundle

```
README.md                          this file (rev 2)
OPTIONS.md                         options for the generation gap, with review response
analysis/
  reproduce_analysis.py            regenerates every figure (read-only)
  analysis_output.txt              captured output
evidence/
  sacct_job_states.txt             Slurm accounting for all four jobs
  slurm_logs/                      stdout/stderr for audit, diagnostic, training
  job_scripts/                     the four submitted scripts + input audit
  failed_collection/
    failure_evidence.json          90 KB — authoritative gate evidence
    SHA256SUMS                     integrity for the failed-collection artifacts
```

**Not included (too large, still on the cluster):**
`diagnostic.all_visited_rows.jsonl` (114 MB) and
`diagnostic.candidate_recovery_rows.jsonl` (31 MB), under the run's
`training_beta025.failed-collection/`.

## Reproducing

```bash
python3 analysis/reproduce_analysis.py > analysis_output.txt
```

Read-only, streams the large files, flat memory.

## Reading order

1. This README — the terminal event and what the censored run can/cannot show.
2. `analysis/analysis_output.txt` §1 — gate-by-gate numbers.
3. `OPTIONS.md` — the counterfactual question, deferred pending an uncensored run.
4. `evidence/failure_evidence.json` — to check any figure directly.
