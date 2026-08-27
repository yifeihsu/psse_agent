# Research BC0 success contract

## Decision

The primary research metric is **truth-audited final task success**, not the
episode's terminal label. An episode succeeds when the offline audit can prove
that:

1. every injected standard physical fault was identified by an accepted
   correction target;
2. every standard physical target finishes within its clean tolerance;
3. no true fault remains;
4. healthy measurements and case components remain preserved;
5. diagnostic-family and localization truth, when an explicit explanation-only
   diagnostic contract applies, match; and
6. the evaluator completed without an error.

The policy may finish with `resolved`, `operator_escalation`, or no terminal
marker. The actual outcome remains recorded for autonomy diagnostics, but it
does not alter the final-state truth audit. This is an offline scoring change;
it does not expose privileged truth to the policy, change the teacher, or
weaken the runtime termination contract.

The versioned contract is `truth_audited_final_task_success_v1`.

## KPI hierarchy

### Primary: truth-audited final task success

```
successful episodes / all evaluated episodes
```

This includes clean controls and faulted episodes. A 95% Wilson interval is
reported because the D0 and D1 suites are small.

### Fault-recovery subset

```
successful initially faulted episodes / all initially faulted episodes
```

This removes clean controls from the denominator and is the clearest episode-
level measure of repair ability.

### Drivers

- **Accepted true-fault target coverage**: accepted true targets divided by all
  true targets. This measures identification plus issuance of an accepted
  correction action; it does not prove the final value is clean.
- **Clean-tolerance target correction**: accepted true targets whose persisted
  final distance is within tolerance, divided by all true targets.

### Operational guardrail

**Safety-clean completion** retains the existing lifecycle requirement: either
strict resolved success or a validated post-correction handoff with no false
commit, false rollback, false finalization, loop, or evaluator error. It should
be reported beside the primary metric, not substituted for it.

**Strict resolved physical success** remains an autonomy/termination
diagnostic. It is intentionally not the primary research performance metric.

## Existing replay results

These figures are conservative backfills from the persisted strict audit and
validated handoff evidence in the repaired D0/D1 replays. They count an episode
only when existing evidence proves success. A fresh replay with the new code
will execute the terminal-independent audit on every episode and report native
evidence coverage.

| Evaluation | Truth-audited task success | Fault-only recovery | Safety-clean completion | Strict resolved |
|---|---:|---:|---:|---:|
| Published D0 | 13/21 (61.9%; 95% CI 40.9-79.2%) | 9/17 (52.9%; 95% CI 31.0-73.8%) | 12/21 (57.1%) | 4/21 (19.0%) |
| Published D1 / checkpoint 128 | 8/15 (53.3%; 95% CI 30.1-75.2%) | 8/15 (53.3%) | 7/15 (46.7%) | 0/15 |
| Checkpoint 192 | **9/15 (60.0%; 95% CI 35.7-80.2%)** | **9/15 (60.0%)** | **9/15 (60.0%)** | 0/15 |
| Checkpoint 256 | 8/15 (53.3%) | 8/15 (53.3%) | 8/15 (53.3%) | 0/15 |
| Checkpoint 320 | 8/15 (53.3%) | 8/15 (53.3%) | 8/15 (53.3%) | 0/15 |
| Checkpoint 64 | 6/15 (40.0%) | 6/15 (40.0%) | 4/15 (26.7%) | 0/15 |

D0 contains four no-error controls, all of which passed; this is why its
all-episode rate is higher than its fault-only recovery rate.

Checkpoint 192 is the strongest existing checkpoint under the aligned contract.
Its D1 family results are 3/3 parameter, 5/6 measurement-plus-parameter, and
1/6 multi-measurement. The corresponding published-checkpoint results are 3/3,
4/6, and 1/6. Multi-measurement recovery remains the dominant failure mode.

Target-level evidence tells the same story:

| Evaluation | Accepted true targets | Clean-tolerance corrected targets |
|---|---:|---:|
| Published D0 | 17/39 (43.6%) | 16/39 (41.0%) |
| Published D1 / checkpoint 128 | 18/38 (47.4%) | 16/38 (42.1%) |
| Checkpoint 192 | **22/38 (57.9%)** | **22/38 (57.9%)** |

Do not describe the accepted-target numerator as fully corrected: an accepted
action can remain outside clean tolerance. That distinction accounts for two
targets in the published D1 replay.

Correction evidence is also reported separately from the correction numerator.
The published D0 replay has complete final-distance evidence for 16/17 faulted
episodes and 38/39 true targets; published D1 has 13/15 episodes and 36/38
targets. Checkpoint 192 has complete evidence for all 15 episodes and all 38
targets. Missing distance evidence remains fail-closed and never earns a
corrected-target count.

## Checkpoint selection rule

Rank checkpoints lexicographically by:

1. truth-audited final task success;
2. fault-only recovery success;
3. clean-tolerance target correction;
4. safety-clean completion;
5. schema-valid action rate and expert action agreement;
6. fewer invalid actions, false commits, and loops.

This keeps physical task performance primary while preserving lifecycle safety
and action-quality diagnostics as explicit guardrails.
