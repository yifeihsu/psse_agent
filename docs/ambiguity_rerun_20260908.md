# Ranked candidate testing under parameter ambiguity: re-run of the full pipeline (2026-09-08)

Research-only results; nothing here is release evidence.

## Decision that drove the re-run

The 2026-09-07 pipeline exposed two problems with how parameter roots were
handled. The development suite had been drawn at the detection threshold
(top-to-runner-up multiplier ratio 1.0) while the environment and the
expert aggregate used the production dominance threshold (1.2), so roughly
five of sixteen development parameter roots were unfixable on the first pass
by design. And the first DAgger round had learned to open topology roots
with a meter edit, because context-fetch labels at exactly those states were
quarantined by the rank-one proof and no environment guard stopped a
masking meter correction under branch dominance.

The user's position: realistic ambiguity between adjacent lines must stay
in the test set even when the rule-based teacher is expected to get some of
it wrong, while the training set may exclude cases the teacher cannot
label reliably. The chosen design is **ranked testing with escalation
fallback**:

- Under ambiguity (ratio below 1.2) the parameter context offers the top
  two ranked lines in rank order instead of standing down.
- The expert tests the candidates in rank order under verification. When
  every candidate carries a verification-rejected hypothesis it escalates
  with `operator_escalation:ambiguous_branch_candidates`, naming the lines.
- The strict audit credits that handoff when the true branch is among at
  most two named candidates and no healthy component was modified.
- Mechanical fixes: the meter route is blocked while the anomaly is
  branch-dominant until both branch families hold a rejected hypothesis; a
  structural context fetched while three-phase screening is pending no
  longer counts as a completed screen; the rank-one proof accepts a
  non-correction ladder-first proposal as a deterministic label.
- Training rounds are drawn at threshold 1.2; the development set at 1.0
  with the true line admitted anywhere in the top two of the deployed
  ranking. Each development root records its stratum, and the expert is
  rolled out on the same roots as the ceiling.

Commits: cb0c0e1 (environment, expert, audit, proof, generator), 4d3a7ee
(suite, summaries, expert ceiling, pipeline cell), f8522a5 (misranked
stratum), 05a091a (checkpoint digest follows a symlinked adapter), 3ad132d
(summary count fix).

## What ran

Directory `/scratch/yx3882/research_full_pipeline_20260908` on Torch. The
expert aggregate (418 roots) and the BC0 adapter were linked from the
2026-09-07 run; the suite was redrawn and both DAgger rounds re-ran.

| Stage | Outcome |
| --- | --- |
| Suite | 122 + 122 training roots at threshold 1.2; 160 development roots at threshold 1.0, rank allowance 2. Development parameter: 11 dominant, 4 ambiguous, 1 misranked (ratio 1.03, true line second). Measurement+parameter: 14 dominant, 2 ambiguous |
| Round 1 collection (BC0) | 122 episodes, label yield 0.92 (previous run 0.82), 536 safe rows, 47 quarantined; tool disagreement with the expert 13% |
| Round 1 training | 268 steps on a 1072-row mixture; preempted once at step 178 and resumed from checkpoint 176 |
| Round 2 collection (R1) | label yield 0.905, 525 safe rows, 55 quarantined; tool disagreement 9% |
| Round 2 training | 263 steps on a 1050-row mixture |
| Evaluation | 160 development roots, 40-step budget, production case loader; BC0, R1, and the expert in round 1, R1 and R2 in round 2 (R1 scored identically in both) |

## Results

Truth-audited task success on the 160 development roots. Measurement+HIF is
scored as fault-identified rather than resolved, so the reachable ceiling
is 152.

| Adapter | Success | Rate | Invalid actions | False commits | Loops |
| --- | --- | --- | --- | --- | --- |
| BC0 | 144/160 | 0.900 | 47 | 27 | 12 |
| R1 | 147/160 | 0.919 | 27 | 18 | 0 |
| R2 | 146/160 | 0.912 | 39 | 31 | 1 |
| Expert (ceiling) | 147/160 | 0.919 | 0 | 0 | 0 |

Per family:

| Family | BC0 | R1 | R2 | Expert |
| --- | --- | --- | --- | --- |
| harmonic | 16/16 | 16/16 | 16/16 | 16/16 |
| hif | 14/16 | 13/16 | 14/16 | 13/16 |
| measurement | 16/16 | 16/16 | 16/16 | 16/16 |
| measurement+hif | 0/8 | 0/8 | 0/8 | 0/8 |
| measurement+parameter | 13/16 | 16/16 | 14/16 | 16/16 |
| measurement+topology | 12/12 | 12/12 | 12/12 | 12/12 |
| multi_measurement | 10/12 | 11/12 | 11/12 | 11/12 |
| no_error | 8/8 | 8/8 | 8/8 | 8/8 |
| parameter | 15/16 | 15/16 | 15/16 | 15/16 |
| telemetry_no_disturbance | 8/8 | 8/8 | 8/8 | 8/8 |
| three_phase_unbalance | 16/16 | 16/16 | 16/16 | 16/16 |
| topology | 16/16 | 16/16 | 16/16 | 16/16 |

Per parameter-ranking stratum (parameter and measurement+parameter roots):

| Stratum | Roots | BC0 | R1 | R2 | Expert |
| --- | --- | --- | --- | --- | --- |
| dominant (true line first, ratio >= 1.2) | 25 | 22 | 25 | 23 | 25 |
| ambiguous (true line first, ratio < 1.2) | 6 | 6 | 6 | 6 | 6 |
| misranked (neighbour outranks the true line) | 1 | 0 | 0 | 0 | 0 |

Comparison with the 2026-09-07 run (same roots for D0 and BC0, different
suite draw): BC0 131/160, R1 117/160, R2 118/160, with topology 16 to 0 to 0
and measurement+topology 12 to 3 to 3. Under the new teacher R1 matches the
expert on every family and the topology regression is gone.

## Reading the strata

- **Ambiguous roots are solved.** All six ambiguous-ranking roots (ratios
  1.02 to 1.12) were corrected on the first candidate by every adapter and
  by the expert. Ranked testing never had to escalate: no bounded handoff
  occurred in either round, so the audit credit for the ambiguous
  escalation was not exercised on this suite.
- **The misranked root is the rule-based ceiling.** On the one root where
  the neighbouring line outranked the true line (ratio 1.03), the expert
  corrected the neighbour, verification accepted that correction because
  it fits the measurements, and the audit failed on the modified healthy
  line. Every adapter did the same. A local probe on freshly drawn roots
  showed the same behaviour for a misranked root at ratio 1.30, so this is
  inherent to multiplier ranking plus residual verification, not to the
  suite draw.
- **The dominant stratum separates the students.** BC0 22/25 and R2 23/25
  fail measurement+parameter roots with a commit-rollback loop: the
  parameter correction is committed, rolled back, and retried. R1 solved
  all 25.

## Where the remaining failures are

Roots below the ceiling for R1 (5 of 152): two HIF roots where the
multi-scan estimator localizes outside tolerance for every policy including
the expert, one further HIF root the expert and R1 miss but BC0 and R2
locate, one multi-meter root nobody finishes within the budget (the expert
escalates after 35 steps with faults remaining), and the misranked
parameter root.

R2 versus R1: R2 recovers one HIF root and loses two measurement+parameter
roots to the commit-rollback loop, with more invalid actions (39 versus 27)
and more false commits (31 versus 18, all but one on measurement+parameter).
Round 2 collected with a student that already agreed with the expert on 91%
of states, so its 525 rows added little new supervision while the extra
pass at lr 3e-5 moved the adapter slightly. The false-commit loop on
measurement+parameter is the one student behaviour the expert never
exhibits and is the natural target for the next round's collection.

## Label quarantine

Round 1 quarantined 47 learner-state rows (42 rank-one proof failures, 5
teacher audit failures), against roughly 100 in the previous run. Topology
lost one context-fetch label instead of 29 and measurement+topology 9
instead of 17. The remaining proof failures sit on measurement+parameter
(15), measurement+topology (9), multi-meter (7), and parameter (7).

## Files

- Cell: `research/hpc/full_pipeline_20260907/` (README describes the
  ambiguity design and the strata).
- Results: `out/pipeline_summary.json`, `out/r1/round_summary.json`
  (with `expert_eval.json` beside the adapter evaluations),
  `out/r2/round_summary.json` in the pipeline directory.
- Tests: `psse_env/test_ambiguity_handoff.py`, generator and cell tests.
