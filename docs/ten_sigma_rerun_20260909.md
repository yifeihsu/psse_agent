# Full pipeline from a 526-root aggregate with ten-sigma meter errors (2026-09-09)

Research-only results; nothing here is release evidence.

## What changed from the 2026-09-08 run

- **Meter errors at ten sigma or more.** Every injected meter error, in the
  corpus rows and in the composed overlays, is lifted to at least ten noise
  sigmas (`MEASUREMENT_ERROR_MIN_SIGMA`). The tracked corpus had centred
  near ten sigma and reached down to five, which is where the multi-meter
  near-threshold masking case lived.
- **HIF localization tolerance 0.15** of line length (was 0.10).
- **Everything regenerated.** A 548-root plan produced a 526-root expert
  aggregate (4966 rows: train 3722, validation 740, test 504) and a fresh
  BC0; the previous runs had reused a 418-root aggregate. The suite is
  drawn as before: 122 + 122 training roots at threshold 1.2, 160
  development roots at 1.0 with rank allowance 2.

Development strata this time: parameter 11 dominant, 3 ambiguous, 2
misranked; measurement+parameter 13 dominant, 1 ambiguous, 2 misranked. All
four misranked roots have ratios between 1.02 and 1.18 with the true line
ranked second.

Commits: 8fd383c (ten-sigma floor, tolerance, plan), 252839d (argument
parsing fix). Cluster directory `research_full_pipeline_20260909`.

## Results

Truth-audited task success on the 160 development roots, 40-step budget,
production case loader. Measurement+HIF is scored fault-identified rather
than resolved, so the reachable ceiling is 152.

| Adapter | Success | Rate | Invalid actions | False commits | Loops |
| --- | --- | --- | --- | --- | --- |
| BC0 | 125/160 | 0.781 | 95 | 13 | 25 |
| R1 | 146/160 | 0.912 | 22 | 10 | 2 |
| R2 | 147/160 | 0.919 | 12 | 11 | 0 |
| Expert (ceiling) | 147/160 | 0.919 | 6 | 0 | 1 |

R1 scored 147 when re-evaluated as round 2's student: one multi-meter root
flips between runs of the same adapter (greedy decoding on different GPU
nodes), so R1 and R2 are at parity and both sit on the expert's ceiling.

| Family | BC0 | R1 | R2 | Expert |
| --- | --- | --- | --- | --- |
| harmonic | 16/16 | 16/16 | 16/16 | 16/16 |
| hif | 16/16 | 15/16 | 15/16 | 15/16 |
| measurement | 16/16 | 16/16 | 16/16 | 16/16 |
| measurement+hif | 0/8 | 0/8 | 0/8 | 0/8 |
| measurement+parameter | 1/16 | 14/16 | 14/16 | 14/16 |
| measurement+topology | 12/12 | 12/12 | 12/12 | 12/12 |
| multi_measurement | 12/12 | 11/12 | 12/12 | 12/12 |
| no_error | 8/8 | 8/8 | 8/8 | 8/8 |
| parameter | 4/16 | 14/16 | 14/16 | 14/16 |
| telemetry_no_disturbance | 8/8 | 8/8 | 8/8 | 8/8 |
| three_phase_unbalance | 16/16 | 16/16 | 16/16 | 16/16 |
| topology | 16/16 | 16/16 | 16/16 | 16/16 |

| Stratum (parameter and measurement+parameter roots) | Roots | BC0 | R1 | R2 | Expert |
| --- | --- | --- | --- | --- | --- |
| dominant (true line first, ratio >= 1.2) | 24 | 5 | 24 | 24 | 24 |
| ambiguous (true line first, ratio < 1.2) | 4 | 0 | 4 | 4 | 4 |
| misranked (neighbour outranks the true line) | 4 | 0 | 0 | 0 | 0 |

## What the two changes bought

- **Multi-meter is solved at the ceiling.** The expert is 12/12 (11/12 on
  the previous suite, where a fifth meter at 1.6 sigma was masked by a
  joint re-fit of the other four). With every error at ten sigma or more
  the residual test localizes each meter after the others are fixed.
- **HIF: 15/16 for everyone but BC0.** On this draw the wider tolerance was
  not what decided it: fifteen roots have alpha errors of 0.088 or less and
  the sixteenth is off by 0.162, which fails at 0.10 and at 0.15 alike.
  Line, phase, and resistance are right on all sixteen (resistance within
  3.4%); the miss is the flat score valley along the line seen before. BC0
  passes that root only because the tie-break landed differently in its
  run.

## The BC0 checkpoint

BC0 is weak on parameter families because checkpoint selection by
validation loss picked step 320 of 931, a third of one epoch: the
validation loss bottomed at 0.008 there and drifted to 0.012 afterwards
while training loss kept falling. That BC0 skips the three-phase and
harmonic screening on parameter roots, its correction is refused because
screening is still pending, and the evaluator's repeated-failure breaker
ends the episode on the retry.

A side evaluation of the end-of-epoch checkpoint on the same roots:

| BC0 checkpoint | Success | Parameter | Meas.+param | Multi-meter | Topology | Invalid | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 320 (selected) | 123/160 | 3/16 | 2/16 | 12/12 | 15/16 | 86 | 22 |
| 931 (final) | 142/160 | 14/16 | 13/16 | 8/12 | 16/16 | 49 | 5 |

The full-epoch adapter matches the expert on parameter families and gives
back four multi-meter roots. Selecting by validation loss on the 128-row
subset is not aligned with task success; the next BC0 should keep the
final checkpoint. Round 1 repaired the gap in this run: DAgger took the
student from 125 to 146 with a 25% tool disagreement against the expert
during collection, and round 2 collected at 7% disagreement.

## Misranked roots are the rule-based ceiling

On all four misranked roots the expert corrected the higher-ranked
neighbour first, verification accepted that correction with 78 to 91
percent global progress, it was committed, and healthy meters were then
rewritten to absorb the residual the wrong line left behind. The
escalation fallback never fires because verification never rejects the
neighbour. Every adapter reproduces this. Together with the flat-valley HIF
root and the measurement+HIF family, these are the only roots R2 misses.

## Pipeline record

| Stage | Outcome |
| --- | --- |
| Stage 0 | 526 roots, 4966 rows in about 110 minutes on the CPU partition; suite from 732 training and 480 development candidates |
| BC0 | 931 steps in 3 h 47 min, one preemption resumed from checkpoint 384 |
| Round 1 | collection 50 min, yield 0.92, 582 rows, 49 quarantined; training 291 steps on 1164 rows in 1 h 47 min; evaluation 2 h 22 min with the expert |
| Round 2 | collection 45 min plus one preemption, yield 0.91, 537 rows, 53 quarantined; training 269 steps on 1074 rows; evaluation resubmitted without preemption opt-in after a 50-minute preemption |

Evaluation jobs keep no ledger and restart from zero when preempted; under
the reclaim rate of 2026-09-09 both the final evaluation and the BC0
comparison only finished once resubmitted without the preemption opt-in.

## Files

- Cell: `research/hpc/full_pipeline_20260907/` (README covers all three runs).
- Results: `out/pipeline_summary.json`, `out/r1/round_summary.json` with
  `expert_eval.json`, `out/r1/round_summary.bc0_final.json`,
  `out/r2/round_summary.json` in the pipeline directory.
- Previous write-up: `docs/ambiguity_rerun_20260908.md`.
