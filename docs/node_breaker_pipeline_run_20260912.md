# Full pipeline with node/breaker topology roots (2026-09-12)

Research-only results; nothing here is release evidence.

## What changed from the 2026-09-09 run

Only the topology family. Every other setting is the ten-sigma run's:
548-root plan, seeds 20260910/11/12, suite thresholds 1.2 (training) and
1.0 with rank allowance 2 (development), `MEASUREMENT_ERROR_MIN_SIGMA=10`,
HIF tolerance 0.15, gemma-4-12B-it, BC0 at 1e-4 and the rounds at 3e-5.

- **Topology roots are breaker-status errors in the full IEEE-14
  node/breaker model** (`Transmission/ieee14_full_topology.py`, 65 nodes,
  73 breakers) instead of line-status flips in the bus/branch case:
  `TOPOLOGY_EFFECTS=dangling_line_terminal,bus_split`. A root carries the
  operator's reported breaker map (schematic normal), the substation
  telemetry synthesised at the AC-OPF dispatch, the operator-model layout
  and the meter nodes; the true breaker status stays in the hidden truth.
- **The agent's topology route is the node/breaker one.**
  `get_topology_context` requests the substation measurements, runs the
  generalized state estimator with normalized Lagrange multipliers on the
  reported map, confirms the top-8 breakers by flipped re-estimation and
  screens the admissible flips on the operator model;
  `correct_topology(cb_name, status[, line_index])` sets the breaker and
  re-renders the operator case (a dangling terminal takes the line out of
  service, a bus split adds a fifteenth bus). Workflow and estimator:
  `docs/ieee14_node_breaker_nlm_20260912.md`.
- **The model-visible protocol carries the breaker.**
  `correct_topology_from_path(case_path, cb_name, desired_status[, line_index1])`.

Commits on `feature/ieee14-full-topology`: 937ac51 (run plan), 5955cfe
(protocol), bd59df4 (non-converged flips), 721fef5 (doc), ae99f03 (suite
metadata). Cluster directory `research_full_pipeline_20260912`.

## Results

Truth-audited task success on the 160 development roots, 40-step budget,
production case loader. Measurement+HIF is scored fault-identified rather
than resolved, so the reachable ceiling is 152.

| Adapter | Success | Rate | Invalid actions | False commits | Loops |
| --- | --- | --- | --- | --- | --- |
| BC0 | 143/160 | 0.894 | 52 | 41 | 12 |
| R1 | 143/160 | 0.894 | 41 | 35 | 2 |
| R2 | 147/160 | 0.919 | 10 | 9 | 0 |
| Expert (ceiling) | 147/160 | 0.919 | 4 | 0 | 1 |

Per family (2026-09-09 run in parentheses where different):

| Family | BC0 | R1 | R2 | Expert |
| --- | --- | --- | --- | --- |
| harmonic | 16/16 | 16/16 | 16/16 | 16/16 |
| hif | 15/16 (16/16) | 15/16 | 15/16 | 15/16 |
| measurement | 16/16 | 16/16 | 16/16 | 16/16 |
| measurement+hif | 0/8 | 0/8 | 0/8 | 0/8 |
| measurement+parameter | 10/16 (1/16) | 11/16 (14/16) | 14/16 | 14/16 |
| measurement+topology | 12/12 | 11/12 (12/12) | 12/12 | 12/12 |
| multi_measurement | 12/12 | 12/12 (11/12) | 12/12 | 12/12 |
| no_error | 8/8 | 8/8 | 8/8 | 8/8 |
| parameter | 14/16 (4/16) | 14/16 | 14/16 | 14/16 |
| telemetry_no_disturbance | 8/8 | 8/8 | 8/8 | 8/8 |
| three_phase_unbalance | 16/16 | 16/16 | 16/16 | 16/16 |
| topology | 16/16 | 16/16 | 16/16 | 16/16 |

| Stratum (parameter and measurement+parameter roots) | Roots | BC0 | R1 | R2 | Expert |
| --- | --- | --- | --- | --- | --- |
| dominant (true line first, ratio >= 1.2) | 21 | 17 | 18 | 21 | 21 |
| ambiguous (true line first, ratio < 1.2) | 7 | 7 | 7 | 7 | 7 |
| misranked (neighbour outranks the true line) | 4 | 0 | 0 | 0 | 0 |

R2 ends exactly where the ten-sigma run's R2 and expert ended: the same
147 roots, and the same thirteen misses (the flat-valley HIF root, the
eight measurement+HIF roots and the four misranked parameter roots, on
which every adapter still corrects the higher-ranked neighbour first).
The node/breaker topology family costs nothing against that ceiling.

## The topology route on the development suite

The 16 topology development roots are 12 dangling terminals and 4 bus
splits (`CB_2R4_2R5` at bus 2, `CB_3_L34_B2` at bus 3, `CB_4_N2_B2` at bus
4, `CB_6_B1_B2` at bus 6); the 12 measurement+topology roots compose a
ten-sigma meter error with a dangling terminal, as the mixed family is
defined.

- **Every adapter solves all 16 topology roots by naming the true
  breaker.** Across BC0, R1, R2 and the expert no accepted breaker differs
  from the true one. On the dangling terminals the accepted correction
  carries the breaker and the line row and the audit retires the truth on
  breaker status. On the four splits the accepted target is the breaker
  alone (`accepted_target_audit` lists no branch row because the
  correction has none); the audit's target evidence records the breaker
  with final distance 0, and the final case and measurements match the
  clean fifteen-bus rendering with healthy components preserved. Each
  split episode takes the same ten steps as a dangling one and ends in
  the bounded operator hand-off.
- **Measurement+topology: 12/12 for BC0, R2 and the expert, 11/12 for
  R1.** On the missed root the R1 adapter emitted two invalid actions,
  tripped the loop detector after seven steps and made no correction at
  all; BC0, R2 and the expert corrected the meter and the breaker on the
  same root in fourteen steps. The miss is a policy loop, not a route or
  protocol failure: no topology-family episode of any adapter was
  quarantined for an unsupported correction.
- **Collection.** On the 22 topology-family round-1 training roots the
  BC0 policy emitted the breaker-level `correct_topology` itself in 15
  steps and the expert supplied it in 7; all 22 episodes were classed
  clean and successful. The canonical export of the 551 round-1 and 550
  round-2 rows had no failures.

## Four attempts at stage 0

The run needed four submissions; the first three stopped in stage 0 on
integration gaps that the unit suites had not covered.

| Attempt | Commit | Failure | Fix |
| --- | --- | --- | --- |
| 1 (jobs 17500137-44) | 937ac51 | Canonical export rejected every breaker-named expert action: `correct_topology requires a numeric branch-row target` | 5955cfe: the executable registry exposes `cb_name` plus optional `line_index1`; branch-id and target-less calls stay rejected |
| 2 (17546080-87) | 5955cfe | Aggregate writer hit NaN in `breaker_findings[k].gse_chi_square_after_flip`: an island-class candidate's flipped re-estimation diverges | bd59df4: such flips are recorded as not converged with null chi-square, never offered; `write_jsonl` names the offending row and fields |
| 3 (17579216-23) | bd59df4 | Aggregate built (526 roots), suite build rejected the six node/breaker metadata keys as unsupported execution metadata | ae99f03: the keys are admitted; the true status stays under `audit.truth` |
| 4 (17591676-83) | ae99f03 | ran through (aggregate reused from attempt 3, suite rebuilt) | |

A local 60-root topology draw with the attempt-2 code reproduced the NaN
in 7 roots, always at that one field and always on an `unsupplied_island`
breaker among the top-8 candidates (`CB_6_I_B2`, `CB_Y1014_14B_14N1`,
`CB_Y1014_I10_10B`, ...).

## Stage 0 and the suite

The aggregate has 526 roots and 4891 rows (train 3685, validation 712,
test 494); the ten-sigma run had 4966 rows from the same plan, the
difference being 75 multi-meter rows. The topology family is 60 roots and
600 rows, measurement+topology 50 roots and 700 rows, exactly as before.
The 110 topology corrections in the expert rows are all breaker-level: 93
dangling terminals (breaker plus line row) and 17 bus splits (breaker
alone). Every topology episode has the same shape as in the ten-sigma
run: WLS, three-phase and harmonic screens, parameter context, topology
context, breaker correction, WLS, commit, measurement context, then the
bounded operator hand-off.

The suite is drawn as before from the regenerated roots: 122 + 122
training roots at threshold 1.2 and 160 development roots at 1.0 with
rank allowance 2 (topology 16, measurement+topology 12), 404 physical
roots in all. Development strata: parameter 11 dominant, 3 ambiguous, 2
misranked; measurement+parameter 10 dominant, 4 ambiguous, 2 misranked.

## The BC0 checkpoint

Checkpoint selection by validation loss picked step 576 of 922 this time
(0.63 epoch, loss 0.0061); the ten-sigma run's selection landed on step
320 of 931 (0.34 epoch) and produced the 4/16 parameter and 1/16
measurement+parameter BC0 rows discussed there. With the later checkpoint
BC0 matches the expert on parameter (14/16) and reaches 10/16 on
measurement+parameter, so the BC0 and R1 columns are not comparable
across the two runs beyond the families that were already at the
ceiling; R1 starts from a different student and lands on the four
dominant-stratum roots the previous R1 had already solved. Round 2
closes the gap in both runs.

A side evaluation of the final checkpoint on the same roots (job
17727218, `out/r1/round_summary.bc0_final.json`; the selected checkpoint
re-evaluates to the identical 143):

| BC0 checkpoint | Success | Parameter | Meas.+param | Multi-meter | Topology | Invalid | Loops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 576 (selected) | 143/160 | 14/16 | 10/16 | 12/12 | 16/16 | 52 | 12 |
| 922 (final) | 136/160 | 14/16 | 10/16 | 5/12 | 16/16 | 57 | 26 |

The two checkpoints differ on one family only. On the seven multi-meter
roots it loses, the final checkpoint keeps re-requesting the parameter
and topology contexts between meter corrections (151 and 103 calls over
the twelve roots, against 19 and 12 for the selected checkpoint), trips
the loop detector and runs out the 40-step budget with only some of the
meters fixed (0 of 3, 3 of 4, 2 of 5, ...); the selected checkpoint
corrects every meter in 13 to 34 steps. This is the same drift the
ten-sigma run saw at the end of its epoch (multi-meter 8/12 for its
final checkpoint), but there the selected checkpoint was so early that
the final one still won overall. So neither rule is safe on its own:
selection by validation loss on the 128-row subset picked well this
time and badly last time, and the last third of the epoch costs
multi-meter roots both times. Selecting on the development suite, or on
a validation slice stratified by family, is the fix.

## Pipeline record

| Stage | Outcome |
| --- | --- |
| Stage 0 | 526 roots, 4891 rows in 1 h 18 min on the CPU partition (attempt 3); suite rebuilt from the reused aggregate in 24 min (attempt 4) |
| BC0 | 922 steps in 4 h 36 min, no preemption |
| Round 1 | collection 51 min, yield 0.90, 551 rows exported; training 276 steps on 1102 rows in 1 h 48 min; evaluation 2 h 27 min with the expert |
| Round 2 | collection 48 min, yield 0.92, 550 rows exported; training 275 steps on 1100 rows in 1 h 45 min; evaluation 2 h 18 min |

No stage was preempted in attempt 4; the whole chain ran in 14 h 40 min.

## Files

- Cell: `research/hpc/full_pipeline_20260907/` (README covers all runs);
  `compare_runs.py` prints the per-family and per-stratum comparison of
  two pipeline summaries.
- Results: `out/pipeline_summary.json`, `out/r1/round_summary.json` and
  `out/r2/round_summary.json` with the `*_eval.json` files under
  `collection/evaluation` (round 2 names the R1 student `bc0_eval.json`
  and the R2 candidate `r1_eval.json`), in the pipeline directory; failed
  attempts' logs under `logs_attempt1..3/`.
- Previous write-up: `docs/ten_sigma_rerun_20260909.md`.
