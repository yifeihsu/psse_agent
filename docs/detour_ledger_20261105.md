# Detour ledger: actions above the protocol floor, attributed (2026-11-05)

The evaluator reports cost totals and protocol violations but never names a
detour: its loop detector resets on every lifecycle advance, so a
correct-verify-reject-rollback cycle counts as progress, and its invalid-action
counter mixes the agent's mistakes with the corrector's failures
(`psse_env/dagger/evaluator.py`).  This note adds a ledger that attributes every
action above the protocol floor to a named cause, computes it from the records
the repository tracks, and gives the commands for the per-episode version on the
episode files that are not tracked.

Code: `research/compound_errors/detour_ledger.py` (standalone, imports nothing
from `psse_env`), tests in `research/compound_errors/test_detour_ledger.py`
(9 tests).  Outputs of this note: `docs/detour_ledger_20261105/`.

## 1. Definitions

| term | definition |
| --- | --- |
| protocol floor | `2 + 4 x atoms`: one opening solve and one handoff, plus per error atom a context fetch, the correction, its verifying solve and the commit (the expert's own minimum, `docs/rl_hard_cases_plan_20261013.md` section 3.1). `control_meter` is taken at 2 atoms, the midpoint of 1 to 3. |
| slack | actions minus the floor |
| hypothesis | one correction attempt, executed or refused |
| rejected cycle | an executed correction whose lifecycle step was a rollback: the correction, the solves in between and the rollback are wasted |
| refused correction | a correction whose execution failed before verification: one wasted action, split into `tool_refused` (corrector not converging, dispatch error) and `policy_invalid` (the gate refused a proposal: unsupported target, lifecycle violation, missing precondition) by error code |
| fruitless route | a context fetch after which no correction of that family was committed before the next fetch or the end; the fetch is wasted (per-episode mode only) |
| comparison | `compare_parameter_candidates` and `compare_routes`: exploration, counted, not wasted |
| wasted | the union of the wasted step sets |
| harm, interface | an accepted wrong target is harm; a true target rejected or refused is an interface limit |

Two modes.  `episodes` reads per-episode records (the compound rollout's compact
record with `trace` and `corrections[*].target_is_true`, or the evaluator's
`EpisodeEvaluation` trace) and attributes step by step.  `summary` reads a
rollout `summary.json` and approximates the ledger per class from the verdict
counts (`tool:true|wrong:verdict`), assuming a rejected candidate cost 3 actions,
a refused one 1 and an undisposed one 2.  The summary mode cannot see fruitless
fetches, repeated solves or the split by solved and failed roots; the residual
column below is what it leaves unattributed.

## 2. What is tracked, and what is not

The per-root `episodes.jsonl` files of every study live under gitignored
`output/` folders (the compound studies on the Windows machine, the DAgger
cells on torch).  Nothing in git holds a trace.  What git holds is the IEEE 118
study's rollout summaries and probes
(`docs/compound_errors_ieee118_20261028/`), so the tables below are the
summary-mode approximation on those, for the IEEE 118 suite and its IEEE 14
bus-branch control, in sample (suite seed 20261027) and on the fresh draw
(20261029).  The IEEE 14 node/breaker study (`claude/nostalgic-kalam-be5f7a`)
tracks no summaries, so its ledger needs the per-episode run of section 6.

## 3. IEEE 118, in sample, 40 actions

`baseline_compare` as deployed (`ieee118_deployed_summary.json`) against the
meter-route rule (`ieee118_meter_route11_40_summary.json`).  Means per root.

| teacher | class | n | solved | actions | floor | slack | est. wasted | wasted / slack | hypotheses per accepted | rejected (true / wrong) | refused (true / wrong) | accepted wrong | comparisons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | control: 1 to 3 meters | 50 | 45 | 21.6 | 10 | 11.6 | 8.6 | 0.74 | 2.49 | 143 (0 / 143) | 0 | 0 | 0.62 |
| baseline | control: one R/X | 50 | 45 | 8.0 | 6 | 2.0 | 0.4 | 0.19 | 1.18 | 5 (0 / 5) | 4 (3 / 1) | 3 | 0.56 |
| baseline | meter on the faulted line + R/X | 75 | 11 | 22.6 | 10 | 12.6 | 10.5 | 0.84 | 4.19 | 237 (70 / 167) | 79 (5 / 74) | 50 | 0.85 |
| baseline | meter + two R/X | 150 | 130 | 17.1 | 14 | 3.1 | 0.7 | 0.22 | 1.11 | 27 (3 / 24) | 23 (10 / 13) | 16 | 1.35 |
| baseline | two R/X, adjacent | 150 | 95 | 15.7 | 10 | 5.7 | 1.9 | 0.34 | 1.41 | 76 (6 / 70) | 60 (35 / 25) | 45 | 1.25 |
| baseline | two R/X, distant | 150 | 130 | 13.0 | 10 | 3.0 | 0.5 | 0.17 | 1.14 | 18 (2 / 16) | 25 (13 / 12) | 21 | 1.22 |
| baseline | **all** | 625 | 456 | 16.1 | | 5.4 | 2.7 | 0.50 | 1.53 | 506 (81 / 425) | 191 (66 / 125) | 135 | 1.11 |
| meter-route | control: 1 to 3 meters | 50 | 50 | 11.2 | 10 | 1.2 | 0.1 | 0.05 | 1.01 | 1 (0 / 1) | 0 | 0 | 0.00 |
| meter-route | meter on the faulted line + R/X | 75 | 11 | 21.9 | 10 | 11.9 | 9.8 | 0.82 | 3.80 | 220 (68 / 152) | 74 (4 / 70) | 55 | 0.79 |
| meter-route | **all** | 625 | 461 | 15.2 | | 4.5 | 1.9 | 0.43 | 1.40 | 344 (78 / 266) | 186 (65 / 121) | 141 | 1.03 |

The other four classes are unchanged under the rule.  Slack decomposition for
the baseline, per root: slack 5.4 = wasted 2.7 + comparisons 1.1 + residual
1.6.  Per class the residual is 1.1 to 1.3 on the one-line control, the distant
pairs and meter + two R/X, 2.4 on the meter controls and 2.5 on the adjacent
pairs.  On a two-atom class one action of the residual is the mandated re-solve
after the first commit, which the floor does not count, so the unattributed
remainder is near zero on the distant pairs and 1.4 to 1.5 actions on the meter
controls and adjacent pairs, where the fruitless context fetches live.

Reading.  Half of all slack is named waste, and it sits in two classes.  The
meter controls pay 143 rejected wrong lines, 8.6 wasted actions per root, 74%
of their slack; the rule removes it (0.1 per root).  The meter on the faulted
line pays 10.5 wasted actions per root, 84% of its slack, with 70 true line
corrections rejected by the verifier and 74 wrong targets refused: that waste
is the agent trying alternatives after the right answer was refused, and no
first guess avoids it.  The adjacent pairs' waste is mostly refusals of true
lines by the corrector (35 of 60).  Everywhere else wasted actions are under one
per root and the comparison is the largest slack item.

## 4. IEEE 14 control, in sample, 40 actions

| teacher | class | n | solved | actions | floor | slack | est. wasted | wasted / slack | hypotheses per accepted | rejected (true / wrong) | refused (true / wrong) | accepted wrong | comparisons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | control: 1 to 3 meters | 50 | 49 | 21.6 | 10 | 11.6 | 7.9 | 0.68 | 2.26 | 131 (0 / 131) | 0 | 0 | 0.68 |
| baseline | control: one R/X | 50 | 48 | 7.7 | 6 | 1.7 | 0.0 | 0.01 | 1.02 | 0 | 1 (1 / 0) | 4 | 0.48 |
| baseline | meter on the faulted line + R/X | 75 | 13 | 19.1 | 10 | 9.1 | 8.3 | 0.91 | 3.33 | 193 (62 / 131) | 42 (2 / 40) | 46 | 0.47 |
| baseline | meter + two R/X | 150 | 88 | 18.6 | 14 | 4.6 | 2.2 | 0.48 | 1.33 | 93 (39 / 54) | 48 (24 / 24) | 22 | 1.10 |
| baseline | two R/X, adjacent | 150 | 61 | 17.6 | 10 | 7.6 | 2.2 | 0.29 | 1.42 | 88 (17 / 71) | 67 (48 / 19) | 62 | 1.76 |
| baseline | two R/X, distant | 150 | 115 | 12.4 | 10 | 2.4 | 0.7 | 0.30 | 1.17 | 28 (7 / 21) | 22 (15 / 7) | 16 | 0.69 |
| baseline | **all** | 625 | 374 | 16.3 | | 5.7 | 2.8 | 0.50 | 1.53 | 533 (125 / 408) | 180 (90 / 90) | 150 | 1.00 |
| meter-route | control: 1 to 3 meters | 50 | 50 | 12.4 | 10 | 2.4 | 0.8 | 0.32 | 1.12 | 13 (0 / 13) | 0 | 0 | 0.10 |
| meter-route | **all** | 625 | 375 | 15.5 | | 4.9 | 2.2 | 0.45 | 1.42 | 400 (118 / 282) | 173 (87 / 86) | 158 | 0.96 |

The same two classes carry the waste.  On this system true corrections are
refused twice as often as on IEEE 118 (90 of 180 refusals), most of them the
adjacent pairs' corrector failures (48), the small network's crowding of
section 5.1 of the IEEE 118 note.  The adjacent pairs also carry the largest
residual, 3.7 actions per root, with 1.76 comparisons per root beside it.

## 5. Budgets and the fresh draw

Under a 16-action budget the floor logic breaks for the cut episodes (a
three-meter root cut at 16 cannot reach its floor), so the wasted actions are
read in absolute terms there:

| run | class | n | solved | actions | est. wasted per root | rejected wrong | hypotheses per accepted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEEE 118 baseline, 16 | control: 1 to 3 meters | 50 | 19 | 12.1 | 4.4 | 74 | 2.42 |
| IEEE 118 meter-route, 16 | control: 1 to 3 meters | 50 | 49 | 11.1 | 0.1 | 1 | 1.01 |
| IEEE 118 baseline, 16 | all | 625 | 424 | 13.2 | 1.6 | 207 | 1.38 |
| IEEE 118 meter-route, 16 | all | 625 | 454 | 13.1 | 1.2 | 133 | 1.30 |
| IEEE 14 control baseline, 16 | control: 1 to 3 meters | 50 | 18 | 12.5 | 4.4 | 73 | 2.28 |
| IEEE 14 control meter-route, 16 | control: 1 to 3 meters | 50 | 46 | 11.4 | 0.3 | 5 | 1.05 |

Fresh draw (seed 20261029), 40 actions, all classes:

| run | n | solved | actions | slack | est. wasted | comparisons | residual | wasted / slack | rejected (true / wrong) | refused (true / wrong) | accepted wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEEE 118 baseline | 625 | 441 | 16.0 | 5.4 | 2.7 | 1.08 | 1.6 | 0.50 | 488 (94 / 394) | 210 (63 / 147) | 138 |
| IEEE 118 meter-route | 625 | 445 | 15.1 | 4.4 | 1.9 | 1.03 | 1.5 | 0.44 | 337 (91 / 246) | 206 (62 / 144) | 143 |
| IEEE 14 control baseline | 625 | 379 | 16.1 | 5.4 | 2.6 | 0.98 | 1.8 | 0.49 | 497 (138 / 359) | 160 (67 / 93) | 141 |
| IEEE 14 control meter-route | 625 | 379 | 15.5 | 4.8 | 2.2 | 0.96 | 1.7 | 0.45 | 406 (129 / 277) | 154 (64 / 90) | 146 |

The fresh draw reproduces the in-sample ledger to within 0.1 action per root on
every total, and the class pattern holds: the IEEE 118 meter controls waste 8.9
of 11.9 slack actions under the baseline and 0.0 under the rule; the meter on
the faulted line wastes 9.9 of 12.6 with 70 true lines rejected.

## 6. Per-root detour on the roots the meter-route rule changed

From `meter_route_probe.json` (in sample) and `fresh_20261029/meter_route_probe.json`,
the roots the rule gained, with the two teachers' actions and true commits on
the same root, plus the mean actions on the roots both teachers solve:

| draw | system | budget | roots gained | baseline actions, true commits | rule actions, true commits | both solved: baseline, rule (n) | roots lost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in sample | IEEE 118 | 16 | 30 | 14.7, 1.00 | 13.3, 2.57 | 7.9, 7.4 (19) | 0 |
| in sample | IEEE 118 | 40 | 5 | 38.0, 2.00 | 15.0, 3.00 | 19.8, 10.8 (45) | 0 |
| in sample | IEEE 14 control | 16 | 28 | 14.5, 1.00 | 12.7, 2.43 | 9.1, 8.6 (18) | 0 |
| in sample | IEEE 14 control | 40 | 1 | 38.0, 2.00 | 15.0, 3.00 | 21.3, 12.4 (49) | 0 |
| fresh | IEEE 118 | 16 | 32 | 14.8, 1.00 | 13.1, 2.53 | 8.0, 7.4 (18) | 0 |
| fresh | IEEE 118 | 40 | 4 | 39.2, 2.00 | 15.0, 3.00 | 20.4, 10.7 (46) | 0 |
| fresh | IEEE 14 control | 16 | 22 | 14.5, 1.05 | 12.8, 2.45 | 9.6, 8.7 (27) | 1 |
| fresh | IEEE 14 control | 40 | 1 | 38.0, 2.00 | 15.0, 3.00 | 16.7, 10.8 (49) | 1 |

These are the only per-root detour measurements the repository holds today.  At
40 actions a three-meter root the baseline loses spends 38 to 39 actions on two
true commits; the rule spends 15 on three.  The one lost root
(`r0_038f9574c00a`, IEEE 14 control, meter + two R/X) is the masking case of the
IEEE 118 note's section 11.3: the rule commits a healthy voltage meter, 15 or 24
actions against the baseline's 15.

## 7. The per-episode ledger, where the episode files are

The same script attributes step by step from `episodes.jsonl`, with the split by
solved and failed roots, fruitless routes, refused actions by error code and
the truth flags of every correction.  On the Windows machine, from the
repository root of the study's worktree (`<D>`, `<C>`, `<F>`, `<G>` as in the
IEEE 118 note's sections 9.6 and 11.7):

```
python research/compound_errors/detour_ledger.py episodes --episodes <D>/expert_baseline_compare/episodes.jsonl --manifest <D>/manifest.json --output-dir <D>/detour_ledger/baseline_40
python research/compound_errors/detour_ledger.py episodes --episodes <D>/meter_route11_40/episodes.jsonl --manifest <D>/manifest.json --output-dir <D>/detour_ledger/meter_route_40
python research/compound_errors/detour_ledger.py episodes --episodes <D>/budget_16/episodes.jsonl --manifest <D>/manifest.json --output-dir <D>/detour_ledger/baseline_16
python research/compound_errors/detour_ledger.py episodes --episodes <C>/expert_baseline_compare/episodes.jsonl --manifest <C>/manifest.json --output-dir <C>/detour_ledger/baseline_40
python research/compound_errors/detour_ledger.py episodes --episodes <F>/baseline_compare_40/episodes.jsonl --manifest <F>/manifest.json --output-dir <F>/detour_ledger/baseline_40
```

For the IEEE 14 node/breaker study (`claude/nostalgic-kalam-be5f7a`, 1,275
roots), the stage-2 and item-7 runs under `output/compound_errors_20261027/`
and `output/item7_step7/`, with that suite's `manifest.json` (its classes carry
breaker atoms, which the floor counts):

```
python research/compound_errors/detour_ledger.py episodes --episodes output/compound_errors_20261027/stage2_item3/expert_baseline_compare/episodes.jsonl --manifest output/compound_errors_20261027/manifest.json --output-dir output/compound_errors_20261027/detour_ledger/baseline
python research/compound_errors/detour_ledger.py episodes --episodes output/item7_step7/<route teacher run>/episodes.jsonl --manifest output/compound_errors_20261027/manifest.json --output-dir output/compound_errors_20261027/detour_ledger/routes
```

For the DAgger cells on torch, the evaluator's full `EpisodeEvaluation` records
(the `*_eval.json` payloads, exported one episode per line) carry the trace; the
cells' compact `episodes.jsonl` does not and gives cost columns only.  The
atoms per family come from `--atoms family=count` there (one for the
single-error families, two for the mixed ones, the meter count for multi-meter
roots).

Each run writes `ledger.jsonl` (one row per episode), `summary.json` and
`summary.md` (means per class and outcome).  Tracking the `summary.*` files
beside the study's other summaries would make the ledger part of every future
teacher comparison.

## 8. What the ledger changes in the earlier reading

- "Little room at 40 actions" was a success statement; the ledger puts the
  efficiency statement beside it.  Half of the slack above the floor is named
  waste on both systems and both draws, about 2.7 actions per root, and it is
  concentrated: two classes of six carry it.
- The removable part was removed by a rule.  The meter-control detour is 8.6
  wasted actions per root under the baseline and 0.1 under the meter-route rule,
  with the roots solved rising from 45 to 50.
- The part that remains is refusals of correct corrections, not wandering: 70
  true line corrections rejected on the meter-on-line class, 35 true lines
  refused by the corrector on the adjacent pairs.  A policy cannot avoid these
  by choosing better; it can only stop sooner, which is the hypothesis
  abandonment decision the RL watchlist already names.
- The loop and invalid-action counters see none of this.  Every rejected cycle
  in these tables advanced the lifecycle, and the refusals it counts are mostly
  the corrector's.
