# The stopping-state diagnostic of the supervised model (2026-11-20)

Asked by the user on 2026-11-20: "Based on the results from your previous training pass, is our current RL approach
likely to improve performance, or should we focus on improving the methodology first?"  The answer given: the
focused RL run has little grip on the unrepaired cases, and one question about the method comes first.  At the
point where each unrepaired episode stops with an error left, (1) does the evidence the model reads there say what
is wrong, and (2) how much probability does the model give to the truth-built way on?  The user said "Yes, build
it" (the diagnostic beside the focused run, which keeps running).

Part 1 runs on Windows, on the episodes the supervised model's two greedy passes stored on torch (development: the
300 cases of `ieee14_dev300.json`; training: the 2,093 training cases a truth-built sequence repairs):

* `research/minimal_env/stopping_states.py` replays every stored episode in the environment, labels every state an
  action was chosen on with what the truth says is still wrong, and at the stopping state of each unrepaired episode
  searches the truth-built continuation (the corrections the truth names for the errors left, in every order, from
  the current version and then, after an `undo`, from other saved versions);
* `research/minimal_env/evidence_probe.py` fits small boosted-tree probes on the evidence columns of the training
  states (folds grouped by case) and reads them at the stopping states: is the case still unrepaired there, and
  which object is wrong.

Part 2 is this cell's one GPU job: `research/minimal_env/score_actions.py` reads, for each stopping state, the
supervised model's probability of its own last action, of the first actions of the truth-built continuation, and of
the other `finish`, teacher-forced in bf16 at the sampling temperature, by the loader and the forward pass the RL
update uses.  Nothing is generated and nothing is trained.

## Files

| File | What it is |
| --- | --- |
| `cell.env` | Paths and the job environment: work directory `/scratch/yx3882/minimal_diag_20261120`, the supervised pass's adapter read in place, the inputs under `data/`, the scores under `out/scores`. |
| `stage_score.sbatch` | The job (`mdg-score`): two processes on a card of 70 GB or more, each with half of the states, then the closing call that scores what is left and writes `summary.json`.  Preemptible and requeued; one file per state. |
| `submit_cell.sh` | `score`: submits the job; refuses before the supervised pass's receipt or the inputs, once the summary is written, and while a job of the name is queued. |
| `status_cell.sh` | Read-only: inputs, queue, `sacct`, states written, the logs' last lines and the summary. |
| `deploy_from_windows.sh` | Clones the RL cell's source into a source of this cell's own (never moving that one: the focused run reads it), applies the incremental bundle, and uploads the two input files with their checksums verified.  Submits nothing. |
| `test_cell.py` | The scripts parse, name this cell and write nothing of another cell; the stage runs the shards and the closing call (with a stub scorer). |

## Order of work, each cluster step on the user's word

1. Part 1 on Windows (done here; outputs under `output/minimal_env_20261115/stopping_states_20261120/`).
2. Deploy: `bash research/hpc/minimal_diag_20261120/deploy_from_windows.sh <commit40>`.
3. The job: `wsl -- ssh torch bash -s -- score < research/hpc/minimal_diag_20261120/submit_cell.sh`.
4. Status: `wsl -- ssh torch bash -s < research/hpc/minimal_diag_20261120/status_cell.sh`.

## State

Built 2026-11-20. Part 1 done on Windows the same day (below).  Part 2 deployed at 96dbf99 on the user's "Deploy
and submit" (2026-11-20; source cloned from the RL cell's, both input files uploaded with checksums verified: 785
states, 2,564 actions); job `mdg-score` 23861813 submitted 18:24:50Z, COMPLETED on gr105 in 2 h 15 min (summary
written 20:45:59Z; 785 of 785 states, no prompt cut, no unrenderable action, no traceback); read below, "Part 2".  The
scoring inputs are in `output/minimal_env_20261115/stopping_states_20261120/`: 74 development states (49 stopping
states, 25 first wrong turns) with 240 actions, and 711 training states (502 and 209) with 2,324 actions.  The
scores (one file per state, copied from `out/scores/states`), the breakdown by pattern and the analysis scripts are
in `part2/` there (local, not committed).

## Part 1, 2026-11-20

**The replay.** All 2,393 stored episodes replay to their stored results, step by step and in the score
(development 300 of 300, training 2,093 of 2,093).  The unrepaired solvable episodes are the supervised model's 49 of
the 247 development cases and 502 of the 2,093 training cases.

**Where they stop.**

| | Development | Training |
| --- | --- | --- |
| A true error never corrected | 45 | 465 |
| Every error corrected, one outside tolerance | 3 | 23 |
| No error left, a healthy object damaged | 1 | 14 |
| The estimate passes at the stop | 16 | 174 |
| A healthy object damaged at the stop | 21 | 160 |
| Declared resolved / unresolved / out of actions | 27 / 20 / 2 | 267 / 217 / 18 |
| The truth-built way on: straight from the stop | 24 | 307 |
| ... after an undo | 25 | 193 |
| ... none found | 0 | 2 |

The errors left at the development stops are 29 breakers, 21 lines and none of the meters never corrected, and 9
lines and 1 meter corrected but outside tolerance.  Training: 264 breakers, 244 lines and 17 meters never
corrected, 41 lines and 6 meters corrected but outside tolerance, 1 meter correction undone.

**The first wrong turn**, the first action that does not act on an unrepaired object, does not undo damage and does
not finish a repaired state (inspections pass).  Development 49: 22 finish with an error left after every earlier
step was on the path; 24 correct a healthy object earlier (18 of them a line fit) and 1 undoes with nothing
damaged, mostly at the third action; 2 never leave the path and run out of actions on refused calls.  Training 502:
278, 194, 15 and 15.  From the 25 development wrong turns before the stop the truth goes on without an undo in 24.

**The probes** (fitted on the training states, out of fold there; read on the development states):

| Reading | Development | Training |
| --- | --- | --- |
| "Still unrepaired?" flags the stop, at 5% of repaired finishes flagged (computed breaker table) | 43 of 49 (10 of the 16 with the estimate passing); repaired finishes 4 of 198 | 426 of 502 (107 of 174) |
| The same with only the shown breaker table | 40 of 49 (8 of 16) | 376 of 502 (64 of 174) |
| "Which object?": the object left placed first among all lines, meters and breakers (computed table) | 37 of 48 (top 3: 43) | 371 of 488 (top 3: 426) |
| ... a breaker left, placed first | 22 of 25 | 231 of 244 |
| ... a line left, placed first | 14 of 22 (top 3: 19) | 130 of 233 (top 3: 179) |
| ... with only the shown breaker table | 19 of 48; breakers 2 of 23 | 156 of 488; breakers 15 of 232 |
| By the kind's first statistic alone (largest breaker score, line multiplier, meter residual), within its kind | 38 of 48; breakers 22 of 25 | 352 of 488 |
| At the first wrong turns before the stop: the object left placed first, and above the object the model corrected (computed table) | 17 of 25, 17 of 25 (top 3: 24) | 167 of 208, 159 of 208 |

**The breaker re-inspection.** After its first breaker correction, in the two-breaker classes of the training pass
the model inspects the breakers again in 140 of 289 episodes (132 of them repaired) and does not in 149 (5
repaired); in the one-breaker classes it inspects in 13 of 846.  The truth-built sequences inspect after the first
breaker correction in 300 of 300 two-breaker cases and in none of the 891 others: the inspection is inserted
where the truth knows a breaker correction follows, so the demonstrations make that decision with the truth.

**Reading (mine).** At most stops the evidence the model could have had says the case is not repaired and which
object is wrong, but for a breaker only on the current version's breaker table, one `inspect` away, which the model
does not ask for.  Part 2 measures how much probability the model gives the step on, at the stop and at the first
wrong turn.

## The failure patterns, 2026-11-20

Asked by the user: "Apart from cases involving multiple breaker errors, are there any other identifiable error
patterns? Can RL learn to correct them?"  Each unrepaired solvable episode gets one pattern, from the truth's
corrections and the episode's accepted calls, first match wins (breaker before line before meter).  "Evidence"
is the state probe and the candidate probe at the stop, with the breaker table computed for the current version
(the "inspect" reading).  "Rollouts" are the focused run's groups (8 samples at T = 1, iterations 0 and 1, 48
failed training cases drawn); a group "reaches a repair" when one of its samples repairs the case.

| Pattern | Development (49) | Training (502) | Ending, training: resolved / unresolved / out of actions | Evidence at the stop, training: flagged; the error left placed first (only the shown table) | Way on, training: direct / after an undo | Rollouts: groups reaching a repair; samples repaired |
| --- | --- | --- | --- | --- | --- | --- |
| A. Two-breaker case, one breaker corrected, the second never | 17 | 144 | 38 / 104 / 2 | 143 of 144; 135 of 144 (1) | 113 / 31 | 9 of 18; 25 of 144 |
| B. A breaker never corrected, no breaker corrected | 7 | 90 | 40 / 36 / 14 | 90; 87 of 90 (17) | 64 / 26 | 2 of 7; 3 of 56 |
| C. A healthy breaker corrected instead | 2 | 14 | 2 / 11 / 1 | 14; 10 of 14 (4) | 3 / 11 | 1 of 1; 2 of 8 |
| D. A healthy line fitted instead of the faulted line | 9 | 56 | 42 / 14 / 0 | 41; 26 of 56 (26), top 3: 40 | 8 / 48 | 3 of 6; 8 of 48 |
| E. A faulted line never fitted, no healthy line fitted | 10 | 159 | 110 / 48 / 1 | 118; 106 of 159 (103), top 3: 130 | 110 / 49 | 8 of 15; 21 of 120 |
| F. Every error corrected, a line fit outside tolerance | 2 | 17 | 15 / 2 / 0 | 4; 1 of 17 (1) | 5 / 12 | none drawn |
| G. A meter error left | 1 | 8 | 6 / 2 / 0 | 6; 6 of 8 | 6 / 2 | 0 of 1; 0 of 8 |
| I. No error left, a healthy correction kept | 1 | 14 | 14 / 0 / 0 | 10; (nothing left to place) | 0 / 14 | none drawn |

Classes: A is cb2_dangling, cb2_mixed and cb2_split; B is mostly the one-breaker mixed classes (training:
rx_cb_distant 31, rx_cb_meter 30, rx_cb_shared_bus 16, cb2 11, control_cb 2); D and E are the two-line classes
(meter_rx2, rx2_distant, rx2_adjacent) and the mixed classes.  The estimate passes at the stop in 23 of A, 28 of
B, 21 of D, 82 of E and 10 of F (training).

What the calls show (training; development in brackets):

* **A.** After the first breaker correction the model turns to a line fit, which the tool refuses (a healthy line,
  `parameter_correction_failure`), in 117 of 144 (10 of 17); after the first refusal it finishes in 72 (6) and
  repeats the refused fit in 35 (3); it inspects the breakers in 1 (none).
* **B.** 13 of the 90 (2 of 7), and 2 cases of A, are one name: the model writes `CB_2R4_R5` for `CB_2R4_2R5` and
  repeats the refused call (`unknown_object`; 214 refused calls in the unrepaired training episodes, 30 in
  development) until the actions run out or it turns elsewhere.  The supervised model repairs 10 of the 30 training
  cases whose truth corrects that breaker (2 of 5 in development), against 998 of the 1,461 case-and-breaker pairs
  of every other breaker (120 of 168).  Over all 90, the breaker left is first on the current version's computed
  table in 87, and with only what was shown in 17.
* **E.** 66 fit one faulted line and stop with the second never fitted (the line counterpart of A); 58 never fit a
  faulted line; 35 tried the faulted line, the tool refused the fit, and did not get it in later (development: 2, 4
  and 4).  From the stop one fit repairs 101 of the 159; for the 35 refused, the way on starts with an undo in 27 (4
  of 4), so an earlier correction stands in the way, and a refit of that line comes first in only 3.
* **D.** The probe places the faulted line first in 26 of 56 at the stop, so the lines are not told apart in about
  half; the way on needs an undo in 48 of 56, and the model undid in 25 but did not find the line.
* **F.** Every error corrected, the estimate passes in 10 of 17 and the probe flags 4 of 17: the evidence reads
  repaired.  The way on is mostly an undo and the fits in another order (the order-dependence of
  r0_220e9e133e72, above).

**Reading (mine).**  A and B share a cause: the breaker left shows on the current version's breaker table, which
the model does not fetch, and a refused line fit, a visible cue, is answered by stopping.  E is mostly an early
stop (124 of 159 never try the line left, 66 of them after fitting the other faulted line) with the cue present at
the stop in about three quarters.  These are learnable from what the model sees, and the rollouts reach a repair in
about half of their groups (A 9 of 18, E 8 of 15), the contrast RL needs; the first update moved the greedy result
by one case, so the rate is the question.  D and the refused part of E need an undo of the model's own wrong
correction and a switch on evidence that is ambiguous in about half: learnable only in part.  F is not visible in
the evidence at the stop, so no policy can learn it from there; it is a limit of the fitting tool's order
dependence.  The misspelled breaker is a copy defect of one name.  Part 2 reads, per pattern, the probability the
supervised model gives the truth-built step.

## Part 2, the scores, 2026-11-20

The job scored every state: the supervised model's probability, at the sampling temperature (T = 1, bf16), of its
own action there, of each first step of the truth-built way on, and (at the stop) of the other `finish`.  Per state:

* **P(own)**: the model's own action (at the stop its `finish`, or the refused call it ran out of actions on; at the
  first wrong turn the wrong call).
* **P(on)**: the steps toward the repair together: the first actions of the truth-built orders, both forms of an
  `undo` where one comes first, and the `inspect` of the breakers where the current version does not show them (the
  demonstrations take that step).
* **1 of 8**: the expected number of states where at least one of 8 samples takes such a step, the sum of
  1 - (1 - P(on))^8: how often a group of the focused run has a sample on the way there.
* **Lead**: ln P(own) - ln P(best step on), in nats: how far an update must move before greedy decoding takes the
  step.  **Tool / object**: the best step's cost split into the tool name's tokens and the arguments' tokens.

| | Development, stop (49) | Development, first wrong turn (25) | Training, stop (502) | Training, first wrong turn (209) |
| --- | --- | --- | --- | --- |
| P(own), median | 0.90 | 0.78 | 0.89 | 0.78 |
| P(on), median | 0.00031 | 0.075 | 0.00089 | 0.031 |
| P(on) at least 0.1 / 0.01 / 0.001 | 5 / 12 / 18 | 12 / 17 / 23 | 45 / 134 / 245 | 66 / 130 / 174 |
| 1 of 8; 1 of 32 (expected states) | 7.2; 10.9 | 11.1; 16.6 | 60.8; 112.3 | 75.2; 115.9 |
| Lead, median; states within 3 nats | 8.2; 10 | 2.4; 16 | 7.0; 72 | 3.2; 97 |
| A step on ahead of the own action | 0 | 2 | 2 | 8 |

The first wrong turn is scored only where it comes before the stop (25 of the 49, 209 of the 502); elsewhere the
stop is the first wrong turn.  Two training states have no step on (no repairing order there).

By pattern, training (development in brackets):

| Pattern | Stop: states; P(own); P(on) median | Stop: 1 of 8; lead (nats) | Stop: tool / object (nats), tool the larger cost in | First wrong turn: states; P(on) median; 1 of 8; lead | First wrong turn: tool / object |
| --- | --- | --- | --- | --- | --- |
| A. second breaker never corrected | 144 (17); 0.81 (0.80); 0.0041 (0.00033) | 29.3 (3.4); 5.4 (8.0) | 5.3 / 0.0, 139 of 144 | 37 (8); 0.0073 (0.067); 8.2 (3.1); 4.4 (2.2) | 5.3 / 0.0 |
| B. breaker never corrected, none corrected | 90 (7); 0.90 (0.85); 0.00030 (0.00031) | 2.8 (0.0); 8.2 (8.2) | 7.9 / 0.0, 64 of 90 | 34 (1); 0.00043 (0.0097); 3.1 (0.1); 7.4 (4.9) | 8.0 / 0.0 |
| C. healthy breaker instead | 14 (2); 0.81 (0.73); 0.00083 (0.0011) | 1.1 (0.0); 6.6 (16.3) | 6.7 / 0.0 | 14 (2); 0.12 (0.10); 8.3 (1.1); 2.4 (2.3) | 0.0 / 2.6 |
| D. healthy line fitted instead | 56 (9); 0.94 (0.95); 6e-9 (8e-9) | 2.6 (0.2); 18.7 (18.5) | 7.3 / 9.2 | 56 (9); 0.10 (0.12); 30.0 (4.6); 1.9 (2.0) | 0.0 / 1.9 |
| E. faulted line never fitted | 159 (10); 0.91 (0.88); 0.0045 (0.070) | 24.4 (3.5); 5.3 (2.6) | 4.2 / 0.4, 126 of 158 | 46 (3); 0.049 (0.0011); 17.2 (0.8); 2.8 (6.8) | 2.4 / 0.1 |
| F. fit outside tolerance | 17 (2); 1.00 (1.00); 6e-12 (1e-11) | 0.0 (0.0); 25.7 (26.2) | 7.6 / 18.1 | 5 (1); 0.12 (0.29); 1.9 (0.9); 2.9 (0.9) | 2.3 / 0.8 |
| G. meter left | 8 (1); 0.99 (0.92); 0.00057 (0.00031) | 0.5 (0.0); 8.2 (8.0) | 5.8 / 2.9 | 3 (0); 4e-5; 0.1; 10.2 | 3.3 / 4.4 |
| I. healthy correction kept | 14 (1); 1.00 (1.00); 5e-12 (6e-13) | 0.1 (0.0); 25.9 (28.2) | 10.1 / 13.9 | 14 (1); 0.063 (0.074); 6.4 (0.5); 2.6 (2.5) | 1.8 / 0.1 |

What it shows:

* **The stop is a confident decision.**  The model gives its own `finish` a median 0.89 (development 0.90) and the
  step on a median 0.00089 (0.00031); no step on leads the model's own action at any development stop, and 2 of the
  502 training stops.  One of 8 samples would step on at about 61 of the 502 training stops (7.2 of 49).
* **A and E: keep going or finish.**  The cost is the tool, not the object: once the model names `inspect` (A) or
  `correct_parameters` (E) the breaker or line follows at no cost (0.0 and 0.4 nats).  One of 8 samples steps on at
  about 29 of 144 A stops and 24 of 159 E stops; the median lead is 5.4 and 5.3 nats.  At A stops the `inspect` of
  the breakers carries the step (P median 0.0075 over 112; correcting the breaker blind 0.0001).
* **B: out of reach.**  The step is the `inspect` of the breakers at P median 0.0003 and a lead of 8.2 nats (7.4 at
  its first wrong turn): about 3 of 90 stops and 3 of 34 first wrong turns would see a sample step on.  **The
  misspelled breaker:** in all 17 scored calls that correct `CB_2R4_2R5` (13 B stops and 1 A stop in training, 2 B
  stops in development, 1 A first wrong turn) the `2` after `R4_` costs 6.0 to 11.3 nats (a probability of 0.0025
  down to 0.00001 per call), while every other token of the name costs at most 0.2 nats (one at 1.8): the model
  knows the breaker and drops one character, and sampling will not write the name.
* **D, C and I: reachable at the first wrong turn, not at the stop.**  At the stop the step is an `undo` of the
  model's own correction, at P median 6e-9 for D (lead 18.7 nats).  At the turn where the model fitted the healthy
  line the faulted line is a close second: P median 0.10, one of 8 samples picks it at 30 of 56 (4.6 of 9), lead 1.9
  nats, and the cost there is the object (which line, 1.9 nats), not the tool.  C and I: 8.3 and 6.4 of 14.
* **F: no.**  At the stop nothing calls for more (lead 25.7 nats); at its first wrong turn (the fit order, 5 cases)
  the other order has P 0.12, but nothing in the evidence says which order is right.
* **Against the focused rollouts** (48 training cases drawn in iterations 0 and 1): P(on) at the stop is higher
  where a sample repaired the case (median 0.013 over 23, against 0.0012 over 25; area under the curve 0.62).  In A
  0.046 against 0.0043 (9 and 9); in E 0.0085 against 0.0061 (8 and 7).  The rollouts sample from the start and leave
  the stored path before the stop, so the stop's number is one part of what a group explores.

**Reading (mine).**  The model stops because it learned to stop there, not because it hesitates.  A and E are
learnable by RL from where they stop, at a slow rate: a group has a sample on the way at about one A stop in five
and one E stop in six or seven, and the greedy choice changes only after the update has moved about 5 nats, which
fits the focused run's one case per update.  D, C and I are learnable at the earlier turn, where the right object
is about 2 nats behind and the episode's reward reaches that step (every step of a sample shares its advantage).  B
and the misspelled name are out of reach by sampling: B needs the demonstrations to look at the breakers (the
unconditional re-inspection of Part 1), and the name needs a direct fix.  F is not learnable from the evidence.  The
demonstration fix (look at the breakers again after each correction and before finishing; answer a refused fit by
looking again) acts on A and B at their source: A needs a 5-nat shift RL makes slowly, B an 8-nat one it does not
make.

## Against the focused run's second evaluation, 2026-11-21

The focused run's second evaluation (after two updates; `research/hpc/minimal_rl_20261116/README.md`) solves 200
of the 247, 7 of the 49 cases this diagnostic scored and 5 fewer of the others (lost 5).  Taking each case's
smallest lead, at the stop or at the first wrong turn:

| Smallest lead (nats) | Cases of the 49 | Solved after two updates | Patterns |
| --- | --- | --- | --- |
| under 1 | 7 | 5 | A 4, D 1, E 1, F 1 |
| 1 to 2 | 6 | 1 | D 3, A 1, C 1, E 1 |
| 2 to 3 | 12 | 1 | A 4, E 4, D 2, C 1, I 1 |
| 3 to 6 | 8 | 0 | A 3, D 3, B 1, E 1 |
| 6 or more | 16 | 0 | B 6, A 5, E 3, F 1, G 1 |

The larger probability of a step on (stop or first wrong turn) tells the 7 solved cases from the 42 others with an
area under the curve of 0.92 (the stop alone 0.81).  Every gained case had a lead of at most 2.1 nats at one of
the two points; the first update's gains that did not hold (r0_7449ca0d5cd6, F, and r0_c35a1e4bfa40, D) were within
0.9 and 0.4 nats at their first wrong turn.  In training, 162 of the 500 scored cases are within 3 nats (A 47 of
144, B 7 of 90, C 10 of 14, D 37 of 56, E 50 of 158, F 2 of 16, G 1 of 8, I 8 of 14).

**Reading (mine).**  The scores predict what the run moves: two updates flipped the cases within about a nat and
none beyond 3.  With 24 of the 49 beyond 3 nats (16 beyond 6, among them 6 of the 7 B cases), the run's reach in a
few more updates is the 25 within 3 nats, less what it loses elsewhere (5 so far).

## Against the focused run's third evaluation, 2026-11-21

The third evaluation (after three updates, the run's last) solves 202 of the 247: 10 of the 49 cases this
diagnostic scored, and 6 fewer of the others.  By each case's smallest lead:

| Smallest lead (nats) | Cases of the 49 | Solved after two updates | Solved after three | Patterns of the solved, after three |
| --- | --- | --- | --- | --- |
| under 1 | 7 | 5 | 5 | A 4, E 1 |
| 1 to 2 | 6 | 1 | 1 | D 1 |
| 2 to 3 | 12 | 1 | 3 | A 2, E 1 |
| 3 to 6 | 8 | 0 | 0 | |
| 6 or more | 16 | 0 | 1 | B 1 |

By pattern, solved after three updates: A 6 of 17, B 1 of 7, D 1 of 9, E 2 of 10, the rest 0 of 6.  The smallest
lead tells the 10 solved cases from the 39 others at an area under the curve of 0.82 (the stop alone 0.76; after
two updates 0.92 and 0.81).

The third update's new gains:
- two A cases at 2.1 nats (r0_018a7be478a1, r0_243985153e00);
- one B case at 8.2 nats (r0_566f34094a8a, cb2_mixed), which now inspects the breakers, corrects one, inspects
  again and corrects the other.

The first update's two gains that did not hold (r0_7449ca0d5cd6, F, 0.9 nats; r0_c35a1e4bfa40, D, 0.4) stayed
unsolved.

**Reading (mine).**  As after two updates, the run moved the near cases: 9 of its 10 gains were within 2.2 nats at
one of the two points.  Over three updates it reached 10 of the 49 and lost 6 of the 198 the supervised model
solved; the 24 cases beyond 3 nats moved once.
