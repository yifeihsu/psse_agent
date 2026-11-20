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

Built 2026-11-20. Part 1 done on Windows the same day (below). Part 2 not yet deployed; nothing submitted.  The
scoring inputs are in `output/minimal_env_20261115/stopping_states_20261120/`: 74 development states (49 stopping
states, 25 first wrong turns) with 240 actions, and 711 training states (502 and 209) with 2,324 actions.

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
