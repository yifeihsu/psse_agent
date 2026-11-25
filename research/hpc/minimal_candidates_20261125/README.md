# The two read-outs of 2026-11-25: partial repairs, and the model's own set of plausible errors

Asked by the user on 2026-11-25: the success criterion asks for every error of a scenario to be resolved; how are
partially resolved scenarios evaluated, and where the agent cannot decide, could it report a set of plausible
errors with the probabilities the LLM gives candidate error types or tool calls at the next decision step, counting
the diagnosis as successful, to the extent the evidence allows, when the true error is in the set?  The answer
given the same day: a partial repair is a plain failure in the count and in the reward; the model's probabilities
can be read exactly; the criterion works as a tier reported beside "solved of the 247" under three conditions (a
reference, not the agent, decides that the evidence is insufficient; the set has a fixed size; credit goes only to
an honest ending).  Two read-outs were offered and the user said "Yes, build both".  Both are built here.  Neither
changes the reward, the scorer or a prompt, and no tier is added to the result: that is the user's decision still.

## 1. The partial-repair breakdown (local, done)

`research/minimal_env/repair_breakdown.py`.  Every evaluation summary of `research/minimal_env/evaluate.py` now
carries it (`summary["repairs"]`), each case's line in `evaluation.json` carries its errors injected, repaired and
healthy objects damaged, and the command reads evaluations that are already stored:

```bash
python -m research.minimal_env.repair_breakdown <start> <evaluation> ... --names A,B,... --reference research/hpc/minimal_sft_20261115/dev300_reference.json
```

It reproduces the counts read by hand on 2026-11-25 (`research/hpc/minimal_rl_20261122/README.md`, section 9) for
the supervised model and stage A's three evaluations, and for the rule expert and the truth-built sequences.
Development suite, 300 cases, 643 injected errors (188 breakers, 346 lines, 109 meters):

| | Supervised (start) | 17 steps | 41 steps | 65 steps | Rule expert | Truth-built |
| --- | --- | --- | --- | --- | --- | --- |
| Solved | 198 | 205 | 198 | 187 | 187 | 247 |
| Every error repaired, a healthy object damaged | 1 | 3 | 7 | 6 | 1 | 0 |
| Some errors repaired | 86 | 78 | 78 | 83 | 90 | 48 |
| No error repaired | 15 | 14 | 17 | 24 | 22 | 5 |
| Errors repaired, of 643 | 520 | 530 | 522 | 506 | 501 | 583 |
| Breakers, of 188 | 159 | 172 | 162 | 159 | 165 | 188 |
| Lines, of 346 | 256 | 259 | 262 | 246 | 247 | 287 |
| Meters, of 109 | 105 | 99 | 98 | 101 | 89 | 108 |
| Mean share of a case's errors repaired | 0.816 | 0.833 | 0.821 | 0.792 | 0.786 | 0.912 |
| Healthy objects damaged (episodes) | 51 (27) | 41 (29) | 46 (34) | 75 (46) | 36 (26) | 0 (0) |
| Unsolved: `resolved` declared (estimator passes at the end) | 72 (52) | 64 (46) | 72 (51) | 76 (52) | 69 (69) | 0 |
| Unsolved: `unresolved` declared (estimator passes) | 27 (0) | 31 (0) | 27 (0) | 31 (0) | 44 (0) | 53 (40) |
| Unsolved: out of actions | 3 | 0 | 3 | 6 | 0 | 0 |
| Some errors repaired, nothing damaged, `unresolved` declared | 13 | 12 | 10 | 5 | 22 | 48 |
| Unsolved with 1 / 2 / 3 errors left (0: damage only) | 81 / 18 / 2 (1) | 73 / 17 / 2 (3) | 72 / 20 / 3 (7) | 79 / 26 / 2 (6) | 86 / 22 / 4 (1) | 47 / 5 / 1 (0) |

The rule expert's 187 is by this object-level audit; its usual 184 is by the strict audit.  New in these rows:

* **Honest partial repairs are rare and RL made them rarer.**  A partial repair that damages nothing and says
  `unresolved` is 13 of the supervised model's 86 partial episodes, 12, 10 and 5 after 17, 41 and 65 steps.  The
  truth-built sequences end that way in 48 of 48.
* **A false claim usually has the estimator's agreement.**  Of the supervised model's 72, the estimator passes at
  the end of 52; of the rule expert's 69, all 69.  Neither a model nor the rule expert declares `unresolved` on a
  state whose estimate passes; the truth-built sequences do in 40 of their 53.
* **Case by case against the supervised model** (all 300): after 17 steps more errors end repaired in 23 cases and
  fewer in 13 (10 more errors, 10 fewer damaged objects); after 41 steps 22 and 18 (+2, -5); after 65 steps 18 and
  29 (-14 errors, +24 damaged objects).  Solved gained and lost are 20 and 13, 19 and 19, 15 and 26 (p = 0.30,
  1.00, 0.12), as recorded before.
* **The 53 cases no sequence repairs:** the supervised model repairs some errors in 48 and none in 5, the same as
  the truth-built sequences (48 and 5), and 63 of their 126 errors against 66.

## 2. The candidate read-out (built, not run)

`research/minimal_env/candidate_sets.py`, three commands: `build` (local), `score` (the model; this cell's job),
`read` (local).

**What is asked.**  At the state an unrepaired episode's last action was chosen on: which corrections does the
model give the most probability, and are the objects still wrong among the 1, 3 and 5 most probable?

**The candidates** come from the observation alone: one `correct_parameters` call per line the prompt lists (20),
one `correct_measurements` call per meter it lists (122 or 125), both statuses of every breaker a table of the
prompt shows or an earlier call of the episode named (0 to 74 breakers), and beside them the two `finish` calls,
`undo` (bare and to each other saved version) and the two `inspect` views.  The development stops list 149 to 297
calls, 206 on average.  A breaker the prompt does not name is no candidate: such an error counts as not covered
and is counted apart.

**The probabilities are exact.**  A call's probability is the product over its tokens, as in `score_actions.py`.
The calls of a state are laid out as a tree of tokens.  One forward pass over a call gives the model's next-token
distribution at every token of it, which is the exact probability of every branch that leaves its path, and a
branch's probability bounds every call under it.  The tree is searched best first: the most probable correction
is found, then the next, until five distinct objects are found; every call left unscored is then no more probable
than the fifth.  The search is checked against scoring every call, on random token trees and on a small torch
model through the RL update's own forward pass (`test_candidate_sets.py`).  A state may take 96 passes at most;
one that needs more is reported as stopped at its limit and its coverage as undecided where it is.

**The truth** (the errors left at the stop, from `stopping_states.py`) rides beside the observation in an input
and never reaches a prompt.  It names which calls are scored whatever their probability (the calls on the objects
still wrong, so their probability is exact wherever they stand) and it counts the coverage.  It changes no
probability and decides no candidate.

**What `read` reports**, per arm: coverage at 1, 3 and 5 objects (an object still wrong in the set; every one in
it), over all stops with an error left and over the stops where every object still wrong can be named; the kind of
error against the three correction tools; whether a first step of the truth-built continuation is among the most
probable calls that go on; the probability of the model's own last action, of each `finish`, of `inspect
breakers`, of any correction, and the share of the correction mass on the objects still wrong; the most probable
object's share against whether it is one still wrong (is a confident set right more often); and, with
`--probe-report`, the model's set against the evidence probe's place for the same stop (the probe of 2026-11-20
places the object first in 37 of 48 solvable stops, in the top three in 43, in the top five in 45).

**Not covered, stated plainly.**  A group of meters is one call the candidates do not list (6 of the 300 cases
hold two or three bad meters): each meter is a candidate on its own.  The stops of *repaired* episodes are not
scored, so the read-out says nothing yet about how much the model would name where nothing is wrong.  `inspect`
with a version or a flip list is counted in the tool's probability and not as a call.

### Files

| File | What it is |
| --- | --- |
| `cell.env` | Paths and the job environment: work directory `/scratch/yx3882/minimal_candidates_20261125`, the two arms and their adapters (read in place), the inputs under `data/`, the scores under `out/<arm>`. |
| `stage_candidates.sbatch` | The job of one arm (`ARM`): two processes on a card of 70 GB or more, each with half of the states, then the closing call that scores what is left and writes `summary.json`.  Preemptible and requeued; one file per state. |
| `submit_cell.sh` | `score <arm>`: submits the arm's job (`mcs-<arm>`); refuses before the arm's adapter or inputs, once its summary is written, and while a job of the name is queued. |
| `status_cell.sh` | Read-only: inputs, queue, `sacct`, and per arm the states written, the logs' last lines and the coverage. |
| `deploy_from_windows.sh` | Clones the RL cell's source into a source of this cell's own (never moving that one), applies the incremental bundle, and uploads the inputs of every arm that has them here, checksums verified.  Submits nothing.  It sends the commit: tracked files that differ in the checkout are named and not sent. |
| `test_cell.py` | The scripts parse, name this cell and write nothing of another cell; the stage runs the shards and the closing call for the arm it is given (with a stub scorer). |

### Arms

| Arm | Model | Its development evaluation | Stops |
| --- | --- | --- | --- |
| `base` | the supervised pass's adapter, every RL run's start | 198 of the 247 | 102 unsolved episodes, 101 with an error left |
| `steps17` | stage A's adapter after 17 optimizer steps (iteration 2) | 205 of the 247 | 95 unsolved episodes, 92 with an error left |

### Order of work, each cluster step on the user's word

1. The inputs, on Windows (done for both arms):
   `python -m research.minimal_env.stopping_states --suite <ieee14_dev300.json> --episodes <evaluation dir> --split dev --output-dir <dir>`,
   then `python -m research.minimal_env.candidate_sets build --scoring-inputs <dir>/scoring_inputs_dev.jsonl --episodes <dir>/episodes_dev.jsonl --output output/minimal_candidates_20261125/candidate_inputs_dev_<arm>.jsonl`.
2. Deploy: `bash research/hpc/minimal_candidates_20261125/deploy_from_windows.sh <commit40>`.
3. An arm: `wsl -- ssh torch bash -s -- score base < research/hpc/minimal_candidates_20261125/submit_cell.sh`.
4. Status: `wsl -- ssh torch bash -s < research/hpc/minimal_candidates_20261125/status_cell.sh`.
5. Read, on Windows, after copying `out/<arm>/states` back:
   `python -m research.minimal_env.candidate_sets read --scores <dir> --reference research/hpc/minimal_sft_20261115/dev300_reference.json --probe-report output/minimal_env_20261115/stopping_states_20261120/part1/probe_report.json`.

### State

Built 2026-11-25.  **Not deployed, nothing submitted.**  The cell has no work directory on torch yet.

Checked on Windows the same day:

* **The inputs of `base`**: 102 stopping states from the replay of 2026-11-20
  (`output/minimal_env_20261115/stopping_states_20261120/part1`), 20,976 candidate calls; 81 states with one error
  left, 18 with two, 2 with three, 1 with none (damage only).  In
  `output/minimal_candidates_20261125/candidate_inputs_dev_base.jsonl` (local, not committed).
* **A dry run on all 102 states with the real prompts and the real call tokens** (the local Gemma 4 tokenizer and
  chat template through the project's own renderers) and a stand-in for the model's probabilities: every candidate
  renders and reads back, no call is the beginning of another, no prompt is cut (9,925 tokens at the median,
  14,558 at most), and a call's tokens are the same at every state (so they are rendered once and checked at each
  state: 0.2 s a state after the first).  The calls on the objects still wrong, the other calls and the model's
  own take 8 passes a state at the median and 12 at most; with a peaked stand-in the search ends after 16 passes at
  the median (20 at most), with a flat one it stops at the limit of 96 at every state and reports it.
* **What the stops allow**: an object still wrong has a candidate for 90 of 90 lines, 4 of 4 meters and 18 of 29
  breakers.  At 90 of the 101 stops every object still wrong can be named, at 9 none can.
* **The inputs of `steps17`**: stage A's evaluation after 17 steps was copied from torch (read-only) and replayed
  here on 2026-11-25: 300 of 300 episodes replay to their stored results, 205 solved, 95 unsolved (73 with one error
  left, 17 with two, 2 with three, 3 with none), 19,609 candidate calls.  The errors left are 87 lines, 16 breakers
  and 10 meters (the supervised model's stops: 90, 29 and 4); every line and meter has a candidate and 14 of the 16
  breakers; at 90 of the 92 stops every object still wrong can be named, at 1 none can.  The same dry run passes on
  all 95 states (16 passes at the median with the peaked stand-in).  In
  `output/minimal_candidates_20261125/candidate_inputs_dev_steps17.jsonl`, the replay beside it in `stops_steps17/`.
* **The model's side is not run here**: no environment on this machine loads this model family.  It is the RL
  update's forward pass with its last step left out (`position_logprobs`), compared with `completion_logprobs` on a
  small torch model; the loader and the renderers are the ones the job of 2026-11-20 ran.  The job's first states
  are its first run on the model.
* **On torch, read-only**: both adapters are there, the job environment is there, the source to clone from is at
  98e6518, and nothing of ours is queued.

**Cost, an estimate.**  A forward pass of these prompts took about 3.2 s with two processes on a card on
2026-11-20.  At 16 to 40 passes a state the arm `base` is 1.5 to 3.5 card-hours; at the limit of 96 at every state
it is 8.7.  The arm `steps17` has 95 states against 102 and costs about the same.  How peaked the model is decides
it, and the job's log gives each state's passes.
