# The two read-outs of 2026-11-25: partial repairs, and the model's own set of plausible errors

Asked by the user on 2026-11-25: the success criterion asks for every error of a scenario to be resolved; how are
partially resolved scenarios evaluated, and where the agent cannot decide, could it report a set of plausible
errors with the probabilities the LLM gives candidate error types or tool calls at the next decision step, counting
the diagnosis as successful, to the extent the evidence allows, when the true error is in the set?  The answer
given the same day: a partial repair is a plain failure in the count and in the reward; the model's probabilities
can be read exactly; the criterion works as a tier reported beside "solved of the 247" under three conditions (a
reference, not the agent, decides that the evidence is insufficient; the set has a fixed size; credit goes only to
an honest ending).  Two read-outs were offered and the user said "Yes, build both".  Both are built here.  Neither
changes the reward, the scorer or a prompt.

**Decided by the user on 2026-11-26, after the result of the arm `base` in section 2: "Keep it a read-out".**  No
tier is added.  "Solved of the 247" stays the one result, the breakdown of section 1 stays in every evaluation
summary, and the candidate read-out of section 2 is a diagnostic that runs when it is asked for.

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

## 2. The candidate read-out (the arm `base` done 2026-11-26; the arm `steps17` submitted 2026-11-26)

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
places the object first in 37 of 48 solvable stops, in the top three in 43, in the top five in 45, when it is
given the substation table an `inspect breakers` would show).

**Not covered, stated plainly.**  A group of meters is one call the candidates do not list (6 of the 300 cases
hold two or three bad meters): each meter is a candidate on its own.  The stops of *repaired* episodes are not
scored, so the read-out says nothing yet about how much the model would name where nothing is wrong.  `inspect`
with a version or a flip list is counted in the tool's probability and not as a call.

### Result of the arm `base`: the supervised model's 102 stops

Job 24643723 ran from 2026-11-25 23:32 to 2026-11-26 01:00 UTC and scored all 102 states.  No search stopped at
its limit, no prompt was cut and no call failed to render: the five most probable objects are exact at every
state.  The scores are in `output/minimal_candidates_20261125/scores_base/` (local, not committed), the reading in
`summary_base.json` beside them.  One model, one evaluation, 101 stops: the counts below are small.

**Coverage.**  The set is the model's 1, 3 or 5 most probable correction objects at the stop.  A state lists 142
to 218 correction objects (20 lines, 122 or 125 meters, 0 to 74 breakers).

| Stops with an error left | Stops | An object still wrong is in the set of 1 / 3 / 5 | Every object still wrong is |
| --- | --- | --- | --- |
| All | 101 | 31 / 45 / 52 | 26 / 39 / 44 |
| Every object still wrong can be named | 90 | 31 / 45 / 51 | 26 / 39 / 44 |
| Reference: the rule expert solves the case | 22 | 3 / 6 / 6 | 1 / 3 / 3 |
| Reference: only the truth-built sequence repairs it | 26 | 9 / 10 / 14 | 9 / 10 / 12 |
| Reference: no sequence repairs it | 53 | 19 / 29 / 32 | 16 / 26 / 29 |
| The model ended with `finish resolved` | 71 | 24 / 34 / 39 | 23 / 33 / 37 |
| The model ended with `finish unresolved` | 27 | 6 / 10 / 12 | 3 / 5 / 6 |
| The model ran out of actions | 3 | 1 / 1 / 1 | 0 / 1 / 1 |

The 102nd stop has no error left: `r0_c2b40c5df2b0` repaired its errors, damaged one healthy object and declared
`resolved`.

**The set is lines; a breaker still wrong is almost never in it.**  The most probable object is a line at 74 of
the 102 stops, a meter at 25 and a breaker at 3.

| Errors left | Errors | Can be named | Among the 1 / 3 / 5 most probable objects |
| --- | --- | --- | --- |
| Lines | 90 | 90 | 31 / 46 / 52 |
| Breakers | 29 | 18 | 0 / 0 / 1 |
| Meters | 4 | 4 | 0 / 1 / 3 |
| Never corrected in the episode | 87 | 76 | 31 / 44 / 49 |
| Corrected, the result not within tolerance | 35 | 35 | 0 / 2 / 6 |
| Corrected, then undone | 1 | 1 | 0 / 1 / 1 |

A line drawn at random among the 20 would be the one still wrong 1 time in 20, and one of five lines 1 time in 4.
The call on a breaker still wrong gets 0.00005 at the median of the 19 such calls scored and 0.0024 at most.

**The kind of error against the tool the model gives the most probability.**

| Errors left at the stop | Stops | `correct_parameters` | `correct_topology` | `correct_measurements` | Two tools equal |
| --- | --- | --- | --- | --- | --- |
| Breakers only | 23 | 17 | 2 | 3 | 1 |
| A breaker and a line | 3 | 2 | 0 | 1 | 0 |
| Lines only | 71 | 49 | 0 | 20 | 2 |
| A line and a meter | 3 | 1 | 1 | 1 | 0 |
| A meter only | 1 | 0 | 0 | 1 | 0 |

At 3 stops the line fit and the meter correction have exactly the same probability (the scores come from
half-precision logits, where equal values occur); until 2026-11-26 this table counted those three with the line
fit, 18 and 51 in its first column.  No two objects are equal across the edge of a set, so the coverage tables
are not touched by it.

At the 23 stops where only breakers are still wrong the line fit holds 0.80 of the correction mass at the median
and the topology tool 0.011.  At 36 of the 101 stops the prompt names no breaker and no topology call is scored
(one of the 29 breakers still wrong is at such a stop): that tool's probability is only bounded there by what the
scored calls leave (0.0007 at the median, 0.12 at most).  The bound decides the most probable tool at 32 of the 36;
the other 4 stand in the table under the leading tool of the two measured.  Over the 97 decided stops the most
probable tool is a tool of an error left at 54, another tool at 40, and two tools are equal at 3 (35, 27 and 3 of
the 65 with all three tools measured; 19 and 13 of the 32); a tool drawn at random would be one of an error left
at 0.35 of them.  (`read` itself reports the tool only where all three are measured and takes the first of two
equal tools in its own order: 37 of 65.)

**The probabilities are small: at the stop the model is sure of the action it took.**

| Probability at the stop | 10th | 25th | Median | 75th | 90th |
| --- | --- | --- | --- | --- | --- |
| The model's own last action | 0.569 | 0.771 | 0.936 | 0.996 | 0.999 |
| `finish resolved` | 0.0020 | 0.117 | 0.914 | 0.996 | 0.999 |
| `finish unresolved` | 0.00001 | 0.00003 | 0.0006 | 0.485 | 0.771 |
| `inspect breakers` | 0.00001 | 0.00003 | 0.00009 | 0.0003 | 0.0013 |
| Any correction | 0.0005 | 0.0013 | 0.012 | 0.118 | 0.319 |
| A correction of an object still wrong (101 stops) | 0.0000004 | 0.00002 | 0.00015 | 0.011 | 0.070 |
| Share of the correction mass on the objects still wrong (101) | 0.00003 | 0.0012 | 0.028 | 0.414 | 0.876 |
| Share of the correction mass on the most probable object (101) | 0.251 | 0.381 | 0.610 | 0.828 | 0.933 |

By tool, at the median: `correct_parameters` 0.0045, `correct_measurements` 0.0008, `correct_topology` 0.00005
(65 stops).  At the 72 stops ending with `finish resolved` that call holds 0.989 at the median and any correction
0.0036; at the 27 ending with `finish unresolved`, 0.702 and 0.047.  A set is therefore a ranking inside about
1% of the probability, not a doubt the model states.  The ranking's own confidence does not mark the right sets:
where the most probable object holds 0.9 or more of the correction mass it is one still wrong at 9 of 15 stops,
between 0.5 and 0.9 at 15 of 46, under 0.5 at 7 of 40.

**Against the evidence probe**, on the 48 solvable stops with an error left (the 22 and the 26 above).  The probe
has two readings: with the substation table of the current state, which is one `inspect breakers` away (the
default of `read`, `--probe-reading candidate_inspect`), and with that table only where the prompt shows it
(`candidate_shown`).

| An object still wrong among the 1 / 3 / 5 first | All 48 | A breaker still wrong (26) | Lines or meters only (22) |
| --- | --- | --- | --- |
| Probe, with the breakers view | 37 / 43 / 45 | 22 / 24 / 24 | 15 / 19 / 21 |
| Probe, on what the prompt shows | 19 / 25 / 26 | 3 / 5 / 6 | 16 / 20 / 20 |
| The model's set | 12 / 16 / 20 | 0 / 0 / 2 | 12 / 16 / 18 |

With the breakers view, model and probe together: both 10 / 15 / 19, the probe only 27 / 28 / 26, the model only
2 / 1 / 1, neither 9 / 4 / 2.  On lines and meters the model's set is close to the probe's.  The difference is the
breakers: the breakers view holds the object (24 of 26 at three objects) and without it the probe finds it at 5.
At those 26 stops the model gives `inspect breakers` 0.0002 at the median (over 0.01 at 4): it neither asks for
the view nor names the breaker.

**A first step of the truth-built continuation** is the most probable call that goes on (any listed call but
`finish`) at 15 of the 51 stops that have such a continuation, among the three most probable at 20 and the five at
22 (the 49 solvable stops: 15, 19 and 21).

**Counting what the agent already corrected** (a reading added on 2026-11-26, not part of `read`).  35 of the 123
errors left (34 lines, 1 meter) are objects the agent itself corrected, the result outside the tolerance, and at
23 stops nothing else is left: a report of its kept corrections names those objects without any set.  With them
counted, every object still wrong is either corrected or in the set of 1 / 3 / 5 at 52 / 65 / 66 of the 101 stops:
37 / 47 / 48 of the 53 no sequence repairs, 15 / 18 / 18 of the 48 solvable ones.

**What a tier would count** under the three conditions named at the top, with the set fixed at three.  The
reference says no sequence repairs the case at 53 stops.  Every object still wrong is in the set at 26 of them,
and corrected or in the set at 47.  The model ends 45 of the 53 with `finish resolved` (the estimator's test
passes at 36 of those), 7 with `finish unresolved` (4 after damaging a healthy object) and 1 out of actions.  Of
the 47, 3 end with `unresolved` and 2 of those damaged nothing.  The tier would count 47 cases without the
condition on the ending and 2 with it.

**The decision (the user, 2026-11-26): no tier; this stays a read-out.**  The recommendation put to the user with
these numbers was to keep it a read-out, for three reasons.  Without the condition on the ending, a tier credits
the model for a doubt it does not state: it ends 45 of the 53 with `finish resolved`, and at the 72 stops it ends
that way all corrections together hold 0.0036 of the probability at the median.  With that condition it counts 2
cases.  And a count built on the set needs this scoring job, about an hour and a half of one card, for every
evaluation.  The user's answer was "Keep it a read-out".  Left open, and not decided here: whether the agent
itself should state the doubt (end `unresolved` where an error is left, ask for the breakers view before it
stops).  That is a training question, the calibrated-abstention item of the RL work.

**The run.**  1,712 forward passes for the 20,976 candidates: 11 to 29 a state, 17 at the median, 7 to 12 of them
for the calls that are always scored.  50 to 229 s a state (97 at the median, 5.4 s a pass with two processes on
one RTX PRO 6000 Blackwell card of 98 GB), 1 h 28 min for the arm.  The prompts are 7,925 to 14,562 tokens: four
more than the local dry run rendered, at the median and at the maximum, and the same as on 2026-11-20 with this
model.  The scored calls of a state hold 0.998 of the probability at the median, 0.88 at least and never more
than 1.

**Checked against the scores of 2026-11-20** (`score_actions.py`, one pass a call): 49 of the 102 stops were
scored then, with 170 calls in both runs.  The prompt lengths are equal at all 49.  162 calls agree to five
decimals of the log-probability, and the 97 calls of probability over 0.001 agree to 0.00005.  Four differ by more
than 0.001, all on calls of probability under 0.0002 (by 0.11 at -9.1, 0.84 at -18.6, 0.03 at -19.4 and 0.001 at
-61.7), where one step of a half-precision logit (0.03 to 0.25) shows in full.  What moves the last bit there (a
shared beginning read from the pass over another call, or the card type) was not determined.  It can move a
coverage count by a few and not more: an object still wrong stands within 0.25 of its set's edge in 8 of the 303
verdicts of the table's first row (2 at one object, 2 at three, 4 at five).  The fifth place itself is often
close: at 59 of the 101 stops the fifth object is within 0.25 of the best one outside the set.

### Files

| File | What it is |
| --- | --- |
| `cell.env` | Paths and the job environment: work directory `/scratch/yx3882/minimal_candidates_20261125`, the two arms and their adapters (read in place), the inputs under `data/`, the scores under `out/<arm>`. |
| `stage_candidates.sbatch` | The job of one arm (`ARM`): two processes on a card of 70 GB or more, each with half of the states, then the closing call that scores what is left and writes `summary.json`.  Preemptible and requeued; one file per state. |
| `submit_cell.sh` | `score <arm>`: submits the arm's job (`mcs-<arm>`); refuses before the arm's adapter or inputs, once its summary is written, and while a job of the name is queued. |
| `status_cell.sh` | Read-only: inputs, queue, `sacct`, and per arm the states written, the logs' last lines and the coverage. |
| `deploy_from_windows.sh` | Clones the RL cell's source into a source of this cell's own (never moving that one), checks the commit out there (where the clone does not hold it, the incremental bundle goes in first), and uploads the inputs of every arm that has them here, checksums verified.  Submits nothing.  The commit is the branch's tip or a commit below it, so another session's later commit on the branch does not come along; tracked files that differ in the checkout are named and not sent. |
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

Built 2026-11-25 and, on the user's word the same day ("Yes, deploy and run", for the supervised model's stops),
deployed at d748912 and the arm `base` submitted at 23:24 UTC as job 24643723 (`mcs-base`).  **The arm `base` is
done: the job ended COMPLETED on 2026-11-26 at 01:00 UTC with all 102 states scored, and its result is above.**
**The tier is decided: none is added** (the user on 2026-11-26: "Keep it a read-out").

**The arm `steps17` is submitted**: on the user's word of 2026-11-26 ("Yes, run it") at 23:19 UTC as job 24782581
(`mcs-steps17`), from the same source (d748912) and the inputs checked below (95 states, the same checksum here
and on torch).  Nothing of it is read yet.

* **The commit on torch is d748912**, the one the tests ran on, not the branch's tip: another session had put
  its stage B commit (600049b) on the branch and redeployed the RL cell at it, so the clone arrived at 600049b
  and was checked out at d748912.  Below this cell's own files, d748912 is the code stage A ran (98e6518) with
  the scorer's loader moved into `score_actions.model_parts`.
* **Before the submit, on the login node with the job's Python** (3.12.13, torch 2.10.0, transformers 5.10.4,
  peft 0.19.1): the source is clean at d748912, both input files arrived with their checksums (102 states and
  20,976 candidates for `base`), and the 32 tests of `test_candidate_sets.py` pass there.

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
  were its first run on the model, and they were read before the rest: the calls also scored on 2026-11-20 came
  out the same (the check above, then on all 49 shared stops).
* **On torch, read-only, before the deploy**: both adapters are there and the job environment is there.

**Cost, measured.**  The arm `base` took 1 h 28 min of one card: 5.4 s a forward pass with two processes on the
card (the estimate before the run was 3.2 s, from 2026-11-20), 17 passes a state at the median and 29 at most, far
from the limit of 96.  The arm `steps17` has 95 states against 102 and should cost about the same if that model is
as peaked.
