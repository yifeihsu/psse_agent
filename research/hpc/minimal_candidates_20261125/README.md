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

## 2. The candidate read-out (both arms done: `base` on 2026-11-26, `steps17` on 2026-11-27)

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
given the substation table an `inspect breakers` would show).  Two objects, calls or tools exactly as probable on
the two sides of a set's edge are reported as not decided (since 2026-11-26; before, the first in the order of
listing was taken): the scores come from half-precision logits, where equal values occur.

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
at 0.35 of them.  (`read` itself reports the tool only where all three are measured: 35 of 65, with 2 not decided
where the two equal tools are the right one and another.)

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

### Result of the arm `steps17`: the model after 17 RL steps, 95 stops

Run on the user's word of 2026-11-26 ("Yes, run it").  Job 24782581 ran from 2026-11-26 23:23 to 2026-11-27 00:47
UTC and scored all 95 states.  No search stopped at its limit, no prompt was cut and no call failed to render.
The scores are in `output/minimal_candidates_20261125/scores_steps17/` (local, not committed), the reading in
`summary_steps17.json` beside them.  The supervised model's numbers of the section above stand beside the new ones.
The two models stop in different cases and at different states, so the tables over all stops compare two sets of
states; only the part on the 44 shared states compares the two models on the same thing.

**In short.**

1. **The set is no better.**  An object still wrong is in the 17-step model's set of 1 / 3 / 5 at 32 / 45 / 49 of
   its 92 stops with an error left; the supervised model's, 31 / 45 / 52 of 101.  A breaker still wrong is in it
   at 0 of 16.
2. **What the 17 steps changed is how sure the model is that it should stop.**  At the 44 states where both
   models stop on the same state, `inspect breakers` is 30 times as probable at the median and higher at all 44,
   `finish unresolved` 5 times, the corrections 2.6 times; which objects lead among the corrections hardly
   changes (13 / 19 / 21 of the 44 against 14 / 20 / 21).
3. **In the episodes the change is a second look at the breakers after a correction.**  56 of the 300 episodes
   hold one against 23, and 14 of the 20 cases the 17 steps gained against 1.

**Coverage**, the two models side by side.  The set is the model's 1, 3 or 5 most probable correction objects;
in brackets, every object still wrong is in it.

| Stops with an error left | Supervised: stops | In the set of 1 / 3 / 5 | 17 steps: stops | In the set of 1 / 3 / 5 |
| --- | --- | --- | --- | --- |
| All | 101 | 31 / 45 / 52 (26 / 39 / 44) | 92 | 32 / 45 / 49 (27 / 36 / 40) |
| Every object still wrong can be named | 90 | 31 / 45 / 51 (26 / 39 / 44) | 90 | 32 / 45 / 49 (27 / 36 / 40) |
| Reference: the rule expert solves the case | 22 | 3 / 6 / 6 (1 / 3 / 3) | 15 | 5 / 6 / 6 (3 / 4 / 5) |
| Reference: only the truth-built sequence repairs it | 26 | 9 / 10 / 14 (9 / 10 / 12) | 24 | 7 / 10 / 13 (6 / 8 / 10) |
| Reference: no sequence repairs it | 53 | 19 / 29 / 32 (16 / 26 / 29) | 53 | 20 / 29 / 30 (18 / 24 / 25) |
| The model ended with `finish resolved` | 71 | 24 / 34 / 39 (23 / 33 / 37) | 61 | 22 / 30 / 33 (22 / 29 / 32) |
| The model ended with `finish unresolved` | 27 | 6 / 10 / 12 (3 / 5 / 6) | 31 | 10 / 15 / 16 (5 / 7 / 8) |
| The model ran out of actions | 3 | 1 / 1 / 1 (0 / 1 / 1) | 0 | |

Three of the 95 stops have no error left: `r0_37ff7e1aa3c5`, `r0_b148f259473a` and `r0_dbf4ff9bc9d9`, all of
the class `control_meter`, repaired their error, damaged one healthy object and declared `resolved` with the
estimator's test passing.

| Errors left | Supervised: errors (can be named) | Among the 1 / 3 / 5 | 17 steps: errors (can be named) | Among the 1 / 3 / 5 |
| --- | --- | --- | --- | --- |
| Lines | 90 (90) | 31 / 46 / 52 | 87 (87) | 30 / 43 / 47 |
| Breakers | 29 (18) | 0 / 0 / 1 | 16 (14) | 0 / 0 / 0 |
| Meters | 4 (4) | 0 / 1 / 3 | 10 (10) | 2 / 3 / 4 |
| Never corrected in the episode | 87 (76) | 31 / 44 / 49 | 71 (69) | 31 / 42 / 43 |
| Corrected, the result not within tolerance | 35 (35) | 0 / 2 / 6 | 39 (39) | 0 / 3 / 7 |
| Corrected, then undone | 1 (1) | 0 / 1 / 1 | 3 (3) | 1 / 1 / 1 |

The most probable object is a line at 71 of the 95 stops and a meter at 24, never a breaker.

**The kind of error against the tool the model gives the most probability** (the 17-step model).

| Errors left at the stop | Stops | `correct_parameters` | `correct_topology` | `correct_measurements` | Two tools equal |
| --- | --- | --- | --- | --- | --- |
| Breakers only | 12 | 8 | 0 | 4 | 0 |
| A breaker and a line | 4 | 3 | 0 | 0 | 1 |
| Lines only | 67 | 47 | 0 | 18 | 2 |
| A line and a meter | 6 | 4 | 0 | 1 | 1 |
| A meter only | 3 | 0 | 0 | 3 | 0 |

At the 12 stops where only breakers are still wrong the line fit holds 0.66 of the correction mass at the median,
the meter correction 0.34 and the topology tool 0.003 (the supervised model's 23: 0.80, 0.07 and 0.011).  At 41 of
the 92 stops no topology call is scored and that tool is only bounded (0.002 at the median of the 44 such states,
0.13 at most); the bound decides the most probable tool at 37 of the 41 and the other 4 stand in the table under
the leading tool of the two measured.  Over the 88 decided stops the most probable tool is a tool of an error left
at 54, another tool at 30, and two tools are equal at 4 (32, 17 and 2 of the 51 with all three tools measured; 22,
13 and 2 of the 37); a tool drawn at random would be one of an error left at 0.37 of them.  The supervised model:
54, 40 and 3 of 97.

**The probabilities at the stop.**

| Probability at the stop | 10th | 25th | Median | 75th | 90th | Supervised, median |
| --- | --- | --- | --- | --- | --- | --- |
| The model's own last action | 0.524 | 0.698 | 0.896 | 0.992 | 0.997 | 0.936 |
| `finish resolved` | 0.0046 | 0.051 | 0.863 | 0.992 | 0.997 | 0.914 |
| `finish unresolved` | 0.0001 | 0.0003 | 0.0032 | 0.519 | 0.775 | 0.0006 |
| `inspect breakers` | 0.0002 | 0.0005 | 0.0016 | 0.0096 | 0.107 | 0.00009 |
| Any correction | 0.0018 | 0.0036 | 0.017 | 0.146 | 0.293 | 0.012 |
| A correction of an object still wrong (92 stops) | 0.00002 | 0.00006 | 0.0005 | 0.011 | 0.092 | 0.00015 |
| Share of the correction mass on the objects still wrong (92) | 0.0006 | 0.0054 | 0.065 | 0.448 | 0.795 | 0.028 |
| Share of the correction mass on the most probable object (92) | 0.188 | 0.272 | 0.394 | 0.652 | 0.813 | 0.610 |

By tool, at the median: `correct_parameters` 0.0065, `correct_measurements` 0.0028, `correct_topology` 0.0001 (51
stops), `undo` 0.0021.  At the 64 stops ending with `finish resolved` that call holds 0.987 at the median, any
correction 0.0057 and `inspect breakers` 0.0008 (the supervised model's 72: 0.989, 0.0036 and 0.00006); at the 31
ending with `finish unresolved`, 0.702, 0.132 and 0.028 (its 27: 0.702, 0.047 and 0.0006).  Where the most
probable object holds 0.9 or more of the correction mass it is one still wrong at 3 of 4 stops, between 0.5 and
0.9 at 19 of 30, under 0.5 at 10 of 58 (the supervised model: 9 of 15, 15 of 46, 7 of 40).  The set is still a
ranking inside one or two percent of the probability.

**The same 44 states under the two models.**  Of the 82 cases both models leave unsolved, 44 stop at the same
state: the same observation, so the same prompt (the token counts are equal at all 44) and the same candidates.
Both models end all 44 the same way, 38 with `finish resolved` and 6 with `finish unresolved`.  31 are cases no
sequence repairs, 10 only the truth-built sequence repairs, 3 the rule expert solves; the errors left are 45 lines
and 4 breakers.  These are states where the 17 steps did not change the decision, so they show what moved under it,
and they are mostly line cases.

| At the same 44 states | Supervised, median | 17 steps, median | Higher at / lower at | Median ratio |
| --- | --- | --- | --- | --- |
| `finish resolved` | 0.992 | 0.980 | 8 / 36 | 0.99 |
| `finish unresolved` | 0.00007 | 0.0004 | 41 / 3 | 5.0 |
| `inspect breakers` | 0.00006 | 0.0010 | 44 / 0 | 30 |
| Any correction | 0.0036 | 0.016 | 36 / 8 | 2.6 |
| `correct_parameters` | 0.0014 | 0.0039 | 34 / 10 | 2.6 |
| `correct_measurements` | 0.0005 | 0.0029 | 33 / 11 | 2.2 |
| `correct_topology` (22 states) | 0.00005 | 0.00009 | 16 / 6 | 1.5 |
| `undo` | 0.0007 | 0.0017 | 37 / 7 | 2.1 |
| A correction of an object still wrong | 0.00006 | 0.0003 | 37 / 7 | 3.5 |
| Share of the correction mass on the objects still wrong | 0.035 | 0.062 | 24 / 20 | 1.3 |
| Share of the correction mass on the most probable object | 0.455 | 0.325 | 4 / 40 | 0.77 |

An object still wrong is in the set of 1 / 3 / 5 at 13 / 19 / 21 of the 44 under the supervised model and at 14 /
20 / 21 under the 17-step model: one state enters the sets of one and of three, none leaves.  The call on a line
still wrong gains what the other corrections gain and keeps its place: +1.0 in log-probability at the median for
the 21 lines never corrected (up at 16, median place 1 and 1), +1.6 for the 24 corrected and not within tolerance
(up at 22, place 6 and 6), +1.2 over all 348 correction calls scored under both models.  The call on a breaker
still wrong falls at 3 of the 4.  So the 17 steps took a little probability from `finish resolved` and gave it to
every other call, most of all to the look at the breakers; they did not teach the model which object to correct.

**Breakers: the model now looks, and still does not name.**

| At the stops with a breaker still wrong | Supervised | 17 steps |
| --- | --- | --- |
| Stops | 26 | 16 |
| Breaker errors left (can be named) | 29 (18) | 16 (14) |
| A breaker still wrong among the 1 / 3 / 5 most probable objects | 0 / 0 / 1 | 0 / 0 / 0 |
| The call on a breaker still wrong: median, highest | 0.00005, 0.0024 (19 scored) | 0.00002, 0.0040 (15 scored) |
| `correct_topology`, median | 0.00006 (25 measured) | 0.0003 (15 measured) |
| `inspect breakers`, median | 0.0002 | 0.054 |
| `inspect breakers` over 0.01 / over 0.1 | 4 / 3 of 26 | 10 / 4 of 16 |
| Ended `finish unresolved` / `finish resolved` / out of actions | 14 / 10 / 2 | 12 / 4 / 0 |
| For comparison, `inspect breakers` at the stops with lines or meters only: median | 0.00008 (75 stops) | 0.0012 (76 stops) |
| ... over 0.01 | 0 of 75 | 13 of 76 |

After the 17 steps the look at the breakers is 45 times as probable where a breaker is still wrong as where only
lines or meters are (0.054 against 0.0012 at the median); for the supervised model the two differ by a factor of
under 3.  The model also ends such a case more often with `unresolved` (12 of 16 against 14 of 26).  All of these
stops are solvable cases.

One episode ended on a tie.  At `r0_ea0fd1033801` (class `cb2_mixed`, breaker `CB_6_B1_B2` still wrong)
`inspect breakers` holds 0.472, `finish unresolved` 0.368 and `finish resolved` 0.106.  It is the one stop of the
95 where the model's last action is not its most probable call: the tools `finish` and `inspect` are exactly
equal at the token that names the tool (0.4738 each), the episode's decoding took `finish`, and the episode
ended with `unresolved`.  The look is a first step of the truth-built continuation there.

**In the episodes** (every call of the two development evaluations, 300 episodes each;
`readings/episode_looks.py`).

| | Supervised | 17 steps |
| --- | --- | --- |
| Solved (of the 247 solvable cases) | 198 | 205 |
| Episodes whose first action is `inspect breakers` | 300 | 300 |
| Episodes with a look at the breakers after a correction | 23 | 56 |
| ... of the 20 cases the 17 steps gained | 1 | 14 |
| ... of the 13 cases they lost | 3 | 3 |
| ... of the 82 cases unsolved under both | 3 | 9 |
| ... of the 185 cases solved under both | 16 | 30 |
| ... of the 35 cases of the two-breaker classes (solved: 15, then 26) | 17 | 30 |
| Actions in all 300 episodes | 1,334 | 1,335 |
| `inspect breakers` / `correct_topology` / `correct_parameters` / `correct_measurements` calls | 323 / 199 / 355 / 133 | 357 / 183 / 360 / 112 |

Every episode of both models opens with the breakers view, taken before any correction.  What the 17 steps added
is the look after a correction: no episode lost it and 33 gained it (13 of them gained cases, 14 solved under
both models, 6 unsolved under both).  In the 20 gained cases the actions fall from 119 to 100, the topology calls
from 44 to 30, the line fits from 23 to 11 and the meter corrections from 10 to 3.  In the 13 lost cases the
second looks do not change and the line fits rise from 11 to 23.  The look comes with the gain; that it made the
gain is not shown by these counts.

The gain could not be read off the supervised model's stops.  In the 20 gained cases the supervised model stops
with breakers only at 14 and with lines only at 6; an object still wrong is in its set of 1 / 3 / 5 at 2 / 5 / 5
of the 20, none of the 16 breakers (9 can be named), and it gives `inspect breakers` 0.0001 there at the median.
The 13 cases the 17 steps lost stop with an error left at 10 (6 lines, 4 breakers, 3 meters; in the set at 3 / 3 /
4) and with damage only at the three `control_meter` stops named above.  The 38 cases both models leave unsolved
at different states: 16 / 21 / 26 of 37 under the supervised model, 15 / 22 / 24 of 38 under the 17-step model.

**Against the evidence probe**, on the 39 solvable stops with an error left.  The probe is the one of 2026-11-20,
fitted on the supervised model's training-split states and read here on the 17-step model's stops
(`output/minimal_candidates_20261125/stops_steps17/probe_report.json`, local; the run reproduces the stored
report's training side exactly, all 502 training stops).

| An object still wrong among the 1 / 3 / 5 first | All 39 | A breaker still wrong (16) | Lines or meters only (23) |
| --- | --- | --- | --- |
| Probe, with the breakers view | 31 / 36 / 37 | 12 / 16 / 16 | 19 / 20 / 21 |
| Probe, on what the prompt shows | 19 / 23 / 24 | 0 / 3 / 4 | 19 / 20 / 20 |
| The model's set | 12 / 16 / 19 | 0 / 1 / 1 | 12 / 15 / 18 |

With the breakers view, model and probe together: both 11 / 15 / 18, the probe only 20 / 21 / 19, the model only
1 / 1 / 1, neither 7 / 2 / 1.  The picture is the supervised model's: on lines and meters the model's set is near
the probe's, and on breakers the current breakers view holds the object (16 of 16 at three objects) while what the
prompt shows does not (3 of 16).  The one object in the model's set at a breaker stop is a line.

**Other readings.**  A first step of the truth-built continuation is the most probable call that goes on at 9 of
the 44 stops that have such a continuation, among the three most probable at 20 and the five at 23 (the 41
solvable ones: 8, 19 and 21; the supervised model: 15, 20 and 22 of 51).  Counting what the agent already
corrected: 39 of the 113 errors left are objects the agent itself corrected with the result outside the
tolerance, and at 27 stops nothing else is left; with them counted every object still wrong is corrected or in
the set of 1 / 3 / 5 at 56 / 65 / 65 of the 92 stops, 42 / 47 / 47 of the 53 no sequence repairs and 14 / 18 / 18
of the 39 solvable ones (the supervised model: 52 / 65 / 66 of 101, 37 / 47 / 48 and 15 / 18 / 18 of 48).  The
endings: 64 `finish resolved`, the estimator's test passing at 46 of them and failing at 18, and 31 `finish
unresolved`, 15 of them after damage (the supervised model: 72 with 52 and 20, 27 with 12 after damage, 3 out of
actions).  Of the 53 cases no sequence repairs the 17-step model ends 44 with `finish resolved` (the test passes
at 35) and 9 with `finish unresolved` (5 after damage).

**The run.**  1,594 forward passes for the 19,609 candidates: 11 to 29 a state, 17 at the median, 7 to 12 of them
for the calls that are always scored.  47 to 225 s a state (94 at the median, 5.4 s a pass), 1 h 24 min for the
arm, on the node the arm `base` ran on.  The prompts are 7,903 to 14,662 tokens.  The scored calls of a state hold
0.997 of the probability at the median and 0.77 at least (under 0.9 at 5 states).

**Checked.**  No earlier scoring of this model exists to compare with, as there was for the supervised one.
Checked instead: the first six states were read before the rest; at all 95 states the prompt is four tokens
longer than the local dry run rendered (the same four as on the arm `base`), the candidates are the same and no
call's tokens differ; the truth in each state file equals the input's; and the model's own last action is its
most probable scored call at 94 of the 95 stops (the arm `base`: 100 of 102, the other two being the two `finish`
calls equal to four decimals), which is what the adapter that wrote the episodes must show.  One state has its
first two objects exactly equal and neither is one still wrong, so no coverage count is undecided; 4 stops have
two tools equal.  An object still wrong stands within 0.25 of its set's edge in 13 of the 276 verdicts (4 at one
object, 7 at three, 2 at five), and at 71 of the 92 stops the fifth object is within 0.25 of the best one outside
the set.

**Not covered.**  The 17-step model is not scored at the supervised model's 102 stops (that would compare the two
on every state, and is another job); the stops of solved episodes are not scored for either model; one RL run,
one evaluation each.

**What it says for the RL work** (a reading, not a decision).  The decision the read-out isolates is whether to
look at the breakers again before stopping.  After 17 steps that call is live where a breaker is still wrong
(over 0.01 at 10 of the 16 stops, over 0.1 at 4, and one tie), the view it would bring holds the object (the
probe, 16 of 16 at three objects), and most of the cases the 17 steps gained are ones in which the model took
that look (14 of 20, against 1 under the supervised model).  Naming the right object from what the prompt already
shows did not move.

### Added readings

`readings/` holds the three scripts the tables above come from that are not part of `candidate_sets read`.  They
are reading scripts without tests, each with its command in its first lines.

| Script | What it reads |
| --- | --- |
| `readings/compare_arms.py` | Two arms side by side, and call by call at the stopping states they share.  Checked by giving it the arm `base` twice (every difference zero); its first block reproduces what `read` prints. |
| `readings/probe_by_kind.py` | The model's set against the evidence probe in both of its readings, by kind of error left, and the probability of `inspect breakers` at the stops.  On the arm `base` it reproduces the table above. |
| `readings/episode_looks.py` | Looks at the breakers and corrections per episode under two evaluations.  The episode records are copied to `output/minimal_candidates_20261125/episodes_base/` and `episodes_steps17/` (local). |

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

**The arm `steps17` is done**: submitted on the user's word of 2026-11-26 ("Yes, run it") at 23:19 UTC as job
24782581 (`mcs-steps17`), from the same source (d748912) and the inputs checked below (95 states, the same
checksum here and on torch); it ended COMPLETED on 2026-11-27 at 00:47 UTC with all 95 states scored, and its
result is above.  Nothing of this cell is queued or running, and no further job is planned: another scoring job
needs the user's word.

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
from the limit of 96.  The arm `steps17`, with 95 states against 102, took 1 h 24 min: the same 5.4 s a pass, 17
passes a state at the median and 29 at most.  The evidence probe on an arm's stops is 17 minutes on this machine.
