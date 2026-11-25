# The RL plan of 2026-11-22: stage A, stage B and the fork probe

**Status: REDEPLOYED 2026-11-22 at 98e6518; stage A's run and both probes were submitted and are COMPLETE** (section 9:
`mrp-steps` 24090332 to 24090338, `mrp-probe-dev` 24090400 to 24090402, `mrp-probe-train` 24090446 to 24090450).
**Both probes are done: one forced step repairs 39 of the 49 unrepaired development cases and 74 of 97
unrepaired training cases (section 9).** **Stage A is COMPLETE (2026-11-25 at 13:04 UTC; nine
iterations, 65 optimizer steps) and has three evaluations (section 9): 205 of the 247 solvable development cases
after 17 steps (20 gained, 13 lost against the supervised model's 198), 198 after 41 steps (19 gained, 19 lost)
and 187 after 65 steps (15 gained, 26 lost; 179 strict).** No evaluation differs from the start by more than the
suite can tell (p = 0.30, 1.0, 0.12). The gain is the two-breaker classes (15, then 26, 22 and 19 of 35); the
loss is cases the supervised model solved in few actions, by added line fits and meter corrections, and it grows
with the steps (tool errors 88, 48, 116, 147; episodes with a damaged healthy object 27, 29, 34, 46). The
planned check did not fire and nothing was cancelled; no `mrp-` job is queued, and every iteration's adapter is
kept (iteration 2's is the 205).
Both smokes completed and passed their checks on d6b7cfc. The probe's smoke showed that stage B would have
picked contradictory rows; how a fork row is picked was changed the same day (section 10) and is in the deployed
source. The user's word so far: "Yes, deploy and smoke", then "Yes, all three" (the redeploy, stage A's run and
both probes), then "Yes, set 0.1" on 2026-11-23: `forks.min_gain` is 0.1 in stage B's two configs, in the
repository only; the cluster's source still holds 0. On 2026-11-24, with the second evaluation read, the user
said "Let it run": stage A goes on to its third evaluation unchanged. Stage B (`forks-smoke`, `forks`), any
deploy, any cancel and any resubmission each still need the user's word.

The plan and its reasons are in `docs/minimal_rl_plan_20261122.md`. The user approved building both stages on
2026-11-22 ("Yes, build both"), with no cluster action.

## 1. What the cell holds

| | Stage A: `rl_steps.json` | Stage B: `rl_forks.json` | The finished focused run, for comparison |
| --- | --- | --- | --- |
| Optimizer steps a batch | 8, dealt by case (4 cases a step); 1 in the first update | the same | 1 |
| Learning rate | 3e-5 | 3e-5 | 1e-5 |
| Evaluation on the 300 development cases | every third iteration | every third iteration | every iteration |
| Truth-built examples for cases without a repaired sample | on (weight 1) | off | on (weight 1) |
| One-step forks of the greedy episode | off | on | off |
| Smallest margin of a fork row (`forks.min_gain`) | | 0.1 since 2026-11-23; 0 when the probes ran | |
| Seed of the draw | 20261122 | 20261122 | 20261118 |

Everything else is the focused run's (`research/hpc/minimal_rl_20261116/rl_focus.json`): the supervised model as
the start and the KL reference (0.04), 24 unrepaired and 8 repaired training cases an iteration, 8 samples a case
at temperature 1, the reward, whole cases per step, the carried optimizer, no clip. `test_cell.py` asserts that
these are the only differences.

The third thing is the **fork probe** (`stage_fork_probe.sbatch`, `research/minimal_env/fork_probe.py`): the forks
alone, with no update, on cases the supervised model leaves unrepaired. It counts the cases one forced step
repairs.

## 2. What was built

**Stage A needed one setting.** `evaluate_every` in `research/minimal_env/rl_train.py`: the adapter of every N-th
iteration is evaluated, the others record that they were skipped and load no model.

**Stage B is new code.**

- `psse_env/dagger/research_policy_factory.py`: `generate_text(observation, forced_prefix=...)`. The prefix's
  tokens are fed after the prompt and the model writes what follows. The generation record carries
  `forced_prefix_tokens` and no token probabilities, so it cannot be taken for a sampled completion.
- `psse_env/minimal/policy.py`: `act_forced(observation, tool)`. The harness gives the tool's name
  (`<|tool_call>call:NAME{`); the model writes the arguments.
- `psse_env/minimal/forks.py`: the model's own greedy episode, then at every state every other tool forced once
  and both outcomes of `finish` given as whole calls (section 10), the model continuing greedily, each branch
  scored with the episode reward. A replay is checked against a digest of what the model read; a branch that
  drifted, a forced call that is not a valid call of its tool, and a call the environment refused are marked and
  never taught.
- `research/minimal_env/rl_train.py`, `forks`: the forks are made in the rollout phase, by the same processes that
  sample (one model load, one file per case, sharded and restartable like the samples). At a state whose best
  branch returned more than going on as the greedy episode did can still reach (the greedy return, or a branch
  kept at a later state), the update gets one supervised row: the branch's call on that state's prompt, weighted
  by that excess (section 10; before the probe's smoke the comparison was with the greedy return alone).
- `research/minimal_env/fork_probe.py`: the forks with no update, and a summary.

**What a fork is not.** The forks are at every state and over every tool; nothing names the states or the tools
that matter, and the truth enters only through the reward. A forced step is the harness's, so nothing after it is
trained. The branching happens in training only: an evaluated episode is one greedy pass.

**Where the build differs from the plan document.**

- The plan named a separate `fork` phase. The forks are made inside the rollout phase instead, so the batch stage
  and the manifest are unchanged and the model is loaded once.
- The plan counted 10 iterations of stage A (73 steps). With an evaluation every third iteration the tenth would
  not be evaluated, so the count here is 9 iterations: 65 optimizer steps (1, then 8 in each of eight updates)
  and evaluations after the third, sixth and ninth.
- A forced call that does not render as the supervised pass taught it gives no row and is counted. It is not a
  stop condition: the model wrote the call itself, and nothing wrong is trained by leaving the row out. The status
  script prints the count; it should be zero.
- `forks.argument_samples` (0 in both configs) adds branches whose forced call is sampled, which tries other
  objects of one tool. It is there for the failure pattern that is a choice of object (a healthy line fitted).
  Only the probe's smoke uses it.

## 3. The order of work

Each step on the user's word.

1. Deploy: `bash research/hpc/minimal_rl_20261122/deploy_from_windows.sh <commit40>` (submits nothing).
2. Stage A's smoke and the probe's smoke: `submit_cell.sh steps-smoke`, `submit_cell.sh probe-smoke`.
3. Read both with `status_cell.sh` (section 4).
4. Stage A: `submit_cell.sh steps 9`. The probe: `submit_cell.sh probe dev` and `submit_cell.sh probe train`.
   They have their own folders and may run at the same time.
5. Read the probe. Then stage B's smoke and run: `submit_cell.sh forks-smoke`, `submit_cell.sh forks 9`.

Stage B starts from the supervised model, like stage A, so the two are compared from one start.

## 4. What each smoke is read for

A smoke is read for what only the model can show, not for a score.

**`steps-smoke`** (two iterations on 6 unrepaired and 2 repaired cases by 2 samples, 8 development cases):

- The first update takes one step (its optimizer is new) and the second takes four with the optimizer carried.
- How far the later steps of one batch are from the policy that sampled it. The status script prints the mean
  absolute log-ratio to the sampler for every step; the first is the numerical gap between the decoding and the
  training pass (0.002 in the focused run), and what the later ones add is the drift of the earlier steps.
- The first iteration's evaluation is skipped without loading the model, and the second is made.

**`probe-smoke`** (4 unrepaired development cases, one sampled call a tool):

- **Forced prefixes that start the call's own rendering.** The tool's name is given to the model as text. If its
  tokens are not the ones the model was trained to write the call with, the arguments are written after a prefix
  the model never saw. The count should equal the number checked.
- **Forced calls that are not valid calls** (`invalid_call`), which should be few, and **drifted branches**, which
  should be none: the environment is deterministic, and a drift means a replay did not reproduce what the model
  read.
- Seconds a case, which replaces the arithmetic of section 5.
- How often a sampled call differs from the one the model's own decoding wrote (`duplicate` counts the repeats).

**`forks-smoke`** (one iteration on 3 unrepaired and 1 repaired cases by 2 samples, 8 development cases):

- The rollout phase samples and forks every drawn case, in shards and in the closing call.
- The fork rows render (`not rendered 0`), the update trains them (a supervised loss above zero), and the
  integrity checks pass.

## 5. Cost

All of this is my arithmetic from the focused run's measured rates; none of these settings has run.

| | GPU-hours |
| --- | --- |
| Stage A, one iteration | 4.6 to 6.1: rollout 1.2, update 3 to 4.5, a third of an evaluation of 1.3 |
| Stage A, nine iterations | 41 to 55 |
| Forks, one iteration of 32 cases | about 3.4: 4.6 states by 5 tools by about 3.5 generated steps at 4.7 s |
| Stage B, nine iterations | 72 to 86 |
| Probe, the 49 development cases | about 5.5 |
| Probe, 100 training cases | about 11 |
| Each smoke | about 1 |

**Measured in the two smokes (2026-11-22, section 9), on few cases.** An update row takes 12.2 to 12.8 s (0.21 h
for 62 rows in one step, 0.37 h for 104 rows in four), inside the focused run's 8.6 to 13.7 s: a row is used once,
so more steps a batch cost nothing. The forks take about 95 to 100 s a state without sampled calls (16 states of
4 cases), which is about 450 s for a case of 4.6 states against the 382 to 404 s the table assumes. With that
rate the probe is about 6 GPU-hours on the 49 development cases and about 12.5 on 100 training cases, the forks
of one iteration about 4, and stage B's nine iterations 77 to 91. Branches whose forced call the environment
refused took 26% of the fork time (22 of the 80 calls in the model's own decoding). They are never taught; ending
such a branch at the refusal would save that time. It is not built, because the probe reads what follows a
refusal (section 9).

One case can cost far more than the average: a greedy episode that runs out of its 16 actions is forked at 16
states, and each branch can run to the budget, about 50 minutes for that case (3 of the supervised model's 300
development episodes ended that way). There is no cap on the states forked.

`submit_cell.sh` counts 7 hours an iteration for stage A and 11 for stage B when it sets the links of a chain
(11 hours of each 12-hour link, plus one spare): 7 links for `steps 9`, 10 for `forks 9`. A link leaves at once
when the target is reached.

## 6. How a result is read

- Solved of the 247 development cases a truth-built sequence repairs, with the cases gained and lost against the
  supervised model's 198. The focused run ended at 202 (201 strict). With 4 cases lost, 13 gained give p = 0.049;
  with 6 lost, 17 gained give p = 0.035: a result this suite can tell from the supervised model is about 207 to
  209 or more.
- Mean actions an episode (4.57 after the focused run).
- Stage A: if the first evaluation (after 17 steps) loses more cases than it gains, the learning rate is halved
  before the run goes on. (It gained 20 and lost 13: the rule did not fire. The second evaluation gained 19 and
  lost 19, a case this rule does not cover; section 9.)
- Stage B, in every iteration: how many unrepaired cases one forced step repairs, how many fork rows the update
  took and by which tool, and how many of them are within one action's cost (a saved action, not a repair).
- The probe: of the unrepaired cases, how many one forced step repairs, by class and by the forced tool. This is
  a labelled upper bound on one round of stage B, not a result.

## 7. Not checked

- The forced call has met the model on 4 cases (section 9): its tokens start the call's own rendering in 176 of
  176 branches, no forced call was invalid and no replay drifted. Stage B's update on fork rows (`forks-smoke`)
  has not run.
- The selection of section 10 and the given outcomes of `finish` have run with stand-ins and on the smoke's
  stored records only until the two probes, which went out at 98e6518 without a second smoke. Both probes read
  as designed (section 9): every branch ran and every forced prefix was canonical. The selection with
  `min_gain` 0.1 was read on the probes' stored records; it has not run inside an update.
- Whether eight steps on one batch at 3e-5 stay close to the sampled policy, and whether 65 steps are safe or
  enough. The smoke took four steps on one small batch: by the fourth, the uncertain tokens were a mean factor of
  1.35 from the policy that sampled the batch (section 9). No earlier RL run of this project took more than a
  few steps. Read on 2026-11-24 (section 9): 17 steps moved 33 development cases for a net gain of 7, and 41
  steps moved 38 for a net of 0, with more tool errors and more episodes that damage a healthy object than at the
  start. Neither kept the cases already solved (13 and 19 of the 198 lost; the focused run lost 6). The third
  evaluation (65 steps) is not in.
- Every cost above; the measured rates are from 4 cases and two small updates.
- Whether supervised rows on fork states, with the KL term, keep the cases already solved.
- The forks read the greedy episode. A case whose repair needs two unlikely steps in a row is not reached by one
  forced step (the probe's tests show such a case): 10 of the 49 unrepaired development cases are out of one
  step's reach (section 9).

## 8. Local checks

- `psse_env/minimal/test_forks.py` (12 tests), `research/minimal_env/test_fork_probe.py` (4),
  `research/minimal_env/test_rl_train.py` (32, seven of them new), `psse_env/dagger/test_research_policy_factory.py`
  (15, one new), `research/hpc/minimal_rl_20261122/test_cell.py` (18): all pass.
- The environment and the scoring in those tests are the real ones, on freshly generated cases. The model is a
  stand-in that decides from the observation alone.

## 9. Cluster record

**Deployed 2026-11-22, about 08:35 UTC**, with `deploy_from_windows.sh d6b7cfc5fe4cbe586c5b41d272bcfd61f3aabff5`.
The cell's source was cloned from the earlier RL cell's (at 2d4919d) and moved to d6b7cfc by a bundle of 210,035
bytes. Read on the cluster afterwards: the source at d6b7cfc; the five changed Python files parse under the job
environment's Python 3.12.13; the five scripts pass `bash -n` and hold no carriage return; the supervised adapter,
its development evaluation, its training pass, the reference and the file of truth-built sequences are present;
both suites have the planned checksums; no job of this user was queued.

**Submitted 2026-11-22 on the user's word ("Yes, deploy and smoke"):**

| Job | Id | Submitted (UTC) | What it runs |
| --- | --- | --- | --- |
| `mrp-steps-smoke` | 24068489 | 08:39:40 | two iterations of `rl_steps_smoke.json`, 4-hour limit |
| `mrp-probe-smoke` | 24068494 | 08:39:55 | the probe on 4 unrepaired development cases with one sampled call a tool, 4-hour limit |

**Both jobs completed** with no preemption and no traceback: `mrp-probe-smoke` in 40 min 16 s on gr102 (peak
memory 33 GB, mean utilization 77%), `mrp-steps-smoke` in 1 h 06 min 13 s on gr105 (peak memory 66 GB, 78%).

**Stage A's smoke, read against section 4.**

| What | Read |
| --- | --- |
| First update | 1 step on 62 rows, the optimizer new, 0.21 h. 29 rows of 6 truth-built examples and 33 rows for the KL term; no policy row, because no case had samples of different reward |
| Second update | 4 steps on 104 rows (18, 27, 32, 27), the optimizer carried from iteration 0 (1 step before, 5 after), 0.37 h |
| Mean absolute log-ratio to the sampler, by step | all tokens 0.0014, 0.0076, 0.029, 0.026; uncertain tokens 0.011, 0.14, 0.18, 0.30 |
| KL to the supervised model, by step | 0.0004, 0.012, 0.009, 0.021 |
| Evaluation | iteration 0 recorded as skipped ("the next evaluation is of iteration 1"); iteration 1 evaluated: 4 of 8 before and after, gained 0, lost 0, mean actions 6.12 to 5.25 |
| Integrity | ok in both iterations; rows that did not render 0 |

The mechanics work. The drift within a batch is real: the first step's 0.011 is the numerical gap between the
decoding and the training pass, and by the fourth step the uncertain tokens are a mean factor of e^0.30 = 1.35
from the policy that sampled the batch, with nothing correcting for it (the clip is off). The run takes eight
steps a batch, so its last steps will be further off. The KL to the supervised model after five steps (0.021) is
6 to 9 times what the focused run reached in all (0.0033 and 0.0024); the KL term's coefficient is 0.04. The
first evaluation of the run (after 17 steps) is the check, and the status script prints these lines for every
update. Eight development cases say nothing about the score.

**The probe's smoke, read against section 4.**

| What | Read |
| --- | --- |
| Forced prefixes that start the call's own rendering | 176 of 176 |
| Branches | 176 at 16 states of 4 cases: 108 ran, 68 repeated a call already tried (all among the 96 sampled calls); invalid 0, drifted 0, no call 0 |
| Forced calls the environment refused | 29 of the 108 (22 of the 80 in the model's own decoding): a breaker named before the breakers were looked at, an undo with nothing to undo, a line fit that fails |
| Unrepaired cases one forced step repairs | 3 of 4, all three with the model's own decoding of the arguments. The sampled calls added one repairing branch and no case |
| Time | 577 s a case with the sampled calls, 0.64 h in all; about 95 to 100 s a state without them |

The four cases (the supervised model's greedy episode, then what one forced step did):

| Case | Greedy episode | One forced step |
| --- | --- | --- |
| `r0_243985153e00`, cb2_split | looks at the breakers, opens one, a line fit fails, declares the case unresolved (-0.06) | a second look at the breakers at the third or fourth state: the model opens the second breaker and the case is repaired (0.95, 0.92) |
| `r0_9c85ff49a8e0`, rx_cb_distant | opens one breaker and declares the case resolved, which is false (-0.53) | `correct_parameters` at the last state: the model writes line 16 and the case is repaired (0.96) |
| `r0_a04d7fb1791e`, rx2_adjacent | a fit of line 1 fails, corrects nine meters (a healthy object damaged), unresolved (-0.56) | a look at the estimate at the second state, or `correct_parameters` at the third (the model writes line 3): the model fits line 3, then line 1 (0.95, 0.93) |
| `r0_d17f8671b359`, rx2_distant | fits line 1, a fit of line 17 fails twice, unresolved (-0.09) | nothing repairs; the fit of line 17 fails in every branch |

**Two cautions on "3 of 4".** Four cases are not a rate. And part of what a forced step does is disturb the
episode. In the cb2_split case a forced correction of a wrong breaker or of a meter also ends repaired (the model
undoes it, looks at the breakers again and finds the right one), and so do four branches whose forced call the
environment refused, which changed nothing but the history the model reads. In that case the model's choice
between a second look and giving up is close. The summary now counts the unrepaired cases repaired after a
refused call (1 of 4 here): the floor to read the one-forced-step count against.

**Redeployed 2026-11-22, about 15:31 UTC, on the user's word ("Yes, all three")**, with
`deploy_from_windows.sh 98e651886657c5a1286f475137bcc6aa3684b26d`: a bundle of 20,709 bytes moved the source
from d6b7cfc to 98e6518. Read on the cluster afterwards: the source at 98e6518 with no local change; `forks.py`,
`rl_train.py` and `fork_probe.py` parse under the job environment's Python; the status script and both stages
pass `bash -n` and hold no carriage return; the inputs are present and both suites have the planned checksums;
no job of this cell was queued before.

**Submitted 2026-11-22 on the same word, source 98e6518:**

| Job | Ids | Submitted (UTC) | What it runs |
| --- | --- | --- | --- |
| `mrp-steps` | 24090332 to 24090338, a chain of 7 links, each after the one before | 15:33:49 | stage A (`rl_steps.json`) from iteration 0 to 9; a link leaves at once when the target is reached |
| `mrp-probe-dev` | parts 24090400 and 24090401, close 24090402 | 15:34:11 | the probe on the 49 unrepaired development cases a truth-built sequence repairs, in 2 parts, then the summary |
| `mrp-probe-train` | parts 24090446 to 24090449, close 24090450 | 15:34:34 | the probe on 100 of the 502 unrepaired training cases, in 4 parts, then the summary |

Every job asks for one card of the cell's pool (a100, h100, h200 or rtx6000, no partition named), 16 CPUs, 96 GB
and 12 hours, with preemption allowed. All 15 were pending when read at 15:35 UTC. Outputs: `out/steps`,
`out/probe_dev`, `out/probe_train`.

**What to read first.**

- The probes: the forced prefixes canonical and no invalid or drifted call, as in the smoke; then the `finish`
  branches the harness gives (`arguments` "given"), which meet the model here for the first time; then how many
  of the 49 and of the 100 one forced step repairs, beside the count repaired after a refused call.
- Stage A: the lines the status script prints for every update (the log-ratio to the sampler by step, the KL to
  the supervised model), and the first evaluation, after iteration 2 (17 steps). If it loses more than it gains,
  the plan is to cancel the chain and halve the learning rate; the cancel needs the user's word.

**The first case the development probe wrote, read at 15:48 UTC** (`r0_891df54d8682`, cb2_dangling; the greedy
episode claims the case resolved after four actions, return -0.79). Contract `minimal_one_step_forks_v2`; 24
branches, all ran; forced prefixes canonical 17 of 17; 446 s for 4 states. The 7 `finish` branches the harness
gave (3 resolved, 4 unresolved; the greedy claim left out) were carried out and ended the episode there. Rows
kept: a look at the breakers at the third state, which repairs the case (margin 1.24), and "declare the case
unresolved" at the fourth, where the greedy episode made the false claim (0.50). The two margins add up to the
best return less the greedy return (0.95 + 0.79). By then stage A was in iteration 0's rollout and three of the
training probe's four parts were running; no traceback in any log.

**The development probe, completed 2026-11-22 at 19:02 UTC** (parts 24090400, 3 h 22 min on gr101, and
24090401, 2 h 22 min on gr103 after one preemption; close 24090402, 28 s; no traceback; about 6.7 card-hours in
all against an estimate of 6). A labelled upper bound: a development fork is never a training row.

| What | Read |
| --- | --- |
| Cases | 49, the greedy episode unrepaired in all 49: 28 end in a false claim, 19 declare the case unresolved, 2 run out of actions |
| States and branches | 263 states, 1,578 branches, all ran: 1,099 with a call the model wrote and 479 `finish` calls the harness gave; invalid 0, drifted 0 |
| Forced prefixes that start the call's own rendering | 1,099 of 1,099 |
| Forced calls the environment refused | 368 of the 1,099 |
| **Cases one forced step repairs** | **39 of 49**, every one by a call the environment carried out |
| Cases repaired after a refused call too | 14 of the 49, all among the 39 |
| Branches that end repaired | after a carried-out forced call 127 of 731 (17.4%); after a refused one 35 of 368 (9.5%) |
| Time | 778 s a case with two workers a card, 10.59 h of worker time |

By class, cases one forced step repairs: cb2_dangling 6 of 8, cb2_mixed 6 of 7, cb2_split 4 of 5, control_meter
1 of 1, meter_on_rx 3 of 3, meter_rx2 4 of 5, rx2_adjacent 5 of 6, rx2_distant 1 of 3, rx_cb_distant 4 of 4,
rx_cb_meter 4 of 4, rx_cb_shared_bus 1 of 3.

What the repairing step is (each case's kept repairing row with the highest return):

| Where | Cases | The greedy action there | The forced step |
| --- | --- | --- | --- |
| Two breaker errors (cb2_*), after the first breaker is corrected: the third state in 15, the second in 1 | 16 of 20 | fits a line (15), names a breaker (1) | a second look at the breakers (13), or the second breaker named at once (3) |
| The first state | 7 (meter_on_rx 3, meter_rx2 2, rx2_distant 1, rx_cb_meter 1) | looks at the breakers | `correct_parameters`; the model names the line itself |
| The last state, where the greedy episode claims the case resolved | 5 (rx_cb_distant 4, meter_rx2 1) | `finish`, resolved | `correct_parameters` (4), a look at the breakers (1) |
| Elsewhere | 11 (rx2_adjacent 5, rx_cb_meter 3, control_meter 1, meter_rx2 1, rx_cb_shared_bus 1) | a line fit (8), a meter correction (2), an undo (1) | undo (4), a look (2), a correction of another kind (5) |

By tool: a look 16, `correct_parameters` 12, `correct_topology` 5, undo 4, `correct_measurements` 2. In 34 of
the 39 the step is before the last state.

**Three cautions.** (1) The forced step is found by trying every tool and scoring the outcome with the truth;
that is what stage B's forks do on training cases, and on development cases it is a measurement only. 198 + 39
= 237 of 247 is what one round would reach if every such step were learned and nothing were lost; it is not a
forecast. (2) Part of the effect is disturbance: a refused call changes only the history the model reads, and
9.5% of such branches end repaired against 17.4% after a carried-out call. Of the 39 cases, 12 have one
(state, tool) pair that repairs and 27 have two or more (up to 9): in many of these states the model's choice
is close. (3) 10 cases are out of one step's reach (cb2_dangling 2, cb2_mixed 1, cb2_split 1, meter_rx2 1,
rx2_adjacent 1, rx2_distant 2, rx_cb_shared_bus 2).

The rows stage B would keep on these records with `min_gain` 0 (for reading only):

| Kind of row | Rows | Weight |
| --- | --- | --- |
| The branch repairs the case | 69 | 46.86 (76%) |
| Declare the case unresolved, margin about 0.5: the false-claim penalty that going on runs into | 20 | 10.02 (16%) |
| Declare the case unresolved, margin about 0.25: one healthy object that going on damages | 7 | 1.82 (3%) |
| Declare the case unresolved, margin at most 0.03: actions saved, nothing else | 77 | 1.59 (3%) |
| Declare the case unresolved, other | 1 | 0.06 |
| Another tool, no repair | 5 | 1.44 (2%) |
| All | 179 | 61.79 |

Of the 77 rows that only save actions, 54 are in the 10 cases no forced step repairs and 10 are at the first
state ("declare the case unresolved before looking at anything"). They are 43% of the rows and 3% of the weight;
an update row costs 12 to 13 s whatever its weight.

What `forks.min_gain` keeps on the same records (a state it skips leaves its small margin to the state before,
so the small rows thin out rather than vanish, and the total weight hardly moves):

| `min_gain` | Rows | Repairing rows | `finish` rows above 0.2 | `finish` rows at most 0.2 | `finish` rows at the first state | Weight |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (as in `rl_forks.json` when the probes ran) | 179 | 69 | 27 | 78 | 10 | 61.79 |
| 0.035 | 101 | 45 | 27 | 26 | 1 | 61.30 |
| 0.1 (set on 2026-11-23) | 77 | 39, one a repaired case | 27 | 8 | 0 | 60.68 |
| 0.2 | 72 | 39 | 31 | 0 | 0 | 60.77 |

With 0.1, the rows are 43% of the count at 98% of the weight, every repaired case keeps one repairing row and
no row says to stop before looking. **The user set it on 2026-11-23 ("Yes, set 0.1"): `rl_forks.json` and
`rl_forks_smoke.json` hold 0.1, in the repository only.** `rl_forks.json` is read from the cell's source, and
the deploy script does not move the source while a job of this cell is queued: the change reaches the cluster
with the next deploy, after stage A's chain or through a second source folder, and a deploy needs the user's
word.

**The training probe, completed 2026-11-22 at 19:30 UTC** (parts 24090446 to 24090449, close 24090450; one
preemption, no traceback; 11.6 card-hours against an estimate of 12.5). A seeded sample of 100 of the 502
training cases the supervised model's training pass left unrepaired. These are cases stage B can draw.

| What | Read |
| --- | --- |
| Cases | 100; the greedy episode left 97 unrepaired here (3 came out repaired on this card, against the training pass): 49 end in a false claim, 47 declare the case unresolved, 1 runs out of actions |
| States and branches | 530 states, 3,180 branches, all ran; invalid 0, drifted 0 |
| Forced prefixes that start the call's own rendering | 2,219 of 2,219 |
| Forced calls the environment refused | 724 of the 2,219 |
| **Cases one forced step repairs** | **74 of the 97**, every one by a call the environment carried out |
| Cases repaired after a refused call too | 16, all among the 74 |
| Branches that end repaired (the 97 cases) | after a carried-out forced call 298 of 1,444 (20.6%); after a refused one 42 of 700 (6.0%) |
| Time | 770 s a case with two workers a card, 21.39 h of worker time |

By class, cases one forced step repairs: cb2_dangling 11 of 12, cb2_mixed 7 of 7, cb2_split 8 of 9,
control_meter 1 of 1, control_rx 2 of 2, meter_on_rx 3 of 3, meter_rx2 13 of 17, rx2_adjacent 2 of 4,
rx2_distant 3 of 5, rx_cb_distant 9 of 14, rx_cb_meter 11 of 18, rx_cb_shared_bus 4 of 5. The 23 out of one
step's reach: rx_cb_meter 7, rx_cb_distant 5, meter_rx2 4, rx2_adjacent 2, rx2_distant 2, and one each of
cb2_dangling, cb2_split and rx_cb_shared_bus.

What the repairing step is, grouped the same way for both probes (each case's kept repairing row with the
highest return):

| Where | Training, of 74 | Development, of 39 | The greedy action there | The forced step (training) |
| --- | --- | --- | --- | --- |
| A line fit where a breaker step repairs, at the second or third state | 42 | 18 | `correct_parameters` | the breaker named at once (30), a look at the breakers (12) |
| The first state | 11 | 7 | a look at the breakers | `correct_parameters` (10), `correct_measurements` (1); the model names the object itself |
| The state where the greedy episode stops | 10 | 5 | `finish` | `correct_parameters` (7), a look at the breakers, a breaker, an undo (1 each) |
| Elsewhere | 11 | 9 | a line fit (7), a meter correction (3), a breaker (1) | undo (7), `correct_parameters` (2), a breaker (1), and one `finish` with the claim that is true there |

The last entry is a control_rx case that is repaired after two actions; the greedy episode goes on and fits a
line that damages it. The claim at that state is one of the `finish` calls the harness gives.

The two probes agree: one forced step repairs 76% of the unrepaired training cases and 80% of the unrepaired
development cases, and more than half of those steps are the same decision, a breaker step where the model
fits a line. At that rate about 380 of the 502 unrepaired training cases hold such a step; stage B draws 24
unrepaired cases an iteration. The cautions of the development probe hold here too: after a refused call 6.0%
of the branches end repaired, and of the 74 cases 9 have one (state, tool) pair that repairs and 65 have two or
more (up to 13).

The rows stage B would keep on the 97 cases with `min_gain` 0:

| Kind of row | Rows | Weight |
| --- | --- | --- |
| The branch repairs the case | 165 | 90.61 (80%) |
| Declare the case unresolved, margin about 0.5 | 29 | 14.51 (13%) |
| Declare the case unresolved, margin about 0.25 | 17 | 4.42 (4%) |
| Declare the case unresolved, margin at most 0.03 | 117 | 1.81 (2%) |
| Declare the case unresolved, other | 2 | 0.15 |
| Another tool, no repair | 5 | 1.63 (1%) |
| All | 335 | 113.13 |

The 3 cases the greedy episode repaired add 4 rows of weight 0.10 in all (a shorter path). Of the 117 rows
that only save actions, 85 are in the 23 cases no forced step repairs and 23 are at the first state.

| `min_gain` (all 100 cases) | Rows | Repairing rows | `finish` rows above 0.2 | `finish` rows at most 0.2 | `finish` rows at the first state | Weight |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (as in `rl_forks.json` when the probes ran) | 339 | 169 | 46 | 119 | 23 | 113.23 |
| 0.035 | 168 | 98 | 46 | 20 | 1 | 111.66 |
| 0.1 (set on 2026-11-23) | 130 | 76 | 46 | 4 | 1 | 110.09 |
| 0.2 | 126 | 74 | 48 | 0 | 0 | 109.78 |

With 0.1 the rows are 38% of the count at 97% of the weight. What it buys is mostly what is taught, not time:
at 24 unrepaired cases an iteration the difference is about 80 rows against about 30, some 10 minutes of an
update. The one row that remains at the first state is a thinned one: a margin of 0.10 at the second state is
not above the threshold and passes to the first state, which is kept at 0.11 (a case no forced step repairs).

**Stage A's first two updates, read 2026-11-23 at 02:24 UTC** (the first link, 24090332; one preemption at
about 22:41 UTC, as the second update began, moved it from gr103 to gh101; no traceback; integrity ok in both
iterations; peak memory 66.9 GB, mean utilization 96%). Iteration 2's rollout began at 02:17 UTC.

| What | Iteration 0 (the supervised model samples) | Iteration 1 (one step later) |
| --- | --- | --- |
| Rollout, 256 episodes | repaired 72, false claims 95, declared unresolved 74, out of actions 15 | repaired 95, false claims 108, declared unresolved 52, out of actions 1 |
| The 24 unrepaired cases drawn of 502 | 5 with a repaired sample, 19 with none; 10 of 192 episodes repaired | 13 with a repaired sample (one with all eight), 11 with none; 42 of 192 episodes repaired |
| The 8 repaired cases drawn of 1,591 | 62 of 64 episodes repaired; 2 cases with an unrepaired sample | 53 of 64; 4 cases with an unrepaired sample |
| Truth-built examples kept | 19, 87 rows | 11, 51 rows |
| Rows | 1,077: 932 policy, 58 for the KL term alone, 87 supervised | 1,223: 1,159 policy, 13 for the KL term alone, 51 supervised |
| Update | 1 step, the optimizer new; 4.03 h | 8 steps on 202, 146, 156, 140, 164, 119, 147 and 149 rows, the optimizer carried (1 step before, 9 after); 3.50 h |
| Gradient norm before the clip at 1.0 | 4.05 | 1.52, 2.28, 1.87, 4.19, 2.44, 4.21, 1.87, 3.80 |

The two rollouts are not a comparison: each iteration draws other cases. For reference, the focused run's three
rollouts, from nearly the same model, repaired 28, 31 and 25 of 192 episodes on the unrepaired cases drawn and
64, 60 and 61 of 64 on the repaired ones. Iteration 1's 42 of 192 is above all of these and its 53 of 64 below
all of them; with 24 and 8 cases a draw, neither is a result.

The eight steps of iteration 1's update:

| Step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mean absolute log-ratio to the sampler, all tokens | 0.0037 | 0.0092 | 0.0091 | 0.0076 | 0.019 | 0.028 | 0.031 | 0.023 |
| The same on uncertain tokens | 0.039 | 0.124 | 0.141 | 0.139 | 0.208 | 0.273 | 0.298 | 0.271 |
| The largest on one token | 0.42 | 1.61 | 1.91 | 1.15 | 3.29 | 2.49 | 3.00 | 7.67 |
| Mean log-ratio to the sampler (signed) | -0.0007 | -0.0020 | -0.0019 | -0.0015 | -0.0091 | -0.0130 | -0.0170 | -0.0145 |
| KL to the supervised model | 0.0044 | 0.0150 | 0.0030 | 0.0075 | 0.0269 | 0.0380 | 0.0632 | 2.764 |
| Largest row gradient norm before the row bound of 1.0 | 1.26 | 0.80 | 22.6 | 0.77 | 1.64 | 5.32 | 2.02 | 508 |
| Rows the row bound cut | 2 | 0 | 3 | 0 | 1 | 1 | 1 | 7 |
| Median row gradient norm | 1.7e-4 | 8.1e-4 | 7.1e-5 | 2.3e-4 | 9.4e-4 | 1.5e-3 | 5.3e-3 | 6.0e-3 |

- **The mean drift is what the smoke showed.** On uncertain tokens it rises to 0.27 to 0.30 by the sixth step
  and stays there (the smoke reached 0.30 at its fourth step). The signed mean is negative and grows: the model
  moves away from what it sampled, as a batch with a mean advantage of -0.034 asks.
- **The model moves much further than in the focused run.** The KL to the supervised model is 0.063 at the
  seventh step: three times the smoke's 0.021 after five steps and about twenty times the focused run's whole
  movement (0.0033). The median row gradient norm grows 35 times over the eight steps.
- **The eighth step's KL of 2.76 is not a general drift.** That step's mean gap is lower than the seventh's
  (0.023 against 0.031). The estimate is exponential in the gap (e^d - d - 1 on each sampled token, d the
  supervised model's log-probability less the model's), and the step holds one token 7.67 nats from the sampler
  and one row whose gradient norm is 508 before the row bound, against 0.8 to 23 for the largest row of the
  other steps. One token about 9 nats below the supervised model, in a row of the mean length of 18 tokens,
  gives 2.76 as the mean of 149 rows. **This is read from the step's totals: the run keeps
  no record by row (`step_diagnostics` is off), so the row and the token are not known.**
- **What it means.** After seven steps the model gives one sampled token of a case this update had not trained
  on yet a probability some thousands of times lower than the model that sampled it: a decision has swung hard
  somewhere. That is the movement the focused run could not make, and it is the risk the halving rule is for.
  The KL term pulled back on that row with the row bound's strength and no more (508 cut to 1.0).

Time: 10.7 h for the two iterations (rollouts about 1.5 and 1.2 h, updates 4.03 and 3.50 h, at most 25
minutes lost to the preemption). At this pace nine iterations and three evaluations take about 52 h, inside the
planned 41 to 55. The first evaluation (iteration 2, after 17 steps) was expected between 08:15 and 09:15 UTC
on 2026-11-23 (it was written at 11:08 UTC, after five more preemptions: the next block); it is the planned
check, and nothing was changed or cancelled on this reading.

**Stage A's first two evaluations, read 2026-11-24 at 17:02 UTC.** The chain is alive: the second link of
seven (24090333) runs on gh102, 24090334 to 24090338 wait on the dependency. The run is in iteration 6's update
(checkpoint at step 3 of 8). Six iterations are done (41 optimizer steps), integrity ok in each, no traceback.
The waiter armed on 2026-11-23 died that day at about 07:35 UTC (exit code 4) and was not re-armed, so both
evaluations were first read here: the first 30 hours after it was written.

| Development suite, greedy | Supervised model (the start) | After 17 steps (iteration 2, written 2026-11-23 11:08 UTC) | After 41 steps (iteration 5, written 2026-11-24 11:56 UTC) |
| --- | --- | --- | --- |
| **Solved of the 247 a truth-built sequence repairs** | **198** | **205** (201 strict) | **198** (198 strict) |
| Gained / lost against the start | | 20 / 13 (exact two-sided p = 0.30) | 19 / 19 (p = 1.0) |
| Of the 184 the rule expert solves | 161 | 166 | 158 |
| Of the 63 only a truth-built sequence repairs | 37 | 39 | 40 |
| Against the rule expert's 184: gained / lost | 37 / 23 (p = 0.09) | 39 / 18 (p = 0.008) | 40 / 26 (p = 0.11) |
| False claims | 72 | 64 | 72 |
| Declared unresolved | 27 | 31 | 27 |
| Out of actions | 3 | 0 | 3 |
| Episodes with a damaged healthy object (objects) | 27 (51) | 29 (41) | 34 (46) |
| Tool errors (unknown object / tool failure) | 88 (30 / 58) | 48 (0 / 48) | 116 (50 / 66) |
| True errors repaired of 643 | 520 | 530 | 522 |
| Mean actions an episode | 4.45 | 4.45 | 4.69 |
| Calls: `inspect` / `correct_topology` / `correct_parameters` / `correct_measurements` / `undo` | 323 / 199 / 355 / 133 / 27 | 357 / 183 / 360 / 112 / 23 | 348 / 223 / 383 / 122 / 34 |

Between the two evaluations: 8 gained, 15 lost (p = 0.21). Against the start, 16 cases are gained in both
evaluations, 4 in the first only and 3 in the second only; 8 are lost in both, 5 in the first only and 11 in the
second only. 174 cases are solved by all three models and 221 by at least one. None of the 53 cases nothing
repairs is solved by any of them. The second evaluation's four outcome counts equal the start's (198, 72, 27,
3); that is a coincidence of totals: it ran on iteration 5's adapter and 38 cases changed.

Solved by class, of the cases a truth-built sequence repairs:

| Class | Solvable | Start | 17 steps | 41 steps |
| --- | --- | --- | --- | --- |
| cb2_dangling | 14 | 6 | 12 | 11 |
| cb2_mixed | 11 | 4 | 7 | 4 |
| cb2_split | 10 | 5 | 7 | 7 |
| control_cb | 12 | 12 | 12 | 10 |
| control_meter | 12 | 11 | 7 | 8 |
| control_rx | 11 | 11 | 11 | 10 |
| meter_on_rx | 16 | 13 | 13 | 12 |
| meter_rx2 | 20 | 15 | 16 | 17 |
| rx2_adjacent | 26 | 20 | 19 | 21 |
| rx2_distant | 24 | 21 | 22 | 22 |
| rx_cb_distant | 30 | 26 | 26 | 26 |
| rx_cb_meter | 29 | 25 | 24 | 23 |
| rx_cb_same_line | 18 | 18 | 18 | 16 |
| rx_cb_shared_bus | 14 | 11 | 11 | 11 |
| The three two-breaker classes | 35 | 15 | 26 | 22 |
| The other eleven | 212 | 183 | 179 | 176 |

- **The gain is the two-breaker classes, through the step the fork probe named.** 13 of the 20 cases gained at
  the first evaluation are two-breaker cases, and 12 of the 13 now run `inspect`, `correct_topology`,
  `inspect`, `correct_topology`, `finish` (the thirteenth adds an `undo` and a third `correct_topology`). The supervised model fitted a
  line after the first breaker correction in 10 of them, repeated `correct_topology` until out of actions in 2
  and went on after the second breaker correction in 1. The development probe found the same step: a second
  look at the breakers or the second breaker at once repairs 16 of its 20 two-breaker cases. Stage A reached it
  by sampling, with no forks. Only the tools' names were read here, not the calls' arguments.
- **20 of the 49 unrepaired cases are repaired at the first evaluation and 19 at the second** (23 in at least
  one); the probe's one forced step repairs 39 of the 49.
- **The loss is cases the supervised model solved in few actions, mostly by an added line fit.** 9 of the 13
  cases lost at the first evaluation and 14 of the 19 lost at the second make more `correct_parameters` calls
  than the supervised model's episode did; in 2 and 4 of them the breaker correction is gone. control_meter
  falls from 11 to 7 of 12 at the first evaluation: all four lost cases add a line fit before or after the meter
  correction. At the second evaluation control_cb and rx_cb_same_line lose two cases each that
  `inspect`, `correct_topology`, `finish` had solved: a `correct_parameters` call now follows the breaker
  correction (3) or replaces it (1), and the episode ends in a false claim.
- **Between the evaluations the model acts more and solves less.** 12 of the 15 cases lost from the first
  evaluation to the second have longer episodes than at the first (4.5 to 6.7 actions on the mean); tool errors
  go from 48 to 116 and episodes with a damaged healthy object from 29 to 34. One case gained at the first
  evaluation (r0_566f34094a8a) is back to repeating `correct_topology` until out of actions.
- **What the counts allow.** 205 is the highest count of any run on this suite (the plain and focused runs
  ended at 202), and under the 207 to 209 the suite can tell from the supervised model. Against the rule
  expert's 184 it is 39 gained and 18 lost (p = 0.008); it is the better of two readings of one run, so that p
  is not a test. The second evaluation cannot be told from the start or from the expert.
- **The planned check did not fire.** The first evaluation gained more than it lost, so nothing was cancelled or
  changed. The plan has no rule for a second evaluation that gains what it loses.

The training rollouts of iterations 2 to 6 (256 sampled episodes each; other cases every iteration, so a weak
reading; the four outcome counts are not exclusive and sum to 258 in iteration 2):

| Iteration | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| Repaired / false claims / declared unresolved / out of actions | 125 / 63 / 70 / 0 | 132 / 75 / 48 / 1 | 134 / 70 / 50 / 2 | 125 / 72 / 58 / 1 | 129 / 55 / 71 / 1 |
| The 24 unrepaired cases drawn: with a repaired sample | 19 | 20 | 21 | 15 | 19 |
| Their episodes repaired, of 192 | 81 | 76 | 83 | 67 | 75 |
| The 8 repaired cases drawn: episodes repaired, of 64 | 44 | 56 | 51 | 58 | 54 |
| Truth-built examples kept (rows) | 6 (25) | 4 (17) | 4 (18) | 9 (41) | 5 |
| Rows: policy / for the KL term alone / supervised | 1,299 / 8 / 25 | 1,257 / 7 / 17 | 1,334 / 0 / 18 | 1,365 / 11 / 41 | update running |

Over iterations 0 to 6 the sampled episodes of the unrepaired cases drawn are repaired 10, 42, 81, 76, 83, 67
and 75 times of 192, and those of the repaired cases drawn 62, 53, 44, 56, 51, 58 and 54 times of 64: the same
two movements as on the development suite. The rows that hold a repaired case in place by the KL term alone
fell from 58 in iteration 0 to between 0 and 13.

The updates of iterations 2 to 5 (eight steps each, the optimizer carried: 17, 25, 33 and 41 steps after):

| Iteration | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- |
| KL to the supervised model by step | 0.028, 0.017, 0.032, 0.043, 0.037, 0.025, 0.035, 0.040 | 0.057, 0.036, 0.021, 0.043, 0.025, 0.063, 0.175, 0.048 | 0.041, 0.035, 0.023, 0.043, 0.008, 0.089, 0.113, 0.052 | 0.068, 0.046, 0.038, 0.066, 0.043, 0.051, 0.041, 0.033 |
| Mean absolute log-ratio to the sampler on uncertain tokens, first and eighth step | 0.057, 0.190 | 0.008, 0.161 | 0.013, 0.136 | 0.008, 0.088 |
| The largest on one token over the eight steps | 3.35 | 2.79 | 5.75 | 1.88 |
| Largest row gradient norm before the row bound of 1.0 | 14.8 | 65.8 | 1,354 | 28.3 |
| Most rows the row bound cut in one step | 7 | 13 | 6 | 4 |
| Gradient norm before the clip at 1.0, range | 1.38 to 4.66 | 1.09 to 6.46 | 1.16 to 3.89 | 1.59 to 3.51 |
| Mean advantage of the batch | -0.023 | -0.065 | -0.018 | -0.036 |

The KL to the supervised model is mostly 0.02 to 0.07 a step, 6 to 20 times the focused run's whole movement
(0.0033), with a low of 0.008 and a high of 0.175; iteration 1's 2.76 did not come back. The drift inside one batch is smaller than in
iteration 1 (0.27 on uncertain tokens at its eighth step). One row of iteration 4's last step has a gradient
norm of 1,354 before the row bound.

Time and cards, in UTC (the start was 2026-11-22 15:37):

| Iteration | Rollout | Update written | Evaluation |
| --- | --- | --- | --- |
| 0 | 11-22 15:39 to 17:07 | 11-22 21:12 | |
| 1 | 11-22 21:14 to 22:31 | 11-23 02:16 | |
| 2 | 11-23 02:17 to 03:43 | 11-23 09:47 | 11-23 09:49 to 11:08 |
| 3 | 11-23 11:09 to 12:33 | 11-23 17:44 | |
| 4 | 11-23 17:45 to 19:11 | 11-24 01:22 | |
| 5 | 11-24 01:24 to 02:59 | 11-24 10:31 | 11-24 10:33 to 11:56 |
| 6 | 11-24 11:57 to 13:31 | step 3 of 8 at 16:49 | |

The run has been preempted 17 times (6 in the first link, 11 in the second) and the first link ended at its
12-hour limit on 2026-11-23 at 19:43 UTC, as a chain is built for; 9 of the 18 finished stints ran under an
hour. Every restart loads the model again and repeats what its phase had not saved. 48.0 card-hours are used in
49.4 hours. Iterations 0 and 1 took 5.3 h each, iterations 2 to 5 took 7.7 h each on the mean. What is left is
five steps of iteration 6's update, iterations 7 and 8 and the third evaluation: 14 to 20 hours, so the third
evaluation (after 65 steps) is expected between about 07:30 and 13:00 UTC on 2026-11-25, and the run will have
used 62 to 68 card-hours against the planned 41 to 55. Each iteration's adapter is kept (iteration 2's is the
205). A waiter is armed on `out/steps/iteration_008/evaluation.json`.

**Stage A is complete: the third evaluation, read 2026-11-25 at 16:00 UTC.** The second link (24090333) ended
at 13:04 UTC with `mrp-steps-complete`, after a last stint of 6 h 20 min without a preemption; the five links
behind it found the run complete and ended at once, and no `mrp-` job is queued. Nine iterations are done (65
optimizer steps), integrity ok in each, no traceback. The evaluation was written at 13:04 UTC and read three
hours later: the SSH master had ended between 06:05 and 06:20 UTC, the local waiter for its return died (exit
code 4), and the user restored the master at 15:57 UTC.

| Development suite, greedy | Supervised model (the start) | After 17 steps (iteration 2) | After 41 steps (iteration 5) | After 65 steps (iteration 8, written 2026-11-25 13:04 UTC) |
| --- | --- | --- | --- | --- |
| **Solved of the 247 a truth-built sequence repairs** | **198** | **205** (201 strict) | **198** (198 strict) | **187** (179 strict) |
| Gained / lost against the start | | 20 / 13 (exact two-sided p = 0.30) | 19 / 19 (p = 1.0) | 15 / 26 (p = 0.12) |
| Of the 184 the rule expert solves | 161 | 166 | 158 | 150 |
| Of the 63 only a truth-built sequence repairs | 37 | 39 | 40 | 37 |
| Against the rule expert's 184: gained / lost | 37 / 23 (p = 0.09) | 39 / 18 (p = 0.008) | 40 / 26 (p = 0.11) | 37 / 34 (p = 0.81) |
| False claims | 72 | 64 | 72 | 76 |
| Declared unresolved | 27 | 31 | 27 | 31 |
| Out of actions | 3 | 0 | 3 | 6 |
| Episodes with a damaged healthy object (objects) | 27 (51) | 29 (41) | 34 (46) | 46 (75) |
| Tool errors (unknown object / tool failure) | 88 (30 / 58) | 48 (0 / 48) | 116 (50 / 66) | 147 (73 / 74) |
| True errors repaired of 643 | 520 | 530 | 522 | 506 |
| Mean actions an episode | 4.45 | 4.45 | 4.69 | 4.87 |
| Calls: `inspect` / `correct_topology` / `correct_parameters` / `correct_measurements` / `undo` | 323 / 199 / 355 / 133 / 27 | 357 / 183 / 360 / 112 / 23 | 348 / 223 / 383 / 122 / 34 | 352 / 253 / 376 / 149 / 38 |

From the second evaluation to the third: 10 gained, 21 lost (p = 0.07). Against the start, 12 cases are gained
in all three evaluations and 25 in at least one; 6 are lost in all three and 39 in at least one. 159 cases are
solved by all four models and 223 by at least one, so 24 of the 247 are solved by none of them. None of the 53
cases nothing repairs is solved by any.

Solved by class, of the cases a truth-built sequence repairs:

| Class | Solvable | Start | 17 steps | 41 steps | 65 steps |
| --- | --- | --- | --- | --- | --- |
| cb2_dangling | 14 | 6 | 12 | 11 | 8 |
| cb2_mixed | 11 | 4 | 7 | 4 | 5 |
| cb2_split | 10 | 5 | 7 | 7 | 6 |
| control_cb | 12 | 12 | 12 | 10 | 8 |
| control_meter | 12 | 11 | 7 | 8 | 10 |
| control_rx | 11 | 11 | 11 | 10 | 9 |
| meter_on_rx | 16 | 13 | 13 | 12 | 13 |
| meter_rx2 | 20 | 15 | 16 | 17 | 13 |
| rx2_adjacent | 26 | 20 | 19 | 21 | 20 |
| rx2_distant | 24 | 21 | 22 | 22 | 20 |
| rx_cb_distant | 30 | 26 | 26 | 26 | 27 |
| rx_cb_meter | 29 | 25 | 24 | 23 | 22 |
| rx_cb_same_line | 18 | 18 | 18 | 16 | 16 |
| rx_cb_shared_bus | 14 | 11 | 11 | 11 | 10 |
| The three two-breaker classes | 35 | 15 | 26 | 22 | 19 |
| The other eleven | 212 | 183 | 179 | 176 | 168 |

- **The solved count falls with the steps: 205, 198, 187.** The first evaluation is the best of the run. No
  evaluation differs from the start by more than the suite can tell (p = 0.30, 1.0, 0.12); the third is the
  nearest to a difference, and it is a loss. Against the rule expert's 184 the third evaluation gains 37 and
  loses 34 (p = 0.81).
- **About half of the gain stays.** 12 cases are gained in all three evaluations, 6 of them two-breaker cases;
  the two-breaker classes go 15, 26, 22, 19 of 35. Of the 49 cases the supervised model leaves unrepaired, 20,
  19 and 15 are repaired at the three evaluations (25 in at least one); the probe's one forced step repairs 39.
- **The loss grows: 13, 19, 26 cases.** Of the 26 lost at the third evaluation, 19 have longer episodes than the
  supervised model's; 12 make more `correct_parameters` calls, 9 more `correct_measurements` calls and 4 more
  `correct_topology` calls (one of them repeats `correct_topology` until out of actions); 3 keep the tool
  sequence with other arguments. 15 end in a false claim, 10 declare unresolved, 1 runs out of actions.
  control_cb, which the supervised model solved 12 of 12 by `inspect`, `correct_topology`, `finish`, is at 8:
  all four lost cases go on after the breaker correction (a line fit in three, a meter correction and a line
  fit in one). Only the tools' names were read here, not the calls' arguments.
- **The model acts more and harms more.** Mean actions an episode 4.45, 4.45, 4.69, 4.87; tool errors 88, 48,
  116, 147; episodes with a damaged healthy object 27, 29, 34, 46 (objects 51, 41, 46, 75); solved cases that
  fail the strict audit 0, 4, 0, 8.
- **The planned check did not fire and the run ended as planned.** Nothing was cancelled. Iteration 2's adapter
  (the 205) is kept at `out/steps/iteration_002/adapter`, as is every iteration's.

How far the episodes got, from each case's episode file (`evaluation/episodes/`, which holds every injected
error's `repaired` flag; the evaluation's summary gives only the suite total). "Damage only" is a case with
every error repaired and a healthy object outside tolerance:

| | Start | 17 steps | 41 steps | 65 steps |
| --- | --- | --- | --- | --- |
| All 300: solved / damage only / some errors repaired / none | 198 / 1 / 86 / 15 | 205 / 3 / 78 / 14 | 198 / 7 / 78 / 17 | 187 / 6 / 83 / 24 |
| The 247: solved / damage only / some / none | 198 / 1 / 38 / 10 | 205 / 3 / 30 / 9 | 198 / 7 / 32 / 10 | 187 / 6 / 39 / 15 |
| The 53 nothing repairs: some / none | 48 / 5 | 48 / 5 | 46 / 7 | 44 / 9 |
| Errors repaired: breakers of 188 | 159 | 172 | 162 | 159 |
| Errors repaired: lines of 346 | 256 | 259 | 262 | 246 |
| Errors repaired: meters of 109 | 105 | 99 | 98 | 101 |
| The 247's errors repaired, of 517 | 457 | 466 | 460 | 445 |
| The 53's errors repaired, of 126 | 63 | 64 | 62 | 61 |
| The 247: false claims / declared unresolved / out of actions | 27 / 20 / 2 | 20 / 22 / 0 | 28 / 19 / 2 | 31 / 23 / 6 |
| "Unresolved" declared on a repaired state | 0 | 0 | 0 | 0 |

The training rollouts of iterations 7 and 8 (256 sampled episodes each; other cases every iteration, so a weak
reading; iteration 8's are sampled by the adapter after 57 steps):

| Iteration | 7 | 8 |
| --- | --- | --- |
| Repaired / false claims / declared unresolved / out of actions | 130 / 53 / 70 / 3 | 103 / 103 / 50 / 0 |
| The 24 unrepaired cases drawn: with a repaired sample | 19 | 14 |
| Their episodes repaired, of 192 | 78 | 60 |
| The 8 repaired cases drawn: episodes repaired, of 64 | 52 | 43 |
| Truth-built examples kept (rows) | 5 (23) | 10 (42) |
| Rows: policy / for the KL term alone / supervised | 1,568 / 0 / 23 | 1,372 / 9 / 42 |
| Mean reward of the sampled episodes | 0.200 | 0.004 |

Iteration 8's rollout is the weakest since iteration 1 (95 repaired, 108 false claims); it agrees with the third
evaluation.

The updates of iterations 6 to 8 (eight steps each, the optimizer carried: 49, 57 and 65 steps after):

| Iteration | 6 | 7 | 8 |
| --- | --- | --- | --- |
| KL to the supervised model by step | 0.024, 0.049, 0.042, 0.068, 0.055, 0.073, 0.041, 0.031 | 0.079, 0.086, 0.053, 0.041, 0.060, 0.064, 0.065, 0.073 | 0.029, 0.030, 0.069, 0.062, 0.049, 0.069, 0.092, 0.073 |
| Mean absolute log-ratio to the sampler on uncertain tokens, first and eighth step | 0.016, 0.097 | 0.035, 0.171 | 0.006, 0.206 |
| The largest on one token over the eight steps | 2.14 | 4.24 | 4.48 |
| Largest row gradient norm before the row bound of 1.0 | 50.0 | 32.9 | 87.8 |
| Most rows the row bound cut in one step | 5 | 8 | 6 |
| Gradient norm before the clip at 1.0, range | 1.34 to 4.14 | 1.99 to 4.66 | 1.83 to 4.27 |
| Mean advantage of the batch | -0.060 | -0.051 | -0.046 |
| Supervised loss, the largest of the eight steps | 0.076 | 0.446 | 0.171 |

The KL to the supervised model stays between 0.02 and 0.09 a step in the last three updates, as in iterations 2
to 5: by this measure the adapter is as near the start after 65 steps as after 17, and the greedy episodes
change all the same (39 cases lost in at least one evaluation).

Time and cards, in UTC:

| Iteration | Rollout | Update written | Evaluation |
| --- | --- | --- | --- |
| 6 | 11-24 11:57 to 13:31 | 11-24 20:07 | |
| 7 | 11-24 20:09 to 22:16 | 11-25 06:25 | |
| 8 | 11-25 06:27 to 07:42 | 11-25 11:59 | 11-25 12:01 to 13:04 |

The run took 69.5 hours (2026-11-22 15:37 to 2026-11-25 13:04 UTC) and 67.3 card-hours against the planned 41
to 55: 29 preemptions (6 in the first link, 23 in the second) and one 12-hour limit; 17 of the 31 stints ran
under an hour. Iteration 7's update took 8 h 9 min through six preemptions.

## 10. What the probe's smoke changed (2026-11-22; deployed at 98e6518 before the probes)

The smoke's records show the rows stage B would have trained on those 4 cases as deployed: 15 rows, 8 of them
`finish`. Two kinds were wrong.

- **A lesson contradicted by a later one.** In the cb2_split case the first two states taught "declare the case
  unresolved" (0.05 and 0.04 above the greedy episode: the actions it saves against an episode that fails anyway),
  beside the look that repairs the case at the third state.
- **A lesson at the wrong state.** In the rx_cb_distant case the greedy episode ends in a false claim. Compared
  with it, the first state taught "declare the case unresolved before looking at anything" with weight 0.52, and
  the second taught the false claim one action earlier (0.01).

Two changes, both in `psse_env/minimal/forks.py`:

1. **A state is compared with what going on can still reach**: the greedy return, or the return of a branch kept
   at a later state. Before a later, better branch the greedy action is the right one, since it leads there.
   The excess is the row's weight (`margin`); with `min_gain` 0 the margins of a case add up to its best return
   less the greedy return, so a repair is paid once, at the state where it is gained.
2. **Both outcomes of `finish` are forked as whole calls** at every state (the greedy action left out), with no
   model call, because the episode ends there. That puts "declare the case unresolved" at the state where the
   greedy episode claimed it resolved. A `finish` whose outcome the model writes is no longer forked.

Neither is a decision rule: the first is arithmetic on returns, the second tries both calls of the one tool that
has only two.

The smoke's stored records read with the new comparison (they predate the given outcomes of `finish`):

| | Against the greedy return (as deployed) | Against what going on reaches (now) |
| --- | --- | --- |
| Rows | 15 | 10 |
| Rows whose branch repairs the case | 6, weight 7.98 | 5, weight 3.52 |
| `finish` rows | 8, weight 0.86 | 4, weight 0.08 |
| Other: an undo that takes back a damaging meter correction before the case is handed off | 1, weight 0.49 | 1, weight 0.49 |
| Total weight | 9.33 | 4.09 |

The four `finish` rows that remain are all in the case nothing repairs. Each says "stop one state earlier" and is
worth the actions it saves (0.01 to 0.03), 2% of the weight. `forks.min_gain` above 0.03 would drop them; it was
0 in `rl_forks.json` then. After the probes the user set it to 0.1 (2026-11-23, section 9), so stage B does not
teach stopping earlier for the actions saved alone.

Also changed: the update reads a stored record with its own `min_gain` (`stored_forks` asks `better_branches`
again), so a case's file keeps the observation of every state with a carried-out branch above the greedy return;
the summary has `margin` (with the share of the branches that repair) beside `gain`, and
`unrepaired_cases_repaired_after_a_refused_call`; the contract is `minimal_one_step_forks_v2`.

**What it means for the order of work.** Stage A does not run this code. The probe's count of cases one forced
step repairs does not depend on the comparison, but a probe on d6b7cfc would fork a `finish` the model writes
and report rows under the old comparison. So the cell was redeployed at 98e6518 before the probes (section 9).

Local tests after the change: `test_forks.py` 13 (one new: the three patterns above as records), the driver 32,
the probe 4, this cell 18, the research policy 15, the earlier RL cell 24.
