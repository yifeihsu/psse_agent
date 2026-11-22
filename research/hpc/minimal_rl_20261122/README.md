# The RL plan of 2026-11-22: stage A, stage B and the fork probe

**Status: REDEPLOYED 2026-11-22 at 98e6518; stage A's run and both probes are SUBMITTED** (section 9:
`mrp-steps` 24090332 to 24090338, `mrp-probe-dev` 24090400 to 24090402, `mrp-probe-train` 24090446 to 24090450).
Both smokes completed and passed their checks on d6b7cfc. The probe's smoke showed that stage B would have
picked contradictory rows; how a fork row is picked was changed the same day (section 10) and is in the deployed
source. The user's word so far: "Yes, deploy and smoke", then "Yes, all three" (the redeploy, stage A's run and
both probes). Stage B (`forks-smoke`, `forks`), any cancel and any resubmission each still need the user's word.

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
  before the run goes on.
- Stage B, in every iteration: how many unrepaired cases one forced step repairs, how many fork rows the update
  took and by which tool, and how many of them are within one action's cost (a saved action, not a repair).
- The probe: of the unrepaired cases, how many one forced step repairs, by class and by the forced tool. This is
  a labelled upper bound on one round of stage B, not a result.

## 7. Not checked

- The forced call has met the model on 4 cases (section 9): its tokens start the call's own rendering in 176 of
  176 branches, no forced call was invalid and no replay drifted. Stage B's update on fork rows (`forks-smoke`)
  has not run.
- The selection of section 10 and the given outcomes of `finish` have run with stand-ins and on the smoke's
  stored records only. The two probes submitted at 98e6518 are their first run with the model; they went out
  without a second smoke.
- Whether eight steps on one batch at 3e-5 stay close to the sampled policy, and whether 65 steps are safe or
  enough. The smoke took four steps on one small batch: by the fourth, the uncertain tokens were a mean factor of
  1.35 from the policy that sampled the batch (section 9). No earlier RL run of this project took more than a
  few steps.
- Every cost above; the measured rates are from 4 cases and two small updates.
- Whether supervised rows on fork states, with the KL term, keep the cases already solved.
- The forks read the greedy episode. A case whose repair needs two unlikely steps in a row is not reached by one
  forced step (the probe's tests show such a case); how many of the 49 are like that is the probe's question.

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
worth the actions it saves (0.01 to 0.03), 2% of the weight. `forks.min_gain` above 0.03 would drop them; it is 0
in `rl_forks.json`. Whether stage B should teach stopping earlier on a case it cannot repair is the user's call.

Also changed: the update reads a stored record with its own `min_gain` (`stored_forks` asks `better_branches`
again), so a case's file keeps the observation of every state with a carried-out branch above the greedy return;
the summary has `margin` (with the share of the branches that repair) beside `gain`, and
`unrepaired_cases_repaired_after_a_refused_call`; the contract is `minimal_one_step_forks_v2`.

**What it means for the order of work.** Stage A does not run this code. The probe's count of cases one forced
step repairs does not depend on the comparison, but a probe on d6b7cfc would fork a `finish` the model writes
and report rows under the old comparison. So the cell was redeployed at 98e6518 before the probes (section 9).

Local tests after the change: `test_forks.py` 13 (one new: the three patterns above as records), the driver 32,
the probe 4, this cell 18, the research policy 15, the earlier RL cell 24.
